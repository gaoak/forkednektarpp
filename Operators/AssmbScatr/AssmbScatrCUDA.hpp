#include "MemoryRegionCUDA.hpp"
#include "Operators/AssmbScatr/AssmbScatrCUDAKernels.cuh"
#include "Operators/OperatorAssmbScatr.hpp"
#include "Operators/OperatorHelper.cuh"

#include <MultiRegions/AssemblyMap/AssemblyMapCG.h>
#include <MultiRegions/ContField.h>

using namespace Nektar;
using namespace Nektar::MultiRegions;
using namespace Nektar::Operators;

namespace Nektar::Operators::detail
{

template <typename TData>
class OperatorAssmbScatrImpl<TData, ImplCUDA> : public OperatorAssmbScatr<TData>
{
public:
    OperatorAssmbScatrImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorAssmbScatr<TData>(expansionList)
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        m_assmbMap = contfield->GetLocalToGlobalMap();

        // Get the solution type
        GlobalSysSolnType solnType = m_assmbMap->GetGlobalSysSolnType();
        auto nloc                  = m_assmbMap->GetNumLocalCoeffs();
        auto nglo                  = (solnType == eIterativeFull)
                                         ? m_assmbMap->GetNumGlobalCoeffs()
                                         : m_assmbMap->GetNumGlobalBndCoeffs();

        // Memory allocation for tmp pointer
        cudaMalloc((void **)&m_tmpptr, sizeof(TData) * nglo);
        cudaMemset(m_tmpptr, 0.0, sizeof(TData) * nglo);

        // Memory allocation for assemble pointer
        cudaMalloc((void **)&m_assmbptr, sizeof(int) * nloc);
        auto assmbptr = m_assmbMap->GetLocalToGlobalMap().get();
        cudaMemcpy(m_assmbptr, assmbptr, sizeof(int) * nloc,
                   cudaMemcpyHostToDevice);

        m_signChange = m_assmbMap->AssemblyMap::GetSignChange();

        if (m_signChange)
        {
            // Memory allocation for sign pointer
            cudaMalloc((void **)&m_signptr, sizeof(TData) * nloc);
            auto signptr = m_assmbMap->GetLocalToGlobalSign().get();
            cudaMemcpy(m_signptr, signptr, sizeof(TData) * nloc,
                       cudaMemcpyHostToDevice);
        }
    }

    ~OperatorAssmbScatrImpl()
    {
        cudaFree(m_tmpptr);
        cudaFree(m_assmbptr);
        m_tmpptr   = nullptr;
        m_assmbptr = nullptr;
        if (m_signChange)
        {
            cudaFree(m_signptr);
            m_signptr = nullptr;
        }
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out,
               const bool &zeroDir = false)
    {
        Assemble(in, m_tmpptr);

        // Zeroing Dirichlet BC
        if (zeroDir)
        {
            size_t nDir = m_assmbMap->GetNumGlobalDirBndCoeffs();
            cudaMemset(m_tmpptr, 0.0, sizeof(TData) * nDir);
        }

        GlobalToLocal(m_tmpptr, out);
    }

    void Assemble(Field<TData, FieldState::Coeff> &in, TData *outptr)
    {
        auto const *inptr =
            in.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();

        // Get the solution type
        GlobalSysSolnType solnType = m_assmbMap->GetGlobalSysSolnType();

        if (solnType == eIterativeFull)
        {
            // Initialise index
            size_t expIdx = 0;
            size_t offset = 0;

            for (size_t block_idx = 0; block_idx < in.GetBlocks().size();
                 ++block_idx)
            {
                // Determine shape and type of the element.
                auto const expPtr = this->m_expansionList->GetExp(expIdx);
                auto nElmts       = in.GetBlocks()[block_idx].num_elements;
                auto numPts       = expPtr->GetNcoeffs();

                // Deterime CUDA grid parameters.
                m_gridSize = nElmts / m_blockSize;
                m_gridSize += (nElmts % m_blockSize == 0) ? 0 : 1;

                if (m_signChange)
                {
                    AssembleKernel<<<m_gridSize, m_blockSize>>>(
                        numPts, nElmts, offset, m_assmbptr, m_signptr, inptr,
                        outptr);
                }
                else
                {
                    AssembleKernel<<<m_gridSize, m_blockSize>>>(
                        numPts, nElmts, offset, m_assmbptr, inptr, outptr);
                }

                // Increment pointer and index for next element type.
                offset += numPts * nElmts;
                expIdx += nElmts;
            }
        }
    }

    void GlobalToLocal(TData *inptr, Field<TData, FieldState::Coeff> &out)
    {
        auto *outptr = out.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();

        // Get the solution type
        GlobalSysSolnType solnType = m_assmbMap->GetGlobalSysSolnType();

        if (solnType == eIterativeFull)
        {
            // Initialise index
            size_t expIdx = 0;
            size_t offset = 0;

            for (size_t block_idx = 0; block_idx < out.GetBlocks().size();
                 ++block_idx)
            {
                // Determine shape and type of the element.
                auto const expPtr = this->m_expansionList->GetExp(expIdx);
                auto nElmts       = out.GetBlocks()[block_idx].num_elements;
                auto numPts       = expPtr->GetNcoeffs();

                // Deterime CUDA grid parameters.
                m_gridSize = nElmts / m_blockSize;
                m_gridSize += (nElmts % m_blockSize == 0) ? 0 : 1;

                if (m_signChange)
                {
                    GlobalToLocalKernel<<<m_gridSize, m_blockSize>>>(
                        numPts, nElmts, offset, m_assmbptr, m_signptr, inptr,
                        outptr);
                }
                else
                {
                    GlobalToLocalKernel<<<m_gridSize, m_blockSize>>>(
                        numPts, nElmts, offset, m_assmbptr, inptr, outptr);
                }

                // Increment pointer and index for next element type.
                offset += numPts * nElmts;
                expIdx += nElmts;
            }
        }
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorAssmbScatrImpl<TData, ImplCUDA>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;

protected:
    AssemblyMapCGSharedPtr m_assmbMap;
    TData *m_tmpptr;
    bool m_signChange;
    TData *m_signptr;
    int *m_assmbptr;
    size_t m_gridSize;
    size_t m_blockSize = 32;
};

} // namespace Nektar::Operators::detail
