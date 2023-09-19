#include <LibUtilities/Foundations/Basis.h>

#include <LibUtilities/BasicUtils/ShapeType.hpp>
#include <LibUtilities/BasicUtils/SharedArray.hpp>

#include "BwdTransMatFreeKernels.hpp"
#include "Operators/OperatorBwdTrans.hpp"
#include <LibUtilities/SimdLib/tinysimd.hpp>

namespace Nektar::Operators::detail
{

using vec_t = tinysimd::simd<double>;

// Matrix-free implementation
template <typename TData>
class OperatorBwdTransImpl<TData, ImplMatFree> : public OperatorBwdTrans<TData>
{
public:
    OperatorBwdTransImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorBwdTrans<TData>(expansionList)
    {
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Phys> &out) override
    {
        // Reshape into vec_t::width, with the right memory alignment
        // requirements, if the data are already interleaved, this method
        // returns
        in.template ReshapeStorage<vec_t::width>();
        out.template ReshapeStorage<vec_t::width>();
        TData *inptr  = in.GetStorage().GetCPUPtr();
        TData *outptr = out.GetStorage().GetCPUPtr();

        size_t exp_idx = 0;
        for (size_t block_idx = 0; block_idx < in.GetBlocks().size();
             ++block_idx)
        {
            auto const expPtr    = this->m_expansionList->GetExp(exp_idx);
            auto const &inblock  = in.GetBlocks()[block_idx];
            auto const &outblock = out.GetBlocks()[block_idx];
            auto const shapeType = expPtr->DetShapeType();
            auto const dimension = expPtr->GetShapeDimension();

            m_nElmtGroup =
                (inblock.num_elements + inblock.num_padding_elements) /
                vec_t::width;

            m_basis.resize(dimension);
            for (int i = 0; i < dimension; ++i)
            {
                m_basis[i]    = expPtr->GetBasis(i);
                auto bdataRAW = expPtr->GetBasis(i)->GetBdata();
                m_B[i].resize(bdataRAW.size());
                for (auto j = 0; j < bdataRAW.size(); ++j)
                {
                    m_B[i][j] = bdataRAW[j];
                }
            }

#include "SwitchNodesPoints.h"

            inptr += inblock.block_size;
            outptr += outblock.block_size;
            exp_idx += inblock.num_elements;
        }
    }

    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorBwdTransImpl<TData, ImplMatFree>>(
            expansionList);
    }

    static std::string className;

private:
    std::vector<LibUtilities::BasisSharedPtr> m_basis;
    std::array<std::vector<vec_t, tinysimd::allocator<vec_t>>, 3> m_B;
    int m_nElmtGroup;
    
    // void copy_to_vec_t(const NekDouble *in, const std::uint32_t nVec,
    //                    std::vector<vec_t, allocator<vec_t>> &out)
    // {
    //     for (size_t i = 0; i < nVec; ++i)
    //     {
    //         out[i].load(in);
    //         in += vec_t::width;
    //     }
    // }
    // void copy_from_vec_t(const std::vector<vec_t, allocator<vec_t>> &in,
    //                      const std::uint32_t nVec, NekDouble *out)
    // {
    //     for (size_t i = 0; i < nVec; ++i)
    //     {
    //         in[i].store(out);
    //         out += vec_t::width;
    //     }
    // }

    // templated operator(), which is instantiated by SwitchNodesPoints.h
    // and used in apply().
    // size based template version
    template <LibUtilities::ShapeType SHAPE_TYPE, int nm0, int nq0>
    void operator1D(const NekDouble *input, NekDouble *output)
    {
        constexpr auto nqTot = nq0;
        constexpr auto nmTot = nm0;
        // constexpr auto nqBlocks = nqTot * vec_t::width;
        // constexpr auto nmBlocks = nmTot * vec_t::width;
        // const auto nElmtGroup = this->m_nElmtGroup;
        // Workspace for kernels - also checks preconditions
        BwdTrans1DWorkspace<SHAPE_TYPE>(nm0, nq0);

        // std::vector<vec_t, allocator<vec_t>> tmpIn(nmTot), tmpOut(nqTot);
        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);
        vec_t::scalarType *tmpOut =
            reinterpret_cast<vec_t::scalarType *>(output);

        for (int e = 0; e < m_nElmtGroup; ++e)
        {
            // // Load and transpose data
            // copy_to_vec_t(input, nmTot, tmpIn);
            // load_interleave(input, nmTot, tmpIn);

            BwdTrans1DKernel<SHAPE_TYPE>(nm0, nq0, tmpIn, this->m_B[0],
                                         tmpOut);

            // // de-interleave and store data
            // copy_from_vec_t(tmpOut, nqTot, output);
            // deinterleave_store(tmpOut, nqTot, output);

            tmpIn += nmTot;
            tmpOut += nqTot * vec_t::width;
        }
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE>
    void operator1D(const NekDouble *input, NekDouble *output)
    {
        const auto nm0 = m_basis[0]->GetNumModes();
        const auto nq0 = m_basis[0]->GetNumPoints();

        const auto nqTot = nq0;
        const auto nmTot = nm0;
        // const auto nqBlocks = nqTot * vec_t::width;
        // const auto nmBlocks = nmTot * vec_t::width;
        // const auto nElmtGroup = this->m_nElmt / vec_t::width;
        // Workspace for kernels - also checks preconditions
        BwdTrans1DWorkspace<SHAPE_TYPE>(nm0, nq0);

        // std::vector<vec_t, allocator<vec_t>> tmpIn(nmTot), tmpOut(nqTot);
        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);
        vec_t::scalarType *tmpOut =
            reinterpret_cast<vec_t::scalarType *>(output);

        for (int e = 0; e < m_nElmtGroup; ++e)
        {
            // // Load and transpose data
            // copy_to_vec_t(input, nmTot, tmpIn);
            // load_interleave(input, nmTot, tmpIn);

            BwdTrans1DKernel<SHAPE_TYPE>(nm0, nq0, tmpIn, this->m_B[0],
                                         tmpOut);

            // // de-interleave and store data
            // copy_from_vec_t(tmpOut, nqTot, output);
            // deinterleave_store(tmpOut, nqTot, output);

            tmpIn += nmTot;
            tmpOut += nqTot * vec_t::width;
        }
    }

    // size based template version
    template <LibUtilities::ShapeType SHAPE_TYPE, int nm0, int nm1, int nq0,
              int nq1>
    void operator2D(const NekDouble *input, NekDouble *output)
    {
        constexpr auto nqTot = nq0 * nq1;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
        // constexpr auto nqBlocks = nqTot * vec_t::width;
        // const auto nmBlocks     = nmTot * vec_t::width;
        // const auto nElmtGroup = this->m_nElmt / vec_t::width;
        const bool correct =
            (m_basis[0]->GetBasisType() == LibUtilities::eModified_A);

        // Workspace for kernels - also checks preconditions
        size_t wsp0Size = 0;
        BwdTrans2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);

        // std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size), tmpIn(nmTot),
        //     tmpOut(nqTot);
        std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size);
        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);
        vec_t::scalarType *tmpOut =
            reinterpret_cast<vec_t::scalarType *>(output);

        for (int e = 0; e < m_nElmtGroup; ++e)
        {
            // // Load and transpose data
            // copy_to_vec_t(input, nmTot, tmpIn);
            // load_interleave(input, nmTot, tmpIn);

            BwdTrans2DKernel<SHAPE_TYPE>(nm0, nm1, nq0, nq1, correct, tmpIn,
                                         this->m_B[0], this->m_B[1],
                                         wsp0, tmpOut);

            // // de-interleave and store data
            // copy_from_vec_t(tmpOut, nqTot, output);
            // deinterleave_store(tmpOut, nqTot, output);

            tmpIn += nmTot;
            tmpOut += nqTot * vec_t::width;
        }
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE>
    void operator2D(const NekDouble *input, NekDouble *output)
    {
        const auto nm0 = m_basis[0]->GetNumModes();
        const auto nm1 = m_basis[1]->GetNumModes();

        const auto nq0 = m_basis[0]->GetNumPoints();
        const auto nq1 = m_basis[1]->GetNumPoints();

        const auto nqTot = nq0 * nq1;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
        // const auto nqBlocks = nqTot * vec_t::width;
        // const auto nmBlocks = nmTot * vec_t::width;
        // const auto nElmtGroup = this->m_nElmt / vec_t::width;
        const bool correct =
            (m_basis[0]->GetBasisType() == LibUtilities::eModified_A);

        // Workspace for kernels - also checks preconditions
        size_t wsp0Size = 0;
        BwdTrans2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);

        // std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size), tmpIn(nmTot),
        //     tmpOut(nqTot);
        std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size);
        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);
        vec_t::scalarType *tmpOut =
            reinterpret_cast<vec_t::scalarType *>(output);

        for (int e = 0; e < m_nElmtGroup; ++e)
        {
            // // Load and transpose data
            // copy_to_vec_t(input, nmTot, tmpIn);
            // load_interleave(input, nmTot, tmpIn);

            BwdTrans2DKernel<SHAPE_TYPE>(nm0, nm1, nq0, nq1, correct, tmpIn,
                                         this->m_B[0], this->m_B[1],
                                         wsp0, tmpOut);

            // // de-interleave and store data
            // copy_from_vec_t(tmpOut, nqTot, output);
            // deinterleave_store(tmpOut, nqTot, output);

            tmpIn += nmTot;
            tmpOut += nqTot * vec_t::width;
        }
    }

    // size based template version
    template <LibUtilities::ShapeType SHAPE_TYPE, int nm0, int nm1, int nm2,
              int nq0, int nq1, int nq2>
    void operator3D(const NekDouble *input, NekDouble *output)
    {
        constexpr auto nqTot = nq0 * nq1 * nq2;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
        // constexpr auto nqBlocks = nqTot * vec_t::width;
        // const auto nmBlocks     = nmTot * vec_t::width;
        // const auto nElmtGroup = this->m_nElmt / vec_t::width;
        const bool correct =
            (m_basis[0]->GetBasisType() == LibUtilities::eModified_A);

        // Workspace for kernels - also checks preconditions
        size_t wsp0Size = 0, wsp1Size = 0;
        BwdTrans3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size);

        // std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size), wsp1(wsp1Size),
        //     tmpIn(nmTot), tmpOut(nqTot);
        std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size), wsp1(wsp1Size);
        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);
        vec_t::scalarType *tmpOut =
            reinterpret_cast<vec_t::scalarType *>(output);

        for (int e = 0; e < m_nElmtGroup; ++e)
        {
            // // Load and transpose data
            // copy_to_vec_t(input, nmTot, tmpIn);
            // load_interleave(input, nmTot, tmpIn);

            BwdTrans3DKernel<SHAPE_TYPE>(
                nm0, nm1, nm2, nq0, nq1, nq2, correct, tmpIn, this->m_B[0],
                this->m_B[1], this->m_B[2], wsp0, wsp1, tmpOut);

            // // de-interleave and store data
            // copy_from_vec_t(tmpOut, nqTot, output);
            // deinterleave_store(tmpOut, nqTot, output);

            tmpIn += nmTot;
            tmpOut += nqTot * vec_t::width;
        }
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE>
    void operator3D(const NekDouble *input, NekDouble *output)
    {
        const auto nm0 = m_basis[0]->GetNumModes();
        const auto nm1 = m_basis[1]->GetNumModes();
        const auto nm2 = m_basis[2]->GetNumModes();

        const auto nq0 = m_basis[0]->GetNumPoints();
        const auto nq1 = m_basis[1]->GetNumPoints();
        const auto nq2 = m_basis[2]->GetNumPoints();

        const auto nqTot = nq0 * nq1 * nq2;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
        // const auto nqBlocks = nqTot * vec_t::width;
        // const auto nmBlocks = nmTot * vec_t::width;
        // const auto nElmtGroup = this->m_nElmt / vec_t::width;
        const bool correct =
            (m_basis[0]->GetBasisType() == LibUtilities::eModified_A);

        // Workspace for kernels - also checks preconditions
        size_t wsp0Size = 0, wsp1Size = 0;
        BwdTrans3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size);

        // std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size), wsp1(wsp1Size),
        //     tmpIn(nmTot), tmpOut(nqTot);
        std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size), wsp1(wsp1Size);
        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);
        vec_t::scalarType *tmpOut =
            reinterpret_cast<vec_t::scalarType *>(output);

        for (int e = 0; e < m_nElmtGroup; ++e)
        {
            // // Load and transpose data
            // copy_to_vec_t(input, nmTot, tmpIn);
            // load_interleave(input, nmTot, tmpIn);

            BwdTrans3DKernel<SHAPE_TYPE>(
                nm0, nm1, nm2, nq0, nq1, nq2, correct, tmpIn, this->m_B[0],
                this->m_B[1], this->m_B[2], wsp0, wsp1, tmpOut);

            // // de-interleave and store data
            // copy_from_vec_t(tmpOut, nqTot, output);
            // deinterleave_store(tmpOut, nqTot, output);

            tmpIn += nmTot;
            tmpOut += nqTot * vec_t::width;
        }
    }
};

} // namespace Nektar::Operators::detail
