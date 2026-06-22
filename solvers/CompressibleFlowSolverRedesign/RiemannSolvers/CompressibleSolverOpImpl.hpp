///////////////////////////////////////////////////////////////////////////////
//
// File: CompressibleSolverOpImpl.hpp
//
// For more information, please see: http://www.nektar.info
//
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
// Department of Aeronautics, Imperial College London (UK), and Scientific
// Computing and Imaging Institute, University of Utah (USA).
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// Description: Compressible Riemann solver.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "RiemannSolvers/CompressibleSolverOp.hpp"
#include <boost/algorithm/string/predicate.hpp>

#include "Operators/LoopExecution/LoopExecution.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

namespace Nektar::Operators::detail
{

template <template <typename, typename, unsigned int> typename RiemannKernel,
          typename EoSParamType, typename ExecSpace, unsigned int NDIM,
          bool ROTMAT, typename TData>
NEK_FORCE_INLINE static void RiemannKernelLauncher(
    const EoSParamType &EoS, const size_t blksize, const TData *normalPtr,
    const TData *rotMatPtr, TData *rotStorage1Ptr, TData *rotStorage2Ptr,
    TData *rotStorage3Ptr, const TData *fwdPtr, const TData *bwdPtr,
    TData *fluxPtr)
{
    using namespace Nektar::Operators::detail;

    // Explicit vectorisation for AVX backend,
    // vec_t = tinysimd::simd<TData> for AVX,
    // vec_t = TData otherwise.
    // using vec_t = typename data_type_if<
    //    std::is_same_v<ExecSpace, NektarSpaces::AVX>, TData>::type;
    constexpr unsigned int vec_width =
        (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
            ? tinysimd::simd<TData>::width
            : 1;

    const size_t groupsize = blksize / vec_width;

    if (NDIM == 3 && ROTMAT)
    {
        Nektar::parallel_for<ExecSpace>(
            0u, groupsize, NEKTAR_LAMBDA(const size_t i) {
                // Only update rotation matrice when necessary.
                GenerateRotationMatrices<ExecSpace>(
                    blksize, normalPtr + i * vec_width,
                    (TData *)rotMatPtr + i * vec_width);

                // Rotate velocity to normal.
                RotateToNormalKernel<ExecSpace, NDIM>(
                    blksize, fwdPtr + i * vec_width, rotMatPtr + i * vec_width,
                    rotStorage1Ptr + i * vec_width);
                RotateToNormalKernel<ExecSpace, NDIM>(
                    blksize, bwdPtr + i * vec_width, rotMatPtr + i * vec_width,
                    rotStorage2Ptr + i * vec_width);

                // Compute flux in rotated frame.
                RiemannKernel<ExecSpace, EoSParamType, NDIM>()(
                    EoS, blksize, rotStorage1Ptr + i * vec_width,
                    rotStorage2Ptr + i * vec_width,
                    rotStorage3Ptr + i * vec_width);

                // Rotate flux back to Cartesian frame.
                RotateFromNormalKernel<ExecSpace, NDIM>(
                    blksize, rotStorage3Ptr + i * vec_width,
                    rotMatPtr + i * vec_width, fluxPtr + i * vec_width);
            });
    }
    else
    {
        Nektar::parallel_for<ExecSpace>(
            0u, groupsize, NEKTAR_LAMBDA(const size_t i) {
                // Rotate velocity to normal.
                RotateToNormalKernel<ExecSpace, NDIM>(
                    blksize, fwdPtr + i * vec_width, rotMatPtr + i * vec_width,
                    rotStorage1Ptr + i * vec_width);
                RotateToNormalKernel<ExecSpace, NDIM>(
                    blksize, bwdPtr + i * vec_width, rotMatPtr + i * vec_width,
                    rotStorage2Ptr + i * vec_width);

                // Compute flux in rotated frame.
                RiemannKernel<ExecSpace, EoSParamType, NDIM>()(
                    EoS, blksize, rotStorage1Ptr + i * vec_width,
                    rotStorage2Ptr + i * vec_width,
                    rotStorage3Ptr + i * vec_width);

                // Rotate flux back to Cartesian frame.
                RotateFromNormalKernel<ExecSpace, NDIM>(
                    blksize, rotStorage3Ptr + i * vec_width,
                    rotMatPtr + i * vec_width, fluxPtr + i * vec_width);
            });
    }
}

template <template <typename, typename, unsigned int> typename RiemannKernel,
          typename EoSParamType, typename ExecSpace, typename TData>
class CompressibleSolverOpImpl : public CompressibleSolverOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    CompressibleSolverOpImpl(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : CompressibleSolverOp<TData>(expansionList, components),
          m_rotStorage1(Field<TData, FieldState::Phys>(
              "Fwd Rot Storage",
              GetBlockAttributes<TData, FieldState::Phys>(
                  expansionList->GetTrace()),
              components, 1)),
          m_rotStorage2(Field<TData, FieldState::Phys>(
              "Bwd Rot Storage",
              GetBlockAttributes<TData, FieldState::Phys>(
                  expansionList->GetTrace()),
              components, 1)),
          m_rotStorage3(Field<TData, FieldState::Phys>(
              "Flux Rot Storage",
              GetBlockAttributes<TData, FieldState::Phys>(
                  expansionList->GetTrace()),
              components, 1)),
          m_rotMat(Field<TData, FieldState::Phys>(
              "Rotation Matrix",
              GetBlockAttributes<TData, FieldState::Phys>(
                  expansionList->GetTrace()),
              9, 1))
    {
        m_dimension = expansionList->GetExp(0)->GetShapeDimension();

        SetUpEquationOfState(expansionList->GetSession(), m_EoS);
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<CompressibleSolverOpImpl<
            RiemannKernel, EoSParamType, ExecSpace, TData>>(expansionList,
                                                            components);
    }

    void SetEquationOfStateParams(EoSParamType EoS)
    {
        m_EoS = EoS;
    }

protected:
    Field<TData, FieldState::Phys> m_rotStorage1, m_rotStorage2, m_rotStorage3,
        m_rotMat;
    unsigned int m_dimension;
    bool m_updateRotMat = true;
    EoSParamType m_EoS;

    void v_Apply(Field<TData, FieldState::Phys> &Fwd,
                 Field<TData, FieldState::Phys> &Bwd,
                 Field<TData, FieldState::Phys> &flux) override
    {
        switch (m_dimension)
        {
            // 1D
            case 1:
            {
                this->template OperatorND<1>(Fwd, Bwd, flux);
                break;
            }
            // 2D
            case 2:
            {
                this->template OperatorND<2>(Fwd, Bwd, flux);
                break;
            }
            // 3D
            case 3:
            {
                this->template OperatorND<3>(Fwd, Bwd, flux);
                break;
            }
        }
    }

    void v_SetTraceNormals(
        Field<TData, FieldState::Phys> &traceNormals) override
    {
        this->m_traceNormals = std::move(traceNormals);
        m_updateRotMat       = true;
    }

    template <unsigned int NDIM>
    void OperatorND(Field<TData, FieldState::Phys> &Fwd,
                    Field<TData, FieldState::Phys> &Bwd,
                    Field<TData, FieldState::Phys> &flux)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < Fwd.GetBlocks().size(); ++blk)
        {
            // Initialize pointers.
            auto &fwdBlk         = Fwd.GetBlocks()[blk];
            auto &bwdBlk         = Bwd.GetBlocks()[blk];
            auto &fluxBlk        = flux.GetBlocks()[blk];
            auto &normalBlk      = this->m_traceNormals.GetBlocks()[blk];
            auto &rotStorage1Blk = m_rotStorage1.GetBlocks()[blk];
            auto &rotStorage2Blk = m_rotStorage2.GetBlocks()[blk];
            auto &rotStorage3Blk = m_rotStorage3.GetBlocks()[blk];

            auto fwdPtr    = fwdBlk.template GetPtr<MemSpace, ReadOnly>();
            auto bwdPtr    = bwdBlk.template GetPtr<MemSpace, ReadOnly>();
            auto fluxPtr   = fluxBlk.template GetPtr<MemSpace, WriteOnly>();
            auto normalPtr = normalBlk.template GetPtr<MemSpace, ReadOnly>();
            auto rotStorage1Ptr =
                rotStorage1Blk.template GetPtr<MemSpace, WriteOnly>();
            auto rotStorage2Ptr =
                rotStorage2Blk.template GetPtr<MemSpace, WriteOnly>();
            auto rotStorage3Ptr =
                rotStorage3Blk.template GetPtr<MemSpace, WriteOnly>();

            const auto blksize = fwdBlk.CompSize();

            auto rotMatPtr =
                (NDIM == 3) ? (m_updateRotMat)
                                  ? m_rotMat.GetBlocks()[blk]
                                        .template GetPtr<MemSpace, WriteOnly>()
                                  : m_rotMat.GetBlocks()[blk]
                                        .template GetPtr<MemSpace, ReadOnly>()
                            : normalPtr;

            if (m_updateRotMat)
            {
                RiemannKernelLauncher<RiemannKernel, EoSParamType, ExecSpace,
                                      NDIM, true>(
                    m_EoS, blksize, normalPtr, rotMatPtr, rotStorage1Ptr,
                    rotStorage2Ptr, rotStorage3Ptr, fwdPtr, bwdPtr, fluxPtr);
            }
            else
            {
                RiemannKernelLauncher<RiemannKernel, EoSParamType, ExecSpace,
                                      NDIM, false>(
                    m_EoS, blksize, normalPtr, rotMatPtr, rotStorage1Ptr,
                    rotStorage2Ptr, rotStorage3Ptr, fwdPtr, bwdPtr, fluxPtr);
            }
        }

        m_updateRotMat = false;
    }
};

} // namespace Nektar::Operators::detail
