///////////////////////////////////////////////////////////////////////////////
//
// File: CompressibleSolverOp.hpp
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
// Description: CompressibleSolver operator base class.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Operators/SolverUtilsOps/RiemannSolvers/RiemannSolverOp.hpp"

namespace Nektar::Operators
{

template <template <typename, unsigned int> typename RiemannKernel,
          typename ExecSpace, unsigned int NDIM, bool ROTMAT, typename TData>
NEK_FORCE_INLINE static void RiemannKernelLauncher(
    const size_t blksize, const TData *normalPtr, const TData *rotMatPtr,
    TData *rotStorage1Ptr, TData *rotStorage2Ptr, TData *rotStorage3Ptr,
    const TData *fwdPtr, const TData *bwdPtr, TData *fluxPtr)
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

    Nektar::parallel_for<ExecSpace>(
        0u, groupsize, NEKTAR_LAMBDA(const size_t i) {
            // Only update rotation matrice when necessary.
            if (NDIM == 3 && ROTMAT)
            {
                GenerateRotationMatrices<ExecSpace>(
                    blksize, normalPtr + i * vec_width,
                    (TData *)rotMatPtr + i * vec_width);
            }

            // Rotate velocity to normal.
            RotateToNormalKernel<ExecSpace, NDIM>(
                blksize, fwdPtr + i * vec_width, rotMatPtr + i * vec_width,
                rotStorage1Ptr + i * vec_width);
            RotateToNormalKernel<ExecSpace, NDIM>(
                blksize, bwdPtr + i * vec_width, rotMatPtr + i * vec_width,
                rotStorage2Ptr + i * vec_width);

            // Compute Lax-Friedrichs flux in rotated frame.
            RiemannKernel<ExecSpace, NDIM>()(
                blksize, rotStorage1Ptr + i * vec_width,
                rotStorage2Ptr + i * vec_width, rotStorage3Ptr + i * vec_width);

            // Rotate flux back to Cartesian frame.
            RotateFromNormalKernel<ExecSpace, NDIM>(
                blksize, rotStorage3Ptr + i * vec_width,
                rotMatPtr + i * vec_width, fluxPtr + i * vec_width);
        });
}

// CompressibleSolver operator base class
template <typename TData>
class CompressibleSolverOp : public RiemannSolverOp<TData>
{
public:
    static std::shared_ptr<CompressibleSolverOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components, const std::string &method,
        const std::string &execStr)
    {
        return std::static_pointer_cast<CompressibleSolverOp<TData>>(
            RiemannSolverOp<TData>::Create(expansionList, components, method,
                                           execStr));
    }

protected:
    Field<TData, FieldState::Phys> m_rotStorage1, m_rotStorage2, m_rotStorage3,
        m_rotMat;
    bool m_updateRotMat = false;

    CompressibleSolverOp(const MultiRegions::ExpListSharedPtr &expansionList,
                         const std::vector<std::string> &components)
        : RiemannSolverOp<TData>(expansionList, components),
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
    }

    ~CompressibleSolverOp() override = default;

    void v_SetTraceNormals(
        Field<TData, FieldState::Phys> &traceNormals) override
    {
        this->m_traceNormals = std::move(traceNormals);
        m_updateRotMat       = true;
    }

    template <template <typename, unsigned int> typename RiemannKernel,
              typename ExecSpace, unsigned int NDIM>
    void OperatorND(Field<TData, FieldState::Phys> &Fwd,
                    Field<TData, FieldState::Phys> &Bwd,
                    Field<TData, FieldState::Phys> &flux)
    {
        using MemSpace = typename ExecSpace::memory_space;

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
                RiemannKernelLauncher<RiemannKernel, ExecSpace, NDIM, true>(
                    blksize, normalPtr, rotMatPtr, rotStorage1Ptr,
                    rotStorage2Ptr, rotStorage3Ptr, fwdPtr, bwdPtr, fluxPtr);
            }
            else
            {
                RiemannKernelLauncher<RiemannKernel, ExecSpace, NDIM, false>(
                    blksize, normalPtr, rotMatPtr, rotStorage1Ptr,
                    rotStorage2Ptr, rotStorage3Ptr, fwdPtr, bwdPtr, fluxPtr);
            }
        }

        m_updateRotMat = false;
    }
};

// Helper function
template <typename TData>
NEK_DEVICE_INLINE TData GetPressure(const TData &rho, const TData &e)
{
    // Ideal gas law: P = (gamma - 1) * rho * e
    const TData gamma = 1.4; // Specific heat ratio for air
    return (gamma - 1) * rho * e;
}

template <typename TData>
NEK_DEVICE_INLINE TData GetSoundSpeed(const TData &rho, const TData &e)
{
    using std::sqrt;

    // Ideal gas law: P = (gamma - 1) * rho * e
    const TData gamma = 1.4; // Specific heat ratio for air
    NekDouble p       = GetPressure(rho, e);
    return std::sqrt(gamma * p / rho);
}

template <typename TData>
NEK_DEVICE_INLINE TData GetRoeSoundSpeed(
    [[maybe_unused]] const TData &rhoL, [[maybe_unused]] const TData &pL,
    [[maybe_unused]] const TData &eL, [[maybe_unused]] const TData &HL,
    [[maybe_unused]] const TData &srL, [[maybe_unused]] const TData &rhoR,
    [[maybe_unused]] const TData &pR, [[maybe_unused]] const TData &eR,
    [[maybe_unused]] const TData &HR, [[maybe_unused]] const TData &srR,
    const TData &HRoe, const TData &URoe2, [[maybe_unused]] const TData &srLR)
{
    using std::sqrt;

    // Specific heat ratio for air
    const TData gamma = 1.4;
    // Calculate sound speed using ideal gas relation
    return sqrt((gamma - 1.0) * (HRoe - 0.5 * URoe2));
}

} // namespace Nektar::Operators
