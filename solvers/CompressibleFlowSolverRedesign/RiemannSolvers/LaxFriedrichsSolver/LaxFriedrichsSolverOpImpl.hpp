///////////////////////////////////////////////////////////////////////////////
//
// File: LaxFriedrichsSolverOpImpl.hpp
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
// Description: LaxFriedrichs Riemann solver.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "RiemannSolvers/LaxFriedrichsSolver/LaxFriedrichsSolverOp.hpp"

#include "Operators/Utils/UtilsKernels.hpp"
#include "RiemannSolvers/LaxFriedrichsSolver/LaxFriedrichsSolverKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class LaxFriedrichsSolverOpImpl : public LaxFriedrichsSolverOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    LaxFriedrichsSolverOpImpl(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : LaxFriedrichsSolverOp<TData>(expansionList, components),
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
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<LaxFriedrichsSolverOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth =
        std::is_same_v<ExecSpace, NektarSpaces::Device>
            ? NektarSpaces::vector_width<ExecSpace, TData>::value
            : 1u;
    unsigned int m_dimension;
    Field<TData, FieldState::Phys> m_rotStorage1, m_rotStorage2, m_rotStorage3,
        m_rotMat;

    void v_Apply(Field<TData, FieldState::Phys> &Fwd,
                 Field<TData, FieldState::Phys> &Bwd,
                 Field<TData, FieldState::Phys> &flux) override
    {
        switch (m_dimension)
        {
            // 1D
            case 1:
            {
                Operator1D(Fwd, Bwd, flux);
                break;
            }
            // 2D
            case 2:
            {
                Operator2D(Fwd, Bwd, flux);
                break;
            }
            // 3D
            case 3:
            {
                Operator3D(Fwd, Bwd, flux);
                break;
            }
        }
    }

    void Operator1D(Field<TData, FieldState::Phys> &Fwd,
                    Field<TData, FieldState::Phys> &Bwd,
                    Field<TData, FieldState::Phys> &flux)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < Fwd.GetBlocks().size(); ++blk)
        {
            // Initialize pointers.
            auto &fwdBlk         = Fwd.GetBlocks()[blk];
            auto &bwdBlk         = Bwd.GetBlocks()[blk];
            auto &normalBlk      = this->m_traceNormals.GetBlocks()[blk];
            auto &fluxBlk        = flux.GetBlocks()[blk];
            auto &rotStorage1Blk = m_rotStorage1.GetBlocks()[blk];
            auto &rotStorage2Blk = m_rotStorage2.GetBlocks()[blk];
            auto &rotStorage3Blk = m_rotStorage3.GetBlocks()[blk];

            auto fwdPtr    = fwdBlk.template GetPtr<MemSpace, ReadOnly>();
            auto bwdPtr    = bwdBlk.template GetPtr<MemSpace, ReadOnly>();
            auto normalPtr = normalBlk.template GetPtr<MemSpace, ReadOnly>();

            auto fluxPtr = fluxBlk.template GetPtr<MemSpace, WriteOnly>();
            auto rotStorage1Ptr =
                rotStorage1Blk.template GetPtr<MemSpace, WriteOnly>();
            auto rotStorage2Ptr =
                rotStorage2Blk.template GetPtr<MemSpace, WriteOnly>();
            auto rotStorage3Ptr =
                rotStorage3Blk.template GetPtr<MemSpace, WriteOnly>();

            const auto blksize = fwdBlk.CompSize();

            // Rotate velocity to normal.
            RotateToNormal1DKernel<ExecSpace>(blksize, fwdPtr, normalPtr,
                                              rotStorage1Ptr);
            RotateToNormal1DKernel<ExecSpace>(blksize, bwdPtr, normalPtr,
                                              rotStorage2Ptr);

            // Compute Lax-Friedrichs flux in rotated frame.
            LaxFriedrichsSolverKernel<ExecSpace, TData, 1>(
                blksize, rotStorage1Ptr, rotStorage2Ptr, rotStorage3Ptr);

            // Rotate flux back to Cartesian frame.
            RotateFromNormal1DKernel<ExecSpace>(blksize, rotStorage3Ptr,
                                                normalPtr, fluxPtr);
        }
    }

    void Operator2D(Field<TData, FieldState::Phys> &Fwd,
                    Field<TData, FieldState::Phys> &Bwd,
                    Field<TData, FieldState::Phys> &flux)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < Fwd.GetBlocks().size(); ++blk)
        {
            // Initialize pointers.
            auto &fwdBlk         = Fwd.GetBlocks()[blk];
            auto &bwdBlk         = Bwd.GetBlocks()[blk];
            auto &normalBlk      = this->m_traceNormals.GetBlocks()[blk];
            auto &fluxBlk        = flux.GetBlocks()[blk];
            auto &rotStorage1Blk = m_rotStorage1.GetBlocks()[blk];
            auto &rotStorage2Blk = m_rotStorage2.GetBlocks()[blk];
            auto &rotStorage3Blk = m_rotStorage3.GetBlocks()[blk];

            auto fwdPtr    = fwdBlk.template GetPtr<MemSpace, ReadOnly>();
            auto bwdPtr    = bwdBlk.template GetPtr<MemSpace, ReadOnly>();
            auto normalPtr = normalBlk.template GetPtr<MemSpace, ReadOnly>();

            auto fluxPtr = fluxBlk.template GetPtr<MemSpace, WriteOnly>();
            auto rotStorage1Ptr =
                rotStorage1Blk.template GetPtr<MemSpace, WriteOnly>();
            auto rotStorage2Ptr =
                rotStorage2Blk.template GetPtr<MemSpace, WriteOnly>();
            auto rotStorage3Ptr =
                rotStorage3Blk.template GetPtr<MemSpace, WriteOnly>();

            const auto blksize = fwdBlk.CompSize();

            // Rotate velocity to normal.
            RotateToNormal2DKernel<ExecSpace>(blksize, fwdPtr, normalPtr,
                                              rotStorage1Ptr);
            RotateToNormal2DKernel<ExecSpace>(blksize, bwdPtr, normalPtr,
                                              rotStorage2Ptr);

            // Compute Lax-Friedrichs flux in rotated frame.
            LaxFriedrichsSolverKernel<ExecSpace, TData, 2>(
                blksize, rotStorage1Ptr, rotStorage2Ptr, rotStorage3Ptr);

            // Rotate flux back to Cartesian frame.
            RotateFromNormal2DKernel<ExecSpace>(blksize, rotStorage3Ptr,
                                                normalPtr, fluxPtr);
        }
    }

    void Operator3D(Field<TData, FieldState::Phys> &Fwd,
                    Field<TData, FieldState::Phys> &Bwd,
                    Field<TData, FieldState::Phys> &flux)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < Fwd.GetBlocks().size(); ++blk)
        {
            // Initialize pointers.
            auto &fwdBlk         = Fwd.GetBlocks()[blk];
            auto &bwdBlk         = Bwd.GetBlocks()[blk];
            auto &normalBlk      = this->m_traceNormals.GetBlocks()[blk];
            auto &fluxBlk        = flux.GetBlocks()[blk];
            auto &rotStorage1Blk = m_rotStorage1.GetBlocks()[blk];
            auto &rotStorage2Blk = m_rotStorage2.GetBlocks()[blk];
            auto &rotStorage3Blk = m_rotStorage3.GetBlocks()[blk];

            auto fwdPtr    = fwdBlk.template GetPtr<MemSpace, ReadOnly>();
            auto bwdPtr    = bwdBlk.template GetPtr<MemSpace, ReadOnly>();
            auto normalPtr = normalBlk.template GetPtr<MemSpace, ReadOnly>();

            auto fluxPtr = fluxBlk.template GetPtr<MemSpace, WriteOnly>();
            auto rotStorage1Ptr =
                rotStorage1Blk.template GetPtr<MemSpace, WriteOnly>();
            auto rotStorage2Ptr =
                rotStorage2Blk.template GetPtr<MemSpace, WriteOnly>();
            auto rotStorage3Ptr =
                rotStorage3Blk.template GetPtr<MemSpace, WriteOnly>();
            auto rotMatPtr = m_rotMat.GetBlocks()[blk]
                                 .template GetPtr<MemSpace, WriteOnly>();

            const auto blksize = fwdBlk.CompSize();

            GenerateRotationMatrices<ExecSpace>(blksize, normalPtr, rotMatPtr);

            // Rotate velocity to normal.
            RotateToNormal3DKernel<ExecSpace>(blksize, fwdPtr, rotMatPtr,
                                              rotStorage1Ptr);
            RotateToNormal3DKernel<ExecSpace>(blksize, bwdPtr, rotMatPtr,
                                              rotStorage2Ptr);

            // Compute Lax-Friedrichs flux in rotated frame.
            LaxFriedrichsSolverKernel<ExecSpace, TData, 3>(
                blksize, rotStorage1Ptr, rotStorage2Ptr, rotStorage3Ptr);

            // Rotate flux back to Cartesian frame.
            RotateFromNormal3DKernel<ExecSpace>(blksize, rotStorage3Ptr,
                                                rotMatPtr, fluxPtr);
        }
    }
};

} // namespace Nektar::Operators::detail
