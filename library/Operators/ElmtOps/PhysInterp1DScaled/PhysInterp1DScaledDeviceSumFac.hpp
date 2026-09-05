///////////////////////////////////////////////////////////////////////////////
//
// File: PhysInterp1DScaledDeviceSumFac.hpp
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
// Description:
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "Operators/ElmtOps/PhysInterp1DScaled/PhysInterp1DScaledBlockOp.hpp"

#include "Operators/ElmtOps/PhysInterp1DScaled/PhysInterp1DScaledDeviceSumFacKernels.hpp"
#include "Operators/ElmtOps/PhysInterp1DScaled/PhysInterp1DScaledDeviceSumFacTOPKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class PhysInterp1DScaledBlockOpImpl : public PhysInterp1DScaledBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    PhysInterp1DScaledBlockOpImpl(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : PhysInterp1DScaledBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        m_streamID = block_idx + 1;

        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
        m_dimension = exp->GetShapeDimension();
        m_coordDim  = exp->GetCoordim();

        // Flag for collapsed coordinate correction.
        m_isModified = (exp->GetBasisType(0) == LibUtilities::eModified_A);

        for (unsigned int d = 0; d < m_dimension; d++)
        {
            m_nm.push_back(exp->GetNumPoints(d));
        }

        m_index = {nullptr, nullptr};
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            PhysInterp1DScaledBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth =
        std::is_same_v<Implementation, Operators::SumFac>
            ? NektarSpaces::Device::warpSize
            : 1u;

    unsigned int m_streamID;
    LibUtilities::ShapeType m_shapeType;
    bool m_isDeformed;
    bool m_isModified;
    unsigned int m_dimension;
    unsigned int m_coordDim;
    std::vector<unsigned int> m_nm;
    std::vector<unsigned int> m_nq;
    std::vector<const TData *> m_B;
    std::vector<const unsigned int *> m_index;

    void v_Apply(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock) override
    {
        ASSERTL1(this->m_scale != -1.0,
                 "Scale factor has not been initialised");
        switch (m_shapeType)
        {
            // Segment
            case LibUtilities::Seg:
            {
                SegBlock(inblock, outblock);
                break;
            }
            // Quads
            case LibUtilities::Quad:
            {
                QuadBlock(inblock, outblock);
                break;
            }
            // Triangles
            case LibUtilities::Tri:
            {
                TriBlock(inblock, outblock);
                break;
            }
            // Nodal Triangles
            case LibUtilities::NodalTri:
            {
                NodalTriBlock(inblock, outblock);
                break;
            }
            // Hexes
            case LibUtilities::Hex:
            {
                HexBlock(inblock, outblock);
                break;
            }
            // Tet
            case LibUtilities::Tet:
            {
                TetBlock(inblock, outblock);
                break;
            }
            // Nodal Tet
            case LibUtilities::NodalTet:
            {
                NodalTetBlock(inblock, outblock);
                break;
            }
            // Pyr
            case LibUtilities::Pyr:
            {
                PyrBlock(inblock, outblock);
                break;
            }
            // Prism
            case LibUtilities::Prism:
            {
                PrismBlock(inblock, outblock);
                break;
            }
            // Nodal Prism
            case LibUtilities::NodalPrism:
            {
                NodalPrismBlock(inblock, outblock);
                break;
            }
            default:
                std::cout << "shapetype not implemented" << std::endl;
        }
    }

    void v_SetScaleFactor(const TData &scale) override
    {
        this->m_scale = scale;
        m_nq.clear();
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            // Fetch element size.
            if (d == 0)
            {
                m_nq.push_back(this->m_scale * m_nm[0]);
            }
            else if (d == 1)
            {
                // if delta between nm0 and nm1 is 1 then keep this delta
                // for new poitns to capitalise on switch templating
                const auto nq1 =
                    (m_nm[0] - m_nm[1] == 1)
                        ? (unsigned int)(this->m_scale * m_nm[0]) - 1
                        : (unsigned int)(this->m_scale * m_nm[1]);
                m_nq.push_back(nq1);
            }
            else if (d == 2)
            {
                const auto nq2 =
                    (m_nm[0] - m_nm[2] == 1)
                        ? (unsigned int)(this->m_scale * m_nm[0]) - 1
                        : (unsigned int)(this->m_scale * m_nm[2]);
                m_nq.push_back(nq2);
            }

            // Fetch basis data.
            m_B.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    this->m_exp->GetBasis(d)->GetBasisKey(),
                    LibUtilities::eInterp, m_nq[d])));
        }
    }

    void SegBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock);

    void QuadBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock);

    void TriBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock);

    void NodalTriBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock);

    void HexBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock);

    void PrismBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock);

    void NodalPrismBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock);

    void PyrBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock);

    void TetBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock);

    void NodalTetBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock);

    // Non-size based operator.
    void Operator1D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        OperatorND<LibUtilities::Seg>(
            inblock, outblock, NonTemplatedSizeParameter1D(m_nm[0], m_nq[0]),
            std::make_integer_sequence<unsigned int, 1>(),
            std::make_integer_sequence<unsigned int, 0>());
    }

    // Size based template version.
    template <unsigned int nm0, unsigned int nq0>
    void Operator1D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        OperatorND<LibUtilities::Seg>(
            inblock, outblock, TemplatedSizeParameter1D<nm0, nq0>(),
            std::make_integer_sequence<unsigned int, 1>(),
            std::make_integer_sequence<unsigned int, 0>());
    }

    // Non-size based operator.
    void Operator2D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        const unsigned int nmTot = LibUtilities::GetNumberOfCoefficients(
            LibUtilities::Quad, m_nm[0], m_nm[1]);

        OperatorND<LibUtilities::Quad>(
            inblock, outblock,
            NonTemplatedSizeParameter2D(m_nm[0], m_nm[1], nmTot, m_nq[0],
                                        m_nq[1]),
            std::make_integer_sequence<unsigned int, 2>(),
            std::make_integer_sequence<unsigned int, 0>());
    }

    // Size based template version.
    template <unsigned int nm0, unsigned int nm1, unsigned int nq0,
              unsigned int nq1>
    void Operator2D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        constexpr unsigned int nmTot =
            LibUtilities::GetNumberOfCoefficients(LibUtilities::Quad, nm0, nm1);

        OperatorND<LibUtilities::Quad>(
            inblock, outblock,
            TemplatedSizeParameter2D<nm0, nm1, nmTot, nq0, nq1>(),
            std::make_integer_sequence<unsigned int, 2>(),
            std::make_integer_sequence<unsigned int, 0>());
    }

    // Non-size based operator.
    void Operator3D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        const unsigned int nmTot = LibUtilities::GetNumberOfCoefficients(
            LibUtilities::Hex, m_nm[0], m_nm[1], m_nm[2]);

        OperatorND<LibUtilities::Hex>(
            inblock, outblock,
            NonTemplatedSizeParameter3D(m_nm[0], m_nm[1], m_nm[2], nmTot,
                                        m_nq[0], m_nq[1], m_nq[2]),
            std::make_integer_sequence<unsigned int, 3>(),
            std::make_integer_sequence<unsigned int, 2>());
    }

    // Size based template version.
    template <unsigned int nm0, unsigned int nm1, unsigned int nm2,
              unsigned int nq0, unsigned int nq1, unsigned int nq2>
    void Operator3D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        constexpr unsigned int nmTot = LibUtilities::GetNumberOfCoefficients(
            LibUtilities::Hex, nm0, nm1, nm2);

        OperatorND<LibUtilities::Hex>(
            inblock, outblock,
            TemplatedSizeParameter3D<nm0, nm1, nm2, nmTot, nq0, nq1, nq2>(),
            std::make_integer_sequence<unsigned int, 3>(),
            std::make_integer_sequence<unsigned int, 2>());
    }

    // Generic operator implementation.
    template <LibUtilities::ShapeType SHAPE_TYPE, typename TSizeParameter,
              unsigned int... ind0, unsigned int... ind1>
    NEK_FORCE_INLINE void OperatorND(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        TSizeParameter sizeParam, std::integer_sequence<unsigned int, ind0...>,
        std::integer_sequence<unsigned int, ind1...>)
    {
        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);

        // Get static workspace pointer.
        const unsigned int ncomp =
            inblock.GetNumComponents() * inblock.GetNumHomoModes();
        auto wspSize =
            BwdTransWorkSpaceSize<SHAPE_TYPE, Implementation>(nelmt, sizeParam);
        auto wspptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                wspSize * ncomp, m_streamID);

        const TData *nodToMod = nullptr;

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int shmemsize =
            sizeof(TData) *
            BwdTransSharedMemorySize<SHAPE_TYPE, Implementation>(sizeParam);
        const unsigned int blocksize =
            GetDeviceBlockSize<Implementation>(sizeParam.nmTot());
        const unsigned int gridsize =
            GetDeviceGridSize<Implementation>(nelmt, blocksize, shmemsize);

        // Reshape, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, interleaveWidth, nelmt * ncomp,
            inblock.GetNumData(), (TData *)inptr, m_streamID);

        // BwdTrans kernel.
        DEVICE_2DGRID_KERNEL_LAUNCHER(
            (BwdTransKernelLauncher<SHAPE_TYPE, Implementation, false>),
            gridsize, ncomp, blocksize, 1, shmemsize, m_streamID, sizeParam,
            nelmt, false, m_index[ind1]..., m_B[ind0]..., nodToMod, inptr,
            outptr, wspptr);

        // Reshape back, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            interleaveWidth, m_implInterleaveWidth, nelmt * ncomp,
            inblock.GetNumData(), (TData *)inptr, m_streamID);
        LibUtilities::ReshapeStorage<ExecSpace>(
            interleaveWidth, m_implInterleaveWidth, nelmt * ncomp,
            outblock.GetNumData(), outptr, m_streamID);

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
