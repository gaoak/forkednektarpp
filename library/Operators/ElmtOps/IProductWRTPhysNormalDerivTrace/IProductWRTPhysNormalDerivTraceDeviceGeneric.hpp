///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTPhysNormalDerivTraceDeviceGeneric.hpp
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
// Description: Device block operator of the lift against the normal
// derivative of the test function
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file IProductWRTPhysNormalDerivTraceDeviceGeneric.hpp
 * @brief Device dispatch and kernel launches of the sum-factorised surface
 * inner product against the normal derivative of the volume cardinal
 * basis.
 *
 * @details
 * This header holds the Device version of
 * detail::IProductWRTPhysNormalDerivTraceBlockOpImpl, the host-side half
 * of the operator. The constructor reads the block's interpolation tables
 * and their derivatives, the trace weights, the collapsed-coordinate
 * factors and the trace derivative factors out of the data warehouse;
 * each application then sizes the workspace and the launch configuration,
 * normalises the storage interleave and dispatches per shape into
 * IProductWRTPhysNormalDerivTraceDeviceGenericKernels.hpp. No arithmetic
 * on the field happens here.
 *
 * CMake serves Operators::Generic from this one header (the Device
 * Generic branch of library/Operators/CMakeLists.txt) and generates one
 * translation unit per shape and data type from
 * LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in. Those units
 * define the per-shape entry points declared below, expanding
 * Common/BlockOpSwitchPhysTraceExtract.h.in, this operator's switch
 * template.
 *
 * This operator has a single implementation, registered under
 * Operators::Generic, so the class is only ever built with that tag and
 * #m_implInterleaveWidth is unconditionally
 * NektarSpaces::Device::warpSize, the interleave the kernels index with.
 * The launch geometry is one element per warp, which is the shape
 * GetDeviceBlockSize() and GetDeviceGridSize() compute for
 * Operators::SumFac, so that is what those helpers are queried with.
 *
 * @see IProductWRTPhysNormalDerivTraceOp.hpp for what the operator
 * computes and how the family is laid out.
 * @see IProductWRTPhysNormalDerivTraceSerialAVXGeneric.hpp for the same
 * decomposition packed for SIMD vectors instead of warp lanes.
 * @see IProductWRTPhysTraceDeviceGeneric.hpp for the plain trace lift this
 * operator is built on, whose layout it follows.
 */

#pragma once

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "LocalRegions/DataWarehouse/GeometricDataWarehouse.hpp"
#include "Operators/ElmtOps/IProductWRTPhysNormalDerivTrace/IProductWRTPhysNormalDerivTraceBlockOp.hpp"
#include "Operators/ElmtOps/IProductWRTPhysNormalDerivTrace/IProductWRTPhysNormalDerivTraceDeviceGenericKernels.hpp"

// Selects the switch construction used by the generated ShapeBlock
// definitions (see LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in).
#define NEKTAR_BLOCKOP_SWITCH_PHYSTRACE

namespace Nektar::Operators::detail
{

/**
 * @brief Device block implementation of the normal-derivative trace lift,
 * one element per warp lane.
 *
 * @details
 * The same operator as the Serial/AVX class, with field, trace and factor
 * data warp interleaved: element \f$e = (i_{warp}, i_{lane})\f$ holds its
 * entry @em n at `buf[warpsize * n + ilane]` inside its warp block, and
 * warp blocks stride by `numData * warpsize`. #m_B holds the eInterp
 * tables, normal-direction tables first and tangential tables after, and
 * #m_DB their derivatives; the kernels choose, term by term, which of the
 * two a slot reads.
 *
 * Streams. This implementation takes its own
 * stream from the block index, #m_streamID, and issues every pointer
 * fetch, reshape and launch on it, so blocks can run concurrently. The
 * workspace is BlockOperator's shared static buffer, kept per stream.
 *
 * @tparam Implementation  Operators::Generic, this operator providing a
 *                         single implementation.
 * @tparam TData           Floating-point type of the field data.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class IProductWRTPhysNormalDerivTraceBlockOpImpl
    : public IProductWRTPhysNormalDerivTraceBlockOp<TData>
{
    using BlockOpBase = IProductWRTPhysNormalDerivTraceBlockOp<TData>;
    using MemSpace    = typename ExecSpace::memory_space;

public:
    /**
     * @brief Cache the block's shape, its point counts, both families of
     * interpolation tables and their derivatives, the trace weights, the
     * collapsed-coordinate factors and the trace derivative factors in
     * device memory.
     *
     * @details
     * The loops follow the Serial/AVX constructor: the normal directions
     * first, then the tangential slots of one representative trace per
     * normal direction. #m_nm and #m_nq are finally padded with zeros to
     * three and six entries; the padded slots are never read.
     *
     * @param   block_idx       Index of this block in the expansion list;
     *                          also selects the block's factor arrays in
     *                          the warehouse.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Warehouse the tables are fetched from.
     */
    IProductWRTPhysNormalDerivTraceBlockOpImpl(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : IProductWRTPhysNormalDerivTraceBlockOp<TData>(block_idx, exp,
                                                        dataWarehouse)
    {
        m_streamID = block_idx + 1;

        // Determine shape and type of the element.
        m_shapeType  = exp->DetShapeType();
        m_dimension  = exp->GetShapeDimension();
        m_isDeformed = LocalRegions::TraceDerivFactorsArePointwise(
            m_shapeType,
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed);

        // Interpolation to the trace positions along the normal
        // direction, and its derivative.
        for (unsigned int dir = 0; dir < m_dimension; ++dir)
        {
            auto bkey = this->m_exp->GetBasis(dir)->GetBasisKey();
            m_nm.push_back(exp->GetNumPoints(dir));

            unsigned int ntrace =
                LibUtilities::ShapeTypeNumTraceInDir[m_shapeType][dir];
            auto endPtsType = (ntrace == 1)
                                  ? LibUtilities::eGaussLegendreWithM
                                  : LibUtilities::eGaussLobattoLegendre;

            m_B.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(bkey, LibUtilities::eInterp,
                                                  ntrace, endPtsType)));

            m_DB.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    bkey, LibUtilities::eInterpDerivative, ntrace,
                    endPtsType)));

            if (LibUtilities::PointsTypeNumEndPts[bkey.GetPointsType()])
            {
                m_endPtsCollocated.push_back(true);
                ASSERTL1(
                    LibUtilities::PointsTypeNumEndPts[bkey.GetPointsType()] ==
                        ntrace,
                    "Trace and number of end points in PointsType differ");
            }
            else
            {
                m_endPtsCollocated.push_back(false);
            }
        }

        // Interpolation onto the trace points along the tangential
        // directions, its derivative, and the trace weights.
        auto expPtsKeys = exp->GetPointsKeys();
        for (unsigned int dim = 0; dim < m_dimension; ++dim)
        {
            unsigned int tr = m_dimension - 1 - dim;
            for (unsigned int d = 0; d < m_dimension - 1; ++d)
            {
                auto dir      = this->m_exp->GetGeom()->GetDir(tr, d);
                auto trBKey   = exp->GetTraceBasisKey(tr, d);
                auto trPtsKey = trBKey.GetPointsKey();
                auto nq       = trPtsKey.GetNumPoints();

                this->m_isCollocated.push_back(expPtsKeys[dir] == trPtsKey);
                m_nq.push_back(nq);
                m_traceDir.push_back(dir);

                auto dirBKey = this->m_exp->GetBasis(dir)->GetBasisKey();

                m_B.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                    LibUtilities::BasisDataKey<TData>(
                        dirBKey, LibUtilities::eInterp, nq,
                        trPtsKey.GetPointsType())));

                m_DB.push_back(
                    this->m_dataWarehouse->template GetData<MemSpace>(
                        LibUtilities::BasisDataKey<TData>(
                            dirBKey, LibUtilities::eInterpDerivative, nq,
                            trPtsKey.GetPointsType())));

                m_W.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                    LibUtilities::BasisDataKey<TData>(trBKey,
                                                      LibUtilities::eWeights)));
            }
        }

        if (m_dimension == 1)
        {
            // nq is not read in 1D; the volume count keeps the generated
            // switch on a compiled instantiation.
            m_nq.push_back(m_nm[0]);
            this->m_isCollocated.push_back(false);
        }

        // Trace Jacobian times normal times geometric factors, one
        // component after the other.
        m_jacNormGeomFacPtr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::JacNormGeomFactorLocTraceKey<TData>(
                block_idx, m_implInterleaveWidth));
        m_jacptr = m_jacNormGeomFacPtr;

        // 2/(1 - eta) at the element's own points for every collapsed
        // direction; only these are safe to evaluate it on, their
        // Gauss-Radau distribution excluding eta = +1. Direction 0 is never
        // collapsed and keeps a null slot, as does any direction the shape
        // does not collapse.
        m_twoOverOneMinusZ.resize(m_dimension, nullptr);
        for (unsigned int dir = 1; dir < m_dimension; ++dir)
        {
            if (IsCollapsedDir(m_shapeType, dir))
            {
                m_twoOverOneMinusZ[dir] =
                    this->m_dataWarehouse->template GetData<MemSpace>(
                        LibUtilities::BasisDataKey<TData>(
                            this->m_exp->GetBasis(dir)->GetBasisKey(),
                            LibUtilities::eTwoOverOneMinusZero));
            }
        }

        // Add dummy entries for 3D switches AFTER 1D logic executes
        while (m_nm.size() < 3)
        {
            m_nm.push_back(0);
        }
        while (m_nq.size() < 6)
        {
            m_nq.push_back(0);
        }
    }

    /// Registration name for BlockOperatorFactory, defined by the
    /// generated factory declaration unit.
    static std::string className;

    /// @brief Creator function registered with BlockOperatorFactory;
    /// builds one block operator for the given block of elements.
    /// Implementation is always Operators::Generic here, this operator
    /// providing a single implementation; see
    /// IProductWRTPhysNormalDerivTraceBlockOp.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<IProductWRTPhysNormalDerivTraceBlockOpImpl<
            ExecSpace, Implementation, TData>>(block_idx, exp, dataWarehouse);
    }

protected:
    /// Interleave width the kernels expect: a full warp, one element per
    /// lane. OperatorNDImpl() reshapes the block storage to this width
    /// around every launch.
    static constexpr unsigned int m_implInterleaveWidth =
        NektarSpaces::Device::warpSize;

    unsigned int m_streamID;

    /// Shape shared by every element of the block.
    LibUtilities::ShapeType m_shapeType;
    /// Whether the factor array holds one value per trace quadrature point
    /// rather than one per trace: deformed geometry, or a collapsed shape
    /// whatever its geometry. Read by the generated dispatch to pick the
    /// DEFORMED instantiation, which is what fixes the name: the switch
    /// template Common/BlockOpSwitchPhysTraceExtract.h.in reads it by name.
    bool m_isDeformed = false;
    /// Dimension of the reference element (1, 2 or 3).
    unsigned int m_dimension = 0;
    /// Volume quadrature points per direction, padded with zeros to three
    /// entries.
    std::vector<unsigned int> m_nm;
    /// Trace quadrature points, indexed `2 * dir + tangential` in three
    /// dimensions and by normal direction in two, padded with zeros to six
    /// entries. In one dimension the single entry is a copy of the volume
    /// point count.
    std::vector<unsigned int> m_nq;
    /// eInterp tables in device memory: first the m_dimension
    /// normal-direction tables, then the tangential tables in the same
    /// (direction, tangential) order as #m_nq.
    std::vector<const TData *> m_B;
    /// eInterpDerivative tables in the same order as #m_B.
    std::vector<const TData *> m_DB;
    /// Trace quadrature weights per (normal direction, tangential
    /// direction), in the same order as #m_nq. Empty in one dimension.
    std::vector<const TData *> m_W;
    /// 2/(1 - eta) at the element points, per direction; null where the
    /// direction is not collapsed.
    std::vector<const TData *> m_twoOverOneMinusZ;
    /// Trace Jacobian times normal times geometric factors of the block in
    /// device memory, one component after the other, warp interleaved.
    const TData *m_jacNormGeomFacPtr = nullptr;
    /// The factor set the next apply consumes: the contracted set above by
    /// default, or one Cartesian direction's uncontracted set when a
    /// vector-input pass has selected it through v_SetActiveDir().
    const TData *m_jacptr = nullptr;
    /// Uncontracted per-direction factors, fetched on first use. Only the
    /// values differ from the contracted set, so the kernels index them
    /// unchanged.
    std::vector<const TData *> m_jacDirGeomFac;
    /// Element direction each tangential trace slot runs along, in the
    /// order of #m_nq; decides which slots carry the derivative table in
    /// each term.
    std::vector<unsigned int> m_traceDir;
    /// One flag per normal direction: that direction's volume rule
    /// contains domain endpoints.
    std::vector<bool> m_endPtsCollocated;

    /// Select the factor set for the next Apply(); see
    /// IProductWRTPhysNormalDerivTraceOp::ApplyVector(). The per-direction
    /// factors share the contracted set's layout, so this is a pointer swap
    /// and the kernels are untouched. They are fetched on first use rather
    /// than at construction, so an operator that never takes the vector
    /// path does not pay for the dim extra factor arrays.
    void v_SetActiveDir(const int dir) override
    {
        if (dir < 0)
        {
            m_jacptr = m_jacNormGeomFacPtr;
            return;
        }

        ASSERTL0(dir < static_cast<int>(m_dimension),
                 "SetActiveDir: direction beyond the shape dimension");

        if (m_jacDirGeomFac.empty())
        {
            m_jacDirGeomFac.resize(m_dimension, nullptr);
            for (unsigned int k = 0; k < m_dimension; ++k)
            {
                m_jacDirGeomFac[k] =
                    this->m_dataWarehouse->template GetData<MemSpace>(
                        LocalRegions::JacNormGeomFactorLocTraceKey<TData>(
                            this->m_block_idx, m_implInterleaveWidth,
                            static_cast<int>(k)));
            }
        }

        m_jacptr = m_jacDirGeomFac[dir];
    }

    /**
     * @brief Bulk path: lift every trace of every element of the block
     * in one launch per shape.
     *
     * Dispatches to the per-shape entry point the generated translation
     * unit defines, which selects the size-templated OperatorND()
     * instantiation.
     */
    void v_Apply(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock) override
    {
        switch (m_shapeType)
        {
            // Segment
            case LibUtilities::Seg:
            {
                ShapeBlock<LibUtilities::Seg>(inblock, outblock);
                break;
            }
            // Quadrilateral
            case LibUtilities::Quad:
            {
                ShapeBlock<LibUtilities::Quad>(inblock, outblock);
                break;
            }
            // Triangle
            case LibUtilities::Tri:
            {
                ShapeBlock<LibUtilities::Tri>(inblock, outblock);
                break;
            }
            // Nodal triangle
            case LibUtilities::NodalTri:
            {
                ShapeBlock<LibUtilities::NodalTri>(inblock, outblock);
                break;
            }
            // Hexahedron
            case LibUtilities::Hex:
            {
                ShapeBlock<LibUtilities::Hex>(inblock, outblock);
                break;
            }
            // Tetrahedron
            case LibUtilities::Tet:
            {
                ShapeBlock<LibUtilities::Tet>(inblock, outblock);
                break;
            }
            // Nodal tetrahedron
            case LibUtilities::NodalTet:
            {
                ShapeBlock<LibUtilities::NodalTet>(inblock, outblock);
                break;
            }
            // Pyramid
            case LibUtilities::Pyr:
            {
                ShapeBlock<LibUtilities::Pyr>(inblock, outblock);
                break;
            }
            // Prism
            case LibUtilities::Prism:
            {
                ShapeBlock<LibUtilities::Prism>(inblock, outblock);
                break;
            }
            // Nodal prism
            case LibUtilities::NodalPrism:
            {
                ShapeBlock<LibUtilities::NodalPrism>(inblock, outblock);
                break;
            }
            default:
                NEKERROR(ErrorUtil::efatal, "shape type not implemented");
                break;
        }
    }

    // Shape specific block operator, specialised for each shape in
    // LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in.
    template <LibUtilities::ShapeType SHAPE_TYPE>
    void ShapeBlock(typename BlockOpBase::InBlock &inblock,
                    typename BlockOpBase::OutBlock &outblock);

    // Tangential directions the kernels index in dim dimensions: none
    // for a segment, whose traces are points, and dim * (dim - 1)
    // otherwise.
    static constexpr unsigned int NumTangentialDir(const unsigned int dim)
    {
        return dim * (dim - 1);
    }

    // Generic operator. Builds the index sequences from the shape dimension
    // and forwards to OperatorNDImpl.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TTraceSizeParameter>
    NEK_FORCE_INLINE void OperatorND(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        TTraceSizeParameter sizeParam)
    {
        constexpr unsigned int DIM = LibUtilities::ShapeTypeDimMap[SHAPE_TYPE];

        static_assert(
            (DIM == 1 && IsTraceSizeParameter1D_v<TTraceSizeParameter>) ||
                (DIM == 2 && IsTraceSizeParameter2D_v<TTraceSizeParameter>) ||
                (DIM == 3 && IsTraceSizeParameter3D_v<TTraceSizeParameter>),
            "OperatorND expects a trace size parameter matching the "
            "dimension of the shape.");

        OperatorNDImpl<SHAPE_TYPE, DEFORMED>(
            inblock, outblock, sizeParam,
            std::make_integer_sequence<unsigned int, DIM>(),
            std::make_integer_sequence<unsigned int, NumTangentialDir(DIM)>());
    }

    /**
     * @brief Bulk path for every dimension: lift every trace of every
     * element of the block in one launch, components along the second
     * grid axis.
     *
     * @details
     * The whole trace loop lives in
     * IProductWRTPhysNormalDerivTraceKernelLauncher, whose one-, two- and
     * three-dimensional arms this one body launches; overload resolution picks
     * the arm from the number of arguments the index sequences expand to.
     *
     * The three per-element strides come from the size parameter alone.
     * @c numDataIn, the packed trace entries of one element, is its
     * nqTotTrace(); @c numDataOut, the volume points, its nmTot();
     * @c numDataJac follows the factor layout: one value per trace point
     * when pointwise, one per trace otherwise. @c jacCompStride is the
     * distance between the components of the factor array, a whole block
     * of those entries. The workspace is BlockOperator's shared static
     * buffer, sized by IProductWRTPhysNormalDerivTraceWorkSpaceSize() and
     * partitioned per warp by the kernels through the same helpers.
     *
     * The output is always reshaped on the way in: the kernels accumulate
     * the later terms onto the first inside the warp layout.
     *
     * @tparam ind0   Normal directions, `0` to `dim - 1`. Selects the
     *                normal-direction tables at the front of #m_B and
     *                #m_DB, the collapsed factors and the
     *                #m_endPtsCollocated flags.
     * @tparam ind1   Tangential directions, `0` to `dim * (dim - 1) - 1`.
     *                Selects the tangential tables that follow them,
     *                offset by `sizeof...(ind0)`, and the #m_W weights,
     *                #m_traceDir and collocation flags. Empty in one
     *                dimension.
     */
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TTraceSizeParameter, unsigned int... ind0,
              unsigned int... ind1>
    NEK_FORCE_INLINE void OperatorNDImpl(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        TTraceSizeParameter sizeParam,
        std::integer_sequence<unsigned int, ind0...>,
        std::integer_sequence<unsigned int, ind1...>)
    {
        const auto nelmtPad = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr =
            (this->m_append)
                ? outblock.template GetPtr<MemSpace, ReadWrite>(m_streamID)
                : outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);

        // Get interleave parameter.
        const auto inInterleaveWidth  = inblock.GetInterleaveWidth();
        const auto outInterleaveWidth = outblock.GetInterleaveWidth();

        const unsigned int ncomp =
            inblock.GetNumComponents() * inblock.GetNumHomoModes();

        // Per-element strides of the packed trace input, the volume-shaped
        // output and the factor array.
        const unsigned int numDataIn =
            sizeParam.template nqTotTrace<SHAPE_TYPE>();
        const unsigned int numDataOut = sizeParam.nmTot();
        const unsigned int numDataJac =
            (DEFORMED) ? numDataIn
                       : LibUtilities::ShapeTypeNumTraces[SHAPE_TYPE];
        const size_t jacCompStride = (size_t)nelmtPad * (size_t)numDataJac;

        // Set Kernel parameters.
        const unsigned int blocksize =
            GetDeviceBlockSize<Operators::SumFac>(sizeParam.nmMax());
        const unsigned int gridsize =
            GetDeviceGridSize<Operators::SumFac>(nelmtPad, blocksize, 0);

        // Get static workspace pointer.
        const size_t wspSize =
            IProductWRTPhysNormalDerivTraceWorkSpaceSize(nelmtPad, sizeParam);
        auto wspptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                wspSize * ncomp, m_streamID);

        // Reshape, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, inInterleaveWidth, nelmtPad * ncomp,
            numDataIn, (TData *)inptr, m_streamID);
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, outInterleaveWidth, nelmtPad * ncomp,
            numDataOut, outptr, m_streamID);

        // IProductWRTPhysNormalDerivTrace kernel.
        DEVICE_2DGRID_KERNEL_LAUNCHER(
            (IProductWRTPhysNormalDerivTraceKernelLauncher<SHAPE_TYPE,
                                                           DEFORMED>),
            gridsize, ncomp, blocksize, 1, 0, m_streamID, sizeParam,
            m_B[ind0]..., m_DB[ind0]..., nelmtPad, numDataIn, numDataOut,
            numDataJac, jacCompStride, m_B[sizeof...(ind0) + ind1]...,
            m_DB[sizeof...(ind0) + ind1]..., m_W[ind1]...,
            m_twoOverOneMinusZ[ind0]..., m_jacptr, wspptr, inptr, outptr,
            m_traceDir[ind1]..., (bool)this->m_isCollocated[ind1]...,
            (bool)m_endPtsCollocated[ind0]..., this->m_append);

        // Reshape back, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            inInterleaveWidth, m_implInterleaveWidth, nelmtPad * ncomp,
            numDataIn, (TData *)inptr, m_streamID);
        LibUtilities::ReshapeStorage<ExecSpace>(
            inInterleaveWidth, m_implInterleaveWidth, nelmtPad * ncomp,
            numDataOut, outptr, m_streamID);

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
