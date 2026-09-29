////////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTPhysTraceDeviceSumFac.hpp
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
////////////////////////////////////////////////////////////////////////////////

/**
 * @file IProductWRTPhysTraceDeviceSumFac.hpp
 * @brief Device dispatch and kernel launchers of the sum-factorised
 * surface inner product against the volume cardinal basis.
 *
 * @details
 * This header holds the Device specialisation of
 * detail::IProductWRTPhysTraceBlockOpImpl, the host-side half of the
 * operator. The constructor reads the block's interpolation tables,
 * trace weights and trace Jacobian out of the data warehouse; each
 * application then sizes the workspace and the launch configuration,
 * normalises the storage interleave and dispatches per shape into
 * IProductWRTPhysTraceDeviceSumFacKernels.hpp. No arithmetic on the
 * field happens here.
 *
 * CMake serves Operators::SumFac and Operators::SumFacTOP from this one
 * header (the Device SumFac branch of library/Operators/CMakeLists.txt)
 * and generates one translation unit per shape and data type from
 * LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in. Those units
 * define the per-shape entry points declared below, expanding
 * LibUtilities/BasicUtils/Switch/BlockOpSwitchPhysTraceExtract.h.in,
 * this operator's switch template.
 *
 * Two entry points reach the kernels:
 * - v_Apply, the bulk path, which lifts all of an element's traces in
 *   one launch per component through OperatorND
 *   ;
 * - v_IProductWRTPhysTrace, the per-trace path, which lifts one named
 *   trace out of a packed trace field, zeroing the volume array or
 *   accumulating onto it according to #m_append as the bulk path does.
 *
 * @note Two implementations are registered from this header:
 * Operators::SumFac, one element per warp lane at the warp-size
 * interleave, and Operators::SumFacTOP, one element per thread block
 * at width one. Implementation selects the kernel family, the launch
 * geometry GetDeviceBlockSize() and GetDeviceGridSize() compute and
 * #m_implInterleaveWidth. A StdMat request reaches the SumFac class
 * through IProductWRTPhysTraceBlockOp::Create(), which maps it.
 *
 * @see IProductWRTPhysTraceOp.hpp for what the operator computes and
 * how the family is laid out.
 * @see IProductWRTPhysTraceDeviceSumFacKernels.hpp for the kernels
 * launched from here.
 * @see IProductWRTPhysTraceSerialAVXSumFac.hpp for the same
 * decomposition packed for SIMD vectors instead of warp lanes.
 * @see PhysTraceExtractDeviceSumFac.hpp for the adjoint operator, which
 * applies the same interpolation tables untransposed.
 */

#pragma once

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "Operators/ElmtOps/IProductWRTPhysTrace/IProductWRTPhysTraceBlockOp.hpp"

#include "Operators/ElmtOps/IProductWRTPhysTrace/IProductWRTPhysTraceDeviceSumFacKernels.hpp"
#include "Operators/ElmtOps/IProductWRTPhysTrace/IProductWRTPhysTraceDeviceSumFacTOPKernels.hpp"

// Selects the switch construction used by the generated ShapeBlock
// definitions (see LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in).
#define NEKTAR_BLOCKOP_SWITCH_PHYSTRACE

namespace Nektar::Operators::detail
{

/**
 * @brief Device block implementation of the trace lifting term, one
 * element per warp lane (Operators::SumFac) or per thread block
 * (Operators::SumFacTOP).
 *
 * @details
 * For every element of the block this accumulates
 * \f[ \Lambda_{\boldsymbol p} \mathrel{+}= \sum_{F \in \partial E}
 *     \int_F \hat f \; h_{\boldsymbol p}\big|_F \; J_F \, \mathrm{d}s ,
 * \f]
 * the trace field tested against each cardinal (hat) function of the
 * volume quadrature grid. IProductWRTPhysTraceOp.hpp derives this and
 * says what the result is and is not. The class itself only marshals
 * data: the constructor caches the tables, OperatorND() and the
 * per-trace launcher fix the launch
 * geometry and the workspace, and every floating-point operation lives
 * in the kernels.
 *
 * Naming, because it trips up every new reader. #m_nm holds
 * GetNumPoints, that is volume quadrature counts per direction and not
 * mode counts. #m_B holds two different families of eInterp tables:
 * its first m_dimension entries are the normal-direction tables
 * \f$h_p(\pm 1)\f$, called nbasis in the kernels, and the remainder are
 * the tangential tables \f$h_p(\xi^{tr})\f$, called tbasis. Neither is
 * an expansion basis evaluated anywhere.
 *
 * Layout. Under Operators::SumFac field, trace and Jacobian data are
 * warp interleaved: element \f$e = (i_{warp}, i_{lane})\f$ holds its
 * entry @em n at `buf[warpsize * n + ilane]` inside its warp block, and
 * warp blocks stride by `numData * warpsize`; under Operators::SumFacTOP
 * an element's entries are contiguous at width one. That is why every
 * per-element offset handed to a kernel is scaled by
 * #m_implInterleaveWidth. The packed
 * trace input, and the deformed trace Jacobian, order traces by normal
 * direction: the N0 pair first, then the N1 pair or single, then N2,
 * face-major within a pair. A regular Jacobian carries one slot per
 * trace in that same order, which is the permutation
 * #m_shapeTraceIDtoJacOff and the face kernel's local copies of it
 * apply.
 *
 * Streams. This implementation takes its
 * own stream from the block index, #m_streamID, and issues every
 * pointer fetch, reshape and launch on it, so blocks can run
 * concurrently. The workspace it uses is BlockOperator's shared static
 * buffer, which is kept per stream and so is private to the block for
 * the same reason.
 *
 * @tparam Implementation  Operators::SumFac or Operators::SumFacTOP,
 *                         selecting the kernel family; see the file-level
 *                         note.
 * @tparam TData           Floating-point type of the field data.
 *
 * @see IProductWRTPhysTraceDeviceSumFacKernels.hpp for the kernel
 * layers this class launches.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class IProductWRTPhysTraceBlockOpImpl
    : public IProductWRTPhysTraceBlockOp<TData>
{
    using BlockOpBase = IProductWRTPhysTraceBlockOp<TData>;
    using MemSpace    = typename ExecSpace::memory_space;

public:
    /**
     * @brief Cache the block's shape, its point counts, both families
     * of interpolation tables, the trace weights and the trace Jacobian
     * in device memory.
     *
     * @details
     * The first loop walks the normal directions. For direction @em dir
     * it records GetNumPoints in #m_nm and fetches the eInterp table
     * taking that direction's volume rule to the positions of its
     * traces: the two Gauss-Lobatto points \f$\xi = \pm 1\f$ where the
     * direction carries a trace pair, and the single point
     * \f$\xi = -1\f$ (a one-point eGaussLegendreWithM rule) where it
     * collapses to one trace. The table is stored tight, so it has as
     * many columns as the direction has traces, which is the stride
     * every kernel indexes it with: the lift kernels take it as an
     * explicit @c tstride argument rather than from their trace-loop
     * bound, which is one on the per-trace route.
     * #m_endPtsCollocated then records whether
     * the direction's own quadrature rule contains domain endpoints, in
     * which case that interpolation is a Kronecker delta and the lift
     * along the normal degenerates to writing a single boundary plane.
     *
     * The second loop walks the same normal directions in packing
     * order, taking one representative trace each, trace
     * `m_dimension - 1 - dim`: edges 1 and 0 in two dimensions, faces
     * 2, 1 and 0 in three, chosen so that the loop index @em dim is the
     * normal direction for every shape. For each of that trace's
     * in-trace directions @em d it records the trace quadrature count
     * in #m_nq, whether those points coincide with the volume points of
     * the volume direction the in-trace direction runs along
     * (SpatialDomains::Geometry::GetDir) in #m_isCollocated, the
     * eInterp table from that volume direction onto the trace points in
     * #m_B, and the trace rule's weights in #m_W. All three are
     * indexed `2 * dim + d` in three dimensions and `dim` in two.
     *
     * In one dimension there are no in-trace directions, so the second
     * loop body never runs and no trace Jacobian is fetched: a segment
     * trace is a point and carries no surface measure. #m_nq is given
     * the volume point count instead, purely so that the generated
     * switch can still match a templated instantiation. #m_nm and
     * #m_nq are finally padded with zeros to three and six entries; the
     * padded slots are never read, since a three-dimensional block
     * fills all of them and the lower-dimensional paths never reach
     * a three-dimensional OperatorND() or index the upper entries.
     *
     * @param   block_idx       Index of this block in the expansion
     *                          list; also selects the block's trace
     *                          Jacobian in the warehouse.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Warehouse the tables are fetched from.
     */
    IProductWRTPhysTraceBlockOpImpl(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : IProductWRTPhysTraceBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        m_streamID = block_idx + 1;

        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
        m_dimension = exp->GetShapeDimension();

        // Fetch interpolation from element points to endpoints (Normal
        // Direction)
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

        // Fetch Trace bases and weights (Tangential Direction)
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

                m_B.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                    LibUtilities::BasisDataKey<TData>(
                        this->m_exp->GetBasis(dir)->GetBasisKey(),
                        LibUtilities::eInterp, nq, trPtsKey.GetPointsType())));

                m_W.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                    LibUtilities::BasisDataKey<TData>(trBKey,
                                                      LibUtilities::eWeights)));
            }
        }

        if (m_dimension > 1)
        {
            auto jackey = LocalRegions::JacobianLocTraceKey<TData>(
                block_idx, m_implInterleaveWidth);
            m_jacptr =
                this->m_dataWarehouse->template GetData<MemSpace>(jackey);
        }
        else
        {
            m_nq.push_back(m_nm[0]);
            this->m_isCollocated.push_back(false);
        }

        // Add dummy entries for 3D switches AFTER 1D logic executes
        while (m_nm.size() < 3)
            m_nm.push_back(0);
        while (m_nq.size() < 6)
            m_nq.push_back(0);
    }

    /// Registration name for BlockOperatorFactory, defined by the
    /// generated factory declaration unit.
    static std::string className;

    /// @brief Creator function registered with BlockOperatorFactory;
    /// builds one block operator for the given block of elements.
    /// Implementation is the tag this class is registered under; see
    /// IProductWRTPhysTraceBlockOp.
    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            IProductWRTPhysTraceBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    /// Interleave width the kernels expect: the warp size for
    /// Operators::SumFac, one element per lane, and one for
    /// Operators::SumFacTOP, one element per thread block. The Operator
    /// methods reshape the block storage to this width around every
    /// launch; the per-trace launchers scale their per-element offsets
    /// by it.
    static constexpr unsigned int m_implInterleaveWidth =
        std::is_same_v<Implementation, Operators::SumFac>
            ? NektarSpaces::Device::warpSize
            : 1u;

    unsigned int m_streamID;

    /// Shape shared by every element of the block.
    LibUtilities::ShapeType m_shapeType;
    /// Whether the block's geometry is deformed, so that the trace
    /// Jacobian holds one value per trace quadrature point rather than
    /// one per trace. Read by the generated dispatch to pick the
    /// DEFORMED instantiation and by the per-trace launchers.
    bool m_isDeformed = false;
    /// Dimension of the reference element (1, 2 or 3).
    unsigned int m_dimension = 0;
    /// Volume quadrature points per direction, read from GetNumPoints:
    /// point counts, not mode counts, padded with zeros to three
    /// entries. A three-dimensional block fills all three itself, so the
    /// padding only ever pads entries the lower-dimensional paths do not
    /// read.
    std::vector<unsigned int> m_nm;
    /// Trace quadrature points, indexed `2 * dir + tangential` by normal
    /// direction and tangential direction in three dimensions and by normal
    /// direction in two, padded with zeros to six entries. In one
    /// dimension the single entry is a copy of the volume point count.
    std::vector<unsigned int> m_nq;
    /// eInterp tables in device memory: first the m_dimension
    /// normal-direction tables \f$h_p(\pm 1)\f$, stored tight with one
    /// column per trace of that direction, then the tangential tables
    /// \f$h_p(\xi^{tr})\f$ in the same (direction, tangential) order as
    /// #m_nq. The kernels call the first group nbasis and the second
    /// tbasis.
    std::vector<const TData *> m_B;
    /// Trace quadrature weights per (normal direction, tangential
    /// direction), in the same order as #m_nq. Empty in one dimension.
    std::vector<const TData *> m_W;
    /// Trace Jacobian of the block in device memory, interleaved at
    /// #m_implInterleaveWidth and ordered by normal direction then trace: one
    /// value per trace quadrature point when deformed, one per trace otherwise.
    /// Null in one dimension, where a trace is a point and carries no measure.
    const TData *m_jacptr = nullptr;
    /// One flag per normal direction: that direction's volume rule
    /// contains domain endpoints, so interpolating it to the trace
    /// position is a Kronecker delta and the lift along the normal
    /// writes a single boundary plane. Set whenever the rule carries
    /// endpoints at all; an ASSERTL1 then requires, in debug builds,
    /// that their count equal the direction's trace count, mirroring
    /// the guard in PhysTraceExtract.
    std::vector<bool> m_endPtsCollocated;
    /// Trace id to regular-geometry Jacobian slot: entry `traceid` is
    /// the position that trace occupies in the packed trace order,
    /// direction by direction and trace within direction. One row per
    /// shape from LibUtilities::Tri onwards, so the row index is
    /// `m_shapeType - LibUtilities::Tri`. The two-dimensional rows are
    /// read by LaunchIProductWRTPhysTraceND()'s regular-geometry
    /// branch; the device face kernel keeps local copies of the same
    /// permutations (faceHex/face5/faceTet), matched by hand.
    static constexpr unsigned int
        m_shapeTraceIDtoJacOff[LibUtilities::SIZE_ShapeType][6] = {
            {2, 1, 0, 0, 0, 0}, // Tri
            {2, 1, 3, 0, 0, 0}, // Quad
            {3, 2, 1, 0, 0, 0}, // Tet
            {4, 2, 1, 3, 0, 0}, // Pyr
            {4, 2, 1, 3, 0, 0}, // Prism
            {4, 2, 1, 3, 0, 5}, // Hex
            {2, 1, 0, 0, 0, 0}, // NodalTri
            {3, 2, 1, 0, 0, 0}, // NodalTet
            {4, 2, 1, 3, 0, 0}  // NodalPrism
        };

    /**
     * @brief Per-trace path: lift trace @p traceid of every element of
     * the block, accumulating into @p outblock.
     *
     * @details
     * Both blocks are reshaped to #m_implInterleaveWidth before the
     * launch and back afterwards, the output as well as the input,
     * because the kernels add into the output rather than overwrite it.
     * The dispatch then builds the block's size parameter and hands
     * LaunchIProductWRTPhysTraceND() the index sequences that select
     * this dimension's slice of #m_B, #m_W, #m_isCollocated and
     * #m_endPtsCollocated, exactly as the bulk route's OperatorND()
     * does for OperatorNDImpl(). Unlike it this route
     * reads the counts at run time only: a per-trace launch is driven
     * by a runtime shape and never reaches a compiled size
     * combination, so every arm passes a NonTemplated parameter.
     *
     * Each size parameter is built inside its own arm rather than
     * ahead of the switch, matching v_ExtractTrace(); here #m_nm and
     * #m_nq are padded to three and six entries, so the choice is one
     * of symmetry rather than of safety.
     *
     * Each nodal shape is folded onto its parent, as PhysTraceExtract's
     * per-trace route does. Nothing on this path distinguishes them:
     * the two agree in ShapeTypeNumTraceInDir, ShapeTypeNumTraces and
     * every row of #m_shapeTraceIDtoJacOff, and the workers test the
     * shape they are handed only for Quad, Hex and the tetrahedra,
     * which they group with their nodal twin already. A segment has no arm at
     * all, its traces being points that carry no surface measure, and falls to
     * the fatal default.
     *
     * Both arms always run their kernel in appending mode: the caller
     * is responsible for zeroing @p outblock before the first trace.
     *
     * The workspace is BlockOperator's shared static device buffer,
     * sized here for the whole block by
     * IProductWRTPhysTraceWorkSpaceSize(). In two dimensions a lane
     * needs one IProductWRTPhysTraceEdgeModeBlockSize(), the trace-mode
     * buffer the edge inner product writes and the lift then reads,
     * hence the @c ntrace of one this path asks for. In three
     * dimensions it needs a face's mode block,
     * IProductWRTPhysTraceFaceModeBlockSize(), plus
     * IProductWRTPhysTraceFaceScratchSize() for the general
     * (non-collocated) face contraction. The kernels partition against
     * the same helpers, so host and device cannot drift apart.
     *
     * @param   traceid     Local trace index in the Nektar edge or face
     *                      numbering of the shape.
     * @param   inblock     Packed trace field of the block.
     * @param   inOffset    Offset of this trace's values within one
     *                      element's packed trace entries, counted in
     *                      entries per element; the launchers scale it
     *                      by #m_implInterleaveWidth. It also selects
     *                      the deformed Jacobian, whose packing
     *                      matches the trace field's.
     * @param   outblock    Volume-shaped output block, accumulated
     *                      into.
     *
     * @note Shapes outside the handled list, that is segments, raise
     * a fatal error.
     */
    void v_IProductWRTPhysTrace(
        const unsigned int traceid,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        const unsigned int inOffset,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock) override
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

        // Reshape, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, inInterleaveWidth, nelmtPad * ncomp,
            inblock.GetNumData(), (TData *)inptr, m_streamID);
        if (this->m_append)
        {
            LibUtilities::ReshapeStorage<ExecSpace>(
                m_implInterleaveWidth, outInterleaveWidth, nelmtPad * ncomp,
                outblock.GetNumData(), outptr, m_streamID);
        }

        switch (m_shapeType)
        {
            case LibUtilities::Quad:
            {
                constexpr auto seq2D =
                    std::make_integer_sequence<unsigned int, 2>();

                if (m_isDeformed)
                {
                    LaunchIProductWRTPhysTraceND<LibUtilities::Quad, true>(
                        traceid, inOffset, nelmtPad, ncomp, inptr, outptr,
                        NonTemplatedTraceSizeParameter2D(m_nm[0], m_nm[1],
                                                         m_nq[0], m_nq[1]),
                        seq2D, seq2D);
                }
                else
                {
                    LaunchIProductWRTPhysTraceND<LibUtilities::Quad, false>(
                        traceid, inOffset, nelmtPad, ncomp, inptr, outptr,
                        NonTemplatedTraceSizeParameter2D(m_nm[0], m_nm[1],
                                                         m_nq[0], m_nq[1]),
                        seq2D, seq2D);
                }
                break;
            }
            case LibUtilities::Tri:
            case LibUtilities::NodalTri:
            {
                constexpr auto seq2D =
                    std::make_integer_sequence<unsigned int, 2>();

                if (m_isDeformed)
                {
                    LaunchIProductWRTPhysTraceND<LibUtilities::Tri, true>(
                        traceid, inOffset, nelmtPad, ncomp, inptr, outptr,
                        NonTemplatedTraceSizeParameter2D(m_nm[0], m_nm[1],
                                                         m_nq[0], m_nq[1]),
                        seq2D, seq2D);
                }
                else
                {
                    LaunchIProductWRTPhysTraceND<LibUtilities::Tri, false>(
                        traceid, inOffset, nelmtPad, ncomp, inptr, outptr,
                        NonTemplatedTraceSizeParameter2D(m_nm[0], m_nm[1],
                                                         m_nq[0], m_nq[1]),
                        seq2D, seq2D);
                }
                break;
            }
            case LibUtilities::Hex:
            {
                constexpr auto seq3D =
                    std::make_integer_sequence<unsigned int, 3>();
                constexpr auto seq6D =
                    std::make_integer_sequence<unsigned int, 6>();

                if (m_isDeformed)
                {
                    LaunchIProductWRTPhysTraceND<LibUtilities::Hex, true>(
                        traceid, inOffset, nelmtPad, ncomp, inptr, outptr,
                        NonTemplatedTraceSizeParameter3D(
                            m_nm[0], m_nm[1], m_nm[2], m_nq[0], m_nq[1],
                            m_nq[2], m_nq[3], m_nq[4], m_nq[5]),
                        seq3D, seq6D);
                }
                else
                {
                    LaunchIProductWRTPhysTraceND<LibUtilities::Hex, false>(
                        traceid, inOffset, nelmtPad, ncomp, inptr, outptr,
                        NonTemplatedTraceSizeParameter3D(
                            m_nm[0], m_nm[1], m_nm[2], m_nq[0], m_nq[1],
                            m_nq[2], m_nq[3], m_nq[4], m_nq[5]),
                        seq3D, seq6D);
                }
                break;
            }
            case LibUtilities::Prism:
            case LibUtilities::NodalPrism:
            {
                constexpr auto seq3D =
                    std::make_integer_sequence<unsigned int, 3>();
                constexpr auto seq6D =
                    std::make_integer_sequence<unsigned int, 6>();

                if (m_isDeformed)
                {
                    LaunchIProductWRTPhysTraceND<LibUtilities::Prism, true>(
                        traceid, inOffset, nelmtPad, ncomp, inptr, outptr,
                        NonTemplatedTraceSizeParameter3D(
                            m_nm[0], m_nm[1], m_nm[2], m_nq[0], m_nq[1],
                            m_nq[2], m_nq[3], m_nq[4], m_nq[5]),
                        seq3D, seq6D);
                }
                else
                {
                    LaunchIProductWRTPhysTraceND<LibUtilities::Prism, false>(
                        traceid, inOffset, nelmtPad, ncomp, inptr, outptr,
                        NonTemplatedTraceSizeParameter3D(
                            m_nm[0], m_nm[1], m_nm[2], m_nq[0], m_nq[1],
                            m_nq[2], m_nq[3], m_nq[4], m_nq[5]),
                        seq3D, seq6D);
                }
                break;
            }
            case LibUtilities::Pyr:
            {
                constexpr auto seq3D =
                    std::make_integer_sequence<unsigned int, 3>();
                constexpr auto seq6D =
                    std::make_integer_sequence<unsigned int, 6>();

                if (m_isDeformed)
                {
                    LaunchIProductWRTPhysTraceND<LibUtilities::Pyr, true>(
                        traceid, inOffset, nelmtPad, ncomp, inptr, outptr,
                        NonTemplatedTraceSizeParameter3D(
                            m_nm[0], m_nm[1], m_nm[2], m_nq[0], m_nq[1],
                            m_nq[2], m_nq[3], m_nq[4], m_nq[5]),
                        seq3D, seq6D);
                }
                else
                {
                    LaunchIProductWRTPhysTraceND<LibUtilities::Pyr, false>(
                        traceid, inOffset, nelmtPad, ncomp, inptr, outptr,
                        NonTemplatedTraceSizeParameter3D(
                            m_nm[0], m_nm[1], m_nm[2], m_nq[0], m_nq[1],
                            m_nq[2], m_nq[3], m_nq[4], m_nq[5]),
                        seq3D, seq6D);
                }
                break;
            }
            case LibUtilities::Tet:
            case LibUtilities::NodalTet:
            {
                constexpr auto seq3D =
                    std::make_integer_sequence<unsigned int, 3>();
                constexpr auto seq6D =
                    std::make_integer_sequence<unsigned int, 6>();

                if (m_isDeformed)
                {
                    LaunchIProductWRTPhysTraceND<LibUtilities::Tet, true>(
                        traceid, inOffset, nelmtPad, ncomp, inptr, outptr,
                        NonTemplatedTraceSizeParameter3D(
                            m_nm[0], m_nm[1], m_nm[2], m_nq[0], m_nq[1],
                            m_nq[2], m_nq[3], m_nq[4], m_nq[5]),
                        seq3D, seq6D);
                }
                else
                {
                    LaunchIProductWRTPhysTraceND<LibUtilities::Tet, false>(
                        traceid, inOffset, nelmtPad, ncomp, inptr, outptr,
                        NonTemplatedTraceSizeParameter3D(
                            m_nm[0], m_nm[1], m_nm[2], m_nq[0], m_nq[1],
                            m_nq[2], m_nq[3], m_nq[4], m_nq[5]),
                        seq3D, seq6D);
                }
                break;
            }
            default:
                NEKERROR(ErrorUtil::efatal, "shape type not implemented");
        }

        // Reshape back, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            inInterleaveWidth, m_implInterleaveWidth, nelmtPad * ncomp,
            inblock.GetNumData(), (TData *)inptr, m_streamID);
        LibUtilities::ReshapeStorage<ExecSpace>(
            inInterleaveWidth, m_implInterleaveWidth, nelmtPad * ncomp,
            outblock.GetNumData(), outptr, m_streamID);

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
    }

    // Helper functions to launch kernels

    /**
     * @brief Per-trace path for every dimension: lift one named trace
     * of every element of the block, accumulating into the output.
     *
     * @details
     * The single-trace counterpart of OperatorND(), and its twin in
     * shape: IProductWRTPhysTraceTraceKernelLauncher's two- and
     * three-dimensional arms are one overload set, and this one body
     * launches whichever the index sequences expand to. Both arms
     * always run in appending mode; the caller is responsible for
     * zeroing the output before the first trace.
     *
     * The three per-element strides come from the size parameter
     * alone, as they do on the bulk route. @c numDataIn, the packed
     * trace entries of one element, is its nqTotTrace(). @c numDataOut,
     * the volume points, is its nmTot(). @c numDataJac follows the
     * geometry: the trace Jacobian holds one value per trace quadrature
     * point when deformed, which is @c numDataIn again, and one per
     * trace otherwise, which is @c ShapeTypeNumTraces.
     *
     * The two geometry branches differ in the DEFORMED instantiation
     * they launch and in where the Jacobian pointer is aimed. A
     * deformed Jacobian is packed exactly like the trace field, so it
     * takes the same @p inOffset; a regular one has one slot per trace,
     * looked up in #m_shapeTraceIDtoJacOff. Both offsets are scaled by
     * #m_implInterleaveWidth, the interleave the data is stored at.
     *
     * The workspace is BlockOperator's shared static buffer, sized by
     * IProductWRTPhysTraceWorkSpaceSize(), which the SumFac kernels
     * partition per warp; the SumFacTOP kernels take the same budget
     * from dynamic shared memory instead. This path covers a single
     * trace, hence the @c ntrace of one in two dimensions; in three the
     * two regions are sized per face already, so the bulk and the
     * per-trace budgets coincide.
     *
     * @tparam SHAPE_TYPE   Shape of the block, with each nodal shape
     *                      folded onto its parent by
     *                      v_IProductWRTPhysTrace(), as
     *                      PhysTraceExtract's per-trace route also
     *                      does; see there.
     * @tparam DEFORMED     Trace Jacobian varies point by point.
     * @tparam TTraceSizeParameter  Trace size parameter of this
     *                      dimension, always the non-templated one on
     *                      this route.
     * @tparam ind0         Normal directions, `0` to `dim - 1`. Selects
     *                      the normal-direction tables at the front of
     *                      #m_B and the #m_endPtsCollocated flags.
     * @tparam ind1         Tangential directions, `0` to
     *                      `dim * (dim - 1) - 1`. Selects the
     *                      tangential tables that follow them in #m_B,
     *                      and the #m_W weights and #m_isCollocated
     *                      flags.
     *
     * @param   traceid     Local trace index in the Nektar edge or face
     *                      numbering of the shape.
     * @param   inOffset    Offset of this trace's values within one
     *                      element's packed trace entries.
     * @param   nelmtPad    Elements in the block including padding.
     * @param   ncomp       Components, including homogeneous planes.
     * @param   inptr       Packed trace field of the block.
     * @param   outptr      Volume-shaped output, accumulated into.
     * @param   sizeParam   Volume and trace point counts of the block.
     */
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TTraceSizeParameter, unsigned int... ind0,
              unsigned int... ind1>
    NEK_FORCE_INLINE void LaunchIProductWRTPhysTraceND(
        const unsigned int traceid, const unsigned int inOffset,
        const size_t nelmtPad, const unsigned int ncomp, const TData *inptr,
        TData *outptr, TTraceSizeParameter sizeParam,
        std::integer_sequence<unsigned int, ind0...>,
        std::integer_sequence<unsigned int, ind1...>)
    {
        // Per-element strides of the packed trace input, the
        // volume-shaped output and the trace Jacobian.
        const unsigned int numDataIn =
            sizeParam.template nqTotTrace<SHAPE_TYPE>();
        const unsigned int numDataOut = sizeParam.nmTot();
        const unsigned int numDataJac =
            (DEFORMED) ? numDataIn
                       : LibUtilities::ShapeTypeNumTraces[SHAPE_TYPE];

        // Get static workspace pointer. This path covers a single
        // trace, hence the ntrace of one in two dimensions; in three
        // the regions are sized per face already.
        size_t wspSize         = 0;
        unsigned int shmemsize = 0;
        if constexpr (IsTraceSizeParameter2D_v<TTraceSizeParameter>)
        {
            wspSize = IProductWRTPhysTraceWorkSpaceSize<Implementation>(
                nelmtPad, sizeParam, 1u);
            shmemsize = sizeof(TData) *
                        IProductWRTPhysTraceSharedMemorySize<Implementation>(
                            sizeParam, 1u);
        }
        else
        {
            wspSize = IProductWRTPhysTraceWorkSpaceSize<Implementation>(
                nelmtPad, sizeParam);
            shmemsize =
                sizeof(TData) *
                IProductWRTPhysTraceSharedMemorySize<Implementation>(sizeParam);
        }
        auto wspptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                wspSize * ncomp, m_streamID);

        // Set Kernel parameters.
        const unsigned int blocksize =
            GetDeviceBlockSize<Implementation>(sizeParam.nmTot());
        const unsigned int gridsize =
            GetDeviceGridSize<Implementation>(nelmtPad, blocksize, shmemsize);

        // A deformed Jacobian is packed like the trace field and takes
        // the same offset; a regular one has one slot per trace.
        const unsigned int jacOffset =
            (DEFORMED) ? inOffset
                       : m_shapeTraceIDtoJacOff[SHAPE_TYPE - LibUtilities::Tri]
                                               [traceid];

        // IProductWRTPhysTrace kernel.
        DEVICE_2DGRID_KERNEL_LAUNCHER(
            (IProductWRTPhysTraceTraceKernelLauncher<Implementation, DEFORMED>),
            gridsize, ncomp, blocksize, 1, shmemsize, m_streamID, traceid,
            SHAPE_TYPE, sizeParam, m_B[ind0]..., nelmtPad, numDataIn,
            numDataOut, numDataJac, m_B[sizeof...(ind0) + ind1]...,
            m_W[ind1]..., m_jacptr + jacOffset * m_implInterleaveWidth, wspptr,
            inptr + inOffset * m_implInterleaveWidth, outptr,
            (bool)this->m_isCollocated[ind1]...,
            (bool)m_endPtsCollocated[ind0]..., this->m_append);
    }

    /**
     * @brief Bulk path: lift every trace of every element of the block
     * in one launch per component.
     *
     * Dispatches to the per-shape entry point the generated
     * translation unit defines, which selects the size-templated
     * OperatorND instantiation.
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

        // sizeParam is built by the switch in
        // LibUtilities/BasicUtils/Switch/BlockOpSwitchPhysTraceExtract.h.in:
        // this operator's trace size parameters carry trace point counts per
        // normal direction.
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

    /// @name Dimensional entry points
    /// Each builds the block's size parameter, non-templated where the
    /// generated switch found no compiled combination and templated
    /// where it did, and hands OperatorND() the index sequences that
    /// select this dimension's slice of #m_B, #m_W, #m_isCollocated and
    /// #m_endPtsCollocated. The first sequence runs over the normal
    /// directions, the second over the tangential directions: one and none
    /// in one dimension, two and two in two, three and six in three.
    /// @{

    /// @}

    /**
     * @brief Bulk path for every dimension: lift every trace of every
     * element of the block in one launch per component.
     *
     * @details
     * The whole trace loop lives in IProductWRTPhysTraceKernelLauncher, whose
     * one-, two- and three-dimensional arms this one body launches;
     * overload resolution picks the arm from the number of arguments
     * the index sequences expand to. Each walks the packed traces of an
     * element in the order the extraction wrote them: in one dimension
     * the two end vertices, whose lift is the rank-1
     * `out[p] += nbasis0[2 * p] * in[0] + nbasis0[2 * p + 1] * in[1]`;
     * in two the direction-0 pair then the direction-1 pair of a
     * quadrilateral or the single edge 0 of a triangle; in three the
     * faces one at a time, in packed order.
     *
     * The three sizes the kernels need to walk the interleaved storage
     * come from the size parameter alone. @c numDataIn, the packed
     * trace entries of one element, is its nqTotTrace(). @c numDataOut,
     * the volume points, is its nmTot(). @c numDataJac follows the
     * geometry: the trace Jacobian holds one value per trace quadrature
     * point when deformed, which is @c numDataIn again, and one per
     * trace otherwise, which is @c ShapeTypeNumTraces. That single
     * expression replaces the per-shape spellings the two- and
     * three-dimensional paths carried separately; they agreed already.
     *
     * The workspace is BlockOperator's shared static buffer, sized by
     * IProductWRTPhysTraceWorkSpaceSize(), which the SumFac kernels
     * partition per warp; the SumFacTOP kernels take the same budget
     * from dynamic shared memory instead. Its two-dimensional arm
     * defaults to an @c ntrace of two, this path integrating a whole
     * edge pair in one call; a segment needs no workspace at all and is
     * given a null pointer, deviceMalloc() returning one for a
     * zero-sized request.
     *
     * The flag IProductWRTPhysTraceBlockOp::m_append reaches the
     * kernel, which zeroes the volume array first unless it is set, and
     * decides whether the output is taken ReadWrite or WriteOnly and
     * whether it is reshaped on the way in.
     *
     * @tparam SHAPE_TYPE   Shape of the block, the nodal enumerators
     *                      included. A NodalPrism shares the prism's
     *                      trace structure and so takes the five-face
     *                      arm of the kernel's tables.
     * @tparam DEFORMED     Trace Jacobian varies point by point.
     * @tparam TTraceSizeParameter  Trace size parameter of this
     *                      dimension, templated or not.
     * @tparam ind0         Normal directions, `0` to `dim - 1`. Selects
     *                      the normal-direction tables at the front of
     *                      #m_B and the #m_endPtsCollocated flags.
     * @tparam ind1         Tangential directions, `0` to
     *                      `dim * (dim - 1) - 1`. Selects the
     *                      tangential tables that follow them in #m_B,
     *                      offset by `sizeof...(ind0)`, and the #m_W
     *                      weights and #m_isCollocated flags. Empty in
     *                      one dimension, where a trace is a point and
     *                      carries no surface measure.
     *
     * @note The component loop carries the GetNumHomoModes factor. That
     * is inert for three-dimensional elements, which support no
     * homogeneous extension and report one plane.
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

        // Get static workspace pointer.
        const unsigned int ncomp =
            inblock.GetNumComponents() * inblock.GetNumHomoModes();
        const size_t wspSize =
            IProductWRTPhysTraceWorkSpaceSize<Implementation>(nelmtPad,
                                                              sizeParam);
        auto wspptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                wspSize * ncomp, m_streamID);

        // Get interleave parameter.
        const auto inInterleaveWidth  = inblock.GetInterleaveWidth();
        const auto outInterleaveWidth = outblock.GetInterleaveWidth();

        // Per-element strides of the packed trace input, the
        // volume-shaped output and the trace Jacobian.
        const unsigned int numDataIn =
            sizeParam.template nqTotTrace<SHAPE_TYPE>();
        const unsigned int numDataOut = sizeParam.nmTot();
        const unsigned int numDataJac =
            (DEFORMED) ? numDataIn
                       : LibUtilities::ShapeTypeNumTraces[SHAPE_TYPE];

        // Set Kernel parameters.
        const unsigned int shmemsize =
            sizeof(TData) *
            IProductWRTPhysTraceSharedMemorySize<Implementation>(sizeParam);
        const unsigned int blocksize =
            GetDeviceBlockSize<Implementation>(sizeParam.nmTot());
        const unsigned int gridsize =
            GetDeviceGridSize<Implementation>(nelmtPad, blocksize, shmemsize);

        // Reshape, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, inInterleaveWidth, nelmtPad * ncomp,
            numDataIn, (TData *)inptr, m_streamID);
        if (this->m_append)
        {
            LibUtilities::ReshapeStorage<ExecSpace>(
                m_implInterleaveWidth, outInterleaveWidth, nelmtPad * ncomp,
                numDataOut, outptr, m_streamID);
        }

        // IProductWRTPhysTrace kernel.
        DEVICE_2DGRID_KERNEL_LAUNCHER(
            (IProductWRTPhysTraceKernelLauncher<SHAPE_TYPE, Implementation,
                                                DEFORMED>),
            gridsize, ncomp, blocksize, 1, shmemsize, m_streamID, sizeParam,
            m_B[ind0]..., nelmtPad, numDataIn, numDataOut, numDataJac,
            m_B[sizeof...(ind0) + ind1]..., m_W[ind1]..., m_jacptr, wspptr,
            inptr, outptr, (bool)this->m_isCollocated[ind1]...,
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
