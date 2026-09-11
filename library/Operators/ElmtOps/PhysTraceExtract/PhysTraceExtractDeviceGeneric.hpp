///////////////////////////////////////////////////////////////////////////////
//
// File: PhysTraceExtractDeviceGeneric.hpp
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

/**
 * @file PhysTraceExtractDeviceGeneric.hpp
 * @brief Device dispatch and kernel launchers of the sum-factorised
 * extraction of a volume field onto the element traces.
 *
 * @details
 * This header holds the Device implementation of
 * detail::PhysTraceExtractBlockOpImpl, the host-side half of the
 * operator. The constructor reads the block's two families of
 * interpolation tables out of the data warehouse and records the
 * collocation properties that select the kernels' fast paths. Each
 * application then sizes the workspace and the launch configuration,
 * normalises the storage interleave and dispatches per shape into
 * PhysTraceExtractDeviceGenericKernels.hpp. No arithmetic on the field
 * happens here.
 *
 * CMake serves Operators::Generic from this one header (the Device Generic
 * branch of library/Operators/CMakeLists.txt) and generates one translation
 * unit per shape and data type from
 * LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in. Those units define
 * the per-shape entry points declared below, expanding
 * LibUtilities/BasicUtils/Switch/BlockOpSwitchPhysTraceExtract.h.in, this
 * operator's switch template.
 *
 * Unlike IProductWRTPhysTrace, which declares a primary template and
 * partially specialises it on NektarSpaces::Device, this header defines
 * the primary PhysTraceExtractBlockOpImpl template itself, as does
 * PhysTraceExtractSerialAVXGeneric.hpp. The two never collide because a
 * generated translation unit includes exactly one of them.
 *
 * Two entry points reach the kernels:
 * - v_Apply, the bulk path, which extracts all of an element's traces
 *   in one launch per component through OperatorND
 *   ;
 * - v_ExtractTrace, the per-trace path, which extracts one named trace
 *   into a caller-chosen offset of a packed trace field.
 *
 * @note This operator has a single implementation, registered under
 * Operators::Generic, so the class is only ever built with that tag and
 * #m_implInterleaveWidth is unconditionally
 * NektarSpaces::Device::warpSize, the interleave the kernels index
 * with. A StdMat, SumFac or SumFacTOP request reaches it through
 * ElmtBlockOp::Create()'s fallback to the `"Generic"` factory key.
 * SumFacTOP in particular must not select a width-one interleave here:
 * that would feed contiguous per-element data to kernels indexing at
 * the warp size, and with one registration there is no tag left that
 * could.
 *
 * The launch geometry is a separate question from the implementation
 * tag: it is one element per warp, which is the shape
 * GetDeviceBlockSize() and GetDeviceGridSize() compute for
 * Operators::SumFac, so that is what those helpers are queried with
 * rather than with Implementation, which is Operators::Generic here and
 * names no launch shape of its own. The kernels in
 * PhysTraceExtractDeviceGenericKernels.hpp take no implementation tag
 * at all: they are written for that one launch geometry, so the tag
 * would have named nothing they read.
 *
 * @see PhysTraceExtractOp.hpp for the operator's place in the family.
 * @see PhysTraceExtractDeviceGenericKernels.hpp for the kernels launched
 * from here.
 * @see PhysTraceExtractSerialAVXGeneric.hpp for the same decomposition
 * packed for SIMD vectors instead of warp lanes.
 * @see IProductWRTPhysTraceDeviceGeneric.hpp for the adjoint operator,
 * which applies the same interpolation tables transposed and carries in
 * addition the trace weights and the trace Jacobian.
 */

#pragma once

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "Operators/ElmtOps/PhysTraceExtract/PhysTraceExtractBlockOp.hpp"

#include "Operators/ElmtOps/PhysTraceExtract/PhysTraceExtractDeviceGenericKernels.hpp"

// Selects the switch construction used by the generated ShapeBlock
// definitions (see LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in).
#define NEKTAR_BLOCKOP_SWITCH_PHYSTRACE

namespace Nektar::Operators::detail
{

/**
 * @brief Device block implementation of the trace extraction, one
 * element per warp lane.
 *
 * @details
 * For every element of the block this evaluates the volume field at the
 * quadrature points of the element's traces,
 * \f[ u(\boldsymbol\xi^{tr}) \;=\; \sum_{\boldsymbol p}
 *     u_{\boldsymbol p}\, h_{\boldsymbol p}(\boldsymbol\xi^{tr}) ,
 * \f]
 * where \f$h_{\boldsymbol p}\f$ is the tensor product of the cardinal
 * (hat) functions of the volume quadrature grid. On a direction-a trace
 * that product factorises into the normal-direction value
 * \f$h_{p_a}(\pm 1)\f$ times the hat functions of the in-trace
 * directions, and that is the two-stage structure the kernels realise:
 * the normal-direction extraction first, writing a face-shaped block
 * still on the volume grid, then the tangential interpolation of that
 * block onto the trace points. PhysTraceExtractOp.hpp names the
 * operator; this class only marshals data. The constructor caches the
 * tables, OperatorND() and LaunchExtractTraceND() fix the launch
 * geometry and the workspace for the bulk and the per-trace route, and
 * every floating-point operation lives in the kernels.
 *
 * Naming, because it trips up every new reader. #m_nm holds
 * GetNumPoints, that is volume quadrature counts per direction and not
 * mode counts. #m_B holds two different families of eInterp tables: its
 * first m_dimension entries are the normal-direction tables
 * \f$h_p(\pm 1)\f$, called ntbasis in the kernels, and the remainder
 * are the tangential tables \f$h_p(\xi^{tr})\f$, called tbasis. Neither
 * is an expansion basis evaluated anywhere.
 *
 * Layout. Field and trace data are warp interleaved: element
 * \f$e = (i_{warp}, i_{lane})\f$ holds its entry @em n at
 * `buf[warpsize * n + ilane]` inside its warp block, and warp blocks
 * stride by `numData * warpsize`. The packed trace output orders traces
 * by normal direction: the N0 pair first, then the N1 pair or single,
 * then N2, face-major within a pair. Unlike IProductWRTPhysTrace this
 * operator reads no Jacobian and no quadrature weights, because an
 * interpolation carries no surface measure, so it fetches no geometric
 * data from the warehouse at all.
 *
 * Streams. This implementation takes its
 * own stream from the block index, #m_streamID, and issues every
 * pointer fetch, reshape and launch on it, so blocks can run
 * concurrently. The workspace it uses is BlockOperator's shared static
 * buffer, which is kept per stream and so is private to the block for
 * the same reason.
 *
 * @tparam ExecSpace       NektarSpaces::Device.
 * @tparam Implementation  Operators::Generic, this operator providing
 *                         a single implementation; see the file-level
 *                         note.
 * @tparam TData           Floating-point type of the field data.
 *
 * @see PhysTraceExtractDeviceGenericKernels.hpp for the kernel layers
 * this class launches.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class PhysTraceExtractBlockOpImpl : public PhysTraceExtractBlockOp<TData>
{
    using BlockOpBase = PhysTraceExtractBlockOp<TData>;
    using MemSpace    = typename ExecSpace::memory_space;

public:
    /**
     * @brief Cache the block's shape, its volume point counts, both
     * families of interpolation tables and the two collocation
     * properties, the tables in device memory.
     *
     * @details
     * A first short loop records GetNumPoints per direction in #m_nm.
     *
     * The second loop walks the normal directions. For direction
     * @em dir it fetches the eInterp table taking that direction's
     * volume rule to the positions of its traces: the two
     * Gauss-Lobatto points \f$\xi = \pm 1\f$ where the
     * direction carries a trace pair, and the single point
     * \f$\xi = -1\f$ (a one-point eGaussLegendreWithM rule) where it
     * collapses to one trace. The table is stored tight, so it has as
     * many columns as the direction has traces. #m_endPtsCollocated
     * then records whether the direction's own quadrature rule contains
     * domain endpoints, in which case that interpolation is a Kronecker
     * delta and the extraction along the normal degenerates to
     * selecting a boundary plane.
     *
     * Where it does, an ASSERTL1 additionally requires the endpoint
     * count to equal the direction's trace count, so that a rule
     * carrying both endpoints is not accepted for a collapsed direction
     * with a single trace. That check is specific to this operator:
     * IProductWRTPhysTrace sets the same flag from the endpoint count
     * alone. Being an ASSERTL1 it is compiled out of release builds.
     *
     * The third loop walks the same normal directions in packing
     * order, taking one representative trace each, trace
     * `m_dimension - 1 - f`: edges 1 and 0 in two dimensions, faces 2,
     * 1 and 0 in three, chosen so that the loop index @em f is the
     * normal direction for every shape. For each of that trace's
     * in-trace directions @em d it records the trace quadrature count
     * in #m_nq, whether those points coincide with the volume points of
     * the volume direction the in-trace direction runs along
     * (SpatialDomains::Geometry::GetDir) in #m_isCollocated, and the
     * eInterp table from that volume direction onto the trace points in
     * #m_B. All three are indexed `2 * f + d` in three dimensions and
     * @em f in two.
     *
     * In one dimension there are no in-trace directions, so the third
     * loop's body never runs and #m_B holds the single normal-direction
     * table alone. #m_nq is given the volume point count rather than a
     * trace count, so that the generated switch can still match a
     * templated instantiation, and #m_isCollocated is given one false
     * entry, which OperatorND() forwards through
     * PhysTraceExtractKernelLauncher to
     * BwdTransSegSumFacKernelTrace, whose isCollocated parameter is
     * marked [[maybe_unused]].
     *
     * @param   block_idx       Index of this block in the expansion
     *                          list.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Warehouse the tables are fetched from.
     */
    PhysTraceExtractBlockOpImpl(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : PhysTraceExtractBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        m_streamID = block_idx + 1;

        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_dimension = exp->GetShapeDimension();

        for (unsigned int d = 0; d < m_dimension; d++)
        {
            m_nm.push_back(exp->GetNumPoints(d));
        }

        // set up end points interpolation expansion in normal direction
        for (unsigned int dir = 0; dir < m_dimension; ++dir)
        {
            auto bkey = exp->GetBasis(dir)->GetBasisKey();
            auto ntrace =
                LibUtilities::ShapeTypeNumTraceInDir[m_shapeType][dir];
            auto endPtsType = (ntrace == 1)
                                  ? LibUtilities::eGaussLegendreWithM
                                  : LibUtilities::eGaussLobattoLegendre;

            m_B.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(bkey, LibUtilities::eInterp,
                                                  ntrace, endPtsType)));

            // Check to see if end point collocated
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

        // set trace interpolation bases
        auto expPtsKeys = exp->GetPointsKeys();
        for (unsigned int f = 0; f < m_dimension; ++f)
        {
            // load basis in normal orientation
            const unsigned int tr = m_dimension - 1 - f;

            // trace directions
            for (unsigned int d = 0; d < m_dimension - 1; ++d)
            {
                const auto dir = exp->GetGeom()->GetDir(tr, d);
                auto trPtsKey  = exp->GetTraceBasisKey(tr, d).GetPointsKey();
                auto nq        = trPtsKey.GetNumPoints();

                if (expPtsKeys[dir] == trPtsKey)
                {
                    this->m_isCollocated.push_back(true);
                }
                else
                {
                    this->m_isCollocated.push_back(false);
                }

                m_nq.push_back(nq);

                // Fetch basis data.
                m_B.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                    LibUtilities::BasisDataKey<TData>(
                        exp->GetBasis(dir)->GetBasisKey(),
                        LibUtilities::eInterp, nq, trPtsKey.GetPointsType())));
            }
        }

        // spatial case for 1D
        if (m_dimension == 1)
        {
            // nq not required in 1D (formally it is 2) use nm value to
            // ensure templated version is adopted when possible
            m_nq.push_back(m_nm[0]);
            this->m_isCollocated.push_back(false);
        }
    }

    /// Registration name for BlockOperatorFactory, defined by the
    /// generated factory declaration unit.
    // className - for BlockOperatorFactory
    static std::string className;

    /// @brief Creator function registered with BlockOperatorFactory;
    /// builds one block operator for the given block of elements.
    /// Implementation is always Operators::Generic here, this operator
    /// providing a single implementation; see PhysTraceExtractBlockOp.
    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            PhysTraceExtractBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    /// Interleave width the kernels expect: a full warp, one element
    /// per lane. The Operator methods and v_ExtractTrace reshape the
    /// block storage to this width around every launch.
    static constexpr unsigned int m_implInterleaveWidth =
        NektarSpaces::Device::warpSize;

    unsigned int m_streamID;

    /// Shape shared by every element of the block.
    LibUtilities::ShapeType m_shapeType;
    /// Read by the generated dispatch to pick between the DEFORMED and
    /// the regular instantiation, but never assigned by this class or
    /// by its Serial/AVX counterpart, so only the regular branch is
    /// ever taken. Nothing here needs it: an extraction uses no
    /// geometric factors, and the DEFORMED template argument is unused
    /// in OperatorND().
    bool m_isDeformed = false;
    /// Dimension of the reference element (1, 2 or 3).
    unsigned int m_dimension;
    /// Volume quadrature points per direction, read from GetNumPoints:
    /// point counts, not mode counts. Exactly m_dimension entries, with
    /// no padding, which is safe because the generated switch reads
    /// #m_nm[2] and #m_nq[5] only in the translation units of the
    /// three-dimensional shapes.
    std::vector<unsigned int> m_nm;
    /// Trace quadrature points, indexed `2 * dir + tangential` by normal
    /// direction and tangential direction in three dimensions (six entries)
    /// and by normal direction in two (two entries). In one dimension
    /// the single entry is a copy of the volume point count.
    std::vector<unsigned int> m_nq;
    /// eInterp tables in device memory: first the m_dimension
    /// normal-direction tables \f$h_p(\pm 1)\f$, stored tight with one
    /// column per trace of that direction, then the tangential tables
    /// \f$h_p(\xi^{tr})\f$ in the same (direction, tangential) order as
    /// #m_nq. The kernels call the first group ntbasis and the second
    /// tbasis. Nine entries in three dimensions, four in two, one in
    /// one.
    std::vector<const TData *> m_B;
    /// One flag per normal direction: that direction's volume rule
    /// contains domain endpoints, so interpolating it to the trace
    /// position is a Kronecker delta and the extraction along the
    /// normal selects a single boundary plane.
    std::vector<bool> m_endPtsCollocated;
    /// Set the first time v_ExtractTrace acquires an output pointer, to
    /// distinguish a not-yet-initialised output block (taken WriteOnly)
    /// from one already holding earlier traces (taken ReadWrite, which
    /// refuses to treat a stale device copy as authoritative). It
    /// belongs to the
    /// block operator, not to any particular output block, and is never
    /// cleared.
    bool m_extractTraceInit = false;

    /**
     * @brief Number of trace quadrature points of one trace of the
     * shape, that is how many values one element contributes for
     * @p traceid.
     *
     * One in one dimension, where a trace is an endpoint; the edge's
     * point count in two; the product of the face's two point counts in
     * three. Read from the expansion rather than from #m_nq, so it is
     * exact for every trace rather than for the representative trace of
     * each normal direction.
     *
     * @param   traceid     Local trace index in the Nektar edge or face
     *                      numbering of the shape.
     *
     * @return Trace quadrature points of that trace.
     */
    unsigned int GetTraceNumPoints(const unsigned int traceid) const
    {
        ASSERTL1(traceid < this->m_exp->GetNtraces(),
                 "Trace id exceeds number of traces");

        if (m_dimension == 1)
        {
            return 1u;
        }

        if (m_dimension == 2)
        {
            return this->m_exp->GetTraceBasisKey(traceid).GetNumPoints();
        }

        return this->m_exp->GetTraceBasisKey(traceid, 0).GetNumPoints() *
               this->m_exp->GetTraceBasisKey(traceid, 1).GetNumPoints();
    }

    /**
     * @brief Per-trace path: extract trace @p traceid of every element
     * of the block into @p outblock at @p outOffset.
     *
     * @details
     * Two ASSERTL1 guards run first: the two blocks must already share
     * an interleave width, and @p outOffset plus the trace's point
     * count, taken from GetTraceNumPoints() rather than from #m_nq,
     * must fit inside the output block's per-element storage.
     *
     * Both blocks are then reshaped to #m_implInterleaveWidth before
     * the launch and back afterwards, the output as well as the input,
     * because a launch writes only its own slice of the packed trace
     * entries and must leave the slices written by earlier traces in
     * the same layout it found them.
     *
     * The output pointer is taken WriteOnly on the first call of this
     * operator's life and on every call with @p outOffset zero, and
     * ReadWrite otherwise. WriteOnly allocates the device buffer if
     * needed (zero filling it once, on that first allocation) and
     * declares the device copy authoritative without transferring
     * anything in. ReadWrite transfers the host copy in only if the
     * device copy has been invalidated, and never declares a stale
     * device copy authoritative, which is what lets the traces written
     * by earlier calls survive when this call fills a later offset. In
     * the usual sequence the device copy is already valid, so nothing
     * is transferred and what preserves them is that neither branch
     * re-zeroes an already allocated buffer. Nothing here zeroes the
     * output otherwise: the region between the packed traces is
     * whatever the caller left.
     *
     * The dispatch then builds the block's size parameter and hands
     * LaunchExtractTraceND() the index sequences that select this
     * dimension's slice of #m_B, #m_isCollocated and
     * #m_endPtsCollocated, exactly as the bulk route's OperatorND()
     * does for OperatorNDImpl(). Unlike it this route
     * reads the counts at run time only: a per-trace launch is driven
     * by a runtime shape and never reaches a compiled size
     * combination, so every arm passes a NonTemplated parameter.
     *
     * Each size parameter is built inside its own arm rather than
     * ahead of the switch. #m_nm and #m_nq are sized to the block's
     * dimension with no padding, so the three-dimensional constructor
     * would read past the end of both for a segment or a
     * quadrilateral.
     *
     * Unlike the bulk route this folds each nodal shape onto its
     * parent: NodalTri goes to the Tri arm, NodalTet and NodalPrism to
     * the Tet and Prism arms. That mapping matters in two dimensions,
     * where the leaf worker BwdTransQuadSumFacKernelTrace tests
     * `SHAPE_TYPE == LibUtilities::Tri` and would put a NodalTri on
     * the quadrilateral arm. In three dimensions it is redundant:
     * SHAPE_TYPE reaches only GetTraceFaceDispatch, whose shape lists
     * already name the nodal enums alongside their parents, and
     * nothing below it reads SHAPE_TYPE at all. The generated bulk
     * path performs no such substitution and passes the nodal enum
     * straight through.
     *
     * A segment's traces are its two end vertices, so there is nothing
     * to interpolate tangentially: its tangential pack is empty, and
     * the whole operation is the rank-1 contraction
     * `out = sum_p ntbasis0[p * 2 + traceid] * in[p]`, which collapses
     * to picking the first or the last volume point when the volume
     * rule has endpoints.
     *
     * Shapes outside the list raise a fatal error.
     *
     * @param   traceid     Local trace index in the Nektar edge or face
     *                      numbering of the shape.
     * @param   inblock     Volume field of the block.
     * @param   outblock    Packed trace field of the block.
     * @param   outOffset   Offset of this trace's values within one
     *                      element's packed trace entries, counted in
     *                      entries per element. The launchers scale it
     *                      by the warp size inside the kernel.
     *
     * @note #m_extractTraceInit is a member of the block operator, not
     * of @p outblock, and is never cleared. An operator reused on a
     * second, freshly created output block whose first extraction is at
     * a non-zero @p outOffset would take the ReadWrite branch on
     * storage that has never been initialised.
     *
     * @note Every window GetTraceFaceDispatch produces is one face
     * wide, so the loop bound cannot double as the @c ntbasis row
     * stride on this route. The general, non-endpoint-collocated arm of
     * the three-dimensional normal stage takes that stride as an
     * explicit @c tstride argument, read from
     * @c ShapeTypeNumTraceInDir, which is what lets the lower face of a
     * two-trace direction address the table correctly.
     * @see PhysExtractEndFacesN0KernelTrace3D
     */
    void v_ExtractTrace(
        const unsigned int traceid,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        const unsigned int outOffset) override
    {
        ASSERTL1(inblock.GetInterleaveWidth() == outblock.GetInterleaveWidth(),
                 "Input and output interleave widths differ");

        ASSERTL1(outOffset + GetTraceNumPoints(traceid) <=
                     outblock.GetNumData(),
                 "Trace output range exceeds output block storage");

        const auto numDataIn       = inblock.GetNumData();
        const auto numDataOut      = outblock.GetNumData();
        const auto nelmtPad        = inblock.GetNumElementsWithPadding();
        const auto interleaveWidth = inblock.GetInterleaveWidth();
        const unsigned int ncomp =
            inblock.GetNumComponents() * inblock.GetNumHomoModes();
        const unsigned int nTracePts = GetTraceNumPoints(traceid);

        // Initialize pointers.
        auto inptr    = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        TData *outptr = nullptr;
        if (!m_extractTraceInit || outOffset == 0)
        {
            outptr = outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);
            m_extractTraceInit = true;
        }
        else
        {
            outptr = outblock.template GetPtr<MemSpace, ReadWrite>(m_streamID);
        }

        // Reshape, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, interleaveWidth, nelmtPad * ncomp, numDataIn,
            (TData *)inptr, m_streamID);
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, interleaveWidth, nelmtPad * ncomp,
            numDataOut, outptr, m_streamID);

        switch (m_shapeType)
        {
            case LibUtilities::Seg:
            {
                constexpr auto seq1D =
                    std::make_integer_sequence<unsigned int, 1>();
                constexpr auto seq0D =
                    std::make_integer_sequence<unsigned int, 0>();

                LaunchExtractTraceND<LibUtilities::Seg>(
                    traceid, nelmtPad, ncomp, numDataIn, numDataOut, nTracePts,
                    inptr, outptr, outOffset,
                    NonTemplatedTraceSizeParameter1D(m_nm[0], m_nq[0]), seq1D,
                    seq0D);
                break;
            }
            case LibUtilities::Quad:
            {
                constexpr auto seq2D =
                    std::make_integer_sequence<unsigned int, 2>();

                LaunchExtractTraceND<LibUtilities::Quad>(
                    traceid, nelmtPad, ncomp, numDataIn, numDataOut, nTracePts,
                    inptr, outptr, outOffset,
                    NonTemplatedTraceSizeParameter2D(m_nm[0], m_nm[1], m_nq[0],
                                                     m_nq[1]),
                    seq2D, seq2D);
                break;
            }
            case LibUtilities::Tri:
            case LibUtilities::NodalTri:
            {
                constexpr auto seq2D =
                    std::make_integer_sequence<unsigned int, 2>();

                LaunchExtractTraceND<LibUtilities::Tri>(
                    traceid, nelmtPad, ncomp, numDataIn, numDataOut, nTracePts,
                    inptr, outptr, outOffset,
                    NonTemplatedTraceSizeParameter2D(m_nm[0], m_nm[1], m_nq[0],
                                                     m_nq[1]),
                    seq2D, seq2D);
                break;
            }
            case LibUtilities::Hex:
            {
                constexpr auto seq3D =
                    std::make_integer_sequence<unsigned int, 3>();
                constexpr auto seq6D =
                    std::make_integer_sequence<unsigned int, 6>();

                LaunchExtractTraceND<LibUtilities::Hex>(
                    traceid, nelmtPad, ncomp, numDataIn, numDataOut, nTracePts,
                    inptr, outptr, outOffset,
                    NonTemplatedTraceSizeParameter3D(m_nm[0], m_nm[1], m_nm[2],
                                                     m_nq[0], m_nq[1], m_nq[2],
                                                     m_nq[3], m_nq[4], m_nq[5]),
                    seq3D, seq6D);
                break;
            }
            case LibUtilities::Prism:
            case LibUtilities::NodalPrism:
            {
                constexpr auto seq3D =
                    std::make_integer_sequence<unsigned int, 3>();
                constexpr auto seq6D =
                    std::make_integer_sequence<unsigned int, 6>();

                LaunchExtractTraceND<LibUtilities::Prism>(
                    traceid, nelmtPad, ncomp, numDataIn, numDataOut, nTracePts,
                    inptr, outptr, outOffset,
                    NonTemplatedTraceSizeParameter3D(m_nm[0], m_nm[1], m_nm[2],
                                                     m_nq[0], m_nq[1], m_nq[2],
                                                     m_nq[3], m_nq[4], m_nq[5]),
                    seq3D, seq6D);
                break;
            }
            case LibUtilities::Pyr:
            {
                constexpr auto seq3D =
                    std::make_integer_sequence<unsigned int, 3>();
                constexpr auto seq6D =
                    std::make_integer_sequence<unsigned int, 6>();

                LaunchExtractTraceND<LibUtilities::Pyr>(
                    traceid, nelmtPad, ncomp, numDataIn, numDataOut, nTracePts,
                    inptr, outptr, outOffset,
                    NonTemplatedTraceSizeParameter3D(m_nm[0], m_nm[1], m_nm[2],
                                                     m_nq[0], m_nq[1], m_nq[2],
                                                     m_nq[3], m_nq[4], m_nq[5]),
                    seq3D, seq6D);
                break;
            }
            case LibUtilities::Tet:
            case LibUtilities::NodalTet:
            {
                constexpr auto seq3D =
                    std::make_integer_sequence<unsigned int, 3>();
                constexpr auto seq6D =
                    std::make_integer_sequence<unsigned int, 6>();

                LaunchExtractTraceND<LibUtilities::Tet>(
                    traceid, nelmtPad, ncomp, numDataIn, numDataOut, nTracePts,
                    inptr, outptr, outOffset,
                    NonTemplatedTraceSizeParameter3D(m_nm[0], m_nm[1], m_nm[2],
                                                     m_nq[0], m_nq[1], m_nq[2],
                                                     m_nq[3], m_nq[4], m_nq[5]),
                    seq3D, seq6D);
                break;
            }
            default:
                NEKERROR(ErrorUtil::efatal, "shape type not implemented");
        }

        // Reshape back, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            interleaveWidth, m_implInterleaveWidth, nelmtPad * ncomp, numDataIn,
            (TData *)inptr, m_streamID);
        LibUtilities::ReshapeStorage<ExecSpace>(
            interleaveWidth, m_implInterleaveWidth, nelmtPad * ncomp,
            numDataOut, outptr, m_streamID);

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    /**
     * @brief Per-trace path for every dimension: extract one named
     * trace of every element of the block.
     *
     * @details
     * The single-trace counterpart of OperatorND(), and its twin in
     * shape: PhysTraceExtractTraceKernelLauncher's one-, two- and
     * three-dimensional arms are one overload set, and this one body
     * launches whichever the index sequences expand to. It offsets the
     * output to `(numDataOut * iwarp + outOffset) * warpsize` so the
     * caller can place each trace in turn within the output block.
     *
     * The workspace comes from PhysTraceExtractWorkSpaceSize(), the
     * same helper and so the same bound the bulk route asks for: it is
     * written for a trace pair even though a per-trace launch fills one
     * trace, and the kernel partitions it through the helpers behind
     * that. A segment needs none and is given a null pointer,
     * deviceMalloc() returning one for a zero-sized request.
     *
     * @tparam SHAPE_TYPE   Shape of the block, with each nodal shape
     *                      folded onto its parent by v_ExtractTrace();
     *                      see there.
     * @tparam TTraceSizeParameter  Trace size parameter of this
     *                      dimension, always the non-templated one on
     *                      this route.
     * @tparam ind0         Normal directions, `0` to `dim - 1`. Selects
     *                      the normal-direction tables at the front of
     *                      #m_B and the #m_endPtsCollocated flags.
     * @tparam ind1         Tangential directions, `0` to
     *                      `dim * (dim - 1) - 1`. Selects the
     *                      tangential tables that follow them in #m_B,
     *                      offset by `sizeof...(ind0)`, and the
     *                      #m_isCollocated flags. Empty in one
     *                      dimension, where a trace is a point.
     *
     * @param   traceid     Local trace index in the Nektar edge or face
     *                      numbering of the shape.
     * @param   nelmtPad    Elements in the block including padding.
     * @param   ncomp       Components, including homogeneous planes.
     * @param   nmTot       Volume points per element, so warp blocks of
     *                      the input stride by `nmTot * warpsize`.
     * @param   numDataOut  Packed trace entries per element, so warp
     *                      blocks of the output stride by
     *                      `numDataOut * warpsize`.
     * @param   nTracePts   Trace points of this trace; sizes the launch
     *                      only, and not even that, since
     *                      GetDeviceBlockSize() ignores its argument
     *                      for Operators::SumFac and returns the warp
     *                      size.
     * @param   inptr       Volume field of the block.
     * @param   outptr      Packed trace field of the block.
     * @param   outOffset   Offset of this trace's values within one
     *                      element's packed trace entries, counted in
     *                      entries per element. The launchers scale it
     *                      by the warp size inside the kernel.
     * @param   sizeParam   Volume and trace point counts of the block.
     */
    template <LibUtilities::ShapeType SHAPE_TYPE, typename TTraceSizeParameter,
              unsigned int... ind0, unsigned int... ind1>
    NEK_FORCE_INLINE void LaunchExtractTraceND(
        const unsigned int traceid, const size_t nelmtPad,
        const unsigned int ncomp, const unsigned int nmTot,
        const unsigned int numDataOut, const unsigned int nTracePts,
        const TData *inptr, TData *outptr, const unsigned int outOffset,
        TTraceSizeParameter sizeParam,
        std::integer_sequence<unsigned int, ind0...>,
        std::integer_sequence<unsigned int, ind1...>)
    {
        // Get static workspace pointer.
        const size_t wspSize =
            PhysTraceExtractWorkSpaceSize(nelmtPad, sizeParam) * ncomp;
        auto wspptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                wspSize, m_streamID);

        // Set Kernel parameters.
        const unsigned int blocksize =
            GetDeviceBlockSize<Operators::SumFac>(nTracePts);
        const unsigned int gridsize =
            GetDeviceGridSize<Operators::SumFac>(nelmtPad, blocksize, 0);

        // PhysTraceExtract kernel.
        DEVICE_2DGRID_KERNEL_LAUNCHER(
            (PhysTraceExtractTraceKernelLauncher<SHAPE_TYPE>), gridsize, ncomp,
            blocksize, 1, 0, m_streamID, traceid, sizeParam, nelmtPad, nmTot,
            numDataOut, outOffset, m_B[ind0]..., m_B[sizeof...(ind0) + ind1]...,
            inptr, outptr, wspptr, (bool)this->m_isCollocated[ind1]...,
            (bool)m_endPtsCollocated[ind0]...);
    }

    /**
     * @brief Bulk path: extract every trace of every element of the
     * block in one launch per component, and per component and
     * homogeneous plane in one and two dimensions.
     *
     * Dispatches to ShapeBlock(), whose per-shape specialisation the
     * generated translation unit defines and which selects the
     * size-templated OperatorND() instantiation. Each nodal shape has
     * its own specialisation and its own generated unit, so SHAPE_TYPE
     * reaching OperatorND() is the nodal enumerator, not the parent one
     * that v_ExtractTrace() substitutes on the per-trace route.
     * Unhandled shapes raise a fatal error.
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
    /// select this dimension's slice of #m_B, #m_isCollocated and
    /// #m_endPtsCollocated. The first sequence runs over the normal
    /// directions, the second over the tangential directions: one and none
    /// in one dimension, two and two in two, three and six in three.
    /// @{

    /// @}

    /**
     * @brief Bulk path for every dimension: extract all of an element's
     * traces in one launch per component.
     *
     * @details
     * The whole trace loop lives in PhysTraceExtractKernelLauncher,
     * whose one-, two- and three-dimensional arms this one body
     * launches; overload resolution picks the arm from the number of
     * arguments the index sequences expand to. Each walks the normal
     * directions in packing order: in two dimensions the direction-0
     * pair first, then the direction-1 pair of a quadrilateral or the
     * single edge 0 of a triangle; in three the direction-0 pair, then
     * the direction-1 faces (one for a tetrahedron, two otherwise),
     * then the direction-2 faces (two for a hexahedron, one otherwise);
     * in one the two end vertices, written contiguously at the start of
     * each element's output.
     *
     * @c nqTotOut, from the size parameter's nqTotTrace(), reproduces
     * the packed output length to size the launch, and the workspace
     * comes from PhysTraceExtractWorkSpaceSize(), which the kernel
     * partitions per warp through the helpers behind it. A segment
     * needs none and is given a null pointer, deviceMalloc() returning
     * one for a zero-sized request.
     *
     * The component loop carries the GetNumHomoModes factor. That is
     * inert for three-dimensional elements, which support no
     * homogeneous extension and report one plane, and correct in one
     * and two dimensions, where it has every plane extracted. This is
     * the factor the corresponding IProductWRTPhysTrace bulk loop
     * omitted before the two were fused.
     *
     * @tparam SHAPE_TYPE   Shape of the block, the nodal enumerators
     *                      included: the generated bulk path passes
     *                      them straight through rather than folding
     *                      them onto their parents, as the per-trace
     *                      route does.
     * @tparam DEFORMED     Passed on but read by neither this method
     *                      nor the kernels: an extraction uses no
     *                      geometric factors.
     * @tparam TTraceSizeParameter  Trace size parameter of this
     *                      dimension, templated or not.
     * @tparam ind0         Normal directions, `0` to `dim - 1`. Selects
     *                      the normal-direction tables at the front of
     *                      #m_B and the #m_endPtsCollocated flags.
     * @tparam ind1         Tangential directions, `0` to
     *                      `dim * (dim - 1) - 1`. Selects the
     *                      tangential tables that follow them in #m_B,
     *                      offset by `sizeof...(ind0)`, and the
     *                      #m_isCollocated flags. Empty in one
     *                      dimension, where a trace is a point.
     *
     * @note Only the input is reshaped to #m_implInterleaveWidth before
     * the launch, and both are reshaped back after it. The output is
     * written wholesale by the kernel, so it needs no incoming layout,
     * but the asymmetry is worth knowing when reading the reshape
     * calls.
     *
     * @note The three-dimensional endpoint interpolation carries an
     * open defect on this bulk route, whose location has not been
     * established: the @c ntbasis row strides this route dispatches are
     * correct for every shape, so it lies elsewhere in the chain. It is
     * masked by the absence of a three-dimensional Gauss fixture, so
     * the suites stay green.
     * @see PhysExtractEndFacesN0KernelTrace3D
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
        const auto nelmt         = inblock.GetNumElementsWithPadding();
        const unsigned int nmTot = sizeParam.nmTot();
        const unsigned int ncomp =
            inblock.GetNumComponents() * inblock.GetNumHomoModes();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Get static workspace pointer.
        const size_t wspSize =
            PhysTraceExtractWorkSpaceSize(nelmt, sizeParam) * ncomp;
        auto wspptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                wspSize, m_streamID);

        // Set Kernel parameters.
        const unsigned int nqTotOut =
            sizeParam.template nqTotTrace<SHAPE_TYPE>();
        const unsigned int blocksize =
            GetDeviceBlockSize<Operators::SumFac>(nqTotOut);
        const unsigned int gridsize =
            GetDeviceGridSize<Operators::SumFac>(nelmt, blocksize, 0);

        // Reshape, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, interleaveWidth, nelmt * ncomp, nmTot,
            (TData *)inptr, m_streamID);

        // PhysTraceExtract kernel.
        DEVICE_2DGRID_KERNEL_LAUNCHER(
            (PhysTraceExtractKernelLauncher<SHAPE_TYPE>), gridsize, ncomp,
            blocksize, 1, 0, m_streamID, sizeParam, nelmt, nmTot, m_B[ind0]...,
            m_B[sizeof...(ind0) + ind1]..., inptr, outptr, wspptr,
            (bool)this->m_isCollocated[ind1]...,
            (bool)m_endPtsCollocated[ind0]...);

        // Reshape back, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            interleaveWidth, m_implInterleaveWidth, nelmt * ncomp, nmTot,
            (TData *)inptr, m_streamID);
        LibUtilities::ReshapeStorage<ExecSpace>(
            interleaveWidth, m_implInterleaveWidth, nelmt * ncomp, nqTotOut,
            outptr, m_streamID);

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
