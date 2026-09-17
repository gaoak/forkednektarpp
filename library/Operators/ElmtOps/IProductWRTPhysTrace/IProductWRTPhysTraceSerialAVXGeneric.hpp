///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTPhysTraceSerialAVXGeneric.hpp
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
// Description: interp in physical space by a scaled number of points
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file IProductWRTPhysTraceSerialAVXGeneric.hpp
 * @brief Serial and AVX dispatch and kernel drivers of the
 * sum-factorised surface inner product against the volume cardinal
 * basis.
 *
 * @details
 * This header defines the primary
 * detail::IProductWRTPhysTraceBlockOpImpl template, which serves the
 * Serial and the AVX execution spaces; the Device space is served by the
 * partial specialisation in IProductWRTPhysTraceDeviceGeneric.hpp. The
 * two headers are the same operator with a different packing of the
 * elements, SIMD vector lanes here and warp lanes there. The vector type
 * is `tinysimd::simd<TData>` for AVX and `tinysimd::scalarT<TData>`, of
 * width one, for Serial, so a Serial build walks one element at a time
 * in plain scalars and every interleave step below degenerates to a
 * no-op.
 *
 * The constructor caches the block's interpolation tables, trace weights
 * and trace Jacobian from the data warehouse. Each application then
 * allocates its workspace, brings the block storage to `simd_t::width` a
 * chunk at a time and calls the kernels of
 * IProductWRTPhysTraceSerialAVXGenericKernels.hpp per shape. No
 * arithmetic on the field happens here.
 *
 * CMake generates one translation unit per shape and data type from
 * LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in (the Serial/AVX
 * Generic branch of library/Operators/CMakeLists.txt). Those units define the
 * per-shape entry points declared below, expanding
 * LibUtilities/BasicUtils/Switch/BlockOpSwitchPhysTraceExtract.h.in, this
 * operator's switch template. This operator registers a single implementation,
 * so the class is only ever instantiated for Operators::Generic and its
 * Implementation template parameter is not read anywhere.
 *
 * Two entry points reach the kernels:
 * - v_Apply, the bulk path, which lifts all of an element's traces
 *   through OperatorND;
 * - v_IProductWRTPhysTrace, the per-trace path, which lifts one named
 *   trace out of a packed trace field, zeroing the volume array or
 *   accumulating onto it according to #m_append as the bulk path does.
 *
 * @note The per-trace path is not implemented for segments: it raises a
 * fatal NEKERROR for LibUtilities::Seg, in both its deformed and its
 * regular branch.
 *
 * @see IProductWRTPhysTraceOp.hpp for what the operator computes and how
 * the family is laid out.
 * @see IProductWRTPhysTraceSerialAVXGenericKernels.hpp for the kernels
 * called from here.
 * @see IProductWRTPhysTraceDeviceGeneric.hpp for the same decomposition
 * packed for warp lanes instead of SIMD vectors.
 * @see PhysTraceExtractSerialAVXGeneric.hpp for the adjoint operator,
 * which applies the same interpolation tables untransposed.
 */

#pragma once

#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "Operators/ElmtOps/IProductWRTPhysTrace/IProductWRTPhysTraceBlockOp.hpp"
#include "Operators/ElmtOps/IProductWRTPhysTrace/IProductWRTPhysTraceSerialAVXGenericKernels.hpp"

// Selects the switch construction used by the generated ShapeBlock
// definitions (see LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in).
#define NEKTAR_BLOCKOP_SWITCH_PHYSTRACE

namespace Nektar::Operators::detail
{

/**
 * @brief Serial and AVX block implementation of the trace lifting term,
 * one element per SIMD lane.
 *
 * @details
 * For every element of the block this accumulates
 * \f[ \Lambda_{\boldsymbol p} \mathrel{+}= \sum_{F \in \partial E}
 *     \int_F \hat f \; h_{\boldsymbol p}\big|_F \; J_F \, \mathrm{d}s ,
 * \f]
 * the trace field tested against each cardinal (hat) function of the
 * volume quadrature grid. IProductWRTPhysTraceOp.hpp derives this and
 * says what the result is and is not. The class itself only marshals
 * data: the constructor caches the tables, OperatorND() sizes the
 * workspace and drives the interleave,
 * and every floating-point operation lives in the kernels.
 *
 * Naming, because it trips up every new reader. #m_nm holds
 * GetNumPoints, that is volume quadrature counts per direction and not
 * mode counts. #m_B holds two different families of eInterp tables:
 * its first #m_dimension entries are the normal-direction tables
 * \f$h_p(\pm 1)\f$, called nbasis in the kernels, and the remainder are
 * the tangential tables \f$h_p(\xi^{tr})\f$, called tbasis. Neither is
 * an expansion basis evaluated anywhere. Both are stored tight: a
 * normal-direction table has one column per trace of that direction,
 * and a tangential table one column per trace quadrature point. The
 * bulk kernels index a normal-direction table with that trace count as
 * its row stride. The per-trace lift kernels of this backend pass their
 * trace-loop bound instead, so a one-wide window on a two-trace
 * direction reads the table with a stride of one; the Device backend
 * threads the stride through as a separate argument and does not share
 * the defect (see IProductWRTPhysTraceSerialAVXGenericKernels.hpp).
 *
 * Layout. The tables and the weights are fetched with a
 * `BasisDataKey<simd_t>`, so each scalar coefficient occupies one whole
 * SIMD vector with the same value in every lane, and one table lookup
 * feeds the `simd_t::width` elements being processed together. Field,
 * trace and Jacobian data are the other way round: they are interleaved,
 * one element per lane, and reach the kernels through a
 * reinterpret_cast of the block's TData storage to simd_t. Since a block
 * may be stored at some other interleave width, every path below
 * reshapes the storage to #m_implInterleaveWidth a chunk at a time
 * around the kernel calls, and scales every per-element offset it hands
 * to a kernel by `simd_t::width`. The packed trace input, and the
 * deformed trace Jacobian, order traces by normal direction: the N0 pair
 * first, then the N1 pair or single, then N2, face-major within a pair.
 * A regular Jacobian carries one slot per trace in that same order,
 * which is the permutation the `jlocoff` tables in
 * v_IProductWRTPhysTrace apply.
 *
 * @tparam ExecSpace       NektarSpaces::Serial or NektarSpaces::AVX,
 *                         which is what selects the vector type.
 * @tparam Implementation  Operators::Generic; nothing else is
 *                         generated, and the parameter is not read.
 * @tparam TData           Floating-point type of the field data.
 *
 * @see IProductWRTPhysTraceSerialAVXGenericKernels.hpp for the kernel
 * layers this class calls.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class IProductWRTPhysTraceBlockOpImpl
    : public IProductWRTPhysTraceBlockOp<TData>
{
    using BlockOpBase = IProductWRTPhysTraceBlockOp<TData>;
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Cache the block's shape, its point counts, both families of
     * interpolation tables, the trace weights and the trace Jacobian, in
     * SIMD-vector form.
     *
     * @details
     * The first loop walks the normal directions. For direction @em dir
     * it records GetNumPoints in #m_nm and fetches the eInterp table
     * taking that direction's volume rule to the positions of its
     * traces: the two Gauss-Lobatto points \f$\xi = \pm 1\f$ where the
     * direction carries a trace pair, and the single point
     * \f$\xi = -1\f$ (a one-point eGaussLegendreWithM rule) where it
     * collapses to one trace. The table is stored tight, so it has as
     * many columns as the direction has traces, which is the stride the
     * bulk kernels index it with. The per-trace lift leaves use their
     * trace-loop bound instead. #m_endPtsCollocated then records whether
     * the direction's own quadrature rule contains domain endpoints, in
     * which case that interpolation is a Kronecker delta and the lift
     * along the normal degenerates to writing a single boundary plane.
     * Unlike the Device implementation this one also asserts, in debug
     * builds, that the rule's endpoint count equals the direction's
     * trace count.
     *
     * The second loop walks the same normal directions in packing order,
     * taking one representative trace each, trace
     * `m_dimension - 1 - dim`: edges 1 and 0 in two dimensions, faces 2,
     * 1 and 0 in three, chosen so that the loop index @em dim is the
     * normal direction for every shape. For each of that trace's
     * in-trace directions @em d it records the trace quadrature count in
     * #m_nq, whether those points coincide with the volume points of the
     * volume direction the in-trace direction runs along
     * (SpatialDomains::Geometry::GetDir) in #m_isCollocated, the eInterp
     * table from that volume direction onto the trace points in #m_B,
     * and the trace rule's weights in #m_W. All three are indexed
     * `2 * dim + d` in three dimensions and `dim` in two.
     *
     * In one dimension there are no in-trace directions, so the second
     * loop body never runs and no trace Jacobian is fetched: a segment
     * trace is a point and carries no surface measure. #m_nq is given
     * the volume point count instead, which is the count the generated
     * segment switch dispatches on, so that choice always lands on a
     * compiled instantiation. The vectors are not padded any further:
     * each generated translation unit reads only the entries its own
     * shape has.
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
        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_dimension = exp->GetShapeDimension();
        m_isDeformed =
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;

        // set up end points interplation expansion in normal direction
        for (unsigned int dir = 0; dir < m_dimension; ++dir)
        {
            auto bkey = this->m_exp->GetBasis(dir)->GetBasisKey();
            m_nm.push_back(exp->GetNumPoints(dir));

            // set up end point interpolation depending on number of faces
            unsigned int ntrace =
                LibUtilities::ShapeTypeNumTraceInDir[exp->DetShapeType()][dir];

            // used GLwithM nodes to get z=-1 as zero if ntrace = 1 otherwise
            // GLL
            LibUtilities::PointsType endPtsType =
                (ntrace == 1) ? LibUtilities::eGaussLegendreWithM
                              : LibUtilities::eGaussLobattoLegendre;

            // Fetch basis data.
            m_B.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(bkey, LibUtilities::eInterp,
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

        // set trace interpolation bases
        auto expPtsKeys = exp->GetPointsKeys();
        for (unsigned int dim = 0; dim < m_dimension; ++dim)
        {
            auto tr = m_dimension - 1 - dim;
            // trace directions
            for (unsigned int d = 0; d < m_dimension - 1; ++d)
            {
                auto dir      = this->m_exp->GetGeom()->GetDir(tr, d);
                auto trBKey   = exp->GetTraceBasisKey(tr, d);
                auto trPtsKey = trBKey.GetPointsKey();
                auto nq       = trPtsKey.GetNumPoints();

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
                    LibUtilities::BasisDataKey<simd_t>(
                        this->m_exp->GetBasis(dir)->GetBasisKey(),
                        LibUtilities::eInterp, nq, trPtsKey.GetPointsType())));

                // Fetch associated Weights
                m_W.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                    LibUtilities::BasisDataKey<simd_t>(
                        trBKey, LibUtilities::eWeights)));
            }
        }

        // Workspaces, sized for the largest intermediate any shape of
        // this dimension can need. A segment's traces are points and
        // need none.
        if (m_dimension == 2)
        {
            // The trace-mode buffer of an edge pair, which the edge
            // inner product writes and the lift reads back.
            m_wsp.push_back(std::vector<simd_t, tinysimd::allocator<simd_t>>(
                std::max(m_nm[0] * 2, m_nm[1] * 2)));
            // The per-trace route hands both buffers to its shape
            // switch before reaching a two- or three-dimensional arm,
            // so the second exists but stays empty in two dimensions,
            // as the local it replaced did. OperatorND()'s third index
            // sequence is one long here and never reaches it.
            m_wsp.push_back(std::vector<simd_t, tinysimd::allocator<simd_t>>());
        }
        else if (m_dimension == 3)
        {
            // The face mode blocks of a face pair, then the scratch of
            // the general face contraction.
            m_wsp.push_back(
                std::vector<simd_t, tinysimd::allocator<simd_t>>(std::max(
                    m_nm[0] * m_nm[1] * 2,
                    std::max(m_nm[1] * m_nm[2] * 2, m_nm[0] * m_nm[2] * 2))));
            m_wsp.push_back(std::vector<simd_t, tinysimd::allocator<simd_t>>(
                std::max(m_nm[1] * m_nq[1],
                         std::max(m_nm[0] * m_nq[3], m_nm[0] * m_nq[5]))));
        }

        if (m_dimension > 1)
        {
            // Fetch trace Jacobian.
            auto jackey = LocalRegions::JacobianLocTraceKey<TData>(
                block_idx, m_implInterleaveWidth);
            m_jacptr =
                this->m_dataWarehouse->template GetData<MemSpace>(jackey);
        }
        else
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
    /// providing a single implementation; see IProductWRTPhysTraceBlockOp.
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
    /// Elements the kernels process at once: `simd_t::width`, the SIMD
    /// vector width in the AVX space and one in Serial. The Operator
    /// methods reshape the block storage to this width around every
    /// kernel call and scale their per-element offsets by it.
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

    /// Shape shared by every element of the block.
    LibUtilities::ShapeType m_shapeType;
    /// Whether the block's geometry is deformed, so that the trace
    /// Jacobian holds one value per trace quadrature point rather than
    /// one per trace. Read by the generated dispatch to pick the
    /// DEFORMED instantiation, and at run time by the per-trace path.
    bool m_isDeformed;
    /// Dimension of the reference element (1, 2 or 3).
    unsigned int m_dimension;
    /// Volume quadrature points per direction, read from GetNumPoints:
    /// point counts, not mode counts. One entry per direction, unpadded.
    std::vector<unsigned int> m_nm;
    /// Trace quadrature points, indexed `2 * dir + tangential` by normal
    /// direction and tangential direction in three dimensions and by normal
    /// direction in two. In one dimension the single entry is a copy of
    /// the volume point count.
    std::vector<unsigned int> m_nq;
    /// eInterp tables, each coefficient broadcast across a whole SIMD
    /// vector: first the #m_dimension normal-direction tables
    /// \f$h_p(\pm 1)\f$, stored tight with one column per trace of that
    /// direction, then the tangential tables \f$h_p(\xi^{tr})\f$ in the
    /// same (direction, tangential) order as #m_nq. The kernels call the first
    /// group nbasis and the second tbasis.
    std::vector<const simd_t *> m_B; // trace basis
    /// Trace quadrature weights per (normal direction, tangential direction),
    /// in the same order as #m_nq and likewise broadcast. Empty in one
    /// dimension.
    std::vector<const simd_t *> m_W; // trace weights
    /// Trace Jacobian of the block, ordered by normal direction then
    /// trace: one value per trace quadrature point when deformed, one
    /// per trace otherwise. Held as TData, interleaved to
    /// #m_implInterleaveWidth, and reinterpret_cast to simd_t at every
    /// call site. Left unset in one dimension, where a trace is a point
    /// and carries no surface measure and no path reads it.
    const TData *m_jacptr; // trace Jacobian in each direction
    /// Kernel workspaces, allocated once by the constructor and indexed
    /// by the third index sequence OperatorND() is handed: none in one
    /// dimension, the trace-mode buffer in two, and in three the face
    /// mode block followed by the contraction scratch. The per-trace
    /// route reads the same buffers.
    std::vector<std::vector<simd_t, tinysimd::allocator<simd_t>>> m_wsp;
    /// One flag per normal direction: that direction's volume rule
    /// contains domain endpoints, so interpolating it to the trace
    /// position is a Kronecker delta and the lift along the normal
    /// writes a single boundary plane.
    std::vector<bool> m_endPtsCollocated;
#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
    // flag to ensure we only get one warning for alignment otherwise CI
    // system is saturated with warnings
    bool m_warnOnce = false;
#endif

    /**
     * @brief Per-trace path: lift trace @p traceid of every element of
     * the block, accumulating into @p outblock.
     *
     * @details
     * The kernels are instantiated with their APPEND parameter fixed to
     * true, so nothing here zeroes anything: a caller assembling the
     * whole boundary trace by trace must zero @p outblock itself before
     * the first call. Both blocks are therefore reshaped to
     * #m_implInterleaveWidth around the kernel calls, the output as well
     * as the input, since the output is read as well as written.
     *
     * Shape and geometry are branched on at run time rather than
     * through the generated switch: the outer branch selects the
     * DEFORMED instantiation from #m_isDeformed, the inner one calls
     * IProductWRTPhysTraceEdgeKernel for the two-dimensional shapes and
     * IProductWRTPhysTraceFaceKernel for the three-dimensional ones,
     * both of which resolve @p traceid to a normal direction and a
     * position themselves. The nodal shapes are mapped onto their parent
     * shape's instantiation here, NodalTri to Tri, NodalTet to Tet and
     * NodalPrism to Prism, which matches their trace structure; the bulk
     * path does not do this.
     *
     * Beyond that DEFORMED instantiation the two branches differ only in
     * where the Jacobian pointer is aimed. A deformed Jacobian is packed
     * exactly like the trace field and so takes the same @p inOffset; a
     * regular one has one slot per trace in packed order, which is the
     * local `jlocoff` permutation. Both offsets are scaled by
     * `simd_t::width`, and @c jacoffset then walks the element groups:
     * by the element's packed trace entries when deformed, by the
     * shape's trace count when regular.
     *
     * The workspaces are stack-local vectors with a tinysimd allocator,
     * sized for one element group. In two dimensions that is
     * `2 * max(nm0, nm1)` vectors, the trace-mode buffer the edge inner
     * product writes and the lift then reads, and the second workspace
     * is unused. In three dimensions the first is
     * `2 * max(nm0 * nm1, nm1 * nm2, nm0 * nm2)`, a face pair's mode
     * block, of which a single-trace call needs half; the second is
     * `max(nm1 * m_nq[1], nm0 * m_nq[3], nm0 * m_nq[5])`, the scratch of
     * the general (non-collocated) face contraction, holding the modes
     * of the first in-trace direction of a face against the trace points
     * of its second.
     *
     * @param   traceid     Local trace index in the Nektar edge or face
     *                      numbering of the shape.
     * @param   inblock     Packed trace field of the block.
     * @param   inOffset    Offset of this trace's values within one
     *                      element's packed trace entries, counted in
     *                      entries per element. It also selects the
     *                      deformed Jacobian, whose packing matches the
     *                      trace field's.
     * @param   outblock    Volume-shaped output block, accumulated into.
     *
     * @note Segments are not served: both branches raise a fatal
     * NEKERROR for LibUtilities::Seg.
     *
     * @see IProductWRTPhysTraceOp::IProductWRTPhysTrace() for the
     * calling protocol.
     */
    // This is a debugging method to test individual trace operators
    // (appending)
    void v_IProductWRTPhysTrace(
        const unsigned int traceid,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        const unsigned int inOffset,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock) override
    {
        // Shape size.
        const auto numDataIn  = inblock.GetNumData();
        const auto numDataOut = outblock.GetNumData();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = (this->m_append)
                          ? outblock.template GetPtr<MemSpace, ReadWrite>()
                          : outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto inInterleaveWidth  = inblock.GetInterleaveWidth();
        const auto outInterleaveWidth = outblock.GetInterleaveWidth();
        const auto width_ratio =
            (inInterleaveWidth == 1)
                ? 1
                : inInterleaveWidth / m_implInterleaveWidth;
        const auto chunkSize =
            std::max(m_implInterleaveWidth, inInterleaveWidth);

        // Loop over components.
        for (unsigned int n = 0;
             n < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++n)
        {
            unsigned int jacoffset = 0;
            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        m_implInterleaveWidth, inInterleaveWidth, chunkSize,
                        numDataIn, (TData *)inptr);
                }

                if (this->m_append)
                {
                    // Reshape, if necessary.
                    if (e % width_ratio == 0)
                    {
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            m_implInterleaveWidth, outInterleaveWidth,
                            chunkSize, numDataOut, (TData *)outptr);
                    }

                    if (m_isDeformed)
                    {
                        LaunchIProductWRTPhysTraceND<true, true>(
                            traceid, inOffset, jacoffset, m_wsp[0], m_wsp[1],
                            inptr, outptr);
                    }
                    else
                    {
                        LaunchIProductWRTPhysTraceND<false, true>(
                            traceid, inOffset, jacoffset, m_wsp[0], m_wsp[1],
                            inptr, outptr);
                    }
                }
                else
                {
                    if (m_isDeformed)
                    {
                        LaunchIProductWRTPhysTraceND<true, false>(
                            traceid, inOffset, jacoffset, m_wsp[0], m_wsp[1],
                            inptr, outptr);
                    }
                    else
                    {
                        LaunchIProductWRTPhysTraceND<false, false>(
                            traceid, inOffset, jacoffset, m_wsp[0], m_wsp[1],
                            inptr, outptr);
                    }
                }

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                        numDataIn,
                        (TData *)inptr -
                            (width_ratio - 1) * numDataIn * simd_t::width);
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                        numDataOut,
                        (TData *)outptr -
                            (width_ratio - 1) * numDataOut * simd_t::width);
                }

                // Increment pointers for the next elmt group.
                inptr += numDataIn * simd_t::width;
                outptr += numDataOut * simd_t::width;
                if (m_isDeformed)
                {
                    jacoffset += numDataIn * simd_t::width;
                }
                else
                {
                    jacoffset += LibUtilities::ShapeTypeNumTraces[m_shapeType] *
                                 simd_t::width;
                }
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
    }

    /// Dispatch one trace of the block to its shape-specific kernel.
    /// DEFORMED selects the trace Jacobian layout and APPEND the
    /// accumulating instantiation, so the caller pairs the latter with
    /// the output reshape, as IProductWRTDerivBase does.
    template <bool DEFORMED, bool APPEND>
    void LaunchIProductWRTPhysTraceND(
        const unsigned int traceid, const unsigned int inOffset,
        const unsigned int jacoffset,
        std::vector<simd_t, tinysimd::allocator<simd_t>> &wsp0,
        std::vector<simd_t, tinysimd::allocator<simd_t>> &wsp1,
        const TData *inptr, TData *outptr)
    {
        switch (m_shapeType)
        {
            // Quads
            case LibUtilities::Quad:
            {
                // A deformed Jacobian is packed like the trace
                // field and takes the same offset; a regular one
                // has one slot per trace, which this table maps
                // the Nektar trace id onto.
                constexpr unsigned int jlocoff[4] = {2, 1, 3, 0};
                const unsigned int joff =
                    (DEFORMED) ? inOffset : jlocoff[traceid];

                IProductWRTPhysTraceEdgeKernel<LibUtilities::Quad, DEFORMED,
                                               APPEND>(
                    traceid, m_nm[0], m_nm[1], m_B[0], m_B[1], m_nq[0], m_nq[1],
                    m_B[2], m_B[3], m_W[0], m_W[1],
                    reinterpret_cast<const simd_t *>(
                        m_jacptr + joff * simd_t::width + jacoffset),
                    wsp0.data(),
                    reinterpret_cast<const simd_t *>(inptr +
                                                     inOffset * simd_t::width),
                    reinterpret_cast<simd_t *>(outptr), this->m_isCollocated[0],
                    this->m_isCollocated[1], m_endPtsCollocated[0],
                    m_endPtsCollocated[1]);
                break;
            }
            // Nodal Triangles
            case LibUtilities::Tri:
            case LibUtilities::NodalTri:
            {
                // A deformed Jacobian is packed like the trace
                // field and takes the same offset; a regular one
                // has one slot per trace, which this table maps
                // the Nektar trace id onto.
                constexpr unsigned int jlocoff[3] = {2, 1, 0};
                const unsigned int joff =
                    (DEFORMED) ? inOffset : jlocoff[traceid];

                IProductWRTPhysTraceEdgeKernel<LibUtilities::Tri, DEFORMED,
                                               APPEND>(
                    traceid, m_nm[0], m_nm[1], m_B[0], m_B[1], m_nq[0], m_nq[1],
                    m_B[2], m_B[3], m_W[0], m_W[1],
                    reinterpret_cast<const simd_t *>(
                        m_jacptr + joff * simd_t::width + jacoffset),
                    wsp0.data(),
                    reinterpret_cast<const simd_t *>(inptr +
                                                     inOffset * simd_t::width),
                    reinterpret_cast<simd_t *>(outptr), this->m_isCollocated[0],
                    this->m_isCollocated[1], m_endPtsCollocated[0],
                    m_endPtsCollocated[1]);
                break;
            }
            // Hexes
            case LibUtilities::Hex:
            {
                // A deformed Jacobian is packed like the trace
                // field and takes the same offset; a regular one
                // has one slot per trace, which this table maps
                // the Nektar trace id onto.
                constexpr unsigned int jlocoff[6] = {4, 2, 1, 3, 0, 5};
                const unsigned int joff =
                    (DEFORMED) ? inOffset : jlocoff[traceid];

                IProductWRTPhysTraceFaceKernel<LibUtilities::Hex, DEFORMED,
                                               APPEND>(
                    traceid, m_nm[0], m_nm[1], m_nm[2], m_B[0], m_B[1], m_B[2],
                    m_nq[0], m_nq[1], m_nq[2], m_nq[3], m_nq[4], m_nq[5],
                    m_B[3], m_B[4], m_B[5], m_B[6], m_B[7], m_B[8], m_W[0],
                    m_W[1], m_W[2], m_W[3], m_W[4], m_W[5],
                    reinterpret_cast<const simd_t *>(
                        m_jacptr + joff * simd_t::width + jacoffset),
                    wsp0.data(), wsp1.data(),
                    reinterpret_cast<const simd_t *>(inptr +
                                                     inOffset * simd_t::width),
                    reinterpret_cast<simd_t *>(outptr), this->m_isCollocated[0],
                    this->m_isCollocated[1], this->m_isCollocated[2],
                    this->m_isCollocated[3], this->m_isCollocated[4],
                    this->m_isCollocated[5], m_endPtsCollocated[0],
                    m_endPtsCollocated[1], m_endPtsCollocated[2]);
                break;
            }
            // Tet
            case LibUtilities::NodalTet:
            case LibUtilities::Tet:
            {
                // A deformed Jacobian is packed like the trace
                // field and takes the same offset; a regular one
                // has one slot per trace, which this table maps
                // the Nektar trace id onto.
                constexpr unsigned int jlocoff[5] = {3, 2, 1, 0};
                const unsigned int joff =
                    (DEFORMED) ? inOffset : jlocoff[traceid];

                IProductWRTPhysTraceFaceKernel<LibUtilities::Tet, DEFORMED,
                                               APPEND>(
                    traceid, m_nm[0], m_nm[1], m_nm[2], m_B[0], m_B[1], m_B[2],
                    m_nq[0], m_nq[1], m_nq[2], m_nq[3], m_nq[4], m_nq[5],
                    m_B[3], m_B[4], m_B[5], m_B[6], m_B[7], m_B[8], m_W[0],
                    m_W[1], m_W[2], m_W[3], m_W[4], m_W[5],
                    reinterpret_cast<const simd_t *>(
                        m_jacptr + joff * simd_t::width + jacoffset),
                    wsp0.data(), wsp1.data(),
                    reinterpret_cast<const simd_t *>(inptr +
                                                     inOffset * simd_t::width),
                    reinterpret_cast<simd_t *>(outptr), this->m_isCollocated[0],
                    this->m_isCollocated[1], this->m_isCollocated[2],
                    this->m_isCollocated[3], this->m_isCollocated[4],
                    this->m_isCollocated[5], m_endPtsCollocated[0],
                    m_endPtsCollocated[1], m_endPtsCollocated[2]);
                break;
            }
            // Pyr
            case LibUtilities::Pyr:
            {
                // A deformed Jacobian is packed like the trace
                // field and takes the same offset; a regular one
                // has one slot per trace, which this table maps
                // the Nektar trace id onto.
                constexpr unsigned int jlocoff[5] = {4, 2, 1, 3, 0};
                const unsigned int joff =
                    (DEFORMED) ? inOffset : jlocoff[traceid];

                IProductWRTPhysTraceFaceKernel<LibUtilities::Pyr, DEFORMED,
                                               APPEND>(
                    traceid, m_nm[0], m_nm[1], m_nm[2], m_B[0], m_B[1], m_B[2],
                    m_nq[0], m_nq[1], m_nq[2], m_nq[3], m_nq[4], m_nq[5],
                    m_B[3], m_B[4], m_B[5], m_B[6], m_B[7], m_B[8], m_W[0],
                    m_W[1], m_W[2], m_W[3], m_W[4], m_W[5],
                    reinterpret_cast<const simd_t *>(
                        m_jacptr + joff * simd_t::width + jacoffset),
                    wsp0.data(), wsp1.data(),
                    reinterpret_cast<const simd_t *>(inptr +
                                                     inOffset * simd_t::width),
                    reinterpret_cast<simd_t *>(outptr), this->m_isCollocated[0],
                    this->m_isCollocated[1], this->m_isCollocated[2],
                    this->m_isCollocated[3], this->m_isCollocated[4],
                    this->m_isCollocated[5], m_endPtsCollocated[0],
                    m_endPtsCollocated[1], m_endPtsCollocated[2]);
                break;
            }
            // Prism
            case LibUtilities::NodalPrism:
            case LibUtilities::Prism:
            {
                // A deformed Jacobian is packed like the trace
                // field and takes the same offset; a regular one
                // has one slot per trace, which this table maps
                // the Nektar trace id onto.
                constexpr unsigned int jlocoff[5] = {4, 2, 1, 3, 0};
                const unsigned int joff =
                    (DEFORMED) ? inOffset : jlocoff[traceid];

                IProductWRTPhysTraceFaceKernel<LibUtilities::Prism, DEFORMED,
                                               APPEND>(
                    traceid, m_nm[0], m_nm[1], m_nm[2], m_B[0], m_B[1], m_B[2],
                    m_nq[0], m_nq[1], m_nq[2], m_nq[3], m_nq[4], m_nq[5],
                    m_B[3], m_B[4], m_B[5], m_B[6], m_B[7], m_B[8], m_W[0],
                    m_W[1], m_W[2], m_W[3], m_W[4], m_W[5],
                    reinterpret_cast<const simd_t *>(
                        m_jacptr + joff * simd_t::width + jacoffset),
                    wsp0.data(), wsp1.data(),
                    reinterpret_cast<const simd_t *>(inptr +
                                                     inOffset * simd_t::width),
                    reinterpret_cast<simd_t *>(outptr), this->m_isCollocated[0],
                    this->m_isCollocated[1], this->m_isCollocated[2],
                    this->m_isCollocated[3], this->m_isCollocated[4],
                    this->m_isCollocated[5], m_endPtsCollocated[0],
                    m_endPtsCollocated[1], m_endPtsCollocated[2]);
                break;
            }
            default:
            {
                NEKERROR(ErrorUtil::efatal, "shape type not implemented");
            }
        }
    }

    /**
     * @brief Bulk path: lift every trace of every element of the block.
     *
     * @details
     * Checks first that both blocks are aligned for simd_t and latches
     * #m_warnOnce afterwards, so that a misaligned field costs at most
     * one warning per block operator. Both the check and the flag exist
     * only under NEKTAR_DEBUG or NEKTAR_FULLDEBUG; elsewhere WARNINGL1
     * expands to nothing. Then dispatches to the per-shape entry point
     * the generated translation unit defines, which selects the
     * size-templated OperatorND() instantiation.
     *
     * @note The default arm raises a fatal error. No shape CMake
     * generates falls through to it (Shapes in
     * library/Operators/CMakeLists.txt), so it is only reachable for a
     * shape outside that list.
     */
    void v_Apply(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock) override
    {
        WARNINGL1(m_warnOnce ||
                      (inblock.GetAlignment() % simd_t::alignment == 0 &&
                       outblock.GetAlignment() % simd_t::alignment == 0),
                  "Input or output Field are not aligned to the required "
                  "alignment "
                  "for the SIMD vector type.");
#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
        m_warnOnce = true;
#endif

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

    // Workspaces #m_wsp holds in dim dimensions.
    static constexpr unsigned int NumWorkspace(const unsigned int dim)
    {
        return (dim == 1) ? 0 : (dim == 2) ? 1 : 2;
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
            std::make_integer_sequence<unsigned int, NumTangentialDir(DIM)>(),
            std::make_integer_sequence<unsigned int, NumWorkspace(DIM)>());
    }

    /// @brief Runtime-sized entry: forwards the stored counts #m_nm and
    /// #m_nq to the core overload.
    /// @name Dimensional entry points
    /// Each builds the block's size parameter, non-templated where the
    /// generated switch found no compiled combination and templated
    /// where it did, and hands OperatorND() the index sequences that
    /// select this dimension's slice of #m_B, #m_W, #m_isCollocated,
    /// #m_endPtsCollocated and #m_wsp. The first sequence runs over the
    /// normal directions, the second over the tangential directions and the
    /// third over the workspaces: one, none and none in one dimension,
    /// two, two and one in two, three, six and two in three.
    /// @{

    /// @}

    /**
     * @brief Bulk path for every dimension: lift every trace of every
     * element of the block onto the volume field.
     *
     * @details
     * The whole trace loop lives in IProductWRTPhysTraceKernelLauncher,
     * whose one-, two- and three-dimensional arms this one body calls;
     * overload resolution picks the arm from the number of arguments
     * the index sequences expand to. Each walks the packed traces of an
     * element in the order the extraction wrote them.
     *
     * IProductWRTPhysTraceBlockOp::m_append selects the accumulating
     * instantiation, and it is paired with the output reshape: only an
     * appending call reads what is already in the output, so only it
     * needs the output brought to #m_implInterleaveWidth on the way in.
     *
     * @c jacoffset walks the trace Jacobian alongside the field: by the
     * element group's worth of trace points when the geometry is
     * deformed, by one slot per trace when it is regular. A segment
     * carries no Jacobian at all, so the pointer stays null rather than
     * being offset into.
     *
     * The component loop carries the GetNumHomoModes factor. That is
     * inert for three-dimensional elements, which support no
     * homogeneous extension and report one plane.
     *
     * @tparam SHAPE_TYPE   Shape of the block, the nodal enumerators
     *                      included.
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
     * @tparam ind2         Workspaces, selecting #m_wsp. Empty in one
     *                      dimension, one in two, two in three.
     */
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TTraceSizeParameter, unsigned int... ind0,
              unsigned int... ind1, unsigned int... ind2>
    NEK_FORCE_INLINE void OperatorNDImpl(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        TTraceSizeParameter sizeParam,
        std::integer_sequence<unsigned int, ind0...>,
        std::integer_sequence<unsigned int, ind1...>,
        std::integer_sequence<unsigned int, ind2...>)
    {
        // Shape size.
        const auto numDataIn  = sizeParam.template nqTotTrace<SHAPE_TYPE>();
        const auto numDataOut = sizeParam.nmTot();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = (this->m_append)
                          ? outblock.template GetPtr<MemSpace, ReadWrite>()
                          : outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto inInterleaveWidth  = inblock.GetInterleaveWidth();
        const auto outInterleaveWidth = outblock.GetInterleaveWidth();
        const auto width_ratio =
            (inInterleaveWidth == 1)
                ? 1
                : inInterleaveWidth / m_implInterleaveWidth;
        const auto chunkSize =
            std::max(m_implInterleaveWidth, inInterleaveWidth);

        // Loop over components.
        for (unsigned int n = 0;
             n < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++n)
        {
            unsigned int jacoffset = 0;

            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        m_implInterleaveWidth, inInterleaveWidth, chunkSize,
                        numDataIn, (TData *)inptr);
                }

                // A segment carries no trace Jacobian, so the pointer
                // stays null rather than being offset into.
                const TData *jacptr =
                    (m_jacptr == nullptr) ? nullptr : m_jacptr + jacoffset;

                if (this->m_append)
                {
                    // Reshape, if necessary.
                    if (e % width_ratio == 0)
                    {
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            m_implInterleaveWidth, outInterleaveWidth,
                            chunkSize, numDataOut, (TData *)outptr);
                    }

                    // IProductWRTPhysTrace kernel.
                    IProductWRTPhysTraceKernelLauncher<SHAPE_TYPE, DEFORMED,
                                                       true>(
                        sizeParam, m_B[ind0]..., m_B[sizeof...(ind0) + ind1]...,
                        m_W[ind1]..., reinterpret_cast<const simd_t *>(jacptr),
                        m_wsp[ind2].data()...,
                        reinterpret_cast<const simd_t *>(inptr),
                        reinterpret_cast<simd_t *>(outptr),
                        (bool)this->m_isCollocated[ind1]...,
                        (bool)m_endPtsCollocated[ind0]...);
                }
                else
                {
                    // IProductWRTPhysTrace kernel.
                    IProductWRTPhysTraceKernelLauncher<SHAPE_TYPE, DEFORMED,
                                                       false>(
                        sizeParam, m_B[ind0]..., m_B[sizeof...(ind0) + ind1]...,
                        m_W[ind1]..., reinterpret_cast<const simd_t *>(jacptr),
                        m_wsp[ind2].data()...,
                        reinterpret_cast<const simd_t *>(inptr),
                        reinterpret_cast<simd_t *>(outptr),
                        (bool)this->m_isCollocated[ind1]...,
                        (bool)m_endPtsCollocated[ind0]...);
                }

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                        numDataIn,
                        (TData *)inptr -
                            (width_ratio - 1) * numDataIn * simd_t::width);
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                        numDataOut,
                        (TData *)outptr -
                            (width_ratio - 1) * numDataOut * simd_t::width);
                }

                // Increment pointers for the next elmt group.
                inptr += numDataIn * simd_t::width;
                outptr += numDataOut * simd_t::width;

                if constexpr (DEFORMED)
                {
                    jacoffset += numDataIn * simd_t::width;
                }
                else
                {
                    jacoffset += LibUtilities::ShapeTypeNumTraces[SHAPE_TYPE] *
                                 simd_t::width;
                }
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
