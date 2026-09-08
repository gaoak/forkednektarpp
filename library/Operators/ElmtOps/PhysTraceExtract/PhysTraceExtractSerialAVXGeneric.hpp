///////////////////////////////////////////////////////////////////////////////
//
// File: PhysTraceExtractSerialAVXGeneric.hpp
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
 * @file PhysTraceExtractSerialAVXGeneric.hpp
 * @brief Serial and AVX dispatch of the sum-factorised trace
 * extraction, the interpolation of an element's volume field onto the
 * quadrature points of its traces.
 *
 * @details
 * This header holds the host-side half of the operator for the Serial
 * and the AVX execution space. The constructor reads the block's point
 * counts and both families of interpolation tables out of the data
 * warehouse; each application then sizes the workspace, normalises the
 * storage interleave to the SIMD width and dispatches per shape into
 * PhysTraceExtractSerialAVXGenericKernels.hpp. No arithmetic on the
 * field happens here.
 *
 * CMake selects this header for the Serial and AVX execution spaces
 * with Operators::Generic and generates one translation unit per shape
 * and data type from PhysTraceExtractGenericBlockOp.cpp.in. Those units
 * define the per-shape entry points declared below, expanding
 * Common/BlockOpSwitchPhysTraceExtract.h.in, the switch template this
 * operator shares with IProductWRTPhysTrace.
 *
 * Two entry points reach the kernels:
 * - v_Apply, the bulk path, which extracts all of an element's traces
 *   in one call per component through OperatorND
 *   , filling the packed trace output in packing order;
 * - v_ExtractTrace, the per-trace path, which extracts one named trace
 *   into a caller-chosen slot of that same output.
 *
 * @note The Implementation tag here is only ever Operators::Generic,
 * this operator providing a single implementation, and nothing in the
 * class reads it: it is only forwarded to name the instantiation. A
 * StdMat, SumFac or SumFacTOP request reaches this class through
 * ElmtBlockOp::Create()'s fallback to the `"Generic"` factory key.
 *
 * @note The class below is the primary
 * detail::PhysTraceExtractBlockOpImpl template, with an unconstrained
 * ExecSpace parameter, and PhysTraceExtractDeviceGeneric.hpp declares a
 * second primary template of the same signature. The two headers are
 * therefore alternatives: a generated translation unit includes
 * exactly one of them, never both.
 *
 * @see PhysTraceExtractOp.hpp for the whole-field interface and for
 * the layouts of the two fields.
 * @see PhysTraceExtractSerialAVXGenericKernels.hpp for the kernels
 * called from here.
 * @see PhysTraceExtractDeviceGeneric.hpp for the same decomposition
 * packed for warp lanes instead of SIMD vectors.
 * @see IProductWRTPhysTraceSerialAVXGeneric.hpp for the adjoint
 * operator, which applies the same interpolation tables transposed and
 * carries the trace weights and trace Jacobian this one has no use
 * for.
 */

#pragma once

#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "Operators/ElmtOps/PhysTraceExtract/PhysTraceExtractBlockOp.hpp"

#include "Operators/ElmtOps/PhysTraceExtract/PhysTraceExtractSerialAVXGenericKernels.hpp"

namespace Nektar::Operators::detail
{

/**
 * @brief Serial and AVX block implementation of the trace extraction,
 * one element per SIMD lane.
 *
 * @details
 * For every element of the block this evaluates the volume field at
 * the quadrature points of the element's traces. A field given at the
 * volume points is \f$u = \sum_{\boldsymbol p} u_{\boldsymbol p}
 * h_{\boldsymbol p}\f$ in the cardinal (hat) functions of the volume
 * grid, so in two dimensions its values on the direction-1 trace pair
 * at \f$\xi_1 = \pm 1\f$, sampled at the trace points
 * \f$\xi^{tr}_{0i}\f$, are
 * \f[ u(\xi^{tr}_{0i}, \pm 1) = \sum_p h_p(\xi^{tr}_{0i})
 *     \sum_q u_{pq} \, h_q(\pm 1) . \f]
 * The two sums are the two stages of every two- and three-dimensional
 * kernel here, and they run in the order they are nested: the inner
 * normal-direction contraction first, the tangential interpolation
 * second. Three dimensions add a second tangential direction, nothing
 * else; a segment has only the first stage, its traces being points.
 *
 * This is the adjoint of IProductWRTPhysTrace and uses the same two
 * families of interpolation tables, here applied untransposed. It is a
 * pure interpolation: no quadrature weights and no trace Jacobian
 * enter it, which is why this class fetches neither. Nothing
 * accumulates either, on either entry point, and the operator carries
 * no append flag.
 *
 * Naming, because it trips up every new reader. #m_nm holds
 * GetNumPoints, that is volume quadrature counts per direction and not
 * mode counts. #m_B holds two different families of eInterp tables:
 * its first m_dimension entries are the normal-direction tables
 * \f$h_p(\pm 1)\f$, called ntbasis in the kernels, and the remainder
 * are the tangential tables \f$h_p(\xi^{tr})\f$, called tbasis.
 * Neither is an expansion basis evaluated anywhere.
 *
 * Layout. The kernels address the data through simd_t pointers, so a
 * group of #m_implInterleaveWidth elements holds entry @em n of
 * element @em ilane at `buf[width * n + ilane]` and one simd_t load
 * fetches that entry for the whole group. The input is a volume field,
 * one entry per volume quadrature point. The output is a packed trace
 * field ordered by normal direction: the N0 pair first, then the N1
 * pair or single, then N2, face-major within a pair.
 *
 * @tparam ExecSpace       NektarSpaces::Serial or NektarSpaces::AVX;
 *                         it selects simd_t and with it
 *                         #m_implInterleaveWidth.
 * @tparam Implementation  Operators::Generic; see the file-level
 *                         note.
 * @tparam TData           Floating-point type of the field data.
 *
 * @see PhysTraceExtractSerialAVXGenericKernels.hpp for the kernel
 * layers this class calls into.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class PhysTraceExtractBlockOpImpl : public PhysTraceExtractBlockOp<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Cache the block's shape, its point counts and both
     * families of interpolation tables.
     *
     * @details
     * A first short loop records GetNumPoints per direction in #m_nm.
     *
     * The second loop walks the normal directions. For direction
     * @em dir it fetches the eInterp table taking that direction's
     * volume rule to the positions of its traces: the two
     * Gauss-Lobatto points \f$\xi = \pm 1\f$ where the direction
     * carries a trace pair, and the single point \f$\xi = -1\f$ (a
     * one-point eGaussLegendreWithM rule) where it collapses to one
     * trace. The table is stored tight, with as many columns as the
     * direction has traces. #m_endPtsCollocated then
     * records whether the direction's own quadrature rule contains
     * domain endpoints, in which case that interpolation is a
     * Kronecker delta and the normal-direction stage degenerates to
     * selecting a boundary plane; a debug build additionally asserts
     * that the rule's endpoint count matches the direction's trace
     * count.
     *
     * The third loop walks the same normal directions in packing
     * order, taking one representative trace each, trace
     * `m_dimension - 1 - f`: edges 1 and 0 in two dimensions, faces 2,
     * 1 and 0 in three, chosen so that the loop index @em f is the
     * normal direction for every shape. For each of that trace's
     * in-trace directions @em d it records the trace quadrature count
     * in #m_nq, whether those points coincide with the volume points
     * of the volume direction the in-trace direction runs along
     * (SpatialDomains::Geometry::GetDir) in #m_isCollocated, and the
     * eInterp table from that volume direction onto the trace points
     * in #m_B. #m_nq and #m_isCollocated are indexed `2 * f + d` in
     * three dimensions and `f` in two, and the tangential tables sit
     * in that same order in #m_B after the m_dimension
     * normal-direction ones.
     *
     * In one dimension there are no in-trace directions, so the third
     * loop's body never runs. #m_nq is given the volume point count
     * instead; the trace count is not needed, because OperatorND
     * passes the literal 2 to its kernel and leaves its own nq0
     * parameter unused, so the value only decides which size-templated
     * instantiation the generated switch selects. #m_isCollocated is
     * given a single false entry.
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
        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_dimension = exp->GetShapeDimension();

        for (unsigned int d = 0; d < m_dimension; d++)
        {
            m_nm.push_back(exp->GetNumPoints(d));
        }

        // set up end points interplation expansion in normal direction
        for (unsigned int dir = 0; dir < m_dimension; ++dir)
        {
            auto bkey = this->m_exp->GetBasis(dir)->GetBasisKey();

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

        // set trace interpolation bases
        auto expPtsKeys = exp->GetPointsKeys();
        for (unsigned int f = 0; f < m_dimension; ++f)
        {
            // load basis in normal orientation
            unsigned int tr = m_dimension - 1 - f;

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
                    LibUtilities::BasisDataKey<TData>(
                        this->m_exp->GetBasis(dir)->GetBasisKey(),
                        LibUtilities::eInterp, nq, trPtsKey.GetPointsType())));
            }
        }

        if (m_dimension == 1)
        {
            // nq not required in 1D (formally it is 2) use nm value to
            // ensure templated version is adopted when possible
            m_nq.push_back(m_nm[0]);
            this->m_isCollocated.push_back(
                false); // cannot have collocated basis in this case since trace
                        // cannot be same as inteior
        }

        // Workspaces, sized for the largest intermediate any shape of
        // this dimension can need. A segment has no tangential stage
        // and so needs none.
        if (m_dimension == 2)
        {
            // The edge block the normal-direction stage produces,
            // before the tangential interpolation reads it.
            m_wsp.push_back(std::vector<simd_t, tinysimd::allocator<simd_t>>(
                std::max(m_nq[0] * m_nm[1], m_nq[1] * m_nm[0])));
        }
        else if (m_dimension == 3)
        {
            // The face blocks of a face pair, then the scratch of the
            // general face interpolation: the first in-trace direction
            // already interpolated while the second is still on the
            // volume points.
            m_wsp.push_back(
                std::vector<simd_t, tinysimd::allocator<simd_t>>(std::max(
                    m_nm[0] * m_nm[1] * 2,
                    std::max(m_nm[1] * m_nm[2] * 2, m_nm[0] * m_nm[2] * 2))));
            m_wsp.push_back(
                std::vector<simd_t, tinysimd::allocator<simd_t>>(std::max(
                    m_nq[4] * m_nm[1] * 2,
                    std::max(m_nq[2] * m_nm[2] * 2, m_nq[0] * m_nm[2] * 2))));
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
    /// SIMD width the kernels work at: one for the Serial space, where
    /// simd_t is tinysimd::scalarT, and the vector width of TData for
    /// AVX. Every entry point reshapes the input to this width around
    /// its kernel calls and reshapes both blocks back afterwards. Only
    /// v_ExtractTrace reshapes the output forwards as well.
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

    /// Shape shared by every element of the block.
    LibUtilities::ShapeType m_shapeType;
    /// Whether the block's geometry is deformed. Read by the generated
    /// dispatch to choose the DEFORMED instantiation, but never
    /// assigned by the constructor, so it stays false and only the
    /// DEFORMED == false half of the switch is ever taken. Nothing in
    /// this class consults DEFORMED, an extraction carrying no
    /// geometric factors, so the two halves are identical anyway.
    bool m_isDeformed = false;
    /// Dimension of the reference element (1, 2 or 3).
    unsigned int m_dimension;
    /// Volume quadrature points per direction, read from GetNumPoints:
    /// point counts, not mode counts. One entry per dimension, not
    /// padded out.
    std::vector<unsigned int> m_nm;
    /// Trace quadrature points, indexed `2 * dir + tangential` by normal
    /// direction and tangential direction in three dimensions and by normal
    /// direction in two. In one dimension the single entry is a copy
    /// of the volume point count, used only to steer the generated
    /// switch.
    std::vector<unsigned int> m_nq;
    /// eInterp tables: first the m_dimension normal-direction tables
    /// \f$h_p(\pm 1)\f$, stored tight with one column per trace of
    /// that direction, then the tangential tables
    /// \f$h_p(\xi^{tr})\f$ in the same (direction, tangential) order as
    /// #m_nq. The kernels call the first group ntbasis and the second
    /// tbasis.
    std::vector<const TData *> m_B;
    /// One flag per normal direction: that direction's volume rule
    /// contains domain endpoints, so interpolating it to the trace
    /// position is a Kronecker delta and the normal-direction stage
    /// degenerates to selecting a boundary plane. It selects the
    /// kernels' END_PTS_COLLOCATED template argument, the fast path that
    /// the general arm's three-dimensional defects do not affect.
    std::vector<bool> m_endPtsCollocated;
    /// Kernel workspaces, allocated once by the constructor and indexed
    /// by the third index sequence OperatorND() is handed: none in one
    /// dimension, one edge block in two, and in three the face blocks
    /// followed by the scratch of the general face interpolation. The
    /// per-trace route reads the same buffers.
    std::vector<std::vector<simd_t, tinysimd::allocator<simd_t>>> m_wsp;
#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
    // flag to ensure we only get one warning for alignment otherwise CI system
    // is saturated with warnings
    bool m_warnOnce = false;
#endif

    /**
     * @brief Per-trace path: extract trace @p traceid of every element
     * of the block into slot @p outOffset of the packed trace output.
     *
     * @details
     * Where v_Apply fills the whole packed output in packing order,
     * this writes one trace and lets the caller say where it lands, so
     * a caller assembling the whole boundary issues one call per trace
     * and advances @p outOffset in packed order rather than in
     * trace-id order.
     *
     * Both blocks are reshaped to #m_implInterleaveWidth at the start
     * of each chunk of `width_ratio` element groups and both are
     * reshaped back at its end. That the output is reshaped forwards
     * too, which the Operator methods do not do, is what lets a call
     * write its own slice while leaving the neighbouring traces in a
     * layout the next call and the field can still read.
     *
     * The workspaces are #m_wsp, which the constructor sizes once for
     * the largest intermediate any shape of this dimension can need
     * and which this route shares with the bulk one. In two dimensions
     * the single buffer holds the plane the normal-direction stage
     * produces before the tangential interpolation reads it. In three
     * dimensions `m_wsp[0]` holds those planes for a face pair,
     * `2 * max(nm0 * nm1, nm1 * nm2, nm0 * nm2)` values, and
     * `m_wsp[1]` the scratch of the general face interpolation, the
     * first in-trace direction already interpolated while the second
     * is still on the volume points,
     * `2 * max(nq20 * nm1, nq10 * nm2, nq00 * nm2)`.
     *
     * Nodal shapes are mapped onto their parent instantiation here,
     * NodalTri onto Tri, NodalTet onto Tet and NodalPrism onto Prism,
     * which the bulk path does not do; see the note on OperatorND.
     *
     * @param   traceid     Local trace index in the Nektar edge or
     *                      face numbering of the shape.
     * @param   inblock     Volume-shaped input block.
     * @param   outblock    Packed trace output block.
     * @param   outOffset   Offset of this trace's values within one
     *                      element's packed trace entries, counted in
     *                      entries per element. It indexes the output
     *                      as a simd_t array, so it is not scaled by
     *                      the interleave width.
     *
     * @note A segment block raises a fatal error: there is no
     * per-trace segment path.
     *
     * @note In three dimensions the general arm of the face extraction
     * this reaches, the one taken when the normal direction's volume
     * rule has no domain endpoints, indexes the normal-direction table
     * as `ntbasis[p * tstride + f]` with @c tstride the direction's
     * trace count rather than the face-loop bound, which is one on this
     * route. The two differ for the lower trace of a direction that
     * carries a pair, so the distinction matters here in a way it does
     * not on the bulk route.
     */
    // This is a debugging method to test individual trace extraction
    void v_ExtractTrace(
        const unsigned int traceid,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        const unsigned int outOffset) override
    {
        // Shape size.
        const auto numDataIn  = inblock.GetNumData();
        const auto numDataOut = outblock.GetNumData();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();
        const auto width_ratio     = (interleaveWidth == 1)
                                         ? 1
                                         : interleaveWidth / m_implInterleaveWidth;
        const auto chunkSize = std::max(m_implInterleaveWidth, interleaveWidth);

        // Loop over components.
        for (unsigned int n = 0;
             n < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++n)
        {
            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        m_implInterleaveWidth, interleaveWidth, chunkSize,
                        numDataIn, (TData *)inptr);
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        m_implInterleaveWidth, interleaveWidth, chunkSize,
                        numDataOut, (TData *)outptr);
                }

                switch (m_shapeType)
                {
                    // Segment
                    // Quads
                    case LibUtilities::Quad:
                        PhysTraceExtractEdgeKernel<LibUtilities::Quad>(
                            traceid, m_nm[0], m_nm[1], m_nq[0], m_nq[1], m_B[0],
                            m_B[1], m_B[2], m_B[3], m_wsp[0].data(),
                            reinterpret_cast<const simd_t *>(inptr),
                            reinterpret_cast<simd_t *>(outptr) + outOffset,
                            this->m_isCollocated[0], this->m_isCollocated[1],
                            m_endPtsCollocated[0], m_endPtsCollocated[1]);
                        break;
                    // Nodal Triangles
                    case LibUtilities::Tri:
                    case LibUtilities::NodalTri:
                    {
                        PhysTraceExtractEdgeKernel<LibUtilities::Tri>(
                            traceid, m_nm[0], m_nm[1], m_nq[0], m_nq[1], m_B[0],
                            m_B[1], m_B[2], m_B[3], m_wsp[0].data(),
                            reinterpret_cast<const simd_t *>(inptr),
                            reinterpret_cast<simd_t *>(outptr) + outOffset,
                            this->m_isCollocated[0], this->m_isCollocated[1],
                            m_endPtsCollocated[0], m_endPtsCollocated[1]);

                        break;
                    }
                    // Hexes
                    case LibUtilities::Hex:
                    {
                        PhysTraceFaceExtractKernel<LibUtilities::Hex>(
                            traceid, m_nm[0], m_nm[1], m_nm[2], m_nq[0],
                            m_nq[1], m_nq[2], m_nq[3], m_nq[4], m_nq[5], m_B[0],
                            m_B[1], m_B[2], m_B[3], m_B[4], m_B[5], m_B[6],
                            m_B[7], m_B[8], m_wsp[0].data(), m_wsp[1].data(),
                            reinterpret_cast<const simd_t *>(inptr),
                            reinterpret_cast<simd_t *>(outptr) + outOffset,
                            this->m_isCollocated[0], this->m_isCollocated[1],
                            this->m_isCollocated[2], this->m_isCollocated[3],
                            this->m_isCollocated[4], this->m_isCollocated[5],
                            m_endPtsCollocated[0], m_endPtsCollocated[1],
                            m_endPtsCollocated[2]);
                        break;
                    }
                    // Tet
                    case LibUtilities::NodalTet:
                    case LibUtilities::Tet:
                    {
                        PhysTraceFaceExtractKernel<LibUtilities::Tet>(
                            traceid, m_nm[0], m_nm[1], m_nm[2], m_nq[0],
                            m_nq[1], m_nq[2], m_nq[3], m_nq[4], m_nq[5], m_B[0],
                            m_B[1], m_B[2], m_B[3], m_B[4], m_B[5], m_B[6],
                            m_B[7], m_B[8], m_wsp[0].data(), m_wsp[1].data(),
                            reinterpret_cast<const simd_t *>(inptr),
                            reinterpret_cast<simd_t *>(outptr) + outOffset,
                            this->m_isCollocated[0], this->m_isCollocated[1],
                            this->m_isCollocated[2], this->m_isCollocated[3],
                            this->m_isCollocated[4], this->m_isCollocated[5],
                            m_endPtsCollocated[0], m_endPtsCollocated[1],
                            m_endPtsCollocated[2]);
                        break;
                    }
                    // Pyr
                    case LibUtilities::Pyr:
                    {
                        PhysTraceFaceExtractKernel<LibUtilities::Pyr>(
                            traceid, m_nm[0], m_nm[1], m_nm[2], m_nq[0],
                            m_nq[1], m_nq[2], m_nq[3], m_nq[4], m_nq[5], m_B[0],
                            m_B[1], m_B[2], m_B[3], m_B[4], m_B[5], m_B[6],
                            m_B[7], m_B[8], m_wsp[0].data(), m_wsp[1].data(),
                            reinterpret_cast<const simd_t *>(inptr),
                            reinterpret_cast<simd_t *>(outptr) + outOffset,
                            this->m_isCollocated[0], this->m_isCollocated[1],
                            this->m_isCollocated[2], this->m_isCollocated[3],
                            this->m_isCollocated[4], this->m_isCollocated[5],
                            m_endPtsCollocated[0], m_endPtsCollocated[1],
                            m_endPtsCollocated[2]);
                        break;
                    }
                    // Prism
                    case LibUtilities::NodalPrism:
                    case LibUtilities::Prism:
                    {
                        PhysTraceFaceExtractKernel<LibUtilities::Prism>(
                            traceid, m_nm[0], m_nm[1], m_nm[2], m_nq[0],
                            m_nq[1], m_nq[2], m_nq[3], m_nq[4], m_nq[5], m_B[0],
                            m_B[1], m_B[2], m_B[3], m_B[4], m_B[5], m_B[6],
                            m_B[7], m_B[8], m_wsp[0].data(), m_wsp[1].data(),
                            reinterpret_cast<const simd_t *>(inptr),
                            reinterpret_cast<simd_t *>(outptr) + outOffset,
                            this->m_isCollocated[0], this->m_isCollocated[1],
                            this->m_isCollocated[2], this->m_isCollocated[3],
                            this->m_isCollocated[4], this->m_isCollocated[5],
                            m_endPtsCollocated[0], m_endPtsCollocated[1],
                            m_endPtsCollocated[2]);
                        break;
                    }
                    default:
                        NEKERROR(ErrorUtil::efatal,
                                 "shape type not implemented");
                }

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        numDataIn,
                        (TData *)inptr -
                            (width_ratio - 1) * numDataIn * simd_t::width);
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        numDataOut,
                        (TData *)outptr -
                            (width_ratio - 1) * numDataOut * simd_t::width);
                }

                // Increment pointers for the next elmt group.
                inptr += numDataIn * simd_t::width;
                outptr += numDataOut * simd_t::width;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    /**
     * @brief Bulk path: extract every trace of every element of the
     * block, filling the packed output in packing order.
     *
     * @details
     * A debug build first checks that both blocks are aligned to the
     * SIMD type's alignment, since the kernels address them through
     * simd_t pointers, and #m_warnOnce keeps that warning to one
     * occurrence per block operator. Neither the check nor the flag
     * exists in a release build. Dispatch is then per shape into
     * ShapeBlock(), whose specialisation the generated translation unit
     * defines and which selects the size-templated OperatorND()
     * instantiation.
     *
     * @param   inblock     Volume-shaped input block.
     * @param   outblock    Packed trace output block.
     *
     * @note The switch covers all ten shapes CMake generates entry
     * points for; the default arm raises a fatal error.
     */
    void v_Apply(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock) override
    {
        WARNINGL1(
            m_warnOnce || (inblock.GetAlignment() % simd_t::alignment == 0 &&
                           outblock.GetAlignment() % simd_t::alignment == 0),
            "Input or output Field are not aligned to the required alignment "
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
    // PhysTraceExtractGenericBlockOp.cpp.in.
    template <LibUtilities::ShapeType SHAPE_TYPE>
    void ShapeBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock);

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
        // Operators/Common/BlockOpSwitchPhysTraceExtract.h.in, which this
        // operator shares with its adjoint rather than taking the
        // LibUtilities/BasicUtils/Switch/BlockOpSwitchCode.h.in the volume
        // operators use: the trace size parameters carry trace point counts
        // per normal direction, which that switch knows nothing of.
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

    /// @brief Runtime-sized entry: passes the stored counts to the
    /// core overload.
    /// @name Dimensional entry points
    /// Each builds the block's size parameter, non-templated where the
    /// generated switch found no compiled combination and templated
    /// where it did, and hands OperatorND() the index sequences that
    /// select this dimension's slice of #m_B, #m_isCollocated,
    /// #m_endPtsCollocated and #m_wsp. The first sequence runs over the
    /// normal directions, the second over the tangential directions and the
    /// third over the workspaces: one, none and none in one dimension,
    /// two, two and one in two, three, six and two in three.
    /// @{

    /// @}

    /**
     * @brief Bulk path for every dimension: extract all of an element's
     * traces, filling the packed output in packing order.
     *
     * @details
     * The whole trace loop lives in PhysTraceExtractKernelLauncher,
     * whose one-, two- and three-dimensional arms this one body calls;
     * overload resolution picks the arm from the number of arguments
     * the index sequences expand to. Each walks the normal directions
     * in packing order: in two dimensions the direction-0 pair first,
     * then the direction-1 pair of a quadrilateral or the single edge 0
     * of a triangle; in three the direction-0 pair, then the
     * direction-1 faces, then the direction-2 faces; in one the two end
     * vertices, written contiguously at the start of each element's
     * output.
     *
     * Both blocks are reshaped to #m_implInterleaveWidth at the start
     * of each chunk of `width_ratio` element groups and back at its
     * end, and the pointers walk one element group per iteration.
     *
     * The component loop carries the GetNumHomoModes factor. That is
     * inert for three-dimensional elements, which support no
     * homogeneous extension and report one plane, and correct in one
     * and two dimensions, where it has every plane extracted.
     *
     * @tparam SHAPE_TYPE   Shape of the block, the nodal enumerators
     *                      included: the generated bulk path passes
     *                      them straight through rather than folding
     *                      them onto their parents, as v_ExtractTrace()
     *                      does.
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
     *                      dimension, where a trace is a point and the
     *                      one-dimensional launcher supplies the flag
     *                      itself.
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
        const auto numDataIn  = sizeParam.nmTot();
        const auto numDataOut = sizeParam.template nqTotTrace<SHAPE_TYPE>();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();
        const auto width_ratio     = (interleaveWidth == 1)
                                         ? 1
                                         : interleaveWidth / m_implInterleaveWidth;
        const auto chunkSize = std::max(m_implInterleaveWidth, interleaveWidth);

        // Loop over components.
        for (unsigned int n = 0;
             n < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++n)
        {
            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        m_implInterleaveWidth, interleaveWidth, chunkSize,
                        numDataIn, (TData *)inptr);
                }

                // PhysTraceExtract kernel.
                PhysTraceExtractKernelLauncher<SHAPE_TYPE>(
                    sizeParam, m_B[ind0]..., m_B[sizeof...(ind0) + ind1]...,
                    m_wsp[ind2].data()...,
                    reinterpret_cast<const simd_t *>(inptr),
                    reinterpret_cast<simd_t *>(outptr),
                    (bool)this->m_isCollocated[ind1]...,
                    (bool)m_endPtsCollocated[ind0]...);

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        numDataIn,
                        (TData *)inptr -
                            (width_ratio - 1) * numDataIn * simd_t::width);
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        numDataOut,
                        (TData *)outptr -
                            (width_ratio - 1) * numDataOut * simd_t::width);
                }

                // Increment pointers for the next elmt group.
                inptr += numDataIn * simd_t::width;
                outptr += numDataOut * simd_t::width;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
