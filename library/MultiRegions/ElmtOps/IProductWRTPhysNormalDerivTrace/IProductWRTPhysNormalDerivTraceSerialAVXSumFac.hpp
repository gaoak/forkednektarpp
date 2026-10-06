///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTPhysNormalDerivTraceSerialAVXSumFac.hpp
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
// Description: Serial/AVX block operator of the lift against the normal
// derivative of the test function
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file IProductWRTPhysNormalDerivTraceSerialAVXSumFac.hpp
 * @brief Serial and AVX dispatch of the sum-factorised surface inner
 * product against the normal derivative of the volume cardinal basis.
 *
 * @details
 * This header defines the primary
 * detail::IProductWRTPhysNormalDerivTraceBlockOpImpl template, which
 * serves the Serial and the AVX execution spaces; the Device space is
 * served by IProductWRTPhysNormalDerivTraceDeviceSumFac.hpp. The two
 * headers are the same operator with a different packing of the elements,
 * SIMD vector lanes here and warp lanes there. The vector type is
 * `tinysimd::simd<TData>` for AVX and `tinysimd::scalarT<TData>`, of width
 * one, for Serial.
 *
 * The constructor caches the block's interpolation tables and their
 * derivatives, the trace weights, the collapsed-coordinate factors and the
 * trace derivative factors from the data warehouse, and allocates the
 * workspaces. Each application brings the block storage to `simd_t::width`
 * a chunk at a time and calls the kernels of
 * IProductWRTPhysNormalDerivTraceSerialAVXSumFacKernels.hpp per shape.
 * No arithmetic on the field happens here.
 *
 * CMake generates one translation unit per shape and data type from
 * LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in (the
 * Serial/AVX SumFac branch of library/MultiRegions/CMakeLists.txt). Those
 * units define the per-shape entry points declared below, expanding
 * LibUtilities/BasicUtils/Switch/BlockOpSwitchPhysTraceExtract.h.in,
 * this operator's switch template. On Serial and AVX this operator
 * registers a single implementation, so the class is only ever
 * instantiated for MultiRegions::SumFac and its Implementation template
 * parameter is not read anywhere.
 *
 * @see IProductWRTPhysNormalDerivTraceOp.hpp for what the operator
 * computes and how the family is laid out.
 * @see IProductWRTPhysTraceSerialAVXSumFac.hpp for the plain trace lift
 * this operator is built on, whose layout it follows.
 */

#pragma once

#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "LocalRegions/DataWarehouse/GeometricDataWarehouse.hpp"
#include <MultiRegions/ElmtOps/IProductWRTPhysNormalDerivTrace/IProductWRTPhysNormalDerivTraceBlockOp.hpp>
#include <MultiRegions/ElmtOps/IProductWRTPhysNormalDerivTrace/IProductWRTPhysNormalDerivTraceSerialAVXSumFacKernels.hpp>

// Selects the switch construction used by the generated ShapeBlock
// definitions (see LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in).
#define NEKTAR_BLOCKOP_SWITCH_PHYSTRACE

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Serial and AVX block implementation of the normal-derivative
 * trace lift, one element per SIMD lane.
 *
 * @details
 * For every element of the block this accumulates
 * \f[ \Lambda_{\boldsymbol p} \mathrel{+}= \sum_{F \in \partial E}
 *     \int_F g \; \partial_n h_{\boldsymbol p}\big|_F \, \mathrm{d}s , \f]
 * the trace field tested against the normal derivative of each cardinal
 * function of the volume quadrature grid, with the normal and the
 * geometric factors folded into the per-point factor array the data
 * warehouse supplies. The class itself only marshals data: the
 * constructor caches the tables, OperatorND() drives the interleave, and
 * every floating-point operation lives in the kernels.
 *
 * #m_nm holds GetNumPoints, that is volume quadrature counts per
 * direction. #m_B holds two families of eInterp tables: its first
 * #m_dimension entries are the normal-direction tables \f$h_p(\pm 1)\f$
 * and the remainder the tangential tables \f$h_p(\xi^{tr})\f$; #m_DB
 * holds their derivatives in the same order, and the kernels choose,
 * term by term, which of the two a slot reads. The tables are fetched
 * with a `BasisDataKey<simd_t>`, so each coefficient occupies one whole
 * SIMD vector; field, trace and factor data are interleaved one element
 * per lane and reach the kernels through a reinterpret_cast of the
 * block's TData storage to simd_t.
 *
 * The factor array orders traces by normal direction as the packed trace
 * input does, one value per trace point when the factors are pointwise
 * and one per trace otherwise, with the #m_dimension components one
 * whole block apart. A collapsed shape needs a factor per point whatever
 * its geometry, so #m_isDeformed follows
 * LocalRegions::TraceDerivFactorsArePointwise rather than the geometry
 * type alone; the warehouse applies the same rule when it builds the
 * array.
 *
 * @tparam ExecSpace       NektarSpaces::Serial or NektarSpaces::AVX,
 *                         which is what selects the vector type.
 * @tparam Implementation  MultiRegions::SumFac; nothing else is
 *                         generated, and the parameter is not read.
 * @tparam TData           Floating-point type of the field data.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class IProductWRTPhysNormalDerivTraceBlockOpImpl
    : public IProductWRTPhysNormalDerivTraceBlockOp<TData>
{
    using BlockOpBase = IProductWRTPhysNormalDerivTraceBlockOp<TData>;
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Cache the block's shape, its point counts, both families of
     * interpolation tables and their derivatives, the trace weights, the
     * collapsed-coordinate factors and the trace derivative factors, and
     * allocate the workspaces.
     *
     * @details
     * The first loop walks the normal directions, recording GetNumPoints
     * in #m_nm and fetching the eInterp and eInterpDerivative tables that
     * take the direction's volume rule to the positions of its traces: the
     * two Gauss-Lobatto points where the direction carries a trace pair
     * and the single point \f$\xi = -1\f$ where it collapses to one trace.
     * #m_endPtsCollocated records whether the direction's own rule contains
     * domain endpoints.
     *
     * The second loop walks the same normal directions in packing order,
     * taking one representative trace each, trace `m_dimension - 1 - dim`,
     * so that the loop index is the normal direction for every shape. For
     * each in-trace direction it records the trace quadrature count in
     * #m_nq, the element direction the slot runs along in #m_traceDir,
     * whether the trace points coincide with the volume points in
     * IProductWRTPhysNormalDerivTraceBlockOp::m_isCollocated, and the
     * eInterp and eInterpDerivative tables onto the trace points in #m_B
     * and #m_DB, with the trace weights in #m_W.
     *
     * A segment has no in-trace directions; #m_nq is given the volume
     * point count so that the generated switch lands on a compiled
     * instantiation.
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
            auto bkey = exp->GetBasis(dir)->GetBasisKey();
            m_nm.push_back(exp->GetNumPoints(dir));

            unsigned int ntrace =
                LibUtilities::ShapeTypeNumTraceInDir[m_shapeType][dir];

            // GLwithM nodes give xi = -1 alone where the direction has one
            // trace, GLL both ends otherwise.
            LibUtilities::PointsType endPtsType =
                (ntrace == 1) ? LibUtilities::eGaussLegendreWithM
                              : LibUtilities::eGaussLobattoLegendre;

            m_B.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(bkey, LibUtilities::eInterp,
                                                   ntrace, endPtsType)));

            m_DB.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
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
            auto tr = m_dimension - 1 - dim;
            for (unsigned int d = 0; d < m_dimension - 1; ++d)
            {
                auto dir      = exp->GetGeom()->GetDir(tr, d);
                auto trBKey   = exp->GetTraceBasisKey(tr, d);
                auto trPtsKey = trBKey.GetPointsKey();
                auto nq       = trPtsKey.GetNumPoints();

                this->m_isCollocated.push_back(expPtsKeys[dir] == trPtsKey);
                m_nq.push_back(nq);
                m_traceDir.push_back(dir);

                auto dirBKey = exp->GetBasis(dir)->GetBasisKey();

                m_B.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                    LibUtilities::BasisDataKey<simd_t>(
                        dirBKey, LibUtilities::eInterp, nq,
                        trPtsKey.GetPointsType())));

                m_DB.push_back(
                    this->m_dataWarehouse->template GetData<MemSpace>(
                        LibUtilities::BasisDataKey<simd_t>(
                            dirBKey, LibUtilities::eInterpDerivative, nq,
                            trPtsKey.GetPointsType())));

                m_W.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                    LibUtilities::BasisDataKey<simd_t>(
                        trBKey, LibUtilities::eWeights)));
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
        bool collapsed = false;
        for (unsigned int dir = 1; dir < m_dimension; ++dir)
        {
            if (IsCollapsedDir(m_shapeType, dir))
            {
                collapsed = true;
                m_twoOverOneMinusZ[dir] =
                    this->m_dataWarehouse->template GetData<MemSpace>(
                        LibUtilities::BasisDataKey<simd_t>(
                            exp->GetBasis(dir)->GetBasisKey(),
                            LibUtilities::eTwoOverOneMinusZero));
            }
        }

        // Workspaces, sized for the largest intermediate any shape of this
        // dimension can need, in the order the third index sequence of
        // OperatorND() selects them. A segment's traces are points, so it
        // has no trace-mode buffer of its own.
        unsigned int numDataOut = 1;
        for (unsigned int dir = 0; dir < m_dimension; ++dir)
        {
            numDataOut *= m_nm[dir];
        }
        if (m_dimension == 2)
        {
            // The trace-mode buffer of an edge pair.
            m_wsp.push_back(std::vector<simd_t, tinysimd::allocator<simd_t>>(
                std::max(m_nm[0] * 2, m_nm[1] * 2)));
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

        // Last of them: a collapsed shape accumulates the trace groups
        // that carry a 2/(1 - eta) factor into a volume-sized buffer and
        // scales them there, at the element's own Gauss-Radau points. A
        // shape with no collapsed direction leaves the slot empty.
        m_wsp.push_back(std::vector<simd_t, tinysimd::allocator<simd_t>>(
            collapsed ? numDataOut : 0));
    }

    /// Registration name for BlockOperatorFactory, defined by the
    /// generated factory declaration unit.
    static std::string className;

    /// @brief Creator function registered with BlockOperatorFactory;
    /// builds one block operator for the given block of elements.
    /// Implementation is the tag this class is registered under; see
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
    /// Elements the kernels process at once: `simd_t::width`, the SIMD
    /// vector width in the AVX space and one in Serial. OperatorNDImpl()
    /// reshapes the block storage to this width around every kernel call
    /// and scales its per-element offsets by it.
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

    /// Shape shared by every element of the block.
    LibUtilities::ShapeType m_shapeType;
    /// Whether the factor array holds one value per trace quadrature point
    /// rather than one per trace: deformed geometry, or a collapsed shape
    /// whatever its geometry. Read by the generated dispatch to pick the
    /// DEFORMED instantiation, which is what fixes the name: the switch
    /// template
    /// LibUtilities/BasicUtils/Switch/BlockOpSwitchPhysTraceExtract.h.in reads
    /// it by name.
    bool m_isDeformed;
    /// Dimension of the reference element (1, 2 or 3).
    unsigned int m_dimension;
    /// Volume quadrature points per direction, unpadded.
    std::vector<unsigned int> m_nm;
    /// Trace quadrature points, indexed `2 * dir + tangential` in three
    /// dimensions and by normal direction in two. In one dimension the
    /// single entry is a copy of the volume point count.
    std::vector<unsigned int> m_nq;
    /// eInterp tables, each coefficient broadcast across a whole SIMD
    /// vector: first the #m_dimension normal-direction tables, then the
    /// tangential tables in the same (direction, tangential) order as
    /// #m_nq.
    std::vector<const simd_t *> m_B;
    /// eInterpDerivative tables in the same order as #m_B.
    std::vector<const simd_t *> m_DB;
    /// Trace quadrature weights per (normal direction, tangential
    /// direction), in the same order as #m_nq and likewise broadcast.
    /// Empty in one dimension.
    std::vector<const simd_t *> m_W;
    /// 2/(1 - eta) at the element points, per direction; null where the
    /// direction is not collapsed.
    std::vector<const simd_t *> m_twoOverOneMinusZ;
    /// Trace Jacobian times normal times geometric factors of the block,
    /// one component after the other, held as TData interleaved to
    /// #m_implInterleaveWidth.
    const TData *m_jacNormGeomFacPtr;
    /// The factor set the next apply consumes: the contracted factors above
    /// by default, or one Cartesian direction's uncontracted factors when a
    /// vector-input pass selected it via v_SetActiveDir().
    const TData *m_jacptr = nullptr;
    /// Uncontracted per-direction factors, fetched on first use.
    std::vector<const TData *> m_jacDirGeomFac;
    /// Element direction each tangential trace slot runs along, in the
    /// order of #m_nq; decides which slots carry the derivative table in
    /// each term.
    std::vector<unsigned int> m_traceDir;
    /// One flag per normal direction: that direction's volume rule
    /// contains domain endpoints.
    std::vector<bool> m_endPtsCollocated;
    /// Kernel workspaces, allocated once by the constructor and indexed
    /// by the third index sequence OperatorND() is handed: the trace-mode
    /// buffer in two dimensions, the face mode block followed by the
    /// contraction scratch in three, none of them in one, and in every
    /// dimension the volume-sized scratch of the collapsed factor scaling
    /// last, empty for a shape with no collapsed direction.
    std::vector<std::vector<simd_t, tinysimd::allocator<simd_t>>> m_wsp;
#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
    // flag to ensure we only get one warning for alignment otherwise CI
    // system is saturated with warnings
    bool m_warnOnce = false;
#endif

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
     * @brief Bulk path: lift every trace of every element of the block.
     *
     * Checks first that both blocks are aligned for simd_t, then
     * dispatches to the per-shape entry point the generated translation
     * unit defines, which selects the size-templated OperatorND()
     * instantiation.
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

    // Workspaces #m_wsp holds in dim dimensions: the trace-mode buffers
    // of the dimension, plus the collapsed-scaling buffer in every one.
    static constexpr unsigned int NumWorkspace(const unsigned int dim)
    {
        return (dim == 1) ? 1 : (dim == 2) ? 2 : 3;
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
            std::make_integer_sequence<unsigned int, NumTangentialDir(DIM)>(),
            std::make_integer_sequence<unsigned int, NumWorkspace(DIM)>());
    }

    /**
     * @brief Bulk path for every dimension: lift every trace of every
     * element of the block onto the volume field.
     *
     * @details
     * The whole trace loop lives in
     * IProductWRTPhysNormalDerivTraceKernelLauncher, whose one-, two- and
     * three-dimensional arms this one body calls; overload resolution picks
     * the arm from the number of arguments the index sequences expand to.
     *
     * The output is always reshaped to #m_implInterleaveWidth on the way
     * in, whether appending or not: the kernels accumulate the later terms
     * onto the first inside the same layout, and the back-reshape assumes
     * the storage was brought to it. The append flag reaches the launcher
     * as a run-time argument, as does the direction term inside it, so the
     * size-specialised body is instantiated once per size and geometry.
     *
     * @c jacoffset walks the factor array alongside the field: by the
     * element group's worth of trace points when the factors are pointwise,
     * by one slot per trace when they are not. @c jacCompStride is the
     * distance between the #m_dimension components of the factor array,
     * which is a whole block of those same entries, counted in SIMD
     * vectors because the launcher advances a simd_t pointer by it.
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
     * @tparam ind2   Workspaces, selecting #m_wsp: one in one dimension,
     *                two in two, three in three, the collapsed-scaling
     *                buffer being the last of them in every dimension.
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

        // Stride between the components of the factor array, in SIMD
        // vectors since the kernels walk the array as simd_t: one value per
        // trace point when pointwise, one per trace otherwise, over every
        // element group of the block. That is not the input component size,
        // which always counts every trace point.
        const size_t jacCompStride =
            inblock.GetNumElmtGroups(m_implInterleaveWidth) *
            (DEFORMED ? numDataIn
                      : LibUtilities::ShapeTypeNumTraces[SHAPE_TYPE]);

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
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        m_implInterleaveWidth, outInterleaveWidth, chunkSize,
                        numDataOut, (TData *)outptr);
                }

                const TData *jacptr = m_jacptr + jacoffset;

                IProductWRTPhysNormalDerivTraceKernelLauncher<SHAPE_TYPE,
                                                              DEFORMED>(
                    sizeParam, m_B[ind0]..., m_DB[ind0]...,
                    m_B[sizeof...(ind0) + ind1]...,
                    m_DB[sizeof...(ind0) + ind1]..., m_W[ind1]...,
                    m_twoOverOneMinusZ[ind0]...,
                    reinterpret_cast<const simd_t *>(jacptr), jacCompStride,
                    m_wsp[ind2].data()...,
                    reinterpret_cast<const simd_t *>(inptr),
                    reinterpret_cast<simd_t *>(outptr), m_traceDir[ind1]...,
                    this->m_append, (bool)this->m_isCollocated[ind1]...,
                    (bool)m_endPtsCollocated[ind0]...);

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

} // namespace Nektar::MultiRegions::detail
