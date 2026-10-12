///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivSerialAVXSumFac.hpp
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
// Description: Serial/AVX sum-factorised implementation of the
// per-block physical derivative.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file PhysDerivSerialAVXSumFac.hpp
 * @brief Serial/AVX sum-factorised (SumFac) implementation of the
 * per-block physical derivative.
 *
 * @details
 * Rather than one dense nqTot x nqTot matrix per standard direction,
 * the sum-factorised path differentiates along one reference direction
 * at a time with the 1D differentiation matrix of that direction
 * (PhysDerivTensor{1,2,3}DKernel in
 * StdRegions/Operators/PhysDerivSumFacStdKernels.hpp), then applies the
 * collapsed-coordinate and geometric chain rules point by point
 * (PhysDerivSerialAVXSumFacKernels.hpp). One template serves both CPU
 * execution spaces: `simd_t` resolves to tinysimd::scalarT<TData> for
 * Serial and tinysimd::simd<TData> for AVX (see simd_type_if), so the
 * same code runs one element at a time or one SIMD group of elements
 * in lock-step, each vector lane owning one element.
 *
 * v_Apply() dispatches on the block's shape type to the ShapeBlock()
 * specialisations, whose definitions CMake generates from
 * LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in, one
 * translation unit per shape: each wraps OperatorND() in the switch of
 * LibUtilities/BasicUtils/Switch/BlockOpSwitch1DCoordsCode.h.in -- over
 * the coordinate dimension and the quadrature-point counts, not over
 * mode counts, this operator working purely in physical space -- so
 * that compile-time constants reach the kernels, falling back to a
 * runtime-sized call for other sizes.
 *
 * The header defines the detail::PhysDerivBlockOpImpl used when the
 * "SumFac" implementation is requested on the Serial or AVX execution
 * space; see PhysDerivSerialAVXStdMat.hpp for the dense-matrix
 * alternative.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, execution space and
 * implementation; it should not normally be included by any other
 * translation unit. Other code goes through PhysDerivOp.hpp and the
 * operator factory: including this header directly instantiates the
 * whole template set in that translation unit, which is what the
 * generated sources exist to avoid.
 */

#pragma once

#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivBlockOp.hpp>

#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivSerialAVXSumFacKernels.hpp>

// Selects the switch construction used by the generated ShapeBlock
// definitions (see LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in).
#define NEKTAR_BLOCKOP_SWITCH_1DCOORDS

namespace Nektar::MultiRegions::detail
{

/**
 * @brief SumFac block implementation of the physical derivative for the
 * Serial and AVX execution spaces: tensorial differentiation of one
 * interleaved element group at a time, followed by the chain rules.
 *
 * @details
 * The constructor caches, per direction, the element's mode and point
 * counts and its 1D differentiation matrix (BasisDataKey with
 * eDerivative), and for the two- and three-dimensional shapes the
 * collapsed-coordinate factor tables
 * \f$\tfrac{1}{2}(1+\eta)\f$ and \f$2/(1-\eta)\f$ (BasisDataKey with
 * eHalfMultOnePlusZero and eTwoOverOneMinusZero) -- in the SIMD vector
 * type, so a lane group's worth of factors is loaded at once. It also
 * fetches the block's derivative factors (DerivFactorKey, untransposed,
 * at this implementation's interleave width). The tables are fetched
 * for every shape of the given dimension, including those that need no
 * collapsed correction; the kernels of the uncollapsed shapes simply
 * ignore them.
 *
 * The generated ShapeBlock() specialisations land in OperatorND(),
 * which forwards to OperatorNDImpl(): that loops over the components
 * and homogeneous modes of the block and, within each, over the
 * element groups, handing every group to PhysDerivKernelLauncher. The
 * tensorial derivatives are written straight into the first
 * `dimension` output planes of the current component and then
 * transformed in place into the `coordDim` Cartesian ones, so no
 * workspace is needed.
 *
 * @tparam ExecSpace       NektarSpaces::Serial or NektarSpaces::AVX;
 *                         sets simd_t via simd_type_if
 *                         (tinysimd::scalarT for Serial, tinysimd::simd
 *                         for AVX).
 * @tparam Implementation  Implementation tag the including translation
 *                         unit registers (MultiRegions::SumFac here).
 * @tparam TData           Floating-point type of the field data.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class PhysDerivBlockOpImpl : public PhysDerivBlockOp<TData>
{
    using BlockOpBase = PhysDerivBlockOp<TData>;
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Gather everything the kernels need from the block's
     * representative expansion and the data warehouse.
     *
     * Per reference direction this fetches the mode and point counts
     * and the 1D differentiation matrix (eDerivative) evaluated at the
     * direction's quadrature points. For 2D and 3D shapes it also
     * fetches the collapsed-coordinate factor tables consumed by the
     * PhysDeriv kernels for the simplex-type shapes: per direction,
     * eHalfMultOnePlusZero holds \f$(1+z_i)/2\f$ and
     * eTwoOverOneMinusZero holds \f$2/(1-z_i)\f$ at the quadrature
     * points, stored as `simd_t` entries so one reference-space factor
     * applies across all lanes. Finally it fetches the block's
     * derivative factors interleaved at the SIMD width.
     *
     * @param   block_idx       Index of the block within the expansion
     *                          list's Collections.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    PhysDerivBlockOpImpl(const unsigned int block_idx,
                         const LocalRegions::ExpansionSharedPtr &exp,
                         LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : PhysDerivBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
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
            // Fetch element size.
            m_nm.push_back(exp->GetBasisNumModes(d));
            m_nq.push_back(exp->GetNumPoints(d));

            // Fetch basis data.
            m_D.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    exp->GetBasis(d)->GetBasisKey(),
                    LibUtilities::eDerivative)));
        }

        if (m_dimension == 2)
        {
            // Fetch geometric factors.
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    this->m_exp->GetBasis(0)->GetBasisKey(),
                    LibUtilities::eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    this->m_exp->GetBasis(1)->GetBasisKey(),
                    LibUtilities::eTwoOverOneMinusZero)));
        }
        else if (m_dimension == 3)
        {
            // Fetch geometric factors.
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    this->m_exp->GetBasis(0)->GetBasisKey(),
                    LibUtilities::eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    this->m_exp->GetBasis(1)->GetBasisKey(),
                    LibUtilities::eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    this->m_exp->GetBasis(1)->GetBasisKey(),
                    LibUtilities::eTwoOverOneMinusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    this->m_exp->GetBasis(2)->GetBasisKey(),
                    LibUtilities::eTwoOverOneMinusZero)));
        }

        // Fetch deriv factors data.
        m_dfptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::DerivFactorKey<TData>(block_idx,
                                                m_implInterleaveWidth, false));
    }

    /// Block-operator class name; defined by generated factory code.
    static std::string className;

    /// Creator function registered with BlockOperatorFactory.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            PhysDerivBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    /// Interleave width the kernels operate at: the SIMD vector width
    /// (1 on Serial). Also the width the derivative factors are fetched
    /// at.
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

    /// Shape shared by every element of the block.
    LibUtilities::ShapeType m_shapeType;
    /// Whether the block's geometry is deformed; selects the DEFORMED
    /// kernel instantiations, in which the derivative factors vary per
    /// quadrature point rather than per element.
    bool m_isDeformed;
    /// First-direction basis is eModified_A; recorded for uniformity
    /// with the other families but not used by this operator, whose
    /// kernels branch on the shape alone.
    bool m_isModified;
    /// Dimension of the reference element (1, 2 or 3); the number of
    /// tensorial derivatives taken per element.
    unsigned int m_dimension;
    /// Coordinate dimension of the (possibly embedded) element; the
    /// number of Cartesian derivative components produced.
    unsigned int m_coordDim;
    /// Modes per direction; recorded at construction but not used by
    /// this operator, which works purely in physical space.
    std::vector<unsigned int> m_nm;
    /// Quadrature points per direction.
    std::vector<unsigned int> m_nq;
    /// 1D differentiation matrix per direction.
    std::vector<const TData *> m_D;
    /// Collapsed-coordinate factor tables, broadcast to SIMD vectors:
    /// in 2D `(f0, f1)`, in 3D `(f0, f1, f1m, f2)` as named in the
    /// kernels; empty in 1D.
    std::vector<const simd_t *> m_f;
    /// Derivative factors of the block, interleaved at
    /// #m_implInterleaveWidth; `coordDim * dimension` values per
    /// element, or per quadrature point when the geometry is deformed.
    const TData *m_dfptr;
#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
    // flag to ensure we only get one warning for alignment otherwise CI system
    // is saturated with warnings
    bool m_warnOnce = false;
#endif

    /**
     * @brief Warn (once, in debug builds) if the fields are not
     * aligned for the SIMD type, then dispatch to the generated
     * ShapeBlock() specialisation matching #m_shapeType. An
     * unsupported shape leaves the output block unwritten and reports
     * on std::cout.
     *
     * @param   inblock     Physical-space input block.
     * @param   outblock    Physical-space output block, holding
     *                      coordDim (or, for 3DH1, three) components
     *                      per input component; overwritten.
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
                std::cout << "shapetype not implemented" << std::endl;
        }
    }

    // Shape specific block operator, specialised for each shape in
    // LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in.
    template <LibUtilities::ShapeType SHAPE_TYPE>
    void ShapeBlock(typename BlockOpBase::InBlock &inblock,
                    typename BlockOpBase::OutBlock &outblock);

    // Number of collapsed coordinate factors used by the kernels in dim
    // dimensions.
    static constexpr unsigned int NumFactor(const unsigned int dim)
    {
        return (dim == 1) ? 0 : (dim == 2) ? 2 : 4;
    }

    // Generic operator. Builds the index sequences from the shape dimension
    // and forwards to OperatorNDImpl.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TPhysSizeParameter>
    NEK_FORCE_INLINE void OperatorND(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        TPhysSizeParameter sizeParam)
    {
        constexpr unsigned int DIM = LibUtilities::ShapeTypeDimMap[SHAPE_TYPE];

        // sizeParam is built by the switch in
        // LibUtilities/BasicUtils/Switch/BlockOpSwitch1DCoordsCode.h.in.
        static_assert(
            (DIM == 1 && IsPhysSizeParameter1D_v<TPhysSizeParameter>) ||
                (DIM == 2 && IsPhysSizeParameter2D_v<TPhysSizeParameter>) ||
                (DIM == 3 && IsPhysSizeParameter3D_v<TPhysSizeParameter>),
            "OperatorND expects a size parameter matching the dimension of the "
            "shape.");

        OperatorNDImpl<SHAPE_TYPE, DEFORMED>(
            inblock, outblock, sizeParam,
            std::make_integer_sequence<unsigned int, DIM>(),
            std::make_integer_sequence<unsigned int, NumFactor(DIM)>());
    }

    /**
     * @brief Worker for every dimension: differentiate all element
     * groups of the block, one component and homogeneous mode at a
     * time.
     *
     * @details
     * The output planes are addressed through `outvec[k]`, plane @em k
     * of the current component starting `k * compOffset` vectors into
     * the output block, `compOffset` being one component's worth of
     * element groups times points times homogeneous modes; `outDim`
     * such planes are tracked, `coordDim` of them, or three for a 3DH1
     * field with more than one homogeneous plane. For each component
     * and homogeneous mode the derivative-factor pointer restarts at
     * the beginning of the block -- the same geometry serves every one
     * of them -- and every element group is handed to
     * PhysDerivKernelLauncher, whose one-, two- and three-dimensional
     * overloads this one body calls; overload resolution picks the arm
     * from the number of arguments the index sequences expand to. The
     * launcher writes the group's tensorial derivatives into
     * `outvec[0 .. ndim-1]` and converts them in place into the
     * `coordDim` Cartesian derivatives, after which the factor pointer
     * advances by one group's factors (`ndim * coordDim` of them,
     * times nqTot for deformed geometry), the input pointer by one
     * group's values and every `outvec[k]` by one group's vectors. Once
     * a component's homogeneous modes are exhausted, every `outvec[k]`
     * additionally skips the `outDim - 1` further planes that
     * component owns, leaving the third free for the z-derivative in
     * the 3DH1 case (see PhysDerivOp).
     *
     * Fields stored at an interleave width other than the
     * implementation width are handled by reshaping `width_ratio`
     * consecutive groups in place: the input is converted to the
     * implementation width before the first group of a chunk, and it
     * and the `outDim` output planes are converted back to the input's
     * width after its last group. The output block's recorded
     * interleave width is set to the input's on return.
     *
     * @tparam SHAPE_TYPE          Shape of the block, the nodal
     *                             enumerators included.
     * @tparam DEFORMED            Derivative factors vary point by
     *                             point.
     * @tparam TPhysSizeParameter  Size parameter of this dimension,
     *                             templated or not.
     * @tparam ind0                Reference directions, 0 to ndim - 1;
     *                             selects the differentiation matrices
     *                             #m_D.
     * @tparam ind1                Collapsed-coordinate factor tables
     *                             #m_f the kernels use: none in 1D, two
     *                             in 2D, four in 3D (see NumFactor()).
     *
     * @param   inblock     Physical-space input block.
     * @param   outblock    Physical-space output block; overwritten.
     * @param   sizeParam   Element sizes, in runtime or compile-time
     *                      form.
     */
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TPhysSizeParameter, unsigned int... ind0,
              unsigned int... ind1>
    NEK_FORCE_INLINE void OperatorNDImpl(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        TPhysSizeParameter sizeParam,
        std::integer_sequence<unsigned int, ind0...>,
        std::integer_sequence<unsigned int, ind1...>)
    {
        constexpr unsigned int ndim = sizeof...(ind0);

        // Shape size.
        const auto nqTot = sizeParam.nqTot();

        unsigned int dfsize = ndim * m_coordDim;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();
        const auto width_ratio     = (interleaveWidth == 1)
                                         ? 1
                                         : interleaveWidth / m_implInterleaveWidth;
        const auto chunkSize = std::max(m_implInterleaveWidth, interleaveWidth);

        // Offsets between the components of a block, in simd_t units.
        const auto nhomo          = outblock.GetNumHomoModes();
        const unsigned int outDim = (nhomo > 1) ? 3u : m_coordDim;
        const auto compOffset     = outblock.CompSize() * nhomo / simd_t::width;
        simd_t *outvec[3];
        for (unsigned int k = 0; k < outDim; ++k)
        {
            outvec[k] = reinterpret_cast<simd_t *>(outptr) + k * compOffset;
        }

        // Loop over components.
        for (unsigned int n = 0; n < inblock.GetNumComponents() * nhomo; ++n)
        {
            auto dfptr = m_dfptr;

            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        m_implInterleaveWidth, interleaveWidth, chunkSize,
                        nqTot, (TData *)inptr);
                }

                // PhysDeriv kernel.
                PhysDerivKernelLauncher<SHAPE_TYPE, DEFORMED>(
                    sizeParam, m_D[ind0]..., m_f[ind1]...,
                    reinterpret_cast<const simd_t *>(dfptr),
                    reinterpret_cast<const simd_t *>(inptr), outvec);

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        nqTot,
                        (TData *)inptr -
                            (width_ratio - 1) * nqTot * simd_t::width);
                    for (unsigned int k = 0; k < outDim; k++)
                    {
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            interleaveWidth, m_implInterleaveWidth, chunkSize,
                            nqTot,
                            (TData *)outvec[k] -
                                (width_ratio - 1) * nqTot * simd_t::width);
                    }
                }

                // Increment pointers for the next elmt group.
                dfptr += dfsize * simd_t::width;
                inptr += nqTot * simd_t::width;
                for (unsigned int k = 0; k < outDim; ++k)
                {
                    outvec[k] += nqTot;
                }
            }

            // Advance by ncoord-1 components since have already
            // advanced one component in the above.
            if ((n + 1) % nhomo == 0)
            {
                for (unsigned int k = 0; k < outDim; ++k)
                {
                    outvec[k] += (outDim - 1) * compOffset;
                }
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::MultiRegions::detail
