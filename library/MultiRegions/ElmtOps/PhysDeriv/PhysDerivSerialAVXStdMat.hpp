///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivSerialAVXStdMat.hpp
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
// Description: Serial/AVX standard-matrix implementation of the
// per-block physical derivative.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file PhysDerivSerialAVXStdMat.hpp
 * @brief Serial/AVX standard-matrix (StdMat) implementation of the
 * per-block physical derivative.
 *
 * @details
 * All elements of a block share the same basis and quadrature, so the
 * `dimension` standard-region derivative matrices serve the whole
 * block; only the geometric derivative factors are per element. This
 * header defines the detail::PhysDerivBlockOpImpl used when the
 * "StdMat" implementation is requested on the Serial or AVX execution
 * space; CMake includes it into the translation units generated for
 * exactly those factory registrations, which is why this header, the
 * SumFac header and the device headers may each define the same
 * primary template without ever colliding.
 *
 * @see PhysDerivSerialAVXSumFac.hpp for the sum-factorised alternative
 * and PhysDerivDeviceStdMat.hpp for the device StdMat path.
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
#include <LibUtilities/LinearAlgebra/NekBlas/libXSMMDispatchWrapper.hpp>
#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivBlockOp.hpp>

#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivSerialAVXStdMatKernels.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief StdMat block implementation of the physical derivative for the
 * Serial and AVX execution spaces: one small dense matrix multiply per
 * standard direction and group of SIMD lanes, then the chain rule.
 *
 * @details
 * The constructor fetches the block's transposed standard-region
 * derivative matrices (StdMatKey with ePhysDerivStdMatTranspose:
 * `dimension` consecutive nqTot x nqTot column-major arrays, one per
 * standard direction) and the block's derivative factors
 * (DerivFactorKey, untransposed, at this implementation's interleave
 * width) from the data warehouse, so both are built once and shared
 * between operators. Applying the operator is then, per group of
 * simd_t::width elements:
 * -# one libxsmm GEMM per standard direction @em d,
 *    `out-panel_d (width x nqTot) = in-panel (width x nqTot) * D_d^T`,
 *    the panels being the group's interleaved storage (lane j of a
 *    panel column holds element j's value at that point), so the
 *    vectorisation is across the elements of the group;
 * -# MultiplyByDerivFactorKernel, which contracts those `dimension`
 *    standard derivatives with the derivative factors into the
 *    `coordDim` Cartesian derivatives, in place in the output.
 *
 * Because the matrices come from StdExpansion::PhysDeriv applied to the
 * unit vectors, any collapsed-coordinate chain rule of the simplex
 * shapes is already contained in them; unlike the sum-factorised
 * implementations this path therefore needs no separate collapsed-
 * coordinate factors, only the geometric ones. On the Serial execution
 * space simd_t is scalar (width 1) and each panel degenerates to a
 * single element.
 *
 * @tparam ExecSpace       NektarSpaces::Serial or NektarSpaces::AVX;
 *                         sets simd_t via simd_type_if
 *                         (tinysimd::scalarT for Serial, tinysimd::simd
 *                         for AVX).
 * @tparam Implementation  Implementation tag the including translation
 *                         unit registers (MultiRegions::StdMat here).
 * @tparam TData           Floating-point type of the field data.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class PhysDerivBlockOpImpl : public PhysDerivBlockOp<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Fetch the stacked transposed derivative matrices
     * (ePhysDerivStdMatTranspose, one nqTot x nqTot block per standard
     * direction, keyed on the block's basis keys and shape) and the
     * block's derivative factors interleaved at the SIMD width.
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
        m_nmTot     = exp->GetNcoeffs();
        m_nqTot     = exp->GetTotPoints();

        // Fetch matrix.
        std::vector<LibUtilities::BasisKey> basisKeys(
            m_dimension, LibUtilities::NullBasisKey);
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            basisKeys[d] = exp->GetBasis(d)->GetBasisKey();
        }

        m_matptr = dataWarehouse->template GetData<MemSpace>(
            StdRegions::StdMatKey<TData>(
                basisKeys, m_shapeType, StdRegions::ePhysDerivStdMatTranspose));

        // Fetch derivative factor.
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
    /// Interleave width the GEMMs and the chain-rule kernel operate at:
    /// the SIMD vector width (1 on Serial). Also the width the
    /// derivative factors are fetched at.
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

    /// Shape shared by every element of the block.
    LibUtilities::ShapeType m_shapeType;
    /// Whether the block's geometry is deformed: selects the DEFORMED
    /// instantiation of MultiplyByDerivFactorKernel, i.e. one set of
    /// derivative factors per quadrature point rather than per element.
    bool m_isDeformed;
    /// Dimension of the reference element (1, 2 or 3); also the number
    /// of standard-region derivative matrices and the number of
    /// standard derivatives contracted per output component.
    unsigned int m_dimension;
    /// Coordinate dimension of the (possibly embedded) element; the
    /// number of Cartesian derivative components produced.
    unsigned int m_coordDim;
    /// Coefficients per element; unused here (this operator never
    /// enters coefficient space).
    unsigned int m_nmTot;
    /// Quadrature points per element.
    unsigned int m_nqTot;
    /// The `dimension` transposed standard-region derivative matrices
    /// (each nqTot x nqTot, column-major, stored back to back), cached
    /// in the data warehouse.
    const TData *m_matptr;
    /// Derivative factors of the block, interleaved at
    /// #m_implInterleaveWidth; `coordDim * dimension` values per
    /// element, or per quadrature point when the geometry is deformed.
    const TData *m_dfptr;

    /**
     * @brief Differentiate every element of the block by dense matrix
     * multiplication followed by the chain rule.
     *
     * The field may be interleaved at a width other than
     * simd_t::width, so work proceeds in chunks of
     * `chunkSize = max(simd width, input width)` elements per
     * component: each chunk is reshaped in place to simd_t::width
     * interleaving, its `width_ratio` lane groups are processed, and
     * the chunk is reshaped back once its last group is done -- the
     * input plane and all `outDim` output planes alike (`outDim` is
     * `coordDim`, or three for a 3DH1 field with more than one
     * homogeneous plane).
     *
     * Per lane group the `dimension` GEMMs write the standard-direction
     * derivatives into the group's slice of output planes 0 ..
     * dimension-1, and MultiplyByDerivFactorKernel then overwrites
     * planes 0 .. coordDim-1 with their contraction against the
     * derivative factors (with `nelmt = 1`, one lane group at a time,
     * and the plane stride given in SIMD vectors). The factor pointer
     * advances by one element group's worth of factors per group and is
     * reset to the start of the block for each component and
     * homogeneous mode, the same geometry serving every one of them.
     *
     * The output pointer advances by one plane -- one homogeneous mode
     * of one component -- per iteration of the component loop and,
     * once a component's homogeneous modes are exhausted, skips the
     * remaining `outDim - 1` output components that input component
     * owns (leaving slot 2 free for the z-derivative in the 3DH1 case;
     * see PhysDerivOp). On return the output block's interleave width
     * is set to the input's.
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
        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();
        const auto width_ratio     = (interleaveWidth == 1)
                                         ? 1
                                         : interleaveWidth / m_implInterleaveWidth;
        const auto chunkSize = std::max(m_implInterleaveWidth, interleaveWidth);

        // Dispatch kernel.
        auto gemm_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
            simd_t::width, m_nqTot, m_nqTot, 1.0, 0.0);

        // Offsets between the components of a block.
        const auto outDim = (outblock.GetNumHomoModes() > 1) ? 3u : m_coordDim;
        const auto outoffset = outblock.CompSize() * outblock.GetNumHomoModes();
        const auto outoffset_vec = outoffset / simd_t::width;

        // Loop over components.
        for (unsigned int n = 0;
             n < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++n)
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
                        m_nqTot, (TData *)inptr);
                }

                // Perform matrix-matrix multiply.
                for (unsigned int d = 0; d < m_dimension; d++)
                {
                    gemm_kernel(inptr, m_matptr + d * m_nqTot * m_nqTot,
                                outptr + d * outoffset);
                }

                // Multiply by derivative factor.
                if (m_isDeformed)
                {
                    MultiplyByDerivFactorKernel<ExecSpace, true>(
                        m_nqTot, m_coordDim, m_dimension, 1, outoffset_vec,
                        outoffset_vec, reinterpret_cast<const simd_t *>(dfptr),
                        reinterpret_cast<const simd_t *>(outptr),
                        reinterpret_cast<simd_t *>(outptr));
                    dfptr += m_coordDim * m_dimension * m_nqTot * simd_t::width;
                }
                else
                {
                    MultiplyByDerivFactorKernel<ExecSpace, false>(
                        m_nqTot, m_coordDim, m_dimension, 1, outoffset_vec,
                        outoffset_vec, reinterpret_cast<const simd_t *>(dfptr),
                        reinterpret_cast<const simd_t *>(outptr),
                        reinterpret_cast<simd_t *>(outptr));
                    dfptr += m_coordDim * m_dimension * simd_t::width;
                }

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        m_nqTot,
                        (TData *)inptr -
                            (width_ratio - 1) * m_nqTot * simd_t::width);
                    for (unsigned int k = 0; k < outDim; k++)
                    {
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            interleaveWidth, m_implInterleaveWidth, chunkSize,
                            m_nqTot,
                            (TData *)outptr + k * outoffset -
                                (width_ratio - 1) * m_nqTot * simd_t::width);
                    }
                }

                // Increment pointers.
                inptr += m_nqTot * simd_t::width;
                outptr += m_nqTot * simd_t::width;
            }

            if ((n + 1) % outblock.GetNumHomoModes() == 0)
            {
                outptr += (outDim - 1) * outoffset;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::MultiRegions::detail
