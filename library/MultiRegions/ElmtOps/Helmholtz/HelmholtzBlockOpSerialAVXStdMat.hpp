///////////////////////////////////////////////////////////////////////////////
//
// File: HelmholtzBlockOpSerialAVXStdMat.hpp
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
 * @file HelmholtzBlockOpSerialAVXStdMat.hpp
 * @brief Serial/AVX standard-matrix (StdMat) implementation of the
 * per-block weak Helmholtz operator.
 *
 * @details
 * This header provides the definition of detail::HelmholtzBlockOpImpl
 * for the Serial and AVX execution spaces with the StdMat
 * implementation tag; see HelmholtzBlockOp.hpp for the interface it
 * implements and HelmholtzOp.hpp for the registration scheme.
 *
 * The operator is decomposed into five stages against standard-region
 * matrices:
 * 1. BwdTrans on the input coefficients to recover physical values
 *    (skipped when lambda is zero);
 * 2. Deriv on the input coefficients to form the reference derivatives
 *    \f$\partial u/\partial \xi_d\f$;
 * 3. ApplyMetricKernel to contract the reference derivatives with the
 *    geometry and diffusion tensor (forming \f$J \mathbf{G} \nabla_\xi
 *    u\f$) and to scale the physical values by `lambda * J`;
 * 4. IProductWRTBase on the scaled physical values to form the mass
 *    term (skipped when lambda is zero);
 * 5. IProductWRTDerivBase on the contracted reference derivatives to
 *    form the weak Laplacian term and accumulate it into the output.
 *
 * The standard-region matrices are fetched from the data warehouse and
 * transposed so that the inner loop can be issued through
 * LibxsmmDispatchWrapper, which evaluates the matrix products across
 * the simd_t::width SIMD lanes of each element group.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, execution space and
 * implementation; it should not normally be included by any other
 * translation unit. Other code goes through HelmholtzOp.hpp and the
 * operator factory: including this header directly instantiates the
 * whole template set in that translation unit, which is what the
 * generated sources exist to avoid.
 */

#pragma once

#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include <LibUtilities/LinearAlgebra/NekBlas/libXSMMDispatchWrapper.hpp>
#include <MultiRegions/ElmtOps/Helmholtz/HelmholtzBlockOp.hpp>

#include <MultiRegions/ElmtOps/Helmholtz/HelmholtzBlockOpSerialAVXStdMatKernels.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Serial/AVX StdMat block implementation of the weak Helmholtz
 * operator: applies the five-stage matrix-matrix decomposition to one
 * group of elements at a time, one element per SIMD lane.
 *
 * @tparam ExecSpace      NektarSpaces::Serial or NektarSpaces::AVX;
 *                        selects scalar or SIMD arithmetic through
 *                        simd_t.
 * @tparam Implementation Implementation tag (StdMat).
 * @tparam TData          Floating-point type of the field data.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class HelmholtzBlockOpImpl : public HelmholtzBlockOp<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Construct the operator, capture element metadata, and
     * fetch the standard-region matrices, Jacobians and derivative
     * factors from the data warehouse.
     *
     * @param   block_idx       Index of the block within the expansion
     *                          list's Collections.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    HelmholtzBlockOpImpl(const unsigned int block_idx,
                         const LocalRegions::ExpansionSharedPtr &exp,
                         LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : HelmholtzBlockOp<TData>(block_idx, exp, dataWarehouse)
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

        LibUtilities::PointsType nodalType =
            (exp->IsNodalNonTensorialExp())
                ? exp->GetNodalPointsKey().GetPointsType()
                : LibUtilities::eNoPointsType;

        m_bwdmat = dataWarehouse->template GetData<MemSpace>(
            StdRegions::StdMatKey<TData>(basisKeys, m_shapeType,
                                         StdRegions::eBwdTransStdMatTranspose,
                                         nodalType));
        m_ipbmat = dataWarehouse->template GetData<MemSpace>(
            StdRegions::StdMatKey<TData>(
                basisKeys, m_shapeType,
                StdRegions::eIProductWRTBaseStdMatTranspose, nodalType));
        m_derivmat = dataWarehouse->template GetData<MemSpace>(
            StdRegions::StdMatKey<TData>(basisKeys, m_shapeType,
                                         StdRegions::eDerivStdMatTranspose,
                                         nodalType));
        m_ipdmat = dataWarehouse->template GetData<MemSpace>(
            StdRegions::StdMatKey<TData>(
                basisKeys, m_shapeType,
                StdRegions::eIProductWRTDerivBaseStdMatTranspose, nodalType));

        // Fetch Jacobian and deriv factors.
        m_jacptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::JacobianKey<TData>(block_idx, m_implInterleaveWidth));
        m_dfptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::DerivFactorKey<TData>(block_idx,
                                                m_implInterleaveWidth, false));
    }

    /// Block-operator class name; defined by generated factory code.
    static std::string className;

    /// Creator function registered with BlockOperatorFactory.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Coeff, FieldState::Coeff, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            HelmholtzBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    /// Vector width this implementation operates at (simd_t::width).
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

    /// Shape type of the element block.
    LibUtilities::ShapeType m_shapeType;
    /// Whether the block's geometry is deformed; selects the branch of
    /// ApplyMetricKernel and the geometry layout it reads.
    bool m_isDeformed;
    /// Dimension of the reference element (1, 2 or 3).
    unsigned int m_dimension;
    /// Coordinate dimension of the (possibly embedded) element.
    unsigned int m_coordDim;
    /// Coefficients per element.
    unsigned int m_nmTot;
    /// Quadrature points per element.
    unsigned int m_nqTot;
    /// Transposed backward-transform matrix (nmTot x nqTot).
    const TData *m_bwdmat;
    /// Transposed inner-product-with-basis matrix (nqTot x nmTot),
    /// quadrature weights included.
    const TData *m_ipbmat;
    /// Transposed coefficients-to-reference-derivative matrices, one
    /// nmTot x nqTot block per reference direction.
    const TData *m_derivmat;
    /// Transposed inner-product-with-basis-derivative matrices, one
    /// nqTot x nmTot block per reference direction, weights included.
    const TData *m_ipdmat;
    /// Jacobian of the block, interleaved at #m_implInterleaveWidth:
    /// one value per element, or one per quadrature point and element
    /// on a deformed block.
    const TData *m_jacptr;
    /// Derivative factors of the block, interleaved and untransposed:
    /// `coordDim * dimension` values per element, or per quadrature
    /// point and element on a deformed block.
    const TData *m_dfptr;

    /**
     * @brief Apply the weak Helmholtz operator to every element of the
     * block by dense matrix multiplication.
     *
     * The five steps are those listed in the class notes. Their
     * intermediates live in the block operators' shared static
     * workspace, sized here for one group of elements: `width * nqTot`
     * values for the physical values followed by
     * `dimension * width * nqTot` for the reference derivatives.
     *
     * The field may be interleaved at a width other than
     * simd_t::width: a field is interleaved at the vector width of the
     * execution space it was prepared for, which need not equal
     * simd_t::width here. Work therefore proceeds in chunks of
     * `chunkSize = max(simd width, input width)` elements per
     * component: each chunk is reshaped in place to simd_t::width
     * interleaving, its `width_ratio` lane groups are put through the
     * five steps, and the chunk is reshaped back once its last group is
     * done. The geometry pointers restart at the beginning of the block
     * for every component and homogeneous mode, since the geometry is
     * shared by all of them. On return the output block's interleave
     * width is set to the input's.
     *
     * @param   inblock     Coefficient-space input block.
     * @param   outblock    Coefficient-space output block;
     *                      overwritten.
     */
    void v_Apply(LibUtilities::BlockAccessor<TData, FieldState::Coeff> &inblock,
                 LibUtilities::BlockAccessor<TData, FieldState::Coeff>
                     &outblock) override
    {
        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto diffCoeffPtr =
            this->m_diffCoeff.template GetPtr<MemSpace, ReadOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();
        const auto width_ratio     = (interleaveWidth == 1)
                                         ? 1
                                         : interleaveWidth / m_implInterleaveWidth;
        const auto chunkSize = std::max(m_implInterleaveWidth, interleaveWidth);

        // Get static workspace pointer.
        auto bwdptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                simd_t::width * m_nqTot +
                m_dimension * simd_t::width * m_nqTot);
        auto derivptr = bwdptr + simd_t::width * m_nqTot;

        // Dispatch kernel.
        auto bwd_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
            simd_t::width, m_nqTot, m_nmTot, 1.0, 0.0);
        auto ipb_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
            simd_t::width, m_nmTot, m_nqTot, 1.0, 0.0);
        auto deriv_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
            simd_t::width, m_nqTot, m_nmTot, 1.0, 0.0);
        auto ipd_kernel1 = LibxsmmDispatchWrapper<TData>::dispatch(
            simd_t::width, m_nmTot, m_nqTot, 1.0, TData(this->m_lambda != 0.0));
        auto ipd_kernel2 = LibxsmmDispatchWrapper<TData>::dispatch(
            simd_t::width, m_nmTot, m_nqTot, 1.0, 1.0);

        // Loop over components.
        const auto derivsize = m_nqTot;
        for (unsigned int n = 0;
             n < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++n)
        {
            auto jacptr = m_jacptr;
            auto dfptr  = m_dfptr;

            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        m_implInterleaveWidth, interleaveWidth, chunkSize,
                        m_nmTot, (TData *)inptr);
                }

                // Step 1: BwdTrans
                // Perform matrix-matrix multiply.
                if (this->m_lambda != 0.0)
                {
                    bwd_kernel(inptr, m_bwdmat, bwdptr);
                }

                // Step 2: Deriv
                // Perform matrix-matrix multiply.
                for (unsigned int d = 0; d < m_dimension; d++)
                {
                    deriv_kernel(inptr, m_derivmat + d * m_nqTot * m_nmTot,
                                 derivptr + d * m_nqTot * simd_t::width);
                }

                // Step 3: Multiply by diffusion coefficient, derivative
                // factor and Jacobian.
                if (m_isDeformed)
                {
                    ApplyMetricKernel<ExecSpace, true>(
                        m_nqTot, m_coordDim, m_dimension, 1, derivsize,
                        derivsize, diffCoeffPtr,
                        reinterpret_cast<const simd_t *>(jacptr),
                        reinterpret_cast<const simd_t *>(dfptr),
                        reinterpret_cast<const simd_t *>(derivptr),
                        reinterpret_cast<simd_t *>(derivptr),
                        reinterpret_cast<simd_t *>(bwdptr), this->m_lambda);
                    jacptr += m_nqTot * simd_t::width;
                    dfptr += m_coordDim * m_dimension * m_nqTot * simd_t::width;
                }
                else
                {
                    ApplyMetricKernel<ExecSpace, false>(
                        m_nqTot, m_coordDim, m_dimension, 1, derivsize,
                        derivsize, diffCoeffPtr,
                        reinterpret_cast<const simd_t *>(jacptr),
                        reinterpret_cast<const simd_t *>(dfptr),
                        reinterpret_cast<const simd_t *>(derivptr),
                        reinterpret_cast<simd_t *>(derivptr),
                        reinterpret_cast<simd_t *>(bwdptr), this->m_lambda);
                    jacptr += simd_t::width;
                    dfptr += m_coordDim * m_dimension * simd_t::width;
                }

                // Step 4: IProduct
                // Perform matrix-matrix multiply.
                if (this->m_lambda != 0.0)
                {
                    ipb_kernel(bwdptr, m_ipbmat, outptr);
                }

                // Step 5: IProductWRTDerivBase
                // Perform matrix-matrix multiply.
                ipd_kernel1(derivptr, m_ipdmat, outptr);
                for (unsigned int d = 1; d < m_dimension; d++)
                {
                    ipd_kernel2(derivptr + d * simd_t::width * m_nqTot,
                                m_ipdmat + d * m_nqTot * m_nmTot, outptr);
                }

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        m_nmTot,
                        (TData *)inptr -
                            (width_ratio - 1) * m_nmTot * simd_t::width);
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        m_nmTot,
                        (TData *)outptr -
                            (width_ratio - 1) * m_nmTot * simd_t::width);
                }

                // Increment pointers.
                inptr += m_nmTot * simd_t::width;
                outptr += m_nmTot * simd_t::width;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::MultiRegions::detail
