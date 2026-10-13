///////////////////////////////////////////////////////////////////////////////
//
// File: LinAdvDiffReactionBlockOpSerialAVXStdMat.hpp
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
// Description: Serial/AVX StdMat implementation of the linearised
// advection-diffusion-reaction block operator.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file LinAdvDiffReactionBlockOpSerialAVXStdMat.hpp
 * @brief Serial/AVX standard-matrix (StdMat) implementation of the
 * per-block linearised advection-diffusion-reaction operator.
 *
 * @details
 * All elements of a block share the same basis, so one set of dense
 * standard-region matrices -- backward transform, reference-space
 * derivative, inner product with the basis and with its gradient --
 * serves the whole block, and the operator becomes a fixed sequence of
 * small matrix multiplies with one pointwise kernel (ApplyMetricKernel,
 * in LinAdvDiffReactionBlockOpSerialAVXStdMatKernels.hpp) in the middle that
 * carries all of the geometry and all three coefficients.
 *
 * This header provides the definition of
 * detail::LinAdvDiffReactionBlockOpImpl for the Serial and AVX execution
 * spaces with the StdMat implementation tag; the sum-factorised
 * definition lives in LinAdvDiffReactionBlockOpSerialAVXSumFac.hpp. The class
 * template name is shared, and each CMake-generated registration
 * translation unit includes exactly one of these headers.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, execution space and
 * implementation; it should not normally be included by any other
 * translation unit. Other code goes through LinAdvDiffReactionOp.hpp and the
 * operator factory: including this header directly instantiates the
 * whole template set in that translation unit, which is what the
 * generated sources exist to avoid.
 */

#pragma once

#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include <LibUtilities/LinearAlgebra/NekBlas/libXSMMDispatchWrapper.hpp>
#include <MultiRegions/ElmtOps/LinAdvDiffReaction/LinAdvDiffReactionBlockOp.hpp>

#include <MultiRegions/ElmtOps/LinAdvDiffReaction/LinAdvDiffReactionBlockOpSerialAVXStdMatKernels.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief StdMat block implementation of the linearised
 * advection-diffusion-reaction operator for the Serial and AVX execution
 * spaces: dense matrix-multiply stages per group of SIMD lanes --
 * 2 + 2 * #m_dimension of them, so four in one dimension, six in two and
 * eight in three -- around one pointwise metric kernel.
 *
 * @details
 * The constructor fetches the block's four transposed standard-region
 * matrices from the data warehouse, so they are built once and shared
 * between operators, together with the block's Jacobians and derivative
 * factors. Per group of simd_t::width elements v_Apply() then runs, on
 * the group's interleaved panels (lane j of a panel column holds
 * element j's value of that mode or point, so the vectorisation is
 * across the elements of the group):
 * -# backward transform of the coefficients to the quadrature points
 *    (#m_bwdmat), giving \f$u\f$ -- skipped when lambda is zero, in
 *    which case the metric kernel does not read it;
 * -# reference-space derivatives of the coefficients in each of the
 *    #m_dimension directions (#m_derivmat), giving
 *    \f$\partial u/\partial\xi_d\f$;
 * -# ApplyMetricKernel, which turns those into Cartesian derivatives
 *    with the derivative factors, dots them with the advection velocity,
 *    adds \f$\lambda u\f$, multiplies by the Jacobian, and separately
 *    forms the diffusion flux pulled back to the reference directions;
 * -# inner product of the advection-plus-reaction values with the basis
 *    (#m_ipbmat), overwriting the output;
 * -# inner product of the diffusion flux with the basis gradient
 *    (#m_ipdmat), accumulated onto the output (GEMM with beta = 1).
 *
 * On the Serial execution space simd_t is scalar (width 1) and each
 * panel degenerates to a single element's data.
 *
 * @tparam ExecSpace       NektarSpaces::Serial or NektarSpaces::AVX;
 *                         sets simd_t via simd_type_if (scalar for
 *                         Serial, tinysimd::simd for AVX).
 * @tparam Implementation  Implementation tag the including translation
 *                         unit registers (MultiRegions::StdMat here).
 * @tparam TData           Floating-point type of the field data.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class LinAdvDiffReactionBlockOpImpl : public LinAdvDiffReactionBlockOp<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Capture the element metadata and fetch the four transposed
     * standard-region matrices, the Jacobians and the derivative factors
     * of the block from the data warehouse.
     *
     * @param   block_idx       Index of the block within the expansion
     *                          list's Collections.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    LinAdvDiffReactionBlockOpImpl(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : LinAdvDiffReactionBlockOp<TData>(block_idx, exp, dataWarehouse)
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

    /// Registration name for the block-operator factory, defined by the
    /// generated registration unit.
    static std::string className;

    /// @brief Creator function registered with the block-operator
    /// factory; builds one block operator for the given block of
    /// elements.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Coeff, FieldState::Coeff, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            LinAdvDiffReactionBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    /// Interleave width the GEMMs operate at: the SIMD vector width
    /// (1 on Serial).
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

    /// Shape shared by every element of the block.
    LibUtilities::ShapeType m_shapeType;
    /// Whether the block's geometry is deformed, i.e. its Jacobian and
    /// derivative factors vary from quadrature point to quadrature
    /// point; selects the ApplyMetricKernel branch and the stride the
    /// geometric data is walked with.
    bool m_isDeformed;
    /// Dimension of the reference element (1, 2 or 3).
    unsigned int m_dimension;
    /// Coordinate dimension of the (possibly embedded) element; the
    /// number of advection-velocity components and the rank of the
    /// diffusivity tensor.
    unsigned int m_coordDim;
    /// Coefficients per element.
    unsigned int m_nmTot;
    /// Quadrature points per element.
    unsigned int m_nqTot;
    /// Transposed backward-transform matrix (nmTot x nqTot,
    /// column-major), cached in the data warehouse.
    const TData *m_bwdmat;
    /// Transposed inner-product-with-basis matrix (nqTot x nmTot).
    const TData *m_ipbmat;
    /// Transposed reference-derivative matrices, one nmTot x nqTot
    /// block per reference direction.
    const TData *m_derivmat;
    /// Transposed inner-product-with-basis-gradient matrices, one
    /// nqTot x nmTot block per reference direction.
    const TData *m_ipdmat;
    /// Jacobians of the block's elements, interleaved at
    /// #m_implInterleaveWidth: one value per element when regular, one
    /// per quadrature point when deformed.
    const TData *m_jacptr;
    /// Derivative factors \f$\partial\xi_d/\partial x_k\f$ of the
    /// block's elements, interleaved at #m_implInterleaveWidth in
    /// point-major order (DerivFactorKey with transpose = false):
    /// element/point stride coordDim * dimension, within which the
    /// entry for coordinate k and reference direction d is at
    /// k * dimension + d.
    const TData *m_dfptr;

    /**
     * @brief Apply the operator to every element of the block by dense
     * matrix multiplication.
     *
     * The advection velocity is first reshaped in place to
     * #m_implInterleaveWidth if it is still held at another width. The
     * field itself may be interleaved at a width other than
     * simd_t::width, so work proceeds in chunks of
     * `chunkSize = max(simd width, field width)` elements per
     * component: each chunk of the input is reshaped in place to
     * simd_t::width interleaving, its `width_ratio` lane groups are run
     * through the five-step sequence described in the class notes, and
     * the input and output chunks are reshaped back to the field width
     * once the chunk's last group is done. The two quadrature-space
     * workspaces -- one nqTot panel for the backward transform,
     * #m_dimension panels for the derivatives -- come from the static
     * workspace shared by all block operators. The Jacobian and
     * derivative-factor pointers advance per element group, by a whole
     * element's worth of data when the geometry is deformed and by one
     * point's worth when it is regular; the velocity pointer
     * starts every component at its plane, `n % nhomo`, so the variables
     * share the velocity of their plane. On
     * return the output block's interleave width is set to the input's.
     *
     * @param   inblock     Coefficient-space input block.
     * @param   outblock    Coefficient-space output block, overwritten.
     */
    void v_Apply(LibUtilities::BlockAccessor<TData, FieldState::Coeff> &inblock,
                 LibUtilities::BlockAccessor<TData, FieldState::Coeff>
                     &outblock) override
    {
        // Reshape advection velocity, if necessary.
        if (this->m_advVel->GetInterleaveWidth() != m_implInterleaveWidth)
        {
            auto advVelPtr =
                this->m_advVel->template GetPtr<MemSpace, ReadWrite>();
            LibUtilities::ReshapeStorage<ExecSpace>(
                m_implInterleaveWidth, this->m_advVel->GetInterleaveWidth(),
                this->m_advVel->GetNumElementsWithPadding() *
                    this->m_advVel->GetNumComponents() *
                    this->m_advVel->GetNumHomoModes(),
                this->m_advVel->GetNumData(), advVelPtr);
            this->m_advVel->template SetInterleaveWidth<TData>(
                m_implInterleaveWidth);
        }

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
        auto ipd_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
            simd_t::width, m_nmTot, m_nqTot, 1.0, 1.0);

        // Loop over components.
        const auto nhomo = inblock.GetNumHomoModes();
        const auto advelsize =
            m_nqTot * inblock.GetNumElmtGroups(m_implInterleaveWidth) * nhomo;
        const auto derivsize = m_nqTot;
        for (unsigned int n = 0; n < inblock.GetNumComponents() * nhomo; ++n)
        {
            // The advection velocity is shared by the variables but not by
            // the planes: the plane this component sits on is n % nhomo, and
            // advelsize steps from there to the next velocity component.
            auto advVelPtr =
                this->m_advVel->template GetPtr<MemSpace, ReadOnly>() +
                (n % nhomo) * this->m_advVel->CompSize();
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
                // factor and Jacobian and add advection.
                if (m_isDeformed)
                {
                    ApplyMetricKernel<ExecSpace, true>(
                        m_nqTot, m_coordDim, m_dimension, 1, derivsize,
                        derivsize, advelsize, diffCoeffPtr,
                        reinterpret_cast<const simd_t *>(jacptr),
                        reinterpret_cast<const simd_t *>(dfptr),
                        reinterpret_cast<const simd_t *>(advVelPtr),
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
                        derivsize, advelsize, diffCoeffPtr,
                        reinterpret_cast<const simd_t *>(jacptr),
                        reinterpret_cast<const simd_t *>(dfptr),
                        reinterpret_cast<const simd_t *>(advVelPtr),
                        reinterpret_cast<const simd_t *>(derivptr),
                        reinterpret_cast<simd_t *>(derivptr),
                        reinterpret_cast<simd_t *>(bwdptr), this->m_lambda);
                    jacptr += simd_t::width;
                    dfptr += m_coordDim * m_dimension * simd_t::width;
                }

                // Step 4: IProduct
                // Perform matrix-matrix multiply.
                ipb_kernel(bwdptr, m_ipbmat, outptr);

                // Step 5: IProductWRTDerivBase
                // Perform matrix-matrix multiply.
                for (unsigned int d = 0; d < m_dimension; d++)
                {
                    ipd_kernel(derivptr + d * simd_t::width * m_nqTot,
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
                advVelPtr += m_nqTot * simd_t::width;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::MultiRegions::detail
