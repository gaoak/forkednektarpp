///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionSerialAVXStdMat.hpp
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
// Description: Serial/AVX StdMat implementation of the advection block
// operator.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file AdvectionSerialAVXStdMat.hpp
 * @brief Serial/AVX StdMat implementation of the advection block operator.
 *
 * @details
 * The StdMat strategy evaluates the reference-space derivatives of the
 * input by multiplying with the dense standard-region physical-derivative
 * matrices, one libxsmm GEMM per direction and element group; a pointwise
 * kernel then applies the geometric derivative factors and the dot product
 * with the advection velocity (see AdvectionSerialAVXStdMatKernels.hpp).
 * On the AVX execution space each SIMD lane owns one element of a group,
 * so the GEMM's left operand is the lane-interleaved batch of element
 * data; on Serial the SIMD type is the width-one tinysimd::scalarT
 * wrapper and the group size is one.
 *
 * This header provides the definition of detail::AdvectionBlockOpImpl for
 * the Serial and AVX execution spaces with the StdMat implementation tag.
 * The sum-factorised Serial/AVX definition lives in
 * AdvectionSerialAVXSumFac.hpp and the Device definitions in
 * AdvectionDeviceStdMat.hpp and AdvectionDeviceSumFac.hpp; the class
 * template name is shared, and each CMake-generated registration
 * translation unit includes exactly one of these headers.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, execution space and
 * implementation; it should not normally be included by any other
 * translation unit. Other code goes through AdvectionOp.hpp and the
 * operator factory: including this header directly instantiates the
 * whole template set in that translation unit, which is what the
 * generated sources exist to avoid.
 */

#pragma once

#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include <LibUtilities/LinearAlgebra/NekBlas/libXSMMDispatchWrapper.hpp>
#include <MultiRegions/ElmtOps/Advection/AdvectionBlockOp.hpp>

#include <MultiRegions/ElmtOps/Advection/AdvectionSerialAVXStdMatKernels.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Serial/AVX StdMat advection block operator: dense standard-region
 * derivative matrices followed by a pointwise derivative-factor and
 * velocity kernel.
 *
 * @details
 * One v_Apply() pass loops over the components and homogeneous modes of
 * the input block and works through each in groups of simd_t::width
 * elements. For each group it
 * -# reshapes the group's storage to the implementation interleave width
 *    when the field is stored at a different width,
 * -# forms the reference-space derivatives of all lanes with one GEMM per
 *    reference direction (#m_dimension of them) against the transposed
 *    standard-region PhysDeriv matrices -- fetched once from the data
 *    warehouse under ePhysDerivStdMatTranspose, one matrix per reference
 *    direction, and shared between operators with equal basis --
 *    collecting them in the static workspace, and
 * -# calls MultiplyByDerivFactorAndAdvecVelKernel to chain-rule the
 *    reference derivatives into Cartesian ones -- through the per-element
 *    or, for deformed geometry, per-point derivative factors -- and
 *    contract them with the advection velocity.
 *
 * The velocity registered by SetAdvVel() is reshaped to the implementation
 * width the first time v_Apply() finds it at another width, and is then
 * read in place; each homogeneous plane reads its own plane of the
 * velocity, and the same velocity serves every component.
 *
 * @tparam ExecSpace      NektarSpaces::Serial or NektarSpaces::AVX; selects
 *                        scalar or SIMD arithmetic through simd_t.
 * @tparam Implementation Implementation tag (StdMat).
 * @tparam TData          Floating-point type of the field data.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class AdvectionBlockOpImpl : public AdvectionBlockOp<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Capture the block's element metadata and fetch the
     * transposed standard-region derivative matrices and the block's
     * derivative factors from the data warehouse.
     *
     * @param   block_idx       Index of the block.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    AdvectionBlockOpImpl(const unsigned int block_idx,
                         const LocalRegions::ExpansionSharedPtr &exp,
                         LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : AdvectionBlockOp<TData>(block_idx, exp, dataWarehouse)
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

    /// Registration name for the block-operator factory, defined by the
    /// generated registration unit.
    static std::string className;

    /// @brief Creator function registered with the block-operator
    /// factory; builds one block operator for the given block of
    /// elements.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            AdvectionBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    /// Element interleave width the kernels expect: the SIMD width (1 on
    /// the Serial execution space).
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

    /// Shape of the block's elements.
    LibUtilities::ShapeType m_shapeType;
    /// Whether the geometry is deformed (per-point derivative factors).
    bool m_isDeformed;
    /// Reference (shape) dimension of the elements.
    unsigned int m_dimension;
    /// Coordinate dimension of the (possibly embedded) elements.
    unsigned int m_coordDim;
    /// Coefficients per element.
    unsigned int m_nmTot;
    /// Quadrature points per element.
    unsigned int m_nqTot;
    /// Transposed standard-region PhysDeriv matrices, one nqTot x nqTot
    /// matrix per reference direction, concatenated.
    const TData *m_matptr;
    /// Derivative factors of the whole block, interleaved at the SIMD
    /// width and fetched untransposed: a deformed block stores the
    /// coordDim * dimension factors of a point together, point by point,
    /// while a regular block stores a single set per element group and
    /// has no point index at all (see DerivFactorKey).
    const TData *m_dfptr;

    /**
     * @brief Apply scale * (V . grad u) to every element of the block (see
     * the class description for the algorithm).
     *
     * Fields stored at an interleave width larger than the implementation
     * width are handled by reshaping width_ratio consecutive groups in
     * place. The two sides are not treated symmetrically: the input is
     * converted to the implementation width before the GEMMs and
     * converted back once the groups have been processed, whereas the
     * output is converted forward only when it is accumulated onto
     * (append set) -- there is nothing to preserve otherwise -- and is in
     * both cases converted from the implementation width to the input's
     * width once the kernel has written it. The output block's recorded
     * interleave width is set to the input's on return.
     *
     * @param   inblock     Physical-space input block.
     * @param   outblock    Physical-space output block; overwritten, or
     *                      accumulated onto when append is set.
     */
    void v_Apply(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock) override
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

        // Get static workspace pointer.
        auto derivptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                m_dimension * simd_t::width * m_nqTot);

        // Dispatch kernel.
        auto gemm_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
            simd_t::width, m_nqTot, m_nqTot, 1.0, 0.0);

        // Loop over components. The advection velocity carries the same
        // planes as the input, so one of its components spans all of them.
        const auto nhomo = inblock.GetNumHomoModes();
        const auto advelsize =
            m_nqTot * inblock.GetNumElmtGroups(m_implInterleaveWidth) * nhomo;
        const auto derivsize = m_nqTot;
        for (unsigned int n = 0; n < inblock.GetNumComponents() * nhomo; ++n)
        {
            // n runs over the components of the input with their planes
            // innermost, so this iteration takes the velocity of plane
            // n % nhomo, and advelsize steps from there to the next velocity
            // component on the same plane.
            auto advVelPtr =
                this->m_advVel->template GetPtr<MemSpace, ReadOnly>() +
                (n % nhomo) * this->m_advVel->CompSize();
            auto dfptr = m_dfptr;

            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        m_implInterleaveWidth, inInterleaveWidth, chunkSize,
                        m_nqTot, (TData *)inptr);
                }

                // Perform matrix-matrix multiply.
                for (unsigned int d = 0; d < m_dimension; d++)
                {
                    gemm_kernel(inptr, m_matptr + d * m_nqTot * m_nqTot,
                                derivptr + d * m_nqTot * simd_t::width);
                }

                // Multiply by derivative factor and Advection Velocity.
                if (m_isDeformed)
                {
                    if (this->m_append)
                    {
                        // Reshape, if necessary.
                        if (e % width_ratio == 0)
                        {
                            LibUtilities::ReshapeStorage<ExecSpace>(
                                m_implInterleaveWidth, outInterleaveWidth,
                                chunkSize, m_nqTot, (TData *)outptr);
                        }

                        MultiplyByDerivFactorAndAdvecVelKernel<ExecSpace, true,
                                                               true>(
                            m_nqTot, m_coordDim, m_dimension, 1, derivsize,
                            reinterpret_cast<const simd_t *>(dfptr),
                            reinterpret_cast<const simd_t *>(advVelPtr),
                            advelsize,
                            reinterpret_cast<const simd_t *>(derivptr),
                            reinterpret_cast<simd_t *>(outptr), this->m_scale);
                    }
                    else
                    {
                        MultiplyByDerivFactorAndAdvecVelKernel<ExecSpace, false,
                                                               true>(
                            m_nqTot, m_coordDim, m_dimension, 1, derivsize,
                            reinterpret_cast<const simd_t *>(dfptr),
                            reinterpret_cast<const simd_t *>(advVelPtr),
                            advelsize,
                            reinterpret_cast<const simd_t *>(derivptr),
                            reinterpret_cast<simd_t *>(outptr), this->m_scale);
                    }
                    dfptr += m_coordDim * m_dimension * m_nqTot * simd_t::width;
                }
                else
                {
                    if (this->m_append)
                    {
                        // Reshape, if necessary.
                        if (e % width_ratio == 0)
                        {
                            LibUtilities::ReshapeStorage<ExecSpace>(
                                m_implInterleaveWidth, outInterleaveWidth,
                                chunkSize, m_nqTot, (TData *)outptr);
                        }

                        MultiplyByDerivFactorAndAdvecVelKernel<ExecSpace, true,
                                                               false>(
                            m_nqTot, m_coordDim, m_dimension, 1, derivsize,
                            reinterpret_cast<const simd_t *>(dfptr),
                            reinterpret_cast<const simd_t *>(advVelPtr),
                            advelsize,
                            reinterpret_cast<const simd_t *>(derivptr),
                            reinterpret_cast<simd_t *>(outptr), this->m_scale);
                    }
                    else
                    {
                        MultiplyByDerivFactorAndAdvecVelKernel<ExecSpace, false,
                                                               false>(
                            m_nqTot, m_coordDim, m_dimension, 1, derivsize,
                            reinterpret_cast<const simd_t *>(dfptr),
                            reinterpret_cast<const simd_t *>(advVelPtr),
                            advelsize,
                            reinterpret_cast<const simd_t *>(derivptr),
                            reinterpret_cast<simd_t *>(outptr), this->m_scale);
                    }
                    dfptr += m_coordDim * m_dimension * simd_t::width;
                }

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                        m_nqTot,
                        (TData *)inptr -
                            (width_ratio - 1) * m_nqTot * simd_t::width);
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                        m_nqTot,
                        (TData *)outptr -
                            (width_ratio - 1) * m_nqTot * simd_t::width);
                }

                // Increment pointers.
                inptr += m_nqTot * simd_t::width;
                outptr += m_nqTot * simd_t::width;
                advVelPtr += m_nqTot * simd_t::width;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
    }
};

} // namespace Nektar::MultiRegions::detail
