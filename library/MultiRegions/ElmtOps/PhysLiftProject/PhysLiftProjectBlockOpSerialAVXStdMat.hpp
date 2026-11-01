///////////////////////////////////////////////////////////////////////////////
//
// File: PhysLiftProjectBlockOpSerialAVXStdMat.hpp
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
// Description: Serial/AVX StdMat implementation of the per-block
// lift-and-project operator.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file PhysLiftProjectBlockOpSerialAVXStdMat.hpp
 * @brief Serial and AVX StdMat block implementation of the lift-and-project
 * operator: the dense projector on affine blocks.
 *
 * @details
 * On an affine block that is not Pointwise (the StdInvMass form) the
 * projection is one dense product per element group with
 * \f$P = B M_{\mathrm{std}}^{-1} B^{\mathsf T}\f$, nqTot x nqTot, fetched
 * from the data warehouse once per shape, basis and rule
 * (StdRegions::ePhysProjectStdMat), after the inner products are scaled by
 * `scale / J`; the libxsmm products are dispatched once in the constructor.
 * The trace is lifted first, onto a scratch copy of the inner products, by
 * the IProductWRTPhysTrace block operator. A block of the Pointwise form,
 * where B is square, is scaled by `scale / (w J)` instead, through the
 * kernels the SumFac implementation shares
 * (PhysLiftProjectBlockOpSerialAVXKernels.hpp); a block of the ElmtInvMass
 * form takes master's four block operators in sequence
 * (PhysLiftProjectBlockOp::ApplyElmtInvMass()).
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, execution space and
 * implementation; other code goes through PhysLiftProjectOp.hpp and the
 * operator factory.
 */

#pragma once

#include <LibUtilities/SimdLib/tinysimd.hpp>

#include <LibUtilities/BasicUtils/SafeDivide.hpp>
#include <LibUtilities/BasicUtils/Utils/UtilsKernels.hpp>
#include <LibUtilities/LinearAlgebra/NekBlas/libXSMMDispatchWrapper.hpp>
#include <MultiRegions/ElmtOps/PhysLiftProject/PhysLiftProjectBlockOp.hpp>
#include <MultiRegions/ElmtOps/PhysLiftProject/PhysLiftProjectBlockOpSerialAVXKernels.hpp>

#include <algorithm>
#include <memory>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Serial and AVX StdMat block implementation of the lift-and-project
 * operator, one element per SIMD lane.
 *
 * @tparam ExecSpace       NektarSpaces::Serial or NektarSpaces::AVX.
 * @tparam Implementation  MultiRegions::StdMat; not read.
 * @tparam TData           Floating-point type of the field data.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class PhysLiftProjectBlockOpImpl : public PhysLiftProjectBlockOp<TData>
{
    using BlockOpBase = PhysLiftProjectBlockOp<TData>;
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Fetch the quadrature weights on a block of the Pointwise form,
     * and the projector on a block of the StdInvMass form; create master's four
     * block operators on a block of the ElmtInvMass form.
     *
     * @param   block_idx       Index of the block.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    PhysLiftProjectBlockOpImpl(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : BlockOpBase(block_idx, exp, dataWarehouse)
    {
        m_form = PhysLiftProjectFormOf(exp);
        if (m_form == PhysLiftProjectForm::ElmtInvMass)
        {
            this->CreateChain(ExecSpace::name, SumFac::name);
        }
        else
        {
            m_isDeformed =
                exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
            m_nqTot = exp->GetTotPoints();
            m_lift  = IProductWRTPhysTraceBlockOp<TData>::Create(
                block_idx, exp, dataWarehouse, ExecSpace::name, StdMat::name);
            m_lift->SetAppend(true);
            m_jacptr = this->m_dataWarehouse->template GetData<MemSpace>(
                LocalRegions::JacobianKey<TData>(block_idx,
                                                 m_implInterleaveWidth));
            m_scaled.resize(m_nqTot);
            if (m_form == PhysLiftProjectForm::Pointwise)
            {
                m_weightsptr =
                    this->m_dataWarehouse->template GetData<MemSpace>(
                        LocalRegions::WeightsKey<TData>(block_idx,
                                                        m_implInterleaveWidth));
            }
            else
            {
                const unsigned int dim = exp->GetShapeDimension();
                std::vector<LibUtilities::BasisKey> basisKeys(
                    dim, LibUtilities::NullBasisKey);
                for (unsigned int d = 0; d < dim; ++d)
                {
                    basisKeys[d] = exp->GetBasis(d)->GetBasisKey();
                }
                LibUtilities::PointsType nodalType =
                    LibUtilities::eNoPointsType;
                if (exp->IsNodalNonTensorialExp())
                {
                    nodalType = exp->GetNodalPointsKey().GetPointsType();
                }
                m_projptr = this->m_dataWarehouse->template GetData<MemSpace>(
                    StdRegions::StdMatKey<TData>(basisKeys, exp->DetShapeType(),
                                                 StdRegions::ePhysProjectStdMat,
                                                 nodalType));
            }
        }
    }

    /// Registration name for BlockOperatorFactory, defined by the
    /// generated factory declaration unit.
    static std::string className;

    /// Creator function registered with BlockOperatorFactory.
    static std::unique_ptr<BlockOperator<TData>> Instantiate(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            PhysLiftProjectBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    /// Elements the kernel processes at once: the SIMD vector width, one on
    /// Serial; also the width the Jacobians are fetched at.
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

    /// The block's form (PhysLiftProjectFormOf()).
    PhysLiftProjectForm m_form;
    bool m_isDeformed    = false;
    unsigned int m_nqTot = 0;
    /// The Jacobians and, Pointwise only, the quadrature weights at every
    /// point, at #m_implInterleaveWidth; StdInvMass only, the projector
    /// \f$B M_{\mathrm{std}}^{-1} B^{\mathsf T}\f$.
    const TData *m_jacptr     = nullptr;
    const TData *m_weightsptr = nullptr;
    const TData *m_projptr    = nullptr;
    /// The lift onto the scratch copy of the inner products.
    std::shared_ptr<IProductWRTPhysTraceBlockOp<TData>> m_lift;
    /// Scratch: the inner products with the trace lifted onto them, sized
    /// on the first Apply() and again if the component count changes; and
    /// one element group of them scaled, or the Pointwise factor of one
    /// element group.
    std::unique_ptr<LibUtilities::BlockAccessor<TData, FieldState::Phys>> m_vol;
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_scaled;

    /**
     * @brief The block's operator with the lift of @p trace: the trace is
     * held for this application alone, then the block is applied as by
     * Apply().
     */
    void v_ApplyLift(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &trace,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock) override
    {
        this->m_trace = &trace;
        v_Apply(inblock, outblock);
        this->m_trace = nullptr;
    }

    void v_Apply(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock) override
    {
        if (m_form == PhysLiftProjectForm::ElmtInvMass)
        {
            this->template ApplyElmtInvMass<MemSpace>(inblock, outblock);
        }
        else
        {
            const unsigned int ncomp =
                inblock.GetNumComponents() * inblock.GetNumHomoModes();
            if (!m_vol ||
                m_vol->GetNumComponents() != inblock.GetNumComponents() ||
                m_vol->GetNumHomoModes() != inblock.GetNumHomoModes())
            {
                m_vol = std::make_unique<
                    LibUtilities::BlockAccessor<TData, FieldState::Phys>>(
                    inblock.GetNumElements(),
                    inblock.GetNumElementsWithPadding(), inblock.GetNumData(),
                    1,
                    LibUtilities::MemoryRegion<TData>(inblock.CompSize() *
                                                      ncomp),
                    nullptr, inblock.GetNumComponents(),
                    inblock.GetNumHomoModes());
            }
            m_vol->template Copy<MemSpace>(inblock);
            if (this->m_trace)
            {
                m_lift->Apply(*this->m_trace, *m_vol);
            }

            // Initialize pointers.
            TData *volPtr = m_vol->template GetPtr<MemSpace, ReadWrite>();
            TData *outPtr =
                this->m_append
                    ? outblock.template GetPtr<MemSpace, ReadWrite>()
                    : outblock.template GetPtr<MemSpace, WriteOnly>();
            const auto jac     = reinterpret_cast<const simd_t *>(m_jacptr);
            const auto weights = reinterpret_cast<const simd_t *>(m_weightsptr);

            // Get interleave parameter: the scratch, and the output when it
            // is appended to, may each arrive at its own width; the output
            // leaves at the input's.
            const unsigned int volInterleaveWidth = m_vol->GetInterleaveWidth();
            const unsigned int inInterleaveWidth = inblock.GetInterleaveWidth();
            unsigned int outInterleaveWidth      = m_implInterleaveWidth;
            if (this->m_append)
            {
                outInterleaveWidth = outblock.GetInterleaveWidth();
            }
            const size_t chunkSize =
                std::max({m_implInterleaveWidth, volInterleaveWidth,
                          outInterleaveWidth, inInterleaveWidth});
            const size_t width_ratio = chunkSize / m_implInterleaveWidth;

            // Dispatch kernel.
            auto gemm_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
                simd_t::width, m_nqTot, m_nqTot, 1.0, (TData)this->m_append);

            // Loop over components.
            for (unsigned int n = 0; n < ncomp; ++n)
            {
                // Loop over element groups.
                for (size_t e = 0;
                     e < m_vol->GetNumElmtGroups(m_implInterleaveWidth); ++e)
                {
                    // Reshape, if necessary.
                    if (e % width_ratio == 0)
                    {
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            m_implInterleaveWidth, volInterleaveWidth,
                            chunkSize, m_nqTot, volPtr);
                        if (this->m_append)
                        {
                            LibUtilities::ReshapeStorage<ExecSpace>(
                                m_implInterleaveWidth, outInterleaveWidth,
                                chunkSize, m_nqTot, outPtr);
                        }
                    }

                    const auto vol = reinterpret_cast<const simd_t *>(volPtr);
                    const auto out = reinterpret_cast<simd_t *>(outPtr);
                    if (m_form == PhysLiftProjectForm::Pointwise)
                    {
                        // out (+)= scale / (w J) * vol.
                        if (m_isDeformed)
                        {
                            PhysLiftProjectPointScaleKernel<true>(
                                m_nqTot, this->m_scale, weights,
                                jac + e * m_nqTot, m_scaled.data());
                        }
                        else
                        {
                            PhysLiftProjectPointScaleKernel<false>(
                                m_nqTot, this->m_scale, weights, jac + e,
                                m_scaled.data());
                        }
                        if (this->m_append)
                        {
                            PhysLiftProjectPointwiseKernel<true>(
                                m_nqTot, m_scaled.data(), vol, out);
                        }
                        else
                        {
                            PhysLiftProjectPointwiseKernel<false>(
                                m_nqTot, m_scaled.data(), vol, out);
                        }
                    }
                    else
                    {
                        // out (+)= scale / J * P vol: the scaled inner
                        // products multiplied by the projector.
                        const simd_t coefScale =
                            SafeDivide(simd_t(this->m_scale), jac[e]);
                        for (unsigned int q = 0; q < m_nqTot; ++q)
                        {
                            m_scaled[q] = vol[q] * coefScale;
                        }
                        gemm_kernel(
                            reinterpret_cast<const TData *>(m_scaled.data()),
                            m_projptr, outPtr);
                    }

                    // Reshape back, if necessary.
                    if (e % width_ratio == width_ratio - 1)
                    {
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                            m_nqTot,
                            outPtr -
                                (width_ratio - 1) * m_nqTot * simd_t::width);
                    }

                    // Increment pointers.
                    volPtr += m_nqTot * simd_t::width;
                    outPtr += m_nqTot * simd_t::width;
                }
            }

            m_vol->template SetInterleaveWidth<TData>(m_implInterleaveWidth);
            outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
        }
    }
};

} // namespace Nektar::MultiRegions::detail
