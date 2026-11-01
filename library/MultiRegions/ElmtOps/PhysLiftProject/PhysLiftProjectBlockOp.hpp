///////////////////////////////////////////////////////////////////////////////
//
// File: PhysLiftProjectBlockOp.hpp
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
// Description: Per-block interface of the lift-and-project element operator.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file PhysLiftProjectBlockOp.hpp
 * @brief Per-block interface of the PhysLiftProject element operator, and
 * the choice of its form for a block.
 *
 * @details
 * A PhysLiftProjectBlockOp computes, for every element of one block,
 * \f[ u \mathrel{(+)=} s\, B M^{-1} B^{\mathsf T} (f + L g), \f]
 * with @c f the physical-space inner products of the element (the
 * pre-\f$B^{\mathsf T}\f$ arrays IProductWRTDerivBase and
 * IProductWRTPhysTrace write), @c g the element-local trace values and
 * @c L the trace lift of IProductWRTPhysTrace; see PhysLiftProjectOp.hpp.
 * Besides the block states and the factory entry the class carries the
 * scale and append settings, ApplyLift(), which applies the block with the
 * lift of its block of trace values, and master's four block operators in
 * sequence for the ElmtInvMass form (CreateChain(), ApplyElmtInvMass()),
 * which every implementation takes there.
 *
 * detail::PhysLiftProjectFormOf() is the one place the form of a block is
 * decided, shared by the whole-field operator (which hands the per-element
 * inverse mass matrices only to the blocks that read them) and the
 * implementations.
 */

#pragma once

#include <MultiRegions/ElmtOps/BwdTrans/BwdTransBlockOp.hpp>
#include <MultiRegions/ElmtOps/ElmtBlockOp.hpp>
#include <MultiRegions/ElmtOps/IProductWRTBase/IProductWRTBaseBlockOp.hpp>
#include <MultiRegions/ElmtOps/IProductWRTPhysTrace/IProductWRTPhysTraceBlockOp.hpp>
#include <MultiRegions/ElmtOps/MultiplyByElmtInvMass/MultiplyByElmtInvMassBlockOp.hpp>

#include <memory>
#include <vector>

namespace Nektar::MultiRegions
{

namespace detail
{

/// How a block applies \f$B M^{-1} B^{\mathsf T}\f$.
enum class PhysLiftProjectForm
{
    /// As many modes as quadrature points: B is square, so
    /// \f$B M^{-1} B^{\mathsf T} = (W J)^{-1}\f$, a pointwise scale by the
    /// quadrature weight and the Jacobian, affine or deformed.
    Pointwise,
    /// Affine elements: the basis transpose, the standard inverse mass over
    /// the element's Jacobian and the backward transform.
    StdInvMass,
    /// Deformed elements: IProductWRTPhysTrace, IProductWRTBase,
    /// MultiplyByElmtInvMass with each element's own inverse mass, and
    /// BwdTrans, in sequence.
    ElmtInvMass
};

/// Which of the Device fused kernels a shared-memory size is for.
enum class PhysLiftProjectKernel
{
    /// The lift and the form in one launch (PhysLiftProjectKernelLauncher()).
    Fused,
    /// The lift and the basis transpose, before the dense product
    /// (PhysLiftProjectLiftBasisTransposeKernelLauncher()).
    LiftBasisTranspose,
    /// The scale and the backward transform, after it
    /// (PhysLiftProjectScaleBwdTransKernelLauncher()).
    ScaleBwdTrans
};

/**
 * @brief The form the block whose representative expansion is @p exp takes.
 * The square test is on the total counts: equal per-direction counts do
 * not make B square on a collapsed-coordinate shape, whose mode count is a
 * fraction of the tensor product.
 */
inline PhysLiftProjectForm PhysLiftProjectFormOf(
    const LocalRegions::ExpansionSharedPtr &exp)
{
    if (exp->GetNcoeffs() == exp->GetTotPoints())
    {
        return PhysLiftProjectForm::Pointwise;
    }
    if (exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed)
    {
        return PhysLiftProjectForm::ElmtInvMass;
    }
    return PhysLiftProjectForm::StdInvMass;
}

} // namespace detail

/**
 * @brief Base class of the per-block lift-and-project operators: physical
 * space in, with the trace values through ApplyLift(), physical space out.
 *
 * @tparam TData  Floating-point type of the field data.
 */
template <typename TData>
class PhysLiftProjectBlockOp
    : public ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>
{
public:
    /// Create a block operator through the block-operator factory; see
    /// ElmtBlockOp::Create().
    static std::shared_ptr<PhysLiftProjectBlockOp<TData>> Create(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse,
        const std::string &execStr, const std::string &implStr)
    {
        return ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>::
            template Create<PhysLiftProjectBlockOp>(
                block_idx, exp, dataWarehouse, execStr, implStr);
    }

    /// Block-operator base name; the factory key appends the execution
    /// space and the implementation.
    static inline const std::string name = "BlockPhysLiftProject";

    /// The factor the result is scaled by.
    void SetScale(const TData &scale)
    {
        m_scale = scale;
    }

    /// Accumulate onto the output instead of overwriting it.
    void SetAppend(const bool &append)
    {
        m_append = append;
    }

    /// Apply the block's operator to @p inblock with the lift of the block
    /// of trace values @p trace onto it; Apply() leaves the lift out.
    void ApplyLift(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &trace,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        v_ApplyLift(inblock, trace, outblock);
    }

    /// The per-element inverse mass matrices of a deformed block, element
    /// after element, each column-major and nmTot x nmTot, the padding
    /// elements zero (MultiplyByElmtInvMassOp::ElmtInvMassMatrices()); read
    /// only by a block of the ElmtInvMass form, whose MultiplyByElmtInvMass
    /// takes them.
    void SetInvMassMatrix(std::vector<TData> &dmat)
    {
        m_chainInvMass->SetInvMassMatrix(dmat);
    }

protected:
    /// The implementation's ApplyLift().
    virtual void v_ApplyLift(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &trace,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock) = 0;

    TData m_scale = 1.0;
    bool m_append = false;
    /// The block of trace values lifted onto the inner products, set by
    /// v_ApplyLift() for that application alone and null otherwise.
    LibUtilities::BlockAccessor<TData, FieldState::Phys> *m_trace = nullptr;
    /// The ElmtInvMass form (CreateChain()): master's four block operators
    /// and the scratch of the inner products with the trace lifted onto
    /// them and of the coefficients before and after the inverse mass,
    /// sized on the first ApplyElmtInvMass().
    std::shared_ptr<IProductWRTPhysTraceBlockOp<TData>> m_chainLift;
    std::shared_ptr<IProductWRTBaseBlockOp<TData>> m_chainBasisTranspose;
    std::shared_ptr<MultiplyByElmtInvMassBlockOp<TData>> m_chainInvMass;
    std::shared_ptr<BwdTransBlockOp<TData>> m_chainBwdTrans;
    std::unique_ptr<LibUtilities::BlockAccessor<TData, FieldState::Phys>>
        m_chainVol;
    std::unique_ptr<LibUtilities::BlockAccessor<TData, FieldState::Coeff>>
        m_chainCoef, m_chainCoef2;

    PhysLiftProjectBlockOp(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>(block_idx, exp,
                                                                 dataWarehouse)
    {
    }

    ~PhysLiftProjectBlockOp() override = default;

    /**
     * @brief Create the block operators of the ElmtInvMass form in
     * execution space @p execStr: IProductWRTPhysTrace appending,
     * IProductWRTBase without the metric, MultiplyByElmtInvMass and
     * BwdTrans, the first two and the last of implementation @p implStr.
     */
    void CreateChain(const std::string &execStr, const std::string &implStr)
    {
        m_chainLift = IProductWRTPhysTraceBlockOp<TData>::Create(
            this->m_block_idx, this->m_exp, this->m_dataWarehouse, execStr,
            implStr);
        m_chainLift->SetAppend(true);
        m_chainBasisTranspose = IProductWRTBaseBlockOp<TData>::Create(
            this->m_block_idx, this->m_exp, this->m_dataWarehouse, execStr,
            implStr);
        m_chainBasisTranspose->SetIntegration(false);
        m_chainInvMass = MultiplyByElmtInvMassBlockOp<TData>::Create(
            this->m_block_idx, this->m_exp, this->m_dataWarehouse, execStr,
            Generic::name);
        m_chainBwdTrans = BwdTransBlockOp<TData>::Create(
            this->m_block_idx, this->m_exp, this->m_dataWarehouse, execStr,
            implStr);
    }

    /**
     * @brief The ElmtInvMass form, master's chain: @p in copied into the
     * scratch, the trace lifted onto it, then the basis transpose with the
     * scale, each element's own inverse mass and the backward transform into
     * @p out, the copy on stream @p streamID. The scratch is sized after
     * @p in; the scale and append settings are set here, where they are used.
     */
    template <typename MemSpace>
    void ApplyElmtInvMass(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &in,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &out,
        const unsigned int streamID = 0)
    {
        const unsigned int ncomp = in.GetNumComponents();
        const unsigned int nhomo = in.GetNumHomoModes();
        if (!m_chainVol || m_chainVol->GetNumComponents() != ncomp ||
            m_chainVol->GetNumHomoModes() != nhomo)
        {
            const size_t nelmtPad    = in.GetNumElementsWithPadding();
            const unsigned int nmTot = this->m_exp->GetNcoeffs();
            m_chainVol               = std::make_unique<
                              LibUtilities::BlockAccessor<TData, FieldState::Phys>>(
                in.GetNumElements(), nelmtPad, in.GetNumData(), 1,
                LibUtilities::MemoryRegion<TData>(in.CompSize() * ncomp *
                                                  nhomo),
                nullptr, ncomp, nhomo);
            m_chainCoef = std::make_unique<
                LibUtilities::BlockAccessor<TData, FieldState::Coeff>>(
                in.GetNumElements(), nelmtPad, nmTot, 1,
                LibUtilities::MemoryRegion<TData>(nelmtPad * nmTot * ncomp *
                                                  nhomo),
                nullptr, ncomp, nhomo);
            m_chainCoef2 = std::make_unique<
                LibUtilities::BlockAccessor<TData, FieldState::Coeff>>(
                in.GetNumElements(), nelmtPad, nmTot, 1,
                LibUtilities::MemoryRegion<TData>(nelmtPad * nmTot * ncomp *
                                                  nhomo),
                nullptr, ncomp, nhomo);
        }
        m_chainVol->template Copy<MemSpace>(in, streamID);
        if (m_trace)
        {
            m_chainLift->Apply(*m_trace, *m_chainVol);
        }
        m_chainBasisTranspose->SetScale(m_scale);
        m_chainBasisTranspose->Apply(*m_chainVol, *m_chainCoef);
        m_chainInvMass->Apply(*m_chainCoef, *m_chainCoef2);
        m_chainBwdTrans->SetAppend(m_append);
        m_chainBwdTrans->Apply(*m_chainCoef2, out);
    }
};

} // namespace Nektar::MultiRegions
