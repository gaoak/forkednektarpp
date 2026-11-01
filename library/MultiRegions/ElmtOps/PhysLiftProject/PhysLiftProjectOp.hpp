///////////////////////////////////////////////////////////////////////////////
//
// File: PhysLiftProjectOp.hpp
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
// Description: Whole-field interface of the lift-and-project element operator,
// from physical-space inner products and element-local trace values to the
// physical-space projection onto the polynomial space.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file PhysLiftProjectOp.hpp
 * @brief Public interface of the PhysLiftProject element operator: the
 * trace lift followed by the projection through coefficient space, the tail
 * of a weak discontinuous Galerkin right-hand side.
 *
 * @details
 * For every element the operator computes
 * \f[ u \mathrel{(+)=} s\, B M^{-1} B^{\mathsf T} (f + L g), \f]
 * where
 * - @c f holds the element's physical-space inner products, the arrays
 *   IProductWRTDerivBase (physical-space output) and IProductWRTPhysTrace
 *   write, which carry the quadrature metric already;
 * - @c g holds values on the element's own traces, laid out by
 *   GetLocTraceBlockAttributes(), and @c L is the trace lift of
 *   IProductWRTPhysTrace, \f$\sum_F \int_F g\, h_{\boldsymbol p} J_F\f$;
 * - \f$B^{\mathsf T}\f$ is the basis transpose without the metric
 *   (IProductWRTBase with SetIntegration(false)), \f$M\f$ the element mass
 *   matrix on the same quadrature and \f$B\f$ the backward transform;
 * - @c s is the SetScale() factor, and SetAppend() accumulates onto @c u.
 *
 * It is therefore the four-operator chain IProductWRTPhysTrace (appending),
 * IProductWRTBase without integration, MultiplyByElmtInvMass and BwdTrans,
 * up to rounding. Each block takes one of three forms, decided at
 * construction by detail::PhysLiftProjectFormOf():
 * - **Pointwise**, where the element has as many modes as quadrature points:
 *   B is square, \f$M = B^{\mathsf T} W J B\f$, so
 *   \f$B M^{-1} B^{\mathsf T} = (W J)^{-1}\f$, a scale by the quadrature
 *   weight and the Jacobian at each point, on affine and deformed elements
 *   alike;
 * - **StdInvMass**, on affine elements otherwise: the sum-factorised basis
 *   transpose, the standard inverse mass over the Jacobian (a scale per
 *   mode where the standard mass is diagonal) and the backward transform;
 * - **ElmtInvMass**, on deformed elements otherwise: the four operators in
 *   sequence, as their block operators, MultiplyByElmtInvMass with each
 *   element's own inverse mass as MultiplyByElmtInvMassOp builds it
 *   (MultiplyByElmtInvMassOp::ElmtInvMassMatrices()).
 *
 * Apply(in, trace, out) lifts @c trace, each block through its block
 * operator's ApplyLift(); the two-argument Apply(in, out) projects @c in
 * alone. @c in and @c out are distinct fields; @c out leaves
 * at the interleave width of @c in.
 *
 * The family follows the standard element-operator layout (see ElmtOp.hpp):
 * this interface, detail::PhysLiftProjectOpImpl for the factory, the
 * per-block interface PhysLiftProjectBlockOp, and two block implementations
 * in each execution space, chosen as for the other element operators
 * (`--opImpl` or the session's per-operator setting):
 * - SumFac: PhysLiftProjectBlockOpSerialAVXSumFac.hpp (Serial and AVX, one
 *   element group at a time) and PhysLiftProjectBlockOpDeviceSumFac.hpp
 *   (Device, through the Device kernels of the four operators, SumFac or
 *   SumFacTOP), the Pointwise and StdInvMass forms;
 * - StdMat: PhysLiftProjectBlockOpSerialAVXStdMat.hpp and
 *   PhysLiftProjectBlockOpDeviceStdMat.hpp, the StdInvMass form as one
 *   dense product with \f$P = B M_{\mathrm{std}}^{-1} B^{\mathsf T}\f$
 *   (nqTot x nqTot, from the data warehouse) scaled by `scale / J`, and the
 *   Pointwise form through the kernels it shares with SumFac;
 * and in both the ElmtInvMass form through PhysLiftProjectBlockOp's
 * CreateChain() and ApplyElmtInvMass().
 */

#pragma once

#include <MultiRegions/ElmtOps/ElmtOp.hpp>

#include <MultiRegions/ElmtOps/MultiplyByElmtInvMass/MultiplyByElmtInvMassOp.hpp>
#include <MultiRegions/ElmtOps/PhysLiftProject/PhysLiftProjectBlockOp.hpp>

namespace Nektar::MultiRegions
{

/**
 * @brief Whole-field interface of the lift-and-project operator: holds one
 * PhysLiftProjectBlockOp per element block and hands each block of the
 * fields to it.
 *
 * @tparam TData  Floating-point type of the field data.
 */
template <typename TData>
class PhysLiftProjectOp
    : public ElmtOp<FieldState::Phys, FieldState::Phys, TData>
{
    friend class ElmtOp<FieldState::Phys, FieldState::Phys, TData>;

public:
    using ElmtOp<FieldState::Phys, FieldState::Phys, TData>::Apply;

    /**
     * @brief Create the operator through the operator and block-operator
     * factories (see ElmtOp::Create()), and hand each block of the
     * ElmtInvMass form its elements' inverse mass matrices.
     */
    static std::shared_ptr<PhysLiftProjectOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "", const std::string &implStr = "")
    {
        auto op =
            ElmtOp<FieldState::Phys, FieldState::Phys, TData>::template Create<
                PhysLiftProjectOp, PhysLiftProjectBlockOp>(
                expansionList, components, execStr, implStr);

        // The per-element inverse mass matrices of the blocks of the
        // ElmtInvMass form, which hand them to their MultiplyByElmtInvMass,
        // as MultiplyByElmtInvMassOp builds them.
        const auto blockAttr =
            MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                expansionList);
        for (unsigned int blk = 0; blk < op->m_blockOp.size(); ++blk)
        {
            const auto exp = MultiRegions::GetCollection(expansionList, blk)
                                 .GetExpVector()[0];
            if (detail::PhysLiftProjectFormOf(exp) ==
                detail::PhysLiftProjectForm::ElmtInvMass)
            {
                std::vector<TData> dmat =
                    MultiplyByElmtInvMassOp<TData>::ElmtInvMassMatrices(
                        expansionList, blk, blockAttr[blk]);
                op->m_blockOp[blk]->SetInvMassMatrix(dmat);
            }
        }

        return op;
    }

    /// Operator base name; Create() appends the execution space to it to
    /// form the factory key.
    static inline const std::string name = "PhysLiftProject";

    /**
     * @brief `out (+)= scale * B M^{-1} B^T (in + L trace)`.
     *
     * @param   in      Volume inner products, laid out by
     *                  GetBlockAttributes().
     * @param   trace   Values on the elements' own traces, laid out by
     *                  GetLocTraceBlockAttributes(), with the components of
     *                  @p in.
     * @param   out     Volume field, written or accumulated onto.
     */
    void Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
               LibUtilities::Field<TData, FieldState::Phys> &trace,
               LibUtilities::Field<TData, FieldState::Phys> &out)
    {
        v_ApplyLift(in, trace, out);
    }

    /// The factor the result is scaled by, on every block.
    void SetScale(const TData &scale)
    {
        for (unsigned int blk = 0; blk < m_blockOp.size(); ++blk)
        {
            m_blockOp[blk]->SetScale(scale);
        }
    }

    /// Accumulate onto the output instead of overwriting it, on every
    /// block.
    void SetAppend(const bool &append)
    {
        for (unsigned int blk = 0; blk < m_blockOp.size(); ++blk)
        {
            m_blockOp[blk]->SetAppend(append);
        }
    }

protected:
    std::vector<std::shared_ptr<PhysLiftProjectBlockOp<TData>>> m_blockOp;

    PhysLiftProjectOp(const MultiRegions::ExpListSharedPtr &expansionList,
                      const std::vector<std::string> &components)
        : ElmtOp<FieldState::Phys, FieldState::Phys, TData>(expansionList,
                                                            components)
    {
    }

    ~PhysLiftProjectOp() override = default;

    /// The projection of @p in alone.
    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
                 LibUtilities::Field<TData, FieldState::Phys> &out) override
    {
        ASSERTL1(in.GetNumComponents() == out.GetNumComponents(),
                 "Number of input and output components differ");
        ASSERTL1(in.GetNumHomoModes() == out.GetNumHomoModes(),
                 "Number of input and output homogeneous modes differ");

        for (unsigned int blk = 0; blk < m_blockOp.size(); ++blk)
        {
            m_blockOp[blk]->Apply(in.GetBlocks()[blk], out.GetBlocks()[blk]);
        }
    }

    /// Implementation hook for the three-argument Apply().
    virtual void v_ApplyLift(
        LibUtilities::Field<TData, FieldState::Phys> &in,
        LibUtilities::Field<TData, FieldState::Phys> &trace,
        LibUtilities::Field<TData, FieldState::Phys> &out)
    {
        ASSERTL1(in.GetNumComponents() == out.GetNumComponents() &&
                     trace.GetNumComponents() == in.GetNumComponents(),
                 "Number of input, trace and output components differ");
        ASSERTL1(in.GetNumHomoModes() == out.GetNumHomoModes() &&
                     trace.GetNumHomoModes() == in.GetNumHomoModes(),
                 "Number of input, trace and output homogeneous modes differ");

        for (unsigned int blk = 0; blk < m_blockOp.size(); ++blk)
        {
            m_blockOp[blk]->ApplyLift(in.GetBlocks()[blk],
                                      trace.GetBlocks()[blk],
                                      out.GetBlocks()[blk]);
        }
    }
};

} // namespace Nektar::MultiRegions
