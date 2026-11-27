///////////////////////////////////////////////////////////////////////////////
//
// File: AdvDiffFusedElmtOp.hpp
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
// Description: Interface of the fused element-local advection-diffusion
// operator: the element-local part of a weak DG advection-diffusion right-hand
// side in one pass.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file AdvDiffFusedElmtOp.hpp
 * @brief Interface of the fused element-local advection-diffusion operator.
 *
 * @details
 * The operator takes the element-local part of a weak DG advection-diffusion
 * right-hand side (interior penalty under the average-of-normal-fluxes
 * viscous trace form) as one pass over the element groups. For each group it
 * forms the state on the traces, the gradient, the advective and viscous
 * fluxes, the derivative inner product of the flux vector, each element's
 * outward normal viscous flux on its traces and, where asked for, the
 * gradient on them. The gradient and the fluxes are held in workspace rather
 * than in fields. The right-hand side closes with the trace flux and the
 * lift and projection (MultiRegions::PhysLiftProjectOp), as it does after the
 * separate operators.
 *
 * The stages it shares with the advection pass (the state on the traces and
 * the derivative inner product) are the kernels of
 * SolverCore/Advection/AdvectionFusedElmt, which this pass calls between its
 * viscous stages; an inviscid flux takes the advection operator
 * (AdvectionFusedElmtOp) instead.
 *
 * The pass is generic over a flux policy (AdvDiffFusedElmtKernels.hpp).
 * The family is laid out as:
 * - AdvDiffFusedElmtOp (this file): the interface, independent of the
 *   policy;
 * - detail::AdvDiffFusedElmtOpImpl (AdvDiffFusedElmtOpImpl.hpp): the
 *   whole-field implementation registered per execution space;
 * - AdvDiffFusedElmtBlockOp (AdvDiffFusedElmtBlockOp.hpp): the per-block
 *   interface, and detail::AdvDiffFusedElmtBlockOpImpl, templated on the
 *   policy and registered for it by the code that supplies it
 *   (ADD_POLICY_ELMT_BLOCK_OPERATOR, cmake/NektarCommon.cmake).
 */

#pragma once

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/BasicUtils/Field/Field.hpp>
#include <MultiRegions/Common/Operator.hpp>
#include <SolverCore/AdvDiffusion/AdvDiffFusedElmt/AdvDiffFusedElmtBlockOp.hpp>
#include <SolverCore/Advection/AdvectionFusedElmt/AdvectionFusedElmtOp.hpp>
#include <SolverCore/SolverCore.hpp>

#include <memory>
#include <string>
#include <vector>

namespace Nektar::SolverCore
{

/**
 * @brief The element-local part of a weak DG advection-diffusion right-hand
 * side as one pass over the elements (see the file notes).
 *
 * @tparam TData Floating-point type of the field data.
 */
template <typename TData>
class AdvDiffFusedElmtOp : public MultiRegions::Operator<TData>
{
public:
    /**
     * @brief Create the operator for the flux policy named @p policyName:
     * the whole-field implementation of execution space @p execStr and one
     * block operator per block, of implementation @p implStr, each with
     * the policy built from the session.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Names of the variables.
     * @param   policyName      The flux policy's name.
     * @param   execStr         Execution space; empty takes the session's.
     * @param   implStr         Implementation; empty takes the session's
     *                          for this operator and each block
     *                          (MultiRegions::ElmtOp::GetOpImpl()).
     *
     * @return The operator, or null when no block operator is registered for
     * the policy, execution space and implementation, or one is that does not
     * take a block of the mesh.
     */
    static std::shared_ptr<AdvDiffFusedElmtOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &policyName, const std::string &execStr = "",
        const std::string &implStr = "")
    {
        // Force a genuine cross-library symbol reference into
        // libSolverCore - see EnsureLinked() in SolverCore.hpp. Without
        // it a linker that keeps only what is referenced can drop the
        // library, and the operator registrations go with it.
        EnsureLinked();

        const auto session     = expansionList->GetSession();
        const std::string exec = detail::FusedElmtExec<TData>(session, execStr);
        const auto impls       = detail::FusedElmtBlockImpls<TData>(
            expansionList, name, policyName, exec, implStr);
        std::vector<std::shared_ptr<AdvDiffFusedElmtBlockOp<TData>>> blockOps;
        std::shared_ptr<AdvDiffFusedElmtOp<TData>> op;
        if (detail::CreateFusedElmtBlockOps(expansionList, policyName, exec,
                                            impls, blockOps))
        {
            for (auto &blockOp : blockOps)
            {
                blockOp->InitialisePolicy(session);
            }
            op = MultiRegions::Operator<TData>::template Create<
                AdvDiffFusedElmtOp>(expansionList, components, exec);
            op->m_blockOp = std::move(blockOps);
        }
        return op;
    }

    /// Operator base name; Create() appends the execution space to it to
    /// form the factory key.
    static inline const std::string name = "AdvDiffFusedElmt";

    /**
     * @brief The fused pass, for a policy with no auxiliary fields.
     *
     * @param   in             The state, one component per variable.
     * @param   trace          Out: the state on the element-local traces,
     *                         laid out by MultiRegions::
     *                         GetLocTraceBlockAttributes().
     * @param   traceNormVisc  Out: each element's outward normal viscous
     *                         flux on its traces, laid out as @p trace.
     * @param   out            Out: the negated derivative inner products of
     *                         the flux vector, `-(grad phi, viscFlux -
     *                         advFlux)` per variable.
     *
     * The outputs leave at the width @p in arrived at. The gradient trace,
     * when one was given (SetGradientTrace()), is written on the listed
     * elements.
     */
    void Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
               LibUtilities::Field<TData, FieldState::Phys> &trace,
               LibUtilities::Field<TData, FieldState::Phys> &traceNormVisc,
               LibUtilities::Field<TData, FieldState::Phys> &out)
    {
        v_Apply(in, nullptr, trace, traceNormVisc, out);
    }

    /**
     * @brief The fused pass, for a policy with auxiliary fields.
     *
     * @param   aux  The policy's auxiliary fields, laid out as @p in.
     *
     * The other arguments are those of Apply(in, trace, traceNormVisc,
     * out).
     */
    void Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
               LibUtilities::Field<TData, FieldState::Phys> &aux,
               LibUtilities::Field<TData, FieldState::Phys> &trace,
               LibUtilities::Field<TData, FieldState::Phys> &traceNormVisc,
               LibUtilities::Field<TData, FieldState::Phys> &out)
    {
        v_Apply(in, &aux, trace, traceNormVisc, out);
    }

    /**
     * @brief Where Apply() writes the gradient on the element-local traces,
     * and of which elements: for each block the sorted indices of its
     * elements, padding excluded. @p gradTrace holds those elements only, in
     * list order, with `NDIM * NVAR` components, component `n * NDIM + d`
     * the derivative of variable n along direction d, on the elements' trace
     * points. Neither is owned, and a refill of @p elements in place is not
     * seen: set it again. A null @p gradTrace writes no gradient trace.
     */
    void SetGradientTrace(
        LibUtilities::Field<TData, FieldState::Phys> *gradTrace,
        const std::vector<std::vector<unsigned int>> *elements)
    {
        m_gradTrace         = gradTrace;
        m_gradTraceElements = elements;
        m_gradTraceStale    = true;
    }

protected:
    /// One block operator per block of the expansion list, built by
    /// Create().
    std::vector<std::shared_ptr<AdvDiffFusedElmtBlockOp<TData>>> m_blockOp;
    /// Where the gradient trace is written, and of which elements; not
    /// owned.
    LibUtilities::Field<TData, FieldState::Phys> *m_gradTrace         = nullptr;
    const std::vector<std::vector<unsigned int>> *m_gradTraceElements = nullptr;
    /// The gradient trace was set since the last Apply().
    bool m_gradTraceStale = false;

    AdvDiffFusedElmtOp(const MultiRegions::ExpListSharedPtr &expansionList,
                       const std::vector<std::string> &components)
        : MultiRegions::Operator<TData>(expansionList, components)
    {
    }

    ~AdvDiffFusedElmtOp() override = default;

    /// Implementation hook for Apply(): each block of the fields to its
    /// block operator, the listed elements handed on first if they changed.
    /// @p aux is null for a policy with no auxiliary fields.
    virtual void v_Apply(
        LibUtilities::Field<TData, FieldState::Phys> &in,
        LibUtilities::Field<TData, FieldState::Phys> *aux,
        LibUtilities::Field<TData, FieldState::Phys> &trace,
        LibUtilities::Field<TData, FieldState::Phys> &traceNormVisc,
        LibUtilities::Field<TData, FieldState::Phys> &out)
    {
        ASSERTL1(in.GetNumComponents() == out.GetNumComponents() &&
                     trace.GetNumComponents() == in.GetNumComponents() &&
                     traceNormVisc.GetNumComponents() == in.GetNumComponents(),
                 "Number of state, trace and output components differ");
        ASSERTL0(in.GetNumHomoModes() == 1,
                 "The fused element-local operator takes no homogeneous "
                 "directions.");

        if (m_gradTraceStale)
        {
            for (unsigned int blk = 0; blk < m_blockOp.size(); ++blk)
            {
                m_blockOp[blk]->SetGradientTraceElements(
                    m_gradTrace && m_gradTraceElements
                        ? (*m_gradTraceElements)[blk]
                        : std::vector<unsigned int>{},
                    in.GetBlocks()[blk].GetNumElementsWithPadding());
            }
            m_gradTraceStale = false;
        }

        for (unsigned int blk = 0; blk < m_blockOp.size(); ++blk)
        {
            m_blockOp[blk]->ApplyFusedElmt(
                in.GetBlocks()[blk], aux ? &aux->GetBlocks()[blk] : nullptr,
                trace.GetBlocks()[blk], traceNormVisc.GetBlocks()[blk],
                m_gradTrace ? &m_gradTrace->GetBlocks()[blk] : nullptr,
                out.GetBlocks()[blk]);
        }
    }
};

} // namespace Nektar::SolverCore
