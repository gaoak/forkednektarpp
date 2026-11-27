///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionFusedElmtOp.hpp
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
// Description: Interface of the fused element-local advection operator: the
// element-local part of a weak DG advection right-hand side in one pass.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file AdvectionFusedElmtOp.hpp
 * @brief Interface of the fused element-local advection operator.
 *
 * @details
 * The operator takes the element-local part of a weak DG advection
 * right-hand side as one pass over the element groups. For each group it
 * forms the state on the traces, the advective flux and the derivative inner
 * product of the flux; the flux is held in workspace rather than in a field.
 * The right-hand side closes with the trace flux and the lift and projection
 * (MultiRegions::PhysLiftProjectOp), as it does after the separate operators.
 *
 * The pass is generic over a flux policy (AdvectionFusedElmtKernels.hpp).
 * Its stages are also those of the advection-diffusion pass
 * (SolverCore/AdvDiffusion/AdvDiffFusedElmt), which adds the gradient and the
 * viscous terms and calls the kernels of this directory for the rest.
 *
 * The family is laid out as:
 * - AdvectionFusedElmtOp (this file): the interface, independent of the
 *   policy;
 * - detail::AdvectionFusedElmtOpImpl (AdvectionFusedElmtOpImpl.hpp): the
 *   whole-field implementation registered per execution space;
 * - AdvectionFusedElmtBlockOp (AdvectionFusedElmtBlockOp.hpp): the per-block
 *   interface, and detail::AdvectionFusedElmtBlockOpImpl, templated on the
 *   policy and registered for it by the code that supplies it
 *   (ADD_POLICY_ELMT_BLOCK_OPERATOR, cmake/NektarCommon.cmake).
 */

#pragma once

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/BasicUtils/Field/Field.hpp>
#include <MultiRegions/Common/Operator.hpp>
#include <MultiRegions/ElmtOps/ElmtOp.hpp>
#include <SolverCore/Advection/AdvectionFusedElmt/AdvectionFusedElmtBlockOp.hpp>
#include <SolverCore/SolverCore.hpp>

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Nektar::SolverCore
{

namespace detail
{

/**
 * @brief The execution space a fused element-local operator runs in:
 * @p execStr or, where empty, the session's
 * (MultiRegions::Operator::GetOpExecSpace()).
 */
template <typename TData>
std::string FusedElmtExec(const LibUtilities::SessionReaderSharedPtr &session,
                          const std::string &execStr)
{
    return execStr.empty()
               ? MultiRegions::Operator<TData>::GetOpExecSpace(session)
               : execStr;
}

/**
 * @brief The implementation of each block of @p expansionList for the fused
 * operator @p name: @p implStr or, where it is empty, the session's for the
 * operator and the block (MultiRegions::ElmtOp::GetOpImpl()), asked once per
 * block shape and order.
 *
 * The default is checked against the operator's block operators for the
 * policy @p policyName, so that it falls back to SumFac where the policy has
 * no SumFacTOP block operator, as it does for an operator without a policy.
 *
 * @param   name  The operator's name, as the session's backend sections
 *                give it.
 * @return  One implementation per block.
 */
template <typename TData>
std::vector<std::string> FusedElmtBlockImpls(
    const MultiRegions::ExpListSharedPtr &expansionList,
    const std::string &name, const std::string &policyName,
    const std::string &execStr, const std::string &implStr)
{
    const auto session = expansionList->GetSession();
    std::map<std::pair<LibUtilities::ShapeType, int>, std::string>
        implByShapeOrder;
    std::vector<std::string> impls;
    for (auto &coll : MultiRegions::GetCollections(expansionList))
    {
        if (!implStr.empty())
        {
            impls.push_back(implStr);
            continue;
        }
        const auto exp = coll.GetExpVector()[0];
        const auto key = std::make_pair(exp->DetShapeType(),
                                        exp->EvalBasisNumModesMax() - 1);
        auto it        = implByShapeOrder.find(key);
        if (it == implByShapeOrder.end())
        {
            it =
                implByShapeOrder
                    .emplace(
                        key,
                        MultiRegions::ElmtOp<FieldState::Phys, FieldState::Phys,
                                             TData>::GetOpImpl(name, execStr,
                                                               session, exp,
                                                               policyName))
                    .first;
        }
        impls.push_back(it->second);
    }
    return impls;
}

/**
 * @brief One block operator per block of @p expansionList, of the family
 * @p TBlockOp, for the policy @p policyName and each block's implementation
 * in @p impls.
 *
 * @param   blockOps  Out: the block operators, or none when a block has no
 *                    block operator registered for its implementation or its
 *                    block operator does not take it; the caller then keeps
 *                    the separate operators for the whole mesh.
 * @return  Whether every block has one.
 */
template <typename TBlockOp>
bool CreateFusedElmtBlockOps(
    const MultiRegions::ExpListSharedPtr &expansionList,
    const std::string &policyName, const std::string &execStr,
    const std::vector<std::string> &impls,
    std::vector<std::shared_ptr<TBlockOp>> &blockOps)
{
    blockOps.clear();
    for (unsigned int blk = 0; blk < impls.size(); ++blk)
    {
        const auto exp =
            MultiRegions::GetCollection(expansionList, blk).GetExpVector()[0];
        auto blockOp = TBlockOp::Create(
            blk, exp, expansionList->GetDataWarehouseSharedPtr(), policyName,
            execStr, impls[blk]);
        if (!blockOp)
        {
            blockOps.clear();
            return false;
        }
        blockOps.push_back(blockOp);
    }
    return true;
}

} // namespace detail

/**
 * @brief The element-local part of a weak DG advection right-hand side as
 * one pass over the elements (see the file notes).
 *
 * @tparam TData Floating-point type of the field data.
 */
template <typename TData>
class AdvectionFusedElmtOp : public MultiRegions::Operator<TData>
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
    static std::shared_ptr<AdvectionFusedElmtOp<TData>> Create(
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
        std::vector<std::shared_ptr<AdvectionFusedElmtBlockOp<TData>>> blockOps;
        std::shared_ptr<AdvectionFusedElmtOp<TData>> op;
        if (detail::CreateFusedElmtBlockOps(expansionList, policyName, exec,
                                            impls, blockOps))
        {
            for (auto &blockOp : blockOps)
            {
                blockOp->InitialisePolicy(session);
            }
            op = MultiRegions::Operator<TData>::template Create<
                AdvectionFusedElmtOp>(expansionList, components, exec);
            op->m_blockOp = std::move(blockOps);
        }
        return op;
    }

    /// Operator base name; Create() appends the execution space to it to
    /// form the factory key.
    static inline const std::string name = "AdvectionFusedElmt";

    /**
     * @brief The fused pass, for a policy with no auxiliary fields.
     *
     * @param   in     The state, one component per variable.
     * @param   trace  Out: the state on the element-local traces, laid out
     *                 by MultiRegions::GetLocTraceBlockAttributes().
     * @param   out    Out: the negated derivative inner products of the
     *                 negated advective flux, `-(grad phi, -advFlux)` per
     *                 variable.
     *
     * The outputs leave at the width @p in arrived at.
     */
    void Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
               LibUtilities::Field<TData, FieldState::Phys> &trace,
               LibUtilities::Field<TData, FieldState::Phys> &out)
    {
        v_Apply(in, nullptr, trace, out);
    }

    /**
     * @brief The fused pass, for a policy with auxiliary fields.
     *
     * @param   aux    The policy's auxiliary fields, laid out as @p in.
     *
     * The other arguments are those of Apply(in, trace, out).
     */
    void Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
               LibUtilities::Field<TData, FieldState::Phys> &aux,
               LibUtilities::Field<TData, FieldState::Phys> &trace,
               LibUtilities::Field<TData, FieldState::Phys> &out)
    {
        v_Apply(in, &aux, trace, out);
    }

protected:
    /// One block operator per block of the expansion list, built by
    /// Create().
    std::vector<std::shared_ptr<AdvectionFusedElmtBlockOp<TData>>> m_blockOp;

    AdvectionFusedElmtOp(const MultiRegions::ExpListSharedPtr &expansionList,
                         const std::vector<std::string> &components)
        : MultiRegions::Operator<TData>(expansionList, components)
    {
    }

    ~AdvectionFusedElmtOp() override = default;

    /// Implementation hook for Apply(): each block of the fields to its
    /// block operator.
    /// @p aux is null for a policy with no auxiliary fields.
    virtual void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
                         LibUtilities::Field<TData, FieldState::Phys> *aux,
                         LibUtilities::Field<TData, FieldState::Phys> &trace,
                         LibUtilities::Field<TData, FieldState::Phys> &out)
    {
        ASSERTL1(in.GetNumComponents() == out.GetNumComponents() &&
                     trace.GetNumComponents() == in.GetNumComponents(),
                 "Number of state, trace and output components differ");
        ASSERTL0(in.GetNumHomoModes() == 1,
                 "The fused element-local operator takes no homogeneous "
                 "directions.");

        for (unsigned int blk = 0; blk < m_blockOp.size(); ++blk)
        {
            m_blockOp[blk]->ApplyFusedElmt(
                in.GetBlocks()[blk], aux ? &aux->GetBlocks()[blk] : nullptr,
                trace.GetBlocks()[blk], out.GetBlocks()[blk]);
        }
    }
};

} // namespace Nektar::SolverCore
