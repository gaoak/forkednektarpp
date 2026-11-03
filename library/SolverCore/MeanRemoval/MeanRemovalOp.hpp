///////////////////////////////////////////////////////////////////////////////
//
// File: MeanRemovalOp.hpp
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

#pragma once

#include "SolverCore/MeanRemoval/MeanRemovalBlockOp.hpp"
#include "SolverCore/SolverCore.hpp"
#include <MultiRegions/Common/Operator.hpp>

namespace Nektar::SolverCore
{

template <typename TData> class LinearSystemOp;

// Subtracts the volume-averaged mean of a physical-space field from itself,
// in place. A PDE variable governed by a singular system (e.g. pressure
// under Neumann/periodic-only boundaries) is only determined up to an
// arbitrary additive constant; this operator removes that constant so that
// norms/errors computed from the field reflect genuine discretisation error
// rather than the arbitrary offset.
//
// Dispatch across execution spaces (Serial/AVX/Device) is resolved entirely
// by Create(), via the same OperatorFactory used by every Operators-library
// operator -- callers just do MeanRemovalOp<TData>::Create(...)->Apply(field),
// with no execution-space conditionals of their own.
template <typename TData>
class MeanRemovalOp : public MultiRegions::Operator<TData>
{
public:
    static std::shared_ptr<MeanRemovalOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "")
    {
        // Force a genuine cross-library symbol reference into libSolverCore so
        // the MeanRemoval*OpImpl static factory registrations are not dropped
        // by the linker/loader -- see EnsureLinked() in SolverCore.hpp.
        EnsureLinked();

        auto session = expansionList->GetSession();

        std::string execStr0 =
            (execStr == "")
                ? MultiRegions::Operator<TData>::GetOpExecSpace(session)
                : execStr;

        auto op = MultiRegions::Operator<TData>::template Create<MeanRemovalOp>(
            expansionList, components, execStr0);

        auto blockAttr =
            MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                expansionList);

        for (unsigned int block_idx = 0; block_idx < blockAttr.size();
             block_idx++)
        {
            const auto exp =
                MultiRegions::GetCollection(expansionList, block_idx)
                    .GetExpVector()[0];

            op->m_blockOp.push_back(MeanRemovalBlockOp<TData>::Create(
                block_idx, exp, expansionList->GetDataWarehouseSharedPtr(),
                execStr0));
            op->m_blockOp.back()->SetNumElements(
                blockAttr[block_idx].GetNumElements());
        }

        return op;
    }

    static inline const std::string name = "MeanRemoval";

    void Apply(LibUtilities::Field<TData, FieldState::Phys> &inout)
    {
        v_Apply(inout);
    }

    void operator()(LibUtilities::Field<TData, FieldState::Phys> &inout)
    {
        v_Apply(inout);
    }

    /**
     * @brief Drive the domain integral of each component to the surface
     * integral of the Neumann boundary values of @p system, instead of to
     * zero.
     *
     * @details
     * A singular pressure system is solvable only when the domain integral
     * of its forcing equals the surface integral of its Neumann boundary
     * values. Apply() then subtracts (integral_c - surface_c) / volume from
     * component c, where surface_c is summed by
     * LinearSystemOp::AddNeumannSurfaceIntegral() into the reduction of the
     * domain integrals, so that the two are reduced across ranks and read
     * back to the host together. A nullptr @p system, the default, removes
     * the mean.
     *
     * The operator keeps a non-owning pointer to @p system, which must
     * outlive its use here.
     *
     * @param system The system with one Neumann value per component, or
     * nullptr.
     */
    void SetNeumannBalance(LinearSystemOp<TData> *system)
    {
        m_neumannSystem = system;
    }

    /**
     * @brief The mean that Apply() drives each component to: the surface
     * integral of the Neumann boundary values of the system set by
     * SetNeumannBalance(), surface_c, divided by the volume of the domain,
     * or zero without a system.
     *
     * @details
     * The constant forcing with this value has the domain integral that
     * the system needs, without a field to balance. Collective over the
     * row communicator: surface_c is reduced across ranks and read back to
     * the host once.
     */
    std::vector<TData> GetBalancedMean()
    {
        return v_GetBalancedMean();
    }

protected:
    // Volume of the domain, computed on the first Apply() or
    // GetBalancedMean() and kept.
    bool m_volumeComputed = false;
    TData m_volume        = 1.0;
    LibUtilities::MemoryRegion<TData> m_data;
    std::vector<std::shared_ptr<MeanRemovalBlockOp<TData>>> m_blockOp;
    LinearSystemOp<TData> *m_neumannSystem = nullptr;

    MeanRemovalOp(const MultiRegions::ExpListSharedPtr &expansionList,
                  const std::vector<std::string> &components)
        : MultiRegions::Operator<TData>(expansionList, components)
    {
    }

    ~MeanRemovalOp() override = default;

    virtual void v_Apply(
        LibUtilities::Field<TData, FieldState::Phys> &inout) = 0;

    virtual std::vector<TData> v_GetBalancedMean() = 0;
};
} // namespace Nektar::SolverCore
