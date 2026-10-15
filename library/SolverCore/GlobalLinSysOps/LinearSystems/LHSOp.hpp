///////////////////////////////////////////////////////////////////////////////
//
// File: LHSOp.hpp
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
// Description: Left-hand side operator of a linear system.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <MultiRegions/AssmbScatr/AssmbScatrOp.hpp>
#include <MultiRegions/BndCondOps/RobBndCond/RobBndCondOp.hpp>
#include <MultiRegions/ElmtOps/ElmtOp.hpp>

namespace Nektar::SolverCore
{

/**
 * @brief Left-hand side operator of a linear system: an element operator
 * plus the Robin boundary contribution, assembled with zero Dirichlet
 * coefficients.
 *
 * The element operator is set with SetElmtOp() and shared with its owner,
 * which keeps setting its parameters directly. The Robin and assembly
 * operators are built here.
 *
 * Apply() returns the assembled product, in the form the preconditioners
 * and the masked inner products expect. ApplyLocal() returns the product
 * before assembly, and Assemble() completes it, for callers that combine
 * the unassembled product with other work first.
 *
 * @tparam TData  Floating-point type of the field data.
 */
template <typename TData> class LHSOp
{
public:
    LHSOp(const MultiRegions::ExpListSharedPtr &expansionList,
          const std::vector<std::string> &components,
          const std::string &execStr = "")
        : m_robBCOp(MultiRegions::RobBndCondOp<TData>::Create(
              expansionList, components, execStr)),
          m_assmbScatrZeroDirOp(
              MultiRegions::AssmbScatrZeroDirOp<TData>::Create(
                  expansionList, components, execStr))
    {
    }

    /// @brief Set the element operator applied before the Robin
    /// contribution.
    void SetElmtOp(const std::shared_ptr<MultiRegions::ElmtOp<
                       FieldState::Coeff, FieldState::Coeff, TData>> &elmtOp)
    {
        m_elmtOp = elmtOp;
    }

    /// @brief Apply the operator and assemble the result with zero
    /// Dirichlet coefficients.
    void Apply(LibUtilities::Field<TData, FieldState::Coeff> &in,
               LibUtilities::Field<TData, FieldState::Coeff> &out)
    {
        ApplyLocal(in, out);
        m_assmbScatrZeroDirOp->Apply(out);
    }

    /// @brief Apply the element operator, then add the Robin contribution,
    /// without assembling the result.
    void ApplyLocal(LibUtilities::Field<TData, FieldState::Coeff> &in,
                    LibUtilities::Field<TData, FieldState::Coeff> &out)
    {
        ASSERTL1(m_elmtOp, "Element operator not set: call SetElmtOp()");

        m_elmtOp->Apply(in, out);
        m_robBCOp->Apply(in, out);
    }

    /// @brief Assemble @p inout in place with zero Dirichlet coefficients.
    void Assemble(LibUtilities::Field<TData, FieldState::Coeff> &inout)
    {
        m_assmbScatrZeroDirOp->Apply(inout);
    }

    /// @brief Assemble @p in with zero Dirichlet coefficients into @p out.
    void Assemble(LibUtilities::Field<TData, FieldState::Coeff> &in,
                  LibUtilities::Field<TData, FieldState::Coeff> &out)
    {
        m_assmbScatrZeroDirOp->Apply(in, out);
    }

protected:
    std::shared_ptr<
        MultiRegions::ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>>
        m_elmtOp;
    std::shared_ptr<MultiRegions::RobBndCondOp<TData>> m_robBCOp;
    std::shared_ptr<MultiRegions::AssmbScatrOp<TData>> m_assmbScatrZeroDirOp;
};

} // namespace Nektar::SolverCore
