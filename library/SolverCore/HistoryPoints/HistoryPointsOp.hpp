///////////////////////////////////////////////////////////////////////////////
//
// File: HistoryPointsOp.hpp
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
// Description: Operator interface evaluating physical fields at a set of
// points.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <array>
#include <vector>

#include <MultiRegions/Common/Operator.hpp>
#include <SolverCore/SolverCore.hpp>

namespace Nektar::SolverCore
{

/**
 * @brief Evaluate physical fields at a fixed set of global points.
 *
 * Point ownership and interpolation data are configured once. Evaluation and
 * the packed MPI reduction execute in the selected operator execution space;
 * only the point values needed for text output are copied to rank zero.
 *
 * After Apply(), GetValues() returns the point values point-major: with
 * @c nComp the number of components the operator was created with, the value
 * of component @c comp at point @c point, numbered in the order passed to
 * Configure(), is entry <tt>point * nComp + comp</tt>, for a size of the
 * number of points times @c nComp. m_result holds them in the same layout.
 *
 * Apply() is collective: every rank must call it. GetValues() reads the
 * reduced values on the host and is valid on rank zero only, the other ranks
 * holding their own local contributions.
 */
template <typename TData>
class HistoryPointsOp : public MultiRegions::Operator<TData>
{
public:
    static std::shared_ptr<HistoryPointsOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "")
    {
        EnsureLinked();
        return MultiRegions::Operator<TData>::template Create<HistoryPointsOp>(
            expansionList, components, execStr);
    }

    static inline const std::string name = "HistoryPoints";

    /// Set the global points at which Apply() evaluates the field, which also
    /// fixes the number of values GetValues() returns.
    void Configure(const std::vector<std::array<TData, 3>> &points)
    {
        v_Configure(points);
    }

    /// Collective: every rank must call it.
    void Apply(LibUtilities::Field<TData, FieldState::Phys> &field)
    {
        v_Apply(field);
    }

    /// Collective: as Apply().
    void operator()(LibUtilities::Field<TData, FieldState::Phys> &field)
    {
        v_Apply(field);
    }

    /// Point values from the last Apply(), point-major, valid on rank zero
    /// only.
    std::vector<TData> GetValues()
    {
        const auto result =
            m_result.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        return std::vector<TData>(result, result + m_result.size());
    }

protected:
    /// Reduced result of Apply(), allocated by Configure().
    LibUtilities::MemoryRegion<TData> m_result;

    HistoryPointsOp(const MultiRegions::ExpListSharedPtr &expansionList,
                    const std::vector<std::string> &components)
        : MultiRegions::Operator<TData>(expansionList, components)
    {
    }

    ~HistoryPointsOp() override = default;

    virtual void v_Configure(
        const std::vector<std::array<TData, 3>> &points) = 0;
    virtual void v_Apply(
        LibUtilities::Field<TData, FieldState::Phys> &field) = 0;
};

} // namespace Nektar::SolverCore
