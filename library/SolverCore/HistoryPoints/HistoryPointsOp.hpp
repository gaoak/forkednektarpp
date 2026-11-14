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
 * the packed MPI reduction, which collects the point values, execute in the
 * selected operator execution space.
 *
 * After Apply(), GetValues() returns the point values point-major: with
 * @c nComp the number of components the operator was created with, the value
 * of component @c comp at point @c point, numbered in the order passed to
 * Configure(), is entry <tt>point * nComp + comp</tt>, for a size of the
 * number of points times @c nComp. m_result holds them in the same layout.
 *
 * With a homogeneous direction (3DH1) a point is located in the 2D mesh by
 * its x and y, and its value is the trigonometric interpolation at its z of
 * its values on the planes. The ranks holding its element each add the share
 * of their own planes, and the reduction sums them.
 *
 * Apply() is collective: every rank must call it. GetValues() reads the
 * reduced values on the host, on every rank, or on rank zero only after
 * SetRootOnly(true), which collects them on rank zero alone.
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

    /// Reduce the results of the next Apply() to rank zero alone if
    /// @p rootOnly, else to every rank.
    void SetRootOnly(const bool rootOnly)
    {
        m_rootOnly = rootOnly;
    }

    /// Point values from the last Apply(), point-major.
    std::vector<TData> GetValues()
    {
        ASSERTL0(!m_rootOnly || this->m_expansionList->GetComm()
                                    ->GetSpaceComm()
                                    ->TreatAsRankZero(),
                 "HistoryPointsOp reduced its results to rank zero only.");
        return m_result.ToVector();
    }

protected:
    /// Whether Apply() reduces the results to rank zero alone.
    bool m_rootOnly = false;
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
