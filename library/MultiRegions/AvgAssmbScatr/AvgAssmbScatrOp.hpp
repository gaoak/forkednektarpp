///////////////////////////////////////////////////////////////////////////////
//
// File: AvgAssmbScatrOp.hpp
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
// Description: Public interface of the averaging assemble-scatter operator,
// which replaces every local copy of each global degree of freedom of a
// coefficient field by the signed average of all its copies.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <MultiRegions/Common/Operator.hpp>

namespace Nektar::MultiRegions
{

/**
 * @brief Base class for the averaging assemble-scatter operator: replaces
 * every local copy of each global degree of freedom by the signed average
 * of all its copies.
 *
 * @details
 * Each local coefficient becomes
 * \f$x_i \leftarrow s_i \sum_j s_j x_j / m\f$, where \f$s\f$ are the
 * orientation signs of the local-to-global assembly map and \f$m\f$ is the
 * number of copies of the degree of freedom with a nonzero sign, counted
 * over every rank and every periodic partner. A coefficient of sign zero,
 * as an unused mode of an edge or face shared with an element of lower
 * order is in a variable-order expansion, is set to zero. The result is
 * continuous, and a continuous field is left unchanged. For a ContField
 * this is AssemblyMap::AvgAssemble() followed by
 * AssemblyMap::GlobalToLocal().
 *
 * The averaging is the sum formed by AssmbScatrOp, weighted by the inverse
 * multiplicity, which the implementation (detail::AvgAssmbScatrOpImpl)
 * builds once, when the operator is created, and reuses on every Apply().
 * Construction and Apply() both communicate in parallel runs, so every rank
 * must construct the operator and call Apply() together.
 *
 * @tparam TData Floating-point type of the coefficient data.
 */
template <typename TData> class AvgAssmbScatrOp : public Operator<TData>
{
public:
    ~AvgAssmbScatrOp() override = default;

    /**
     * @brief Create an averaging assemble-scatter operator through the
     * operator factory.
     *
     * @param   expansionList   Expansion list the operator acts on; in
     *                          parallel runs a MultiRegions::ContField.
     * @param   components      Session variable names of the field
     *                          components the operator is built for.
     * @param   execStr         Execution-space suffix, e.g. "Serial",
     *                          "AVX" or "Device". If empty, it is taken
     *                          from the "opExecSpace" command-line
     *                          argument; without it the operator takes the
     *                          space the build provides and warns.
     *
     * @return The operator implementation for the requested execution
     * space. Creation raises a fatal error if none is registered.
     */
    static std::shared_ptr<AvgAssmbScatrOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "")
    {
        return Operator<TData>::template Create<AvgAssmbScatrOp>(
            expansionList, components, execStr);
    }

    /// Operator base name; Create() appends the execution space to it to
    /// form the factory key.
    static inline const std::string name = "AvgAssmbScatr";

    /**
     * @brief Average the local copies of @p inout in place.
     *
     * @param   inout   Coefficient-space field with the components the
     *                  operator was created for; averaged in place, at the
     *                  interleave width it has on entry.
     */
    void Apply(LibUtilities::Field<TData, FieldState::Coeff> &inout)
    {
        v_Apply(inout);
    }

    /**
     * @brief Average the local copies of @p in, writing the result to
     * @p out.
     *
     * @param   in      Coefficient-space input field with the components
     *                  the operator was created for; left unmodified unless
     *                  it is the same object as @p out.
     * @param   out     Coefficient-space output field of the same size as
     *                  @p in; on return it holds the average at the
     *                  interleave width of @p in.
     */
    void Apply(LibUtilities::Field<TData, FieldState::Coeff> &in,
               LibUtilities::Field<TData, FieldState::Coeff> &out)
    {
        v_Apply(in, out);
    }

    /// @brief Call operator; equivalent to the in-place Apply().
    void operator()(LibUtilities::Field<TData, FieldState::Coeff> &inout)
    {
        v_Apply(inout);
    }

    /// @brief Call operator; equivalent to the two-field Apply().
    void operator()(LibUtilities::Field<TData, FieldState::Coeff> &in,
                    LibUtilities::Field<TData, FieldState::Coeff> &out)
    {
        v_Apply(in, out);
    }

protected:
    /// @brief Construct the interface part of a concrete implementation;
    /// called by the factory-registered creator functions.
    AvgAssmbScatrOp(const MultiRegions::ExpListSharedPtr &expansionList,
                    const std::vector<std::string> &components)
        : Operator<TData>(expansionList, components)
    {
    }

    /// @brief Implementation hook for the in-place Apply().
    virtual void v_Apply(
        LibUtilities::Field<TData, FieldState::Coeff> &inout) = 0;

    /// @brief Implementation hook for the two-field Apply().
    virtual void v_Apply(
        LibUtilities::Field<TData, FieldState::Coeff> &in,
        LibUtilities::Field<TData, FieldState::Coeff> &out) = 0;
};

} // namespace Nektar::MultiRegions
