///////////////////////////////////////////////////////////////////////////////
//
// File: AssmbScatrOp.hpp
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

/**
 * @file AssmbScatrOp.hpp
 * @brief Public interface for the assemble-scatter operator, which makes
 * element-wise coefficient storage globally consistent, and for its
 * Dirichlet-zeroing variant.
 *
 * @details
 * A coefficient-space Field stores each element's coefficients
 * separately, so a global degree of freedom shared by several elements
 * (or, in parallel, by several ranks) exists as multiple local copies.
 * Solvers for continuous (C0) discretisations repeatedly need those
 * copies summed and made consistent, which is done in two phases:
 *
 * - **Assemble**: sum the local contributions to every global degree
 *   of freedom, each weighted by its orientation sign;
 * - **Scatter**: write the summed value, weighted by each copy's
 *   orientation sign, back to every local copy.
 *
 * AssmbScatrOp is the abstract, execution-space-agnostic interface to
 * this operation. Concrete implementations are provided by
 * detail::AssmbScatrOpImpl (AssmbScatrOpImpl.hpp), built on the kernels
 * in AssmbScatrDeviceKernels.hpp and AssmbScatrSerialAVXKernels.hpp,
 * and are registered with the operator factory by translation units
 * that CMake generates from AssmbScatrFactoryDec.cpp.in. When run in
 * parallel, the implementation additionally exchanges partition-
 * boundary contributions between ranks, overlapping that communication
 * with the assembly of the purely interior degrees of freedom.
 *
 * AssmbScatrZeroDirOp, declared below, is the factory-facing class of
 * the variant whose result has the global Dirichlet boundary degrees of
 * freedom zeroed; its implementations are registered from the same
 * generated translation units.
 */

#pragma once

#include <MultiRegions/ContField.h>

#include "Operators/Common/Operator.hpp"

namespace Nektar::Operators
{

/**
 * @brief Base class for the assemble-scatter operator: sums the local
 * contributions to every global degree of freedom and writes the sum
 * back to each contributing local coefficient.
 *
 * The operator acts on local (element-wise) coefficient storage. After
 * Apply(), every local copy of a shared degree of freedom holds the
 * fully assembled value in its own sign convention. In the
 * factory-registered implementations each contribution and each
 * write-back is weighted by the orientation sign defined in the
 * local-to-global assembly map; for a ContField the behaviour is then
 * equivalent to AssemblyMap::Assemble() followed by
 * AssemblyMap::GlobalToLocal(). Overrides may adopt a different sign
 * convention: the in-tree detail::AssmbScatrNoSignOpImpl sums with
 * every orientation sign replaced by its absolute value.
 *
 * As with the other operator base classes in this library, this class
 * exists to fix the Apply() parameter types: both overloads accept
 * only coefficient-space fields (FieldState::Coeff), enforced at
 * compile time. The per-execution-space implementations
 * (detail::AssmbScatrOpImpl) override v_Apply() and are obtained
 * through Create(). The derived AssmbScatrZeroDirOp performs the same
 * operation but with the global Dirichlet boundary degrees of freedom
 * zeroed.
 *
 * @tparam TData Floating-point type of the field data.
 */
template <typename TData> class AssmbScatrOp : public Operator<TData>
{
public:
    ~AssmbScatrOp() override = default;

    /**
     * @brief Create an assemble-scatter operator through the operator
     * factory.
     *
     * The factory key is #name followed by the execution-space string,
     * so this returns whichever implementation was registered for that
     * execution space (e.g. "Serial", "AVX", "Device").
     *
     * @param   expansionList   Expansion list the operator acts on. In
     *                          parallel runs this must be a
     *                          MultiRegions::ContField, since the
     *                          implementation derives its inter-rank
     *                          assembly communication from the field's
     *                          assembly map.
     * @param   components      Names of the field components (solution
     *                          variables); used to key the assembly-map
     *                          data cached in the data warehouse.
     * @param   execStr         Execution-space part of the factory key.
     *                          If empty, the session's "opExecSpace"
     *                          command-line argument is used instead;
     *                          without it the operator takes the space
     *                          the build provides and warns.
     *
     * @return The newly created operator. Creation raises a fatal
     * error (throws ErrorUtil::NekError) if no implementation is
     * registered under the requested key.
     */
    static std::shared_ptr<AssmbScatrOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "")
    {
        return Operator<TData>::template Create<AssmbScatrOp>(
            expansionList, components, execStr);
    }

    /// Operator base name; Create() appends the execution space to it
    /// to form the factory key.
    static inline const std::string name = "AssmbScatr";

    /**
     * @brief Assemble and scatter @p inout in place.
     *
     * On return every local coefficient of @p inout holds the assembled
     * sum of all local contributions to its global degree of freedom,
     * weighted by its own orientation sign under the implementation's
     * sign convention (see the class description).
     *
     * @param   inout   Coefficient-space field, overwritten with its
     *                  assembled values.
     */
    void Apply(LibUtilities::Field<TData, FieldState::Coeff> &inout)
    {
        v_Apply(inout);
    }

    /**
     * @brief Assemble and scatter @p in, writing the result to @p out.
     *
     * The two fields must have the same block sizes and number of
     * components. They may refer to the same object, in which case the
     * operation is performed in place.
     *
     * @param   in      Coefficient-space input field; left unmodified
     *                  unless it is the same object as @p out.
     * @param   out     Coefficient-space output field receiving the
     *                  assembled values.
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
    /**
     * @brief Construct the interface part of a concrete implementation;
     * called by the factory-registered creator functions.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Component names the operator is set up
     *                          for.
     */
    AssmbScatrOp(const MultiRegions::ExpListSharedPtr &expansionList,
                 const std::vector<std::string> &components)
        : Operator<TData>(expansionList, components)
    {
    }

    /**
     * @brief Implementation hook for the in-place Apply().
     *
     * Overrides must replace every local coefficient of @p inout, for
     * every component, with the sum of all local contributions to its
     * global degree of freedom, weighted by that coefficient's own
     * orientation sign under whichever sign convention the override
     * implements (see the class description), and must restore the
     * field's interleaved storage layout if they change it internally.
     *
     * @param   inout   Coefficient-space field to assemble and scatter
     *                  in place.
     */
    virtual void v_Apply(
        LibUtilities::Field<TData, FieldState::Coeff> &inout) = 0;

    /**
     * @brief Implementation hook for the two-field Apply().
     *
     * Overrides must write the assembled and scattered form of @p in
     * to @p out, accept @p in and @p out referring to the same field,
     * and leave @p in unmodified when the two are distinct.
     *
     * @param   in      Coefficient-space input field.
     * @param   out     Coefficient-space output field.
     */
    virtual void v_Apply(
        LibUtilities::Field<TData, FieldState::Coeff> &in,
        LibUtilities::Field<TData, FieldState::Coeff> &out) = 0;
};

/**
 * @brief Assemble-scatter operator that also zeroes the global Dirichlet
 * boundary degrees of freedom.
 *
 * Implementations perform the same in-place assemble-scatter as
 * AssmbScatrOp, whose Apply() overloads this class inherits: for every
 * global degree of freedom (DOF), all local coefficients mapping to it
 * are summed with the +-1 sign factors held by the assembly map, and the
 * sum is written back to each of them, weighted by each copy's
 * orientation sign. When running on more than one rank,
 * partition-boundary contributions are exchanged as well.
 *
 * The difference lies in the global Dirichlet boundary DOFs (per
 * component, the leading AssemblyMapCG::GetNumGlobalDirBndCoeffs()
 * entries of the global numbering): their assembly-table entries carry a
 * sign of zero, so their contributions are discarded and every local
 * coefficient mapping to such a DOF is overwritten with zero. The result
 * is equivalent to AssemblyMap::Assemble(), zeroing the Dirichlet block
 * of the global vector, then AssemblyMap::GlobalToLocal().
 *
 * A plain assemble-scatter is the identity on DOFs with a single local
 * contribution, so AssmbScatrOp's tables omit them; this variant's tables
 * additionally include single-contribution Dirichlet DOFs so that they
 * are zeroed too. The ZERODIR flag in the data-warehouse keys selects
 * these enlarged tables (see detail::AssmbScatrOpImpl::SetUpMaps).
 *
 * The class itself carries no implementation: it fixes the factory name
 * under which detail::AssmbScatrZeroDirOpImpl (AssmbScatrOpImpl with
 * ZERODIR = true) is registered from AssmbScatrFactoryDec.cpp.in. The
 * iterative solvers in GlobalLinSysOps construct that implementation
 * directly and apply it to their residuals and other Krylov work vectors.
 *
 * @note detail::AssmbScatrZeroDirOpImpl derives from AssmbScatrOp, not
 * from this class, while Operator::Create() hands the factory's product
 * back through a static_pointer_cast to this class. The cast is not
 * backed by the object's actual type; it works because this class adds
 * no data members or virtual functions to AssmbScatrOp, so the inherited
 * Apply() overloads still dispatch to the implementation's v_Apply().
 *
 * @tparam TData Floating-point type of the coefficient data.
 *
 * @see AssmbScatrOp for the variant that assembles Dirichlet DOFs like
 * any other DOF (no zeroing).
 */
template <typename TData> class AssmbScatrZeroDirOp : public AssmbScatrOp<TData>
{
public:
    /**
     * @brief Create an implementation of this operator through the
     * operator factory.
     *
     * Looks up the key formed from #name plus the execution-space suffix
     * and instantiates whatever implementation was registered under it
     * (see AssmbScatrFactoryDec.cpp.in).
     *
     * @param   expansionList  Expansion list the operator acts on. In
     *                         parallel runs this must be a
     *                         MultiRegions::ContField, since the
     *                         implementation derives its inter-rank
     *                         assembly communication from the field's
     *                         assembly map.
     * @param   components     Session variable names the assembly maps
     *                         are built for, one per field component.
     * @param   execStr        Execution-space suffix, e.g. "Serial",
     *                         "AVX" or "Device". If empty, it is taken
     *                         from the "opExecSpace" command-line
     *                         argument; without it the operator takes
     *                         the space the build provides and warns.
     *
     * @return  The operator implementation for the requested execution
     *          space.
     */
    static std::shared_ptr<AssmbScatrZeroDirOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "")
    {
        return Operator<TData>::template Create<AssmbScatrZeroDirOp>(
            expansionList, components, execStr);
    }

    /// Factory base name; implementations register under this name with an
    /// execution-space suffix appended (e.g. "AssmbScatrZeroDirSerial").
    static inline const std::string name = "AssmbScatrZeroDir";

protected:
    /// @brief Protected constructor; forwards the expansion list and
    /// component names to AssmbScatrOp.
    AssmbScatrZeroDirOp(const MultiRegions::ExpListSharedPtr &expansionList,
                        const std::vector<std::string> &components)
        : AssmbScatrOp<TData>(expansionList, components)
    {
    }

    ~AssmbScatrZeroDirOp() override = default;
};

} // namespace Nektar::Operators
