///////////////////////////////////////////////////////////////////////////////
//
// File: Operator.hpp
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
 * @file Operator.hpp
 * @brief Whole-field operator base class and factory machinery shared by
 * every operator family in the Operators library.
 *
 * @details
 * Operator fixes no input or output field type; it only provides what
 * every family's interface class needs regardless of what kind of
 * operator it is: the expansion list, component names and data
 * warehouse an instance was built with, the factory lookup performed
 * by Create(), and the execution-space default returned by
 * GetOpExecSpace(). Families that act element by element derive their
 * public interface from ElmtOp (ElmtOp.hpp) instead, which adds the
 * per-block factory stage on top of this class.
 */

#pragma once

#include <algorithm>
#include <cctype>
#include <string>

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/BasicUtils/NekFactory.hpp>
#include <LibUtilities/BasicUtils/NekInline.hpp>
#include <LibUtilities/SimdLib/tinysimd.hpp>
#include <MultiRegions/ExpList.h>
#include <MultiRegions/MultiRegionsDeclspec.h>

#include "LibUtilities/BasicUtils/DataWarehouse/NekDataWarehouse.hpp"

#include <LibUtilities/Backends/Backends.hpp>
#include <LibUtilities/BasicUtils/Field/Field.hpp>

namespace Nektar::MultiRegions
{

// Forward-declare the Operator base class so we can define the factory
template <typename TData> class Operator;

/// @brief Factory of Operator<TData> interface objects, keyed by
/// `TOperator::name + execStr`.
template <typename TData>
using OperatorFactory =
    LibUtilities::NekFactory<std::string, Operator<TData>,
                             const MultiRegions::ExpListSharedPtr &,
                             const std::vector<std::string> &>;

/// @brief Return the process-wide singleton OperatorFactory<TData>.
template <typename TData> OperatorFactory<TData> &GetOperatorFactory();

/**
 * @brief Common base class of the operator family interfaces: holds the
 * expansion list, component names and data warehouse an operator was
 * built with, and provides the family-agnostic Create() and
 * GetOpExecSpace().
 *
 * @tparam TData  Floating-point type of the field data.
 *
 * @see ElmtOp for the element-by-element specialisation that adds the
 * per-block factory stage.
 */
template <typename TData> class Operator
{
public:
    virtual ~Operator() = default;

    /**
     * @brief Construct the interface part of a concrete operator;
     * called by the factory-registered creator functions.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Names of the field components the
     *                          operator is set up for.
     */
    Operator(const MultiRegions::ExpListSharedPtr &expansionList,
             const std::vector<std::string> components)
        : m_expansionList(expansionList), m_components(components),
          m_dataWarehouse(expansionList->GetDataWarehouseSharedPtr())
    {
    }

    /**
     * @brief Create a concrete operator instance through the operator
     * factory.
     *
     * The factory key is `TOperator::name + execStr`, where an empty
     * @p execStr is resolved by GetOpExecSpace(). The product is
     * handed back through a static_pointer_cast to @p TOperator, so
     * the creator registered under the key must construct that class
     * or one derived from it.
     *
     * @tparam TOperator  Family interface class supplying the static
     *                    `name` string.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Names of the field components the
     *                          operator is set up for.
     * @param   execStr         Execution space ("Serial", "AVX" or
     *                          "Device"); if empty, GetOpExecSpace()
     *                          supplies it from the session's
     *                          "opExecSpace" command-line argument or,
     *                          failing that, the build's default, with
     *                          a warning.
     *
     * @return The newly created operator. Creation raises a fatal
     * error (throws ErrorUtil::NekError) if no implementation is
     * registered under the requested key.
     */
    template <template <typename> typename TOperator>
    static std::shared_ptr<TOperator<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> components, const std::string &execStr)
    {
        auto session = expansionList->GetSession();

        std::string execStr0 = (execStr == "")
                                   ? Operator<TData>::GetOpExecSpace(session)
                                   : execStr;

        std::string requestedKey = TOperator<TData>::name + execStr0;

        OperatorFactory<TData> &factory = GetOperatorFactory<TData>();

        // No suitable operator was found.
        if (!factory.ModuleExists(requestedKey))
        {
            std::stringstream msg;
            msg << "No such operator: " << requestedKey << std::endl;
            factory.PrintAvailableClasses(msg);
            NEKERROR(ErrorUtil::efatal, msg.str());
        }

        return std::static_pointer_cast<TOperator<TData>>(
            factory.CreateInstance(requestedKey, expansionList, components));
    }

    /**
     * @brief Return the execution space name ("opExecSpace") for an
     * operator.
     *
     * The space comes from the command-line argument "opExecSpace". Without
     * it the operator takes the space the build provides, "Device" when a
     * device is present and otherwise "AVX" or "Serial", and warns that it
     * has done so.
     *
     * @param session  Session reader to recover the relevant command-line
     * argument.
     *
     * @return std::string containing the implementation name (e.g. "Serial",
     * "Device").
     */
    static std::string GetOpExecSpace(
        std::shared_ptr<LibUtilities::SessionReader> session)
    {
        std::string OpExec = (nekGetNumDevice()) ? "Device" :
#if defined(NEKTAR_ENABLE_SIMD)
                                                 "AVX";
#else
                                                 "Serial";
#endif

        if (session->DefinesCmdLineArgument("opExecSpace"))
        {
            OpExec = session->GetCmdLineArgument<std::string>("opExecSpace");
        }
        else
        {
            NEKERROR(ErrorUtil::ewarning,
                     "No execution space specified for operator. Defaulting "
                     "to " +
                         OpExec);
        }
        return OpExec;
    }

    /**
     * @brief Return the default interleave width that persistent Fields
     * (e.g. solution storage, time-integration and linear-solver workspace)
     * should be constructed with for the given session.
     *
     * The SumFac implementation on the Device backend (and any AVX-backed
     * execution space) natively operates on interleaved data, so Fields are
     * pre-interleaved to that native width to avoid repeated reshape
     * operations. All other configurations use a trivial width of 1.
     *
     * @param session  Session reader to recover the relevant command-line
     * arguments.
     *
     * @return unsigned int interleave width.
     */
    static unsigned int GetDefaultInterleaveWidth(
        std::shared_ptr<LibUtilities::SessionReader> session)
    {
        // Discontinuous (DG) projections default to interleavewidth = 1
        // as the current DG operator implementations cannot use
        // interleaving and hence would cause a slowdown
        // due to many interleaving/deinterleaving calls.
        if (IsDiscontinuousProjection(session))
        {
            return 1u;
        }

        std::string execName = GetOpExecSpace(session);
        std::string implName =
            session->DefinesCmdLineArgument("opImpl")
                ? session->GetCmdLineArgument<std::string>("opImpl")
                : "";

        return (implName == "SumFac" || execName == "AVX")
                   ? NektarSpaces::GetVectorWidth<TData>(execName)
                   : 1u;
    }

    /**
     * @brief Return true if the session uses a Discontinuous (DG) projection.
     *
     * @param session  Session reader to recover the relevant SolverInfo.
     *
     * @return bool true if the "Projection" SolverInfo is "DG" or
     * "Discontinuous".
     */
    static bool IsDiscontinuousProjection(
        std::shared_ptr<LibUtilities::SessionReader> session)
    {
        if (!session->DefinesSolverInfo("Projection"))
        {
            return false;
        }

        std::string projection = session->GetSolverInfo("Projection");
        std::transform(projection.begin(), projection.end(), projection.begin(),
                       [](unsigned char c) { return std::toupper(c); });
        return projection == "DG" || projection == "DISCONTINUOUS";
    }

protected:
    /// Expansion list the operator acts on.
    MultiRegions::ExpListSharedPtr m_expansionList;
    /// Names of the field components the operator was built for.
    std::vector<std::string> m_components;
    /// Data warehouse shared with the other operators on the expansion
    /// list, used to build and cache shared tables (basis matrices,
    /// mode indices, geometric factors) once.
    LibUtilities::NekDataWarehouseSharedPtr m_dataWarehouse;
};

} // namespace Nektar::MultiRegions
