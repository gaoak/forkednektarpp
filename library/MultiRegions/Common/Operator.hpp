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
 * every operator family in the MultiRegions library.
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

#include <string>
#include <vector>

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/BasicUtils/NekFactory.hpp>
#include <LibUtilities/BasicUtils/NekInline.hpp>
#include <LibUtilities/BasicUtils/StringUtils.hpp>
#include <LibUtilities/SimdLib/tinysimd.hpp>
#include <MultiRegions/ExpList.h>
#include <MultiRegions/MultiRegionsDeclspec.h>

#include <LibUtilities/BasicUtils/DataWarehouse/NekDataWarehouse.hpp>

#include <LibUtilities/Backends/Backends.hpp>
#include <LibUtilities/Backends/DeviceProperties.hpp>
#include <LibUtilities/BasicUtils/Field/Field.hpp>
#include <MultiRegions/Common/DefaultOpImplTable.hpp>

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
     * device is present and otherwise "AVX" or "Serial", and, if @p warn,
     * warns that it has done so.
     *
     * @param session  Session reader to recover the relevant command-line
     * argument.
     * @param warn     Whether to warn when the space is defaulted.
     *
     * @return std::string containing the implementation name (e.g. "Serial",
     * "Device").
     */
    static std::string GetOpExecSpace(
        std::shared_ptr<LibUtilities::SessionReader> session,
        const bool warn = true)
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
        else if (warn)
        {
            NEKERROR(ErrorUtil::ewarning,
                     "No execution space specified for operator. Defaulting "
                     "to " +
                         OpExec);
        }
        return OpExec;
    }

    /**
     * @brief Return the row of the default implementation table
     * (DefaultOpImplTable.hpp) that a block of elements takes, or null if
     * none does.
     *
     * @param exp        The block's representative element, whose shape and
     *                   highest order over its directions are used, or null.
     * @param execSpace  Execution space ("Serial", "AVX" or "Device").
     */
    static const DefaultOpImplRule *GetDefaultOpImplRule(
        const StdRegions::StdExpansionSharedPtr &exp,
        const std::string &execSpace)
    {
        const DefaultOpImplRule *rule = nullptr;
        if (exp)
        {
            rule = FindDefaultOpImplRule(
                execSpace, execSpace == "Device" ? GetDeviceArchitecture() : "",
                exp->DetShapeType(), exp->EvalBasisNumModesMax() - 1);
        }
        return rule;
    }

    /**
     * @brief Return the implementation a block of elements takes when
     * neither the command line nor the session names one for its operator.
     *
     * This is the implementation of the block's row of the default
     * implementation table (DefaultOpImplTable.hpp), by execution space,
     * device architecture, element shape and expansion order, and SumFac
     * where no row applies or without an expansion. On Device the table
     * currently gives SumFacTOP from order three in 3D and order six in 1D
     * and 2D. The default is taken per block, so the blocks of a mesh of
     * mixed order or shape can take different implementations.
     *
     * @param exp        The block's representative element, or null.
     * @param execSpace  Execution space ("Serial", "AVX" or "Device").
     *
     * @return std::string containing the implementation name.
     */
    static std::string GetDefaultOpImpl(
        const StdRegions::StdExpansionSharedPtr &exp,
        const std::string &execSpace)
    {
        const DefaultOpImplRule *rule = GetDefaultOpImplRule(exp, execSpace);
        return rule ? std::string(rule->opImpl) : "SumFac";
    }

    /**
     * @brief Return the default interleave widths, one per block of
     * @p expansionList, that persistent Fields (e.g. solution storage,
     * time-integration and linear-solver workspace) built on it should be
     * constructed with.
     *
     * A block takes the vector width of the session's execution space when
     * its implementation is SumFac or the execution space is AVX, so that
     * Fields are created at that width rather than reshaped on every
     * Apply; 1 otherwise. The implementation is the command line's
     * "opImpl" or, without it, the block's default of GetDefaultOpImpl(),
     * so that each block matches the operators' default for it; a session
     * that names implementations per operator (a backend section) leaves
     * that default out, since one width cannot follow a mix. A DG
     * projection takes the same interleave widths as a continuous one: the
     * trace operators read a field at whatever width each of its blocks
     * carries.
     *
     * @param expansionList  Expansion list the fields are built on; its
     *                       session supplies the command-line arguments
     *                       and the backend sections.
     *
     * @return std::vector<unsigned int> of interleave widths, one per
     * block, for GetBlockAttributes().
     */
    static std::vector<unsigned int> GetDefaultInterleaveWidths(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        const auto session = expansionList->GetSession();
        std::vector<unsigned int> interleave_widths;
        if (!session)
        {
            // Without a session there is no execution space to follow.
            interleave_widths.assign(GetCollections(expansionList).size(), 1u);
        }
        else
        {
            const std::string execName = GetOpExecSpace(session, false);

            const LibUtilities::BackendMap &backend =
                session->GetBackendMap(execName);
            const bool cmdLineImpl = session->DefinesCmdLineArgument("opImpl");
            const std::string cmdLineImplName =
                cmdLineImpl ? session->GetCmdLineArgument<std::string>("opImpl")
                            : "";

            for (auto &coll : GetCollections(expansionList))
            {
                const std::string implName =
                    cmdLineImpl ? cmdLineImplName
                    : backend.empty()
                        ? GetDefaultOpImpl(coll.GetExpVector()[0], execName)
                        : ""; // "" for now as different block operators might
                              // use different implementations
                interleave_widths.push_back(
                    (implName == "SumFac" || execName == "AVX")
                        ? NektarSpaces::GetVectorWidth<TData>(execName)
                        : 1u);
            }
        }
        return interleave_widths;
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

        std::string projection =
            StringUtils::ToUpperCopy(session->GetSolverInfo("Projection"));
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
