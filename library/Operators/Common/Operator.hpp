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

#pragma once

#include <algorithm>
#include <cctype>
#include <string>

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/BasicUtils/NekFactory.hpp>
#include <LibUtilities/BasicUtils/NekInline.hpp>
#include <LibUtilities/SimdLib/tinysimd.hpp>
#include <MultiRegions/ExpList.h>

#include "LibUtilities/BasicUtils/DataWarehouse/NekDataWarehouse.hpp"

#include "Operators/Common/OperatorsDeclspec.hpp"
#include <LibUtilities/Backends/Backends.hpp>
#include <LibUtilities/BasicUtils/Field/Field.hpp>

namespace Nektar::Operators
{

// Forward-declare the Operator base class so we can define the factory
template <typename TData> class Operator;

// Typename alias for the factory
template <typename TData>
using OperatorFactory =
    LibUtilities::NekFactory<std::string, Operator<TData>,
                             const MultiRegions::ExpListSharedPtr &,
                             const std::vector<std::string> &>;

// Operator factory singleton
template <typename TData> OperatorFactory<TData> &GetOperatorFactory();

template <typename TData> class Operator
{
public:
    virtual ~Operator() = default;

    Operator(const MultiRegions::ExpListSharedPtr &expansionList,
             const std::vector<std::string> components)
        : m_expansionList(expansionList), m_components(components),
          m_dataWarehouse(expansionList->GetDataWarehouseSharedPtr())
    {
    }

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
    MultiRegions::ExpListSharedPtr m_expansionList;
    std::vector<std::string> m_components;
    LibUtilities::NekDataWarehouseSharedPtr m_dataWarehouse;
};

} // namespace Nektar::Operators
