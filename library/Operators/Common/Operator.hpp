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

#include <string>

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/BasicUtils/NekFactory.hpp>
#include <LibUtilities/BasicUtils/NekInline.hpp>
#include <LibUtilities/SimdLib/tinysimd.hpp>
#include <MultiRegions/ExpList.h>

#include "Operators/Common/DataWarehouse/NekDataWarehouse.hpp"

#include "Operators/Common/OperatorsDeclspec.hpp"
#include "Operators/Common/Spaces.hpp"
#include "Operators/Field/Field.hpp"

namespace Nektar::Operators
{

// Forward-declare the Operator base class so we can define the factory
template <typename TData> class Operator;

// Typename alias for the factory
template <typename TData>
using OperatorFactory =
    Nektar::LibUtilities::NekFactory<std::string, Operator<TData>,
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
     * This function returns the execution space selected for a given operator
     * from the command-line argument "opExecSpace".
     *
     * otherwise ASSERTL1 is triggered.
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
        if (!session->DefinesCmdLineArgument("opExecSpace"))
        {
            NEKERROR(ErrorUtil::efatal,
                     "No execution space specified for operator. Please "
                     "specify using the command-line argument 'opExecSpace'.");
        }
        return session->GetCmdLineArgument<std::string>("opExecSpace");
    }

protected:
    MultiRegions::ExpListSharedPtr m_expansionList;
    std::vector<std::string> m_components;
    NekDataWarehouseSharedPtr m_dataWarehouse;
};

} // namespace Nektar::Operators
