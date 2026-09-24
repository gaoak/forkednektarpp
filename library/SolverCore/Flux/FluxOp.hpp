///////////////////////////////////////////////////////////////////////////////
//
// File: FluxOp.hpp
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
// Description: Flux operator factory base class.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Operators/Common/Operator.hpp"

namespace Nektar::SolverCore
{

// Flux operator factory base class.
template <typename TData> class FluxOp : public Operators::Operator<TData>
{
public:
    static std::shared_ptr<FluxOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &method = "", const std::string &execStr = "")
    {
        auto session = expansionList->GetSession();

        std::string method0 = method;

        std::string execStr0 =
            (execStr == "")
                ? Operators::Operator<TData>::GetOpExecSpace(session)
                : execStr;

        std::string requestedKey = method0 + execStr0;

        Operators::OperatorFactory<TData> &factory =
            Operators::GetOperatorFactory<TData>();

        // No suitable operator was found.
        if (!factory.ModuleExists(requestedKey))
        {
            std::stringstream msg;
            msg << "No such operator: " << requestedKey << std::endl;
            factory.PrintAvailableClasses(msg);
            NEKERROR(ErrorUtil::efatal, msg.str());
        }

        return std::static_pointer_cast<FluxOp<TData>>(
            factory.CreateInstance(requestedKey, expansionList, components));
    }

    void SetScale(const TData &scale)
    {
        m_scale = scale;
    }

    /// Select whether Apply() overwrites its output or accumulates onto it.
    void SetAppend(const bool &append)
    {
        m_append = append;
    }

protected:
    TData m_scale = 1.0;
    bool m_append = false;

    FluxOp(const MultiRegions::ExpListSharedPtr &expansionList,
           const std::vector<std::string> &components)
        : Operators::Operator<TData>(expansionList, components)
    {
    }

    ~FluxOp() override = default;
};

} // namespace Nektar::SolverCore
