///////////////////////////////////////////////////////////////////////////////
//
// File: ElmtOp.hpp
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

#include "Operators/Common/Operator.hpp"

namespace Nektar::Operators
{

template <FieldState TFieldIn, FieldState TFieldOut, typename TData>
class ElmtOp : public Operator<TData>
{
public:
    ~ElmtOp() override = default;

    template <template <typename> typename TOperator,
              template <typename> typename TBlockOperator>
    static std::shared_ptr<TOperator<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "", const std::string &implStr = "")
    {
        auto session = expansionList->GetSession();

        std::string execStr0 = (execStr == "")
                                   ? Operator<TData>::GetOpExecSpace(session)
                                   : execStr;

        std::string implStr0 =
            (implStr == "")
                ? GetOpImpl(TOperator<TData>::name, execStr0, session)
                : implStr;

        auto op = Operator<TData>::template Create<TOperator>(
            expansionList, components, execStr0);

        auto blockAttr =
            MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                expansionList);

        // Loop over the blocks.
        for (unsigned int block_idx = 0; block_idx < blockAttr.size();
             block_idx++)
        {
            const auto exp =
                MultiRegions::GetCollection(expansionList, block_idx)
                    .GetExpVector()[0];
            op->m_blockOp.push_back(TBlockOperator<TData>::Create(
                block_idx, exp, expansionList->GetDataWarehouseSharedPtr(),
                execStr0, implStr0));
        }

        return op;
    }

    void Apply(LibUtilities::Field<TData, TFieldIn> &in,
               LibUtilities::Field<TData, TFieldOut> &out)
    {
        v_Apply(in, out);
    }

    void operator()(LibUtilities::Field<TData, TFieldIn> &in,
                    LibUtilities::Field<TData, TFieldOut> &out)
    {
        v_Apply(in, out);
    }

    /**
     * @brief Return the implementation for an operator.
     *
     * This looks up the implementation (e.g. "StdMat", "SumFac") for a
     * given operator name and session from maps populated by
     * ParseOptimisations().
     *
     * Lookup rules:
     * - The command-line argument "opImpl" overrides everything.
     * - Otherwise the map for @p opExecSpace is consulted:
     *   m_serialBackendInfo for "Serial", m_avxBackendInfo for "AVX" and
     *   deviceBackendInfo for "Device". The map keys use the UPPERCASE of
     *   @p opName (e.g. "Mass" -> "MASS").
     * - An operator named in neither takes "SumFac" and warns that it has
     *   done so.
     *
     * @param opName  Operator name (e.g. "Mass", "Helmholtz").
     * @param session  Session reader to recover the relevant the maps and
     * command-line arguments.
     *
     * @return std::string containing the implementation name (e.g. "SumFac",
     * "StdMat").
     */
    static std::string GetOpImpl(const std::string &opName,
                                 const std::string &opExecSpace,
                                 LibUtilities::SessionReaderSharedPtr session)
    {
        ASSERTL0(
            opExecSpace == "Serial" || opExecSpace == "AVX" ||
                opExecSpace == "Device",
            "Operator execution space must be 'Serial', 'AVX', or, 'Device'");

        std::string OpImpl = "SumFac";

        // command line overrides everything
        if (session->DefinesCmdLineArgument("opImpl"))
        {
            OpImpl = session->GetCmdLineArgument<std::string>("opImpl");
        }
        else
        {
            bool OpImplSet = false;

            // Keys are stored uppercased by ParseOptimisations().
            const std::string opNameUpper = boost::to_upper_copy(opName);
            if (opExecSpace == "Serial")
            {
                LibUtilities::BackendMap &serialBackendInfo =
                    session->GetSerialBackendMap();
                auto opImplIter = serialBackendInfo.find(opNameUpper);
                if (opImplIter != serialBackendInfo.end())
                {
                    OpImpl    = opImplIter->second;
                    OpImplSet = true;
                }
            }
            else if (opExecSpace == "AVX")
            {
                LibUtilities::BackendMap &avxBackendInfo =
                    session->GetAVXBackendMap();
                auto opImplIter = avxBackendInfo.find(opNameUpper);
                if (opImplIter != avxBackendInfo.end())
                {
                    OpImpl    = opImplIter->second;
                    OpImplSet = true;
                }
            }
            else if (opExecSpace == "Device")
            {
                LibUtilities::BackendMap &deviceBackendInfo =
                    session->GetDeviceBackendMap();
                auto opImplIter = deviceBackendInfo.find(opNameUpper);
                if (opImplIter != deviceBackendInfo.end())
                {
                    OpImpl    = opImplIter->second;
                    OpImplSet = true;
                }
            }

            if (OpImplSet == false)
            {
                NEKERROR(ErrorUtil::ewarning,
                         "No implementation space specified "
                         "for operator. Defaulting to " +
                             OpImpl);
            }
        }

        return OpImpl;
    }

protected:
    ElmtOp(const MultiRegions::ExpListSharedPtr &expansionList,
           const std::vector<std::string> &components)
        : Operator<TData>(expansionList, components)
    {
    }

    virtual void v_Apply(LibUtilities::Field<TData, TFieldIn> &in,
                         LibUtilities::Field<TData, TFieldOut> &out) = 0;
};

} // namespace Nektar::Operators
