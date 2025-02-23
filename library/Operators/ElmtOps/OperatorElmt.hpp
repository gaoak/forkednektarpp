///////////////////////////////////////////////////////////////////////////////
//
// File: OperatorElmt.hpp
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

// Forward-declare the BlockOperator base class so we can define the factory
template <typename TData> class BlockOperator;

// BlockOperator factory singleton
template <typename TData>
using BlockOperatorFactory =
    Nektar::LibUtilities::NekFactory<std::string, BlockOperator<TData>,
                                     const LocalRegions::ExpansionSharedPtr &,
                                     NekDataWarehouseSharedPtr>;

// BlockOperator factory singleton
template <typename TData>
BlockOperatorFactory<TData> &GetBlockOperatorFactory();

template <typename TData> class BlockOperator
{
public:
    virtual ~BlockOperator() = default;

    BlockOperator(const LocalRegions::ExpansionSharedPtr &exp,
                  NekDataWarehouseSharedPtr dataWarehouse)
        : m_exp(exp), m_dataWarehouse(dataWarehouse)
    {
    }

    template <typename TOperator>
    static std::shared_ptr<TOperator> Create(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse, std::string execStr,
        std::string implStr)
    {
        std::string requestedKey = TOperator::name + execStr + implStr;

        BlockOperatorFactory<TData> &factory = GetBlockOperatorFactory<TData>();

        bool notFound = true;

        constexpr size_t nOpTests = 2;

        std::string key;
        for (size_t i = 0; i < nOpTests; ++i)
        {
            switch (i)
            {
                case 0:
                    // Find the operator with the requested ExecSpace and the
                    // same implementation.
                    key = TOperator::name + execStr + implStr;
                    break;
                case 1:
                    // Find the operator with the requested ExecSpace and a
                    // general implementation.
                    key = TOperator::name + execStr + "Generic";
                    break;
                default:
                    break;
            }

            if (factory.ModuleExists(key))
            {
                if (key != requestedKey && i != 1)
                {
                    std::string msg;
                    msg += "The requested operator: " + requestedKey +
                           " was not found. Using operator: " + key +
                           " instead";

                    WARNINGL0(false, msg);
                }

                notFound = false;

                break;
            }
        }

        // No suitible operator was found.
        if (notFound)
        {
            std::stringstream msg;
            msg << "No such operator: " << requestedKey
                << " and no default operator: " << key << "." << std::endl;
            factory.PrintAvailableClasses(msg);
            NEKERROR(ErrorUtil::efatal, msg.str());
        }

        return std::static_pointer_cast<TOperator>(
            factory.CreateInstance(key, exp, dataWarehouse));
    }

protected:
    LocalRegions::ExpansionSharedPtr m_exp;
    NekDataWarehouseSharedPtr m_dataWarehouse;
};

template <FieldState TFieldIn, FieldState TFieldOut, typename TData>
class OperatorElmt : public Operator<TData>
{

public:
    ~OperatorElmt() override = default;

    OperatorElmt(const MultiRegions::ExpListSharedPtr &expansionList)
        : Operator<TData>(expansionList)
    {
    }

    virtual void apply(Field<TData, TFieldIn> &in,
                       Field<TData, TFieldOut> &out) = 0;
};

} // namespace Nektar::Operators
