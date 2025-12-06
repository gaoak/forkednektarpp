///////////////////////////////////////////////////////////////////////////////
//
// File: ExpressionBlockOp.hpp
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

#include "LibUtilities/BasicUtils/Equation.h"
#include "Operators/ElmtOps/BlockOperator.hpp"

namespace Nektar::Operators
{

template <typename TData> class ExpressionBlockOp : public BlockOperator<TData>
{
public:
    static std::shared_ptr<ExpressionBlockOp<TData>> Create(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse, std::string execStr,
        std::string implStr)
    {
        return BlockOperator<TData>::template Create<ExpressionBlockOp>(
            block_idx, exp, dataWarehouse, execStr, implStr);
    }

    static inline const std::string name = "BlockExpression";

    void SetExpressions(
        const std::vector<LibUtilities::EquationSharedPtr> &exprs)
    {
        v_SetExpressions(exprs);
    }

    void SetTime(const TData &time)
    {
        v_SetTime(time);
    }

    void SetScale(const TData &scale)
    {
        v_SetScale(scale);
    }

    void SetNumEvars(const std::vector<unsigned int> &numEvars)
    {
        v_SetNumEvars(numEvars);
    }

protected:
    std::vector<LibUtilities::EquationSharedPtr> m_expressions;

    ExpressionBlockOp(const unsigned int block_idx,
                      const LocalRegions::ExpansionSharedPtr &exp,
                      NekDataWarehouseSharedPtr dataWarehouse)
        : BlockOperator<TData>(block_idx, exp, dataWarehouse)
    {
    }

    ~ExpressionBlockOp() override = default;

    virtual void v_SetExpressions(
        const std::vector<LibUtilities::EquationSharedPtr> &exprs) = 0;

    virtual void v_SetTime(const TData &time) = 0;

    virtual void v_SetScale(const TData &scale) = 0;

    virtual void v_SetNumEvars(const std::vector<unsigned int> &numEvars) = 0;
};

} // namespace Nektar::Operators
