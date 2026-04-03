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
#include "Operators/ElmtOps/ElmtBlockOp.hpp"

namespace Nektar::Operators
{

template <typename TData>
class ExpressionBlockOp
    : public ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>
{
public:
    static std::shared_ptr<ExpressionBlockOp<TData>> Create(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse, std::string execStr,
        std::string implStr)
    {
        return ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>::
            template Create<ExpressionBlockOp>(block_idx, exp, dataWarehouse,
                                               execStr, implStr);
    }

    static inline const std::string name = "BlockExpression";

    void SetExpressions(
        const std::vector<LibUtilities::EquationSharedPtr> &exprs)
    {
        v_SetExpressions(exprs);
    }

    void SetTime(const TData &time)
    {
        this->m_time = time;
    }

    void SetScale(const TData &scale)
    {
        this->m_scale = scale;
    }

    void SetAppend(const bool &append)
    {
        this->m_append = append;
    }

    void SetNumEvars(const std::vector<unsigned int> &numEvars)
    {
        this->m_numEvars = numEvars;
    }

    void SetComponentMask(const std::vector<bool> &cmask)
    {
        this->m_cmask = cmask;
    }

protected:
    std::vector<LibUtilities::EquationSharedPtr> m_expressions;
    std::vector<unsigned int> m_numEvars;
    std::vector<bool> m_cmask;

    TData m_time  = 0.0;
    TData m_scale = 1.0;
    bool m_append = false;

    ExpressionBlockOp(const unsigned int block_idx,
                      const LocalRegions::ExpansionSharedPtr &exp,
                      NekDataWarehouseSharedPtr dataWarehouse)
        : ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>(block_idx, exp,
                                                                 dataWarehouse)
    {
    }

    ~ExpressionBlockOp() override = default;

    virtual void v_SetExpressions(
        const std::vector<LibUtilities::EquationSharedPtr> &exprs) = 0;
};

} // namespace Nektar::Operators
