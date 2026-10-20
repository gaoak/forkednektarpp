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

/**
 * @file ExpressionBlockOp.hpp
 * @brief Per-block interface of the Expression operator, holding the
 * evaluation settings shared by all back-ends.
 *
 * See ExpressionOp.hpp for the family overview; the concrete
 * evaluators are in ExpressionBlockOpSerialAVXGeneric.hpp and
 * ExpressionBlockOpDeviceGeneric.hpp.
 */

#pragma once

#include <LibUtilities/BasicUtils/StringUtils.hpp>

#include "LibUtilities/BasicUtils/Equation.h"
#include <MultiRegions/ElmtOps/ElmtBlockOp.hpp>

namespace Nektar::MultiRegions
{

/**
 * @brief Block-level interface of the Expression operator: evaluates
 * the configured expressions at every quadrature point of one element
 * block.
 *
 * @details
 * Besides setting the BlockAccessor states accepted by Apply() to
 * physical space on both sides, this class holds the evaluation
 * settings -- expressions, per-expression variable counts, component
 * mask, time, scale and append flag. One instance per block is created
 * by ElmtOp::Create via ExpressionOp::Create, and the settings are
 * written through the whole-field ExpressionOp setters, which forward
 * to every block.
 *
 * All setters are plain stores except SetExpressions(), which goes
 * through the virtual v_SetExpressions() so a back-end can prepare for
 * evaluation.
 *
 * @tparam TData Floating-point type of the field data.
 *
 * @see detail::ExpressionBlockOpImpl (ExpressionBlockOpSerialAVXGeneric.hpp /
 * ExpressionBlockOpDeviceGeneric.hpp) for the v_Apply() implementations.
 */
template <typename TData>
class ExpressionBlockOp
    : public ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>
{
public:
    /**
     * @brief Create one block operator through the block-operator
     * factory.
     *
     * The factory key is #name + @p execStr + @p implStr; since the
     * family registers only "Generic" implementations, any other
     * @p implStr falls back to the Generic key of the same execution
     * space (see ElmtBlockOp::Create).
     *
     * @param   block_idx       Index of the block within the expansion
     *                          list's Collections.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     * @param   execStr         Execution-space part of the factory
     *                          key.
     * @param   implStr         Implementation part of the factory key.
     */
    static std::shared_ptr<ExpressionBlockOp<TData>> Create(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse,
        const std::string &execStr, const std::string &implStr)
    {
        return ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>::
            template Create<ExpressionBlockOp>(block_idx, exp, dataWarehouse,
                                               execStr, implStr);
    }

    /// Block-operator base name; Create() appends the execution space
    /// and implementation to it to form the factory key.
    static inline const std::string name = "BlockExpression";

    /**
     * @brief Split an expression's variable list into the individual
     * variable names.
     *
     * Entries 0-3 are the built-in x, y, z and t; entries from 4
     * onwards name the extra field variables (EVARS).
     *
     * @param   expression  Expression whose variable list is split.
     *
     * @return The comma/space-separated names, in list order.
     */
    static std::vector<std::string> GetEvarsNames(
        const LibUtilities::EquationSharedPtr &expression)
    {
        std::vector<std::string> vnames;
        StringUtils::Split(vnames, expression->GetVlist(),
                           StringUtils::IsAnyOf(", "));
        return vnames;
    }

    /**
     * @brief Set the expressions to evaluate (one per component) via
     * the back-end hook v_SetExpressions().
     */
    void SetExpressions(
        const std::vector<LibUtilities::EquationSharedPtr> &exprs)
    {
        v_SetExpressions(exprs);
    }

    /// @brief Set the value substituted for the built-in variable `t`.
    void SetTime(const TData &time)
    {
        this->m_time = time;
    }

    /// @brief Set the factor every evaluated value is multiplied by.
    void SetScale(const TData &scale)
    {
        this->m_scale = scale;
    }

    /**
     * @brief Choose between accumulating onto the output and
     * overwriting it.
     */
    void SetAppend(const bool &append)
    {
        this->m_append = append;
    }

    /**
     * @brief Set the variable count of each expression, as computed by
     * ExpressionOp::GetNumberEvars().
     */
    void SetNumEvars(const std::vector<unsigned int> &numEvars)
    {
        this->m_numEvars = numEvars;
    }

    /**
     * @brief Set the per-component enable flags (see the mask notes in
     * ExpressionOp::SetComponentMask, including the requirement that
     * the mask be final before SetExpressions()).
     */
    void SetComponentMask(const std::vector<bool> &cmask)
    {
        this->m_cmask = cmask;
    }

protected:
    /// Parsed expressions, one per field component.
    std::vector<LibUtilities::EquationSharedPtr> m_expressions;
    /// Variable-list length of each expression: the built-in x, y, z
    /// and t plus any field-valued EVARS. The implementations read
    /// entry 0 only, assuming all expressions share one variable list.
    std::vector<unsigned int> m_numEvars;
    /// Per-component enable flags.
    std::vector<bool> m_cmask;

    /// Value of the built-in time variable `t`.
    TData m_time = 0.0;
    /// Factor applied to every evaluated expression value.
    TData m_scale = 1.0;
    /// Accumulate onto the output (true) or overwrite it (false).
    bool m_append = false;

    /**
     * @brief Construct the interface part of a concrete block
     * operator; called by the factory-registered creator functions.
     *
     * @param   block_idx       Index of the block within the expansion
     *                          list's Collections.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    ExpressionBlockOp(const unsigned int block_idx,
                      const LocalRegions::ExpansionSharedPtr &exp,
                      LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>(block_idx, exp,
                                                                 dataWarehouse)
    {
    }

    ~ExpressionBlockOp() override = default;

    /**
     * @brief Implementation hook for SetExpressions().
     *
     * Overrides must arrange that #m_expressions, and whatever the
     * back-end derives from them, are in place before the work queued
     * by the next Apply() runs; that work need not have finished by
     * the time this call returns.
     *
     * @param   exprs   Parsed expressions, one per component.
     */
    virtual void v_SetExpressions(
        const std::vector<LibUtilities::EquationSharedPtr> &exprs) = 0;
};

} // namespace Nektar::MultiRegions
