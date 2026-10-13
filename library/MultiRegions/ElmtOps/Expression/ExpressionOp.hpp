///////////////////////////////////////////////////////////////////////////////
//
// File: ExpressionOp.hpp
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
 * @file ExpressionOp.hpp
 * @brief Public interface of the Expression operator, which evaluates
 * analytical expressions at the quadrature points of a physical-space
 * field.
 *
 * @details
 * The operator writes to every output component `nc` the value
 * `scale * f_nc(x, y, z, t, ...)`, where `f_nc` is a parsed
 * session-file expression (LibUtilities::Equation), `(x, y, z)` are
 * the quadrature-point coordinates, `t` is the time set through
 * SetTime(), and any expression variables beyond those built-in four
 * ("EVARS") take their values from the components of the input field.
 * Optionally the result is accumulated onto the output instead of
 * overwriting it (SetAppend()). SolverCore uses the operator to
 * evaluate a session's initial-condition and body-forcing functions,
 * and the boundary-condition operators in MultiRegions/BndCondOps use it
 * to evaluate Dirichlet and Neumann data on boundary expansions.
 *
 * The operation is purely pointwise -- there is no basis, matrix or
 * tensor-product structure to exploit -- so this family provides a
 * single "Generic" block implementation per execution space instead of
 * the StdMat/SumFac/SumFacTOP split of the transform-type families
 * (ExpressionBlockOpSerialAVXGeneric.hpp, ExpressionBlockOpDeviceGeneric.hpp).
 * The whole-field interface classes are registered through the CMake-generated
 * factory declarations shared by all element operators; the family has no
 * factory template of its own.
 *
 * @see ExpressionBlockOp.hpp for the per-block interface holding the
 * evaluation settings.
 */

#pragma once

#include <boost/algorithm/string.hpp>

#include <MultiRegions/ElmtOps/ElmtOp.hpp>

#include <MultiRegions/ElmtOps/Expression/ExpressionBlockOp.hpp>

namespace Nektar::MultiRegions
{

/**
 * @brief Whole-field interface of the Expression operator: evaluates
 * one analytical expression per field component at every quadrature
 * point.
 *
 * @details
 * As with the other element-operator interfaces, deriving from
 * ElmtOp<FieldState::Phys, FieldState::Phys, TData> sets the Apply()
 * parameter types at compile time: both fields must be in physical
 * space. Unusually for an element operator, the input field is not
 * transformed -- it supplies the values of any extra expression
 * variables (EVARS) beyond the built-in x, y, z and t, and merely
 * sets the storage layout when there are none. The operator holds one
 * ExpressionBlockOp per element block (m_blockOp) and its v_Apply()
 * delegates each block of the input/output fields to the matching
 * block operator.
 *
 * The evaluation settings live in the block operators; every setter
 * here simply forwards its value to all of them. The intended workflow
 * is Create(), then optionally SetComponentMask(), then
 * SetExpressions() -- mandatory, Apply() asserts otherwise -- then
 * optionally SetTime(), SetScale() and SetAppend(), then Apply().
 * Expressions may be swapped at any time between applications.
 *
 * SetTime(), SetScale() and SetAppend() are order-free and may be
 * changed between applications. SetComponentMask() is not: the mask
 * must be final before SetExpressions() is called, and changing it
 * afterwards requires calling SetExpressions() again (the in-tree
 * callers in MultiRegions/BndCondOps set the mask first).
 *
 * @tparam TData Floating-point type of the field data.
 *
 * @see ExpressionBlockOp for the per-block settings and the
 * v_SetExpressions() hook the back-ends implement.
 */
template <typename TData>
class ExpressionOp : public ElmtOp<FieldState::Phys, FieldState::Phys, TData>
{
    friend class ElmtOp<FieldState::Phys, FieldState::Phys, TData>;

public:
    /**
     * @brief Create an Expression operator through the two-stage
     * element-operator factory and initialise its default settings.
     *
     * After the interface object and its per-block operators are built
     * by ElmtOp::Create, every block operator is given the defaults:
     * all components enabled in the mask, time 0 and scale 1 (the
     * append flag defaults to false in ExpressionBlockOp). No
     * expressions are set -- SetExpressions() must be called before
     * the first Apply().
     *
     * @param   expansionList   Expansion list the operator acts on;
     *                          its Collections define the blocks.
     * @param   components      Names of the field components; their
     *                          count sizes the default component mask.
     * @param   execStr         Execution space ("Serial", "AVX" or
     *                          "Device"); if empty, taken from the
     *                          session's "opExecSpace" command-line
     *                          argument or, failing that, the build's
     *                          default (see ElmtOp::Create).
     * @param   implStr         Block implementation part of the
     *                          factory key. This family registers only
     *                          "Generic", so any other request falls
     *                          back to it (see ElmtBlockOp::Create).
     *
     * @return The newly created operator, ready for SetExpressions().
     */
    static std::shared_ptr<ExpressionOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "", const std::string &implStr = "")
    {
        auto op =
            ElmtOp<FieldState::Phys, FieldState::Phys, TData>::template Create<
                ExpressionOp, ExpressionBlockOp>(expansionList, components,
                                                 execStr, implStr);

        // Default component mask, true for all components
        std::vector<bool> cmask;
        for (unsigned int i = 0; i < components.size(); ++i)
        {
            cmask.push_back(true);
        }

        // Loop over the blocks.
        for (unsigned int blk = 0; blk < op->m_blockOp.size(); ++blk)
        {
            // Default component mask
            op->m_blockOp[blk]->SetComponentMask(cmask);

            // Default time set to 0.0
            op->m_blockOp[blk]->SetTime(0.0);

            // Default scale set to 1.0
            op->m_blockOp[blk]->SetScale(1.0);
        }
        return op;
    }

    /// Operator base name; Create() appends the execution space to it
    /// to form the factory key.
    static inline const std::string name = "Expression";

    /**
     * @brief Set the expressions to evaluate, one per field component
     * in component order.
     *
     * The variable count of each expression is extracted with
     * GetNumberEvars() and stored on the block operators with
     * SetNumEvars(); the expressions go through their
     * v_SetExpressions() hook. Also marks the operator ready for Apply().
     *
     * @param   exprs   Parsed expressions; their number must equal the
     *                  number of components of the fields later passed
     *                  to Apply().
     */
    void SetExpressions(
        const std::vector<LibUtilities::EquationSharedPtr> &exprs)
    {
        // Check number of EVARS for new expressions
        std::vector<unsigned int> numEvars;
        for (unsigned int ne = 0; ne < exprs.size(); ++ne)
        {
            numEvars.push_back(this->GetNumberEvars(exprs[ne]));
        }

        // Loop over the blocks.
        for (unsigned int blk = 0; blk < m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetExpressions(exprs);
            this->m_blockOp[blk]->SetNumEvars(numEvars);
        }

        m_isExpressionsDefined = true;
    }

    /**
     * @brief Set the value substituted for the built-in time variable
     * `t` in every expression (default 0).
     */
    void SetTime(const TData &time)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetTime(time);
        }
    }

    /**
     * @brief Set the factor every evaluated expression value is
     * multiplied by before it is written or appended (default 1).
     */
    void SetScale(const TData &scale)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetScale(scale);
        }
    }

    /**
     * @brief Choose between accumulating onto the output
     * (`out += scale * f`) and overwriting it (`out = scale * f`, the
     * default).
     */
    void SetAppend(const bool &append)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetAppend(append);
        }
    }

    /**
     * @brief Select which output components the expressions are
     * evaluated for (default: all enabled).
     *
     * @param   cmask   One flag per component, in component order;
     *                  Apply() asserts its size matches the number of
     *                  expressions.
     *
     * @note Call this before SetExpressions() (see the class notes on
     * the setter ordering).
     *
     * @note Masked-out output components are not guaranteed to be left
     * untouched.
     */
    void SetComponentMask(const std::vector<bool> &cmask)
    {
        // Loop over the blocks.
        for (unsigned int blk = 0; blk < m_blockOp.size(); ++blk)
        {
            this->m_blockOp[blk]->SetComponentMask(cmask);
        }
    }

    /**
     * @brief Count the variables of an expression, i.e. the length of
     * its variable list.
     *
     * The list contains the built-in `x y z t` (the
     * LibUtilities::Equation default) plus any extra field variables
     * (EVARS) declared in the session; the count is therefore at
     * least four; the back-ends assume those four are present, and
     * use the count to decide how many input-field components to
     * gather per point.
     *
     * @param   expression  Expression whose variable list is counted.
     *
     * @return Number of comma/space-separated names in the list.
     */
    static unsigned int GetNumberEvars(
        const LibUtilities::EquationSharedPtr &expression)
    {
        // Extract the number of expression variables (EVARS).
        // Note that we use four by default: x, y, z, t
        std::vector<std::string> vars;
        auto variableList = expression->GetVlist();
        boost::split(vars, variableList, boost::is_any_of(", "));
        return vars.size();
    }

protected:
    /// Whether SetExpressions() has been called; Apply() asserts this.
    bool m_isExpressionsDefined = false;
    /// One block operator per element block, filled by ElmtOp::Create.
    std::vector<std::shared_ptr<ExpressionBlockOp<TData>>> m_blockOp;

    /**
     * @brief Construct the interface part of a concrete implementation;
     * called by the factory-registered creator functions.
     *
     * @param   expansionList   Expansion list the operator acts on.
     * @param   components      Component names the operator is set up
     *                          for.
     */
    ExpressionOp(const MultiRegions::ExpListSharedPtr &expansionList,
                 const std::vector<std::string> &components)
        : ElmtOp<FieldState::Phys, FieldState::Phys, TData>(expansionList,
                                                            components)
    {
    }

    ~ExpressionOp() override = default;

    /**
     * @brief Evaluate the expressions block by block: checks that the
     * fields conform and that expressions have been set, then hands
     * each input/output block pair to its block operator.
     *
     * @param   in      Physical-space field supplying the EVAR
     *                  values; not modified unless it is the same
     *                  object as @p out.
     * @param   out     Physical-space field receiving the evaluated
     *                  expressions; in append mode its existing values
     *                  are accumulated onto.
     */
    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
                 LibUtilities::Field<TData, FieldState::Phys> &out) override
    {
        ASSERTL1(in.GetNumComponents() == out.GetNumComponents(),
                 "Number of input and output components differ");

        ASSERTL1(in.GetNumHomoModes() == out.GetNumHomoModes(),
                 "Number of input and output homogeneous modes differ");

        ASSERTL1(this->m_isExpressionsDefined,
                 "No expressions defined for this Operator. Define with "
                 "SetExpressions().")

        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_blockOp.size(); ++blk)
        {
            // Block dependent.
            auto &inblock  = in.GetBlocks()[blk];
            auto &outblock = out.GetBlocks()[blk];

            this->m_blockOp[blk]->Apply(inblock, outblock);
        }
    }
};

} // namespace Nektar::MultiRegions
