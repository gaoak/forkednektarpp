///////////////////////////////////////////////////////////////////////////////
//
// File: ExpressionDeviceGenericHelper.hpp
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
// Description: The runtime compiled expression kernel: the variants of it, the
// arguments it is launched with, and the parts of its source that every device
// back-end generates alike.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file ExpressionDeviceGenericHelper.hpp
 * @brief The parts of the runtime-compiled expression kernel shared by
 * every device back-end: its variants, its launch arguments and the
 * generators of its constants, parameter list and body.
 *
 * @details
 * The back-end helpers, ExpressionDeviceGenericHIPCUDAHelper.hpp and
 * ExpressionDeviceGenericSYCLHelper.hpp, wrap these pieces in a kernel
 * declaration of their own, then compile and launch the result.
 */

#pragma once

#include <array>
#include <map>
#include <string>
#include <vector>

#include <boost/lexical_cast.hpp>

#include "LibUtilities/Backends/Backends.hpp"
#include "LibUtilities/BasicUtils/Equation.h"

namespace Nektar::MultiRegions::detail
{

/**
 * @brief The variants of the expression kernel that are compiled, in
 * the order Apply() selects them: appending to or assigning the output,
 * reading storage interleaved at width one or at the device warp size.
 */
enum ExpressionKernelIndex
{
    eAppendScalar = 0,
    eAssignScalar,
    eAppendInterleaved,
    eAssignInterleaved,
    eNumExpressionKernels
};

/**
 * @brief The arguments an expression kernel is launched with, in the
 * order its parameter list declares them (see
 * GetExpressionKernelParameters()).
 *
 * @tparam TData    Floating-point type of the field data.
 */
template <typename TData> struct ExpressionKernelArgs
{
    /// Stored values of one component over every homogeneous mode.
    size_t nsize;
    /// Stored values of one component in one mode.
    size_t compSize;
    /// Number of elements of the block, without padding.
    size_t nelmt;
    /// Quadrature points per element.
    size_t ndata;
    /// Factor applied to every evaluated value.
    TData scale;
    /// Coordinate table of the block, for one mode.
    const TData *coordptr;
    /// Value of the built-in time variable `t`.
    TData time;
    /// One input pointer per EVAR, at the component's first mode.
    std::vector<const TData *> evarptr;
    /// One output pointer per unmasked component, at its first mode.
    std::vector<TData *> outptr;
};

/**
 * @brief One variant of the expression kernel: the template arguments
 * instantiating it, and the suffix naming it where a back-end needs a
 * kernel function of its own per variant.
 */
struct ExpressionKernelVariant
{
    /// APPEND template argument, "true" or "false".
    std::string append;
    /// INTERLEAVEWIDTH template argument.
    std::string interleaveWidth;
    /// Suffix of the per-variant kernel name.
    std::string suffix;
};

/// The name the generated kernel function is compiled under.
inline const std::string expressionKernelName = "expression_kernel";

/// @brief The kernel variants, in ExpressionKernelIndex order.
inline std::array<ExpressionKernelVariant, eNumExpressionKernels>
GetExpressionKernelVariants(void)
{
    const std::string interleaveWidth =
        std::to_string(NektarSpaces::Device::warpSize);

    return {
        ExpressionKernelVariant{"true", "1", "_append_scalar"},
        ExpressionKernelVariant{"false", "1", "_assign_scalar"},
        ExpressionKernelVariant{"true", interleaveWidth, "_append_interleaved"},
        ExpressionKernelVariant{"false", interleaveWidth,
                                "_assign_interleaved"}};
}

// The pieces of a generated expression kernel below are shared by every device
// back-end: the kernel only differs in how it is declared and how it obtains
// its thread index, which the caller supplies. The generated kernel is
// templated on APPEND and INTERLEAVEWIDTH, and writes in the data type named
// TData, all three of which the caller declares.

/**
 * @brief Find the bracket matching the one at @p start by scanning
 * in the given direction.
 *
 * @param   s           Expression string to scan.
 * @param   start       Position of the opening or closing bracket
 *                      to match.
 * @param   direction   +1 to scan forwards from an opening
 *                      bracket, -1 to scan backwards from a
 *                      closing one.
 *
 * @return Position of the matching bracket, or @p start if none is
 * found.
 */
inline int findMatching(const std::string &s, int start, int direction)
{
    int count = 0;
    for (int i = start; i >= 0 && i < s.length(); i += direction)
    {
        if (s[i] == '(')
            count += direction;
        else if (s[i] == ')')
            count -= direction;
        if (count == 0)
            return i;
    }
    return start;
}

/**
 * @brief Rewrite every caret power `x^y` in @p expr into a
 * `pow(x, y)` call, since `^` is exclusive-or rather than
 * exponentiation in the generated kernel source.
 *
 * Each operand is taken to be either a bracketed group or a run of
 * alphanumeric/'.' characters adjacent to the caret; a leading
 * minus sign of the exponent is included.
 *
 * @param   expr    Expression string in the session-file notation.
 *
 * @return The rewritten expression.
 */
inline std::string convertPow(std::string expr)
{
    size_t caret;
    while ((caret = expr.find('^')) != std::string::npos)
    {
        int left_start, right_end;

        // 1. Identify Left Operand (Base)
        if (expr[caret - 1] == ')')
        {
            left_start = findMatching(expr, caret - 1, -1);
        }
        else
        {
            left_start = caret - 1;
            while (left_start > 0 && (isalnum(expr[left_start - 1]) ||
                                      expr[left_start - 1] == '.'))
            {
                left_start--;
            }
        }
        std::string base = expr.substr(left_start, caret - left_start);

        // 2. Identify Right Operand (Exponent)
        if (expr[caret + 1] == '(')
        {
            right_end = findMatching(expr, caret + 1, 1);
        }
        else
        {
            right_end = caret + 1;
            while (right_end < expr.length() - 1 &&
                   (isalnum(expr[right_end + 1]) || expr[right_end + 1] == '.'))
            {
                right_end++;
            }
        }
        std::string exponent = expr.substr(caret + 1, right_end - caret);

        // 3. Replace x^y with pow(x, y)
        std::string replacement = "pow(" + base + ", " + exponent + ")";
        expr.replace(left_start, (right_end - left_start + 1), replacement);
    }
    return expr;
}

/**
 * @brief Emit the constants and the parameters of the expressions as
 * macro variables, so that they are substituted when the kernel is
 * compiled. A name defined by several expressions is emitted once.
 *
 * @param   exprs   Parsed expressions, one per component.
 *
 * @return One `#define` per name, followed by a blank line.
 */
inline std::string GetExpressionKernelConstants(
    const std::vector<LibUtilities::EquationSharedPtr> &exprs)
{
    std::string kernel_src = "";

    std::map<std::string, double> constmap;
    for (unsigned int nc = 0; nc < exprs.size(); nc++)
    {
        auto constants = exprs[nc]->GetConstants();
        for (auto &constant : constants)
        {
            if (constmap.find(constant.first) == constmap.end())
            {
                kernel_src +=
                    "#define " + constant.first + " " +
                    boost::lexical_cast<std::string>(constant.second) + "\n";
                constmap.emplace(constant);
            }
        }
    }
    for (unsigned int nc = 0; nc < exprs.size(); nc++)
    {
        auto parameters = exprs[nc]->GetParameters();
        for (auto &parameter : parameters)
        {
            if (constmap.find(parameter.first) == constmap.end())
            {
                kernel_src +=
                    "#define " + parameter.first + " " +
                    boost::lexical_cast<std::string>(parameter.second) + "\n";
                constmap.emplace(parameter);
            }
        }
    }
    kernel_src += "\n";

    return kernel_src;
}

/**
 * @brief Emit the parameter list of an expression kernel, without the
 * enclosing brackets: the fields of ExpressionKernelArgs in order, the
 * EVAR pointers named `<name>ptr` and the outputs `out<nc>`.
 *
 * @param   vnames  Variable names of the expressions (see
 *                  ExpressionBlockOp::GetEvarsNames()); entries from 4
 *                  on are the EVARs.
 * @param   cmask   Per-component enable flags; only enabled components
 *                  get an output pointer.
 *
 * @return The parameter list.
 */
inline std::string GetExpressionKernelParameters(
    const std::vector<std::string> &vnames, const std::vector<bool> &cmask)
{
    std::string kernel_src = "";

    // 0. size.
    kernel_src += "const size_t nsize, ";
    kernel_src += "const size_t compSize, ";
    kernel_src += "const size_t nelmt, ";
    kernel_src += "const size_t ndata, ";
    kernel_src += "const TData scale, ";
    // 1. Coordinates.
    kernel_src += "const TData* coordptr, ";
    // 2. time variable.
    kernel_src += "const TData t, ";
    // 3. Pointers.
    for (unsigned int i = 4; i < vnames.size(); i++)
    {
        kernel_src += "const TData* " + vnames[i] + "ptr, ";
    }
    // 4. Output variable.
    for (unsigned int nc = 0; nc < cmask.size(); nc++)
    {
        if (cmask[nc])
        {
            kernel_src += "TData* out" + std::to_string(nc) + ", ";
        }
    }
    kernel_src.pop_back();
    kernel_src.pop_back();

    return kernel_src;
}

/**
 * @brief Emit the argument list matching GetExpressionKernelParameters(),
 * without the enclosing brackets, so that a kernel can forward its
 * parameters on.
 *
 * @param   vnames  Variable names of the expressions.
 * @param   cmask   Per-component enable flags.
 *
 * @return The argument list.
 */
inline std::string GetExpressionKernelArguments(
    const std::vector<std::string> &vnames, const std::vector<bool> &cmask)
{
    std::string kernel_src = "";

    kernel_src += "nsize, compSize, nelmt, ndata, scale, coordptr, t, ";
    for (unsigned int i = 4; i < vnames.size(); i++)
    {
        kernel_src += vnames[i] + "ptr, ";
    }
    for (unsigned int nc = 0; nc < cmask.size(); nc++)
    {
        if (cmask[nc])
        {
            kernel_src += "out" + std::to_string(nc) + ", ";
        }
    }
    kernel_src.pop_back();
    kernel_src.pop_back();

    return kernel_src;
}

/**
 * @brief Emit the body of an expression kernel, without the enclosing
 * braces.
 *
 * The body takes its thread index from @p threadIndex, returns past
 * `nsize`, and on a padded element writes zero to every enabled output
 * (when not appending) without evaluating the expressions. Otherwise it
 * gathers the coordinates for the interleave width the kernel is
 * instantiated on and the EVAR values, then assigns or appends
 * `scale * (expression)` to each enabled output. `#define`s make the
 * common maths functions take TData arguments, so integer literals do
 * not make calls like `sqrt(61)` ambiguous, and caret powers are
 * rewritten by convertPow().
 *
 * One kernel covers every homogeneous mode of the block: the modes of a
 * component are stored back to back, so the thread index walks them,
 * while the coordinates are held for one mode only and are read at the
 * index within the mode.
 *
 * @param   exprs       Parsed expressions, one per component.
 * @param   cmask       Per-component enable flags.
 * @param   vnames      Variable names of the expressions.
 * @param   coordDim    Coordinate dimension of the block.
 * @param   threadIndex Expression yielding the global thread index.
 * @param   mathPrefix  Qualification of the maths functions, e.g.
 *                      `sycl::`.
 *
 * @return The kernel body.
 */
inline std::string GetExpressionKernelBody(
    const std::vector<LibUtilities::EquationSharedPtr> &exprs,
    const std::vector<bool> &cmask, const std::vector<std::string> &vnames,
    const unsigned int coordDim, const std::string &threadIndex,
    const std::string &mathPrefix)
{
    std::string kernel_src = "";

    // Assign local variables:
    // 1. Thread index.
    kernel_src += "    const size_t tid = " + threadIndex + ";\n";
    kernel_src += "    if (tid >= nsize) return;\n";
    kernel_src += "    const size_t cid = tid % compSize;\n";
    kernel_src += "    const size_t elmt = (INTERLEAVEWIDTH == 1)\n";
    kernel_src += "        ? cid / ndata\n";
    kernel_src +=
        "        : (cid / (ndata * INTERLEAVEWIDTH)) * INTERLEAVEWIDTH"
        " + cid % INTERLEAVEWIDTH;\n";
    kernel_src += "    if (elmt >= nelmt)\n";
    kernel_src += "    {\n";
    kernel_src += "        if constexpr (!APPEND)\n";
    kernel_src += "        {\n";
    for (unsigned int nc = 0; nc < cmask.size(); nc++)
    {
        if (cmask[nc])
        {
            kernel_src +=
                "            out" + std::to_string(nc) + "[tid] = 0;\n";
        }
    }
    kernel_src += "        }\n";
    kernel_src += "        return;\n";
    kernel_src += "    }\n";
    // 2. Coordinates.
    if (coordDim == 1)
    {
        kernel_src += "    TData x;\n";
    }
    else if (coordDim == 2)
    {
        kernel_src += "    TData x, y;\n";
    }
    else if (coordDim == 3)
    {
        kernel_src += "    TData x, y, z;\n";
    }
    kernel_src += "    if constexpr (INTERLEAVEWIDTH == 1) {\n";
    if (coordDim == 1)
    {
        kernel_src += "        x = coordptr[cid];\n";
    }
    else if (coordDim == 2)
    {
        kernel_src += "        x = coordptr[2 * cid];\n";
        kernel_src += "        y = coordptr[2 * cid + 1];\n";
    }
    else if (coordDim == 3)
    {
        kernel_src += "        x = coordptr[3 * cid];\n";
        kernel_src += "        y = coordptr[3 * cid + 1];\n";
        kernel_src += "        z = coordptr[3 * cid + 2];\n";
    }
    kernel_src += "    } else if constexpr (INTERLEAVEWIDTH != 1) {\n";
    kernel_src += "        const size_t ilane = cid % INTERLEAVEWIDTH;\n";
    kernel_src += "        const size_t iwarp = cid / INTERLEAVEWIDTH;\n";
    if (coordDim == 1)
    {
        kernel_src +=
            "        x = coordptr[iwarp * INTERLEAVEWIDTH + ilane];\n";
    }
    else if (coordDim == 2)
    {
        kernel_src +=
            "        x = coordptr[iwarp * 2 * INTERLEAVEWIDTH + ilane];\n";
        kernel_src += "        y = coordptr[iwarp * 2 * INTERLEAVEWIDTH + "
                      "INTERLEAVEWIDTH + ilane];\n";
    }
    else if (coordDim == 3)
    {
        kernel_src +=
            "        x = coordptr[iwarp * 3 * INTERLEAVEWIDTH + ilane];\n";
        kernel_src += "        y = coordptr[iwarp * 3 * INTERLEAVEWIDTH + "
                      "INTERLEAVEWIDTH + ilane];\n";
        kernel_src += "        z = coordptr[iwarp * 3 * INTERLEAVEWIDTH + 2 * "
                      "INTERLEAVEWIDTH + ilane];\n";
    }
    kernel_src += "    } \n";
    kernel_src += "\n";
    // 3. Pointers.
    for (unsigned int i = 4; i < vnames.size(); i++)
    {
        kernel_src +=
            "    const TData " + vnames[i] + " = " + vnames[i] + "ptr[tid];\n";
    }

    // Expressions are emitted from the original session string, so numeric
    // literals keep their source spelling. In runtime compilation, calls such
    // as sqrt(61) are ambiguous because 61 is an int and both float and double
    // overloads are viable. Cast math-function arguments to TData so overload
    // resolution follows the precision of this generated kernel.
    kernel_src +=
        "#define sqrt(x) " + mathPrefix + "sqrt(static_cast<TData>(x))\n";
    kernel_src +=
        "#define sin(x) " + mathPrefix + "sin(static_cast<TData>(x))\n";
    kernel_src +=
        "#define cos(x) " + mathPrefix + "cos(static_cast<TData>(x))\n";
    kernel_src +=
        "#define tan(x) " + mathPrefix + "tan(static_cast<TData>(x))\n";
    kernel_src +=
        "#define exp(x) " + mathPrefix + "exp(static_cast<TData>(x))\n";
    kernel_src +=
        "#define log(x) " + mathPrefix + "log(static_cast<TData>(x))\n";
    kernel_src +=
        "#define log10(x) " + mathPrefix + "log10(static_cast<TData>(x))\n";
    kernel_src += "#define pow(x,y) " + mathPrefix +
                  "pow(static_cast<TData>(x), static_cast<TData>(y))\n";

    // Define expression.
    for (unsigned int nc = 0; nc < exprs.size(); nc++)
    {
        if (cmask[nc])
        {
            std::string expression = convertPow(exprs[nc]->GetExpression());
            kernel_src += "    if constexpr (APPEND)\n";
            kernel_src += "    {\n";
            kernel_src += "        out" + std::to_string(nc) +
                          "[tid] += scale * (" + expression + ");\n";
            kernel_src += "    }\n";
            kernel_src += "    else\n";
            kernel_src += "    {\n";
            kernel_src += "        out" + std::to_string(nc) +
                          "[tid] = scale * (" + expression + ");\n";
            kernel_src += "    };\n";
        }
    }

    return kernel_src;
}

} // namespace Nektar::MultiRegions::detail
