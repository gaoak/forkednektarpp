///////////////////////////////////////////////////////////////////////////////
//
// File: MassBlockOp.hpp
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
// Description: Per-block interface of the Mass element operator.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file MassBlockOp.hpp
 * @brief Per-block interface of the Mass element operator.
 *
 * @details
 * A MassBlockOp applies the element mass matrix to every element of one
 * block (see ElmtBlockOp.hpp for the block scheme and MassOp.hpp for the
 * mathematical action). The family has no per-block settings, so the class
 * only sets the block states and provides the factory entry. The
 * per-implementation subclasses are the detail::MassBlockOpImpl definitions
 * in Mass{SerialAVX,Device}{StdMat,SumFac}.hpp.
 */

#pragma once

#include <MultiRegions/ElmtOps/ElmtBlockOp.hpp>

namespace Nektar::MultiRegions
{

/**
 * @brief Base class of the per-block mass operators: applies the element
 * mass matrix to every element of one block, coefficient space to
 * coefficient space.
 *
 * @details
 * Sets the block states accepted by Apply() at compile time -- coefficient
 * space in and out. Unlike BwdTransBlockOp it carries no append flag, and
 * unlike IProductWRTBaseBlockOp no scale factor or quadrature-metric switch:
 * the mass operator always overwrites its output with the full inner
 * product. The concrete implementations are the detail::MassBlockOpImpl
 * definitions listed in MassOp.hpp.
 *
 * CMake generates one registration translation unit per execution space,
 * implementation and data type (from ElmtOps/ElmtBlockOpFactoryDec.cpp.in)
 * that includes exactly one detail::MassBlockOpImpl header and registers it
 * under the matching execution-space/implementation key.
 *
 * @tparam TData Floating-point type of the field data.
 */
template <typename TData>
class MassBlockOp
    : public ElmtBlockOp<FieldState::Coeff, FieldState::Coeff, TData>
{
public:
    /**
     * @brief Create one Mass block operator through the block-operator
     * factory.
     *
     * The factory key is #name + @p execStr + @p implStr; see
     * ElmtBlockOp::Create for the lookup and fall-back rules.
     *
     * @param   block_idx       Index of the block within the expansion
     *                          list's Collections.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     * @param   execStr         Execution-space part of the key.
     * @param   implStr         Implementation part of the key.
     *
     * @return The newly created block operator.
     */
    static std::shared_ptr<MassBlockOp<TData>> Create(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse,
        const std::string &execStr, const std::string &implStr)
    {
        return ElmtBlockOp<FieldState::Coeff, FieldState::Coeff,
                           TData>::template Create<MassBlockOp>(block_idx, exp,
                                                                dataWarehouse,
                                                                execStr,
                                                                implStr);
    }

    /// Block-operator base name; Create() appends the execution space and
    /// implementation to it to form the factory key.
    static inline const std::string name = "BlockMass";

protected:
    /**
     * @brief Construct the interface part of a concrete block operator;
     * called by the factory-registered creator functions.
     *
     * @param   block_idx       Index of the block within the expansion
     *                          list's Collections.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    MassBlockOp(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : ElmtBlockOp<FieldState::Coeff, FieldState::Coeff, TData>(
              block_idx, exp, dataWarehouse)
    {
    }

    ~MassBlockOp() override = default;
};

} // namespace Nektar::MultiRegions
