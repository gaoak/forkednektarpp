///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseBlockOp.hpp
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
// Description: Per-block interface of the IProductWRTBase element
// operator.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file IProductWRTBaseBlockOp.hpp
 * @brief Per-block interface of the IProductWRTBase element operator.
 *
 * @details
 * An IProductWRTBaseBlockOp takes the inner product with the basis over the
 * elements of one block (see ElmtBlockOp.hpp for the block scheme and
 * IProductWRTBaseOp.hpp for the mathematical action). It carries the
 * family-specific state the whole-field IProductWRTBaseOp forwards to every
 * block -- the scale factor and the quadrature-metric switch. The
 * per-implementation subclasses are the detail::IProductWRTBaseBlockOpImpl
 * definitions in IProductWRTBase{SerialAVX,Device}{StdMat,SumFac}.hpp.
 */

#pragma once

#include "Operators/ElmtOps/ElmtBlockOp.hpp"

namespace Nektar::Operators
{

/**
 * @brief Base class of the per-block inner products with the basis: takes
 * physical-space blocks to coefficient-space blocks.
 *
 * @details
 * Besides fixing the block states accepted by Apply() at compile time --
 * physical space in, coefficient space out -- this class stores the two
 * settings shared by every implementation: the scale factor #m_scale and
 * the quadrature-metric switch #m_integration, which selects between the
 * full inner product (the input weighted by the quadrature metric before
 * the basis transpose) and the basis transpose alone.
 * IProductWRTBaseOp::SetScale and SetIntegration forward to the setters
 * here block by block; the implementations read the settings in their
 * v_Apply().
 *
 * CMake generates one registration translation unit per execution space,
 * implementation and data type (from ElmtOps/ElmtBlockOpFactoryDec.cpp.in)
 * that includes exactly one detail::IProductWRTBaseBlockOpImpl header and
 * registers it under the matching execution-space/implementation key.
 *
 * @tparam TData Floating-point type of the field data.
 */
template <typename TData>
class IProductWRTBaseBlockOp
    : public ElmtBlockOp<FieldState::Phys, FieldState::Coeff, TData>
{
public:
    /**
     * @brief Create one IProductWRTBase block operator through the
     * block-operator factory.
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
    static std::shared_ptr<IProductWRTBaseBlockOp<TData>> Create(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse,
        const std::string &execStr, const std::string &implStr)
    {
        return ElmtBlockOp<FieldState::Phys, FieldState::Coeff, TData>::
            template Create<IProductWRTBaseBlockOp>(
                block_idx, exp, dataWarehouse, execStr, implStr);
    }

    /// Block-operator base name; Create() appends the execution space and
    /// implementation to it to form the factory key.
    static inline const std::string name = "BlockIProductWRTBase";

    /// @brief Set the scalar factor applied to the result; see
    /// IProductWRTBaseOp::SetScale.
    void SetScale(const TData &scale)
    {
        m_scale = scale;
    }

    /// @brief Enable (true, the initial state) or disable the quadrature
    /// metric; see IProductWRTBaseOp::SetIntegration.
    void SetIntegration(bool value)
    {
        m_integration = value;
    }

protected:
    /// Scalar factor applied to the result (default 1); how each
    /// implementation applies it is set out in IProductWRTBaseOp::SetScale.
    TData m_scale = 1.0;
    /// Quadrature metric on (the initial state) or off; see the class
    /// description.
    bool m_integration = true;

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
    IProductWRTBaseBlockOp(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : ElmtBlockOp<FieldState::Phys, FieldState::Coeff, TData>(
              block_idx, exp, dataWarehouse)
    {
    }

    ~IProductWRTBaseBlockOp() override = default;
};

} // namespace Nektar::Operators
