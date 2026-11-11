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
 * mathematical action). It carries the family-specific state the
 * whole-field MassOp forwards to every block -- the scale factor and the
 * append flag. The
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
 * space in and out -- and stores the two settings shared by every
 * implementation: the scale factor #m_scale and the append flag
 * #m_append. MassOp::SetScale and SetAppend forward to the setters here
 * block by block; the implementations read the settings in their
 * v_Apply(). The concrete implementations are the detail::MassBlockOpImpl
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

    /// @brief Set overwrite (false, the default) or accumulate (true)
    /// mode for subsequent Apply() calls; see MassOp::SetAppend().
    void SetAppend(const bool &append)
    {
        m_append = append;
    }

    /// @brief Set the scalar factor applied to the result; see
    /// MassOp::SetScale.
    void SetScale(const TData &scale)
    {
        m_scale = scale;
    }

protected:
    /// Scalar factor applied to the result (default 1).
    TData m_scale = 1.0;
    /// When true, Apply() adds the scaled result onto the output block's
    /// existing values instead of overwriting them.
    bool m_append = false;

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
