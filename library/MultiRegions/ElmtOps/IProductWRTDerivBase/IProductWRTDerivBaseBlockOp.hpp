///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseBlockOp.hpp
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
// Description: Per-block interface of the IProductWRTDerivBase element
// operator.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file IProductWRTDerivBaseBlockOp.hpp
 * @brief Per-block interface of the IProductWRTDerivBase element operator.
 *
 * @details
 * An IProductWRTDerivBaseBlockOp applies the inner product with the
 * gradient of the basis to the elements of one block (see ElmtBlockOp.hpp
 * for the block scheme and IProductWRTDerivBaseOp.hpp for the mathematical
 * action). It carries the family-specific state the whole-field
 * IProductWRTDerivBaseOp forwards to every block -- the append flag and
 * the scale factor. The per-implementation subclasses are the
 * detail::IProductWRTDerivBaseBlockOpImpl definitions in
 * IProductWRTDerivBase{SerialAVX,Device}{StdMat,SumFac}.hpp.
 */

#pragma once

#include <MultiRegions/ElmtOps/ElmtBlockOp.hpp>

namespace Nektar::MultiRegions
{

/**
 * @brief Base class of the per-block inner products with the gradient of
 * the basis: takes physical-space blocks of coordDim components per output
 * component to blocks in state @p TFieldOut.
 *
 * @details
 * Besides setting the block states accepted by Apply() at compile time,
 * this class stores the two settings shared by every implementation --
 * the accumulate flag #m_append and the scale factor #m_scale.
 * IProductWRTDerivBaseOp::SetAppend and SetScale forward to the setters
 * here block by block; the implementations read the settings in their
 * v_Apply().
 *
 * The class is templated on its output state, so CMake generates one
 * registration translation unit per execution space, implementation and
 * data type (from ElmtOps/ElmtBlockOpTwoOutStates.cpp.in) that includes
 * exactly one detail::IProductWRTDerivBaseBlockOpImpl header and registers
 * both output states under the same execution-space/implementation key.
 *
 * @tparam TFieldOut Field state of the output block.
 * @tparam TData     Floating-point type of the field data.
 */
template <FieldState TFieldOut, typename TData>
class IProductWRTDerivBaseBlockOp
    : public ElmtBlockOp<FieldState::Phys, TFieldOut, TData>
{
    template <typename TDataOp>
    using TIProductWRTDerivBaseBlockOp =
        IProductWRTDerivBaseBlockOp<TFieldOut, TDataOp>;

public:
    /**
     * @brief Create one IProductWRTDerivBase block operator through the
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
    static std::shared_ptr<IProductWRTDerivBaseBlockOp<TFieldOut, TData>> Create(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse,
        const std::string &execStr, const std::string &implStr)
    {
        return ElmtBlockOp<FieldState::Phys, TFieldOut, TData>::template Create<
            TIProductWRTDerivBaseBlockOp>(block_idx, exp, dataWarehouse,
                                          execStr, implStr);
    }

    /// Block-operator base name used in the factory key; carries the
    /// output state so that the Coeff and Phys instantiations register
    /// separately.
    static inline const std::string name =
        "BlockIProductWRTDerivBase" + FieldStateToString<TFieldOut>();

    /// @brief Accumulate onto (true) or overwrite (false) the output; see
    /// IProductWRTDerivBaseOp::SetAppend.
    void SetAppend(bool append)
    {
        m_append = append;
    }

    /// @brief Set the scalar factor applied to the result.
    void SetScale(const TData &scale)
    {
        m_scale = scale;
    }

protected:
    /// If true, Apply() accumulates onto the output block.
    bool m_append = false;
    /// Scalar factor applied to the result (default 1).
    TData m_scale = 1.0;

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
    IProductWRTDerivBaseBlockOp(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : ElmtBlockOp<FieldState::Phys, TFieldOut, TData>(block_idx, exp,
                                                          dataWarehouse)
    {
    }

    ~IProductWRTDerivBaseBlockOp() override = default;
};

} // namespace Nektar::MultiRegions
