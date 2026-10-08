///////////////////////////////////////////////////////////////////////////////
//
// File: BwdTransBlockOp.hpp
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
// Description: Per-block interface of the backward transform
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file BwdTransBlockOp.hpp
 * @brief Per-block interface class of the backward-transform operator.
 */

#pragma once

#include <MultiRegions/ElmtOps/ElmtBlockOp.hpp>

namespace Nektar::MultiRegions
{

/**
 * @brief Per-block interface of the backward transform: takes every
 * element of one block from its expansion coefficients to its values at
 * the quadrature points.
 *
 * Sets the BlockAccessor states accepted by Apply() -- coefficient
 * space in, physical space out -- and carries the append flag shared by
 * all implementations. The concrete implementations are the
 * detail::BwdTransBlockOpImpl definitions in the headers listed in
 * BwdTransOp.hpp. CMake includes exactly one of them per generated
 * translation unit, picking the header by execution space and
 * implementation name, so that the unit registers the pair it was
 * generated for; the SumFac headers serve both the SumFac and the
 * Device-only SumFacTOP key.
 *
 * @tparam TData Floating-point type of the field data.
 *
 * @see ElmtBlockOp for the Apply()/v_Apply() contract; BwdTransOp for
 * the whole-field interface driving this class.
 */
template <typename TData>
class BwdTransBlockOp
    : public ElmtBlockOp<FieldState::Coeff, FieldState::Phys, TData>
{
public:
    /**
     * @brief Create one backward-transform block operator through the
     * block-operator factory, under the key
     * `"BlockBwdTrans" + execStr + implStr`.
     *
     * @param   block_idx       Index of the block within the expansion
     *                          list's Collections.
     * @param   exp             Representative expansion of the block
     *                          (its first element).
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     * @param   execStr         Execution-space part of the factory key.
     * @param   implStr         Implementation part of the factory key.
     *
     * @return The newly created block operator.
     */
    static std::shared_ptr<BwdTransBlockOp<TData>> Create(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse,
        const std::string &execStr, const std::string &implStr)
    {
        return ElmtBlockOp<FieldState::Coeff, FieldState::Phys, TData>::
            template Create<BwdTransBlockOp>(block_idx, exp, dataWarehouse,
                                             execStr, implStr);
    }

    /// Block-operator base name; Create() appends the execution space
    /// and implementation to it to form the factory key.
    static inline const std::string name = "BlockBwdTrans";

    /// @brief Set overwrite (false, the default) or accumulate (true)
    /// mode for subsequent Apply() calls; see BwdTransOp::SetAppend().
    void SetAppend(const bool &append)
    {
        this->m_append = append;
    }

protected:
    /// When true, Apply() adds the transform onto the output block's
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
    BwdTransBlockOp(const unsigned int block_idx,
                    const LocalRegions::ExpansionSharedPtr &exp,
                    LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : ElmtBlockOp<FieldState::Coeff, FieldState::Phys, TData>(
              block_idx, exp, dataWarehouse)
    {
    }

    ~BwdTransBlockOp() override = default;
};

} // namespace Nektar::MultiRegions
