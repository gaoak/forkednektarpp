///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionBlockOp.hpp
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
// Description: Per-block interface of the advection element operator.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file AdvectionBlockOp.hpp
 * @brief Per-block interface of the advection element operator.
 *
 * @details
 * An AdvectionBlockOp applies scale * (V . grad u) to the elements of one
 * block (see ElmtBlockOp.hpp for the block scheme); on a 3DH1 expansion
 * only the in-plane terms, the homogeneous one being added by AdvectionOp
 * after the blocks. It carries the family-specific state the whole-field
 * AdvectionOp forwards to every block -- the scale factor, the append flag
 * and the per-block view of the advection velocity. The per-implementation
 * subclasses are the detail::AdvectionBlockOpImpl definitions in
 * Advection{SerialAVX,Device}{StdMat,SumFac}.hpp.
 */

#pragma once

#include <MultiRegions/ElmtOps/ElmtBlockOp.hpp>

namespace Nektar::MultiRegions
{

/**
 * @brief Base class of the per-block advection operators: evaluates
 * scale * (V . grad u) on every element of one block.
 *
 * @details
 * Besides setting the block states (physical space in and out), this class
 * stores the three settings shared by every implementation -- the scale
 * factor #m_scale, the accumulate flag #m_append and the pointer #m_advVel
 * to the block's advection velocity. AdvectionOp::SetScale, SetAppend and
 * SetAdvVel forward to the setters here block by block; the
 * implementations read the settings in their v_Apply().
 *
 * @tparam TData Floating-point type of the field data.
 */
template <typename TData>
class AdvectionBlockOp
    : public ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>
{
public:
    /**
     * @brief Create one advection block operator through the
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
    static std::shared_ptr<AdvectionBlockOp<TData>> Create(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse,
        const std::string &execStr, const std::string &implStr)
    {
        return ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>::
            template Create<AdvectionBlockOp>(block_idx, exp, dataWarehouse,
                                              execStr, implStr);
    }

    /// Block-operator base name used in the factory key.
    static inline const std::string name = "BlockAdvection";

    /// @brief Set the scalar factor applied to the advection term.
    void SetScale(const TData &scale)
    {
        m_scale = scale;
    }

    /// @brief Accumulate onto (true) or overwrite (false) the output.
    void SetAppend(const bool &append)
    {
        this->m_append = append;
    }

    /**
     * @brief Register the block's advection velocity.
     *
     * Stores a non-owning pointer to @p Vel, so the accessor must outlive
     * the operator. The implementations reshape its storage in place to
     * their interleave width when Apply() first finds it at another
     * width.
     *
     * @param   Vel     Velocity block accessor with one component per
     *                  coordinate dimension (three on 3DH1, the last
     *                  unused here).
     */
    void SetAdvVel(LibUtilities::BlockAccessor<TData, FieldState::Phys> &Vel)
    {
        m_advVel = &Vel;
    }

protected:
    /// Scalar factor applied to the advection term (default 1).
    TData m_scale = 1.0;
    /// If true, Apply() accumulates onto the output block.
    bool m_append = false;
    /// Advection velocity of the block, registered by SetAdvVel(); not
    /// owned.
    LibUtilities::BlockAccessor<TData, FieldState::Phys> *m_advVel = nullptr;

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
    AdvectionBlockOp(const unsigned int block_idx,
                     const LocalRegions::ExpansionSharedPtr &exp,
                     LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>(block_idx, exp,
                                                                 dataWarehouse)
    {
    }

    ~AdvectionBlockOp() override = default;
};

} // namespace Nektar::MultiRegions
