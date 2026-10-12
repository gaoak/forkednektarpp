///////////////////////////////////////////////////////////////////////////////
//
// File: LinAdvDiffReactionBlockOp.hpp
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
// Description: Per-block interface of the linearised
// advection-diffusion-reaction element operator.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file LinAdvDiffReactionBlockOp.hpp
 * @brief Per-block interface of the linearised
 * advection-diffusion-reaction element operator.
 *
 * @details
 * A LinAdvDiffReactionBlockOp applies the weak
 * advection-diffusion-reaction form to the elements of one block (see
 * ElmtBlockOp.hpp for the block scheme). It carries the family-specific
 * state the whole-field LinAdvDiffReactionOp forwards to every block --
 * the reaction coefficient, the packed diffusivity tensor and the
 * per-block view of the advection velocity. The per-implementation
 * subclasses are the detail::LinAdvDiffReactionBlockOpImpl definitions
 * in LinAdvDiffReaction{SerialAVX,Device}{StdMat,SumFac}.hpp; CMake
 * includes exactly one of them per generated translation unit, matching
 * the execution-space and implementation pair that unit registers with
 * the block-operator factory.
 */

#pragma once

#include <MultiRegions/ElmtOps/ElmtBlockOp.hpp>

namespace Nektar::MultiRegions
{

/**
 * @brief Base class of the per-block linearised
 * advection-diffusion-reaction operators: applies the weak
 * advection-diffusion-reaction form to every element of one block.
 *
 * @details
 * Besides setting the block states (coefficient space in and out), this
 * class stores the three settings shared by every implementation -- the
 * reaction coefficient #m_lambda, the packed diffusivity tensor
 * #m_diffCoeff and the pointer #m_advVel to the block's advection
 * velocity. LinAdvDiffReactionOp::SetLambda, SetDiffCoeff and SetAdvVel
 * forward to the setters here block by block; the implementations read
 * the settings in their v_Apply().
 *
 * @tparam TData Floating-point type of the field data.
 *
 * @see ElmtBlockOp for the Apply()/v_Apply() contract;
 * LinAdvDiffReactionOp for the whole-field interface driving this class
 * and for the mathematical form of the operator.
 */
template <typename TData>
class LinAdvDiffReactionBlockOp
    : public ElmtBlockOp<FieldState::Coeff, FieldState::Coeff, TData>
{
public:
    /**
     * @brief Create one block operator through the block-operator
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
    static std::shared_ptr<LinAdvDiffReactionBlockOp<TData>> Create(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse,
        const std::string &execStr, const std::string &implStr)
    {
        return ElmtBlockOp<FieldState::Coeff, FieldState::Coeff, TData>::
            template Create<LinAdvDiffReactionBlockOp>(
                block_idx, exp, dataWarehouse, execStr, implStr);
    }

    /// Block-operator base name used in the factory key.
    static inline const std::string name = "BlockLinAdvDiffReaction";

    /// @brief Set the reaction coefficient for subsequent Apply()
    /// calls; see LinAdvDiffReactionOp::SetLambda for its sign.
    void SetLambda(const TData &lambda)
    {
        this->m_lambda = lambda;
    }

    /**
     * @brief Set the packed constant diffusivity tensor for subsequent
     * Apply() calls.
     *
     * Asserts that @p diffCoeff has coordDim * (coordDim + 1) / 2 entries
     * -- one, three or six -- and copies it into a host-space memory
     * region; the implementations obtain a pointer to it in their own
     * memory space when they run, so the vector itself need not be kept
     * alive.
     *
     * @param   diffCoeff   Lower triangle of the symmetric tensor in the
     *                      order documented at
     *                      LinAdvDiffReactionOp::SetDiffCoeff.
     */
    void SetDiffCoeff(std::vector<TData> &diffCoeff)
    {
        const auto coordDim      = this->m_exp->GetCoordim();
        const auto diffCoeffSize = coordDim * (coordDim + 1) / 2;
        ASSERTL0(diffCoeff.size() == diffCoeffSize,
                 "The number of diffusion coefficients must match 1, 3 or 6 "
                 "for a 1D, 2D or 3D case, respectively.")

        this->m_diffCoeff = LibUtilities::MemoryRegion<
            TData>::template FromVector<NektarSpaces::HostSpace>(diffCoeff);
    }

    /**
     * @brief Register the block's advection velocity.
     *
     * Stores a non-owning pointer to @p Vel, so the accessor must outlive
     * the operator. @p Vel holds this block's velocity at the quadrature
     * points of every plane, one component per coordinate direction (three
     * on 3DH1, the last unused here). The
     * implementations reshape its storage in place to their interleave
     * width when Apply() first finds it at another width, and record the
     * new width on the accessor.
     *
     * @param   Vel     Physical-space block of the velocity field.
     */
    void SetAdvVel(LibUtilities::BlockAccessor<TData, FieldState::Phys> &Vel)
    {
        m_advVel = &Vel;
    }

protected:
    /// Reaction coefficient; unset until SetLambda() is called.
    TData m_lambda;
    /// Packed lower triangle of the symmetric diffusivity tensor, in
    /// host memory; unset until SetDiffCoeff() is called.
    LibUtilities::MemoryRegion<TData> m_diffCoeff;
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
    LinAdvDiffReactionBlockOp(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : ElmtBlockOp<FieldState::Coeff, FieldState::Coeff, TData>(
              block_idx, exp, dataWarehouse)
    {
    }

    ~LinAdvDiffReactionBlockOp() override = default;
};

} // namespace Nektar::MultiRegions
