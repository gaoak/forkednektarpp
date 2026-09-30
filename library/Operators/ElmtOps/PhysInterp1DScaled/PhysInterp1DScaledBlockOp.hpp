///////////////////////////////////////////////////////////////////////////////
//
// File: PhysInterp1DScaledBlockOp.hpp
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
 * @file PhysInterp1DScaledBlockOp.hpp
 * @brief Per-block interface class of the scaled physical-space
 * interpolation operator.
 */

#pragma once

#include <vector>

#include "Operators/ElmtOps/ElmtBlockOp.hpp"

namespace Nektar::Operators
{

/**
 * @brief Per-block interface of the scaled interpolation: interpolates
 * every element of one block onto its rescaled quadrature grid.
 *
 * Sets the BlockAccessor states accepted by Apply() -- physical space in
 * and out -- and carries the scale factor shared by all implementations,
 * together with the hook through which setting it lets an implementation
 * prepare its interpolation data. The concrete implementations are the
 * detail::PhysInterp1DScaledBlockOpImpl definitions in the headers listed
 * in PhysInterp1DScaledOp.hpp; CMake includes exactly one of them per
 * generated registration translation unit, matching the
 * execution-space/implementation pair that unit registers with the
 * block-operator factory.
 *
 * @tparam TData Floating-point type of the field data.
 *
 * @see ElmtBlockOp for the Apply()/v_Apply() contract;
 * PhysInterp1DScaledOp for the whole-field interface driving this class
 * and for the meaning of the scale factor.
 */
template <typename TData>
class PhysInterp1DScaledBlockOp
    : public ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>
{
public:
    /**
     * @brief Create one scaled-interpolation block operator through the
     * block-operator factory, under the key #name + @p execStr +
     * @p implStr.
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
    static std::shared_ptr<PhysInterp1DScaledBlockOp<TData>> Create(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse,
        const std::string &execStr, const std::string &implStr)
    {
        return ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>::
            template Create<PhysInterp1DScaledBlockOp>(
                block_idx, exp, dataWarehouse, execStr, implStr);
    }

    /// Block-operator base name; Create() appends the execution space
    /// and implementation to it to form the factory key.
    static inline const std::string name = "BlockPhysInterp1DScaled";

    /**
     * @brief Set this block's scale factor, and with it the number of
     * output points per direction.
     *
     * Must be called before Apply(): the implementations derive their
     * output point counts, workspaces and interpolation data from it (see
     * v_SetScaleFactor), none of which is known at construction. The
     * sum-factorised implementations assert on an unset factor; the
     * StdMat ones do not.
     *
     * @param   scale   Multiplier applied to the per-direction
     *                  quadrature-point counts; see the file notes of
     *                  PhysInterp1DScaledOp.hpp for the exact rule.
     */
    void SetScaleFactor(const TData &scale)
    {
        v_SetScaleFactor(scale);
    }

protected:
    /// Scale factor most recently set, or -1 while unset -- the value the
    /// sum-factorised implementations assert against.
    TData m_scale = -1.0; // scaling factor

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
    PhysInterp1DScaledBlockOp(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>(block_idx, exp,
                                                                 dataWarehouse)
    {
    }

    ~PhysInterp1DScaledBlockOp() override = default;

    /// \brief The per-direction point counts of the @p scale over-integrated
    /// grid this operator's interpolation and projection matrices are built
    /// for.
    ///
    /// Direction 0 scales outright, and a direction carrying one point fewer
    /// than direction 0 keeps that offset, which is what lets the switch
    /// templating see the same relationship on the over-integrated grid as on
    /// the native one.
    static std::vector<unsigned int> GetScaledNumPoints(
        const std::vector<unsigned int> &nq, const NekDouble scale)
    {
        std::vector<unsigned int> nqScaled;
        nqScaled.reserve(nq.size());

        for (size_t d = 0; d < nq.size(); ++d)
        {
            nqScaled.push_back((d != 0 && nq[0] - nq[d] == 1)
                                   ? static_cast<unsigned int>(scale * nq[0]) -
                                         1
                                   : static_cast<unsigned int>(scale * nq[d]));
        }

        return nqScaled;
    }

    /**
     * @brief Implementation hook for SetScaleFactor().
     *
     * Overrides must record @p scale in #m_scale and put the block
     * operator into a state where Apply() interpolates onto the point
     * counts that factor implies: fix the per-direction output sizes,
     * fetch the matching interpolation data from the data warehouse and
     * size any workspace, replacing whatever an earlier call set up.
     *
     * @param   scale   Multiplier applied to the per-direction
     *                  quadrature-point counts.
     */
    virtual void v_SetScaleFactor(const TData &scale) = 0;
};

} // namespace Nektar::Operators
