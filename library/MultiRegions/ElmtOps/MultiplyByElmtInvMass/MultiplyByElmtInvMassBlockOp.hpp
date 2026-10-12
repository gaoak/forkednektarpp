///////////////////////////////////////////////////////////////////////////////
//
// File: MultiplyByElmtInvMassBlockOp.hpp
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
 * @file MultiplyByElmtInvMassBlockOp.hpp
 * @brief Per-block interface of the MultiplyByElmtInvMass element
 * operator.
 *
 * @details
 * A MultiplyByElmtInvMassBlockOp applies each element's inverse mass
 * matrix to that element's coefficients, for every element of one block
 * (see ElmtBlockOp.hpp for the block scheme and
 * MultiplyByElmtInvMassOp.hpp for the mathematical action). Besides setting
 * the block states and providing the factory entry, the class adds the
 * SetInvMassMatrix() hook through which the whole-field operator supplies
 * the per-element matrices a block of deformed geometry needs. The
 * per-execution-space subclasses are the
 * detail::MultiplyByElmtInvMassBlockOpImpl definitions in
 * MultiplyByElmtInvMassSerialAVXGeneric.hpp and
 * MultiplyByElmtInvMassDeviceGeneric.hpp.
 */

#pragma once

#include <MultiRegions/ElmtOps/ElmtBlockOp.hpp>

namespace Nektar::MultiRegions
{

/**
 * @brief Base class of the per-block inverse mass operators: applies each
 * element's inverse mass matrix to that element's coefficients, for every
 * element of one block, coefficient space to coefficient space.
 *
 * @details
 * Sets the block states accepted by Apply() at compile time -- coefficient
 * space in and out -- and adds the SetInvMassMatrix() hook through which
 * MultiplyByElmtInvMassOp::Create supplies the per-element matrices.
 *
 * How the inverse is formed depends on the block's geometry type, which is
 * uniform over a block:
 * - for a block of regular elements the element mass matrix is the
 *   standard-region mass matrix scaled by the element's constant Jacobian,
 *   \f$M^e = J^e M^{\mathrm{std}}\f$, so its inverse is the standard-region
 *   inverse mass matrix divided by that Jacobian. The implementations
 *   apply the single cached standard-region inverse to the whole block and
 *   then divide element by element by the Jacobian; no per-element matrix
 *   is stored;
 * - for a block of deformed elements no such factorisation exists, and the
 *   implementations apply the element-specific inverse mass matrices
 *   received through SetInvMassMatrix(), which already account for the
 *   geometry.
 *
 * CMake generates one registration translation unit per execution space
 * and data type (from ElmtOps/ElmtBlockOpFactoryDec.cpp.in) that includes
 * exactly one detail::MultiplyByElmtInvMassBlockOpImpl header and
 * registers it under the execution space's "Generic" key; the family has
 * no StdMat or SumFac implementation headers, so no other key exists for
 * it.
 *
 * @note On the deformed path the same per-element matrices are applied
 * to every homogeneous mode of a field.
 *
 * @tparam TData Floating-point type of the field data.
 *
 * @see ElmtBlockOp for the Apply()/v_Apply() contract;
 * MultiplyByElmtInvMassOp for the whole-field interface driving this class.
 */
template <typename TData>
class MultiplyByElmtInvMassBlockOp
    : public ElmtBlockOp<FieldState::Coeff, FieldState::Coeff, TData>
{
public:
    /**
     * @brief Create one inverse mass block operator through the
     * block-operator factory.
     *
     * The factory key is #name + @p execStr + @p implStr; see
     * ElmtBlockOp::Create for the lookup and fall-back rules. Only the
     * "Generic" implementation is registered for this family, so any other
     * @p implStr resolves to it through that fall-back.
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
    static std::shared_ptr<MultiplyByElmtInvMassBlockOp<TData>> Create(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse,
        const std::string &execStr, const std::string &implStr)
    {
        return ElmtBlockOp<FieldState::Coeff, FieldState::Coeff, TData>::
            template Create<MultiplyByElmtInvMassBlockOp>(
                block_idx, exp, dataWarehouse, execStr, implStr);
    }

    /// Block-operator base name; Create() appends the execution space and
    /// implementation to it to form the factory key.
    static inline const std::string name = "BlockMultiplyByElmtInvMass";

    /**
     * @brief Hand the block operator the inverse mass matrices of its
     * elements.
     *
     * Called once per block by MultiplyByElmtInvMassOp::Create, after the
     * block operator has been constructed and before any Apply().
     *
     * @param   dmat    For a block of deformed elements, one dense
     *                  nmTot x nmTot inverse mass matrix per element in
     *                  element order, copied verbatim from the raw storage
     *                  of the matrices Expansion::GetLocMatrix returns and
     *                  padded with zero matrices to the block's padded
     *                  element count; empty for a block of regular
     *                  elements, which needs no per-element matrix (see
     *                  the class description).
     *
     * @note @p dmat holds one matrix per padded element, not one per
     * padded element and homogeneous mode; every homogeneous mode uses the
     * same matrices.
     */
    void SetInvMassMatrix(std::vector<TData> &dmat)
    {
        v_SetInvMassMatrix(dmat);
    }

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
    MultiplyByElmtInvMassBlockOp(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : ElmtBlockOp<FieldState::Coeff, FieldState::Coeff, TData>(
              block_idx, exp, dataWarehouse)
    {
    }

    ~MultiplyByElmtInvMassBlockOp() override = default;

    /**
     * @brief Implementation hook for SetInvMassMatrix().
     *
     * Overrides must take a copy of @p dmat into whatever storage their
     * execution space applies it from: the vector the whole-field operator
     * passes is a local one that does not outlive the call. An empty
     * @p dmat is legitimate and means the block is regular, so no
     * per-element matrix will be used.
     *
     * @param   dmat    Per-element inverse mass matrices; see
     *                  SetInvMassMatrix().
     */
    virtual void v_SetInvMassMatrix(std::vector<TData> &dmat) = 0;
};

} // namespace Nektar::MultiRegions
