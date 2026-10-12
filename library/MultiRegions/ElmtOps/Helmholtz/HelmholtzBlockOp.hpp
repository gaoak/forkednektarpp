///////////////////////////////////////////////////////////////////////////////
//
// File: HelmholtzBlockOp.hpp
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
 * @file HelmholtzBlockOp.hpp
 * @brief Per-block interface class of the weak Helmholtz operator.
 *
 * @details
 * A HelmholtzBlockOp applies the weak Helmholtz operator over the elements
 * of one block (see ElmtBlockOp.hpp for the block scheme and HelmholtzOp.hpp
 * for the mathematical action). It carries the family-specific state the
 * whole-field HelmholtzOp forwards to every block -- the mass-term weight
 * m_lambda and the diffusion tensor m_diffCoeff. The per-implementation
 * subclasses are the detail::HelmholtzBlockOpImpl definitions in
 * Helmholtz{SerialAVX,Device}{StdMat,SumFac}.hpp.
 */

#pragma once

#include <MultiRegions/ElmtOps/ElmtBlockOp.hpp>

namespace Nektar::MultiRegions
{

/**
 * @brief Per-block interface of the weak Helmholtz operator: applies the
 * operator to every element of one block.
 *
 * @details
 * Sets the block states accepted by Apply() -- coefficient space in and
 * out -- and carries the two parameters shared by all implementations, the
 * mass weight m_lambda and the diffusion tensor m_diffCoeff. The concrete
 * implementations are the detail::HelmholtzBlockOpImpl definitions in the
 * headers listed in HelmholtzOp.hpp; CMake includes exactly one of them per
 * generated translation unit, matching the execution-space/implementation
 * pair that unit registers with the block-operator factory.
 *
 * @tparam TData Floating-point type of the field data.
 *
 * @see ElmtBlockOp for the Apply()/v_Apply() contract; HelmholtzOp for the
 * whole-field interface driving this class.
 */
template <typename TData>
class HelmholtzBlockOp
    : public ElmtBlockOp<FieldState::Coeff, FieldState::Coeff, TData>
{
public:
    /**
     * @brief Create one Helmholtz block operator through the block-operator
     * factory, under the key `"BlockHelmholtz" + execStr + implStr`.
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
    static std::shared_ptr<HelmholtzBlockOp<TData>> Create(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse,
        const std::string &execStr, const std::string &implStr)
    {
        return ElmtBlockOp<FieldState::Coeff, FieldState::Coeff, TData>::
            template Create<HelmholtzBlockOp>(block_idx, exp, dataWarehouse,
                                              execStr, implStr);
    }

    /// Block-operator base name; Create() appends the execution space
    /// and implementation to it to form the factory key.
    static inline const std::string name = "BlockHelmholtz";

    /// @brief Set the mass-term weight for subsequent Apply() calls; see
    /// HelmholtzOp::SetLambda.
    void SetLambda(const TData &lambda)
    {
        this->m_lambda = lambda;
    }

    /**
     * @brief Set the packed diffusion tensor for subsequent Apply() calls;
     * see HelmholtzOp::SetDiffCoeff for the packing order.
     *
     * The entries are copied into a MemoryRegion, from which the
     * implementations obtain a pointer in the memory space they run in.
     *
     * @param   diffCoeff   Packed tensor entries.
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

protected:
    /// Weight of the mass term; set by SetLambda() before Apply().
    TData m_lambda;
    /// Packed symmetric diffusion tensor; set by SetDiffCoeff() before
    /// Apply().
    LibUtilities::MemoryRegion<TData> m_diffCoeff;

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
    HelmholtzBlockOp(const unsigned int block_idx,
                     const LocalRegions::ExpansionSharedPtr &exp,
                     LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : ElmtBlockOp<FieldState::Coeff, FieldState::Coeff, TData>(
              block_idx, exp, dataWarehouse)
    {
    }

    ~HelmholtzBlockOp() override = default;
};

} // namespace Nektar::MultiRegions
