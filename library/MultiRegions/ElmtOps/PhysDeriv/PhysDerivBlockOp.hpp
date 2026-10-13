///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivBlockOp.hpp
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
// Description: Per-block interface class of the physical-derivative
// operator.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file PhysDerivBlockOp.hpp
 * @brief Per-block interface class of the physical-derivative operator.
 *
 * @details
 * See PhysDerivOp.hpp for the mathematical action and the layout of
 * the family. The concrete per-block implementations
 * (detail::PhysDerivBlockOpImpl) live in the per-execution-space
 * headers PhysDerivBlockOpSerialAVXStdMat.hpp,
 * PhysDerivBlockOpSerialAVXSumFac.hpp, PhysDerivBlockOpDeviceStdMat.hpp and
 * PhysDerivBlockOpDeviceSumFac.hpp.
 */

#pragma once

#include <MultiRegions/ElmtOps/ElmtBlockOp.hpp>

namespace Nektar::MultiRegions
{

/**
 * @brief Per-block interface of the physical derivative: differentiates
 * every element of one block.
 *
 * Sets the BlockAccessor states accepted by Apply() -- physical space
 * in, physical space out, the output block holding coordDim (or, for
 * 3DH1, three) components per input component (see PhysDerivOp) -- and
 * provides the factory hook through which PhysDerivOp::Create builds
 * one implementation per block. Unlike BwdTransBlockOp it carries no
 * append flag: the implementations always overwrite. The numerical
 * work happens in the detail::PhysDerivBlockOpImpl definitions, which
 * override v_Apply(): the sum-factorised ones with a shape-type switch
 * into the generated ShapeBlock() specialisations, the standard-matrix
 * ones with dense matrix products. CMake includes exactly one of those
 * headers per generated translation unit, matching the
 * execution-space/implementation pair that unit registers with the
 * block-operator factory.
 *
 * @tparam TData Floating-point type of the field data.
 *
 * @see ElmtBlockOp for the Apply()/v_Apply() contract; PhysDerivOp for
 * the whole-field interface driving this class.
 */
template <typename TData>
class PhysDerivBlockOp
    : public ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>
{
public:
    /**
     * @brief Create one physical-derivative block operator through the
     * block-operator factory, under the key
     * `"BlockPhysDeriv" + execStr + implStr` (see ElmtBlockOp::Create
     * for the Generic fall-back rules).
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
    static std::shared_ptr<PhysDerivBlockOp<TData>> Create(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse,
        const std::string &execStr, const std::string &implStr)
    {
        return ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>::
            template Create<PhysDerivBlockOp>(block_idx, exp, dataWarehouse,
                                              execStr, implStr);
    }

    /// Block-operator base name; Create() appends the execution space
    /// and implementation to it to form the factory key.
    static inline const std::string name = "BlockPhysDeriv";

protected:
    /**
     * @brief Construct the interface part of a concrete block
     * operator; called by the factory-registered creator functions.
     *
     * @param   block_idx       Index of the block within the expansion
     *                          list's Collections.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    PhysDerivBlockOp(const unsigned int block_idx,
                     const LocalRegions::ExpansionSharedPtr &exp,
                     LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>(block_idx, exp,
                                                                 dataWarehouse)
    {
    }

    ~PhysDerivBlockOp() override = default;
};

} // namespace Nektar::MultiRegions
