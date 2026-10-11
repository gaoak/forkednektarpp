///////////////////////////////////////////////////////////////////////////////
//
// File: CurlCurlBlockOp.hpp
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
 * @file CurlCurlBlockOp.hpp
 * @brief Per-block interface of the curl-curl element operator.
 *
 * @details
 * A CurlCurlBlockOp applies curl(curl u) to the elements of one block (see
 * ElmtBlockOp.hpp for the block scheme and CurlCurlOp.hpp for the
 * mathematical action); on a multi-plane 3DH1 block it applies only the
 * plane part of one curl, which CurlCurlOpImpl completes. The family has
 * no per-block settings, so the class adds nothing to its ElmtBlockOp base
 * beyond the factory entry point and the base name. The per-implementation
 * subclasses are the detail::CurlCurlBlockOpImpl definitions in
 * CurlCurl{SerialAVX,Device}{StdMat,SumFac}.hpp.
 */

#pragma once

#include <MultiRegions/ElmtOps/ElmtBlockOp.hpp>

namespace Nektar::MultiRegions
{

/**
 * @brief Base class of the per-block curl-curl operators: evaluates
 * curl(curl u), or on a multi-plane 3DH1 block the plane part of one curl,
 * on every element of one block, from and to physical-space values.
 *
 * @details
 * Sets the block states (physical space in and out) accepted by Apply()
 * at compile time; the numerical work happens in the
 * detail::CurlCurlBlockOpImpl definitions, of which each CMake-generated
 * registration translation unit includes exactly one.
 *
 * @tparam TData Floating-point type of the field data.
 */
template <typename TData>
class CurlCurlBlockOp
    : public ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>
{
public:
    /**
     * @brief Create one curl-curl block operator through the
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
    static std::shared_ptr<CurlCurlBlockOp<TData>> Create(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse,
        const std::string &execStr, const std::string &implStr)
    {
        return ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>::
            template Create<CurlCurlBlockOp>(block_idx, exp, dataWarehouse,
                                             execStr, implStr);
    }

    /// Block-operator base name used in the factory key.
    static inline const std::string name = "BlockCurlCurl";

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
    CurlCurlBlockOp(const unsigned int block_idx,
                    const LocalRegions::ExpansionSharedPtr &exp,
                    LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>(block_idx, exp,
                                                                 dataWarehouse)
    {
    }

    ~CurlCurlBlockOp() override = default;
};

} // namespace Nektar::MultiRegions
