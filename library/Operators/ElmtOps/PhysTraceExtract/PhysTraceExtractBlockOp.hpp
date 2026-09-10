///////////////////////////////////////////////////////////////////////////////
//
// File: PhysTraceExtractBlockOp.hpp
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
 * @file PhysTraceExtractBlockOp.hpp
 * @brief Per-block interface class of the trace extraction operator
 * PhysTraceExtract.
 */

#pragma once

#include <vector>

#include "Operators/ElmtOps/ElmtBlockOp.hpp"

namespace Nektar::Operators
{

/**
 * @brief Per-block interface of the trace extraction: interpolates the
 * volume field of every element of one block onto that block's packed
 * trace storage.
 *
 * @details
 * Fixes the BlockAccessor states accepted by Apply() -- physical space
 * in, physical space out -- with the input block holding the element's
 * volume quadrature points and the output block its packed trace values.
 * See PhysTraceExtractOp for both layouts and for the mathematics.
 *
 * On top of ElmtBlockOp this class adds two things:
 * - ExtractTrace(), the per-trace entry point;
 * - #m_isCollocated, the tangential collocation flags the
 *   implementations fill, with SetIsCollocated() to overwrite them.
 *
 * The per-trace entry point is a virtual hook whose base implementation
 * raises a fatal error, so an implementation that does not provide it
 * fails loudly rather than silently doing nothing. Clearing the
 * collocation flags needs no such hook: the flags live here and every
 * implementation reads them the same way. There is no append flag here,
 * unlike the adjoint IProductWRTPhysTraceBlockOp: this operator's paths
 * write the trace entries they are responsible for.
 *
 * The concrete implementations are the
 * detail::PhysTraceExtractBlockOpImpl definitions in
 * PhysTraceExtractSerialAVXGeneric.hpp (Serial and AVX) and
 * PhysTraceExtractDeviceGeneric.hpp (Device). Their per-shape entry
 * points, the ShapeBlock() specialisations, are generated from
 * LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in, one
 * translation unit per shape, execution space and data type, so the
 * size-templated OperatorND() instantiations of the SumFac families are
 * kept.
 *
 * There is a single implementation, registered under
 * Operators::Generic: only one trace-extraction algorithm exists, and
 * no StdMat/SumFac/SumFacTOP choice is to be made here. A caller asking
 * for another implementation -- StdMat or SumFac from the whole-field
 * operator selection, SumFacTOP from a Device run -- is served that one
 * registration by ElmtBlockOp::Create(), which falls back to the
 * `"Generic"` factory key whenever the requested key is absent.
 *
 * Registering under Generic is also what keeps a SumFacTOP run correct.
 * The kernels index the trace data at warpSize interleave;
 * instantiating them with the SumFacTOP tag would have set an
 * implementation width of one and produced wrong numbers on real
 * hardware -- invisibly in a DEVICEONHOST build, where warpSize is 1
 * and the two interleaves coincide. With one registration there is no
 * such tag to instantiate against.
 *
 * @tparam TData Floating-point type of the field data.
 *
 * @see ElmtBlockOp for the Apply()/v_Apply() contract;
 * PhysTraceExtractOp for the whole-field interface driving this class.
 */
template <typename TData>
class PhysTraceExtractBlockOp
    : public ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>
{
public:
    /**
     * @brief Create one trace extraction block operator through the
     * block-operator factory, under the key
     * `"BlockPhysTraceExtract" + execStr + implStr`.
     *
     * @param   block_idx       Index of the block within the expansion
     *                          list's Collections.
     * @param   exp             Representative expansion of the block
     *                          (its first element).
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list; the
     *                          implementations take their interpolation
     *                          tables from it.
     * @param   execStr         Execution-space part of the factory key.
     * @param   implStr         Implementation part of the factory key.
     *
     * @return The newly created block operator.
     */
    static std::shared_ptr<PhysTraceExtractBlockOp<TData>> Create(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse,
        std::string execStr, std::string implStr)
    {
        return ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>::
            template Create<PhysTraceExtractBlockOp>(
                block_idx, exp, dataWarehouse, execStr, implStr);
    }

    /// Block-operator base name; Create() appends the execution space
    /// and implementation to it to form the factory key.
    static inline const std::string name = "BlockPhysTraceExtract";

    /**
     * @brief Extract a single trace of every element of this block into
     * @p outblock at @p outOffset.
     *
     * @param   traceid     Local trace index within the element's shape.
     * @param   inblock     Volume-shaped input block.
     * @param   outblock    Packed trace output block, written at
     *                      @p outOffset.
     * @param   outOffset   Offset of this trace's data within one
     *                      element's packed trace entries, in entries per
     *                      element.
     *
     * @see PhysTraceExtractOp::ExtractTrace() for the calling protocol,
     * for the shapes that are not served and for the open
     * three-dimensional defect this path shares.
     */
    void ExtractTrace(
        const unsigned traceid,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        const unsigned outOffset)
    {
        v_ExtractTrace(traceid, inblock, outblock, outOffset);
    }

    /**
     * @brief Set every one of this block's tangential collocation flags,
     * so that passing false runs the general interpolation where the
     * fast path would otherwise be taken.
     *
     * @details
     * Every implementation stores the same flags in the same way and
     * reads them the same way, so #m_isCollocated lives here and this
     * needs no virtual hook. The setter is wholesale: it overwrites
     * every flag, discarding what the constructor worked out from the
     * point distributions, so passing true does not restore the flags
     * of a block whose points do not in fact coincide.
     *
     * @param   isCollocated    Value written to every flag.
     *
     * @see PhysTraceExtractOp::SetIsCollocated() for what the flags
     * mean and why a caller would want them off.
     */
    void SetIsCollocated(const bool isCollocated)
    {
        for (unsigned i = 0; i < m_isCollocated.size(); ++i)
        {
            m_isCollocated[i] = isCollocated;
        }
    }

protected:
    /// One flag per (normal direction, tangential direction), filled by the
    /// implementation's constructor: the trace points of that direction
    /// coincide with the volume points of the direction they run along,
    /// so that the tangential stage reduces to a straight copy. Six
    /// entries in three dimensions, two in two, one entry in one.
    /// Overwritten wholesale by SetIsCollocated().
    std::vector<bool> m_isCollocated;

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
    PhysTraceExtractBlockOp(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>(block_idx, exp,
                                                                 dataWarehouse)
    {
    }

    ~PhysTraceExtractBlockOp() override = default;

    /// @brief Default hook: an implementation without a per-trace entry
    /// point raises a fatal error. Both the SerialAVX and the Device
    /// implementation override it; see ExtractTrace() for the contract.
    virtual void v_ExtractTrace(
        const unsigned traceid,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        const unsigned outOffset) = 0;
};

} // namespace Nektar::Operators
