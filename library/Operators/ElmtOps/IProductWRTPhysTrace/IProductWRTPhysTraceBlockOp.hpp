///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTPhysTraceBlockOp.hpp
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
 * @file IProductWRTPhysTraceBlockOp.hpp
 * @brief Per-block interface class of the discontinuous Galerkin lifting
 * operator IProductWRTPhysTrace.
 */

#pragma once

#include <vector>

#include "Operators/ElmtOps/ElmtBlockOp.hpp"

namespace Nektar::Operators
{

/**
 * @brief Per-block interface of the trace inner product: lifts the
 * traces of every element of one block into that block's volume
 * physical-space storage.
 *
 * @details
 * Fixes the BlockAccessor states accepted by Apply() -- physical space
 * in, physical space out -- with the input block holding the element's
 * packed trace values and the output block its volume quadrature points.
 * See IProductWRTPhysTraceOp for both layouts and for the mathematics.
 *
 * On top of ElmtBlockOp this class adds three things:
 * - #m_append, the accumulate flag of the bulk path, with SetAppend();
 * - IProductWRTPhysTrace(), the per-trace entry point;
 * - #m_isCollocated, the tangential collocation flags the
 *   implementations fill, with SetIsCollocated() to overwrite them.
 *
 * The per-trace entry point is a virtual hook whose base implementation
 * raises a fatal error, so an implementation that does not provide it
 * fails loudly rather than silently doing nothing. Clearing the
 * collocation flags needs no such hook: the flags live here and every
 * implementation reads them the same way.
 *
 * The concrete implementations are the
 * detail::IProductWRTPhysTraceBlockOpImpl definitions in
 * IProductWRTPhysTraceSerialAVXGeneric.hpp (Serial and AVX) and
 * IProductWRTPhysTraceDeviceGeneric.hpp (Device). Their per-shape entry
 * points, the ShapeBlock() specialisations, are generated from
 * IProductWRTPhysTraceGenericBlockOp.cpp.in, one translation unit per
 * shape, execution space and data type, so the size-templated
 * OperatorND() instantiations of the SumFac families are kept.
 *
 * There is a single implementation, registered under
 * Operators::Generic: only one trace inner-product algorithm exists,
 * and no StdMat/SumFac/SumFacTOP choice is to be made here. A caller
 * asking for another implementation -- StdMat or SumFac from the
 * whole-field operator selection, SumFacTOP from a Device run -- is
 * served that one registration by ElmtBlockOp::Create(), which falls
 * back to the `"Generic"` factory key whenever the requested key is
 * absent.
 *
 * Registering under Generic is also what keeps a SumFacTOP run correct.
 * The kernels index the trace data at warpSize interleave;
 * instantiating them with the SumFacTOP tag would have set an
 * implementation width of one and produced wrong answers for every
 * SumFacTOP DG case on real hardware -- invisibly in a DEVICEONHOST
 * build, where warpSize is 1 and the two interleaves coincide. With one
 * registration there is no such tag to instantiate against.
 *
 * @tparam TData Floating-point type of the field data.
 *
 * @see ElmtBlockOp for the Apply()/v_Apply() contract;
 * IProductWRTPhysTraceOp for the whole-field interface driving this
 * class.
 */
template <typename TData>
class IProductWRTPhysTraceBlockOp
    : public ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>
{
public:
    /**
     * @brief Create one trace inner-product block operator through the
     * block-operator factory, under the key
     * `"BlockIProductWRTPhysTrace" + execStr + implStr`.
     *
     * @param   block_idx       Index of the block within the expansion
     *                          list's Collections.
     * @param   exp             Representative expansion of the block
     *                          (its first element).
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list; the
     *                          implementations take their interpolation
     *                          tables, trace weights and trace Jacobian
     *                          from it.
     * @param   execStr         Execution-space part of the factory key.
     * @param   implStr         Implementation part of the factory key.
     *
     * @return The newly created block operator.
     */
    static std::shared_ptr<IProductWRTPhysTraceBlockOp<TData>> Create(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse,
        std::string execStr, std::string implStr)
    {
        return ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>::
            template Create<IProductWRTPhysTraceBlockOp>(
                block_idx, exp, dataWarehouse, execStr, implStr);
    }

    /// Block-operator base name; Create() appends the execution space
    /// and implementation to it to form the factory key.
    static inline const std::string name = "BlockIProductWRTPhysTrace";

    /**
     * @brief Lift a single trace of every element of this block into
     * @p outblock.
     *
     * Whether @p outblock is zeroed first or accumulated onto is
     * selected by #m_append, exactly as it is on the bulk path. Because
     * every trace contributes to the same volume-shaped array, clearing
     * it is only meaningful on the first trace of a sequence: a caller
     * assembling the whole boundary trace by trace therefore clears on
     * that first trace and appends on the rest, or pre-clears the array
     * and appends throughout.
     *
     * @param   traceid     Local trace index within the element's shape.
     * @param   inblock     Packed trace input block.
     * @param   inOffset    Offset of this trace's data within one
     *                      element's packed trace entries, in entries
     *                      per element.
     * @param   outblock    Volume-shaped output block, accumulated into.
     *
     * @see IProductWRTPhysTraceOp::IProductWRTPhysTrace() for the
     * calling protocol and for the shapes that are not served.
     */
    void IProductWRTPhysTrace(
        const unsigned traceid,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        const unsigned inOffset,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock)

    {
        v_IProductWRTPhysTrace(traceid, inblock, inOffset, outblock);
    }

    /**
     * @brief Set every one of this block's tangential collocation flags,
     * so that passing false runs the general contraction where the fast
     * path would otherwise be taken.
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
     * @see IProductWRTPhysTraceOp::SetIsCollocated() for what the flags
     * mean and why a caller would want them off.
     */
    void SetIsCollocated(const bool isCollocated)
    {
        for (unsigned i = 0; i < m_isCollocated.size(); ++i)
        {
            m_isCollocated[i] = isCollocated;
        }
    }

    /**
     * @brief Set the accumulate flag of the bulk path.
     *
     * @param   append  New value of #m_append.
     */
    void SetAppend(bool append)
    {
        m_append = append;
    }

protected:
    /// Whether the bulk Apply() path accumulates onto the output block
    /// rather than zeroing it and writing. Default false.
    ///
    /// Read by the Serial/AVX one-, two- and three-dimensional paths and
    /// by the Device one- and three-dimensional paths. The Device
    /// two-dimensional bulk path is not among them: it instantiates its
    /// kernel with the append template argument fixed to `false` and so
    /// always overwrites, whatever this flag holds. The per-trace
    /// IProductWRTPhysTrace() entry does not read it either, always
    /// appending instead. No unit test sets the flag on this operator,
    /// so the appending bulk path is currently unexercised.
    bool m_append = false;

    /// One flag per (normal direction, tangential direction), filled by the
    /// implementation's constructor: the trace points of that direction
    /// coincide with the volume points of the direction they run along,
    /// so that the tangential contraction reduces to a pointwise
    /// multiply by the trace weights and the Jacobian. Six entries in
    /// three dimensions, two in two, one entry in one. Cleared
    /// wholesale by SetIsCollocated().
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
    IProductWRTPhysTraceBlockOp(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>(block_idx, exp,
                                                                 dataWarehouse)
    {
    }

    ~IProductWRTPhysTraceBlockOp() override = default;

    /// @brief Default hook: an implementation without a per-trace entry
    /// point raises a fatal error. Both SumFac implementations override
    /// it; see IProductWRTPhysTrace() for the contract.
    virtual void v_IProductWRTPhysTrace(
        const unsigned traceid,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        const unsigned inOffset,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock) = 0;
};

} // namespace Nektar::Operators
