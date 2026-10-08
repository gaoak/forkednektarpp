///////////////////////////////////////////////////////////////////////////////
//
// File: ElmtBlockOp.hpp
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
 * @file ElmtBlockOp.hpp
 * @brief Block-level base class, implementation-strategy tags and
 * device launch helpers shared by every element-operator family.
 *
 * @details
 * ### Blocks
 * A block is a contiguous run of elements with identical shape, basis
 * and quadrature -- one block per Collection of the expansion list.
 * Grouping like elements lets one kernel process a whole batch: within
 * a block every element carries the same number of data values, so the
 * block's storage is a rectangular array of `num_data` values per
 * element and component, padded to a whole multiple of the maximum
 * vector width and optionally interleaved so that vector lanes (SIMD
 * on AVX, warp lanes on Device) each own one element (see
 * LibUtilities::BlockAttributes and LibUtilities::BlockAccessor in
 * LibUtilities/BasicUtils/Field/Block.hpp). ElmtBlockOp is the base
 * class of the per-family block operators (e.g. BwdTransBlockOp) that
 * apply one element operation to one such block; the whole-field
 * ElmtOp (ElmtOp.hpp) holds one of them per block and drives them in
 * its Apply loop.
 *
 * ### Implementation strategies
 * The tag structs StdMat, SumFac, SumFacTOP and Generic name the
 * block-level implementation strategies. The tag's `name` forms part
 * of the block-operator factory key, and on the device back-ends the
 * tag type also selects between kernel variants at compile time.
 *
 * ### Device launch helpers
 * When NEKTAR_ENABLE_DEVICE is set this file additionally provides the
 * launch-configuration helpers GetDeviceBlockSize() and
 * GetDeviceGridSize(), which the device implementations call to size
 * their kernel launches for the SumFac and SumFacTOP strategies.
 * The kernel size-parameter types those launches take (runtime and
 * compile-time forms of the element's mode and quadrature counts) and
 * the launch-bounds helper detail::GetMaxThreadPerBlock() are in
 * ElmtHelper.hpp.
 */

#pragma once

#include <algorithm>

#include "LibUtilities/Backends/DeviceProperties.hpp"
#include <MultiRegions/Common/BlockOperator.hpp>

#include <LibUtilities/BasicUtils/DataWarehouse/BasisDataWarehouse.hpp>
#include <LibUtilities/BasicUtils/DataWarehouse/ModeIndexDataWarehouse.hpp>
#include <LocalRegions/DataWarehouse/GeometricDataWarehouse.hpp>
#include <StdRegions/DataWarehouse/StdMatDataWarehouse.hpp>

namespace Nektar::MultiRegions
{

/// @brief Elements each device thread block is assumed to take per pass
/// in the SumFacTOP grid-size heuristic (see GetDeviceGridSize()); tuned
/// per vendor back-end.
#if defined(NEKTAR_ENABLE_CUDA) || defined(SYCL_ENABLE_CUDA)
static constexpr unsigned int defaultElmtsPerBlock = 8u;
#elif defined(NEKTAR_ENABLE_HIP) || defined(SYCL_ENABLE_HIP)
static constexpr unsigned int defaultElmtsPerBlock = 4u;
#elif defined(SYCL_ENABLE_INTEL)
static constexpr unsigned int defaultElmtsPerBlock = 4u;
#else
static constexpr unsigned int defaultElmtsPerBlock = 1u;
#endif

// Core implementation types.
/**
 * @brief Implementation tag: apply the operator as one dense
 * standard-region matrix multiplied across all elements of the block.
 */
struct StdMat
{
    static inline const std::string name = "StdMat";
};

/**
 * @brief Implementation tag: sum-factorised tensor-product evaluation
 * with one element per vector lane.
 *
 * Element data is interleaved at the vector width -- the SIMD width on
 * the AVX back-end, the warp size on the Device back-end -- so that
 * each lane performs the full per-element evaluation on its own
 * element while the batch progresses in lock-step.
 */
struct SumFac
{
    static inline const std::string name = "SumFac";
};

/**
 * @brief Implementation tag: sum-factorised evaluation in which the
 * threads of one device block cooperate on a single element.
 *
 * Threads are indexed over the work items (output points or modes)
 * within an element, with intermediates staged in shared memory and
 * synchronised between the sum-factorisation stages, rather than over
 * elements as in SumFac. This exposes more parallelism per element;
 * it is used by the device back-ends only.
 */
struct SumFacTOP
{
    static inline const std::string name = "SumFacTOP";
};

/**
 * @brief Implementation tag: strategy-agnostic implementation, used by
 * families with a single implementation (e.g. Expression and
 * MultiplyByElmtInvMass) and as the fall-back key when a requested
 * implementation is not registered (see ElmtBlockOp::Create()).
 */
struct Generic
{
    static inline const std::string name = "Generic";
};

/**
 * @brief Base class of the per-family block operators: applies one
 * element operation to every element of one block.
 *
 * @details
 * A block operator is created per block by ElmtOp::Create and bound to
 * the block's index, its representative expansion -- the block's first
 * element, from which the shape, basis and quadrature shared by every
 * element of the block are read, along with the block's geometry type
 * (regular or deformed) and coordinate dimension -- and the expansion
 * list's data warehouse, through which basis matrices and similar
 * tables are built once and shared between operators. Geometric
 * factors (Jacobians, derivative factors) are per element, not
 * shared; implementations needing them fetch them for the whole block
 * through the data warehouse (see
 * LocalRegions/DataWarehouse/GeometricDataWarehouse.hpp).
 *
 * As with ElmtOp, this class fixes the BlockAccessor states accepted
 * by Apply() at compile time; the numerical work happens in the
 * detail:: implementation classes (e.g. detail::BwdTransBlockOpImpl),
 * which override v_Apply(). The sum-factorised implementations, and
 * the Generic ones of the trace operators, dispatch from v_Apply() on
 * the block's shape type into per-shape ShapeBlock definitions that
 * CMake generates from
 * LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in, one
 * translation unit per shape; each ShapeBlock switches on the
 * element's size counts and calls the implementation's OperatorND
 * with a compile-time size parameter when they fall inside the
 * configured switch range and a runtime one otherwise, warning once
 * per block operator in the latter case (see #m_warnOnceTemplate and
 * ElmtHelper.hpp). The StdMat and the remaining Generic
 * implementations declare no ShapeBlock and act on the block directly.
 *
 * @tparam TFieldIn  Field state of the input block.
 * @tparam TFieldOut Field state of the output block.
 * @tparam TData     Floating-point type of the field data.
 *
 * @see BlockOperator for the static per-stream workspace shared by all
 * block operators; ElmtOp for the whole-field loop this class is
 * driven from.
 */
template <FieldState TFieldIn, FieldState TFieldOut, typename TData>
class ElmtBlockOp : public BlockOperator<TData>
{
public:
    /// @brief Accessor type of this operator's input block.
    ///
    /// Derived classes and the generated ShapeBlock definitions use
    /// #InBlock and #OutBlock rather than restating the field states, so
    /// each operator states them in exactly one place: its ElmtBlockOp
    /// base clause in the family's BlockOp header.
    using InBlock = LibUtilities::BlockAccessor<TData, TFieldIn>;
    /// Accessor type of this operator's output block; see #InBlock.
    using OutBlock = LibUtilities::BlockAccessor<TData, TFieldOut>;

    ~ElmtBlockOp() override = default;

    /**
     * @brief Create one block operator through the block-operator
     * factory.
     *
     * The factory key is `TOperator::name + execStr + implStr`. If no
     * implementation is registered under that key, the "Generic"
     * implementation for the same execution space is tried before
     * failing, so families providing only a Generic implementation
     * (e.g. Expression) work whatever implementation is requested.
     * The product is handed back through a static_pointer_cast to
     * @p TOperator, so the registered creator must construct that
     * class or one derived from it.
     *
     * @tparam TOperator  Family block-operator class (e.g.
     *                    BwdTransBlockOp) supplying the static `name`.
     *
     * @param   block_idx       Index of the block within the expansion
     *                          list's Collections.
     * @param   exp             Representative expansion of the block
     *                          (its first element).
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     * @param   execStr         Execution-space part of the factory
     *                          key.
     * @param   implStr         Implementation part of the factory key.
     *
     * @return The newly created block operator. Creation raises a
     * fatal error (throws ErrorUtil::NekError) if neither the
     * requested nor the Generic key is registered.
     */
    template <template <typename> typename TOperator>
    static std::shared_ptr<TOperator<TData>> Create(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse,
        const std::string &execStr, const std::string &implStr)
    {
        std::string requestedKey = TOperator<TData>::name + execStr + implStr;

        BlockOperatorFactory<TData> &factory = GetBlockOperatorFactory<TData>();

        // No suitable operator was found.
        if (!factory.ModuleExists(requestedKey))
        {
            // See if there is a Generic implementation.
            auto requestedKey0 = requestedKey;
            requestedKey       = TOperator<TData>::name + execStr + "Generic";

            if (!factory.ModuleExists(requestedKey))
            {
                std::stringstream msg;
                msg << "No such operator: " << requestedKey0 << std::endl;
                factory.PrintAvailableClasses(msg);
                NEKERROR(ErrorUtil::efatal, msg.str());
            }
        }

        return std::static_pointer_cast<TOperator<TData>>(
            factory.CreateInstance(requestedKey, block_idx, exp,
                                   dataWarehouse));
    }

    /**
     * @brief Apply the element operation to every element of the
     * block.
     *
     * @param   inblock     Input block accessor in state @p TFieldIn.
     * @param   outblock    Output block accessor in state
     *                      @p TFieldOut; must belong to the same block
     *                      of a conforming field, so that element
     *                      count and padding match @p inblock. Its
     *                      interleave width need not match: the
     *                      implementations reshape the data to the
     *                      width they work at, restore it, and set the
     *                      output's recorded width to the input's on
     *                      return (see v_Apply()).
     */
    void Apply(LibUtilities::BlockAccessor<TData, TFieldIn> &inblock,
               LibUtilities::BlockAccessor<TData, TFieldOut> &outblock)
    {
        this->v_Apply(inblock, outblock);
    }

    /// @brief Call operator; equivalent to Apply().
    void operator()(LibUtilities::BlockAccessor<TData, TFieldIn> &inblock,
                    LibUtilities::BlockAccessor<TData, TFieldOut> &outblock)
    {
        this->v_Apply(inblock, outblock);
    }

protected:
    /**
     * @brief Set once the generated ShapeBlock switch code has warned
     * that an element size outside the templated switch range fell
     * back to the runtime-sized OperatorND call, so that the warning
     * is issued only once per block operator (see the BlockOpSwitch
     * templates in LibUtilities/BasicUtils/Switch).
     */
    bool m_warnOnceTemplate = false;

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
    ElmtBlockOp(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : BlockOperator<TData>(block_idx, exp, dataWarehouse)
    {
    }

    /**
     * @brief Implementation hook for Apply().
     *
     * Overrides must apply the element operation to all elements of
     * the block, for every component of @p inblock, and leave
     * @p outblock's recorded interleave width consistent with the
     * layout they wrote (the implementations reshape back to the
     * input's width and record it; see e.g.
     * detail::BwdTransBlockOpImpl::OperatorNDImpl). Padding elements
     * may be processed as full elements -- their storage is allocated
     * and their results are ignored.
     *
     * @param   inblock     Input block accessor in state @p TFieldIn.
     * @param   outblock    Output block accessor in state
     *                      @p TFieldOut.
     */
    virtual void v_Apply(
        LibUtilities::BlockAccessor<TData, TFieldIn> &inblock,
        LibUtilities::BlockAccessor<TData, TFieldOut> &outblock) = 0;
};

#if defined(NEKTAR_ENABLE_DEVICE)
/**
 * @brief Number of threads per device block for a kernel launch of the
 * given implementation.
 *
 * For SumFac one warp is used, with each lane owning one element; the
 * kernels grid-stride over any remaining elements, so the grid size
 * from GetDeviceGridSize() supplies the rest of the parallelism. For
 * SumFacTOP the requested per-element work size is rounded up to a
 * whole number of warps and capped at the device's default block size;
 * the kernels stride their local thread index over the element's work
 * items, so the cap is safe when @p elmtBlockSize exceeds it. Other
 * implementations return 0 (they do not launch through this path).
 *
 * @tparam Implementation   Implementation tag (SumFac or SumFacTOP).
 *
 * @param   elmtBlockSize   Per-element work-item count the block
 *                          should cover (the element's total mode or
 *                          quadrature-point count, whichever the
 *                          kernel is indexed over); unused for SumFac.
 *
 * @return Threads per block. Constexpr, so with a compile-time
 * argument it is also usable inside `__LAUNCH_BOUNDS__` (see
 * detail::GetMaxThreadPerBlock() in ElmtHelper.hpp).
 */
template <typename Implementation>
NEK_FORCE_INLINE static constexpr unsigned int GetDeviceBlockSize(
    [[maybe_unused]] const unsigned int elmtBlockSize)
{
    if constexpr (std::is_same_v<Implementation, MultiRegions::SumFac>)
    {
        constexpr auto warpsize = NektarSpaces::Device::warpSize;
        return warpsize;
    }
    else if constexpr (std::is_same_v<Implementation, MultiRegions::SumFacTOP>)
    {
        constexpr auto warpsize = NektarSpaces::Device::warpSize;
        return std::min(((elmtBlockSize + warpsize - 1u) / warpsize) * warpsize,
                        NektarSpaces::Device::defaultBlockSize);
    }
    else
    {
        return 0;
    }
}

/**
 * @brief Number of thread blocks (grid size) for a kernel launch of
 * the given implementation.
 *
 * For SumFac, with one element per thread, this is simply
 * `ceil(nelmt / blockSize)`, capped at the maximum grid dimension.
 * For SumFacTOP a persistent-blocks heuristic is used: the number of
 * blocks resident per multiprocessor is bounded by both the thread and
 * the shared-memory occupancy limits, and enough blocks are launched
 * to keep every multiprocessor's queue filled (scaled by
 * `persistentQueueFactor`), but never more than one block per element
 * and never fewer than one per @p elmtsPerBlock elements -- the
 * in-code comments walk through the three regimes. The requested
 * shared memory is additionally validated against the device limit.
 *
 * @tparam Implementation   Implementation tag (SumFac or SumFacTOP;
 *                          others return 0).
 * @tparam elmtsPerBlock    Elements each block is assumed to process
 *                          per pass (SumFacTOP only); defaults to the
 *                          vendor-tuned defaultElmtsPerBlock.
 *
 * @param   nelmt       Number of elements to process, including
 *                      padding.
 * @param   blockSize   Threads per block, from GetDeviceBlockSize().
 * @param   shmemsize   Dynamic shared memory per block in bytes
 *                      (SumFacTOP only).
 *
 * @return Number of blocks to launch.
 */
template <typename Implementation,
          unsigned int elmtsPerBlock = defaultElmtsPerBlock>
NEK_FORCE_INLINE static unsigned int GetDeviceGridSize(
    const size_t nelmt, const unsigned int blockSize,
    [[maybe_unused]] const unsigned int shmemsize)
{
    constexpr size_t maxGridSize                 = 2147483647;
    constexpr unsigned int persistentQueueFactor = 2u;

    if constexpr (std::is_same_v<Implementation, MultiRegions::SumFac>)
    {
        return std::min((nelmt + blockSize - 1u) / blockSize, maxGridSize);
    }
    else if constexpr (std::is_same_v<Implementation, MultiRegions::SumFacTOP>)
    {
        GetDeviceProperties::CheckSharedMemoryUsage(shmemsize);

        const auto smCount =
            static_cast<size_t>(GetDeviceProperties::NumMultiProcessors());
        // Number of blocks that can achieve full occupancy on each SM
        const size_t blocksByThreads =
            GetDeviceProperties::MaxThreadsPerMultiprocessor() / blockSize;
        // Number of blocks that can fully utilize shared memory on each SM
        const size_t blocksByShared =
            (shmemsize == 0)
                ? blocksByThreads
                : GetDeviceProperties::SharedMemoryPerMultiprocessor() /
                      shmemsize;
        const size_t residentBlocks = std::min(blocksByThreads, blocksByShared);
        // Number of blocks to launch to keep all SMs busy, controlled by
        // a customizable QueueFactor. This serves as a lower bound on grid size
        const size_t launchBlocks =
            smCount * residentBlocks * persistentQueueFactor;
        const size_t cappedLaunchBlocks =
            std::max<size_t>(1u, std::min(launchBlocks, maxGridSize));

        // Each block processes elmtsPerBlock elements per queue
        // iteration.
        const size_t ngroups = (nelmt + elmtsPerBlock - 1u) / elmtsPerBlock;

        // 1. If nelmt is big, then grid size is determined by ngroups
        // to ensure each block has enough work to do;
        // 2. If nelmt is small, then we try to ensure SMs are busy by
        // launching enough blocks;
        // 3. If nelmt is very small, then each block only process
        // one element to avoid unnecessary idling.
        const unsigned int gridsize = static_cast<unsigned int>(
            std::min(nelmt, std::max(ngroups, cappedLaunchBlocks)));

        return gridsize;
    }
    else
    {
        return 0;
    }
}
#endif

} // namespace Nektar::MultiRegions
