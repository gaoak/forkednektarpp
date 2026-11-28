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
 * vector width and optionally interleaved so that each vector lane
 * owns one element (see
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
 * launch-configuration helpers GetElmtsPerBlock() and
 * GetDeviceKernelParameters(), which the device implementations call to
 * size their kernel launches for the SumFac and SumFacTOP strategies.
 * The kernel size-parameter types those launches take (runtime and
 * compile-time forms of the element's mode and quadrature counts) and
 * the launch-bounds helper detail::GetMaxThreadPerBlock() are in
 * ElmtHelper.hpp.
 */

#pragma once

#include <algorithm>
#include <type_traits>

#include <LibUtilities/Backends/DeviceProperties.hpp>
#include <MultiRegions/Common/BlockOperator.hpp>

#include <LibUtilities/BasicUtils/DataWarehouse/BasisDataWarehouse.hpp>
#include <LibUtilities/BasicUtils/DataWarehouse/ModeIndexDataWarehouse.hpp>
#include <LibUtilities/BasicUtils/ShapeType.hpp>
#include <LocalRegions/DataWarehouse/GeometricDataWarehouse.hpp>
#include <StdRegions/DataWarehouse/StdMatDataWarehouse.hpp>

namespace Nektar::MultiRegions
{

/// @brief Passes over its element groups each device thread block is
/// assumed to make in the SumFacTOP grid-size heuristic (see
/// GetDeviceKernelParameters()); tuned per vendor back-end.
#if defined(NEKTAR_ENABLE_CUDA) || defined(SYCL_ENABLE_CUDA)
static constexpr unsigned int defaultPassesPerBlock = 8u;
#elif defined(NEKTAR_ENABLE_HIP) || defined(SYCL_ENABLE_HIP)
static constexpr unsigned int defaultPassesPerBlock = 4u;
#elif defined(SYCL_ENABLE_INTEL)
static constexpr unsigned int defaultPassesPerBlock = 4u;
#else
static constexpr unsigned int defaultPassesPerBlock = 1u;
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
 * Element data is interleaved at the execution space's vector width so
 * that each lane performs the full per-element evaluation on its own
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
 * As with ElmtOp, this class sets the BlockAccessor states accepted
 * by Apply() at compile time; the numerical work happens in the
 * detail:: implementation classes (e.g. detail::BwdTransBlockOpImpl),
 * which override v_Apply(). The sum-factorised implementations, and
 * the Generic ones of the trace operators, dispatch from v_Apply() on
 * the block's shape type, through ShapeSwitch(), into per-shape
 * ShapeBlock definitions that CMake generates from
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
    /// #InBlock and #OutBlock are for the ShapeBlock declarations and
    /// their generated definitions only. The shared ShapeBlock template
    /// that generates the switch cannot restate each operator's field
    /// states, so it takes them from these aliases, set once in the
    /// operator's ElmtBlockOp base clause. Other code should name the
    /// BlockAccessor type directly.
    using InBlock = LibUtilities::BlockAccessor<TData, TFieldIn>;
    /// Accessor type of this operator's output block; see #InBlock.
    using OutBlock = LibUtilities::BlockAccessor<TData, TFieldOut>;

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

    ~ElmtBlockOp() override = default;

    /**
     * @brief Call @p func with the runtime @p shape as a compile-time
     * constant, `std::integral_constant<LibUtilities::ShapeType, ...>`.
     *
     * Used by the v_Apply() overrides that dispatch to ShapeBlock.
     *
     * @param   shape   Shape of the block's elements.
     * @param   func    Callable taking the shape constant.
     */
    template <typename TFunc>
    static void ShapeSwitch(const LibUtilities::ShapeType shape, TFunc &&func)
    {
        using LibUtilities::ShapeType;
        switch (shape)
        {
            case LibUtilities::Seg:
                func(std::integral_constant<ShapeType, LibUtilities::Seg>{});
                break;
            case LibUtilities::Quad:
                func(std::integral_constant<ShapeType, LibUtilities::Quad>{});
                break;
            case LibUtilities::Tri:
                func(std::integral_constant<ShapeType, LibUtilities::Tri>{});
                break;
            case LibUtilities::NodalTri:
                func(std::integral_constant<ShapeType,
                                            LibUtilities::NodalTri>{});
                break;
            case LibUtilities::Hex:
                func(std::integral_constant<ShapeType, LibUtilities::Hex>{});
                break;
            case LibUtilities::Tet:
                func(std::integral_constant<ShapeType, LibUtilities::Tet>{});
                break;
            case LibUtilities::NodalTet:
                func(std::integral_constant<ShapeType,
                                            LibUtilities::NodalTet>{});
                break;
            case LibUtilities::Pyr:
                func(std::integral_constant<ShapeType, LibUtilities::Pyr>{});
                break;
            case LibUtilities::NodalPyr:
                func(std::integral_constant<ShapeType,
                                            LibUtilities::NodalPyr>{});
                break;
            case LibUtilities::Prism:
                func(std::integral_constant<ShapeType, LibUtilities::Prism>{});
                break;
            case LibUtilities::NodalPrism:
                func(std::integral_constant<ShapeType,
                                            LibUtilities::NodalPrism>{});
                break;
            default:
                NEKERROR(ErrorUtil::efatal, "Shape type not implemented");
        }
    }

    /**
     * @brief Call @p func with `std::true_type` if @p append and
     * `std::false_type` otherwise.
     *
     * Used by the v_Apply() overrides to select a kernel's `APPEND`
     * template parameter.
     *
     * @param   append  Add to the output rather than overwrite it.
     * @param   func    Callable taking the append constant.
     */
    template <typename TFunc>
    static void DispatchAppendFlag(const bool append, TFunc &&func)
    {
        if (append)
        {
            func(std::true_type{});
        }
        else
        {
            func(std::false_type{});
        }
    }

    /**
     * @brief Call @p func with `std::false_type` if @p scale is one and
     * `std::true_type` otherwise.
     *
     * Used by the v_Apply() overrides to select a kernel's `SCALE`
     * template parameter.
     *
     * @param   scale   Factor the output is multiplied by.
     * @param   func    Callable taking the scaling constant.
     */
    template <typename TFunc>
    static void DispatchScaleFlag(const TData scale, TFunc &&func)
    {
        if (scale == 1.0)
        {
            func(std::false_type{});
        }
        else
        {
            func(std::true_type{});
        }
    }

    /**
     * @brief Call @p func with `std::true_type` if @p deformed and
     * `std::false_type` otherwise.
     *
     * Used by the v_Apply() overrides to select a kernel's `DEFORMED`
     * template parameter.
     *
     * @param   deformed    The block's elements have deformed geometry.
     * @param   func        Callable taking the deformed constant.
     */
    template <typename TFunc>
    static void DispatchDeformedFlag(const bool deformed, TFunc &&func)
    {
        if (deformed)
        {
            func(std::true_type{});
        }
        else
        {
            func(std::false_type{});
        }
    }

    /// The creators return a std::unique_ptr to this class, which the
    /// factory converts to the shared_ptr it hands out; their deleter
    /// destroys the operator through this destructor.
    friend struct std::default_delete<ElmtBlockOp>;

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
 * @brief Launch geometry of a device kernel, from
 * GetDeviceKernelParameters().
 */
struct DeviceKernelParameters
{
    /// Threads per block.
    unsigned int blockSize = 0u;
    /// Thread blocks.
    unsigned int gridSize = 0u;
    /// Dynamic shared memory per block in bytes.
    unsigned int shmemSize = 0u;
    /// Elements each thread block takes per pass.
    unsigned int elmtsPerBlock = 0u;
};

/// Threads GetElmtsPerBlock() fills with whole SumFacTOP elements by default.
constexpr unsigned int defaultPackingThreads = 128u;

/**
 * @brief Elements each thread block takes per pass.
 *
 * For SumFac one warp, a lane per element. For SumFacTOP as many whole
 * elements, of @p dofsPerElmt threads rounded up to a power of two each, as
 * fit in @p PACKING_THREADS threads, one at least, where an element takes at
 * most half a warp; one otherwise, as more would leave as many threads idle
 * and couple the elements' barriers, and one where @p PACKING_THREADS is zero.
 *
 * Pass the same @p dofsPerElmt to GetDeviceKernelParameters(), usually the
 * GetDofsPerElmt() of the launch's size parameter.
 *
 * @tparam  PACKING_THREADS  Threads to fill with whole elements
 *                           (SumFacTOP), at most the default block size,
 *                           or zero for no packing.
 * @param   dofsPerElmt      Degrees of freedom of one element, a thread
 *                           each (SumFacTOP).
 */
template <typename Implementation,
          unsigned int PACKING_THREADS = defaultPackingThreads>
NEK_FORCE_INLINE static constexpr unsigned int GetElmtsPerBlock(
    [[maybe_unused]] const unsigned int dofsPerElmt)
{
    static_assert(std::is_same_v<Implementation, MultiRegions::SumFac> ||
                      std::is_same_v<Implementation, MultiRegions::SumFacTOP>,
                  "Only SumFac and SumFacTOP kernels use GetElmtsPerBlock().");
    // Packed elements share a block whose launch bounds
    // (GetMaxThreadPerBlock()) are the default block size.
    static_assert(PACKING_THREADS <= NektarSpaces::Device::defaultBlockSize,
                  "PACKING_THREADS exceeds the default block size of the "
                  "launch bounds.");

    if constexpr (std::is_same_v<Implementation, MultiRegions::SumFac>)
    {
        return NektarSpaces::Device::warpSize;
    }
    else
    {
        if (dofsPerElmt <= NektarSpaces::Device::warpSize / 2)
        {
            // Round to power of two.
            unsigned int elmtThreads = 1u;
            while (elmtThreads < dofsPerElmt)
            {
                elmtThreads *= 2u;
            }

            // As many whole elements as fit, one at least.
            return std::max(1u, PACKING_THREADS / elmtThreads);
        }
        else
        {
            return 1u;
        }
    }
}

/**
 * @brief GetElmtsPerBlock() of a compile-time @p DOFS, as a
 * `std::integral_constant`, for a templated GetDofsPerElmt().
 */
template <typename Implementation,
          unsigned int PACKING_THREADS = defaultPackingThreads,
          unsigned int DOFS>
NEK_FORCE_INLINE static constexpr auto GetElmtsPerBlock(
    std::integral_constant<unsigned int, DOFS>)
{
    return std::integral_constant<
        unsigned int,
        GetElmtsPerBlock<Implementation, PACKING_THREADS>(DOFS)>{};
}

/**
 * @brief Launch geometry of a SumFac or SumFacTOP kernel.
 *
 * For SumFac a block is one warp of @p elmtsPerBlock lanes, one element
 * each, and the grid covers the elements in @p passesPerBlock passes, the
 * kernels striding over the grid. For SumFacTOP the block is the
 * elements' threads, @p dofsPerElmt rounded up to a power of two each where
 * it takes several, rounded up to whole warps, at most @p maxBlockSize, by
 * default the device's default block size, and the grid follows a
 * persistent-blocks heuristic bounded by thread and shared-memory occupancy,
 * at most one block per group of @p elmtsPerBlock elements.
 *
 * @tparam Implementation  SumFac or SumFacTOP.
 *
 * @param   dofsPerElmt     Degrees of freedom of one element, a thread
 *                          each (SumFacTOP).
 * @param   nelmt           Number of elements, including padding.
 * @param   shmemSize       Dynamic shared memory per block in bytes.
 * @param   elmtsPerBlock   Elements per block per pass, from
 *                          GetElmtsPerBlock(); zero, the default, for a
 *                          warp under SumFac and one element under
 *                          SumFacTOP.
 * @param   passesPerBlock  Passes, one group of elmtsPerBlock elements
 *                          each, each block is assumed to make; zero, the
 *                          default, for one under SumFac and
 *                          defaultPassesPerBlock under SumFacTOP.
 * @param   maxBlockSize    Most threads per block (SumFacTOP). The kernels
 *                          stride their threads over an element's degrees
 *                          of freedom, so the default block size is safe
 *                          for any element; a kernel that keeps one thread
 *                          on each and cannot stride passes
 *                          NektarSpaces::Device::maximumBlockSize and
 *                          refuses larger elements.
 */
template <typename Implementation>
NEK_FORCE_INLINE static DeviceKernelParameters GetDeviceKernelParameters(
    [[maybe_unused]] const unsigned int dofsPerElmt, const size_t nelmt,
    const unsigned int shmemSize = 0u, const unsigned int elmtsPerBlock = 0u,
    const unsigned int passesPerBlock = 0u,
    [[maybe_unused]] const unsigned int maxBlockSize =
        NektarSpaces::Device::defaultBlockSize)
{
    static_assert(std::is_same_v<Implementation, MultiRegions::SumFac> ||
                      std::is_same_v<Implementation, MultiRegions::SumFacTOP>,
                  "Only SumFac and SumFacTOP kernels launch through "
                  "GetDeviceKernelParameters().");
    constexpr size_t maxGridSize                 = 2147483647;
    constexpr unsigned int persistentQueueFactor = 2u;

    // UseSharedWorkSpace() sized the launch from these limits.
    CheckDeviceArchProperties();

    DeviceKernelParameters kernelParams;

    if constexpr (std::is_same_v<Implementation, MultiRegions::SumFac>)
    {
        // Zero elmtsPerBlock defaults to a warp, one lane per element, as
        // GetElmtsPerBlock() gives.
        kernelParams.elmtsPerBlock =
            elmtsPerBlock ? elmtsPerBlock : NektarSpaces::Device::warpSize;
        ASSERTL0(kernelParams.elmtsPerBlock == NektarSpaces::Device::warpSize,
                 "SumFac kernels take one warp of elements per block.");

        // Each block processes elmtsPerBlock elements per pass.
        const size_t ngroups = (nelmt + kernelParams.elmtsPerBlock - 1u) /
                               kernelParams.elmtsPerBlock;

        // Zero passesPerBlock defaults to one pass, a block per group.
        const unsigned int passes = passesPerBlock ? passesPerBlock : 1u;

        // One lane per element: elmtsPerBlock is the warp size.
        kernelParams.blockSize = kernelParams.elmtsPerBlock;
        kernelParams.gridSize  = static_cast<unsigned int>(
            std::min((ngroups + passes - 1u) / passes, maxGridSize));
        kernelParams.shmemSize = shmemSize;
    }
    else if constexpr (std::is_same_v<Implementation, MultiRegions::SumFacTOP>)
    {
        // Zero elmtsPerBlock defaults to one element per block, as
        // GetElmtsPerBlock() gives with a PACKING_THREADS of zero.
        kernelParams.elmtsPerBlock = elmtsPerBlock ? elmtsPerBlock : 1u;

        // Each block processes elmtsPerBlock elements per pass.
        const size_t ngroups = (nelmt + kernelParams.elmtsPerBlock - 1u) /
                               kernelParams.elmtsPerBlock;

        // Zero passesPerBlock defaults to defaultPassesPerBlock.
        const unsigned int passes =
            passesPerBlock ? passesPerBlock : defaultPassesPerBlock;

        // The elements' threads, rounded up to whole warps, at most
        // maxBlockSize.
        unsigned int threadsPerBlock;
        if (kernelParams.elmtsPerBlock == 1u)
        {
            threadsPerBlock = dofsPerElmt;
        }
        else
        {
            // Several elements of at most half a warp each, as
            // GetElmtsPerBlock() packs them: round each up to a power of two
            // so that a warp holds whole elements.
            unsigned int elmtThreads = 1u;
            while (elmtThreads < dofsPerElmt)
            {
                elmtThreads *= 2u;
            }
            threadsPerBlock = kernelParams.elmtsPerBlock * elmtThreads;
        }

        constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;
        kernelParams.blockSize =
            std::min((threadsPerBlock + warpsize - 1u) / warpsize * warpsize,
                     maxBlockSize);

        GetDeviceProperties::CheckSharedMemoryUsage(shmemSize);

        const auto smCount =
            static_cast<size_t>(GetDeviceProperties::NumMultiProcessors());
        // Number of blocks that can achieve full occupancy on each SM
        const size_t blocksByThreads =
            GetDeviceProperties::MaxThreadsPerMultiprocessor() /
            kernelParams.blockSize;
        // Number of blocks that can fully utilize shared memory on each SM
        const size_t blocksByShared =
            (shmemSize == 0)
                ? blocksByThreads
                : GetDeviceProperties::SharedMemoryPerMultiprocessor() /
                      shmemSize;
        const size_t residentBlocks = std::min(blocksByThreads, blocksByShared);
        // Number of blocks to launch to keep all SMs busy, controlled by
        // a customizable QueueFactor. This serves as a lower bound on grid size
        const size_t launchBlocks =
            smCount * residentBlocks * persistentQueueFactor;
        const size_t cappedLaunchBlocks =
            std::max<size_t>(1u, std::min(launchBlocks, maxGridSize));

        // 1. If nelmt is big, then grid size is determined by ngroups
        // over the passes to ensure each block has enough work to do;
        // 2. If nelmt is small, then we try to ensure SMs are busy by
        // launching enough blocks;
        // 3. If nelmt is very small, then each block only process
        // one group to avoid unnecessary idling.
        kernelParams.gridSize = static_cast<unsigned int>(
            std::min(ngroups, std::max((ngroups + passes - 1u) / passes,
                                       cappedLaunchBlocks)));
        kernelParams.shmemSize = shmemSize;
    }

    return kernelParams;
}

/**
 * @brief Whether a kernel holds its stage intermediates in shared
 * memory rather than in the global-memory workspace.
 *
 * For SumFac, true when the thread block's share of the workspace fits
 * in the device's shared memory per block and leaves at least
 * `1 / minOccupancyDivisor` of the multiprocessor's threads resident.
 * For SumFacTOP, true when the thread block's whole request, workspace
 * included, fits in the device's shared memory per block. Always false
 * for the other implementations.
 *
 * The device limits are the compile-time ones of GetDeviceArchProperties()
 * where the build's target architecture is listed, so that the result is
 * then a constant expression for a constant @p shmemsize, and are queried
 * from the device otherwise.
 *
 * @tparam Implementation   Implementation tag.
 *
 * @param   shmemsize   Shared memory of one thread block in bytes with
 *                      the workspace staged in it.
 *
 * @return Whether to place the workspace in shared memory.
 */
template <typename Implementation>
NEK_FORCE_INLINE static constexpr bool UseSharedWorkSpace(
    const size_t shmemsize)
{
    constexpr size_t minOccupancyDivisor = 4u;

    constexpr DeviceArchProperties arch = GetDeviceArchProperties();
    DeviceArchProperties device         = arch;
    if constexpr (!arch.known)
    {
        device.sharedMemoryPerBlock =
            GetDeviceProperties::SharedMemoryPerBlock();
        device.sharedMemoryPerMultiprocessor =
            GetDeviceProperties::SharedMemoryPerMultiprocessor();
        device.maxThreadsPerMultiprocessor = static_cast<size_t>(
            GetDeviceProperties::MaxThreadsPerMultiprocessor());
    }

    if constexpr (std::is_same_v<Implementation, MultiRegions::SumFac>)
    {
        if (shmemsize == 0 || shmemsize > device.sharedMemoryPerBlock)
        {
            return false;
        }
        else
        {
            constexpr size_t warpsize = NektarSpaces::Device::warpSize;

            const size_t threadsByShared =
                device.sharedMemoryPerMultiprocessor / shmemsize * warpsize;

            return threadsByShared * minOccupancyDivisor >=
                   device.maxThreadsPerMultiprocessor;
        }
    }
    else if constexpr (std::is_same_v<Implementation, MultiRegions::SumFacTOP>)
    {
        return shmemsize <= device.sharedMemoryPerBlock;
    }
    else
    {
        return false;
    }
}

/**
 * @brief Elements whose workspace the global-memory workspace holds at
 * once.
 *
 * Every element of the block for SumFac. For SumFacTOP the elements of
 * one group per thread block, each block reusing its slice for the groups
 * it takes.
 *
 * @tparam Implementation   Implementation tag.
 *
 * @param   nelmt          Elements in the block, padding included.
 * @param   gridsize       Thread blocks in the grid's first dimension.
 * @param   elmtsPerBlock  Elements per thread block, from
 *                         GetDeviceKernelParameters().
 *
 * @return Number of per-element workspaces to allocate per component.
 */
template <typename Implementation>
NEK_FORCE_INLINE static size_t WorkSpaceNumElmts(
    const size_t nelmt, const unsigned int gridsize,
    [[maybe_unused]] const unsigned int elmtsPerBlock)
{
    if constexpr (std::is_same_v<Implementation, MultiRegions::SumFac>)
    {
        return nelmt;
    }
    else if constexpr (std::is_same_v<Implementation, MultiRegions::SumFacTOP>)
    {
        return gridsize * elmtsPerBlock;
    }
}
#endif

} // namespace Nektar::MultiRegions
