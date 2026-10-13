///////////////////////////////////////////////////////////////////////////////
//
// File: ExpressionBlockOpDeviceGenericSYCLHelper.hpp
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
// Description: Generation, runtime compilation and launch of the expression
// kernels on a SYCL device.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file ExpressionBlockOpDeviceGenericSYCLHelper.hpp
 * @brief Generation, runtime compilation and launch of the expression
 * kernels on the SYCL back-ends.
 *
 * @details
 * Offloading to CUDA or HIP, the kernels are those of
 * ExpressionBlockOpDeviceGenericHIPCUDAHelper.hpp, and only the compilation
 * and launch are wrapped in host operations ordered on the block's
 * queue. On Intel, SYCL source is generated and compiled through the
 * oneAPI kernel-compiler extension, which the device must support.
 */

#pragma once

#include <string>
#include <vector>

#include <MultiRegions/ElmtOps/Expression/ExpressionBlockOpDeviceGenericHelper.hpp>

#if defined(SYCL_ENABLE_CUDA) || defined(SYCL_ENABLE_HIP)
// Offloading to CUDA or HIP, the kernels are those the CUDA/HIP runtime
// compiler builds. SYCL only carries the work to the native stream behind the
// block's queue, through an operation the host runs in queue order, which
// AdaptiveCpp and DPC++ spell differently.
#if defined(SYCL_ENABLE_CUDA)
#include "LibUtilities/Backends/CUDA_Host_API.hpp"
#elif defined(SYCL_ENABLE_HIP)
#include "LibUtilities/Backends/HIP_Host_API.hpp"
#endif

#include <MultiRegions/ElmtOps/Expression/ExpressionBlockOpDeviceGenericHIPCUDAHelper.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Compile the generated source with BuildExpressionKernels(), in
 * a host operation enqueued on the block's queue.
 *
 * @tparam TData    Floating-point type the kernels are compiled for.
 *
 * @param   kernels     Compiled kernels, replaced when the operation
 *                      runs; must outlive it.
 * @param   kernel_src  Source from GetExpressionKernelSource().
 * @param   streamID    Stream index of the block.
 */
template <typename TData>
void CompileExpressionKernels(ExpressionKernels &kernels,
                              const std::string &kernel_src,
                              const unsigned int streamID)
{
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);
    sycl::event e  = Q.submit([&](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
#if defined(__ADAPTIVECPP__)
        cgh.AdaptiveCpp_enqueue_custom_operation(
            [=, &kernels]([[maybe_unused]] sycl::interop_handle ih) {
                BuildExpressionKernels<TData>(kernels, kernel_src);
            });
#elif defined(__DPCPP_COMPILER)
        cgh.host_task([=, &kernels]() {
            BuildExpressionKernels<TData>(kernels, kernel_src);
        });
#endif
    });
    SYCLQueue::SetEvent(streamID, e);
}

/**
 * @brief Launch a variant of the compiled kernels on the native stream
 * behind the block's queue. The kernels are read when the launch runs
 * rather than when it is enqueued, as the compilation above is itself
 * enqueued on the block's queue.
 *
 * @tparam TData    Floating-point type of the field data.
 *
 * @param   kernels     Compiled kernels.
 * @param   index       Variant to launch.
 * @param   kernelArgs  Launch arguments.
 * @param   gridSize    Number of thread blocks.
 * @param   blockSize   Threads per block.
 * @param   streamID    Stream index of the block.
 */
template <typename TData>
void LaunchExpressionKernel(const ExpressionKernels &kernels,
                            const ExpressionKernelIndex index,
                            const ExpressionKernelArgs<TData> &kernelArgs,
                            const unsigned int gridSize,
                            const unsigned int blockSize,
                            const unsigned int streamID)
{
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);
    sycl::event e  = Q.submit([&](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);
        auto launch = [=, &kernels](sycl::interop_handle ih) {
            EnqueueExpressionKernel(kernels, index, kernelArgs, gridSize,
                                     blockSize,
                                     ih.get_native_queue<sycl_backend>());
        };
#if defined(__ADAPTIVECPP__)
        cgh.AdaptiveCpp_enqueue_custom_operation(launch);
#elif defined(__DPCPP_COMPILER)
        cgh.host_task(launch);
#endif
    });
    SYCLQueue::SetEvent(streamID, e);
}

} // namespace Nektar::MultiRegions::detail

#else // defined(SYCL_ENABLE_INTEL)

namespace Nektar::MultiRegions::detail
{

/// The kernels retrieved from the runtime-compiled bundle, which they
/// keep alive.
struct ExpressionKernels
{
    /// Kernels, in ExpressionKernelIndex order; empty until compiled.
    std::vector<sycl::kernel> kernels;
};

/// @brief Initialise the device context the runtime compiler needs.
inline void InitExpressionKernelCompiler(void)
{
    // The SYCL queue carries its own context, so there is nothing to do.
}

/**
 * @brief Generate the SYCL source of the expression kernels.
 *
 * A free function kernel cannot itself be templated and be looked up by
 * name, so the kernel body (GetExpressionKernelBody()) is written once
 * as a function template on APPEND and INTERLEAVEWIDTH, and each
 * variant is entered through an `extern "C"` kernel of its own, named
 * with the variant's suffix.
 *
 * @tparam TData    Floating-point type the kernels are generated for.
 *
 * @param   exprs       Parsed expressions, one per component; the
 *                      first expression's variable list defines the
 *                      EVAR names for all of them.
 * @param   cmask       Per-component enable flags.
 * @param   coordDim    Coordinate dimension of the block.
 *
 * @return The kernel source.
 */
template <typename TData>
std::string GetExpressionKernelSource(
    const std::vector<LibUtilities::EquationSharedPtr> &exprs,
    const std::vector<bool> &cmask, const unsigned int coordDim)
{
    // Generate a kernel function from the string expressions, and compile
    // it at runtime.
    const auto vnames = ExpressionBlockOp<TData>::GetEvarsNames(exprs[0]);

    // The generated source is compiled on its own, so it carries its own
    // includes and namespace aliases, and names the data type the kernels are
    // generated for.
    std::string kernel_src = "";
    kernel_src += "#include <sycl/sycl.hpp>\n";
    kernel_src += "\n";
    kernel_src += "namespace syclext = sycl::ext::oneapi;\n";
    kernel_src += "namespace syclexp = sycl::ext::oneapi::experimental;\n";
    kernel_src += "\n";
    kernel_src += "using TData = " + DataTypeToString<TData>() + ";\n";
    kernel_src += "\n";

    // Specify constants and parameters as macro variables.
    kernel_src += GetExpressionKernelConstants(exprs);

    const std::string parameters = GetExpressionKernelParameters(vnames, cmask);
    const std::string arguments  = GetExpressionKernelArguments(vnames, cmask);

    // Declare the kernel body, templated on whether it appends to its output
    // and on the interleave width of the storage it reads.
    kernel_src += "template<bool APPEND, unsigned int INTERLEAVEWIDTH>\n";
    kernel_src +=
        "static void " + expressionKernelName + "(" + parameters + ")\n";
    kernel_src += "{\n";
    kernel_src += GetExpressionKernelBody(
        exprs, cmask, vnames, coordDim,
        "syclext::this_work_item::get_nd_item<1>().get_global_linear_id()",
        "sycl::");
    kernel_src += "}\n";
    kernel_src += "\n";

    // Declare one kernel per variant of the body.
    for (const auto &variant : GetExpressionKernelVariants())
    {
        kernel_src += "extern \"C\" SYCL_EXT_ONEAPI_FUNCTION_PROPERTY("
                      "(syclexp::nd_range_kernel<1>))\n";
        kernel_src += "void " + expressionKernelName + variant.suffix + "(" +
                      parameters + ")\n";
        kernel_src += "{\n";
        kernel_src += "    " + expressionKernelName + "<" + variant.append +
                      ", " + variant.interleaveWidth + ">(" + arguments +
                      ");\n";
        kernel_src += "}\n";
    }

    return kernel_src;
}

/**
 * @brief Release the compiled kernels.
 *
 * @param   kernels     Compiled kernels, emptied.
 */
inline void ReleaseExpressionKernels(ExpressionKernels &kernels)
{
    kernels.kernels.clear();
}

/**
 * @brief Compile the generated source in the context of the block's
 * queue and retrieve the kernel of each variant from it, releasing the
 * kernels compiled for any previous expression. Raises an error if the
 * device cannot compile SYCL source or the compilation fails.
 *
 * @tparam TData    Floating-point type the kernels are compiled for.
 *
 * @param   kernels     Compiled kernels, replaced.
 * @param   kernel_src  Source from GetExpressionKernelSource().
 * @param   streamID    Stream index of the block.
 */
template <typename TData>
void CompileExpressionKernels(ExpressionKernels &kernels,
                              const std::string &kernel_src,
                              const unsigned int streamID)
{
    namespace syclexp = sycl::ext::oneapi::experimental;

    ReleaseExpressionKernels(kernels);

    sycl::queue &Q = SYCLQueue::GetInstance(streamID);

    ASSERTL0(
        Q.get_device().ext_oneapi_can_compile(syclexp::source_language::sycl),
        "The SYCL device cannot compile SYCL source at runtime, which the "
        "expression operator requires.")

    try
    {
        // Compile program.
        auto source_bundle = syclexp::create_kernel_bundle_from_source(
            Q.get_context(), syclexp::source_language::sycl, kernel_src);
        auto exec_bundle = syclexp::build(source_bundle);

        // Retrieve the kernel of each variant.
        for (const auto &variant : GetExpressionKernelVariants())
        {
            kernels.kernels.push_back(exec_bundle.ext_oneapi_get_kernel(
                expressionKernelName + variant.suffix));
        }
    }
    catch (const sycl::exception &e)
    {
        NEKERROR(ErrorUtil::efatal,
                 "Failed to compile the expression kernels: " +
                     std::string(e.what()));
    }
}

/**
 * @brief Launch a variant of the compiled kernels on the block's queue.
 *
 * @tparam TData    Floating-point type of the field data.
 *
 * @param   kernels     Compiled kernels.
 * @param   index       Variant to launch.
 * @param   kernelArgs  Launch arguments.
 * @param   gridSize    Number of work-groups.
 * @param   blockSize   Work-items per work-group.
 * @param   streamID    Stream index of the block.
 */
template <typename TData>
void LaunchExpressionKernel(const ExpressionKernels &kernels,
                            const ExpressionKernelIndex index,
                            const ExpressionKernelArgs<TData> &kernelArgs,
                            const unsigned int gridSize,
                            const unsigned int blockSize,
                            const unsigned int streamID)
{
    sycl::queue &Q = SYCLQueue::GetInstance(streamID);
    sycl::event e  = Q.submit([&](sycl::handler &cgh) {
        setSYCLDefaultExecutionDependency(streamID, cgh);

        // Set input parameters.
        const size_t numEvar = kernelArgs.evarptr.size();
        cgh.set_arg(0, kernelArgs.nsize);
        cgh.set_arg(1, kernelArgs.compSize);
        cgh.set_arg(2, kernelArgs.nelmt);
        cgh.set_arg(3, kernelArgs.ndata);
        cgh.set_arg(4, kernelArgs.scale);
        cgh.set_arg(5, kernelArgs.coordptr);
        cgh.set_arg(6, kernelArgs.time);

        // Set input pointers.
        for (size_t i = 0; i < numEvar; ++i)
        {
            cgh.set_arg(static_cast<int>(7 + i), kernelArgs.evarptr[i]);
        }

        // Set output pointers.
        for (size_t i = 0; i < kernelArgs.outptr.size(); ++i)
        {
            cgh.set_arg(static_cast<int>(7 + numEvar + i),
                         kernelArgs.outptr[i]);
        }

        // Launch kernel.
        cgh.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(static_cast<size_t>(gridSize) * blockSize),
                sycl::range<1>(blockSize)),
            kernels.kernels[index]);
    });
    SYCLQueue::SetEvent(streamID, e);
}

} // namespace Nektar::MultiRegions::detail

#endif
