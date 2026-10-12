///////////////////////////////////////////////////////////////////////////////
//
// File: ExpressionDeviceGenericHIPCUDAHelper.hpp
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
// kernels with NVRTC (Nvidia Runtime Compilation) or HIPRTC (HIP Runtime
// Compilation).
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file ExpressionDeviceGenericHIPCUDAHelper.hpp
 * @brief Generation, runtime compilation (NVRTC/HIPRTC) and launch of
 * the expression kernels on CUDA and HIP.
 *
 * @details
 * Used directly on the native CUDA/HIP builds, which also take
 * CompileExpressionKernels() and LaunchExpressionKernel() from here.
 * The SYCL-on-CUDA/HIP builds reuse the rest through
 * ExpressionDeviceGenericSYCLHelper.hpp, which supplies those two
 * entry points itself.
 */

#pragma once

#include <array>
#include <string>
#include <vector>

#include <MultiRegions/ElmtOps/Expression/ExpressionDeviceGenericHelper.hpp>

// The runtime compiler is declared by the CUDA or HIP host API header, which
// Backends.hpp includes on a CUDA or HIP build. A SYCL build offloading to
// CUDA or HIP compiles and launches through the same runtime compiler, and
// includes that header before this one.

namespace Nektar::MultiRegions::detail
{

/// The name the generated source is compiled under. It only names the
/// source in the runtime compiler's diagnostics.
#if defined(NEKTAR_ENABLE_CUDA) || defined(SYCL_ENABLE_CUDA)
inline const std::string expressionKernelSourceName = "expression_kernels.cu";
#elif defined(NEKTAR_ENABLE_HIP) || defined(SYCL_ENABLE_HIP)
inline const std::string expressionKernelSourceName = "expression_kernels.hip";
#endif

/// The runtime-compiled module, and the kernels retrieved from it.
struct ExpressionKernels
{
    /// Driver-API module; null until compiled.
    NEKmodule nekModule = nullptr;
    /// Kernel handles, in ExpressionKernelIndex order.
    std::array<NEKfunction, eNumExpressionKernels> kernels;
};

/**
 * @brief Initialise the device context the runtime compiler needs: a
 * `cudaFree(0)`/`hipFree(0)` on the native builds, nothing on SYCL.
 */
inline void InitExpressionKernelCompiler(void)
{
#if defined(NEKTAR_ENABLE_CUDA)
    CHECK_HIPCUDA_ERROR(cudaFree(0));
#elif defined(NEKTAR_ENABLE_HIP)
    CHECK_HIPCUDA_ERROR(hipFree(0));
#endif
}

#if defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP)
/**
 * @brief The native stream a block issues its work on.
 *
 * @param   streamID    Stream index of the block.
 */
inline auto GetExpressionKernelStream(const unsigned int streamID)
{
#if defined(NEKTAR_ENABLE_CUDA)
    return CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    return HIPStream::GetInstance(streamID);
#endif
}
#endif

/**
 * @brief The name expressions the kernel variants are registered and
 * looked up under, in ExpressionKernelIndex order.
 *
 * @tparam TData    Floating-point type the kernels are compiled for.
 */
template <typename TData>
inline std::array<std::string, eNumExpressionKernels>
GetExpressionKernelNameExpressions(void)
{
    const auto variants = GetExpressionKernelVariants();

    std::array<std::string, eNumExpressionKernels> name_exprs;
    for (unsigned int i = 0; i < eNumExpressionKernels; ++i)
    {
        name_exprs[i] = expressionKernelName + "<" + variants[i].append + ", " +
                        variants[i].interleaveWidth + ", " +
                        DataTypeToString<TData>() + ">";
    }

    return name_exprs;
}

/**
 * @brief Generate the CUDA/HIP source of the expression kernel.
 *
 * The emitted source consists of:
 * -# a `#define` for every constant and every parameter of each
 *    expression, so those names resolve to their session values
 *    (GetExpressionKernelConstants());
 * -# a kernel `expression_kernel` templated on an APPEND flag, the
 *    interleave width and the data type, taking the parameters of
 *    GetExpressionKernelParameters(): the EVAR input pointers are
 *    named after the variable-list entries from index 4 on, and there
 *    is one output pointer per unmasked component;
 * -# its body (GetExpressionKernelBody()): a guarded thread index, a
 *    zero write (unless appending) and early return on padded
 *    elements, and the coordinate gather, emitted for @p coordDim in
 *    both flat (width 1) and element-interleaved addressing, selected
 *    by the width template parameter; `#define`s forcing the common
 *    maths functions to take TData arguments, so integer literals in
 *    the expression strings do not make calls like `sqrt(61)`
 *    ambiguous under runtime compilation; and, per unmasked component,
 *    the expression itself -- caret powers rewritten into `pow()`
 *    calls by convertPow() -- assigned or appended to `out<nc>[tid]`,
 *    scaled.
 *
 * @tparam TData    Floating-point type the kernel is generated for.
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

    std::string kernel_src = "";

    // Specify constants and parameters as macro variables.
    kernel_src += GetExpressionKernelConstants(exprs);

    // Declare the kernel, templated on whether it appends to its output and on
    // the interleave width of the storage it reads.
    kernel_src += "template<bool APPEND, unsigned int INTERLEAVEWIDTH, "
                  "typename TData>\n";
    kernel_src += "__global__ void " + expressionKernelName;
    kernel_src += "(" + GetExpressionKernelParameters(vnames, cmask) + ")\n";
    kernel_src += "{\n";
    kernel_src += GetExpressionKernelBody(exprs, cmask, vnames, coordDim,
                                          "blockIdx.x * blockDim.x + "
                                          "threadIdx.x",
                                          "");
    kernel_src += "}\n";

    return kernel_src;
}

/**
 * @brief Unload the module holding the compiled kernels, if any.
 *
 * @param   kernels     Compiled kernels, reset to the uncompiled state.
 */
inline void ReleaseExpressionKernels(ExpressionKernels &kernels)
{
    if (kernels.nekModule != nullptr)
    {
        CHECK_HIPCUDA_DRIVER_ERROR(nekModuleUnload(kernels.nekModule));
        kernels.nekModule = nullptr;
    }
}

/**
 * @brief Compile the generated source and retrieve the kernels from it,
 * releasing the kernels compiled for any previous expression.
 *
 * The four name expressions <APPEND, INTERLEAVEWIDTH, TData> for
 * APPEND in {true, false} and INTERLEAVEWIDTH in {1, warp size},
 * with TData set by this operator's instantiation, are
 * registered; the program is then compiled with NVRTC/HIPRTC,
 * loaded as a driver module and the lowered names resolved into
 * the kernel handles for the launch to use.
 *
 * @tparam TData    Floating-point type the kernels are compiled for.
 *
 * @param   kernels     Compiled module and kernel handles, replaced.
 * @param   kernel_src  Source from GetExpressionKernelSource().
 */
template <typename TData>
void BuildExpressionKernels(ExpressionKernels &kernels,
                            const std::string &kernel_src)
{
    ReleaseExpressionKernels(kernels);

    // Create program.
    nekrtcProgram prog;
    CHECK_NEKRTC_ERROR(nekrtcCreateProgram(&prog, kernel_src.c_str(),
                                           expressionKernelSourceName.c_str(),
                                           0, nullptr, nullptr));

    // Register specialisation.
    const auto name_exprs = GetExpressionKernelNameExpressions<TData>();
    for (const auto &name_expr : name_exprs)
    {
        CHECK_NEKRTC_ERROR(nekrtcAddNameExpression(prog, name_expr.c_str()));
    }

    // Compile program.
    const char *opts[] = {"--std=c++17"};
    CHECK_NEKRTC_ERROR(nekrtcCompileProgram(prog, 1, opts));

    // Load module.
    size_t code_size;
    CHECK_NEKRTC_ERROR(nekrtcGetCodeSize(prog, &code_size));
    std::vector<char> code(code_size);
    CHECK_NEKRTC_ERROR(nekrtcGetCode(prog, code.data()));
    CHECK_HIPCUDA_DRIVER_ERROR(
        nekModuleLoadData(&kernels.nekModule, code.data()));

    // Retrieve mangled name and register function.
    for (unsigned int i = 0; i < eNumExpressionKernels; ++i)
    {
        const char *mangled_name;
        CHECK_NEKRTC_ERROR(
            nekrtcGetLoweredName(prog, name_exprs[i].c_str(), &mangled_name));
        CHECK_HIPCUDA_DRIVER_ERROR(nekModuleGetFunction(
            &kernels.kernels[i], kernels.nekModule, mangled_name));
    }
    CHECK_NEKRTC_ERROR(nekrtcDestroyProgram(&prog));
}

/**
 * @brief Launch a variant of the compiled kernels on a native stream.
 *
 * @tparam TData    Floating-point type of the field data.
 * @tparam TStream  Native stream type.
 *
 * @param   kernels     Compiled kernels.
 * @param   index       Variant to launch.
 * @param   kernelArgs  Launch arguments; the launch takes their
 *                      addresses.
 * @param   gridSize    Number of thread blocks.
 * @param   blockSize   Threads per block.
 * @param   stream      Stream to launch on.
 */
template <typename TData, typename TStream>
void EnqueueExpressionKernel(const ExpressionKernels &kernels,
                             const ExpressionKernelIndex index,
                             const ExpressionKernelArgs<TData> &kernelArgs,
                             const unsigned int gridSize,
                             const unsigned int blockSize, TStream stream)
{
    // The launch takes the address of each argument, so the arguments must
    // outlive it: they are held by kernelArgs.
    const size_t numEvar = kernelArgs.evarptr.size();
    const size_t numArgs = 7 + numEvar + kernelArgs.outptr.size();
    void **args          = new void *[numArgs];
    args[0]              = (void *)&kernelArgs.nsize;
    args[1]              = (void *)&kernelArgs.compSize;
    args[2]              = (void *)&kernelArgs.nelmt;
    args[3]              = (void *)&kernelArgs.ndata;
    args[4]              = (void *)&kernelArgs.scale;
    args[5]              = (void *)&kernelArgs.coordptr;
    args[6]              = (void *)&kernelArgs.time;

    // Set input pointers.
    for (size_t i = 0; i < numEvar; ++i)
    {
        args[7 + i] = (void *)&kernelArgs.evarptr[i];
    }

    // Set output pointers.
    for (size_t i = 0; i < kernelArgs.outptr.size(); ++i)
    {
        args[7 + numEvar + i] = (void *)&kernelArgs.outptr[i];
    }

    // Launch kernel.
    CHECK_HIPCUDA_DRIVER_ERROR(nekLaunchKernel(kernels.kernels[index], gridSize,
                                               1, 1, blockSize, 1, 1, 0, stream,
                                               args, nullptr));

    delete[] args;
}

#if !defined(NEKTAR_ENABLE_SYCL)
/**
 * @brief Compile the generated source; see BuildExpressionKernels().
 *
 * @tparam TData    Floating-point type the kernels are compiled for.
 *
 * @param   kernels     Compiled kernels, replaced.
 * @param   kernel_src  Source from GetExpressionKernelSource().
 * @param   streamID    Stream index of the block; unused.
 */
template <typename TData>
void CompileExpressionKernels(ExpressionKernels &kernels,
                              const std::string &kernel_src,
                              [[maybe_unused]] const unsigned int streamID)
{
    BuildExpressionKernels<TData>(kernels, kernel_src);
}

/**
 * @brief Launch a variant of the compiled kernels on the block's
 * stream.
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
    EnqueueExpressionKernel(kernels, index, kernelArgs, gridSize, blockSize,
                            GetExpressionKernelStream(streamID));
}
#endif

} // namespace Nektar::MultiRegions::detail
