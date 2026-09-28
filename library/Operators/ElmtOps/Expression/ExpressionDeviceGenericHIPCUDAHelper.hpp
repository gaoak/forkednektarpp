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

#pragma once

#include <array>
#include <string>
#include <vector>

#include "Operators/ElmtOps/Expression/ExpressionDeviceGenericHelper.hpp"

// The runtime compiler is declared by the CUDA or HIP host API header, which
// Backends.hpp includes on a CUDA or HIP build. A SYCL build offloading to
// CUDA or HIP compiles and launches through the same runtime compiler, and
// includes that header before this one.

namespace Nektar::Operators::detail
{

// The name the generated source is compiled under. It only names the source in
// the runtime compiler's diagnostics, and follows the back-end in use.
#if defined(NEKTAR_ENABLE_CUDA) || defined(SYCL_ENABLE_CUDA)
inline const std::string expressionKernelSourceName = "expression_kernels.cu";
#elif defined(NEKTAR_ENABLE_HIP) || defined(SYCL_ENABLE_HIP)
inline const std::string expressionKernelSourceName = "expression_kernels.hip";
#endif

// The runtime compiled module, and the kernels retrieved from it in
// ExpressionKernelIndex order.
struct ExpressionKernels
{
    NEKmodule nekModule = nullptr;
    std::array<NEKfunction, eNumExpressionKernels> kernels;
};

// Initialises the device context the runtime compiler needs.
inline void InitExpressionKernelCompiler(void)
{
#if defined(NEKTAR_ENABLE_CUDA)
    CHECK_HIPCUDA_ERROR(cudaFree(0));
#elif defined(NEKTAR_ENABLE_HIP)
    CHECK_HIPCUDA_ERROR(hipFree(0));
#endif
}

#if defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP)
// The stream a block issues its work on.
inline auto GetExpressionKernelStream(const unsigned int streamID)
{
#if defined(NEKTAR_ENABLE_CUDA)
    return CUDAStream::GetInstance(streamID);
#elif defined(NEKTAR_ENABLE_HIP)
    return HIPStream::GetInstance(streamID);
#endif
}
#endif

// The name expressions the kernel variants are registered and looked up under,
// in ExpressionKernelIndex order.
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

// Generates the source of a kernel function evaluating the expressions, as one
// function template instantiated on each of the kernel variants.
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

// Releases the runtime compiled kernels.
inline void ReleaseExpressionKernels(ExpressionKernels &kernels)
{
    if (kernels.nekModule != nullptr)
    {
        CHECK_HIPCUDA_DRIVER_ERROR(nekModuleUnload(kernels.nekModule));
        kernels.nekModule = nullptr;
    }
}

// Compiles the generated source and retrieves the kernels from it, releasing
// the kernels compiled for any previous expression.
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

// Launches the given variant of the runtime compiled kernels on a stream.
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
    const size_t numArgs = 5 + numEvar + kernelArgs.outptr.size();
    void **args          = new void *[numArgs];
    args[0]              = (void *)&kernelArgs.nsize;
    args[1]              = (void *)&kernelArgs.compSize;
    args[2]              = (void *)&kernelArgs.scale;
    args[3]              = (void *)&kernelArgs.coordptr;
    args[4]              = (void *)&kernelArgs.time;

    // Set input pointers.
    for (size_t i = 0; i < numEvar; ++i)
    {
        args[5 + i] = (void *)&kernelArgs.evarptr[i];
    }

    // Set output pointers.
    for (size_t i = 0; i < kernelArgs.outptr.size(); ++i)
    {
        args[5 + numEvar + i] = (void *)&kernelArgs.outptr[i];
    }

    // Launch kernel.
    CHECK_HIPCUDA_DRIVER_ERROR(nekLaunchKernel(kernels.kernels[index], gridSize,
                                               1, 1, blockSize, 1, 1, 0, stream,
                                               args, nullptr));

    delete[] args;
}

#if !defined(NEKTAR_ENABLE_SYCL)
template <typename TData>
void CompileExpressionKernels(ExpressionKernels &kernels,
                              const std::string &kernel_src,
                              [[maybe_unused]] const unsigned int streamID)
{
    BuildExpressionKernels<TData>(kernels, kernel_src);
}

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

} // namespace Nektar::Operators::detail
