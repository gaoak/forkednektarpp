///////////////////////////////////////////////////////////////////////////////
//
// File: ExpressionDeviceGeneric.hpp
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
// Description: Implementation of the math expression evaluator.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Operators/ElmtOps/Expression/ExpressionBlockOp.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#if defined(SYCL_ENABLE_CPU) || defined(NEKTAR_ENABLE_DEVICEONHOST)
#include "Operators/ElmtOps/Expression/ExpressionSerialAVXGeneric.hpp"
#else
#if defined(SYCL_ENABLE_CUDA) && defined(__ADAPTIVECPP__)
#define sycl_backend sycl::backend::cuda
#elif defined(SYCL_ENABLE_CUDA)
#define sycl_backend sycl::backend::ext_oneapi_cuda
#elif defined(SYCL_ENABLE_HIP) && defined(__ADAPTIVECPP__)
#define sycl_backend sycl::backend::hip
#elif defined(SYCL_ENABLE_HIP)
#define sycl_backend sycl::backend::ext_oneapi_hip
#endif

#if defined(SYCL_ENABLE_CUDA)
#include "Operators/Common/Backends/CUDA_Host_API.hpp"
#elif defined(SYCL_ENABLE_HIP)
#include "Operators/Common/Backends/HIP_Host_API.hpp"
#endif

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class ExpressionBlockOpImpl : public ExpressionBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    ExpressionBlockOpImpl(const unsigned int block_idx,
                          const LocalRegions::ExpansionSharedPtr &exp,
                          NekDataWarehouseSharedPtr dataWarehouse)
        : ExpressionBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        // Determine shape and type of the element.
        this->m_dimension = exp->GetShapeDimension();
        this->m_coordDim  = exp->GetCoordim();
        this->m_nqTot     = exp->GetTotPoints();

        // Initialise CUDA/HIP context for JIT.
#if defined(NEKTAR_ENABLE_CUDA) || defined(SYCL_ENABLE_CUDA)
        cudaFree(0);
#elif defined(NEKTAR_ENABLE_CUDA) || defined(SYCL_ENABLE_CUDA)
        hipFree(0);
#endif
    }

    ~ExpressionBlockOpImpl(void)
    {
        CHECK_HIPCUDA_DRIVER_ERROR(nekModuleUnload(m_nekModule));
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            ExpressionBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    bool m_isExpressionSet = false;
    unsigned int m_dimension;
    unsigned int m_coordDim;
    unsigned int m_nqTot;
    NEKmodule m_nekModule;
    NEKfunction m_kernel_handle1;
    NEKfunction m_kernel_handle2;
    NEKfunction m_kernel_handle3;
    NEKfunction m_kernel_handle4;

    void v_Apply(BlockAccessor<TData, FieldState::Phys> &inblock,
                 BlockAccessor<TData, FieldState::Phys> &outblock) override
    {
        ASSERTL1(this->m_expressions.size() == inblock.GetNumComponents() &&
                     this->m_expressions.size() == outblock.GetNumComponents(),
                 "Number of expressions must match number of components in "
                 "input and output Field when calling Apply().")

        ASSERTL1(this->m_expressions.size() == this->m_cmask.size(),
                 "Number of expressions must match size of component mask when "
                 "calling Apply().")

        const auto compSize = inblock.CompSize();

        // Initialize pointers.
        auto inptr = (&inblock != &outblock)
                         ? inblock.template GetPtr<MemSpace, ReadOnly>()
                         : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr =
            (&inblock != &outblock)
                ? (this->m_append)
                      ? outblock.template GetPtr<MemSpace, ReadWrite>()
                      : outblock.template GetPtr<MemSpace, WriteOnly>()
                : (TData *)inptr;

        // Get interleave parameter.
        const auto inInterleaveWidth  = inblock.GetInterleaveWidth();
        const auto outInterleaveWidth = outblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
        const unsigned int gridSize  = (compSize + blockSize - 1u) / blockSize;

        const auto coordptr = this->m_dataWarehouse->template GetData<MemSpace>(
            CoordKey<TData>(this->m_block_idx, inInterleaveWidth, false));

        // Loop over components.
        for (unsigned int n = 0; n < inblock.GetNumHomoModes(); ++n)
        {
            // Reshape, if necessary.
            for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
            {
                if (&inblock != &outblock && this->m_append)
                {
                    ReshapeStorage<ExecSpace>(
                        inInterleaveWidth, outInterleaveWidth,
                        outblock.GetNumElementsWithPadding(),
                        outblock.GetNumData(),
                        (TData *)outptr +
                            (n + nc * outblock.GetNumHomoModes()) * compSize);
                }
            }

            unsigned int numOutVar = 0;
            for (unsigned int nc = 0; nc < outblock.GetNumComponents(); ++nc)
            {
                // Check component mask
                if (this->m_cmask[nc])
                {
                    numOutVar++;
                }
            }

            // Set input parameters.
            unsigned int numEvar = this->m_numEvars[0];
            void **args          = new void *[numEvar + numOutVar];
            args[0]              = (void *)&compSize;
            args[1]              = (void *)&this->m_scale;
            args[2]              = (void *)&coordptr;
            args[3]              = (void *)&this->m_time;

            // Set input pointers.
            void **ptr = new void *[numEvar + numOutVar - 4];
            for (unsigned int i = 0; i < numEvar - 4; i++)
            {
                ptr[i] = (void *)(inptr + (n + i * inblock.GetNumHomoModes()) *
                                              compSize);
                args[4 + i] = &ptr[i];
            }

            // Set output pointers.
            for (unsigned int nc = 0, cnt = 0; nc < outblock.GetNumComponents();
                 ++nc)
            {
                // Check component mask
                if (this->m_cmask[nc])
                {
                    ptr[numEvar - 4 + cnt] =
                        (void *)(outptr +
                                 (n + nc * outblock.GetNumHomoModes()) *
                                     compSize);
                    args[numEvar + cnt] = &ptr[numEvar - 4 + cnt];
                    cnt++;
                }
            }

            // Launch kernel.
            auto kernel_handle =
                (inInterleaveWidth == 1)
                    ? (this->m_append) ? m_kernel_handle1 : m_kernel_handle2
                : (this->m_append) ? m_kernel_handle3
                                   : m_kernel_handle4;
#if defined(SYCL_ENABLE_CUDA) || defined(SYCL_ENABLE_HIP)
            SYCLQueue::GetInstance().submit([&](sycl::handler &h) {
#if defined(__ADAPTIVECPP__)
                h.AdaptiveCpp_enqueue_custom_operation(
                    [=](sycl::interop_handle ih) {
#else
                h.host_task([=](sycl::interop_handle ih) {
#endif
                        auto stream = ih.get_native_queue<sycl_backend>();
                        nekLaunchKernel(kernel_handle, gridSize, 1, 1,
                                        blockSize, 1, 1, 0, stream, args,
                                        nullptr);
                    });
            });
#if !defined(__ADAPTIVECPP__)
            SYCLQueue::GetInstance().wait();
#endif
#else
            CHECK_HIPCUDA_DRIVER_ERROR(nekLaunchKernel(kernel_handle, gridSize,
                                                       1, 1, blockSize, 1, 1, 0,
                                                       nullptr, args, nullptr));
#endif
            delete[] ptr;
            delete[] args;
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
    }

    void v_SetExpressions(
        const std::vector<LibUtilities::EquationSharedPtr> &exprs) override
    {
        // Use NVRTC (Nvidia Runtime Compilation) / HIPRTC (HIP Runtime
        // Compilation) to generate a kernel function from a string expression.

        // Clean-up previous expression.
        if (m_isExpressionSet)
        {
            CHECK_HIPCUDA_DRIVER_ERROR(nekModuleUnload(m_nekModule));
        }

        this->m_expressions = exprs;
        m_isExpressionSet   = true;

        // Define expression kernels.
        std::string kernel_src = "";
        // Specify constant as macro variable.
        std::map<std::string, double> constmap;
        for (unsigned int nc = 0; nc < this->m_expressions.size(); nc++)
        {
            auto constants = this->m_expressions[nc]->GetConstants();
            for (auto &constant : constants)
            {
                if (constmap.find(constant.first) == constmap.end())
                {
                    kernel_src +=
                        "#define " + constant.first + " " +
                        boost::lexical_cast<std::string>(constant.second) +
                        "\n";
                }
            }
        }
        // Specify parameters as constant variables.
        for (unsigned int nc = 0; nc < this->m_expressions.size(); nc++)
        {
            auto parameters = this->m_expressions[nc]->GetParameters();
            for (auto &parameter : parameters)
            {
                if (constmap.find(parameter.first) == constmap.end())
                {
                    kernel_src +=
                        "#define " + parameter.first + " " +
                        boost::lexical_cast<std::string>(parameter.second) +
                        "\n";
                }
            }
        }
        kernel_src += "\n";
        std::string kernel_name = "expression_kernel";
        auto vnames =
            ExpressionBlockOp<TData>::GetEvarsNames(this->m_expressions[0]);

        kernel_src += "template<bool APPEND, unsigned int INTERLEAVEWIDTH, "
                      "typename TData>\n";
        kernel_src += "__global__ void " + kernel_name;
        // Set-up arguments:
        kernel_src += "(";
        // 0. size.
        kernel_src += "const size_t nsize, ";
        kernel_src += "const TData scale, ";
        // 1. Coordinates.
        kernel_src += "const TData* coordptr, ";
        // 2. time variable.
        kernel_src += "const TData t, ";
        // 3. Pointers.
        for (unsigned int i = 4; i < vnames.size(); i++)
        {
            kernel_src += "const TData* " + vnames[i] + "ptr, ";
        }
        // 4. Output variable.
        for (unsigned int nc = 0; nc < this->m_expressions.size(); nc++)
        {
            if (this->m_cmask[nc])
            {
                kernel_src += "TData* out" + std::to_string(nc) + ", ";
            }
        }
        kernel_src.pop_back();
        kernel_src.pop_back();
        kernel_src += ")\n";

        kernel_src += "{\n";

        // Assign local variables:
        // 1. Thread index.
        kernel_src += "    const size_t tid = blockIdx.x * blockDim.x + "
                      "threadIdx.x;\n";
        kernel_src += "    if (tid >= nsize) return;\n";
        // 2. Coordinates.
        if (m_coordDim == 1)
        {
            kernel_src += "    TData x;\n";
        }
        else if (m_coordDim == 2)
        {
            kernel_src += "    TData x, y;\n";
        }
        else if (m_coordDim == 3)
        {
            kernel_src += "    TData x, y, z;\n";
        }
        kernel_src += "    if constexpr (INTERLEAVEWIDTH == 1) {\n";
        if (m_coordDim == 1)
        {
            kernel_src += "        x = coordptr[tid];\n";
        }
        else if (m_coordDim == 2)
        {
            kernel_src += "        x = coordptr[2 * tid];\n";
            kernel_src += "        y = coordptr[2 * tid + 1];\n";
        }
        else if (m_coordDim == 3)
        {
            kernel_src += "        x = coordptr[3 * tid];\n";
            kernel_src += "        y = coordptr[3 * tid + 1];\n";
            kernel_src += "        z = coordptr[3 * tid + 2];\n";
        }
        kernel_src += "    } else if constexpr (INTERLEAVEWIDTH != 1) {\n";
        kernel_src += "        const size_t ilane = tid % INTERLEAVEWIDTH;\n";
        kernel_src += "        const size_t iwarp = tid / INTERLEAVEWIDTH;\n";
        if (m_coordDim == 1)
        {
            kernel_src +=
                "        x = coordptr[iwarp * INTERLEAVEWIDTH + ilane];\n";
        }
        else if (m_coordDim == 2)
        {
            kernel_src +=
                "        x = coordptr[iwarp * 2 * INTERLEAVEWIDTH + ilane];\n";
            kernel_src += "        y = coordptr[iwarp * 2 * INTERLEAVEWIDTH + "
                          "INTERLEAVEWIDTH + ilane];\n";
        }
        else if (m_coordDim == 3)
        {
            kernel_src +=
                "        x = coordptr[iwarp * 3 * INTERLEAVEWIDTH + ilane];\n";
            kernel_src += "        y = coordptr[iwarp * 3 * INTERLEAVEWIDTH + "
                          "INTERLEAVEWIDTH + ilane];\n";
            kernel_src +=
                "        z = coordptr[iwarp * 3 * INTERLEAVEWIDTH + 2 * "
                "INTERLEAVEWIDTH + ilane];\n";
        }
        kernel_src += "    } \n";
        kernel_src += "\n";
        // 3. Pointers.
        for (unsigned int i = 4; i < vnames.size(); i++)
        {
            kernel_src += "    const TData " + vnames[i] + " = " + vnames[i] +
                          "ptr[tid];\n";
        }

        // Define expression.
        for (unsigned int nc = 0; nc < this->m_expressions.size(); nc++)
        {
            if (this->m_cmask[nc])
            {
                std::string expression =
                    convertPow(this->m_expressions[nc]->GetExpression());
                kernel_src += "    if constexpr (APPEND)\n";
                kernel_src += "    {\n";
                kernel_src += "        out" + std::to_string(nc) +
                              "[tid] += scale * (" + expression + ");\n";
                kernel_src += "    }\n";
                kernel_src += "    else\n";
                kernel_src += "    {\n";
                kernel_src += "        out" + std::to_string(nc) +
                              "[tid] = scale * (" + expression + ");\n";
                kernel_src += "    };\n";
            }
        }
        kernel_src += "}\n";

        // Print kernel
        // std::cout << kernel_src << std::endl;

        // Create program.
        nekrtcProgram prog;
#if defined(NEKTAR_ENABLE_CUDA) || defined(SYCL_ENABLE_CUDA)
        CHECK_NEKRTC_ERROR(nekrtcCreateProgram(&prog, kernel_src.c_str(),
                                               "expression_kernels.cu", 0,
                                               nullptr, nullptr));
#elif defined(NEKTAR_ENABLE_HIP) || defined(SYCL_ENABLE_HIP)
        CHECK_NEKRTC_ERROR(nekrtcCreateProgram(&prog, kernel_src.c_str(),
                                               "expression_kernels.hip", 0,
                                               nullptr, nullptr));
#endif

        // Register specialisation
        std::string name_expr1 =
            kernel_name + "<true, 1, " + DataTypeToString<TData>() + ">";
        CHECK_NEKRTC_ERROR(nekrtcAddNameExpression(prog, name_expr1.c_str()));
        std::string name_expr2 =
            kernel_name + "<false, 1 ," + DataTypeToString<TData>() + ">";
        CHECK_NEKRTC_ERROR(nekrtcAddNameExpression(prog, name_expr2.c_str()));
        std::string name_expr3 =
            kernel_name + "<true, " +
            std::to_string(NektarSpaces::Device::warpSize) + ", " +
            DataTypeToString<TData>() + ">";
        CHECK_NEKRTC_ERROR(nekrtcAddNameExpression(prog, name_expr3.c_str()));
        std::string name_expr4 =
            kernel_name + "<false, " +
            std::to_string(NektarSpaces::Device::warpSize) + ", " +
            DataTypeToString<TData>() + ">";
        CHECK_NEKRTC_ERROR(nekrtcAddNameExpression(prog, name_expr4.c_str()));

        // Compile program.
        const char *opts[] = {"--std=c++17"};
        CHECK_NEKRTC_ERROR(nekrtcCompileProgram(prog, 1, opts));

        // Load module.
        size_t code_size;
        CHECK_NEKRTC_ERROR(nekrtcGetCodeSize(prog, &code_size));
        std::vector<char> code(code_size);
        CHECK_NEKRTC_ERROR(nekrtcGetCode(prog, code.data()));
        CHECK_HIPCUDA_DRIVER_ERROR(
            nekModuleLoadData(&m_nekModule, code.data()));

        // Retrive mangled name and register function.
        const char *mangled_name;
        CHECK_NEKRTC_ERROR(
            nekrtcGetLoweredName(prog, name_expr1.c_str(), &mangled_name));
        CHECK_HIPCUDA_DRIVER_ERROR(
            nekModuleGetFunction(&m_kernel_handle1, m_nekModule, mangled_name));
        CHECK_NEKRTC_ERROR(
            nekrtcGetLoweredName(prog, name_expr2.c_str(), &mangled_name));
        CHECK_HIPCUDA_DRIVER_ERROR(
            nekModuleGetFunction(&m_kernel_handle2, m_nekModule, mangled_name));
        CHECK_NEKRTC_ERROR(
            nekrtcGetLoweredName(prog, name_expr3.c_str(), &mangled_name));
        CHECK_HIPCUDA_DRIVER_ERROR(
            nekModuleGetFunction(&m_kernel_handle3, m_nekModule, mangled_name));
        CHECK_NEKRTC_ERROR(
            nekrtcGetLoweredName(prog, name_expr4.c_str(), &mangled_name));
        CHECK_HIPCUDA_DRIVER_ERROR(
            nekModuleGetFunction(&m_kernel_handle4, m_nekModule, mangled_name));
        CHECK_NEKRTC_ERROR(nekrtcDestroyProgram(&prog));
    }

    // From Google AI.
    // To following helper functions are used to convert x^y expressions into
    // pow(x, y) such that they can be evaluated in kernel functions.

    // Finds the matching bracket by scanning in a specific direction
    int findMatching(const std::string &s, int start, int direction)
    {
        int count = 0;
        for (int i = start; i >= 0 && i < s.length(); i += direction)
        {
            if (s[i] == '(')
                count += direction;
            else if (s[i] == ')')
                count -= direction;
            if (count == 0)
                return i;
        }
        return start;
    }

    std::string convertPow(std::string expr)
    {
        size_t caret;
        while ((caret = expr.find('^')) != std::string::npos)
        {
            int left_start, right_end;

            // 1. Identify Left Operand (Base)
            if (expr[caret - 1] == ')')
            {
                left_start = findMatching(expr, caret - 1, -1);
            }
            else
            {
                left_start = caret - 1;
                while (left_start > 0 && (isalnum(expr[left_start - 1]) ||
                                          expr[left_start - 1] == '.'))
                {
                    left_start--;
                }
            }
            std::string base = expr.substr(left_start, caret - left_start);

            // 2. Identify Right Operand (Exponent)
            if (expr[caret + 1] == '(')
            {
                right_end = findMatching(expr, caret + 1, 1);
            }
            else
            {
                right_end = caret + 1;
                while (right_end < expr.length() - 1 &&
                       (isalnum(expr[right_end + 1]) ||
                        expr[right_end + 1] == '.'))
                {
                    right_end++;
                }
            }
            std::string exponent = expr.substr(caret + 1, right_end - caret);

            // 3. Replace x^y with pow(x, y)
            std::string replacement = "pow(" + base + ", " + exponent + ")";
            expr.replace(left_start, (right_end - left_start + 1), replacement);
        }
        return expr;
    }
};

} // namespace Nektar::Operators::detail
#endif
