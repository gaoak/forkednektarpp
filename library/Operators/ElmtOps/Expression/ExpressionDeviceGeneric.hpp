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

#include <LibUtilities/SimdLib/tinysimd.hpp>
#include <boost/algorithm/string.hpp>

#include "Operators/ElmtOps/Expression/ExpressionBlockOp.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#if defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
#include "Operators/ElmtOps/Expression/ExpressionSerialGeneric.hpp"
#elif defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP)
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

        this->m_coordptr = this->m_dataWarehouse->template GetData<MemSpace>(
            CoordKey<TData>(block_idx, this->m_implInterleaveWidth, false));

        // Initialise CUDA/HIP context for JIT.
        nekCtxGetCurrent(&m_context);
    }

    ~ExpressionBlockOpImpl(void)
    {
        nekrtcDestroyProgram(&m_prog);
        nekModuleUnload(m_nekModule);
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
    static constexpr unsigned int m_implInterleaveWidth = 1;

    bool m_isExpressionSet = false;
    unsigned int m_dimension;
    unsigned int m_coordDim;
    unsigned int m_nqTot;
    const TData *m_coordptr;
    nekrtcProgram m_prog;
    NEKmodule m_nekModule;
    NEKcontext m_context;
    std::vector<NEKfunction> m_kernel_handle1;
    std::vector<NEKfunction> m_kernel_handle2;

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

        const auto nelmt    = inblock.GetNumElements();
        const auto compSize = inblock.CompSize();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = (this->m_append)
                          ? outblock.template GetPtr<MemSpace, ReadWrite>()
                          : outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
        const unsigned int gridSize  = (compSize + blockSize - 1u) / blockSize;

        // Loop over components.
        unsigned int nc;
        for (unsigned int n = 0;
             n < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++n)
        {
            // Get component index
            nc = n / inblock.GetNumHomoModes();

            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(this->m_implInterleaveWidth,
                                      interleaveWidth,
                                      inblock.GetNumElementsWithPadding(),
                                      inblock.GetNumData(), (TData *)inptr);

            if (this->m_append)
            {
                ReshapeStorage<ExecSpace>(
                    this->m_implInterleaveWidth, interleaveWidth,
                    outblock.GetNumElementsWithPadding(), outblock.GetNumData(),
                    (TData *)outptr);
            }

            // Check component mask
            if (this->m_cmask[nc])
            {
                unsigned int numEvar = this->m_numEvars[nc];
                void **args          = new void *[numEvar + 1];
                args[0]              = (void *)&compSize;
                args[1]              = (void *)&this->m_scale;
                args[2]              = (void *)&m_coordptr;
                args[3]              = (void *)&this->m_time;

                // Set input pointers.
                void **ptr = new void *[numEvar - 4];
                for (unsigned int i = 0; i < numEvar - 4; i++)
                {
                    ptr[i]      = (void *)(inptr + i * compSize);
                    args[4 + i] = &ptr[i];
                }
                // Set output pointers.
                args[numEvar] = (void *)&outptr;
                if (this->m_append)
                {
                    nekLaunchKernel(m_kernel_handle1[nc], gridSize, 1, 1,
                                    blockSize, 1, 1, 0, NULL, args, NULL);
                }
                else
                {
                    nekLaunchKernel(m_kernel_handle2[nc], gridSize, 1, 1,
                                    blockSize, 1, 1, 0, NULL, args, NULL);
                }
                delete[] ptr;
                delete[] args;
            }

            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(interleaveWidth,
                                      this->m_implInterleaveWidth,
                                      inblock.GetNumElementsWithPadding(),
                                      inblock.GetNumData(), (TData *)inptr);

            ReshapeStorage<ExecSpace>(interleaveWidth,
                                      this->m_implInterleaveWidth,
                                      outblock.GetNumElementsWithPadding(),
                                      outblock.GetNumData(), (TData *)outptr);

            // Increment pointer.
            outptr += outblock.CompSize();
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    void v_SetExpressions(
        const std::vector<LibUtilities::EquationSharedPtr> &exprs) override
    {
        // Clean-up previous expression.
        if (m_isExpressionSet)
        {
            nekrtcDestroyProgram(&m_prog);
            nekModuleUnload(m_nekModule);
            m_kernel_handle1.clear();
            m_kernel_handle2.clear();
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
        // kernel_src += "#define PI  3.14159265358979323846 \n";
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
        for (unsigned int nc = 0; nc < this->m_expressions.size(); nc++)
        {
            std::string kernel_name = "expression_kernel" + std::to_string(nc);
            std::vector<std::string> vnames;
            boost::split(vnames, this->m_expressions[nc]->GetVlist(),
                         boost::is_any_of(", "));

            kernel_src += "template<bool APPEND, typename TData>\n";
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
            kernel_src += "TData* out)\n";

            kernel_src += "{\n";

            // Assign local variables:
            // 1. Thread index.
            kernel_src += "    const size_t tid = blockIdx.x * blockDim.x + "
                          "threadIdx.x;\n";
            kernel_src += "    if (tid >= nsize) return;\n";
            // 2. Coordinates.
            if (m_coordDim == 1)
            {
                kernel_src += "    const TData x = coordptr[tid];\n";
            }
            else if (m_coordDim == 2)
            {
                kernel_src += "    const TData x = coordptr[2 * tid];\n";
                kernel_src += "    const TData y = coordptr[2 * tid + 1];\n";
            }
            else if (m_coordDim == 3)
            {
                kernel_src += "    const TData x = coordptr[3 * tid];\n";
                kernel_src += "    const TData y = coordptr[3 * tid + 1];\n";
                kernel_src += "    const TData z = coordptr[3 * tid + 2];\n";
            }
            // 3. Pointers.
            for (unsigned int i = 4; i < vnames.size(); i++)
            {
                kernel_src += "    const TData " + vnames[i] + " = " +
                              vnames[i] + "ptr[tid];\n";
            }

            // Define expression.
            std::string expression = this->m_expressions[nc]->GetExpression();
            kernel_src += "    if constexpr (APPEND)\n";
            kernel_src += "    {\n";
            kernel_src += "        out[tid] = scale * (" + expression + ");\n";
            kernel_src += "    }\n";
            kernel_src += "    else\n";
            kernel_src += "    {\n";
            kernel_src += "        out[tid] += scale * (" + expression + ");\n";
            kernel_src += "    };\n";
            kernel_src += "}\n";
        }

        // Print kernel
        // std::cout << kernel_src << std::endl;

        // Create program.
#if defined(NEKTAR_ENABLE_CUDA)
        CHECK_NEKRTC_ERROR(nekrtcCreateProgram(&m_prog, kernel_src.c_str(),
                                               "expression_kernels.cu", 0, NULL,
                                               NULL));
#elif defined(NEKTAR_ENABLE_HIP)
        CHECK_NEKRTC_ERROR(nekrtcCreateProgram(&m_prog, kernel_src.c_str(),
                                               "expression_kernels.hip", 0,
                                               NULL, NULL));
#endif

        // Register specialisation
        for (unsigned int nc = 0; nc < this->m_expressions.size(); nc++)
        {
            std::string kernel_name = "expression_kernel" + std::to_string(nc);
            std::string name_expr1 =
                kernel_name + "<true, " + DataTypeToString<TData>() + ">";
            CHECK_NEKRTC_ERROR(
                nekrtcAddNameExpression(m_prog, name_expr1.c_str()));
            std::string name_expr2 =
                kernel_name + "<false, " + DataTypeToString<TData>() + ">";
            CHECK_NEKRTC_ERROR(
                nekrtcAddNameExpression(m_prog, name_expr2.c_str()));
        }

        // Compile program.
        const char *opts[] = {"", ""};
        CHECK_NEKRTC_ERROR(nekrtcCompileProgram(m_prog, 0, opts));

        // Load module.
        size_t ptx_size;
        CHECK_NEKRTC_ERROR(nekrtcGetCodeSize(m_prog, &ptx_size));
        std::vector<char> ptx(ptx_size);
        CHECK_NEKRTC_ERROR(nekrtcGetCode(m_prog, ptx.data()));
        nekModuleLoadData(&m_nekModule, ptx.data());

        // Retrive mangled name and register function.
        for (unsigned int nc = 0; nc < this->m_expressions.size(); nc++)
        {
            std::string kernel_name = "expression_kernel" + std::to_string(nc);
            NEKfunction func;
            const char *mangled_name;
            std::string name_expr1 =
                kernel_name + "<true, " + DataTypeToString<TData>() + ">";
            CHECK_NEKRTC_ERROR(nekrtcGetLoweredName(m_prog, name_expr1.c_str(),
                                                    &mangled_name));
            nekModuleGetFunction(&func, m_nekModule, mangled_name);
            m_kernel_handle1.push_back(func);
            std::string name_expr2 =
                kernel_name + "<false, " + DataTypeToString<TData>() + ">";
            CHECK_NEKRTC_ERROR(nekrtcGetLoweredName(m_prog, name_expr2.c_str(),
                                                    &mangled_name));
            nekModuleGetFunction(&func, m_nekModule, mangled_name);
            m_kernel_handle2.push_back(func);
        }
    }
};

} // namespace Nektar::Operators::detail
#endif
