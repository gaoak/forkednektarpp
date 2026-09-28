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

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "Operators/ElmtOps/Expression/ExpressionBlockOp.hpp"

#if defined(SYCL_ENABLE_CPU)
// The CPU back-end evaluates the expressions on the host.
#include "Operators/ElmtOps/Expression/ExpressionSerialAVXGeneric.hpp"
#else
#if defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP)
#include "Operators/ElmtOps/Expression/ExpressionDeviceGenericHIPCUDAHelper.hpp"
#else
#include "Operators/ElmtOps/Expression/ExpressionDeviceGenericSYCLHelper.hpp"
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
                          LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : ExpressionBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        m_streamID = block_idx + 1;

        // Determine shape and type of the element.
        this->m_dimension = exp->GetShapeDimension();
        this->m_coordDim  = exp->GetCoordim();
        this->m_nqTot     = exp->GetTotPoints();

        // Initialise the device context for JIT.
        InitExpressionKernelCompiler();
    }

    ~ExpressionBlockOpImpl(void)
    {
        nekStreamSynchronize(m_streamID);
        ReleaseExpressionKernels(m_kernels);
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            ExpressionBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    unsigned int m_streamID;
    unsigned int m_dimension;
    unsigned int m_coordDim;
    unsigned int m_nqTot;
    ExpressionKernels m_kernels;

    void v_Apply(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock) override
    {
        ASSERTL1(this->m_expressions.size() == inblock.GetNumComponents() &&
                     this->m_expressions.size() == outblock.GetNumComponents(),
                 "Number of expressions must match number of components in "
                 "input and output Field when calling Apply().")

        ASSERTL1(this->m_expressions.size() == this->m_cmask.size(),
                 "Number of expressions must match size of component mask when "
                 "calling Apply().")

        ASSERTL1(inblock.GetNumHomoModes() == outblock.GetNumHomoModes(),
                 "Number of homogeneous modes must match in input and output "
                 "Field when calling Apply().")

        const auto compSize = inblock.CompSize();
        const auto nhomo    = inblock.GetNumHomoModes();
        const auto noutcomp = outblock.GetNumComponents();

        // Initialize pointers.
        auto inptr =
            (&inblock != &outblock)
                ? inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID)
                : inblock.template GetPtr<MemSpace, ReadWrite>(m_streamID);
        auto outptr = (&inblock != &outblock)
                          ? (this->m_append)
                                ? outblock.template GetPtr<MemSpace, ReadWrite>(
                                      m_streamID)
                                : outblock.template GetPtr<MemSpace, WriteOnly>(
                                      m_streamID)
                          : (TData *)inptr;

        // Get interleave parameter.
        const auto inInterleaveWidth  = inblock.GetInterleaveWidth();
        const auto outInterleaveWidth = outblock.GetInterleaveWidth();

        // Get coord pointer.
        const auto coordptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::CoordKey<TData>(this->m_block_idx, inInterleaveWidth,
                                          false));

        // Set Kernel parameters.
        const size_t nsize           = compSize * nhomo;
        const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
        const unsigned int gridSize =
            static_cast<unsigned int>((nsize + blockSize - 1u) / blockSize);

        // Select the kernel matching the interleave format that is read and
        // whether the operator appends to its output.
        const auto index =
            (inInterleaveWidth == 1)
                ? (this->m_append) ? eAppendScalar : eAssignScalar
            : (this->m_append) ? eAppendInterleaved
                               : eAssignInterleaved;

        // Reshape, if necessary.
        if (&inblock != &outblock && this->m_append)
        {
            LibUtilities::ReshapeStorage<ExecSpace>(
                inInterleaveWidth, outInterleaveWidth,
                outblock.GetNumElementsWithPadding() * nhomo * noutcomp,
                outblock.GetNumData(), (TData *)outptr, m_streamID);
        }

        // Set input parameters.
        ExpressionKernelArgs<TData> kernelArgs;
        kernelArgs.nsize    = nsize;
        kernelArgs.compSize = compSize;
        kernelArgs.scale    = this->m_scale;
        kernelArgs.coordptr = coordptr;
        kernelArgs.time     = this->m_time;

        // Set input pointers, at the first mode of the component.
        const unsigned int numEvar = this->m_numEvars[0];
        for (unsigned int i = 0; i < numEvar - 4; i++)
        {
            kernelArgs.evarptr.push_back(inptr + i * nhomo * compSize);
        }

        // Set output pointers, at the first mode of the component.
        for (unsigned int nc = 0; nc < noutcomp; ++nc)
        {
            // Check component mask.
            if (this->m_cmask[nc])
            {
                kernelArgs.outptr.push_back(outptr + nc * nhomo * compSize);
            }
        }

        // Launch kernel.
        LaunchExpressionKernel(m_kernels, index, kernelArgs, gridSize,
                               blockSize, m_streamID);

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
    }

    void v_SetExpressions(
        const std::vector<LibUtilities::EquationSharedPtr> &exprs) override
    {
        this->m_expressions = exprs;

        CompileExpressionKernels<TData>(
            m_kernels,
            GetExpressionKernelSource<TData>(this->m_expressions, this->m_cmask,
                                             m_coordDim),
            m_streamID);
    }
};

} // namespace Nektar::Operators::detail
#endif
