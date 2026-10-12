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

/**
 * @file ExpressionDeviceGeneric.hpp
 * @brief Device Generic implementation of the Expression operator:
 * compiles the expression strings at run time into a pointwise device
 * kernel.
 *
 * @details
 * Kernel source for the configured expressions is generated once, at
 * SetExpressions() time, compiled at run time and cached; Apply() then
 * only marshals arguments and launches the cached kernel, one thread
 * per stored value.
 *
 * Generation, compilation and launch are done by the back-end helpers:
 * ExpressionDeviceGenericHIPCUDAHelper.hpp compiles with NVRTC/HIPRTC
 * and loads a driver-API module, and ExpressionDeviceGenericSYCLHelper.hpp
 * either wraps those native calls in operations ordered on the
 * per-block SYCL queue (SYCL-on-CUDA/HIP) or compiles SYCL source
 * (SYCL on Intel).
 *
 * When the Device execution space actually runs on the host
 * (SYCL_ENABLE_CPU), this file reduces to including
 * ExpressionSerialAVXGeneric.hpp.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, execution space and
 * implementation; it should not normally be included by any other
 * translation unit. Other code goes through ExpressionOp.hpp and the
 * operator factory: including this header directly instantiates the
 * whole template set in that translation unit, which is what the
 * generated sources exist to avoid.
 */

#pragma once

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include <MultiRegions/ElmtOps/Expression/ExpressionBlockOp.hpp>

#if defined(SYCL_ENABLE_CPU)
// The CPU back-end evaluates the expressions on the host.
#include <MultiRegions/ElmtOps/Expression/ExpressionSerialAVXGeneric.hpp>
#else
#if defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP)
#include <MultiRegions/ElmtOps/Expression/ExpressionDeviceGenericHIPCUDAHelper.hpp>
#else
#include <MultiRegions/ElmtOps/Expression/ExpressionDeviceGenericSYCLHelper.hpp>
#endif

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Device block-level Expression evaluator: runs a runtime-compiled
 * kernel that evaluates every component's expression at every stored
 * point of the block.
 *
 * @details
 * ### The generated kernel
 * v_SetExpressions() generates one kernel templated on an APPEND flag
 * and the interleave width, with the expression strings pasted into its
 * body, and compiles four variants of it (ExpressionKernelIndex):
 * append/overwrite crossed with interleave width 1 (flat storage) and
 * the device warp size (the width Device-space fields are interleaved
 * at), all at TData. The variants are cached in #m_kernels so v_Apply()
 * picks one without recompiling.
 * Masked-out components are omitted from the kernel signature and
 * body entirely, so the component mask costs nothing at evaluation
 * time. The mask must therefore be in its final state before
 * SetExpressions() and not change afterwards: v_Apply() marshals the
 * output pointers against the compiled signature using the current
 * mask.
 *
 * Each thread owns one stored value of one homogeneous mode: it reads
 * its point's coordinates from the data warehouse table (CoordKey, held
 * for one mode; flat or element-interleaved addressing is chosen by the
 * width template parameter), reads its EVAR values at the same flat
 * index of the input components, and assigns or increments each
 * unmasked output component with scale times its expression. A thread
 * on a padded element evaluates nothing, and writes zero to the
 * unmasked outputs unless appending. Because the evaluation is
 * pointwise and input and output share one layout, only the coordinate
 * reads and the padding test need to be interleave-aware.
 *
 * Expressions are evaluated in TData precision, since the generated
 * kernel is instantiated with TData.
 *
 * ### Interoperation
 * On the plain CUDA/HIP back-ends the kernel is launched directly on
 * the per-block stream. On SYCL-on-CUDA/HIP, compilation and launch
 * are enqueued on the per-block SYCL queue as host operations on the
 * native stream, so they order correctly against SYCL kernels. On
 * SYCL on Intel, SYCL source is compiled through the oneAPI
 * kernel-compiler extension and launched on the per-block queue.
 *
 * @tparam ExecSpace      Execution back-end; NektarSpaces::Device.
 * @tparam Implementation Implementation tag; only Generic is
 *                        registered for this family.
 * @tparam TData          Floating-point type of the field data and of
 *                        the generated kernel's arithmetic.
 *
 * @see ExpressionBlockOp for the settings this class evaluates with.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class ExpressionBlockOpImpl : public ExpressionBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Construct the evaluator for one block: caches the
     * block-wide sizes and sets the per-block stream.
     *
     * InitExpressionKernelCompiler() is also called: on the native
     * CUDA/HIP builds it issues a `cudaFree(0)`/`hipFree(0)`, to prime
     * the driver context the later module load needs; on the SYCL
     * back-ends it does nothing and the context comes from the SYCL
     * queue.
     *
     * @param   block_idx       Index of the block within the expansion
     *                          list's Collections; stream
     *                          `block_idx + 1` is used for all
     *                          launches.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse the coordinate table is
     *                          fetched from.
     */
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

    /**
     * @brief Synchronise the block's stream and release the compiled
     * kernels.
     */
    ~ExpressionBlockOpImpl(void)
    {
        nekStreamSynchronize(m_streamID);
        ReleaseExpressionKernels(m_kernels);
    }

    /// Block-operator class name; defined by generated factory code.
    static std::string className;

    /// Creator function registered with BlockOperatorFactory.
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
    /// Per-block stream (block index + 1) used for all launches.
    unsigned int m_streamID;
    /// Reference-element (shape) dimension of the block's elements.
    unsigned int m_dimension;
    /// Coordinate dimension of the embedding space; may exceed
    /// #m_dimension for boundary or manifold elements.
    unsigned int m_coordDim;
    /// Quadrature points per element.
    unsigned int m_nqTot;
    /// Compiled kernel variants, in ExpressionKernelIndex order.
    ExpressionKernels m_kernels;

    /**
     * @brief Launch the pre-compiled expression kernel over every
     * homogeneous mode of the block.
     *
     * An ExpressionKernelArgs is filled with the sizes, the scale
     * factor, the coordinate-table pointer (CoordKey, keyed on the
     * input's interleave width), the time, one input pointer per EVAR
     * and one output pointer per unmasked component, each at the
     * component's first mode. A single launch covers every mode, one
     * thread per stored value, in blocks of the device's default block
     * size, picking the variant for the append flag together with the
     * width-1 variant when the input's interleave width is 1 and the
     * warp-size one otherwise. Only those two widths are compiled, so
     * only they are addressed correctly; any other width is not
     * detected and runs the warp-size kernel.
     *
     * @p inblock and @p outblock may refer to the same block (in-place
     * evaluation). When appending into a distinct output block, the
     * output is first reshaped from its interleave width to the
     * input's, in one call over every component and mode, so the
     * accumulation reads and writes one layout (the block's padded
     * element count); the output block's interleave width is set to the
     * input's on return in all cases.
     *
     * On the SYCL back-ends the launch is enqueued on the per-block
     * SYCL queue (see LaunchExpressionKernel()).
     *
     * @param   inblock     Physical-space input block supplying the
     *                      EVAR values.
     * @param   outblock    Physical-space output block written with
     *                      `scale * f` (or incremented by it in append
     *                      mode).
     */
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
        kernelArgs.nelmt    = inblock.GetNumElements();
        kernelArgs.ndata    = inblock.GetNumData();
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

    /**
     * @brief Store the expressions and compile the expression kernels
     * for them.
     *
     * The source is generated by GetExpressionKernelSource() for the
     * current component mask and the block's coordinate dimension, and
     * compiled by CompileExpressionKernels(), replacing any kernels
     * compiled before. On the SYCL-on-CUDA/HIP back-ends the
     * compilation is enqueued on the per-block SYCL queue as a host
     * operation, so it is ordered against any launches already in
     * flight; elsewhere it runs directly.
     *
     * @param   exprs   Parsed expressions, one per component; the
     *                  first expression's variable list defines the
     *                  EVAR names for all of them.
     */
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

} // namespace Nektar::MultiRegions::detail
#endif
