///////////////////////////////////////////////////////////////////////////////
//
// File: ExpressionSerialAVXGeneric.hpp
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
 * @file ExpressionSerialAVXGeneric.hpp
 * @brief Host-side Generic implementation of the Expression operator:
 * interprets the parsed expressions point by point.
 *
 * @details
 * Following the element operators' file-naming scheme, CMake picks
 * this single "SerialAVX" file up for both the Serial and the AVX
 * execution spaces. It also serves the Device execution space when that
 * runs on the host (SYCL_ENABLE_CPU), for which the implementation
 * carries Device-only stream-synchronisation barriers.
 *
 * Unlike the sum-factorised operator families there is no SIMD
 * vectorisation here: every quadrature point is evaluated by a scalar
 * call into the LibUtilities::Equation interpreter, in double
 * precision regardless of TData. Interleaved (AVX-width) storage is
 * handled purely in the addressing.
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

#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include <MultiRegions/ElmtOps/Expression/ExpressionBlockOp.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Host (Serial/AVX) block-level Expression evaluator: for every
 * quadrature point of the block, gathers the expression variables and
 * evaluates each component's parsed expression through the
 * LibUtilities::Equation interpreter.
 *
 * @details
 * The quadrature-point coordinates are fetched once per configuration
 * from the data warehouse (CoordKey, keyed on the block and the
 * input's interleave width) and shared between operators. The table
 * stores, per element group, per point, per coordinate direction, one
 * value for each of the `interleave width` elements of the group, with
 * padding lanes set to zero -- the same element-interleaved layout as
 * the field data, so kernel and table walk in step.
 *
 * For each point the variable vector handed to Equation::Evaluate is
 * assembled as [x, y, z, t, EVARS...]: coordinates above the block's
 * coordinate dimension default to zero (relevant for boundary or
 * manifold elements embedded in a higher-dimensional domain), `t` is
 * the configured time, and EVAR i takes the value of input component i
 * at the same point. Each output component is then assigned (or, with
 * the append flag, incremented by) scale times its expression's value.
 *
 * @tparam ExecSpace      Execution back-end. Normally
 *                        NektarSpaces::Serial or NektarSpaces::AVX;
 *                        NektarSpaces::Device when that space runs
 *                        on the host (see the file notes), in which
 *                        case the per-block stream is synchronised
 *                        around the evaluation loop.
 * @tparam Implementation Implementation tag; only Generic is
 *                        registered for this family.
 * @tparam TData          Floating-point type of the field data. The
 *                        interpreter evaluates in double regardless
 *                        and the result is converted on store.
 *
 * @see ExpressionBlockOp for the settings this class evaluates with.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class ExpressionBlockOpImpl : public ExpressionBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Construct the evaluator for one block, caching the
     * block-wide sizes read from the representative expansion.
     *
     * @param   block_idx       Index of the block within the expansion
     *                          list's Collections.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse the coordinate table is
     *                          fetched from.
     */
    ExpressionBlockOpImpl(const unsigned int block_idx,
                          const LocalRegions::ExpansionSharedPtr &exp,
                          LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : ExpressionBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        // Determine shape and type of the element.
        this->m_dimension = exp->GetShapeDimension();
        this->m_coordDim  = exp->GetCoordim();
        this->m_nqTot     = exp->GetTotPoints();
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
    /// Reference-element (shape) dimension of the block's elements.
    unsigned int m_dimension;
    /// Coordinate dimension of the embedding space; may exceed
    /// #m_dimension for boundary or manifold elements.
    unsigned int m_coordDim;
    /// Quadrature points per element.
    unsigned int m_nqTot;

    /**
     * @brief Evaluate the expressions at every quadrature point of the
     * block (see the class notes for the per-point variable
     * gathering).
     *
     * @p inblock and @p outblock may refer to the same block, in which
     * case the evaluation is performed in place on the shared storage.
     * When appending into a distinct output block, the output storage
     * of each component is first reshaped from the output's to the
     * input's interleave width -- ReshapeStorage() takes the target
     * width as its first argument, see
     * LibUtilities/BasicUtils/Utils/UtilsKernels.hpp -- so that the
     * accumulation reads and writes one layout. In all cases the
     * output block's interleave width is set to the input's on return,
     * since the results are written in the input's layout.
     *
     * @note The arithmetic of that ReshapeStorage() call has not been
     * reconciled with the storage it addresses. The element count
     * passed is chunkSize, i.e. the input's interleave width, whereas
     * the span between two successive calls covers width_ratio
     * element groups; when the output's width exceeds the input's,
     * the group count `nelmt / currInterleaveWidth` that
     * ReshapeStorage() hands to deInterleave() therefore truncates to
     * zero. The pointer is likewise advanced by chunkSize values per
     * element group rather than by one group's worth of storage
     * (#m_nqTot * chunkSize). The in-place and overwrite paths do not
     * go through this call.
     *
     * Padded elements are not evaluated: unless appending, every
     * output component is set to zero there.
     *
     * The component mask is checked for size but not consulted: every
     * component's expression is evaluated (cf. the note in
     * ExpressionOp::SetComponentMask). When this class serves the
     * Device execution space on the host, the block's stream is
     * synchronised before and after each element group's evaluation so
     * the host-side interpreter cannot overlap with device-side work
     * on the same data.
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
        const auto width_ratio        = (outInterleaveWidth == 1)
                                            ? 1
                                            : outInterleaveWidth / inInterleaveWidth;
        const auto chunkSize          = inInterleaveWidth;

        const auto coordptr0 =
            this->m_dataWarehouse->template GetData<MemSpace>(
                LocalRegions::CoordKey<TData>(this->m_block_idx,
                                              inInterleaveWidth, false));

        // Loop over homogeneous modes.
        for (unsigned int n = 0; n < inblock.GetNumHomoModes(); ++n)
        {
            // Pre-allocate vector for point-wise fielddata
            std::vector<double> fielddata(this->m_numEvars[0]);

            auto coordptr = coordptr0;
            for (size_t e = 0, cnt = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                for (unsigned int nc = 0; nc < outblock.GetNumComponents();
                     ++nc)
                {
                    if (&inblock != &outblock && this->m_append &&
                        e % width_ratio == 0)
                    {
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            inInterleaveWidth, outInterleaveWidth, chunkSize,
                            outblock.GetNumData(),
                            (TData *)outptr +
                                (n + nc * outblock.GetNumHomoModes()) *
                                    compSize +
                                e * chunkSize);
                    }
                }

                // Synchronization barrier to allow SYCL-CPU back-end to
                // reuse Serial/AVX code.
                if constexpr (std::is_same_v<ExecSpace, NektarSpaces::Device>)
                {
                    nekStreamSynchronize(this->m_block_idx + 1);
                }

                // Kernel operation.
                for (unsigned int pt = 0; pt < this->m_nqTot; ++pt)
                {
                    for (unsigned int i = 0; i < chunkSize; ++i, ++cnt)
                    {
                        // Padded elements are not evaluated; their padding is
                        // left zero.
                        if (e * chunkSize + i >= inblock.GetNumElements())
                        {
                            for (unsigned int nc = 0;
                                 !this->m_append &&
                                 nc < outblock.GetNumComponents();
                                 ++nc)
                            {
                                outptr[(n + nc * outblock.GetNumHomoModes()) *
                                           compSize +
                                       cnt] = 0.0;
                            }
                            continue;
                        }

                        // Gather fielddata
                        // Note we set y and z coordinate only if the
                        // coordinate dimension is large enough otherwise
                        // they default to zero This is relevant for example
                        // for boundary elements where the domain uses one
                        // more dimension than the boundary
                        fielddata[0] = coordptr[i];
                        fielddata[1] = (this->m_coordDim > 1)
                                           ? coordptr[chunkSize + i]
                                           : 0.0;
                        fielddata[2] = (this->m_coordDim > 2)
                                           ? coordptr[2 * chunkSize + i]
                                           : 0.0;
                        fielddata[3] = this->m_time;

                        // Add EVARS, if required
                        // Note we assume that inblock holds all fields as
                        // components

                        for (unsigned int nc = 0; nc < this->m_numEvars[0] - 4;
                             nc++)
                        {
                            fielddata[4 + nc] =
                                inptr[(n + nc * inblock.GetNumHomoModes()) *
                                          compSize +
                                      cnt];
                        }

                        for (unsigned int nc = 0;
                             nc < outblock.GetNumComponents(); ++nc)
                        {
                            // Evaluate the function assuming fixed input of
                            // x, y and z coordinate.
                            auto fce =
                                this->m_expressions[nc]->Evaluate(fielddata);

                            // Add fce to outptr.
                            const size_t ind =
                                (n + nc * outblock.GetNumHomoModes()) *
                                    compSize +
                                cnt;
                            outptr[ind] =
                                (this->m_append)
                                    ? outptr[ind] + this->m_scale * fce
                                    : this->m_scale * fce;
                        }
                    }
                    coordptr += this->m_coordDim * chunkSize;
                }

                // Synchronization barrier to allow SYCL-CPU back-end to
                // reuse Serial/AVX code.
                if constexpr (std::is_same_v<ExecSpace, NektarSpaces::Device>)
                {
                    nekStreamSynchronize(this->m_block_idx + 1);
                }
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
    }

    /**
     * @brief Store the expressions; no further preparation is needed,
     * since they are interpreted afresh at every v_Apply().
     *
     * @param   exprs   Parsed expressions, one per component.
     */
    void v_SetExpressions(
        const std::vector<LibUtilities::EquationSharedPtr> &exprs) override
    {
        this->m_expressions = exprs;
    }
};

} // namespace Nektar::MultiRegions::detail
