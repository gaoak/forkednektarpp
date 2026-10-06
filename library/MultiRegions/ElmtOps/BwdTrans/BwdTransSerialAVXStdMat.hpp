///////////////////////////////////////////////////////////////////////////////
//
// File: BwdTransSerialAVXStdMat.hpp
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
// Description: Serial/AVX standard-matrix implementation of the
// per-block backward transform
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file BwdTransSerialAVXStdMat.hpp
 * @brief Serial/AVX standard-matrix (StdMat) implementation of the
 * per-block backward transform.
 *
 * All elements of a block share the same basis, so one dense
 * standard-region backward-transform matrix serves the whole block.
 * This header defines the detail::BwdTransBlockOpImpl used when the
 * "StdMat" implementation is requested on the Serial or AVX execution
 * space; CMake includes it into the translation units generated for
 * exactly those factory registrations, which is why this header, the
 * SumFac header and the device headers may each define the same primary
 * template without ever colliding.
 *
 * @see BwdTransSerialAVXSumFac.hpp for the sum-factorised alternative
 * and BwdTransDeviceStdMat.hpp for the device StdMat path.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, execution space and
 * implementation; it should not normally be included by any other
 * translation unit. Other code goes through BwdTransOp.hpp and the
 * operator factory: including this header directly instantiates the
 * whole template set in that translation unit, which is what the
 * generated sources exist to avoid.
 */

#pragma once

#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp"
#include <MultiRegions/ElmtOps/BwdTrans/BwdTransBlockOp.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief StdMat block implementation of the backward transform for the
 * Serial and AVX execution spaces: one small dense matrix multiply per
 * group of SIMD lanes.
 *
 * The constructor fetches the block's transposed standard-region
 * backward-transform matrix -- a StdRegions::StdMatKey with
 * StdRegions::eBwdTransStdMatTranspose, an nmTot x nqTot column-major
 * array \f$B^T\f$ -- from the data warehouse, so it is built once and
 * shared between operators. One libxsmm GEMM per group of
 * simd_t::width elements then computes
 *
 *     out-panel (width x nqTot) = coeff-panel (width x nmTot) * B^T,
 *
 * where the panels are the group's interleaved storage: lane j of a
 * panel column holds element j's value of that mode or point, so the
 * vectorisation is across the elements of the group. On the Serial
 * execution space simd_t is scalar (width 1) and the panel degenerates
 * to a single element's coefficient vector. In append mode the GEMM
 * runs with beta = 1 and accumulates onto the existing output.
 *
 * The element geometry never enters: the backward transform is a
 * standard-region operation, so #m_isDeformed and #m_coordDim are
 * recorded only for uniformity with the other element operators.
 *
 * @tparam ExecSpace       NektarSpaces::Serial or NektarSpaces::AVX;
 *                         sets simd_t via simd_type_if (scalar for
 *                         Serial, tinysimd::simd for AVX).
 * @tparam Implementation  Implementation tag the including translation
 *                         unit registers (MultiRegions::StdMat here).
 * @tparam TData           Floating-point type of the field data.
 *
 * @see BwdTransDeviceStdMat.hpp for the device StdMat path.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class BwdTransBlockOpImpl : public BwdTransBlockOp<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Fetch the block's transposed backward-transform matrix and
     * record the shape sizes the GEMM is dispatched for.
     *
     * @param   block_idx       Index of the block within the expansion
     *                          list's Collections.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse the matrix is cached in.
     */
    BwdTransBlockOpImpl(const unsigned int block_idx,
                        const LocalRegions::ExpansionSharedPtr &exp,
                        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : BwdTransBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
        m_dimension = exp->GetShapeDimension();
        m_coordDim  = exp->GetCoordim();
        m_nmTot     = exp->GetNcoeffs();
        m_nqTot     = exp->GetTotPoints();

        // Fetch matrix.
        std::vector<LibUtilities::BasisKey> basisKeys(
            m_dimension, LibUtilities::NullBasisKey);
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            basisKeys[d] = exp->GetBasis(d)->GetBasisKey();
        }

        LibUtilities::PointsType nodalType =
            (exp->IsNodalNonTensorialExp())
                ? exp->GetNodalPointsKey().GetPointsType()
                : LibUtilities::eNoPointsType;

        m_matptr = dataWarehouse->template GetData<MemSpace>(
            StdRegions::StdMatKey<TData>(basisKeys, m_shapeType,
                                         StdRegions::eBwdTransStdMatTranspose,
                                         nodalType));
    }

    /// Block-operator factory registration key,
    /// `"BlockBwdTrans" + execution space + implementation`. Defined by
    /// the CMake-generated declaration file, whose initialiser performs
    /// the registration.
    static std::string className;

    /**
     * @brief Creator function registered with the block-operator factory
     * under #className.
     *
     * @param   block_idx       Index of the block within the expansion
     *                          list's Collections.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     *
     * @return The newly created block operator.
     */
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Coeff, FieldState::Phys, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            BwdTransBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    /// Interleave width the GEMM operates at: the SIMD vector width
    /// (1 on Serial).
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

    /// Shape shared by every element of the block.
    LibUtilities::ShapeType m_shapeType;
    /// Whether the block's geometry is deformed; unused here (see the
    /// class notes).
    bool m_isDeformed;
    /// Dimension of the reference element (1, 2 or 3).
    unsigned int m_dimension;
    /// Coordinate dimension of the (possibly embedded) element; unused
    /// here.
    unsigned int m_coordDim;
    /// Coefficients per element.
    unsigned int m_nmTot;
    /// Quadrature points per element.
    unsigned int m_nqTot;
    /// Transposed backward-transform matrix (nmTot x nqTot,
    /// column-major), cached in the data warehouse.
    const TData *m_matptr;

    /**
     * @brief Backward-transform every element of the block by dense
     * matrix multiplication.
     *
     * The field may be interleaved at a width other than simd_t::width:
     * a field is interleaved at the vector width of the execution space
     * it was prepared for, which need not equal simd_t::width here. Work
     * therefore proceeds in chunks of `chunkSize = max(simd width, input
     * width)` elements per component: each chunk is reshaped in place to
     * simd_t::width interleaving, its `width_ratio` lane groups are
     * multiplied through the GEMM kernel, and the chunk is reshaped back
     * once its last group is done. In append mode the output chunk is
     * also reshaped forward first, so the beta = 1 accumulation reads it
     * in the layout it is written in. On return the output block's
     * interleave width is set to the input's.
     *
     * @param   inblock     Coefficient-space input block.
     * @param   outblock    Physical-space output block; overwritten, or
     *                      accumulated into in append mode.
     */
    void v_Apply(
        LibUtilities::BlockAccessor<TData, FieldState::Coeff> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock) override
    {
        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = (this->m_append)
                          ? outblock.template GetPtr<MemSpace, ReadWrite>()
                          : outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto inInterleaveWidth  = inblock.GetInterleaveWidth();
        const auto outInterleaveWidth = outblock.GetInterleaveWidth();
        const auto width_ratio =
            (inInterleaveWidth == 1)
                ? 1
                : inInterleaveWidth / m_implInterleaveWidth;
        const auto chunkSize =
            std::max(m_implInterleaveWidth, inInterleaveWidth);

        // Dispatch kernel.
        auto gemm_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
            simd_t::width, m_nqTot, m_nmTot, 1.0, (TData)this->m_append);

        // Loop over components.
        for (unsigned int n = 0;
             n < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++n)
        {
            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        m_implInterleaveWidth, inInterleaveWidth, chunkSize,
                        m_nmTot, (TData *)inptr);
                    if (this->m_append)
                    {
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            m_implInterleaveWidth, outInterleaveWidth,
                            chunkSize, m_nqTot, (TData *)outptr);
                    }
                }

                // Perform matrix-matrix multiply.
                gemm_kernel(inptr, m_matptr, outptr);

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                        m_nmTot,
                        (TData *)inptr -
                            (width_ratio - 1) * m_nmTot * simd_t::width);
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                        m_nqTot,
                        (TData *)outptr -
                            (width_ratio - 1) * m_nqTot * simd_t::width);
                }

                // Increment pointers.
                inptr += m_nmTot * simd_t::width;
                outptr += m_nqTot * simd_t::width;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
    }
};

} // namespace Nektar::MultiRegions::detail
