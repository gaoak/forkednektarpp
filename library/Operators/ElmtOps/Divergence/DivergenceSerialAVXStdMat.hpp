///////////////////////////////////////////////////////////////////////////////
//
// File: DivergenceSerialAVXStdMat.hpp
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

#pragma once

#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp"
#include "Operators/ElmtOps/Divergence/DivergenceBlockOp.hpp"

#include "Operators/ElmtOps/PhysDeriv/PhysDerivSerialAVXStdMatKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class DivergenceBlockOpImpl : public DivergenceBlockOp<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    DivergenceBlockOpImpl(const unsigned int block_idx,
                          const LocalRegions::ExpansionSharedPtr &exp,
                          LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : DivergenceBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
        m_dimension = exp->GetShapeDimension();
        m_coordDim  = exp->GetCoordim();
        m_nmTot     = exp->GetNcoeffs();
        m_nqTot     = exp->GetTotPoints();

        ASSERTL1(m_dimension == m_coordDim,
                 "Assuming that these valeus are the same");

        // Fetch matrix.
        std::vector<LibUtilities::BasisKey> basisKeys(
            m_dimension, LibUtilities::NullBasisKey);
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            basisKeys[d] = exp->GetBasis(d)->GetBasisKey();
        }

        m_matptr = dataWarehouse->template GetData<MemSpace>(
            StdRegions::StdMatKey<TData>(
                basisKeys, m_shapeType, StdRegions::ePhysDerivStdMatTranspose));

        // Fetch derivative factor.
        m_dfptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::DerivFactorKey<TData>(block_idx,
                                                m_implInterleaveWidth, false));
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
            DivergenceBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

    LibUtilities::ShapeType m_shapeType;
    bool m_isDeformed;
    unsigned int m_dimension;
    unsigned int m_coordDim;
    unsigned int m_nmTot;
    unsigned int m_nqTot;
    const TData *m_matptr;
    const TData *m_dfptr;

    void v_Apply(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock) override
    {
        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();
        const auto width_ratio     = (interleaveWidth == 1)
                                         ? 1
                                         : interleaveWidth / m_implInterleaveWidth;
        const auto chunkSize = std::max(m_implInterleaveWidth, interleaveWidth);

        // Get static workspace pointer.
        auto wspptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                m_dimension * simd_t::width * m_nqTot);

        // Dispatch kernel.
        auto gemm_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
            simd_t::width, m_nqTot, m_nqTot, 1.0, 0.0);

        // Loop over components.
        const auto inoffset = inblock.CompSize() * inblock.GetNumHomoModes();

        auto dfptr = m_dfptr;

        // Loop over element groups.
        for (size_t e = 0; e < inblock.GetNumElmtGroups(m_implInterleaveWidth);
             ++e)
        {

            // du/dx
            // Reshape, if necessary.
            if (e % width_ratio == 0)
            {
                LibUtilities::ReshapeStorage<ExecSpace>(
                    m_implInterleaveWidth, interleaveWidth, chunkSize, m_nqTot,
                    (TData *)inptr);
            }

            // calculate dudx
            // Perform matrix-matrix multiply.
            for (unsigned int d = 0; d < m_dimension; d++)
            {
                gemm_kernel(inptr, m_matptr + d * m_nqTot * m_nqTot,
                            wspptr + d * m_nqTot * simd_t::width);
            }

            // Multiply by derivative factor.
            if (m_isDeformed)
            {
                MultiplyByDirDerivFactorKernel<ExecSpace, false, true>(
                    0, m_nqTot, m_coordDim, m_dimension, 1,
                    reinterpret_cast<const simd_t *>(dfptr),
                    reinterpret_cast<const simd_t *>(wspptr),
                    reinterpret_cast<simd_t *>(outptr));
            }
            else
            {
                MultiplyByDirDerivFactorKernel<ExecSpace, false, false>(
                    0, m_nqTot, m_coordDim, m_dimension, 1,
                    reinterpret_cast<const simd_t *>(dfptr),
                    reinterpret_cast<const simd_t *>(wspptr),
                    reinterpret_cast<simd_t *>(outptr));
            }

            // Calculate dv/dy & dw/dz
            for (unsigned c = 1; c < m_dimension; ++c)
            {

                if (e % width_ratio == 0)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        m_implInterleaveWidth, interleaveWidth, chunkSize,
                        m_nqTot, (TData *)inptr + c * inoffset);
                }

                // Perform matrix-matrix multiply.
                for (unsigned int d = 0; d < m_dimension; d++)
                {
                    gemm_kernel(inptr + c * inoffset,
                                m_matptr + d * m_nqTot * m_nqTot,
                                wspptr + d * m_nqTot * simd_t::width);
                }

                // Multiply by derivative factor.
                if (m_isDeformed)
                {
                    MultiplyByDirDerivFactorKernel<ExecSpace, true, true>(
                        c, m_nqTot, m_coordDim, m_dimension, 1,
                        reinterpret_cast<const simd_t *>(dfptr),
                        reinterpret_cast<const simd_t *>(wspptr),
                        reinterpret_cast<simd_t *>(outptr));
                }
                else
                {
                    MultiplyByDirDerivFactorKernel<ExecSpace, true, false>(
                        c, m_nqTot, m_coordDim, m_dimension, 1,
                        reinterpret_cast<const simd_t *>(dfptr),
                        reinterpret_cast<const simd_t *>(wspptr),
                        reinterpret_cast<simd_t *>(outptr));
                }
            }

            // Reshape back, if necessary.
            if (e % width_ratio == width_ratio - 1)
            {
                for (unsigned c = 0; c < m_dimension; ++c)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        m_nqTot,
                        (TData *)inptr + c * inoffset -
                            (width_ratio - 1) * m_nqTot * simd_t::width);
                }

                LibUtilities::ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, chunkSize, m_nqTot,
                    (TData *)outptr -
                        (width_ratio - 1) * m_nqTot * simd_t::width);
            }

            // Increment pointer.
            dfptr += (m_isDeformed)
                         ? m_coordDim * m_dimension * m_nqTot * simd_t::width
                         : m_coordDim * m_dimension * simd_t::width;
            inptr += m_nqTot * simd_t::width;
            outptr += m_nqTot * simd_t::width;
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
