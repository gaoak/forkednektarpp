///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionSerialAVXStdMat.hpp
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
#include "Operators/ElmtOps/Advection/AdvectionBlockOp.hpp"

#include "Operators/ElmtOps/Advection/AdvectionSerialAVXStdMatKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class AdvectionBlockOpImpl : public AdvectionBlockOp<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    AdvectionBlockOpImpl(const unsigned int block_idx,
                         const LocalRegions::ExpansionSharedPtr &exp,
                         LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : AdvectionBlockOp<TData>(block_idx, exp, dataWarehouse)
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
            AdvectionBlockOpImpl<ExecSpace, Implementation, TData>>(
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
        // Reshape advection velocity, if necessary.
        if (this->m_advVel->GetInterleaveWidth() != m_implInterleaveWidth)
        {
            auto advVelPtr =
                this->m_advVel->template GetPtr<MemSpace, ReadWrite>();
            LibUtilities::ReshapeStorage<ExecSpace>(
                m_implInterleaveWidth, this->m_advVel->GetInterleaveWidth(),
                this->m_advVel->GetNumElementsWithPadding() *
                    this->m_advVel->GetNumComponents() *
                    this->m_advVel->GetNumHomoModes(),
                this->m_advVel->GetNumData(), advVelPtr);
            this->m_advVel->template SetInterleaveWidth<TData>(
                m_implInterleaveWidth);
        }

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

        // Get static workspace pointer.
        auto derivptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                m_dimension * simd_t::width * m_nqTot);

        // Dispatch kernel.
        auto gemm_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
            simd_t::width, m_nqTot, m_nqTot, 1.0, 0.0);

        // Loop over components. The advection velocity carries the same
        // planes as the input, so one of its components spans all of them.
        const auto nhomo = inblock.GetNumHomoModes();
        const auto advelsize =
            m_nqTot * inblock.GetNumElmtGroups(m_implInterleaveWidth) * nhomo;
        const auto derivsize = m_nqTot;
        for (unsigned int n = 0; n < inblock.GetNumComponents() * nhomo; ++n)
        {
            // n runs over the components of the input with their planes
            // innermost, so this iteration takes the velocity of plane
            // n % nhomo, and advelsize steps from there to the next velocity
            // component on the same plane.
            auto advVelPtr =
                this->m_advVel->template GetPtr<MemSpace, ReadOnly>() +
                (n % nhomo) * this->m_advVel->CompSize();
            auto dfptr = m_dfptr;

            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        m_implInterleaveWidth, inInterleaveWidth, chunkSize,
                        m_nqTot, (TData *)inptr);
                }

                // Perform matrix-matrix multiply.
                for (unsigned int d = 0; d < m_dimension; d++)
                {
                    gemm_kernel(inptr, m_matptr + d * m_nqTot * m_nqTot,
                                derivptr + d * m_nqTot * simd_t::width);
                }

                // Multiply by derivative factor and Advection Velocity.
                if (m_isDeformed)
                {
                    if (this->m_append)
                    {
                        // Reshape, if necessary.
                        if (e % width_ratio == 0)
                        {
                            LibUtilities::ReshapeStorage<ExecSpace>(
                                m_implInterleaveWidth, outInterleaveWidth,
                                chunkSize, m_nqTot, (TData *)outptr);
                        }

                        MultiplyByDerivFactorAndAdvecVelKernel<ExecSpace, true,
                                                               true>(
                            m_nqTot, m_coordDim, m_dimension, 1, derivsize,
                            reinterpret_cast<const simd_t *>(dfptr),
                            reinterpret_cast<const simd_t *>(advVelPtr),
                            advelsize,
                            reinterpret_cast<const simd_t *>(derivptr),
                            reinterpret_cast<simd_t *>(outptr), this->m_scale);
                    }
                    else
                    {
                        MultiplyByDerivFactorAndAdvecVelKernel<ExecSpace, false,
                                                               true>(
                            m_nqTot, m_coordDim, m_dimension, 1, derivsize,
                            reinterpret_cast<const simd_t *>(dfptr),
                            reinterpret_cast<const simd_t *>(advVelPtr),
                            advelsize,
                            reinterpret_cast<const simd_t *>(derivptr),
                            reinterpret_cast<simd_t *>(outptr), this->m_scale);
                    }
                    dfptr += m_coordDim * m_dimension * m_nqTot * simd_t::width;
                }
                else
                {
                    if (this->m_append)
                    {
                        // Reshape, if necessary.
                        if (e % width_ratio == 0)
                        {
                            LibUtilities::ReshapeStorage<ExecSpace>(
                                m_implInterleaveWidth, outInterleaveWidth,
                                chunkSize, m_nqTot, (TData *)outptr);
                        }

                        MultiplyByDerivFactorAndAdvecVelKernel<ExecSpace, true,
                                                               false>(
                            m_nqTot, m_coordDim, m_dimension, 1, derivsize,
                            reinterpret_cast<const simd_t *>(dfptr),
                            reinterpret_cast<const simd_t *>(advVelPtr),
                            advelsize,
                            reinterpret_cast<const simd_t *>(derivptr),
                            reinterpret_cast<simd_t *>(outptr), this->m_scale);
                    }
                    else
                    {
                        MultiplyByDerivFactorAndAdvecVelKernel<ExecSpace, false,
                                                               false>(
                            m_nqTot, m_coordDim, m_dimension, 1, derivsize,
                            reinterpret_cast<const simd_t *>(dfptr),
                            reinterpret_cast<const simd_t *>(advVelPtr),
                            advelsize,
                            reinterpret_cast<const simd_t *>(derivptr),
                            reinterpret_cast<simd_t *>(outptr), this->m_scale);
                    }
                    dfptr += m_coordDim * m_dimension * simd_t::width;
                }

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                        m_nqTot,
                        (TData *)inptr -
                            (width_ratio - 1) * m_nqTot * simd_t::width);
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                        m_nqTot,
                        (TData *)outptr -
                            (width_ratio - 1) * m_nqTot * simd_t::width);
                }

                // Increment pointers.
                inptr += m_nqTot * simd_t::width;
                outptr += m_nqTot * simd_t::width;
                advVelPtr += m_nqTot * simd_t::width;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
