///////////////////////////////////////////////////////////////////////////////
//
// File: MeanRemovalSerialAVX.hpp
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

#include <SolverCore/MeanRemoval/MeanRemovalBlockOp.hpp>

#include <LibUtilities/BasicUtils/DataWarehouse/BasisDataWarehouse.hpp>
#include <LocalRegions/DataWarehouse/GeometricDataWarehouse.hpp>

#include <LibUtilities/BasicUtils/Math/Math.hpp>
#include <LibUtilities/BasicUtils/Utils/UtilsKernels.hpp>

namespace Nektar::SolverCore::detail
{

template <typename ExecSpace, typename TData>
class MeanRemovalBlockOpImpl : public MeanRemovalBlockOp<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    MeanRemovalBlockOpImpl(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : MeanRemovalBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
        m_nqTot     = exp->GetTotPoints();
        m_dimension = exp->GetShapeDimension();
        m_coordDim  = exp->GetCoordim();

        // Get basis keys for fetching matrices
        std::vector<LibUtilities::BasisKey> basisKeys(
            m_dimension, LibUtilities::NullBasisKey);
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            basisKeys[d] = exp->GetBasis(d)->GetBasisKey();

            // Fetch element size.
            m_nq.push_back(exp->GetNumPoints(d));

            // Fetch basis data.
            m_W.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(basisKeys[d],
                                                  LibUtilities::eWeights)));
        }

        // Fetch Jacobian.
        m_jacptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::JacobianKey<TData>(block_idx, m_implInterleaveWidth));
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<MeanRemovalBlockOp<TData>> Instantiate(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<MeanRemovalBlockOpImpl<ExecSpace, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

    LibUtilities::ShapeType m_shapeType;
    bool m_isDeformed;
    unsigned int m_nqTot;
    unsigned int m_dimension;
    unsigned int m_coordDim;
    const TData *m_jacptr;
    std::vector<unsigned int> m_nq;
    std::vector<const TData *> m_W;

#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
    // flag to ensure we only get one warning for alignment otherwise CI system
    // is saturated with warnings
    bool m_warnOnce = false;
#endif

    void v_Apply(LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
                 LibUtilities::MemoryRegion<TData> &data) override
    {
        WARNINGL1(m_warnOnce ||
                      (inblock.GetAlignment() % simd_t::alignment == 0),
                  "Input Field is not aligned to the required alignment "
                  "for the SIMD vector type.");
#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
        m_warnOnce = true;
#endif
        switch (m_dimension)
        {
            case 1:
            {
                OperatorND<1>(inblock, data);
                break;
            }
            case 2:
            {
                OperatorND<2>(inblock, data);
                break;
            }
            case 3:
            {
                OperatorND<3>(inblock, data);
                break;
            }
            default:
                ASSERTL0(false, "MeanRemoval SerialAVX only implemented for "
                                "dimension 1, 2 or 3.");
        }
    }

    template <unsigned int NDIM>
    void OperatorND(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::MemoryRegion<TData> &data)
    {
        OperatorNDImpl(inblock, data,
                       std::make_integer_sequence<unsigned int, NDIM>());
    }

    template <unsigned int... ind>
    void OperatorNDImpl(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::MemoryRegion<TData> &data,
        std::integer_sequence<unsigned int, ind...>)
    {
        unsigned int jacSize = 1;
        if (m_isDeformed)
        {
            jacSize *= m_nqTot;
        }

        // Initialize pointers.
        auto inptr   = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto dataptr = data.template GetPtr<MemSpace, ReadWrite>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();
        const auto width_ratio     = (interleaveWidth == 1)
                                         ? 1
                                         : interleaveWidth / m_implInterleaveWidth;
        const auto chunkSize = std::max(m_implInterleaveWidth, interleaveWidth);

        const unsigned int nComp = inblock.GetNumComponents();
        const unsigned int nHomo = inblock.GetNumHomoModes();

        // Compute mesh volume.
        simd_t vol  = 0.0;
        auto jacptr = m_jacptr;

        // Loop over element groups.
        for (size_t e = 0; e < inblock.GetNumElmtGroups(m_implInterleaveWidth);
             ++e)
        {
            if (m_isDeformed)
            {
                vol += VolumeKernel<true>(
                    m_nq[ind]..., m_W[ind]...,
                    reinterpret_cast<const simd_t *>(jacptr));
            }
            else
            {
                vol += VolumeKernel<false>(
                    m_nq[ind]..., m_W[ind]...,
                    reinterpret_cast<const simd_t *>(jacptr));
            }

            // Increment pointers for the next elmt group.
            jacptr += jacSize * simd_t::width;
        }

        // Accumulate over vector width.
        for (unsigned int i = 0; i < simd_t::width; i++)
        {
            dataptr[nComp] += vol[i];
        }

        // Compute the weighted volume integral of every component.
        for (unsigned int nc = 0; nc < nComp * nHomo; ++nc)
        {
            simd_t acc = 0.0;
            jacptr     = m_jacptr;

            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        m_implInterleaveWidth, interleaveWidth, chunkSize,
                        m_nqTot, (TData *)inptr);
                }

                // Zeroing padding element.
                for (unsigned int q = 0; q < m_nqTot; ++q)
                {
                    const auto offset = q * m_implInterleaveWidth;
                    for (unsigned int i = 0; i < m_implInterleaveWidth; i++)
                    {
                        if (e * m_implInterleaveWidth + i >=
                            inblock.GetNumElements())
                        {
                            ((TData *)inptr)[offset + i] = 0.0;
                        }
                    }
                }

                if (m_isDeformed)
                {
                    acc += IntegralKernel<IntegralOp::Default, true>(
                        m_nq[ind]..., m_W[ind]...,
                        reinterpret_cast<const simd_t *>(jacptr),
                        reinterpret_cast<const simd_t *>(inptr));
                }
                else
                {
                    acc += IntegralKernel<IntegralOp::Default, false>(
                        m_nq[ind]..., m_W[ind]...,
                        reinterpret_cast<const simd_t *>(jacptr),
                        reinterpret_cast<const simd_t *>(inptr));
                }

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        m_nqTot,
                        (TData *)inptr -
                            (width_ratio - 1) * m_nqTot * simd_t::width);
                }

                // Increment pointers for the next elmt group.
                inptr += m_nqTot * simd_t::width;
                jacptr += jacSize * simd_t::width;
            }

            // Accumulate over vector width.
            for (unsigned int i = 0; i < simd_t::width; i++)
            {
                dataptr[nc] += acc[i];
            }
        }
    }
};

} // namespace Nektar::SolverCore::detail
