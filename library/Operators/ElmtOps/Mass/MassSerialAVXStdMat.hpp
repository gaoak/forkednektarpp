///////////////////////////////////////////////////////////////////////////////
//
// File: MassSerialAVXStdMat.hpp
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

#include "Operators/ElmtOps/Mass/MassOp.hpp"
#include "Operators/NekBlas/NekBlas.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseSerialAVXStdMatKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class MassBlockOpImpl : public MassBlockOp<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    MassBlockOpImpl(const LocalRegions::ExpansionSharedPtr &exp,
                    NekDataWarehouseSharedPtr dataWarehouse)
        : MassBlockOp<TData>(exp, dataWarehouse)
    {
        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetMetricInfo()->GetGtype() == SpatialDomains::eDeformed;
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

        if (m_isDeformed)
        {
            m_bwdmat = dataWarehouse->template GetData<ExecSpace>(
                StdMatKey<TData>(basisKeys, m_shapeType,
                                 eBwdTransStdMatTranspose, nodalType));
            m_ipbmat = dataWarehouse->template GetData<ExecSpace>(
                StdMatKey<TData>(basisKeys, m_shapeType,
                                 eIProductWRTBaseStdMatTranspose, nodalType));
        }
        else
        {
            m_massmat =
                dataWarehouse->template GetData<ExecSpace>(StdMatKey<TData>(
                    basisKeys, m_shapeType, eMassStdMatTranspose, nodalType));
        }
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<BlockOperator<TData>> Instantiate(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            MassBlockOpImpl<ExecSpace, Implementation, TData>>(exp,
                                                               dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

    LibUtilities::ShapeType m_shapeType;
    bool m_isDeformed;
    unsigned int m_dimension;
    unsigned int m_coordDim;
    unsigned int m_nmTot;
    unsigned int m_nqTot;
    const TData *m_massmat;
    const TData *m_bwdmat;
    const TData *m_ipbmat;
    MemoryRegion<TData> m_wsp;

    void v_Apply(BlockAccessor<TData> &inblock,
                 BlockAccessor<TData> &outblock) override
    {
        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch Jacobian.
        auto jacptr_init = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));

        // Allocate storage.
        if (m_wsp.size() == 0)
        {
            m_wsp = MemoryRegion<TData>(simd_t::width * m_nqTot,
                                        ExecSpace::alignment);
        }

        // Get workspace pointer.
        auto wspptr = m_wsp.template GetPtr<MemSpace, WriteOnly>();

        // Dispatch kernel.
        auto bwd_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
            simd_t::width, m_nqTot, m_nmTot, 1.0, 0.0);
        auto ipb_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
            simd_t::width, m_nmTot, m_nqTot, 1.0, 0.0);
        auto mass_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
            simd_t::width, m_nmTot, m_nmTot, 1.0, 0.0);

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();
        const auto width_ratio     = (interleaveWidth == 1)
                                         ? 1
                                         : interleaveWidth / m_implInterleaveWidth;
        const auto chunkSize = std::max(m_implInterleaveWidth, interleaveWidth);

        // Loop over components.
        for (unsigned int n = 0;
             n < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++n)
        {
            auto jacptr = jacptr_init;

            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                              interleaveWidth, chunkSize,
                                              m_nmTot, (TData *)inptr);
                }

                if (m_isDeformed)
                {
                    // Step 1: BwdTrans
                    // Perform matrix-matrix multiply.
                    bwd_kernel(inptr, m_bwdmat, wspptr);

                    // Multiply by jacobian.
                    MultiplyByJacobianKernel<ExecSpace, true>(
                        1, m_nqTot, reinterpret_cast<const simd_t *>(jacptr),
                        reinterpret_cast<const simd_t *>(wspptr),
                        reinterpret_cast<simd_t *>(wspptr), 1.0);

                    // Step 2: IProduct
                    // Perform matrix-matrix multiply.
                    ipb_kernel(wspptr, m_ipbmat, outptr);

                    // Increment pointers.
                    jacptr += m_nqTot * simd_t::width;
                }
                else
                {
                    // Perform matrix-matrix multiply.
                    mass_kernel(inptr, m_massmat, outptr);

                    // Multiply by jacobian.
                    MultiplyByJacobianKernel<ExecSpace, false>(
                        1, m_nmTot, reinterpret_cast<const simd_t *>(jacptr),
                        reinterpret_cast<const simd_t *>(outptr),
                        reinterpret_cast<simd_t *>(outptr), 1.0);

                    // Increment pointers.
                    jacptr += simd_t::width;
                }

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        m_nmTot,
                        (TData *)inptr -
                            (width_ratio - 1) * m_nmTot * simd_t::width);
                    ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        m_nmTot,
                        (TData *)outptr -
                            (width_ratio - 1) * m_nmTot * simd_t::width);
                }

                // Increment pointers.
                inptr += m_nmTot * simd_t::width;
                outptr += m_nmTot * simd_t::width;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
