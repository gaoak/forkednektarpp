///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseImplShared.hpp
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

#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseCUDASumFacKernels.cuh"
#include "Operators/ElmtOps/OperatorIProductWRTBase.hpp"

#include "Operators/Common/OperatorHelper.hpp"
#include "Operators/Field/MemoryRegion.hpp"

namespace Nektar::Operators::detail
{

// Shared implementation
template <typename ExecSpace, typename Implementation, typename TData,
          typename = typename std::enable_if<
              (std::is_same<ExecSpace, NektarSpaces::CUDA>::value &&
               std::is_same<Implementation, Operators::SumFac>::value) ||
              (std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value &&
               std::is_same<Implementation, Operators::StdMat>::value)>::type>
class OperatorIProductWRTBaseImpl : public OperatorIProductWRTBase<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorIProductWRTBaseImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorIProductWRTBase<TData>(expansionList)
    {
        // Initialise the jacobian.
        size_t jacSize = Operator<TData>::GetGeometricFactorSize();
        auto jac       = Operator<TData>::SetJacobian(jacSize);

        m_jac = MemoryRegion<TData>::template fromArray<MemSpace, TData>(
            jac, EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());

        // Initialize the basis data.
        m_basisMap =
            GetBasisData<MemSpace, TData>(expansionList, BASIS_BASIS_DATA);
        m_weightMap =
            GetBasisData<MemSpace, TData>(expansionList, BASIS_WEIGHT_DATA);
    }

    ~OperatorIProductWRTBaseImpl(void)
    {
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out,
               const TData lambda = 1.0) override
    {
        // Copy memory to the device, if necessary and get raw pointers.
        const TData *inPtr = in.template GetPtr<MemSpace, ReadOnly>();
        TData *outPtr      = (lambda == 1.0)
                                 ? out.template GetPtr<MemSpace, WriteOnly>()
                                 : out.template GetPtr<MemSpace, ReadWrite>();

        const TData *jacPtr = m_jac.template GetPtr<MemSpace, ReadOnly>();

        TData *wspPtr = nullptr;

        // Initialize index.
        size_t exp_idx = 0;

        // Initialize basiskey.
        std::vector<LibUtilities::BasisKey> basisKeys(
            3, LibUtilities::NullBasisKey);

        // Loop over the blocks.
        for (auto const &block : in.GetBlocks())
        {
            // Block dependent
            auto const nElmts    = block.num_elements;
            auto const nPadElmts = block.num_padding_elements;

            // Determine shape and type of the element.
            auto const expPtr    = this->m_expansionList->GetExp(exp_idx);
            auto const shapeType = expPtr->DetShapeType();
            auto const dimension = expPtr->GetShapeDimension();
            auto const deformed  = expPtr->GetMetricInfo()->GetGtype() ==
                                  SpatialDomains::eDeformed;
            auto const nqTot   = expPtr->GetTotPoints();
            auto const nmTot   = expPtr->GetNcoeffs();
            auto const ptsKeys = expPtr->GetPointsKeys();

            // Flag for collapsed coordinate correction.
            bool correct = expPtr->GetBasis(0)->GetBasisType() ==
                           LibUtilities::eModified_A;

            // Fetch basis key for the current element type.
            for (size_t d = 0; d < dimension; d++)
            {
                basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
            }

            // Function call to kernel functions.
            if (dimension == 1)
            {
                auto basis0 = m_basisMap[basisKeys[0]]
                                  .template GetPtr<MemSpace, ReadOnly>();
                auto w0 = m_weightMap[basisKeys[0]]
                              .template GetPtr<MemSpace, ReadOnly>();

                auto nm0 = expPtr->GetBasisNumModes(0);
                auto nq0 = expPtr->GetNumPoints(0);

                if (deformed)
                {
                    if (lambda == 1.0)
                    {
                        IProductWRTBase1DKernel<ExecSpace, TData, false, false,
                                                true>(
                            nm0, nq0, nElmts + nPadElmts, basis0, w0, jacPtr,
                            inPtr, outPtr);
                    }
                    else
                    {
                        IProductWRTBase1DKernel<ExecSpace, TData, true, false,
                                                true>(
                            nm0, nq0, nElmts + nPadElmts, basis0, w0, jacPtr,
                            inPtr, outPtr, lambda);
                    }
                }
                else
                {
                    if (lambda == 1.0)
                    {
                        IProductWRTBase1DKernel<ExecSpace, TData, false, false,
                                                false>(
                            nm0, nq0, nElmts + nPadElmts, basis0, w0, jacPtr,
                            inPtr, outPtr);
                    }
                    else
                    {
                        IProductWRTBase1DKernel<ExecSpace, TData, true, false,
                                                false>(
                            nm0, nq0, nElmts + nPadElmts, basis0, w0, jacPtr,
                            inPtr, outPtr, lambda);
                    }
                }
            }
            else if (dimension == 2)
            {
                auto basis0 = m_basisMap[basisKeys[0]]
                                  .template GetPtr<MemSpace, ReadOnly>();
                auto basis1 = m_basisMap[basisKeys[1]]
                                  .template GetPtr<MemSpace, ReadOnly>();
                auto w0 = m_weightMap[basisKeys[0]]
                              .template GetPtr<MemSpace, ReadOnly>();
                auto w1 = m_weightMap[basisKeys[1]]
                              .template GetPtr<MemSpace, ReadOnly>();

                auto nm0 = expPtr->GetBasisNumModes(0);
                auto nm1 = expPtr->GetBasisNumModes(1);
                auto nq0 = expPtr->GetNumPoints(0);
                auto nq1 = expPtr->GetNumPoints(1);

                if constexpr (!FLAG_QP)
                {
                    size_t wspsize = 0;

                    if (shapeType == LibUtilities::Quad)
                    {
                        wspsize = nq1 * (nElmts + nPadElmts);
                    }
                    else if (shapeType == LibUtilities::Tri)
                    {
                        wspsize = nq0 * (nElmts + nPadElmts);
                    }

                    if (m_wspsize < wspsize)
                    {
                        m_wspsize = wspsize;
                        m_wsp = MemoryRegion<TData>::template create<MemSpace>(
                            m_wspsize,
                            EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());
                    }

                    wspPtr = m_wsp.template GetPtr<MemSpace, WriteOnly>();
                }

                if (deformed)
                {
                    if (lambda == 1.0)
                    {
                        IProductWRTBase2DKernel<ExecSpace, TData, false, false,
                                                true>(
                            shapeType, nm0, nm1, nq0, nq1, nElmts + nPadElmts,
                            correct, basis0, basis1, w0, w1, jacPtr, wspPtr,
                            inPtr, outPtr);
                    }
                    else
                    {
                        IProductWRTBase2DKernel<ExecSpace, TData, true, false,
                                                true>(
                            shapeType, nm0, nm1, nq0, nq1, nElmts + nPadElmts,
                            correct, basis0, basis1, w0, w1, jacPtr, wspPtr,
                            inPtr, outPtr, lambda);
                    }
                }
                else
                {
                    if (lambda == 1.0)
                    {
                        IProductWRTBase2DKernel<ExecSpace, TData, false, false,
                                                false>(
                            shapeType, nm0, nm1, nq0, nq1, nElmts + nPadElmts,
                            correct, basis0, basis1, w0, w1, jacPtr, wspPtr,
                            inPtr, outPtr);
                    }
                    else
                    {
                        IProductWRTBase2DKernel<ExecSpace, TData, true, false,
                                                false>(
                            shapeType, nm0, nm1, nq0, nq1, nElmts + nPadElmts,
                            correct, basis0, basis1, w0, w1, jacPtr, wspPtr,
                            inPtr, outPtr, lambda);
                    }
                }
            }
            else if (dimension == 3)
            {
                auto basis0 = m_basisMap[basisKeys[0]]
                                  .template GetPtr<MemSpace, ReadOnly>();
                auto basis1 = m_basisMap[basisKeys[1]]
                                  .template GetPtr<MemSpace, ReadOnly>();
                auto basis2 = m_basisMap[basisKeys[2]]
                                  .template GetPtr<MemSpace, ReadOnly>();
                auto w0 = m_weightMap[basisKeys[0]]
                              .template GetPtr<MemSpace, ReadOnly>();
                auto w1 = m_weightMap[basisKeys[1]]
                              .template GetPtr<MemSpace, ReadOnly>();
                auto w2 = m_weightMap[basisKeys[2]]
                              .template GetPtr<MemSpace, ReadOnly>();

                auto nm0 = expPtr->GetBasisNumModes(0);
                auto nm1 = expPtr->GetBasisNumModes(1);
                auto nm2 = expPtr->GetBasisNumModes(2);
                auto nq0 = expPtr->GetNumPoints(0);
                auto nq1 = expPtr->GetNumPoints(1);
                auto nq2 = expPtr->GetNumPoints(2);

                if constexpr (!FLAG_QP)
                {
                    size_t wspsize = 0;

                    if (shapeType == LibUtilities::Hex)
                    {
                        wspsize = (nq2 * nq1 + nq2) * (nElmts + nPadElmts);
                    }
                    else if (shapeType == LibUtilities::Tet)
                    {
                        wspsize =
                            (nq2 * nq1 + nq2 + nm2) * (nElmts + nPadElmts);
                    }
                    else if (shapeType == LibUtilities::Prism)
                    {
                        wspsize =
                            (nq2 * nq1 + nq2 + nm1) * (nElmts + nPadElmts);
                    }
                    else if (shapeType == LibUtilities::Pyr)
                    {
                        wspsize = (nq2 * nq1 + nq2) * (nElmts + nPadElmts);
                    }

                    if (m_wspsize < wspsize)
                    {
                        m_wspsize = wspsize;
                        m_wsp = MemoryRegion<TData>::template create<MemSpace>(
                            m_wspsize,
                            EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());
                    }

                    wspPtr = m_wsp.template GetPtr<MemSpace, WriteOnly>();
                }

                if (deformed)
                {
                    if (lambda == 1.0)
                    {
                        IProductWRTBase3DKernel<ExecSpace, TData, false, false,
                                                true>(
                            shapeType, nm0, nm1, nm2, nq0, nq1, nq2,
                            nElmts + nPadElmts, correct, basis0, basis1, basis2,
                            w0, w1, w2, jacPtr, wspPtr, inPtr, outPtr);
                    }
                    else
                    {
                        IProductWRTBase3DKernel<ExecSpace, TData, true, false,
                                                true>(
                            shapeType, nm0, nm1, nm2, nq0, nq1, nq2,
                            nElmts + nPadElmts, correct, basis0, basis1, basis2,
                            w0, w1, w2, jacPtr, wspPtr, inPtr, outPtr, lambda);
                    }
                }
                else
                {
                    if (lambda == 1.0)
                    {
                        IProductWRTBase3DKernel<ExecSpace, TData, false, false,
                                                false>(
                            shapeType, nm0, nm1, nm2, nq0, nq1, nq2,
                            nElmts + nPadElmts, correct, basis0, basis1, basis2,
                            w0, w1, w2, jacPtr, wspPtr, inPtr, outPtr);
                    }
                    else
                    {
                        IProductWRTBase3DKernel<ExecSpace, TData, true, false,
                                                false>(
                            shapeType, nm0, nm1, nm2, nq0, nq1, nq2,
                            nElmts + nPadElmts, correct, basis0, basis1, basis2,
                            w0, w1, w2, jacPtr, wspPtr, inPtr, outPtr, lambda);
                    }
                }
            }

            // Increment pointer and index for next element type.
            jacPtr += deformed ? nqTot * nElmts : nElmts;
            inPtr += nqTot * (nElmts + nPadElmts);
            outPtr += nmTot * (nElmts + nPadElmts);
            exp_idx += nElmts;
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorIProductWRTBaseImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

private:
    BasisDataMap<TData> m_basisMap;
    BasisDataMap<TData> m_weightMap;
    MemoryRegion<TData> m_jac;
    MemoryRegion<TData> m_wsp;
    size_t m_wspsize = 0;
};

} // namespace Nektar::Operators::detail
