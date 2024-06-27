///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseImplShared.hpp
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

#include "Operators/OperatorIProductWRTDerivBase.hpp"

#include "Operators/MemoryRegion.hpp"
#include "Operators/OperatorHelper.hpp"

#include "Operators/IProductWRTBase/IProductWRTBaseCUDASumFacKernels.cuh"
#include "Operators/IProductWRTDerivBase/IProductWRTDerivBaseCUDASumFacKernels.cuh"

namespace Nektar::Operators::detail
{

// Shared implementation
template <typename ExecSpace, typename Implementation, typename TData,
          typename = typename std::enable_if<
#if defined(NEKTAR_ENABLE_CUDA)
              (std::is_same<ExecSpace, NektarSpaces::CUDA>::value &&
               std::is_same<Implementation, Operators::SumFac>::value) ||
#endif
              (std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value &&
               std::is_same<Implementation, Operators::StdMat>::value)>::type>
class OperatorIProductWRTDerivBaseImpl
    : public OperatorIProductWRTDerivBase<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorIProductWRTDerivBaseImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorIProductWRTDerivBase<TData>(expansionList)
    {
        size_t nDim   = this->m_expansionList->GetShapeDimension();
        size_t nCoord = this->m_expansionList->GetCoordim(0);

        // Initialise the jacobian and the derivative factor.
        m_dfSize = Operator<TData>::GetGeometricFactorSize();

        auto jac      = Operator<TData>::SetJacobian(m_dfSize);
        auto derivFac = Operator<TData>::SetDerivativeFactor(m_dfSize);

        // Initialise the jacobian.
        m_jac = MemoryRegion<TData>::template fromArray<MemSpace, TData>(
            jac, EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());

        // Initialise the derivative factor.
        m_derivFac = MemoryRegion<TData>::template fromArray<MemSpace, TData>(
            derivFac, EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());

        // Initialize the basis data.
        m_basisMap =
            GetBasisData<MemSpace, TData>(expansionList, BASIS_BASIS_DATA);
        m_dbasisMap = GetBasisData<MemSpace, TData>(
            expansionList, BASIS_BASIS_DERIVATIVE_DATA);
        m_weightMap =
            GetBasisData<MemSpace, TData>(expansionList, BASIS_WEIGHT_DATA);
        m_pointMap =
            GetBasisData<MemSpace, TData>(expansionList, BASIS_POINT_DATA);
        m_derivativeMap =
            GetBasisData<MemSpace, TData>(expansionList, BASIS_DERIVATIVE_DATA);
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out,
               bool APPEND = false) override
    {
        // Zero output.
        if (!APPEND)
        {
            out.initialize(0);
        }

        // Copy memory to the device, if necessary and get raw pointers.
        const TData *inPtr = in.template GetConstPtr<MemSpace>();
        TData *outPtr      = out.template GetPtr<MemSpace>();

        const TData *jacPtr = m_jac.template GetConstPtr<MemSpace>();
        const TData *dfPtr  = m_derivFac.template GetConstPtr<MemSpace>();

        // Initialize the workspace memory.
        auto nSize = in.GetFieldSize();
        if (m_tmpsize < nSize)
        {
            m_tmpsize = nSize;
            m_tmp     = MemoryRegion<TData>::template create<MemSpace>(
                nSize * this->m_expansionList->GetExp(0)->GetCoordim());
        }

        TData *tmpPtr = m_tmp.template GetPtr<MemSpace>();
        TData *wspPtr = nullptr;

        // Initialize basiskey.
        std::vector<LibUtilities::BasisKey> basisKeys(
            3, LibUtilities::NullBasisKey);

        // Initialize index.
        size_t exp_idx = 0;

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
            auto const nqTot  = expPtr->GetTotPoints();
            auto const nmTot  = expPtr->GetNcoeffs();
            auto const nCoord = expPtr->GetCoordim();

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
                auto dbasis0 =
                    m_dbasisMap[basisKeys[0]].template GetConstPtr<MemSpace>();
                auto w0 =
                    m_weightMap[basisKeys[0]].template GetConstPtr<MemSpace>();
                auto D0 = m_derivativeMap[basisKeys[0]]
                              .template GetConstPtr<MemSpace>();

                auto nm0 = expPtr->GetBasisNumModes(0);
                auto nq0 = expPtr->GetNumPoints(0);

                if (deformed)
                {
                    IProductWRTDerivBase1DKernel<ExecSpace, TData, true>(
                        nq0, nCoord, nElmts, nSize, m_dfSize, dfPtr, inPtr,
                        tmpPtr);
                    IProductWRTBase1DKernel<ExecSpace, TData, false, true,
                                            true>(nm0, nq0, nElmts + nPadElmts,
                                                  dbasis0, w0, jacPtr, tmpPtr,
                                                  outPtr);
                }
                else
                {
                    IProductWRTDerivBase1DKernel<ExecSpace, TData, false>(
                        nq0, nCoord, nElmts, nSize, m_dfSize, dfPtr, inPtr,
                        tmpPtr);
                    IProductWRTBase1DKernel<ExecSpace, TData, false, true,
                                            false>(nm0, nq0, nElmts + nPadElmts,
                                                   dbasis0, w0, jacPtr, tmpPtr,
                                                   outPtr);
                }
            }
            else if (dimension == 2)
            {
                auto basis0 =
                    m_basisMap[basisKeys[0]].template GetConstPtr<MemSpace>();
                auto basis1 =
                    m_basisMap[basisKeys[1]].template GetConstPtr<MemSpace>();
                auto dbasis0 =
                    m_dbasisMap[basisKeys[0]].template GetConstPtr<MemSpace>();
                auto dbasis1 =
                    m_dbasisMap[basisKeys[1]].template GetConstPtr<MemSpace>();
                auto w0 =
                    m_weightMap[basisKeys[0]].template GetConstPtr<MemSpace>();
                auto w1 =
                    m_weightMap[basisKeys[1]].template GetConstPtr<MemSpace>();
                auto D0 = m_derivativeMap[basisKeys[0]]
                              .template GetConstPtr<MemSpace>();
                auto D1 = m_derivativeMap[basisKeys[1]]
                              .template GetConstPtr<MemSpace>();
                auto Z0 =
                    m_pointMap[basisKeys[0]].template GetConstPtr<MemSpace>();
                auto Z1 =
                    m_pointMap[basisKeys[1]].template GetConstPtr<MemSpace>();

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

                    wspPtr = m_wsp.template GetPtr<MemSpace>();
                }
                if (deformed)
                {
                    IProductWRTDerivBase2DKernel<ExecSpace, TData, true>(
                        shapeType, nq0, nq1, nCoord, nElmts, nSize, m_dfSize,
                        Z0, Z1, dfPtr, inPtr, tmpPtr);
                    IProductWRTBase2DKernel<ExecSpace, TData, false, true,
                                            true>(
                        shapeType, nm0, nm1, nq0, nq1, nElmts + nPadElmts,
                        correct, dbasis0, basis1, w0, w1, jacPtr, wspPtr,
                        tmpPtr, outPtr);
                    IProductWRTBase2DKernel<ExecSpace, TData, false, true,
                                            true>(
                        shapeType, nm0, nm1, nq0, nq1, nElmts + nPadElmts,
                        correct, basis0, dbasis1, w0, w1, jacPtr, wspPtr,
                        tmpPtr + nSize, outPtr);
                }
                else
                {
                    IProductWRTDerivBase2DKernel<ExecSpace, TData, false>(
                        shapeType, nq0, nq1, nCoord, nElmts, nSize, m_dfSize,
                        Z0, Z1, dfPtr, inPtr, tmpPtr);
                    IProductWRTBase2DKernel<ExecSpace, TData, false, true,
                                            false>(
                        shapeType, nm0, nm1, nq0, nq1, nElmts + nPadElmts,
                        correct, dbasis0, basis1, w0, w1, jacPtr, wspPtr,
                        tmpPtr, outPtr);
                    IProductWRTBase2DKernel<ExecSpace, TData, false, true,
                                            false>(
                        shapeType, nm0, nm1, nq0, nq1, nElmts + nPadElmts,
                        correct, basis0, dbasis1, w0, w1, jacPtr, wspPtr,
                        tmpPtr + nSize, outPtr);
                }
            }
            else if (dimension == 3)
            {
                auto basis0 =
                    m_basisMap[basisKeys[0]].template GetConstPtr<MemSpace>();
                auto basis1 =
                    m_basisMap[basisKeys[1]].template GetConstPtr<MemSpace>();
                auto basis2 =
                    m_basisMap[basisKeys[2]].template GetConstPtr<MemSpace>();
                auto dbasis0 =
                    m_dbasisMap[basisKeys[0]].template GetConstPtr<MemSpace>();
                auto dbasis1 =
                    m_dbasisMap[basisKeys[1]].template GetConstPtr<MemSpace>();
                auto dbasis2 =
                    m_dbasisMap[basisKeys[2]].template GetConstPtr<MemSpace>();
                auto w0 =
                    m_weightMap[basisKeys[0]].template GetConstPtr<MemSpace>();
                auto w1 =
                    m_weightMap[basisKeys[1]].template GetConstPtr<MemSpace>();
                auto w2 =
                    m_weightMap[basisKeys[2]].template GetConstPtr<MemSpace>();
                auto D0 = m_derivativeMap[basisKeys[0]]
                              .template GetConstPtr<MemSpace>();
                auto D1 = m_derivativeMap[basisKeys[1]]
                              .template GetConstPtr<MemSpace>();
                auto D2 = m_derivativeMap[basisKeys[2]]
                              .template GetConstPtr<MemSpace>();
                auto Z0 =
                    m_pointMap[basisKeys[0]].template GetConstPtr<MemSpace>();
                auto Z1 =
                    m_pointMap[basisKeys[1]].template GetConstPtr<MemSpace>();
                auto Z2 =
                    m_pointMap[basisKeys[2]].template GetConstPtr<MemSpace>();

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

                    wspPtr = m_wsp.template GetPtr<MemSpace>();
                }

                if (deformed)
                {
                    IProductWRTDerivBase3DKernel<ExecSpace, TData, true>(
                        shapeType, nq0, nq1, nq2, nCoord, nElmts, nSize,
                        m_dfSize, Z0, Z1, Z2, dfPtr, inPtr, tmpPtr);
                    IProductWRTBase3DKernel<ExecSpace, TData, false, true,
                                            true>(
                        shapeType, nm0, nm1, nm2, nq0, nq1, nq2,
                        nElmts + nPadElmts, correct, dbasis0, basis1, basis2,
                        w0, w1, w2, jacPtr, wspPtr, tmpPtr, outPtr);
                    IProductWRTBase3DKernel<ExecSpace, TData, false, true,
                                            true>(
                        shapeType, nm0, nm1, nm2, nq0, nq1, nq2,
                        nElmts + nPadElmts, correct, basis0, dbasis1, basis2,
                        w0, w1, w2, jacPtr, wspPtr, tmpPtr + nSize, outPtr);
                    IProductWRTBase3DKernel<ExecSpace, TData, false, true,
                                            true>(
                        shapeType, nm0, nm1, nm2, nq0, nq1, nq2,
                        nElmts + nPadElmts, correct, basis0, basis1, dbasis2,
                        w0, w1, w2, jacPtr, wspPtr, tmpPtr + 2 * nSize, outPtr);
                }
                else
                {
                    IProductWRTDerivBase3DKernel<ExecSpace, TData, false>(
                        shapeType, nq0, nq1, nq2, nCoord, nElmts, nSize,
                        m_dfSize, Z0, Z1, Z2, dfPtr, inPtr, tmpPtr);
                    IProductWRTBase3DKernel<ExecSpace, TData, false, true,
                                            false>(
                        shapeType, nm0, nm1, nm2, nq0, nq1, nq2,
                        nElmts + nPadElmts, correct, dbasis0, basis1, basis2,
                        w0, w1, w2, jacPtr, wspPtr, tmpPtr, outPtr);
                    IProductWRTBase3DKernel<ExecSpace, TData, false, true,
                                            false>(
                        shapeType, nm0, nm1, nm2, nq0, nq1, nq2,
                        nElmts + nPadElmts, correct, basis0, dbasis1, basis2,
                        w0, w1, w2, jacPtr, wspPtr, tmpPtr + nSize, outPtr);
                    IProductWRTBase3DKernel<ExecSpace, TData, false, true,
                                            false>(
                        shapeType, nm0, nm1, nm2, nq0, nq1, nq2,
                        nElmts + nPadElmts, correct, basis0, basis1, dbasis2,
                        w0, w1, w2, jacPtr, wspPtr, tmpPtr + 2 * nSize, outPtr);
                }
            }

            // Increment pointer and index for next element type.
            jacPtr += deformed ? nqTot * nElmts : nElmts;
            dfPtr += deformed ? nqTot * nElmts : nElmts;

            tmpPtr += nqTot * (nElmts + nPadElmts);
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
            OperatorIProductWRTDerivBaseImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

private:
    BasisDataMap<TData> m_basisMap;
    BasisDataMap<TData> m_dbasisMap;
    BasisDataMap<TData> m_weightMap;
    BasisDataMap<TData> m_derivativeMap;
    BasisDataMap<TData> m_pointMap;

    MemoryRegion<TData> m_jac;
    MemoryRegion<TData> m_derivFac;
    MemoryRegion<TData> m_wsp;
    MemoryRegion<TData> m_tmp;

    size_t m_dfSize;
    size_t m_wspsize = 0;
    size_t m_tmpsize = 0;
};

} // namespace Nektar::Operators::detail
