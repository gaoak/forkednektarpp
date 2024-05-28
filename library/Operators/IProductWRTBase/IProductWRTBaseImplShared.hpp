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

#include "Operators/OperatorIProductWRTBase.hpp"

#include "Operators/IProductWRTBase/IProductWRTBaseCUDASumFacKernels.cuh"
#include "Operators/MemoryRegion.hpp"
#include "Operators/OperatorHelper.hpp"

namespace Nektar::Operators::detail
{

// Shared implementation
template <typename ExecSpace, typename Implementation, typename TData,
          typename = typename std::enable_if<
              std::is_same<ExecSpace, NektarSpaces::CUDA>::value &&
              std::is_same<Implementation, Operators::SumFac>::value>::type>
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
        const TData *inptr = in.template GetConstPtr<MemSpace>();
        TData *outptr      = out.template GetPtr<MemSpace>();

        const TData *jacptr = m_jac.template GetConstPtr<MemSpace>();

        TData *wspptr = nullptr;

        // Initialize index.
        size_t expIdx = 0;

        // Initialize basiskey.
        std::vector<LibUtilities::BasisKey> basisKeys(
            3, LibUtilities::NullBasisKey);

        // Loop over the blocks.
        for (auto const &block : in.GetBlocks())
        {
            // Determine shape and type of the element.
            auto nElmts = block.num_elements;

            auto const expPtr = this->m_expansionList->GetExp(expIdx);
            auto nqTot        = expPtr->GetTotPoints();
            auto nmTot        = expPtr->GetNcoeffs();
            auto ptsKeys      = expPtr->GetPointsKeys();
            auto deformed     = expPtr->GetMetricInfo()->GetGtype() ==
                            SpatialDomains::eDeformed;

            // Deterime CUDA grid size.
            m_gridSize = GetCUDAGridSize(nElmts, m_blockSize);

            // Flag for collapsed coordinate correction.
            bool correct = expPtr->GetBasis(0)->GetBasisType() ==
                           LibUtilities::eModified_A;

            // Fetch basis key for the current element type.
            for (size_t d = 0; d < expPtr->GetShapeDimension(); d++)
            {
                basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
            }

            // Function call to kernel functions.
            if (expPtr->GetShapeDimension() == 1)
            {
                auto basis0 =
                    m_basisMap[basisKeys[0]].template GetConstPtr<MemSpace>();
                auto w0 =
                    m_weightMap[basisKeys[0]].template GetConstPtr<MemSpace>();

                auto nm0 = expPtr->GetBasisNumModes(0);
                auto nq0 = expPtr->GetNumPoints(0);

                if (deformed)
                {
                    if (lambda == 1.0)
                    {
                        IProductWRTBase1DKernel<ExecSpace, TData, false, false,
                                                true>(
                            m_gridSize, m_blockSize, nm0, nq0, nElmts, basis0,
                            w0, jacptr, inptr, outptr);
                    }
                    else
                    {
                        IProductWRTBase1DKernel<ExecSpace, TData, true, false,
                                                true>(
                            m_gridSize, m_blockSize, nm0, nq0, nElmts, basis0,
                            w0, jacptr, inptr, outptr, lambda);
                    }
                }
                else
                {
                    if (lambda == 1.0)
                    {
                        IProductWRTBase1DKernel<ExecSpace, TData, false, false,
                                                false>(
                            m_gridSize, m_blockSize, nm0, nq0, nElmts, basis0,
                            w0, jacptr, inptr, outptr);
                    }
                    else
                    {
                        IProductWRTBase1DKernel<ExecSpace, TData, true, false,
                                                false>(
                            m_gridSize, m_blockSize, nm0, nq0, nElmts, basis0,
                            w0, jacptr, inptr, outptr, lambda);
                    }
                }
            }
            else if (expPtr->GetShapeDimension() == 2)
            {
                auto basis0 =
                    m_basisMap[basisKeys[0]].template GetConstPtr<MemSpace>();
                auto basis1 =
                    m_basisMap[basisKeys[1]].template GetConstPtr<MemSpace>();
                auto w0 =
                    m_weightMap[basisKeys[0]].template GetConstPtr<MemSpace>();
                auto w1 =
                    m_weightMap[basisKeys[1]].template GetConstPtr<MemSpace>();

                auto shape = expPtr->DetShapeType();
                auto nm0   = expPtr->GetBasisNumModes(0);
                auto nm1   = expPtr->GetBasisNumModes(1);
                auto nq0   = expPtr->GetNumPoints(0);
                auto nq1   = expPtr->GetNumPoints(1);

                if constexpr (!FLAG_QP)
                {
                    size_t wspsize = 0;

                    if (shape == LibUtilities::Quad)
                    {
                        wspsize = nq1 * nElmts;
                    }
                    else if (shape == LibUtilities::Tri)
                    {
                        wspsize = nq0 * nElmts;
                    }

                    if (m_wspsize < wspsize)
                    {
                        m_wspsize = wspsize;
                        m_wsp = MemoryRegion<TData>::template create<MemSpace>(
                            m_wspsize,
                            EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());
                    }

                    wspptr = m_wsp.template GetPtr<MemSpace>();
                }

                if (deformed)
                {
                    if (lambda == 1.0)
                    {
                        IProductWRTBase2DKernel<ExecSpace, TData, false, false,
                                                true>(
                            m_gridSize, m_blockSize, shape, nm0, nm1, nq0, nq1,
                            nElmts, correct, basis0, basis1, w0, w1, jacptr,
                            wspptr, inptr, outptr);
                    }
                    else
                    {
                        IProductWRTBase2DKernel<ExecSpace, TData, true, false,
                                                true>(
                            m_gridSize, m_blockSize, shape, nm0, nm1, nq0, nq1,
                            nElmts, correct, basis0, basis1, w0, w1, jacptr,
                            wspptr, inptr, outptr, lambda);
                    }
                }
                else
                {
                    if (lambda == 1.0)
                    {
                        IProductWRTBase2DKernel<ExecSpace, TData, false, false,
                                                false>(
                            m_gridSize, m_blockSize, shape, nm0, nm1, nq0, nq1,
                            nElmts, correct, basis0, basis1, w0, w1, jacptr,
                            wspptr, inptr, outptr);
                    }
                    else
                    {
                        IProductWRTBase2DKernel<ExecSpace, TData, true, false,
                                                false>(
                            m_gridSize, m_blockSize, shape, nm0, nm1, nq0, nq1,
                            nElmts, correct, basis0, basis1, w0, w1, jacptr,
                            wspptr, inptr, outptr, lambda);
                    }
                }
            }
            else if (expPtr->GetShapeDimension() == 3)
            {
                auto basis0 =
                    m_basisMap[basisKeys[0]].template GetConstPtr<MemSpace>();
                auto basis1 =
                    m_basisMap[basisKeys[1]].template GetConstPtr<MemSpace>();
                auto basis2 =
                    m_basisMap[basisKeys[2]].template GetConstPtr<MemSpace>();
                auto w0 =
                    m_weightMap[basisKeys[0]].template GetConstPtr<MemSpace>();
                auto w1 =
                    m_weightMap[basisKeys[1]].template GetConstPtr<MemSpace>();
                auto w2 =
                    m_weightMap[basisKeys[2]].template GetConstPtr<MemSpace>();

                auto shape = expPtr->DetShapeType();
                auto nm0   = expPtr->GetBasisNumModes(0);
                auto nm1   = expPtr->GetBasisNumModes(1);
                auto nm2   = expPtr->GetBasisNumModes(2);
                auto nq0   = expPtr->GetNumPoints(0);
                auto nq1   = expPtr->GetNumPoints(1);
                auto nq2   = expPtr->GetNumPoints(2);

                if constexpr (!FLAG_QP)
                {
                    size_t wspsize = 0;

                    if (shape == LibUtilities::Hex)
                    {
                        wspsize = (nq2 * nq1 + nq2) * nElmts;
                    }
                    else if (shape == LibUtilities::Tet)
                    {
                        wspsize = (nq2 * nq1 + nq2 + nm2) * nElmts;
                    }
                    else if (shape == LibUtilities::Prism)
                    {
                        wspsize = (nq2 * nq1 + nq2 + nm1) * nElmts;
                    }
                    else if (shape == LibUtilities::Pyr)
                    {
                        wspsize = (nq2 * nq1 + nq2) * nElmts;
                    }

                    if (m_wspsize < wspsize)
                    {
                        m_wspsize = wspsize;
                        m_wsp = MemoryRegion<TData>::template create<MemSpace>(
                            m_wspsize,
                            EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());
                    }

                    wspptr = m_wsp.template GetPtr<MemSpace>();
                }

                if (deformed)
                {
                    if (lambda == 1.0)
                    {
                        IProductWRTBase3DKernel<ExecSpace, TData, false, false,
                                                true>(
                            m_gridSize, m_blockSize, shape, nm0, nm1, nm2, nq0,
                            nq1, nq2, nElmts, correct, basis0, basis1, basis2,
                            w0, w1, w2, jacptr, wspptr, inptr, outptr);
                    }
                    else
                    {
                        IProductWRTBase3DKernel<ExecSpace, TData, true, false,
                                                true>(
                            m_gridSize, m_blockSize, shape, nm0, nm1, nm2, nq0,
                            nq1, nq2, nElmts, correct, basis0, basis1, basis2,
                            w0, w1, w2, jacptr, wspptr, inptr, outptr, lambda);
                    }
                }
                else
                {
                    if (lambda == 1.0)
                    {
                        IProductWRTBase3DKernel<ExecSpace, TData, false, false,
                                                false>(
                            m_gridSize, m_blockSize, shape, nm0, nm1, nm2, nq0,
                            nq1, nq2, nElmts, correct, basis0, basis1, basis2,
                            w0, w1, w2, jacptr, wspptr, inptr, outptr);
                    }
                    else
                    {
                        IProductWRTBase3DKernel<ExecSpace, TData, true, false,
                                                false>(
                            m_gridSize, m_blockSize, shape, nm0, nm1, nm2, nq0,
                            nq1, nq2, nElmts, correct, basis0, basis1, basis2,
                            w0, w1, w2, jacptr, wspptr, inptr, outptr, lambda);
                    }
                }
            }

            // Increment pointer and index for next element type.
            jacptr += deformed ? nqTot * nElmts : nElmts;
            inptr += nqTot * nElmts;
            outptr += nmTot * nElmts;
            expIdx += nElmts;
        }
    }

    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorIProductWRTBaseImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

    static std::string className;

private:
    BasisDataMap<TData> m_basisMap;
    BasisDataMap<TData> m_weightMap;

    MemoryRegion<TData> m_jac;
    MemoryRegion<TData> m_wsp;

    size_t m_wspsize   = 0;
    size_t m_gridSize  = 1024;
    size_t m_blockSize = 32;
};

} // namespace Nektar::Operators::detail
