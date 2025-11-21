///////////////////////////////////////////////////////////////////////////////
//
// File: HelmholtzSerialAVXSumFac.hpp
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

#include "Operators/ElmtOps/Helmholtz/HelmholtzOp.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/Helmholtz/HelmholtzSerialAVXSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class HelmholtzBlockOpImpl : public HelmholtzBlockOp<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    HelmholtzBlockOpImpl(const LocalRegions::ExpansionSharedPtr &exp,
                         NekDataWarehouseSharedPtr dataWarehouse)
        : HelmholtzBlockOp<TData>(exp, dataWarehouse)
    {
        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
        m_dimension = exp->GetShapeDimension();
        m_coordDim  = exp->GetCoordim();

        // Flag for collapsed coordinate correction.
        m_isModified = (exp->GetBasisType(0) == LibUtilities::eModified_A);

        for (unsigned int d = 0; d < m_dimension; d++)
        {
            // Fetch element size.
            m_nm.push_back(exp->GetBasisNumModes(d));
            m_nq.push_back(exp->GetNumPoints(d));

            // Fetch basis data.
            m_B.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(exp->GetBasis(d)->GetBasisKey(), eBasis)));
            m_DB.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(exp->GetBasis(d)->GetBasisKey(),
                                     eBasisDerivative)));
            m_D.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(exp->GetBasis(d)->GetBasisKey(),
                                     eDerivative)));
            m_W.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(exp->GetBasis(d)->GetBasisKey(),
                                     eWeights)));
        }

        if (m_dimension == 2)
        {
            // Fetch geometric factors.
            m_f.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(exp->GetBasis(0)->GetBasisKey(),
                                     eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(exp->GetBasis(1)->GetBasisKey(),
                                     eTwoOverOneMinusZero)));
        }
        else if (m_dimension == 3)
        {
            // Fetch geometric factors.
            m_f.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(exp->GetBasis(0)->GetBasisKey(),
                                     eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(exp->GetBasis(1)->GetBasisKey(),
                                     eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(exp->GetBasis(1)->GetBasisKey(),
                                     eTwoOverOneMinusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<ExecSpace>(
                BasisDataKey<simd_t>(exp->GetBasis(2)->GetBasisKey(),
                                     eTwoOverOneMinusZero)));
        }

        // Set diffusion coefficient.
        m_diffCoeff =
            std::vector<TData>(m_coordDim * (m_coordDim + 1) / 2, 0.0);

        // Set up temprary solution.
        m_diffCoeff[0] = 1.0; // m_D[0]0
        if (m_coordDim >= 2)
        {
            m_diffCoeff[2] = 1.0; // m_D[1]1
            if (m_coordDim == 3)
            {
                m_diffCoeff[5] = 1.0; // m_D[2]2
            }
        }

        if ((m_shapeType == LibUtilities::eNodalTri) ||
            (m_shapeType == LibUtilities::eNodalTet) ||
            (m_shapeType == LibUtilities::eNodalPrism))
        {
            // Fetch NodalToModal Matrix if required.
            m_nodToMod = this->m_dataWarehouse->template GetData<ExecSpace>(
                VandemondeKey<simd_t>(eNodalToModal, exp->GetElmtId()));

            // Fetch NodalToModal Matrix if required.
            m_nodToModTrans =
                this->m_dataWarehouse->template GetData<ExecSpace>(
                    VandemondeKey<simd_t>(eNodalToModalTranspose,
                                          exp->GetElmtId()));
        }
        else
        {
            m_nodToMod      = (const simd_t *)nullptr;
            m_nodToModTrans = (const simd_t *)nullptr;
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
            HelmholtzBlockOpImpl<ExecSpace, Implementation, TData>>(
            exp, dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

    LibUtilities::ShapeType m_shapeType;
    bool m_isDeformed;
    bool m_isModified;
    unsigned int m_dimension;
    unsigned int m_coordDim;
    std::vector<unsigned int> m_nm;
    std::vector<unsigned int> m_nq;
    std::vector<const simd_t *> m_B;
    std::vector<const simd_t *> m_DB;
    std::vector<const simd_t *> m_D;
    std::vector<const simd_t *> m_W;
    std::vector<const simd_t *> m_f;
    std::vector<TData> m_diffCoeff;
    std::vector<TData> NullTDataVector;
    const simd_t *m_nodToMod;
    const simd_t *m_nodToModTrans;
#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
    // flag to ensure we only get one warning for alignment otherwise CI system
    // is saturated with warnings
    bool m_warnOnce = false;
#endif

    void v_Apply(BlockAccessor<TData> &inblock,
                 BlockAccessor<TData> &outblock) override
    {
        WARNINGL1(
            m_warnOnce || (inblock.GetAlignment() == simd_t::alignment &&
                           outblock.GetAlignment() == simd_t::alignment),
            "Input or output Field are not aligned to the required alignment "
            "for the SIMD vector type.");
#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
        m_warnOnce = true;
#endif
        switch (m_shapeType)
        {
            // Segment
            case LibUtilities::Seg:
            {
                SegBlock(inblock, outblock);
                break;
            }
            // Quads
            case LibUtilities::Quad:
            {
                QuadBlock(inblock, outblock);
                break;
            }
            // Triangles
            case LibUtilities::Tri:
            {
                TriBlock(inblock, outblock);
                break;
            }
            // Nodal Triangles
            case LibUtilities::NodalTri:
            {
                NodalTriBlock(inblock, outblock);
                break;
            }
            // Hexes
            case LibUtilities::Hex:
            {
                HexBlock(inblock, outblock);
                break;
            }
            // Tet
            case LibUtilities::Tet:
            {
                TetBlock(inblock, outblock);
                break;
            }
            // NodalTet
            case LibUtilities::NodalTet:
            {
                NodalTetBlock(inblock, outblock);
                break;
            }
            // Pyr
            case LibUtilities::Pyr:
            {
                PyrBlock(inblock, outblock);
                break;
            }
            // Prism
            case LibUtilities::Prism:
            {
                PrismBlock(inblock, outblock);
                break;
            }
            // NodalPrism
            case LibUtilities::NodalPrism:
            {
                NodalPrismBlock(inblock, outblock);
                break;
            }
            default:
                std::cout << "shapetype not implemented" << std::endl;
        }
    }

    void v_SetLambda(const TData &lambda) override
    {
        this->m_lambda = lambda;
    }

    void SegBlock(BlockAccessor<TData> &inblock,
                  BlockAccessor<TData> &outblock);

    void TriBlock(BlockAccessor<TData> &inblock,
                  BlockAccessor<TData> &outblock);

    void NodalTriBlock(BlockAccessor<TData> &inblock,
                       BlockAccessor<TData> &outblock);

    void QuadBlock(BlockAccessor<TData> &inblock,
                   BlockAccessor<TData> &outblock);

    void HexBlock(BlockAccessor<TData> &inblock,
                  BlockAccessor<TData> &outblock);

    void PrismBlock(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock);

    void NodalPrismBlock(BlockAccessor<TData> &inblock,
                         BlockAccessor<TData> &outblock);

    void PyrBlock(BlockAccessor<TData> &inblock,
                  BlockAccessor<TData> &outblock);

    void TetBlock(BlockAccessor<TData> &inblock,
                  BlockAccessor<TData> &outblock);

    void NodalTetBlock(BlockAccessor<TData> &inblock,
                       BlockAccessor<TData> &outblock);

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator1D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        const auto nm0 = m_nm[0];
        const auto nq0 = m_nq[0];

        const auto nmTot = nm0;
        const auto nqTot = nq0;

        unsigned int ndf    = m_coordDim;
        unsigned int dfSize = 1;
        if constexpr (DEFORMED)
        {
            dfSize *= nqTot;
        }

        // Fetch Jacobian and deriv factors.
        auto jacptr_init = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));
        auto dfptr_init = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), false));

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Allocate workspace.
        std::vector<simd_t, tinysimd::allocator<simd_t>> bwd(nqTot);
        std::vector<simd_t, tinysimd::allocator<simd_t>> deriv0(nqTot);

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
            auto dfptr  = dfptr_init;

            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                              interleaveWidth, chunkSize, nmTot,
                                              (TData *)inptr);
                }

                // Step 1: BwdTrans.
                BwdTrans1DKernel<SHAPE_TYPE>(
                    nm0, nq0, m_B[0], reinterpret_cast<const simd_t *>(inptr),
                    bwd.data());

                // Step 2: Inner product for mass matrix operation.
                IProduct1DKernel<SHAPE_TYPE, true, false, DEFORMED>(
                    nm0, nq0, bwd.data(), m_B[0], m_W[0],
                    reinterpret_cast<const simd_t *>(jacptr),
                    reinterpret_cast<simd_t *>(outptr), this->m_lambda);

                // Step 3: Take derivatives in collapsed coordinate space.
                PhysDerivTensor1DKernel(nq0, bwd.data(), m_D[0], deriv0.data());

                // Step 4: Apply diffusion coefficiets.
                DiffusionCoeffSegKernel<DEFORMED, simd_t>(
                    m_coordDim, nq0, true, this->m_diffCoeff, false,
                    NullTDataVector, NullTDataVector, NullTDataVector,
                    NullTDataVector, NullTDataVector, NullTDataVector,
                    reinterpret_cast<const simd_t *>(dfptr), deriv0.data());

                // Step 5: Apply Laplacian metrics & inner product.
                IProduct1DKernel<SHAPE_TYPE, false, true, DEFORMED>(
                    nm0, nq0, deriv0.data(), m_DB[0], m_W[0],
                    reinterpret_cast<const simd_t *>(jacptr),
                    reinterpret_cast<simd_t *>(outptr));

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        nmTot,
                        (TData *)inptr -
                            (width_ratio - 1) * nmTot * simd_t::width);
                    ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        nmTot,
                        (TData *)outptr -
                            (width_ratio - 1) * nmTot * simd_t::width);
                }

                // Increment pointers.
                dfptr += dfSize * ndf * simd_t::width;
                jacptr += dfSize * simd_t::width;
                inptr += nmTot * simd_t::width;
                outptr += nmTot * simd_t::width;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nm0, unsigned int nq0>
    void Operator1D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        constexpr auto nmTot = nm0;
        constexpr auto nqTot = nq0;

        unsigned int ndf    = m_coordDim;
        unsigned int dfSize = 1;
        if constexpr (DEFORMED)
        {
            dfSize *= nqTot;
        }

        // Fetch Jacobian and deriv factors.
        auto jacptr_init = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));
        auto dfptr_init = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), false));

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Allocate workspace.
        std::vector<simd_t, tinysimd::allocator<simd_t>> bwd(nqTot);
        std::vector<simd_t, tinysimd::allocator<simd_t>> deriv0(nqTot);

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
            auto dfptr  = dfptr_init;

            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                              interleaveWidth, chunkSize, nmTot,
                                              (TData *)inptr);
                }

                // Step 1: BwdTrans.
                BwdTrans1DKernel<SHAPE_TYPE>(
                    nm0, nq0, m_B[0], reinterpret_cast<const simd_t *>(inptr),
                    bwd.data());

                // Step 2: Inner product for mass matrix operation.
                IProduct1DKernel<SHAPE_TYPE, true, false, DEFORMED>(
                    nm0, nq0, bwd.data(), m_B[0], m_W[0],
                    reinterpret_cast<const simd_t *>(jacptr),
                    reinterpret_cast<simd_t *>(outptr), this->m_lambda);

                // Step 3: Take derivatives in collapsed coordinate space.
                PhysDerivTensor1DKernel(nq0, bwd.data(), m_D[0], deriv0.data());

                // Step 4: Apply diffusion coefficiets.
                DiffusionCoeffSegKernel<DEFORMED, simd_t>(
                    m_coordDim, nq0, true, this->m_diffCoeff, false,
                    NullTDataVector, NullTDataVector, NullTDataVector,
                    NullTDataVector, NullTDataVector, NullTDataVector,
                    reinterpret_cast<const simd_t *>(dfptr), deriv0.data());

                // Step 5: Apply Laplacian metrics & inner product.
                IProduct1DKernel<SHAPE_TYPE, false, true, DEFORMED>(
                    nm0, nq0, deriv0.data(), m_DB[0], m_W[0],
                    reinterpret_cast<const simd_t *>(jacptr),
                    reinterpret_cast<simd_t *>(outptr));

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        nmTot,
                        (TData *)inptr -
                            (width_ratio - 1) * nmTot * simd_t::width);
                    ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        nmTot,
                        (TData *)outptr -
                            (width_ratio - 1) * nmTot * simd_t::width);
                }

                // Increment pointers.
                dfptr += dfSize * ndf * simd_t::width;
                jacptr += dfSize * simd_t::width;
                inptr += nmTot * simd_t::width;
                outptr += nmTot * simd_t::width;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator2D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        const auto nm0 = m_nm[0];
        const auto nm1 = m_nm[1];

        const auto nq0 = m_nq[0];
        const auto nq1 = m_nq[1];

        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
        const auto nqTot = nq0 * nq1;

        unsigned int ndf    = 2 * m_coordDim;
        unsigned int dfSize = 1;
        if constexpr (DEFORMED)
        {
            dfSize *= nqTot;
        }

        // Fetch Jacobian and deriv factors.
        auto jacptr_init = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));
        auto dfptr_init = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), false));

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Workspace for kernels - also checks preconditions.
        unsigned int wsp0Size = 0;
        BwdTrans2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);
        IProduct2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size);
        std::vector<simd_t, tinysimd::allocator<simd_t>> bwd(nqTot);
        std::vector<simd_t, tinysimd::allocator<simd_t>> deriv0(nqTot);
        std::vector<simd_t, tinysimd::allocator<simd_t>> deriv1(nqTot);

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
            auto dfptr  = dfptr_init;

            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                              interleaveWidth, chunkSize, nmTot,
                                              (TData *)inptr);
                }

                // Step 1: BwdTrans.
                BwdTrans2DKernel<SHAPE_TYPE>(
                    nm0, nm1, nq0, nq1, m_isModified, m_B[0], m_B[1],
                    m_nodToMod, wsp0.data(),
                    reinterpret_cast<const simd_t *>(inptr), bwd.data());

                // Step 2: Get tensor derivatives
                PhysDerivTensor2DKernel<simd_t>(nq0, nq1, bwd.data(), m_D[0],
                                                m_D[1], deriv0.data(),
                                                deriv1.data());

                // Step 3: apply diffusion coeff and WJ
                DiffusionCoeffwithWJ2DKernel<SHAPE_TYPE, DEFORMED, true,
                                             simd_t>(
                    m_coordDim, nq0, nq1, true, this->m_diffCoeff, false,
                    NullTDataVector, NullTDataVector, NullTDataVector,
                    NullTDataVector, NullTDataVector, NullTDataVector,
                    reinterpret_cast<const simd_t *>(jacptr), m_W[0], m_W[1],
                    reinterpret_cast<const simd_t *>(dfptr), m_f[0], m_f[1],
                    deriv0.data(), deriv1.data(), bwd.data(), this->m_lambda);

                // Step 4: apply derivative and sum up.
                SumDerivTensor2DKernel<simd_t>(nq0, nq1, deriv0.data(),
                                               deriv1.data(), m_D[0], m_D[1],
                                               bwd.data(), 1.0);

                // Step 5 : inner product without WJ.
                IProduct2DKernel<SHAPE_TYPE, false, false, simd_t>(
                    nm0, nm1, nq0, nq1, m_isModified, bwd.data(), m_B[0],
                    m_B[1], m_nodToModTrans, wsp0.data(),
                    reinterpret_cast<simd_t *>(outptr), 1.0);

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        nmTot,
                        (TData *)inptr -
                            (width_ratio - 1) * nmTot * simd_t::width);
                    ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        nmTot,
                        (TData *)outptr -
                            (width_ratio - 1) * nmTot * simd_t::width);
                }

                // Increment pointers.
                dfptr += dfSize * ndf * simd_t::width;
                jacptr += dfSize * simd_t::width;
                inptr += nmTot * simd_t::width;
                outptr += nmTot * simd_t::width;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nm0, unsigned int nm1, unsigned int nq0,
              unsigned int nq1>
    void Operator2D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        constexpr auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
        constexpr auto nqTot = nq0 * nq1;

        unsigned int ndf    = 2 * m_coordDim;
        unsigned int dfSize = 1;
        if constexpr (DEFORMED)
        {
            dfSize *= nqTot;
        }

        // Fetch Jacobian and deriv factors.
        auto jacptr_init = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));
        auto dfptr_init = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), false));

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Workspace for kernels - also checks preconditions.
        unsigned int wsp0Size = 0;
        BwdTrans2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);
        IProduct2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size);
        std::vector<simd_t, tinysimd::allocator<simd_t>> bwd(nqTot);
        std::vector<simd_t, tinysimd::allocator<simd_t>> deriv0(nqTot);
        std::vector<simd_t, tinysimd::allocator<simd_t>> deriv1(nqTot);

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
            auto dfptr  = dfptr_init;

            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                              interleaveWidth, chunkSize, nmTot,
                                              (TData *)inptr);
                }

                // Step 1: BwdTrans.
                BwdTrans2DKernel<SHAPE_TYPE>(
                    nm0, nm1, nq0, nq1, m_isModified, m_B[0], m_B[1],
                    m_nodToMod, wsp0.data(),
                    reinterpret_cast<const simd_t *>(inptr), bwd.data());

                // Step 2: Get tensor derivatives
                PhysDerivTensor2DKernel<simd_t>(nq0, nq1, bwd.data(), m_D[0],
                                                m_D[1], deriv0.data(),
                                                deriv1.data());

                // Step 3: apply diffusion coeff and WJ
                DiffusionCoeffwithWJ2DKernel<SHAPE_TYPE, DEFORMED, true,
                                             simd_t>(
                    m_coordDim, nq0, nq1, true, this->m_diffCoeff, false,
                    NullTDataVector, NullTDataVector, NullTDataVector,
                    NullTDataVector, NullTDataVector, NullTDataVector,
                    reinterpret_cast<const simd_t *>(jacptr), m_W[0], m_W[1],
                    reinterpret_cast<const simd_t *>(dfptr), m_f[0], m_f[1],
                    deriv0.data(), deriv1.data(), bwd.data(), this->m_lambda);

                // Step 4: apply derivative and sum up.
                SumDerivTensor2DKernel<simd_t>(nq0, nq1, deriv0.data(),
                                               deriv1.data(), m_D[0], m_D[1],
                                               bwd.data(), 1.0);

                // Step 5 : inner product without WJ.
                IProduct2DKernel<SHAPE_TYPE, false, false, simd_t>(
                    nm0, nm1, nq0, nq1, m_isModified, bwd.data(), m_B[0],
                    m_B[1], m_nodToModTrans, wsp0.data(),
                    reinterpret_cast<simd_t *>(outptr), 1.0);

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        nmTot,
                        (TData *)inptr -
                            (width_ratio - 1) * nmTot * simd_t::width);
                    ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        nmTot,
                        (TData *)outptr -
                            (width_ratio - 1) * nmTot * simd_t::width);
                }

                // Increment pointers.
                dfptr += dfSize * ndf * simd_t::width;
                jacptr += dfSize * simd_t::width;
                inptr += nmTot * simd_t::width;
                outptr += nmTot * simd_t::width;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator3D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        const auto nm0 = m_nm[0];
        const auto nm1 = m_nm[1];
        const auto nm2 = m_nm[2];

        const auto nq0 = m_nq[0];
        const auto nq1 = m_nq[1];
        const auto nq2 = m_nq[2];

        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
        const auto nqTot = nq0 * nq1 * nq2;

        constexpr unsigned int ndf = 9;
        unsigned int dfSize        = 1;
        if constexpr (DEFORMED)
        {
            dfSize *= nqTot;
        }

        // Fetch Jacobian and deriv factors.
        auto jacptr_init = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));
        auto dfptr_init = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), false));

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Workspace for kernels - also checks preconditions.
        unsigned int wsp0Size = 0, wsp1Size = 0, wsp2Size = 0;
        BwdTrans3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size);
        IProduct3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size, wsp2Size);

        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size),
            wsp1(wsp1Size), wsp2(wsp2Size);
        std::vector<simd_t, tinysimd::allocator<simd_t>> bwd(nqTot);
        std::vector<simd_t, tinysimd::allocator<simd_t>> deriv0(nqTot);
        std::vector<simd_t, tinysimd::allocator<simd_t>> deriv1(nqTot);
        std::vector<simd_t, tinysimd::allocator<simd_t>> deriv2(nqTot);

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();
        const auto width_ratio     = (interleaveWidth == 1)
                                         ? 1
                                         : interleaveWidth / m_implInterleaveWidth;
        const auto chunkSize = std::max(m_implInterleaveWidth, interleaveWidth);

        // Loop over components.
        for (unsigned int n = 0; n < inblock.GetNumComponents(); ++n)
        {
            auto jacptr = jacptr_init;
            auto dfptr  = dfptr_init;

            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                              interleaveWidth, chunkSize, nmTot,
                                              (TData *)inptr);
                }

                // Step 1: BwdTrans.
                BwdTrans3DKernel<SHAPE_TYPE>(
                    nm0, nm1, nm2, nq0, nq1, nq2, m_isModified, m_B[0], m_B[1],
                    m_B[2], m_nodToMod, wsp0.data(), wsp1.data(),
                    reinterpret_cast<const simd_t *>(inptr), bwd.data());

                // Step 2: Get tensor derivatives
                PhysDerivTensor3DKernel<simd_t>(
                    nq0, nq1, nq2, bwd.data(), m_D[0], m_D[1], m_D[2],
                    deriv0.data(), deriv1.data(), deriv2.data());

                // Step 2 + 3 : apply diffusion coeff and WJ
                DiffusionCoeffwithWJ3DKernel<SHAPE_TYPE, DEFORMED, true,
                                             simd_t>(
                    nq0, nq1, nq2, true, this->m_diffCoeff, false,
                    NullTDataVector, NullTDataVector, NullTDataVector,
                    NullTDataVector, NullTDataVector, NullTDataVector,
                    reinterpret_cast<const simd_t *>(jacptr), m_W[0], m_W[1],
                    m_W[2], reinterpret_cast<const simd_t *>(dfptr), m_f[0],
                    m_f[1], m_f[2], m_f[3], deriv0.data(), deriv1.data(),
                    deriv2.data(), bwd.data(), this->m_lambda);

                // Step 4: apply WJ, derivative and sum up.
                SumDerivTensor3DKernel<simd_t>(
                    nq0, nq1, nq2, deriv0.data(), deriv1.data(), deriv2.data(),
                    m_D[0], m_D[1], m_D[2], bwd.data(), 1.0);

                // Step 5 : inner product without WJ.
                IProduct3DKernel<SHAPE_TYPE, false, false, simd_t>(
                    nm0, nm1, nm2, nq0, nq1, nq2, m_isModified, bwd.data(),
                    m_B[0], m_B[1], m_B[2], m_nodToModTrans, wsp0.data(),
                    wsp1.data(), wsp2.data(),
                    reinterpret_cast<simd_t *>(outptr), 1.0);

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        nmTot,
                        (TData *)inptr -
                            (width_ratio - 1) * nmTot * simd_t::width);
                    ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        nmTot,
                        (TData *)outptr -
                            (width_ratio - 1) * nmTot * simd_t::width);
                }

                // Increment pointers.
                dfptr += dfSize * ndf * simd_t::width;
                jacptr += dfSize * simd_t::width;
                inptr += nmTot * simd_t::width;
                outptr += nmTot * simd_t::width;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nm0, unsigned int nm1, unsigned int nm2,
              unsigned int nq0, unsigned int nq1, unsigned int nq2>
    void Operator3D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        constexpr auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
        constexpr auto nqTot = nq0 * nq1 * nq2;

        constexpr unsigned int ndf = 9;
        unsigned int dfSize        = 1;
        if constexpr (DEFORMED)
        {
            dfSize *= nqTot;
        }

        // Fetch Jacobian and deriv factors.
        auto jacptr_init = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));
        auto dfptr_init = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), false));

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Workspace for kernels - also checks preconditions.
        unsigned int wsp0Size = 0, wsp1Size = 0, wsp2Size = 0;
        BwdTrans3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size);
        IProduct3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size, wsp2Size);
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size),
            wsp1(wsp1Size), wsp2(wsp2Size);
        std::vector<simd_t, tinysimd::allocator<simd_t>> bwd(nqTot);
        std::vector<simd_t, tinysimd::allocator<simd_t>> deriv0(nqTot);
        std::vector<simd_t, tinysimd::allocator<simd_t>> deriv1(nqTot);
        std::vector<simd_t, tinysimd::allocator<simd_t>> deriv2(nqTot);

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();
        const auto width_ratio     = (interleaveWidth == 1)
                                         ? 1
                                         : interleaveWidth / m_implInterleaveWidth;
        const auto chunkSize = std::max(m_implInterleaveWidth, interleaveWidth);

        // Loop over components.
        for (unsigned int n = 0; n < inblock.GetNumComponents(); ++n)
        {
            auto jacptr = jacptr_init;
            auto dfptr  = dfptr_init;

            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                              interleaveWidth, chunkSize, nmTot,
                                              (TData *)inptr);
                }

                // Step 1: BwdTrans.
                BwdTrans3DKernel<SHAPE_TYPE>(
                    nm0, nm1, nm2, nq0, nq1, nq2, m_isModified, m_B[0], m_B[1],
                    m_B[2], m_nodToMod, wsp0.data(), wsp1.data(),
                    reinterpret_cast<const simd_t *>(inptr), bwd.data());

                // Step 2: Get tensor derivatives
                PhysDerivTensor3DKernel<simd_t>(
                    nq0, nq1, nq2, bwd.data(), m_D[0], m_D[1], m_D[2],
                    deriv0.data(), deriv1.data(), deriv2.data());

                // Step 2 + 3 : apply diffusion coeff and WJ
                DiffusionCoeffwithWJ3DKernel<SHAPE_TYPE, DEFORMED, true,
                                             simd_t>(
                    nq0, nq1, nq2, true, this->m_diffCoeff, false,
                    NullTDataVector, NullTDataVector, NullTDataVector,
                    NullTDataVector, NullTDataVector, NullTDataVector,
                    reinterpret_cast<const simd_t *>(jacptr), m_W[0], m_W[1],
                    m_W[2], reinterpret_cast<const simd_t *>(dfptr), m_f[0],
                    m_f[1], m_f[2], m_f[3], deriv0.data(), deriv1.data(),
                    deriv2.data(), bwd.data(), this->m_lambda);

                // Step 4: apply WJ, derivative and sum up.
                SumDerivTensor3DKernel<simd_t>(
                    nq0, nq1, nq2, deriv0.data(), deriv1.data(), deriv2.data(),
                    m_D[0], m_D[1], m_D[2], bwd.data(), 1.0);

                // Step 5 : inner product without WJ.
                IProduct3DKernel<SHAPE_TYPE, false, false, simd_t>(
                    nm0, nm1, nm2, nq0, nq1, nq2, m_isModified, bwd.data(),
                    m_B[0], m_B[1], m_B[2], m_nodToModTrans, wsp0.data(),
                    wsp1.data(), wsp2.data(),
                    reinterpret_cast<simd_t *>(outptr), 1.0);

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        nmTot,
                        (TData *)inptr -
                            (width_ratio - 1) * nmTot * simd_t::width);
                    ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        nmTot,
                        (TData *)outptr -
                            (width_ratio - 1) * nmTot * simd_t::width);
                }

                // Increment pointers.
                dfptr += dfSize * ndf * simd_t::width;
                jacptr += dfSize * simd_t::width;
                inptr += nmTot * simd_t::width;
                outptr += nmTot * simd_t::width;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
