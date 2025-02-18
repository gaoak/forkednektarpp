///////////////////////////////////////////////////////////////////////////////
//
// File: LinAdvDiffReactionSerialAVXSumFac.hpp
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

#include "Operators/ElmtOps/OperatorLinAdvDiffReaction.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/LinAdvDiffReaction/LinAdvDiffReactionSerialAVXSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class BlockOperatorLinAdvDiffReactionImpl
    : public BlockOperatorLinAdvDiffReaction<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    BlockOperatorLinAdvDiffReactionImpl(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
        : BlockOperatorLinAdvDiffReaction<TData>(exp, dataWarehouse)
    {
        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetMetricInfo()->GetGtype() == SpatialDomains::eDeformed;
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
    }

    void apply(BlockAccessor<TData> &inblock,
               BlockAccessor<TData> &outblock) override
    {
        // Check alignment.
        WARNINGL1(inblock.GetAlignment() == simd_t::alignment,
                  "Input Field are not aligned to the required alignment "
                  "for the SIMD vector type.");
        WARNINGL1(outblock.GetAlignment() == simd_t::alignment,
                  "Output Field are not aligned to the required alignment "
                  "for the SIMD vector type.");

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
            default:
                std::cout << "shapetype not implemented" << std::endl;
        }
    }

    void v_SetAdvVel(const unsigned int nVel,
                     BlockAccessor<TData> &advVel) override
    {
        this->m_advVel = advVel.template GetPtr<MemSpace, ReadWrite>();
        for (unsigned int n = 0; n < nVel; n++)
        {
            ReshapeStorage<ExecSpace, simd_t::width>(
                1, advVel.GetNumElementsWithPadding(), advVel.GetNumData(),
                this->m_advVel + n * advVel.size());
        }
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<BlockOperator<TData>> instantiate(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<BlockOperatorLinAdvDiffReactionImpl<
            ExecSpace, Implementation, TData>>(exp, dataWarehouse);
    }

protected:
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
    TData *m_advVel;

    void SegBlock(BlockAccessor<TData> &inblock,
                  BlockAccessor<TData> &outblock);

    void TriBlock(BlockAccessor<TData> &inblock,
                  BlockAccessor<TData> &outblock);

    void QuadBlock(BlockAccessor<TData> &inblock,
                   BlockAccessor<TData> &outblock);

    void HexBlock(BlockAccessor<TData> &inblock,
                  BlockAccessor<TData> &outblock);

    void PrismBlock(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock);

    void PyrBlock(BlockAccessor<TData> &inblock,
                  BlockAccessor<TData> &outblock);

    void TetBlock(BlockAccessor<TData> &inblock,
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

        constexpr unsigned int ndf = 1;
        unsigned int dfSize        = 1;
        if constexpr (DEFORMED)
        {
            dfSize *= nqTot;
        }

        // Fetch Jacobian and deriv factors.
        auto jacptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                JacobianKey<TData>(inblock.GetExpIdx(), simd_t::width,
                                   inblock.GetNumElements())));
        auto dfptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                DerivFactorKey<TData>(inblock.GetExpIdx(), simd_t::width,
                                      inblock.GetNumElements(), false)));

        // Get interleave parameter.
        const unsigned int interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        const auto chunkSize = std::max(simd_t::width, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Allocate workspace.
        auto bwd = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto bwdvec = reinterpret_cast<typename simd_t::vectorType *>(bwd);
        auto deriv0 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto deriv0vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv0);

        // Initialize advVel pointers.
        auto advVel = m_advVel;
        auto advVelPtr_init =
            reinterpret_cast<const typename simd_t::vectorType *>(advVel);

        // Initialize pointers.
        auto input  = (interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto outptr = reinterpret_cast<typename simd_t::scalarType *>(output);

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            auto jacptr    = jacptr_init;
            auto dfptr     = dfptr_init;
            auto advVelPtr = advVelPtr_init;
            for (unsigned int e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        interleave_width, chunkSize, nmTot, (TData *)inptr);
                }

                // Step 1: BwdTrans.
                BwdTrans1DKernel<SHAPE_TYPE>(nm0, nq0, m_B[0], inptr, bwd);

                // Step 2: Take derivatives in collapsed coordinate space.
                PhysDerivTensor1DKernel(nq0, bwdvec, m_D[0], deriv0);

                // Step 3: add Advect solution to  this->m_lambda * bwd.
                AddAdvectionSegKernel<DEFORMED>(nq0, advVelPtr, dfptr,
                                                deriv0vec, bwd, this->m_lambda);

                // Step 4: Inner product for mass matrix operation.
                IProduct1DKernel<SHAPE_TYPE, true, false, DEFORMED>(
                    nm0, nq0, bwdvec, m_B[0], m_W[0], jacptr, outptr);

                // Step 5: Apply diffusion coefficiets.
                DiffusionCoeffSegKernel<DEFORMED, simd_t>(
                    nq0, true, this->m_diffCoeff, false, NullTDataVector, dfptr,
                    deriv0);

                // Step 6: Apply Laplacian metrics & inner product.
                IProduct1DKernel<SHAPE_TYPE, false, true, DEFORMED>(
                    nm0, nq0, deriv0vec, m_DB[0], m_W[0], jacptr, outptr);

                // Increment pointers.
                dfptr += dfSize * ndf;
                jacptr += dfSize;
                advVelPtr += nqTot;
                inptr += nmTot;
                outptr += nmTot * simd_t::width;
            }
        }

        // Free aligned memory.
        ::operator delete[](bwd, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv0, std::align_val_t(simd_t::alignment));
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

        constexpr unsigned int ndf = 1;
        unsigned int dfSize        = 1;
        if constexpr (DEFORMED)
        {
            dfSize *= nqTot;
        }

        // Fetch Jacobian and deriv factors.
        auto jacptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                JacobianKey<TData>(inblock.GetExpIdx(), simd_t::width,
                                   inblock.GetNumElements())));
        auto dfptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                DerivFactorKey<TData>(inblock.GetExpIdx(), simd_t::width,
                                      inblock.GetNumElements(), false)));

        // Get interleave parameter.
        const unsigned int interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        const auto chunkSize = std::max(simd_t::width, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Allocate workspace.
        auto bwd = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto bwdvec = reinterpret_cast<typename simd_t::vectorType *>(bwd);
        auto deriv0 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto deriv0vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv0);

        // Initialize advVel pointers.
        auto advVel = m_advVel;
        auto advVelPtr_init =
            reinterpret_cast<const typename simd_t::vectorType *>(advVel);

        // Initialize pointers.
        auto input  = (interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto outptr = reinterpret_cast<typename simd_t::scalarType *>(output);

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            auto jacptr    = jacptr_init;
            auto dfptr     = dfptr_init;
            auto advVelPtr = advVelPtr_init;
            for (unsigned int e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        interleave_width, chunkSize, nmTot, (TData *)inptr);
                }

                // Step 1: BwdTrans.
                BwdTrans1DKernel<SHAPE_TYPE>(nm0, nq0, m_B[0], inptr, bwd);

                // Step 2: Take derivatives in collapsed coordinate space.
                PhysDerivTensor1DKernel(nq0, bwdvec, m_D[0], deriv0);

                // Step 3: add Advect solution to  this->m_lambda * bwd.
                AddAdvectionSegKernel<DEFORMED>(nq0, advVelPtr, dfptr,
                                                deriv0vec, bwd, this->m_lambda);

                // Step 4: Inner product for mass matrix operation.
                IProduct1DKernel<SHAPE_TYPE, true, false, DEFORMED>(
                    nm0, nq0, bwdvec, m_B[0], m_W[0], jacptr, outptr);

                // Step 5: Apply diffusion coefficiets.
                DiffusionCoeffSegKernel<DEFORMED, simd_t>(
                    nq0, true, this->m_diffCoeff, false, NullTDataVector, dfptr,
                    deriv0);

                // Step 6: Apply Laplacian metrics & inner product.
                IProduct1DKernel<SHAPE_TYPE, false, true, DEFORMED>(
                    nm0, nq0, deriv0vec, m_DB[0], m_W[0], jacptr, outptr);

                // Increment pointers.
                dfptr += dfSize * ndf;
                jacptr += dfSize;
                advVelPtr += nqTot;
                inptr += nmTot;
                outptr += nmTot * simd_t::width;
            }
        }

        // Free aligned memory.
        ::operator delete[](bwd, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv0, std::align_val_t(simd_t::alignment));
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

        constexpr unsigned int ndf = 4;
        unsigned int dfSize        = 1;
        if constexpr (DEFORMED)
        {
            dfSize *= nqTot;
        }

        // Fetch Jacobian and deriv factors.
        auto jacptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                JacobianKey<TData>(inblock.GetExpIdx(), simd_t::width,
                                   inblock.GetNumElements())));
        auto dfptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                DerivFactorKey<TData>(inblock.GetExpIdx(), simd_t::width,
                                      inblock.GetNumElements(), false)));

        // Get interleave parameter.
        const unsigned int interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        const auto chunkSize = std::max(simd_t::width, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Workspace for kernels - also checks preconditions.
        unsigned int wsp0Size = 0;
        BwdTrans2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);
        IProduct2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size);

        auto bwd = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto bwdvec = reinterpret_cast<typename simd_t::vectorType *>(bwd);
        auto deriv0 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto deriv0vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv0);
        auto deriv1 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto deriv1vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv1);

        auto diffderiv0 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto diffderiv0vec =
            reinterpret_cast<typename simd_t::vectorType *>(diffderiv0);

        auto diffderiv1 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto diffderiv1vec =
            reinterpret_cast<typename simd_t::vectorType *>(diffderiv1);

        // Initialize advVel pointers.
        auto advVelOffset = inblock.GetNumElmtGroups() * nqTot;
        auto advVel       = m_advVel;
        auto advVelPtr_init =
            reinterpret_cast<const typename simd_t::vectorType *>(advVel);

        // Initialize pointers.
        auto input  = (interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto outptr = reinterpret_cast<typename simd_t::scalarType *>(output);

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            auto jacptr    = jacptr_init;
            auto dfptr     = dfptr_init;
            auto advVelPtr = advVelPtr_init;
            for (unsigned int e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        interleave_width, chunkSize, nmTot, (TData *)inptr);
                }

                // Step 1: BwdTrans.
                BwdTrans2DKernel<SHAPE_TYPE>(nm0, nm1, nq0, nq1, m_isModified,
                                             m_B[0], m_B[1], wsp0, inptr, bwd);

                // Step 2 + 3: Get tensor derivative (deriv0, deriv1)  and apply
                // diffusion coeff (diffderiv0, diffderiv1).
                TensorDerivWithDiffuCoeff2DKernel<SHAPE_TYPE, DEFORMED, simd_t>(
                    nq0, nq1, true, this->m_diffCoeff, false, NullTDataVector,
                    NullTDataVector, NullTDataVector, bwdvec, m_D[0], m_D[1],
                    dfptr, m_f[0], m_f[1], diffderiv0, diffderiv1, deriv0,
                    deriv1);

                // Step 4 evaluate advection term and add to bwd * lambda.
                AddAdvection2DKernel<SHAPE_TYPE, DEFORMED>(
                    nq0, nq1, m_f[0], m_f[1], advVelPtr,
                    advVelPtr + advVelOffset, dfptr, deriv0vec, deriv1vec, bwd,
                    this->m_lambda);

                // Step 5: apply WJ, derivative and sum up.
                SumDerivTensor2DKernel<DEFORMED, simd_t>(
                    nq0, nq1, diffderiv0vec, diffderiv1vec, m_W[0], m_W[1],
                    jacptr, m_D[0], m_D[1], bwd, 1.0);

                // Step 6 : inner product without WJ.
                IProduct2DKernel<SHAPE_TYPE, false, false, simd_t>(
                    nm0, nm1, nq0, nq1, m_isModified, bwdvec, m_B[0], m_B[1],
                    wsp0, outptr);

                // Increment pointers.
                dfptr += dfSize * ndf;
                jacptr += dfSize;
                advVelPtr += nqTot;
                inptr += nmTot;
                outptr += nmTot * simd_t::width;
            }
        }

        // Free aligned memory.
        ::operator delete[](bwd, std::align_val_t(simd_t::alignment));
        ::operator delete[](diffderiv0, std::align_val_t(simd_t::alignment));
        ::operator delete[](diffderiv1, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv0, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv1, std::align_val_t(simd_t::alignment));
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

        constexpr unsigned ndf = 4;
        unsigned int dfSize    = 1;
        if constexpr (DEFORMED)
        {
            dfSize *= nqTot;
        }

        // Fetch Jacobian and deriv factors.
        auto jacptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                JacobianKey<TData>(inblock.GetExpIdx(), simd_t::width,
                                   inblock.GetNumElements())));
        auto dfptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                DerivFactorKey<TData>(inblock.GetExpIdx(), simd_t::width,
                                      inblock.GetNumElements(), false)));

        // Get interleave parameter.
        const unsigned int interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        const auto chunkSize = std::max(simd_t::width, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Workspace for kernels - also checks preconditions.
        unsigned int wsp0Size = 0;
        BwdTrans2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);
        IProduct2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size);

        auto bwd = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto bwdvec = reinterpret_cast<typename simd_t::vectorType *>(bwd);

        auto deriv0 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto deriv0vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv0);

        auto deriv1 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto deriv1vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv1);

        auto diffderiv0 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto diffderiv0vec =
            reinterpret_cast<typename simd_t::vectorType *>(diffderiv0);

        auto diffderiv1 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto diffderiv1vec =
            reinterpret_cast<typename simd_t::vectorType *>(diffderiv1);

        // Initialize advVel pointers.
        auto advVelOffset = inblock.GetNumElmtGroups() * nqTot;
        auto advVel       = m_advVel;
        auto advVelPtr_init =
            reinterpret_cast<const typename simd_t::vectorType *>(advVel);

        // Initialize pointers.
        auto input  = (interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto outptr = reinterpret_cast<typename simd_t::scalarType *>(output);

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            auto jacptr    = jacptr_init;
            auto dfptr     = dfptr_init;
            auto advVelPtr = advVelPtr_init;
            for (unsigned int e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        interleave_width, chunkSize, nmTot, (TData *)inptr);
                }

                // Step 1: BwdTrans.
                BwdTrans2DKernel<SHAPE_TYPE>(nm0, nm1, nq0, nq1, m_isModified,
                                             m_B[0], m_B[1], wsp0, inptr, bwd);

                // Step 2 + 3: Get tensor derivative (deriv0, deriv1)  and apply
                // diffusion coeff (diffderiv0, diffderiv1).
                TensorDerivWithDiffuCoeff2DKernel<SHAPE_TYPE, DEFORMED, simd_t>(
                    nq0, nq1, true, this->m_diffCoeff, false, NullTDataVector,
                    NullTDataVector, NullTDataVector, bwdvec, m_D[0], m_D[1],
                    dfptr, m_f[0], m_f[1], diffderiv0, diffderiv1, deriv0,
                    deriv1);

                // Step 4 evaluate advection term and add to bwd * lambda.
                AddAdvection2DKernel<SHAPE_TYPE, DEFORMED>(
                    nq0, nq1, m_f[0], m_f[1], advVelPtr,
                    advVelPtr + advVelOffset, dfptr, deriv0vec, deriv1vec, bwd,
                    this->m_lambda);

                // Step 5: apply WJ, derivative and sum up.
                SumDerivTensor2DKernel<DEFORMED, simd_t>(
                    nq0, nq1, diffderiv0vec, diffderiv1vec, m_W[0], m_W[1],
                    jacptr, m_D[0], m_D[1], bwd, 1.0);

                // Step 6 : inner product without WJ.
                IProduct2DKernel<SHAPE_TYPE, false, false, simd_t>(
                    nm0, nm1, nq0, nq1, m_isModified, bwdvec, m_B[0], m_B[1],
                    wsp0, outptr);

                // Increment pointers.
                dfptr += dfSize * ndf;
                jacptr += dfSize;
                advVelPtr += nqTot;
                inptr += nmTot;
                outptr += nmTot * simd_t::width;
            }
        }

        // Free aligned memory.
        ::operator delete[](bwd, std::align_val_t(simd_t::alignment));
        ::operator delete[](diffderiv0, std::align_val_t(simd_t::alignment));
        ::operator delete[](diffderiv1, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv0, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv1, std::align_val_t(simd_t::alignment));
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
        auto jacptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                JacobianKey<TData>(inblock.GetExpIdx(), simd_t::width,
                                   inblock.GetNumElements())));
        auto dfptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                DerivFactorKey<TData>(inblock.GetExpIdx(), simd_t::width,
                                      inblock.GetNumElements(), false)));

        // Get interleave parameter.
        const unsigned int interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        const auto chunkSize = std::max(simd_t::width, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Workspace for kernels - also checks preconditions.
        unsigned int wsp0Size = 0, wsp1Size = 0, wsp2Size = 0;
        BwdTrans3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size);
        IProduct3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size, wsp2Size);

        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size),
            wsp1(wsp1Size), wsp2(wsp2Size);

        auto bwd = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto bwdvec = reinterpret_cast<typename simd_t::vectorType *>(bwd);

        auto deriv0 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto deriv0vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv0);
        auto deriv1 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto deriv1vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv1);
        auto deriv2 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto deriv2vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv2);

        auto diffderiv0 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto diffderiv0vec =
            reinterpret_cast<typename simd_t::vectorType *>(diffderiv0);

        auto diffderiv1 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto diffderiv1vec =
            reinterpret_cast<typename simd_t::vectorType *>(diffderiv1);

        auto diffderiv2 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto diffderiv2vec =
            reinterpret_cast<typename simd_t::vectorType *>(diffderiv2);

        // Initialize advVel pointers.
        auto advVelOffset = inblock.GetNumElmtGroups() * nqTot;
        auto advVel       = m_advVel;
        auto advVelPtr_init =
            reinterpret_cast<const typename simd_t::vectorType *>(advVel);

        // Initialize pointers.
        auto input  = (interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto outptr = reinterpret_cast<typename simd_t::scalarType *>(output);

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            auto jacptr    = jacptr_init;
            auto dfptr     = dfptr_init;
            auto advVelPtr = advVelPtr_init;
            for (unsigned int e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        interleave_width, chunkSize, nmTot, (TData *)inptr);
                }

                // Step 1: BwdTrans.
                BwdTrans3DKernel<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2,
                                             m_isModified, m_B[0], m_B[1],
                                             m_B[2], wsp0, wsp1, inptr, bwd);

                // Step 2 + 3 : Get tensor derivative and apply diffusion
                // coeff.
                TensorDerivWithDiffuCoeff3DKernel<SHAPE_TYPE, DEFORMED, simd_t>(
                    nq0, nq1, nq2, true, this->m_diffCoeff, false,
                    NullTDataVector, NullTDataVector, NullTDataVector,
                    NullTDataVector, NullTDataVector, NullTDataVector, bwdvec,
                    m_D[0], m_D[1], m_D[2], dfptr, m_f[0], m_f[1], m_f[2],
                    m_f[3], diffderiv0, diffderiv1, diffderiv2, deriv0, deriv1,
                    deriv2);

                // Step 4 evaluate advection term and add to bwd * lambda.
                AddAdvection3DKernel<SHAPE_TYPE, DEFORMED>(
                    nq0, nq1, nq2, m_f[0], m_f[1], m_f[2], m_f[3], advVelPtr,
                    advVelPtr + advVelOffset, advVelPtr + 2 * advVelOffset,
                    dfptr, deriv0vec, deriv1vec, deriv2vec, bwd,
                    this->m_lambda);

                // Step 5: apply WJ, derivative and sum up.
                SumDerivTensor3DKernel<DEFORMED, simd_t>(
                    nq0, nq1, nq2, diffderiv0vec, diffderiv1vec, diffderiv2vec,
                    m_W[0], m_W[1], m_W[2], jacptr, m_D[0], m_D[1], m_D[2], bwd,
                    1.0);

                // Step 6 : inner product without WJ.
                IProduct3DKernel<SHAPE_TYPE, false, false, simd_t>(
                    nm0, nm1, nm2, nq0, nq1, nq2, m_isModified, bwdvec, m_B[0],
                    m_B[1], m_B[2], wsp0, wsp1, wsp2, outptr);

                // Increment pointers.
                dfptr += dfSize * ndf;
                jacptr += dfSize;
                advVelPtr += nqTot;
                inptr += nmTot;
                outptr += nmTot * simd_t::width;
            }
        }

        // Free aligned memory.
        ::operator delete[](bwd, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv0, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv1, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv2, std::align_val_t(simd_t::alignment));
        ::operator delete[](diffderiv0, std::align_val_t(simd_t::alignment));
        ::operator delete[](diffderiv1, std::align_val_t(simd_t::alignment));
        ::operator delete[](diffderiv2, std::align_val_t(simd_t::alignment));
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
        auto jacptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                JacobianKey<TData>(inblock.GetExpIdx(), simd_t::width,
                                   inblock.GetNumElements())));
        auto dfptr_init = reinterpret_cast<const simd_t *>(
            this->m_dataWarehouse->template GetData<ExecSpace>(
                DerivFactorKey<TData>(inblock.GetExpIdx(), simd_t::width,
                                      inblock.GetNumElements(), false)));

        // Get interleave parameter.
        const unsigned int interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        const auto chunkSize = std::max(simd_t::width, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Workspace for kernels - also checks preconditions.
        unsigned int wsp0Size = 0, wsp1Size = 0, wsp2Size = 0;
        BwdTrans3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size);
        IProduct3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size, wsp2Size);
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size),
            wsp1(wsp1Size), wsp2(wsp2Size);

        auto bwd = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto bwdvec = reinterpret_cast<typename simd_t::vectorType *>(bwd);

        auto deriv0 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto deriv0vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv0);
        auto deriv1 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto deriv1vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv1);
        auto deriv2 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto deriv2vec =
            reinterpret_cast<typename simd_t::vectorType *>(deriv2);

        auto diffderiv0 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto diffderiv0vec =
            reinterpret_cast<typename simd_t::vectorType *>(diffderiv0);

        auto diffderiv1 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto diffderiv1vec =
            reinterpret_cast<typename simd_t::vectorType *>(diffderiv1);

        auto diffderiv2 = static_cast<TData *>(
            ::operator new[](nqTot *simd_t::width * sizeof(TData),
                             std::align_val_t(simd_t::alignment)));
        auto diffderiv2vec =
            reinterpret_cast<typename simd_t::vectorType *>(diffderiv2);

        // Initialize advVel pointers.
        auto advVelOffset = inblock.GetNumElmtGroups() * nqTot;
        auto advVel       = m_advVel;
        auto advVelPtr_init =
            reinterpret_cast<const typename simd_t::vectorType *>(advVel);

        // Initialize pointers.
        auto input  = (interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto outptr = reinterpret_cast<typename simd_t::scalarType *>(output);

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            auto jacptr    = jacptr_init;
            auto dfptr     = dfptr_init;
            auto advVelPtr = advVelPtr_init;
            for (unsigned int e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        interleave_width, chunkSize, nmTot, (TData *)inptr);
                }

                // Step 1: BwdTrans.
                BwdTrans3DKernel<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2,
                                             m_isModified, m_B[0], m_B[1],
                                             m_B[2], wsp0, wsp1, inptr, bwd);

                // Step 2 + 3 : Get tensor derivative and apply diffusion
                // coeff.
                TensorDerivWithDiffuCoeff3DKernel<SHAPE_TYPE, DEFORMED, simd_t>(
                    nq0, nq1, nq2, true, this->m_diffCoeff, false,
                    NullTDataVector, NullTDataVector, NullTDataVector,
                    NullTDataVector, NullTDataVector, NullTDataVector, bwdvec,
                    m_D[0], m_D[1], m_D[2], dfptr, m_f[0], m_f[1], m_f[2],
                    m_f[3], diffderiv0, diffderiv1, diffderiv2, deriv0, deriv1,
                    deriv2);

                // Step 4 evaluate advection term and add to bwd * lambda.
                AddAdvection3DKernel<SHAPE_TYPE, DEFORMED>(
                    nq0, nq1, nq2, m_f[0], m_f[1], m_f[2], m_f[3], advVelPtr,
                    advVelPtr + advVelOffset, advVelPtr + 2 * advVelOffset,
                    dfptr, deriv0vec, deriv1vec, deriv2vec, bwd,
                    this->m_lambda);

                // Step 5: apply WJ, derivative and sum up.
                SumDerivTensor3DKernel<DEFORMED, simd_t>(
                    nq0, nq1, nq2, diffderiv0vec, diffderiv1vec, diffderiv2vec,
                    m_W[0], m_W[1], m_W[2], jacptr, m_D[0], m_D[1], m_D[2], bwd,
                    1.0);

                // Step 6 : inner product without WJ.
                IProduct3DKernel<SHAPE_TYPE, false, false, simd_t>(
                    nm0, nm1, nm2, nq0, nq1, nq2, m_isModified, bwdvec, m_B[0],
                    m_B[1], m_B[2], wsp0, wsp1, wsp2, outptr);

                // Increment pointers.
                dfptr += dfSize * ndf;
                jacptr += dfSize;
                advVelPtr += nqTot;
                inptr += nmTot;
                outptr += nmTot * simd_t::width;
            }
        }

        // Free aligned memory.
        ::operator delete[](bwd, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv0, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv1, std::align_val_t(simd_t::alignment));
        ::operator delete[](deriv2, std::align_val_t(simd_t::alignment));
        ::operator delete[](diffderiv0, std::align_val_t(simd_t::alignment));
        ::operator delete[](diffderiv1, std::align_val_t(simd_t::alignment));
        ::operator delete[](diffderiv2, std::align_val_t(simd_t::alignment));
    }
};

} // namespace Nektar::Operators::detail
