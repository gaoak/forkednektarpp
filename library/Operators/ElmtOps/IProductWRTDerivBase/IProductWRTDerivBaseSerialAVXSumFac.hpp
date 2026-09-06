///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseSerialAVXSumFac.hpp
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

#include "LibUtilities/BasicUtils/Math/MathKernels.hpp"
#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseBlockOp.hpp"

#include "Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseSerialAVXSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, FieldState TFieldOut,
          typename TData>
class IProductWRTDerivBaseBlockOpImpl
    : public IProductWRTDerivBaseBlockOp<TFieldOut, TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    IProductWRTDerivBaseBlockOpImpl(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : IProductWRTDerivBaseBlockOp<TFieldOut, TData>(block_idx, exp,
                                                        dataWarehouse)
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
            m_B.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    exp->GetBasis(d)->GetBasisKey(), LibUtilities::eBasis)));
            m_D.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    exp->GetBasis(d)->GetBasisKey(),
                    LibUtilities::eDerivative)));
            m_W.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    exp->GetBasis(d)->GetBasisKey(), LibUtilities::eWeights)));
        }

        if (m_dimension == 2)
        {
            // Fetch geometric factors.
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    exp->GetBasis(0)->GetBasisKey(),
                    LibUtilities::eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    exp->GetBasis(1)->GetBasisKey(),
                    LibUtilities::eTwoOverOneMinusZero)));
        }
        else if (m_dimension == 3)
        {
            // Fetch geometric factors.
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    exp->GetBasis(0)->GetBasisKey(),
                    LibUtilities::eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    exp->GetBasis(1)->GetBasisKey(),
                    LibUtilities::eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    exp->GetBasis(1)->GetBasisKey(),
                    LibUtilities::eTwoOverOneMinusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    exp->GetBasis(2)->GetBasisKey(),
                    LibUtilities::eTwoOverOneMinusZero)));
        }

        if ((m_shapeType == LibUtilities::eNodalTri) ||
            (m_shapeType == LibUtilities::eNodalTet) ||
            (m_shapeType == LibUtilities::eNodalPrism))
        {
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

            // Fetch NodalToModal Matrix if required.
            m_nodToModTrans = dataWarehouse->template GetData<MemSpace>(
                StdRegions::StdMatKey<simd_t>(
                    basisKeys, m_shapeType, StdRegions::eNodalToModalTranspose,
                    nodalType));
        }
        else
        {
            m_nodToModTrans = (const simd_t *)nullptr;
        }

        // Fetch Jacobian and deriv factors.
        m_jacptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::JacobianKey<TData>(block_idx, m_implInterleaveWidth));
        m_dfptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::DerivFactorKey<TData>(block_idx,
                                                m_implInterleaveWidth, false));

        // Workspace for kernels - also checks preconditions.
        m_df = std::vector<simd_t, tinysimd::allocator<simd_t>>(m_dimension *
                                                                m_coordDim);
        if (m_dimension == 2)
        {
            unsigned int wsp0Size = 0;
            IProduct2DWorkspace(m_shapeType, m_nm[0], m_nm[1], m_nq[0], m_nq[1],
                                wsp0Size);
            m_wsp.push_back(
                std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp0Size));
        }
        else if (m_dimension == 3)
        {
            unsigned int wsp0Size = 0, wsp1Size = 0, wsp2Size = 0;
            IProduct3DWorkspace(m_shapeType, m_nm[0], m_nm[1], m_nm[2], m_nq[0],
                                m_nq[1], m_nq[2], wsp0Size, wsp1Size, wsp2Size);
            m_wsp.push_back(
                std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp0Size));
            m_wsp.push_back(
                std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp1Size));
            m_wsp.push_back(
                std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp2Size));
        }

        // One tensorial derivative per direction, plus the array holding
        // their sum.
        unsigned int nqTot = 1;
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            nqTot *= m_nq[d];
        }
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            m_tmp.push_back(
                std::vector<simd_t, tinysimd::allocator<simd_t>>(nqTot));
        }
        m_tmpsum = std::vector<simd_t, tinysimd::allocator<simd_t>>(nqTot);
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<ElmtBlockOp<FieldState::Phys, TFieldOut, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<IProductWRTDerivBaseBlockOpImpl<
            ExecSpace, Implementation, TFieldOut, TData>>(block_idx, exp,
                                                          dataWarehouse);
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
    std::vector<const TData *> m_B;
    std::vector<const TData *> m_D;
    std::vector<const TData *> m_W;
    std::vector<const simd_t *> m_f;
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_df;
    std::vector<std::vector<simd_t, tinysimd::allocator<simd_t>>> m_wsp;
    std::vector<std::vector<simd_t, tinysimd::allocator<simd_t>>> m_tmp;
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_tmpsum;
    const simd_t *m_nodToModTrans;
    const TData *m_jacptr;
    const TData *m_dfptr;
#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
    // flag to ensure we only get one warning for alignment otherwise CI system
    // is saturated with warnings
    bool m_warnOnce = false;
#endif

    void v_Apply(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, TFieldOut> &outblock) override
    {
        WARNINGL1(
            m_warnOnce || (inblock.GetAlignment() % simd_t::alignment == 0 &&
                           outblock.GetAlignment() % simd_t::alignment == 0),
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
            // NodalTriangles
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

    void SegBlock(LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
                  LibUtilities::BlockAccessor<TData, TFieldOut> &outblock);

    void QuadBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, TFieldOut> &outblock);

    void TriBlock(LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
                  LibUtilities::BlockAccessor<TData, TFieldOut> &outblock);

    void NodalTriBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, TFieldOut> &outblock);

    void HexBlock(LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
                  LibUtilities::BlockAccessor<TData, TFieldOut> &outblock);

    void PrismBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, TFieldOut> &outblock);

    void NodalPrismBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, TFieldOut> &outblock);

    void PyrBlock(LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
                  LibUtilities::BlockAccessor<TData, TFieldOut> &outblock);

    void TetBlock(LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
                  LibUtilities::BlockAccessor<TData, TFieldOut> &outblock);

    void NodalTetBlock(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, TFieldOut> &outblock);

    // Non-size based operator with Coeff output.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator1D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Coeff> &outblock)
    {
        OperatorND<SHAPE_TYPE, DEFORMED>(
            inblock, outblock, NonTemplatedSizeParameter1D(m_nm[0], m_nq[0]),
            std::make_integer_sequence<unsigned int, 1>(),
            std::make_integer_sequence<unsigned int, 0>(),
            std::make_integer_sequence<unsigned int, 0>());
    }

    // Non-size based operator with Phys output.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator1D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        OperatorND<SHAPE_TYPE, DEFORMED>(
            inblock, outblock,
            NonTemplatedPhysSizeParameter1D(m_coordDim, m_nq[0]),
            std::make_integer_sequence<unsigned int, 1>(),
            std::make_integer_sequence<unsigned int, 0>(),
            std::make_integer_sequence<unsigned int, 0>());
    }

    // Size based template versions.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nm0, unsigned int nq0>
    void Operator1D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Coeff> &outblock)
    {
        OperatorND<SHAPE_TYPE, DEFORMED>(
            inblock, outblock, TemplatedSizeParameter1D<nm0, nq0>(),
            std::make_integer_sequence<unsigned int, 1>(),
            std::make_integer_sequence<unsigned int, 0>(),
            std::make_integer_sequence<unsigned int, 0>());
    }

    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int coordDim, unsigned int nq0>
    void Operator1D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        OperatorND<SHAPE_TYPE, DEFORMED>(
            inblock, outblock, TemplatedPhysSizeParameter1D<coordDim, nq0>(),
            std::make_integer_sequence<unsigned int, 1>(),
            std::make_integer_sequence<unsigned int, 0>(),
            std::make_integer_sequence<unsigned int, 0>());
    }

    // Non-size based operator with Coeff output.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator2D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Coeff> &outblock)
    {
        const unsigned int nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, m_nm[0], m_nm[1]);

        OperatorND<SHAPE_TYPE, DEFORMED>(
            inblock, outblock,
            NonTemplatedSizeParameter2D(m_nm[0], m_nm[1], nmTot, m_nq[0],
                                        m_nq[1]),
            std::make_integer_sequence<unsigned int, 2>(),
            std::make_integer_sequence<unsigned int, 1>(),
            std::make_integer_sequence<unsigned int, 2>());
    }

    // Non-size based operator with Phys output.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator2D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        OperatorND<SHAPE_TYPE, DEFORMED>(
            inblock, outblock,
            NonTemplatedPhysSizeParameter2D(m_coordDim, m_nq[0], m_nq[1]),
            std::make_integer_sequence<unsigned int, 2>(),
            std::make_integer_sequence<unsigned int, 1>(),
            std::make_integer_sequence<unsigned int, 2>());
    }

    // Size based template versions.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nm0, unsigned int nm1, unsigned int nq0,
              unsigned int nq1>
    void Operator2D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Coeff> &outblock)
    {
        constexpr unsigned int nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);

        OperatorND<SHAPE_TYPE, DEFORMED>(
            inblock, outblock,
            TemplatedSizeParameter2D<nm0, nm1, nmTot, nq0, nq1>(),
            std::make_integer_sequence<unsigned int, 2>(),
            std::make_integer_sequence<unsigned int, 1>(),
            std::make_integer_sequence<unsigned int, 2>());
    }

    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int coordDim, unsigned int nq0, unsigned int nq1>
    void Operator2D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        OperatorND<SHAPE_TYPE, DEFORMED>(
            inblock, outblock,
            TemplatedPhysSizeParameter2D<coordDim, nq0, nq1>(),
            std::make_integer_sequence<unsigned int, 2>(),
            std::make_integer_sequence<unsigned int, 1>(),
            std::make_integer_sequence<unsigned int, 2>());
    }

    // Non-size based operator with Coeff output.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator3D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Coeff> &outblock)
    {
        const unsigned int nmTot = LibUtilities::GetNumberOfCoefficients(
            SHAPE_TYPE, m_nm[0], m_nm[1], m_nm[2]);

        OperatorND<SHAPE_TYPE, DEFORMED>(
            inblock, outblock,
            NonTemplatedSizeParameter3D(m_nm[0], m_nm[1], m_nm[2], nmTot,
                                        m_nq[0], m_nq[1], m_nq[2]),
            std::make_integer_sequence<unsigned int, 3>(),
            std::make_integer_sequence<unsigned int, 3>(),
            std::make_integer_sequence<unsigned int, 4>());
    }

    // Non-size based operator with Phys output.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator3D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        OperatorND<SHAPE_TYPE, DEFORMED>(
            inblock, outblock,
            NonTemplatedPhysSizeParameter3D(m_nq[0], m_nq[1], m_nq[2]),
            std::make_integer_sequence<unsigned int, 3>(),
            std::make_integer_sequence<unsigned int, 3>(),
            std::make_integer_sequence<unsigned int, 4>());
    }

    // Size based template versions.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nm0, unsigned int nm1, unsigned int nm2,
              unsigned int nq0, unsigned int nq1, unsigned int nq2>
    void Operator3D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Coeff> &outblock)
    {
        constexpr unsigned int nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);

        OperatorND<SHAPE_TYPE, DEFORMED>(
            inblock, outblock,
            TemplatedSizeParameter3D<nm0, nm1, nm2, nmTot, nq0, nq1, nq2>(),
            std::make_integer_sequence<unsigned int, 3>(),
            std::make_integer_sequence<unsigned int, 3>(),
            std::make_integer_sequence<unsigned int, 4>());
    }

    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              unsigned int nq0, unsigned int nq1, unsigned int nq2>
    void Operator3D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        OperatorND<SHAPE_TYPE, DEFORMED>(
            inblock, outblock, TemplatedPhysSizeParameter3D<nq0, nq1, nq2>(),
            std::make_integer_sequence<unsigned int, 3>(),
            std::make_integer_sequence<unsigned int, 3>(),
            std::make_integer_sequence<unsigned int, 4>());
    }

    // Generic operator implementation.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TSizeParameter, unsigned int... ind0,
              unsigned int... ind1, unsigned int... ind2>
    NEK_FORCE_INLINE void OperatorND(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, TFieldOut> &outblock,
        TSizeParameter sizeParam, std::integer_sequence<unsigned int, ind0...>,
        std::integer_sequence<unsigned int, ind1...>,
        std::integer_sequence<unsigned int, ind2...>)
    {
        constexpr unsigned int ndim = sizeof...(ind0);

        // Shape size.
        const auto nqTot = sizeParam.nqTot();

        const auto ndf       = ndim * m_coordDim;
        unsigned int jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
        }

        // Initialize pointers.
        auto inptr    = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr   = (this->m_append)
                            ? outblock.template GetPtr<MemSpace, ReadWrite>()
                            : outblock.template GetPtr<MemSpace, WriteOnly>();
        auto outndata = outblock.GetNumData();

        // Get interleave parameter.
        const auto inInterleaveWidth  = inblock.GetInterleaveWidth();
        const auto outInterleaveWidth = outblock.GetInterleaveWidth();
        const auto width_ratio =
            (inInterleaveWidth == 1)
                ? 1
                : inInterleaveWidth / m_implInterleaveWidth;
        const auto chunkSize =
            std::max(m_implInterleaveWidth, inInterleaveWidth);

        // Loop over components.
        const auto inoffset = inblock.CompSize() * inblock.GetNumHomoModes();
        const auto inoffset_vec = inoffset / simd_t::width;
        for (unsigned int n = 0;
             n < outblock.GetNumComponents() * outblock.GetNumHomoModes(); ++n)
        {
            auto jacptr = m_jacptr;
            auto dfptr  = m_dfptr;

            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    for (unsigned int k = 0; k < m_coordDim; ++k)
                    {
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            m_implInterleaveWidth, inInterleaveWidth, chunkSize,
                            nqTot, (TData *)(inptr + k * inoffset));
                    }
                }

                // IProductWRTDerivBase kernel.
                IProductWRTDerivBaseKernelLauncher<SHAPE_TYPE, DEFORMED>(
                    sizeParam, m_coordDim,
                    reinterpret_cast<const simd_t *>(dfptr), m_df, inoffset_vec,
                    reinterpret_cast<const simd_t *>(inptr), m_f[ind2]...,
                    reinterpret_cast<const simd_t *>(jacptr), m_W[ind0]...,
                    m_D[ind0]..., m_tmp[ind0].data()..., m_tmpsum.data());

                if (this->m_append)
                {
                    // Reshape, if necessary.
                    if (e % width_ratio == 0)
                    {
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            m_implInterleaveWidth, outInterleaveWidth,
                            chunkSize, outndata, (TData *)outptr);
                    }
                }

                if constexpr (TFieldOut == FieldState::Phys)
                {
                    simd_t *vecout = reinterpret_cast<simd_t *>(outptr);
                    if (this->m_append)
                    {
                        for (unsigned int i = 0; i < nqTot; ++i)
                        {
                            vecout[i] += this->m_scale * m_tmpsum[i];
                        }
                    }
                    else
                    {
                        for (unsigned int i = 0; i < nqTot; ++i)
                        {
                            vecout[i] = this->m_scale * m_tmpsum[i];
                        }
                    }
                }
                else
                {
                    // Inner product without the quadrature metric wJ.
                    if (this->m_append)
                    {
                        IProductWRTBaseKernelLauncher<SHAPE_TYPE, true, true>(
                            sizeParam, m_isModified, m_tmpsum.data(),
                            m_B[ind0]..., m_nodToModTrans,
                            m_wsp[ind1].data()...,
                            reinterpret_cast<simd_t *>(outptr), this->m_scale);
                    }
                    else
                    {
                        IProductWRTBaseKernelLauncher<SHAPE_TYPE, true, false>(
                            sizeParam, m_isModified, m_tmpsum.data(),
                            m_B[ind0]..., m_nodToModTrans,
                            m_wsp[ind1].data()...,
                            reinterpret_cast<simd_t *>(outptr), this->m_scale);
                    }
                }

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    for (unsigned int k = 0; k < m_coordDim; ++k)
                    {
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                            nqTot,
                            (TData *)inptr + k * inoffset -
                                (width_ratio - 1) * nqTot * simd_t::width);
                    }
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                        outndata,
                        (TData *)outptr -
                            (width_ratio - 1) * outndata * simd_t::width);
                }

                // Increment pointers for the next elmt group.
                inptr += nqTot * simd_t::width;
                outptr += outndata * simd_t::width;
                jacptr += jacSize * simd_t::width;
                dfptr += jacSize * ndf * simd_t::width;
            }

            // Advance input by coordDim-1 componennts since have already
            // advanced one component in the above.
            if ((n + 1) % inblock.GetNumHomoModes() == 0)
            {
                inptr += (m_coordDim - 1) * inoffset;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
