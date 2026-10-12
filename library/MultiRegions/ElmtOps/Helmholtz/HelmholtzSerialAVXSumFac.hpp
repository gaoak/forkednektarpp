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

/**
 * @file HelmholtzSerialAVXSumFac.hpp
 * @brief Serial/AVX sum-factorised (SumFac) implementation of the
 * per-block weak Helmholtz operator.
 *
 * @details
 * Instead of the dense standard-region matrices of the StdMat path, the
 * operator is assembled from the one-dimensional tensor-product stages
 * of the simpler families -- the backward transform
 * (BwdTransKernelLauncher), the tensor derivative
 * (PhysDerivTensor{1,2,3}DKernel), the diffusion metric of the Helmholtz
 * family (DiffusionCoeffwithWJ{1,2,3}DKernel), the summed transposed
 * derivative (SumDerivTensor{1,2,3}DKernel) and the inner product
 * (IProductWRTBaseKernelLauncher). The HelmholtzKernelLauncher
 * overloads in HelmholtzSerialAVXSumFacKernels.hpp string the stages
 * together for one element group; each SIMD lane owns one element, so a
 * group of simd_t::width elements advances in lock-step (width one on
 * the Serial execution space).
 *
 * v_Apply() dispatches on the block's shape type to the ShapeBlock()
 * specialisations, whose definitions CMake generates from
 * LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in, one
 * translation unit per shape: each wraps OperatorND() in a switch over
 * the element sizes so that compile-time constants reach the kernels,
 * falling back to a runtime-sized call for other sizes.
 *
 * This header provides the definition of detail::HelmholtzBlockOpImpl
 * for the Serial and AVX execution spaces with the SumFac implementation
 * tag; see HelmholtzSerialAVXStdMat.hpp for the StdMat sibling and the
 * note on the one-definition-per-translation-unit scheme.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, execution space and
 * implementation; it should not normally be included by any other
 * translation unit. Other code goes through HelmholtzOp.hpp and the
 * operator factory: including this header directly instantiates the
 * whole template set in that translation unit, which is what the
 * generated sources exist to avoid.
 */
#pragma once

#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include <MultiRegions/ElmtOps/Helmholtz/HelmholtzBlockOp.hpp>

#include <MultiRegions/ElmtOps/Helmholtz/HelmholtzSerialAVXSumFacKernels.hpp>

// Selects the switch construction used by the generated ShapeBlock
// definitions (see LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in).
#define NEKTAR_BLOCKOP_SWITCH_CODE

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Serial/AVX sum-factorised block implementation of the weak
 * Helmholtz operator, one element per SIMD lane.
 *
 * @details
 * The constructor caches, per direction, the 1D basis table, the 1D
 * collocation derivative matrix and the 1D quadrature weights, plus the
 * collapsed-coordinate factors #m_f, the nodal-to-modal matrices of the
 * nodal shapes, and the block's Jacobians and derivative factors -- all
 * from the data warehouse, so they are built once and shared between
 * operators. The factor tables are fetched for every 2D block (two of
 * them) and every 3D block (four of them) whatever the shape, since the
 * warehouse keys are uniform.
 *
 * @tparam ExecSpace      NektarSpaces::Serial or NektarSpaces::AVX.
 * @tparam Implementation Implementation tag (SumFac).
 * @tparam TData          Floating-point type of the field data.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class HelmholtzBlockOpImpl : public HelmholtzBlockOp<TData>
{
    using BlockOpBase = HelmholtzBlockOp<TData>;
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Construct the operator, capture element metadata, and
     * fetch 1D basis data, quadrature weights and geometric factors
     * from the data warehouse, and size the kernel workspaces.
     *
     * @param   block_idx       Index of the block within the expansion
     *                          list's Collections.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    HelmholtzBlockOpImpl(const unsigned int block_idx,
                         const LocalRegions::ExpansionSharedPtr &exp,
                         LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : HelmholtzBlockOp<TData>(block_idx, exp, dataWarehouse)
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

        if ((m_shapeType == LibUtilities::NodalTri) ||
            (m_shapeType == LibUtilities::NodalPrism) ||
            (m_shapeType == LibUtilities::NodalTet))
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
            m_nodToMod = dataWarehouse->template GetData<MemSpace>(
                StdRegions::StdMatKey<simd_t>(basisKeys, m_shapeType,
                                              StdRegions::eNodalToModal,
                                              nodalType));
            m_nodToModTrans = dataWarehouse->template GetData<MemSpace>(
                StdRegions::StdMatKey<simd_t>(
                    basisKeys, m_shapeType, StdRegions::eNodalToModalTranspose,
                    nodalType));
        }
        else
        {
            m_nodToMod      = (const simd_t *)nullptr;
            m_nodToModTrans = (const simd_t *)nullptr;
        }

        // Fetch Jacobian and deriv factors.
        m_jacptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::JacobianKey<TData>(block_idx, m_implInterleaveWidth));
        m_dfptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::DerivFactorKey<TData>(block_idx,
                                                m_implInterleaveWidth, false));

        // Workspace for kernels - also checks preconditions.
        if (m_dimension == 2)
        {
            unsigned int wsp0Size = 0;
            BwdTrans2DWorkspace(m_shapeType, m_nm[0], m_nm[1], m_nq[0], m_nq[1],
                                wsp0Size);
            IProduct2DWorkspace(m_shapeType, m_nm[0], m_nm[1], m_nq[0], m_nq[1],
                                wsp0Size);
            m_wsp.push_back(
                std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp0Size));
        }
        else if (m_dimension == 3)
        {
            unsigned int wsp0Size = 0, wsp1Size = 0, wsp2Size = 0;
            BwdTrans3DWorkspace(m_shapeType, m_nm[0], m_nm[1], m_nm[2], m_nq[0],
                                m_nq[1], m_nq[2], wsp0Size, wsp1Size);
            IProduct3DWorkspace(m_shapeType, m_nm[0], m_nm[1], m_nm[2], m_nq[0],
                                m_nq[1], m_nq[2], wsp0Size, wsp1Size, wsp2Size);
            m_wsp.push_back(
                std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp0Size));
            m_wsp.push_back(
                std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp1Size));
            m_wsp.push_back(
                std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp2Size));
        }

        // One tensorial derivative per direction, plus the backward
        // transform of the input.
        unsigned int nqTot = 1;
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            nqTot *= m_nq[d];
        }
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            m_deriv.push_back(
                std::vector<simd_t, tinysimd::allocator<simd_t>>(nqTot));
        }
        m_bwd = std::vector<simd_t, tinysimd::allocator<simd_t>>(nqTot);
    }

    /// Block-operator class name; defined by generated factory code.
    static std::string className;

    /// Creator function registered with BlockOperatorFactory.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Coeff, FieldState::Coeff, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            HelmholtzBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    /// Vector width this implementation operates at (simd_t::width).
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

    /// Shape type of the element block.
    LibUtilities::ShapeType m_shapeType;
    /// Whether the block's geometry is deformed; selects the branch of
    /// OperatorNDImpl and the geometry layout it reads.
    bool m_isDeformed;
    /// Whether the basis is modified; selects the collapsed-coordinate
    /// chain-rule correction.
    bool m_isModified;
    /// Dimension of the reference element (1, 2 or 3).
    unsigned int m_dimension;
    /// Coordinate dimension of the (possibly embedded) element.
    unsigned int m_coordDim;
    /// Coefficients per direction.
    std::vector<unsigned int> m_nm;
    /// Quadrature points per direction.
    std::vector<unsigned int> m_nq;
    /// Basis matrix per direction.
    std::vector<const TData *> m_B;
    /// Basis-derivative matrix per direction.
    std::vector<const TData *> m_D;
    /// Quadrature weights per direction.
    std::vector<const TData *> m_W;
    /// Collapsed-coordinate chain-rule factors used by the kernels.
    std::vector<const simd_t *> m_f;
    /// Kernel workspaces, sized per NumWorkspace(#m_dimension).
    std::vector<std::vector<simd_t, tinysimd::allocator<simd_t>>> m_wsp;
    /// Per-direction reference-derivative workspace.
    std::vector<std::vector<simd_t, tinysimd::allocator<simd_t>>> m_deriv;
    /// Backward-transform workspace.
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_bwd;
    /// Empty placeholder passed where a diffusion tensor is not used.
    std::vector<TData> NullTDataVector;
    /// Nodal-to-modal conversion matrix; null for tensor-product shapes.
    const simd_t *m_nodToMod;
    /// Transposed nodal-to-modal conversion matrix; null for
    /// tensor-product shapes.
    const simd_t *m_nodToModTrans;
    /// Jacobian of the block, interleaved at #m_implInterleaveWidth:
    /// one value per element, or one per quadrature point and element
    /// on a deformed block.
    const TData *m_jacptr;
    /// Derivative factors of the block, interleaved and untransposed:
    /// `coordDim * dimension` values per element, or per quadrature
    /// point and element on a deformed block.
    const TData *m_dfptr;
#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
    // flag to ensure we only get one warning for alignment otherwise CI system
    // is saturated with warnings
    bool m_warnOnce = false;
#endif

    /// Apply the weak Helmholtz operator to every element of the block
    /// by sum factorisation; dispatches on #m_shapeType to ShapeBlock().
    void v_Apply(LibUtilities::BlockAccessor<TData, FieldState::Coeff> &inblock,
                 LibUtilities::BlockAccessor<TData, FieldState::Coeff>
                     &outblock) override
    {
        // Check alignment.
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
                ShapeBlock<LibUtilities::Seg>(inblock, outblock);
                break;
            }
            // Quadrilateral
            case LibUtilities::Quad:
            {
                ShapeBlock<LibUtilities::Quad>(inblock, outblock);
                break;
            }
            // Triangle
            case LibUtilities::Tri:
            {
                ShapeBlock<LibUtilities::Tri>(inblock, outblock);
                break;
            }
            // Nodal triangle
            case LibUtilities::NodalTri:
            {
                ShapeBlock<LibUtilities::NodalTri>(inblock, outblock);
                break;
            }
            // Hexahedron
            case LibUtilities::Hex:
            {
                ShapeBlock<LibUtilities::Hex>(inblock, outblock);
                break;
            }
            // Tetrahedron
            case LibUtilities::Tet:
            {
                ShapeBlock<LibUtilities::Tet>(inblock, outblock);
                break;
            }
            // Nodal tetrahedron
            case LibUtilities::NodalTet:
            {
                ShapeBlock<LibUtilities::NodalTet>(inblock, outblock);
                break;
            }
            // Pyramid
            case LibUtilities::Pyr:
            {
                ShapeBlock<LibUtilities::Pyr>(inblock, outblock);
                break;
            }
            // Prism
            case LibUtilities::Prism:
            {
                ShapeBlock<LibUtilities::Prism>(inblock, outblock);
                break;
            }
            // Nodal prism
            case LibUtilities::NodalPrism:
            {
                ShapeBlock<LibUtilities::NodalPrism>(inblock, outblock);
                break;
            }
            default:
                std::cout << "shapetype not implemented" << std::endl;
        }
    }

    /// Shape-specific block operator, specialised for each shape in
    /// LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in.
    template <LibUtilities::ShapeType SHAPE_TYPE>
    void ShapeBlock(typename BlockOpBase::InBlock &inblock,
                    typename BlockOpBase::OutBlock &outblock);

    /// Number of workspaces used by the kernels in dim dimensions.
    static constexpr unsigned int NumWorkspace(const unsigned int dim)
    {
        return (dim == 1) ? 0 : (dim == 2) ? 1 : 3;
    }

    /// Number of collapsed coordinate factors used by the kernels in
    /// dim dimensions.
    static constexpr unsigned int NumFactor(const unsigned int dim)
    {
        return (dim == 1) ? 0 : (dim == 2) ? 2 : 4;
    }

    /// Generic operator. Builds the index sequences from the shape
    /// dimension and forwards to OperatorNDImpl.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TSizeParameter>
    NEK_FORCE_INLINE void OperatorND(
        LibUtilities::BlockAccessor<TData, FieldState::Coeff> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Coeff> &outblock,
        TSizeParameter sizeParam)
    {
        constexpr unsigned int DIM = LibUtilities::ShapeTypeDimMap[SHAPE_TYPE];

        // sizeParam is built by the switch in
        // LibUtilities/BasicUtils/Switch/BlockOpSwitchCode.h.in.
        static_assert((DIM == 1 && IsSizeParameter1D_v<TSizeParameter>) ||
                          (DIM == 2 && IsSizeParameter2D_v<TSizeParameter>) ||
                          (DIM == 3 && IsSizeParameter3D_v<TSizeParameter>),
                      "OperatorND expects a size parameter matching the "
                      "dimension of the shape.");

        OperatorNDImpl<SHAPE_TYPE, DEFORMED>(
            inblock, outblock, sizeParam,
            std::make_integer_sequence<unsigned int, DIM>(),
            std::make_integer_sequence<unsigned int, NumWorkspace(DIM)>(),
            std::make_integer_sequence<unsigned int, NumFactor(DIM)>());
    }

    /// Generic operator implementation. ind0 indexes each direction,
    /// ind1 the workspaces used by the kernels,
    /// ind2 the collapsed coordinate factors used by the kernels.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TSizeParameter, unsigned int... ind0,
              unsigned int... ind1, unsigned int... ind2>
    NEK_FORCE_INLINE void OperatorNDImpl(
        LibUtilities::BlockAccessor<TData, FieldState::Coeff> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Coeff> &outblock,
        TSizeParameter sizeParam, std::integer_sequence<unsigned int, ind0...>,
        std::integer_sequence<unsigned int, ind1...>,
        std::integer_sequence<unsigned int, ind2...>)
    {
        constexpr unsigned int ndim = sizeof...(ind0);

        // Shape size.
        const auto nmTot                  = sizeParam.nmTot();
        [[maybe_unused]] const auto nqTot = sizeParam.nqTot();

        const unsigned int ndf = ndim * m_coordDim;
        unsigned int dfsize    = 1;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto diffCoeffPtr =
            this->m_diffCoeff.template GetPtr<MemSpace, ReadOnly>();

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
            auto jacptr = m_jacptr;
            auto dfptr  = m_dfptr;

            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        m_implInterleaveWidth, interleaveWidth, chunkSize,
                        nmTot, (TData *)inptr);
                }

                // Helmholtz kernel.
                HelmholtzKernelLauncher<SHAPE_TYPE, DEFORMED>(
                    sizeParam, m_isModified, m_coordDim, diffCoeffPtr,
                    NullTDataVector, m_B[ind0]..., m_D[ind0]..., m_W[ind0]...,
                    m_f[ind2]..., m_nodToMod, m_nodToModTrans,
                    reinterpret_cast<const simd_t *>(jacptr),
                    reinterpret_cast<const simd_t *>(dfptr),
                    m_wsp[ind1].data()..., m_deriv[ind0].data()...,
                    m_bwd.data(), reinterpret_cast<const simd_t *>(inptr),
                    reinterpret_cast<simd_t *>(outptr), this->m_lambda);

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        nmTot,
                        (TData *)inptr -
                            (width_ratio - 1) * nmTot * simd_t::width);
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        nmTot,
                        (TData *)outptr -
                            (width_ratio - 1) * nmTot * simd_t::width);
                }

                // Increment pointers for the next elmt group.
                dfptr += dfsize * ndf * simd_t::width;
                jacptr += dfsize * simd_t::width;
                inptr += nmTot * simd_t::width;
                outptr += nmTot * simd_t::width;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::MultiRegions::detail
