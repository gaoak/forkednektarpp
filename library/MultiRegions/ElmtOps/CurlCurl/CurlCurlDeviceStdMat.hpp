///////////////////////////////////////////////////////////////////////////////
//
// File: CurlCurlDeviceStdMat.hpp
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
 * @file CurlCurlDeviceStdMat.hpp
 * @brief Device StdMat implementation of the curl-curl block operator.
 *
 * @details
 * The Device StdMat path forms the reference-space derivatives of the input
 * by multiplying with the dense standard-region physical-derivative
 * matrices, one matrix product per reference direction, and pointwise
 * kernels then apply the geometric derivative factors and take the curl
 * (see CurlCurlDeviceStdMatKernels.hpp); the two steps are run twice, for
 * omega = curl u and for out = curl omega. A multi-plane 3DH1 block instead
 * gets the plane part of a single curl on every plane, which CurlCurlOpImpl
 * completes with the z parts.
 *
 * Each set of products is issued as a single strided-batched GEMM through
 * NekBlas -- the vendor BLAS behind it depends on the build -- covering
 * every element and component of the block at once, so the matrices are
 * fetched untransposed (ePhysDerivStdMat) and the block data is read as one
 * column-major matrix: the implementation interleave width is one. All
 * device work is issued on the block's own stream, so different blocks may
 * overlap.
 *
 * This header provides the definition of detail::CurlCurlBlockOpImpl for
 * the Device execution space with the StdMat implementation tag; the
 * sum-factorised Device definitions live in CurlCurlDeviceSumFac.hpp. The
 * class template name is shared, and each CMake-generated registration
 * translation unit includes exactly one of these headers.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, execution space and
 * implementation; it should not normally be included by any other
 * translation unit. Other code goes through CurlCurlOp.hpp and the
 * operator factory: including this header directly instantiates the
 * whole template set in that translation unit, which is what the
 * generated sources exist to avoid.
 */

#pragma once

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp"
#include <MultiRegions/ElmtOps/CurlCurl/CurlCurlBlockOp.hpp>

#include <MultiRegions/ElmtOps/CurlCurl/CurlCurlDeviceStdMatKernels.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Device StdMat curl-curl block operator: dense standard-region
 * derivative matrices through the vendor BLAS, followed by pointwise curl
 * kernels, applied twice.
 *
 * @details
 * One v_Apply() pass
 * -# reshapes the block's #m_coordDim input components to the
 *    implementation interleave width (one) when the field is stored
 *    interleaved,
 * -# forms the reference-space derivatives of the whole block with one
 *    strided-batched GEMM -- the nqTot x nqTot standard-region derivative
 *    matrix of each reference direction against the block's data read as
 *    an nqTot x (components times elements) matrix -- writing one plane
 *    per direction and component into the static workspace,
 * -# calls Curl2DScalarStdMatKernel (2D) or Curl3DStdMatKernel (3D) to
 *    chain-rule those reference derivatives into Cartesian ones -- through
 *    the per-element or, for deformed geometry, per-point derivative
 *    factors -- and combine them into the vorticity omega, a single
 *    out-of-plane component in 2D and three components in 3D, also held in
 *    the workspace,
 * -# differentiates the components of omega with a second batched GEMM of
 *    the same form, and
 * -# calls Curl2DVectorStdMatKernel (2D) or Curl3DStdMatKernel (3D) again
 *    to write the output components curl(omega).
 *
 * The workspace holds #m_dimension * #m_coordDim reference-derivative
 * planes of nqTot * nelmt values -- plane d * coordDim + c is the
 * direction-d derivative of component c, the batched GEMM writing the
 * directions one after the other -- followed by one plane per component of
 * omega; the derivatives of omega overwrite those of u. The derivative
 * factors are fetched with the transpose flag set (the last argument of
 * DerivFactorKey). That flag only bites on deformed blocks, where it makes
 * each factor component contiguous over the element's quadrature points;
 * for regular geometry the warehouse ignores it and stores one value per
 * factor component per element, with no point index at all. The kernels
 * index the two cases differently -- see CurlCurlDeviceStdMatKernels.hpp.
 * A block with more than one homogeneous mode goes to ApplyCurl3DH1()
 * instead.
 *
 * @tparam ExecSpace      NektarSpaces::Device.
 * @tparam Implementation Implementation tag (StdMat).
 * @tparam TData          Floating-point type of the field data.
 *
 * @see CurlCurlDeviceSumFac.hpp for the sum-factorised Device path.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class CurlCurlBlockOpImpl : public CurlCurlBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Bind the operator to the block's device stream, capture the
     * element metadata and fetch the standard-region derivative matrices
     * and the block's derivative factors from the data warehouse.
     *
     * The curl-curl operator is only defined for two- and
     * three-dimensional expansions whose coordinate dimension equals the
     * shape dimension; both are asserted here.
     *
     * @param   block_idx       Index of the block; the stream used for
     *                          all device work is block_idx + 1.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    CurlCurlBlockOpImpl(const unsigned int block_idx,
                        const LocalRegions::ExpansionSharedPtr &exp,
                        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : CurlCurlBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        m_streamID = block_idx + 1;

        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
        m_dimension = exp->GetShapeDimension();
        m_coordDim  = exp->GetCoordim();
        m_nmTot     = exp->GetNcoeffs();
        m_nqTot     = exp->GetTotPoints();

        ASSERTL1(m_coordDim == 2 || m_coordDim == 3,
                 "CurlCurl operator only defined for 2D and 3D.");

        ASSERTL1(m_dimension == m_coordDim,
                 "Shape dimension and coordinate dimension are not the same.");

        // Fetch matrix.
        std::vector<LibUtilities::BasisKey> basisKeys(
            m_dimension, LibUtilities::NullBasisKey);
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            basisKeys[d] = exp->GetBasis(d)->GetBasisKey();
        }

        m_matptr = dataWarehouse->template GetData<MemSpace>(
            StdRegions::StdMatKey<TData>(basisKeys, m_shapeType,
                                         StdRegions::ePhysDerivStdMat));

        // Fetch derivative factor.
        m_dfptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::DerivFactorKey<TData>(block_idx,
                                                m_implInterleaveWidth, true));
    }

    /// Registration name for the block-operator factory, defined by the
    /// generated registration unit.
    static std::string className;

    /// @brief Creator function registered with the block-operator
    /// factory; builds one block operator for the given block of
    /// elements.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            CurlCurlBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    /// Element interleave width the GEMM and the kernels expect: the
    /// block data is read element by element, not interleaved.
    static constexpr unsigned int m_implInterleaveWidth = 1u;

    /// Device stream all work of this block operator is issued on, one
    /// per block (block index + 1).
    unsigned int m_streamID;
    /// Shape of the block's elements.
    LibUtilities::ShapeType m_shapeType;
    /// Whether the geometry is deformed (per-point derivative factors).
    bool m_isDeformed;
    /// Reference (shape) dimension of the elements (2 or 3).
    unsigned int m_dimension;
    /// Coordinate dimension of the elements; asserted equal to
    /// #m_dimension.
    unsigned int m_coordDim;
    /// Coefficients per element.
    unsigned int m_nmTot;
    /// Quadrature points per element.
    unsigned int m_nqTot;
    /// Standard-region PhysDeriv matrices, one nqTot x nqTot matrix per
    /// reference direction, concatenated.
    const TData *m_matptr;
    /// Derivative factors of the whole block, not interleaved and
    /// fetched with the transpose flag set: for a deformed block each
    /// factor component is contiguous over the element's points, while
    /// a regular block ignores the flag and stores one value per
    /// component per element (see DerivFactorKey).
    const TData *m_dfptr;

    /**
     * @brief Apply curl(curl u) to every element of the block (see the
     * class description for the algorithm).
     *
     * Every BLAS call, kernel launch and storage reshape is queued on
     * #m_streamID. Block storage held at a larger interleave width is
     * reshaped in place to width one before the first GEMM -- the
     * #m_coordDim input components in one pass -- and both the input and
     * the output components are reshaped back to the input's width once
     * the kernels have been queued; the output block's recorded width is
     * set to the input's on return.
     *
     * @param   inblock     Physical-space input block holding at least
     *                      #m_coordDim components of u, or the three
     *                      components on a multi-plane 3DH1 block.
     * @param   outblock    Physical-space output block receiving the
     *                      #m_coordDim components of curl(curl u), or the
     *                      plane part of one curl on a multi-plane 3DH1
     *                      block.
     */
    void v_Apply(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock) override
    {
        if (inblock.GetNumHomoModes() > 1)
        {
            ApplyCurl3DH1(inblock, outblock);
            return;
        }

        // Get BLAS handle.
        auto handle = NekBlas::Handle<ExecSpace>::GetInstance(m_streamID);

        // Get block sizes. omega = curl(u) is a scalar in 2D and a vector
        // in 3D.
        const auto nelmt          = inblock.GetNumElementsWithPadding();
        const unsigned int nOmega = (m_dimension == 2u) ? 1u : 3u;

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);

        // Get static workspace pointer. The standard derivatives of every
        // component are held at once so that the chain rule and the curl can
        // be applied by a single kernel.
        auto wspptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                (m_dimension * m_coordDim + nOmega) * m_nqTot * nelmt,
                m_streamID);
        auto derivptr = wspptr;
        auto omegaptr = derivptr + m_dimension * m_coordDim * m_nqTot * nelmt;

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
        const unsigned int gridSize =
            (nelmt * m_nqTot + blockSize - 1u) / blockSize;

        // Offsets between the components of a block.
        const auto outoffset   = outblock.CompSize();
        const auto derivoffset = m_nqTot * nelmt;

        // Reshape, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, interleaveWidth, m_coordDim * nelmt,
            inblock.GetNumData(), (TData *)inptr, m_streamID);

        // Standard derivatives of all the components.
        // Perform batched matrix-matrix multiply, one multiply per direction,
        // with the components held in the columns.
        NekBlas::GemmStridedBatched(
            handle, "N", "N", m_nqTot, m_coordDim * nelmt, m_nqTot, (TData)1.0,
            m_matptr, m_nqTot, m_nqTot * m_nqTot, inptr, m_nqTot, 0, (TData)0.0,
            derivptr, m_nqTot, m_coordDim * derivoffset, m_dimension);

        // omega = curl(u)
        if (m_dimension == 2u)
        {
            if (m_isDeformed)
            {
                DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (Curl2DScalarStdMatKernel<true>), gridSize, 1u, blockSize,
                    1, m_streamID, m_nqTot, nelmt, derivoffset, m_dfptr,
                    derivptr, omegaptr);
            }
            else
            {
                DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (Curl2DScalarStdMatKernel<false>), gridSize, 1u, blockSize,
                    1, m_streamID, m_nqTot, nelmt, derivoffset, m_dfptr,
                    derivptr, omegaptr);
            }
        }
        else
        {
            if (m_isDeformed)
            {
                DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (Curl3DStdMatKernel<true>), gridSize, blockSize, m_streamID,
                    m_nqTot, nelmt, derivoffset, derivoffset, m_dfptr, derivptr,
                    omegaptr);
            }
            else
            {
                DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (Curl3DStdMatKernel<false>), gridSize, blockSize,
                    m_streamID, m_nqTot, nelmt, derivoffset, derivoffset,
                    m_dfptr, derivptr, omegaptr);
            }
        }

        // Standard derivatives of omega.
        // Perform batched matrix-matrix multiply, one multiply per direction,
        // with the components held in the columns.
        NekBlas::GemmStridedBatched(
            handle, "N", "N", m_nqTot, nOmega * nelmt, m_nqTot, (TData)1.0,
            m_matptr, m_nqTot, m_nqTot * m_nqTot, omegaptr, m_nqTot, 0,
            (TData)0.0, derivptr, m_nqTot, nOmega * derivoffset, m_dimension);

        // out = curl(omega)
        if (m_dimension == 2u)
        {
            if (m_isDeformed)
            {
                DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (Curl2DVectorStdMatKernel<true>), gridSize, 1u, blockSize,
                    1, m_streamID, m_nqTot, nelmt, derivoffset, outoffset,
                    m_dfptr, derivptr, outptr);
            }
            else
            {
                DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (Curl2DVectorStdMatKernel<false>), gridSize, 1u, blockSize,
                    1, m_streamID, m_nqTot, nelmt, derivoffset, outoffset,
                    m_dfptr, derivptr, outptr);
            }
        }
        else
        {
            if (m_isDeformed)
            {
                DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (Curl3DStdMatKernel<true>), gridSize, blockSize, m_streamID,
                    m_nqTot, nelmt, derivoffset, outoffset, m_dfptr, derivptr,
                    outptr);
            }
            else
            {
                DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (Curl3DStdMatKernel<false>), gridSize, blockSize,
                    m_streamID, m_nqTot, nelmt, derivoffset, outoffset, m_dfptr,
                    derivptr, outptr);
            }
        }

        // Reshape back, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            interleaveWidth, m_implInterleaveWidth, m_coordDim * nelmt,
            inblock.GetNumData(), (TData *)inptr, m_streamID);
        LibUtilities::ReshapeStorage<ExecSpace>(
            interleaveWidth, m_implInterleaveWidth, m_coordDim * nelmt,
            outblock.GetNumData(), outptr, m_streamID);

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    /// \brief One curl's plane part on each plane of a 3DH1 block: three
    /// components in, three out.
    ///
    /// The curl of a 3DH1 field splits into a part that differentiates in the
    /// plane, \f$(\partial_y f_z, -\partial_x f_z,
    /// \partial_x f_y - \partial_y f_x)\f$, and a part that differentiates
    /// along z. This writes the first; the operator's z-op adds the second,
    /// and CurlCurlOpImpl takes the pair twice, the output carrying omega in
    /// between. Both halves of the plane part are already in the 2D kernels:
    /// the scalar one is the third component, taken from the derivatives of
    /// the first two, and the vector one is the first two components, taken
    /// from the derivatives of the third.
    ///
    /// The two kernels want different strides between directions, two
    /// components for the scalar and one for the vector, so the derivatives
    /// come from two multiplies rather than one. Every plane rides inside
    /// them: a component's planes are adjacent, so they simply lengthen the
    /// element count.
    void ApplyCurl3DH1(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock)
    {
        // A curl needs all three components whichever way round it is taken.
        constexpr unsigned int ncomp = 3u;

        ASSERTL1(inblock.GetNumComponents() == ncomp &&
                     outblock.GetNumComponents() == ncomp,
                 "The homogeneous curl needs all three components");

        // Get BLAS handle.
        auto handle = NekBlas::Handle<ExecSpace>::GetInstance(m_streamID);

        // Get block sizes.
        const auto nhomo    = inblock.GetNumHomoModes();
        const auto nelmt    = inblock.GetNumElementsWithPadding();
        const auto nelmtTot = nelmt * nhomo;

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);

        // Get static workspace pointer. The standard derivatives of every
        // component are held at once so that the chain rule and the curl can
        // be applied by a single kernel: the first two components in the
        // layout the scalar kernel reads, the third in the vector kernel's.
        const auto derivoffset = m_nqTot * nelmtTot;
        auto derivptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                ncomp * m_dimension * derivoffset, m_streamID);
        auto derivzptr = derivptr + 2u * m_dimension * derivoffset;

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
        const unsigned int gridSize =
            (nelmt * m_nqTot + blockSize - 1u) / blockSize;

        // Offsets between the components of a block: one component spans all
        // of the planes.
        const auto inoffset  = inblock.CompSize() * nhomo;
        const auto outoffset = outblock.CompSize() * nhomo;

        // Reshape, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, interleaveWidth, nelmtTot * ncomp,
            inblock.GetNumData(), (TData *)inptr, m_streamID);

        // Standard derivatives of the first two components, ordered as the
        // scalar kernel reads them.
        NekBlas::GemmStridedBatched(
            handle, "N", "N", m_nqTot, 2u * nelmtTot, m_nqTot, (TData)1.0,
            m_matptr, m_nqTot, m_nqTot * m_nqTot, inptr, m_nqTot, 0, (TData)0.0,
            derivptr, m_nqTot, 2u * derivoffset, m_dimension);

        // Standard derivatives of the third component, ordered as the vector
        // kernel reads them.
        NekBlas::GemmStridedBatched(
            handle, "N", "N", m_nqTot, nelmtTot, m_nqTot, (TData)1.0, m_matptr,
            m_nqTot, m_nqTot * m_nqTot, inptr + 2u * inoffset, m_nqTot, 0,
            (TData)0.0, derivzptr, m_nqTot, derivoffset, m_dimension);

        // The third component, df_y/dx - df_x/dy.
        if (m_isDeformed)
        {
            DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(
                (Curl2DScalarStdMatKernel<true>), gridSize, nhomo, blockSize, 1,
                m_streamID, m_nqTot, nelmt, derivoffset, m_dfptr, derivptr,
                outptr + 2u * outoffset);
        }
        else
        {
            DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(
                (Curl2DScalarStdMatKernel<false>), gridSize, nhomo, blockSize,
                1, m_streamID, m_nqTot, nelmt, derivoffset, m_dfptr, derivptr,
                outptr + 2u * outoffset);
        }

        // The first two components, {df_z/dy, -df_z/dx}.
        if (m_isDeformed)
        {
            DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(
                (Curl2DVectorStdMatKernel<true>), gridSize, nhomo, blockSize, 1,
                m_streamID, m_nqTot, nelmt, derivoffset, outoffset, m_dfptr,
                derivzptr, outptr);
        }
        else
        {
            DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(
                (Curl2DVectorStdMatKernel<false>), gridSize, nhomo, blockSize,
                1, m_streamID, m_nqTot, nelmt, derivoffset, outoffset, m_dfptr,
                derivzptr, outptr);
        }

        // Reshape back, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            interleaveWidth, m_implInterleaveWidth, nelmtTot * ncomp,
            inblock.GetNumData(), (TData *)inptr, m_streamID);
        LibUtilities::ReshapeStorage<ExecSpace>(
            interleaveWidth, m_implInterleaveWidth, nelmtTot * ncomp,
            outblock.GetNumData(), outptr, m_streamID);

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::MultiRegions::detail
