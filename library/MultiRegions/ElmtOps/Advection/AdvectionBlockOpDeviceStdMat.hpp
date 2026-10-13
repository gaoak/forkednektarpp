///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionBlockOpDeviceStdMat.hpp
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
// Description: Device StdMat implementation of the advection block operator.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file AdvectionBlockOpDeviceStdMat.hpp
 * @brief Device StdMat implementation of the advection block operator.
 *
 * @details
 * The Device StdMat path takes two steps: the reference-space derivatives
 * of the input are formed by multiplying with the dense standard-region
 * physical-derivative matrices, one matrix product per reference
 * direction, and a pointwise kernel then applies the geometric derivative
 * factors and the dot product with the advection velocity (see
 * AdvectionBlockOpDeviceStdMatKernels.hpp).
 *
 * The products are issued as a single strided-batched GEMM through NekBlas
 * -- the vendor BLAS behind it depends on the build -- covering every
 * element, component and homogeneous mode of the block at once, so the
 * matrices are fetched untransposed (ePhysDerivStdMat) and the block data
 * is read as one column-major matrix: the implementation interleave width
 * is one. All device work is issued on the block's own stream, so
 * different blocks may overlap.
 *
 * This header provides the definition of detail::AdvectionBlockOpImpl for
 * the Device execution space with the StdMat implementation tag; the
 * sum-factorised Device definitions live in AdvectionBlockOpDeviceSumFac.hpp.
 * The class template name is shared, and each CMake-generated registration
 * translation unit includes exactly one of these headers.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, execution space and
 * implementation; it should not normally be included by any other
 * translation unit. Other code goes through AdvectionOp.hpp and the
 * operator factory: including this header directly instantiates the
 * whole template set in that translation unit, which is what the
 * generated sources exist to avoid.
 */

#pragma once

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp"
#include <MultiRegions/ElmtOps/Advection/AdvectionBlockOp.hpp>

#include <MultiRegions/ElmtOps/Advection/AdvectionBlockOpDeviceStdMatKernels.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Device StdMat advection block operator: dense standard-region
 * derivative matrices through the vendor BLAS, followed by a pointwise
 * derivative-factor and velocity kernel.
 *
 * @details
 * One v_Apply() pass
 * -# reshapes the block's storage to the implementation interleave width
 *    (one) when the field is stored interleaved,
 * -# forms the reference-space derivatives of the whole block with one
 *    strided-batched GEMM -- the nqTot x nqTot standard-region derivative
 *    matrix of each reference direction against the block's data read as
 *    an nqTot x (elements times components times homogeneous modes)
 *    matrix -- writing one plane per direction into the static
 *    workspace, and
 * -# launches MultiplyByDerivFactorAndAdvecVelKernel once, with the
 *    components and homogeneous modes on the second grid dimension, to
 *    chain-rule those reference derivatives into Cartesian ones --
 *    through the per-element or, for deformed geometry, per-point
 *    derivative factors -- and contract them with the advection velocity.
 *
 * The derivative factors are fetched with the transpose flag set (the last
 * argument of DerivFactorKey). That flag only bites on deformed blocks,
 * where it makes each factor component contiguous over the element's
 * quadrature points; for regular geometry the warehouse ignores it and
 * stores one value per factor component per element, with no point index
 * at all. The kernel indexes the two cases differently -- see
 * MultiplyByDerivFactorAndAdvecVelKernel in
 * AdvectionBlockOpDeviceStdMatKernels.hpp. The velocity registered by
 * SetAdvVel() is reshaped to the implementation width the first time v_Apply()
 * finds it at another width, and is then read in place; each homogeneous mode
 * reads its own plane of the velocity, and the same velocity serves every
 * component.
 *
 * @tparam ExecSpace      NektarSpaces::Device.
 * @tparam Implementation Implementation tag (StdMat).
 * @tparam TData          Floating-point type of the field data.
 *
 * @see AdvectionBlockOpDeviceSumFac.hpp for the sum-factorised Device path.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class AdvectionBlockOpImpl : public AdvectionBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Bind the operator to the block's device stream, capture the
     * element metadata and fetch the standard-region derivative matrices
     * and the block's derivative factors from the data warehouse.
     *
     * @param   block_idx       Index of the block; the stream used for
     *                          all device work is block_idx + 1.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    AdvectionBlockOpImpl(const unsigned int block_idx,
                         const LocalRegions::ExpansionSharedPtr &exp,
                         LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : AdvectionBlockOp<TData>(block_idx, exp, dataWarehouse)
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
            AdvectionBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    /// Element interleave width the GEMM and the kernel expect: the
    /// block data is read element by element, not interleaved.
    static constexpr unsigned int m_implInterleaveWidth = 1u;

    /// Device stream all work of this block operator is issued on, one
    /// per block (block index + 1).
    unsigned int m_streamID;
    /// Shape of the block's elements.
    LibUtilities::ShapeType m_shapeType;
    /// Whether the geometry is deformed (per-point derivative factors).
    bool m_isDeformed;
    /// Reference (shape) dimension of the elements.
    unsigned int m_dimension;
    /// Coordinate dimension of the (possibly embedded) elements.
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
     * @brief Apply scale * (V . grad u) to every element of the block (see
     * the class description for the algorithm).
     *
     * Every BLAS call, kernel launch and storage reshape is queued on
     * #m_streamID. Block storage held at a larger interleave width is
     * reshaped in place to width one before the GEMM -- the input always,
     * the output only when it is accumulated onto (append set) -- and
     * both are reshaped back to the input's width once the kernels have
     * been queued; the output block's recorded width is set to the
     * input's on return. Homogeneous modes are not iterated over
     * separately: they extend the column dimension of the GEMM and share
     * the kernel's second grid dimension with the components.
     *
     * @param   inblock     Physical-space input block.
     * @param   outblock    Physical-space output block; overwritten, or
     *                      accumulated onto when append is set.
     */
    void v_Apply(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock) override
    {
        // Reshape advection velocity, if necessary.
        if (this->m_advVel->GetInterleaveWidth() != m_implInterleaveWidth)
        {
            auto advVelPtr =
                this->m_advVel->template GetPtr<MemSpace, ReadWrite>(
                    m_streamID);
            LibUtilities::ReshapeStorage<ExecSpace>(
                m_implInterleaveWidth, this->m_advVel->GetInterleaveWidth(),
                this->m_advVel->GetNumElementsWithPadding() *
                    this->m_advVel->GetNumComponents() *
                    this->m_advVel->GetNumHomoModes(),
                this->m_advVel->GetNumData(), advVelPtr, m_streamID);
            this->m_advVel->template SetInterleaveWidth<TData>(
                m_implInterleaveWidth);
        }

        // Get BLAS handle.
        auto handle = NekBlas::Handle<ExecSpace>::GetInstance(m_streamID);

        // Get block sizes.
        const auto nhomo = inblock.GetNumHomoModes();
        const auto ncomp = inblock.GetNumComponents() * nhomo;
        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr =
            (this->m_append)
                ? outblock.template GetPtr<MemSpace, ReadWrite>(m_streamID)
                : outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);
        auto advVelPtr =
            this->m_advVel->template GetPtr<MemSpace, ReadOnly>(m_streamID);

        // Get static workspace pointer.
        auto derivptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                m_dimension * nelmt * ncomp * m_nqTot, m_streamID);

        // Get interleave parameter.
        const auto inInterleaveWidth  = inblock.GetInterleaveWidth();
        const auto outInterleaveWidth = outblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
        const unsigned int gridSize =
            (nelmt * m_nqTot + blockSize - 1u) / blockSize;

        // Offsets between the components of a block. The derivatives of every
        // component are held at once, so the offset between two directions
        // spans all of them, and the advection velocity carries the same
        // planes as the input, so one of its components spans all of them.
        const auto advelsize =
            m_nqTot * inblock.GetNumElmtGroups(m_implInterleaveWidth) * nhomo;
        const auto derivoffset = m_nqTot * nelmt * ncomp;

        // Reshape, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, inInterleaveWidth, nelmt * ncomp,
            inblock.GetNumData(), (TData *)inptr, m_streamID);
        if (this->m_append)
        {
            LibUtilities::ReshapeStorage<ExecSpace>(
                m_implInterleaveWidth, outInterleaveWidth, nelmt * ncomp,
                outblock.GetNumData(), (TData *)outptr, m_streamID);
        }

        // Perform batched matrix-matrix multiply, one multiply per direction,
        // with the components and homogeneous modes held in the columns.
        NekBlas::GemmStridedBatched(
            handle, "N", "N", m_nqTot, nelmt * ncomp, m_nqTot, (TData)1.0,
            m_matptr, m_nqTot, m_nqTot * m_nqTot, inptr, m_nqTot, 0, (TData)0.0,
            derivptr, m_nqTot, derivoffset, m_dimension);

        // Multiply by derivative factor, the variables and planes together on
        // the second grid dimension.
        if (m_isDeformed)
        {
            if (this->m_append)
            {
                DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (MultiplyByDerivFactorAndAdvecVelKernel<true, true>),
                    gridSize, ncomp, blockSize, 1, m_streamID, m_nqTot,
                    m_coordDim, m_dimension, nelmt, derivoffset, m_dfptr,
                    advVelPtr, advelsize, nhomo, derivptr, outptr,
                    this->m_scale);
            }
            else
            {
                DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (MultiplyByDerivFactorAndAdvecVelKernel<false, true>),
                    gridSize, ncomp, blockSize, 1, m_streamID, m_nqTot,
                    m_coordDim, m_dimension, nelmt, derivoffset, m_dfptr,
                    advVelPtr, advelsize, nhomo, derivptr, outptr,
                    this->m_scale);
            }
        }
        else
        {
            if (this->m_append)
            {
                DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (MultiplyByDerivFactorAndAdvecVelKernel<true, false>),
                    gridSize, ncomp, blockSize, 1, m_streamID, m_nqTot,
                    m_coordDim, m_dimension, nelmt, derivoffset, m_dfptr,
                    advVelPtr, advelsize, nhomo, derivptr, outptr,
                    this->m_scale);
            }
            else
            {
                DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (MultiplyByDerivFactorAndAdvecVelKernel<false, false>),
                    gridSize, ncomp, blockSize, 1, m_streamID, m_nqTot,
                    m_coordDim, m_dimension, nelmt, derivoffset, m_dfptr,
                    advVelPtr, advelsize, nhomo, derivptr, outptr,
                    this->m_scale);
            }
        }

        // Reshape back, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            inInterleaveWidth, m_implInterleaveWidth, nelmt * ncomp,
            inblock.GetNumData(), (TData *)inptr, m_streamID);
        LibUtilities::ReshapeStorage<ExecSpace>(
            inInterleaveWidth, m_implInterleaveWidth, nelmt * ncomp,
            outblock.GetNumData(), outptr, m_streamID);

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
    }
};

} // namespace Nektar::MultiRegions::detail
