///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseMatFree.hpp
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

#include <LibUtilities/BasicUtils/ShapeType.hpp>
#include <LibUtilities/BasicUtils/SharedArray.hpp>
#include <LibUtilities/Foundations/Basis.h>
#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "IProductWRTBaseMatFreeKernels.hpp"
#include "Operators/OperatorIProductWRTBase.hpp"

namespace Nektar::Operators::detail
{

using vec_t = tinysimd::simd<double>;

// Matrix-free implementation
template <typename TData>
class OperatorIProductWRTBaseImpl<TData, ImplMatFree>
    : public OperatorIProductWRTBase<TData>
{
public:
    OperatorIProductWRTBaseImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorIProductWRTBase<TData>(expansionList)
    {
        // Initialise jacobian with paddings
        auto blocks =
            GetBlockAttributes(FieldState::Phys, expansionList, vec_t::width);
        size_t jacSize = Operator<TData>::GetVectorizedGeomFactorSize(blocks);
        m_jac = Operator<TData>::SetVectorizedJacobian(jacSize, blocks);
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out,
               [[maybe_unused]] const TData lambda) override
    {
        // check alignment
        if (in.GetAlignment() != vec_t::alignment)
        {
            NEKERROR(ErrorUtil::efatal,
                     "Input Field are not aligned to the required alignment "
                     "for the SIMD vector type.");
        }
        if (out.GetAlignment() != vec_t::alignment)
        {
            NEKERROR(ErrorUtil::efatal,
                     "Output Field are not aligned to the required alignment "
                     "for the SIMD vector type.");
        }
        // Reshape into vec_t::width. If the Field is already
        // interleaved, this method returns.
        in.template ReshapeStorage<vec_t::width>();
        out.template ReshapeStorage<vec_t::width>();

        TData *inptr  = in.GetStorage().GetCPUPtr();
        TData *outptr = out.GetStorage().GetCPUPtr();

        //---debug-----
        // std::cout << "Print Vectorized Jacobian:" << std::endl;
        // size_t jac_idx = 0;

        size_t exp_idx = 0; // accumulates over blocks, not used in operatorND()
        m_jac_idx      = 0; // accumulates over blocks, accessed in operatorND()

        for (size_t block_idx = 0; block_idx < in.GetBlocks().size();
             ++block_idx)
        {
            auto const expPtr    = this->m_expansionList->GetExp(exp_idx);
            auto const &inblock  = in.GetBlocks()[block_idx];
            auto const &outblock = out.GetBlocks()[block_idx];
            auto const shapeType = expPtr->DetShapeType();
            auto const dimension = expPtr->GetShapeDimension();
            auto const deformed  = expPtr->GetMetricInfo()->GetGtype() ==
                                  SpatialDomains::eDeformed;
            m_nElmtGroup =
                (inblock.num_elements + inblock.num_padding_elements) /
                vec_t::width;

            m_basis.resize(dimension);
            for (int i = 0; i < dimension; ++i)
            {
                // prepare bdata
                m_basis[i]    = expPtr->GetBasis(i);
                auto bdataRAW = expPtr->GetBasis(i)->GetBdata();
                m_B[i].resize(bdataRAW.size());
                for (auto j = 0; j < bdataRAW.size(); ++j)
                {
                    m_B[i][j] = bdataRAW[j];
                }
                // prepare weights
                NekDouble fac = 1.0;
                if (m_basis[i]->GetPointsType() ==
                    LibUtilities::eGaussRadauMAlpha1Beta0)
                {
                    fac = 0.5;
                }
                else if (m_basis[i]->GetPointsType() ==
                         LibUtilities::eGaussRadauMAlpha2Beta0)
                {
                    fac = 0.25;
                }
                const auto w = m_basis[i]->GetW();
                m_w[i].resize(w.size());
                for (auto j = 0; j < w.size(); ++j)
                {
                    m_w[i][j] = fac * w[j];
                }
            }

            //---debug-----
            /*
            std::cout << "block_idx: " << block_idx << std::endl;
            if (expPtr->GetMetricInfo()->GetGtype() ==
                SpatialDomains::eDeformed)
            {
                alignas(vec_t::alignment) NekDouble tmp[vec_t::width];
                for (size_t e = 0; e < m_nElmtGroup; ++e)
                {
                    for (size_t pt = 0; pt < expPtr->GetTotPoints(); ++pt)
                    {
                        (*m_jac)[jac_idx++].store(&tmp[0]);
                        for (auto g : tmp)
                        {
                            std::cout << g << " ";
                        }
                    }
                }
                std::cout << std::endl;
            }
            else
            {
                alignas(vec_t::alignment) NekDouble tmp[vec_t::width];
                for (size_t e = 0; e < m_nElmtGroup; ++e)
                {
                    (*m_jac)[jac_idx++].store(&tmp[0]);
                    for (auto g : tmp)
                    {
                        std::cout << g << " ";
                    }
                }
                std::cout << std::endl;
            }
            */

#include "SwitchNodesPoints.h"

            inptr += inblock.block_size;
            outptr += outblock.block_size;
            exp_idx += inblock.num_elements;
        }
    }

    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorIProductWRTBaseImpl<TData, ImplMatFree>>(expansionList);
    }

    static std::string className;

private:
    std::vector<LibUtilities::BasisSharedPtr> m_basis;
    std::array<std::vector<vec_t, tinysimd::allocator<vec_t>>, 3> m_B;
    int m_nElmtGroup;
    int m_jac_idx;
    std::shared_ptr<std::vector<vec_t, tinysimd::allocator<vec_t>>> m_jac;
    std::array<std::vector<vec_t, tinysimd::allocator<vec_t>>, 3> m_w;

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void operator1D(const NekDouble *input, NekDouble *output)
    {
        const auto nm0   = m_basis[0]->GetNumModes();
        const auto nq0   = m_basis[0]->GetNumPoints();
        const auto nmTot = nm0;
        const auto nqTot = nq0;
        // const auto nqBlocks = nqTot * vec_t::width;
        // const auto nmBlocks = m_nmTot * vec_t::width;

        // std::vector<vec_t, allocator<vec_t>> tmpIn(nqTot), tmpOut(m_nmTot);
        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);
        vec_t::scalarType *tmpOut =
            reinterpret_cast<vec_t::scalarType *>(output);

        vec_t *jac_ptr;

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            if constexpr (DEFORMED)
            {
                jac_ptr = &((*this->m_jac)[m_jac_idx]);
                m_jac_idx += nq0;
            }
            else
            {
                jac_ptr = &((*this->m_jac)[m_jac_idx++]);
            }

            // Load and transpose data
            // load_interleave(inptr, nqTot, tmpIn);

            IProduct1DKernel<SHAPE_TYPE, false, false, DEFORMED>(
                nm0, nq0, tmpIn, this->m_B[0], this->m_w[0], jac_ptr, tmpOut);

            // de-interleave and store data
            // deinterleave_store(tmpOut, m_nmTot, outptr);

            tmpIn += nqTot;
            tmpOut += nmTot * vec_t::width;
        }
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, int nm0,
              int nq0>
    void operator1D(const NekDouble *input, NekDouble *output)
    {
        constexpr auto nmTot = nm0;
        constexpr auto nqTot = nq0;
        // constexpr auto nqBlocks = nqTot * vec_t::width;
        // const auto nmBlocks     = m_nmTot * vec_t::width;

        // std::vector<vec_t, allocator<vec_t>> tmpIn(nqTot), tmpOut(m_nmTot);
        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);
        vec_t::scalarType *tmpOut =
            reinterpret_cast<vec_t::scalarType *>(output);

        vec_t *jac_ptr;

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            if constexpr (DEFORMED)
            {
                jac_ptr = &((*this->m_jac)[m_jac_idx]);
                m_jac_idx += nq0;
            }
            else
            {
                jac_ptr = &((*this->m_jac)[m_jac_idx++]);
            }

            // Load and transpose data
            // load_interleave(inptr, nqTot, tmpIn);

            IProduct1DKernel<SHAPE_TYPE, false, false, DEFORMED>(
                nm0, nq0, tmpIn, this->m_B[0], this->m_w[0], jac_ptr, tmpOut);

            // de-interleave and store data
            // deinterleave_store(tmpOut, m_nmTot, outptr);

            tmpIn += nqTot;
            tmpOut += nmTot * vec_t::width;
        }
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void operator2D(const NekDouble *input, NekDouble *output)
    {
        const auto nm0 = m_basis[0]->GetNumModes();
        const auto nm1 = m_basis[1]->GetNumModes();

        const auto nq0 = m_basis[0]->GetNumPoints();
        const auto nq1 = m_basis[1]->GetNumPoints();

        const auto nqTot = nq0 * nq1;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
        // const auto nqBlocks = nqTot * vec_t::width;
        // const auto nmBlocks = m_nmTot * vec_t::width;

        // auto *inptr  = &input[0];
        // auto *outptr = &output[0];

        const bool correct =
            (m_basis[0]->GetBasisType() == LibUtilities::eModified_A);

        // Workspace for kernels - also checks preconditions
        size_t wsp0Size = 0;
        IProduct2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);

        // std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size), tmpIn(nqTot),
        //     tmpOut(m_nmTot);
        std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size);
        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);
        vec_t::scalarType *tmpOut =
            reinterpret_cast<vec_t::scalarType *>(output);

        vec_t *jac_ptr;

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            if constexpr (DEFORMED)
            {
                jac_ptr = &((*this->m_jac)[m_jac_idx]);
                m_jac_idx += nqTot;
            }
            else
            {
                jac_ptr = &((*this->m_jac)[m_jac_idx++]);
            }

            // Load and transpose data
            // load_interleave(inptr, nqTot, tmpIn);

            IProduct2DKernel<SHAPE_TYPE, false, false, DEFORMED>(
                nm0, nm1, nq0, nq1, correct, tmpIn, this->m_B[0], this->m_B[1],
                this->m_w[0], this->m_w[1], jac_ptr, wsp0, tmpOut);

            // de-interleave and store data
            // deinterleave_store(tmpOut, m_nmTot, outptr);

            tmpIn += nqTot;
            tmpOut += nmTot * vec_t::width;
        }
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, int nm0,
              int nm1, int nq0, int nq1>
    void operator2D(const NekDouble *input, NekDouble *output)
    {
        constexpr auto nqTot = nq0 * nq1;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
        // constexpr auto nqBlocks = nqTot * vec_t::width;
        // const auto nmBlocks     = m_nmTot * vec_t::width;

        // auto *inptr  = &input[0];
        // auto *outptr = &output[0];

        const bool correct =
            (m_basis[0]->GetBasisType() == LibUtilities::eModified_A);

        // Workspace for kernels - also checks preconditions
        size_t wsp0Size = 0;
        IProduct2DWorkspace<SHAPE_TYPE>(nm0, nm1, nq0, nq1, wsp0Size);

        // std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size), tmpIn(nmTot),
        //     tmpOut(nqTot);
        std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size);
        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);
        vec_t::scalarType *tmpOut =
            reinterpret_cast<vec_t::scalarType *>(output);

        vec_t *jac_ptr;

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            if constexpr (DEFORMED)
            {
                jac_ptr = &((*this->m_jac)[m_jac_idx]);
                m_jac_idx += nqTot;
            }
            else
            {
                jac_ptr = &((*this->m_jac)[m_jac_idx++]);
            }

            // Load and transpose data
            // load_interleave(inptr, nqTot, tmpIn);

            IProduct2DKernel<SHAPE_TYPE, false, false, DEFORMED>(
                nm0, nm1, nq0, nq1, correct, tmpIn, this->m_B[0], this->m_B[1],
                this->m_w[0], this->m_w[1], jac_ptr, wsp0, tmpOut);

            // de-interleave and store data
            // deinterleave_store(tmpOut, m_nmTot, outptr);

            tmpIn += nqTot;
            tmpOut += nmTot * vec_t::width;
        }
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void operator3D(const NekDouble *input, NekDouble *output)
    {
        const auto nm0 = m_basis[0]->GetNumModes();
        const auto nm1 = m_basis[1]->GetNumModes();
        const auto nm2 = m_basis[2]->GetNumModes();

        const auto nq0 = m_basis[0]->GetNumPoints();
        const auto nq1 = m_basis[1]->GetNumPoints();
        const auto nq2 = m_basis[2]->GetNumPoints();

        const auto nqTot = nq0 * nq1 * nq2;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
        // const auto nqBlocks = nqTot * vec_t::width;
        // const auto nmBlocks = m_nmTot * vec_t::width;

        // auto *inptr  = &input[0];
        // auto *outptr = &output[0];

        const bool correct =
            (m_basis[0]->GetBasisType() == LibUtilities::eModified_A);

        // Workspace for kernels - also checks preconditions
        size_t wsp0Size = 0, wsp1Size = 0, wsp2Size = 0;
        IProduct3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size, wsp2Size);

        // std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size), wsp1(wsp1Size),
        //     wsp2(wsp2Size), tmpIn(nqTot), tmpOut(m_nmTot);
        std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size), wsp1(wsp1Size),
            wsp2(wsp2Size);
        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);
        vec_t::scalarType *tmpOut =
            reinterpret_cast<vec_t::scalarType *>(output);

        vec_t *jac_ptr;

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            if constexpr (DEFORMED)
            {
                jac_ptr = &((*this->m_jac)[m_jac_idx]);
                m_jac_idx += nqTot;
            }
            else
            {
                jac_ptr = &((*this->m_jac)[m_jac_idx++]);
            }

            // Load and transpose data
            // load_interleave(inptr, nqTot, tmpIn);

            IProduct3DKernel<SHAPE_TYPE, false, false, DEFORMED>(
                nm0, nm1, nm2, nq0, nq1, nq2, correct, tmpIn, this->m_B[0],
                this->m_B[1], this->m_B[2], this->m_w[0], this->m_w[1],
                this->m_w[2], jac_ptr, wsp0, wsp1, wsp2, tmpOut);

            // de-interleave and store data
            // deinterleave_store(tmpOut, m_nmTot, outptr);

            tmpIn += nqTot;
            tmpOut += nmTot * vec_t::width;
        }
    }

    // Size based template version.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, int nm0,
              int nm1, int nm2, int nq0, int nq1, int nq2>
    void operator3D(const NekDouble *input, NekDouble *output)
    {
        constexpr auto nqTot = nq0 * nq1 * nq2;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
        // constexpr auto nqBlocks = nqTot * vec_t::width;
        // const auto nmBlocks     = m_nmTot * vec_t::width;

        // auto *inptr  = &input[0];
        // auto *outptr = &output[0];

        const bool correct =
            (m_basis[0]->GetBasisType() == LibUtilities::eModified_A);

        // Workspace for kernels - also checks preconditions
        size_t wsp0Size = 0, wsp1Size = 0, wsp2Size = 0;
        IProduct3DWorkspace<SHAPE_TYPE>(nm0, nm1, nm2, nq0, nq1, nq2, wsp0Size,
                                        wsp1Size, wsp2Size);

        // std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size), wsp1(wsp1Size),
        //     wsp2(wsp2Size), tmpIn(nqTot), tmpOut(m_nmTot);
        std::vector<vec_t, allocator<vec_t>> wsp0(wsp0Size), wsp1(wsp1Size),
            wsp2(wsp2Size);
        const vec_t::vectorType *tmpIn =
            reinterpret_cast<const vec_t::vectorType *>(input);
        vec_t::scalarType *tmpOut =
            reinterpret_cast<vec_t::scalarType *>(output);

        vec_t *jac_ptr;

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            if constexpr (DEFORMED)
            {
                jac_ptr = &((*this->m_jac)[m_jac_idx]);
                m_jac_idx += nqTot;
            }
            else
            {
                jac_ptr = &((*this->m_jac)[m_jac_idx++]);
            }

            // Load and transpose data
            // load_interleave(inptr, nqTot, tmpIn);

            IProduct3DKernel<SHAPE_TYPE, false, false, DEFORMED>(
                nm0, nm1, nm2, nq0, nq1, nq2, correct, tmpIn, this->m_B[0],
                this->m_B[1], this->m_B[2], this->m_w[0], this->m_w[1],
                this->m_w[2], jac_ptr, wsp0, wsp1, wsp2, tmpOut);

            // de-interleave and store data
            // deinterleave_store(tmpOut, m_nmTot, outptr);

            tmpIn += nqTot;
            tmpOut += nmTot * vec_t::width;
        }
    }
};

} // namespace Nektar::Operators::detail
