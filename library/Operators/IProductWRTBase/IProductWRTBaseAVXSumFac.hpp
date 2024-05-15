///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseAVXSumFac.hpp
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

#include "IProductWRTBaseAVXSumFacKernels.hpp"
#include "Operators/OperatorHelper.hpp"

#include <LibUtilities/BasicUtils/ShapeType.hpp>
#include <LibUtilities/BasicUtils/SharedArray.hpp>
#include <LibUtilities/Foundations/Basis.h>
#include <LibUtilities/SimdLib/tinysimd.hpp>

namespace Nektar::Operators::detail
{

// Matrix-free implementation
template <typename ExecSpace, typename Implementation, typename TData,
          typename = typename std::enable_if<
              std::is_same<ExecSpace, NektarSpaces::AVX>::value &&
              std::is_same<Implementation, Operators::SumFac>::value>::type>
class OperatorIProductWRTBaseImpl : public OperatorIProductWRTBase<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorIProductWRTBaseImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorIProductWRTBase<TData>(expansionList)
    {
        // Initialise jacobian with paddings
        auto blocks =
            GetBlockAttributes(FieldState::Phys, expansionList, vec_t::width);

        size_t jacSize         = Operator<TData>::GetGeometricFactorSize();
        Array<OneD, TData> jac = Operator<TData>::SetJacobian(jacSize);

        m_jac = MemoryRegion<vec_t>::template fromArray<MemSpace, TData>(jac);

        // Initialize the basis data.
        m_basisMap  = GetBasisData<MemSpace, TData, vec_t>(expansionList,
                                                          BASIS_BASIS_DATA);
        m_weightMap = GetBasisData<MemSpace, TData, vec_t>(expansionList,
                                                           BASIS_WEIGHT_DATA);
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

        auto *inptr  = in.template GetConstPtr<MemSpace>();
        auto *outptr = out.template GetPtr<MemSpace>();

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

            // Fetch basis key for the current element type.
            m_basisKeys.clear();
            for (size_t d = 0; d < dimension; ++d)
            {
                // m_basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
                m_basisKeys.push_back(expPtr->GetBasis(d)->GetBasisKey());
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

#include "../Common/SwitchLevel2Deformed.h"

            inptr += inblock.block_size;
            outptr += outblock.block_size;
            exp_idx += inblock.num_elements;
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
    int m_nElmtGroup;
    int m_jac_idx;

    MemoryRegion<vec_t> m_jac;
    BasisDataMap<vec_t> m_basisMap;
    BasisDataMap<vec_t> m_weightMap;
    // std::array<LibUtilities::BasisKey, 3> m_basisKeys;
    std::vector<LibUtilities::BasisKey> m_basisKeys;

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void operator1D(const NekDouble *input, NekDouble *output)
    {
        const size_t exp_idx = 0;
        const auto expPtr    = this->m_expansionList->GetExp(exp_idx);

        const auto nm0 = expPtr->GetBasisNumModes(0);
        const auto nq0 = expPtr->GetNumPoints(0);

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

        auto jptr = m_jac.template GetPtr<MemSpace>();
        auto bptr0 =
            m_basisMap[m_basisKeys[0]].template GetConstPtr<MemSpace>();
        auto wptr0 =
            m_weightMap[m_basisKeys[0]].template GetConstPtr<MemSpace>();

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            jac_ptr = &(jptr[m_jac_idx]);

            if constexpr (DEFORMED)
            {
                m_jac_idx += nqTot;
            }
            else
            {
                m_jac_idx++;
            }

            // Load and transpose data
            // load_interleave(inptr, nqTot, tmpIn);

            IProduct1DKernel<SHAPE_TYPE, false, false, DEFORMED>(
                nm0, nq0, tmpIn, bptr0, wptr0, jac_ptr, tmpOut);

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

        auto jptr = m_jac.template GetPtr<MemSpace>();
        auto bptr0 =
            m_basisMap[m_basisKeys[0]].template GetConstPtr<MemSpace>();
        auto wptr0 =
            m_weightMap[m_basisKeys[0]].template GetConstPtr<MemSpace>();

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            jac_ptr = &(jptr[m_jac_idx]);

            if constexpr (DEFORMED)
            {
                m_jac_idx += nqTot;
            }
            else
            {
                m_jac_idx++;
            }

            // Load and transpose data
            // load_interleave(inptr, nqTot, tmpIn);

            IProduct1DKernel<SHAPE_TYPE, false, false, DEFORMED>(
                nm0, nq0, tmpIn, bptr0, wptr0, jac_ptr, tmpOut);

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
        const size_t exp_idx = 0;
        const auto expPtr    = this->m_expansionList->GetExp(exp_idx);

        const auto nm0 = expPtr->GetBasisNumModes(0);
        const auto nm1 = expPtr->GetBasisNumModes(1);

        const auto nq0 = expPtr->GetNumPoints(0);
        const auto nq1 = expPtr->GetNumPoints(1);

        const auto nqTot = nq0 * nq1;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
        // const auto nqBlocks = nqTot * vec_t::width;
        // const auto nmBlocks = m_nmTot * vec_t::width;

        // auto *inptr  = &input[0];
        // auto *outptr = &output[0];

        const bool correct =
            (expPtr->GetBasisType(0) == LibUtilities::eModified_A);

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

        auto jptr = m_jac.template GetPtr<MemSpace>();
        auto bptr0 =
            m_basisMap[m_basisKeys[0]].template GetConstPtr<MemSpace>();
        auto bptr1 =
            m_basisMap[m_basisKeys[1]].template GetConstPtr<MemSpace>();
        auto wptr0 =
            m_weightMap[m_basisKeys[0]].template GetConstPtr<MemSpace>();
        auto wptr1 =
            m_weightMap[m_basisKeys[1]].template GetConstPtr<MemSpace>();
        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            jac_ptr = &(jptr[m_jac_idx]);

            if constexpr (DEFORMED)
            {
                m_jac_idx += nqTot;
            }
            else
            {
                m_jac_idx++;
            }

            // Load and transpose data
            // load_interleave(inptr, nqTot, tmpIn);

            IProduct2DKernel<SHAPE_TYPE, false, false, DEFORMED>(
                nm0, nm1, nq0, nq1, correct, tmpIn, bptr0, bptr1, wptr0, wptr1,
                jac_ptr, wsp0, tmpOut);

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
        const size_t exp_idx = 0;
        const auto expPtr    = this->m_expansionList->GetExp(exp_idx);

        constexpr auto nqTot = nq0 * nq1;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
        // constexpr auto nqBlocks = nqTot * vec_t::width;
        // const auto nmBlocks     = m_nmTot * vec_t::width;

        // auto *inptr  = &input[0];
        // auto *outptr = &output[0];

        const bool correct =
            (expPtr->GetBasisType(0) == LibUtilities::eModified_A);

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

        auto jptr = m_jac.template GetPtr<MemSpace>();
        auto bptr0 =
            m_basisMap[m_basisKeys[0]].template GetConstPtr<MemSpace>();
        auto bptr1 =
            m_basisMap[m_basisKeys[1]].template GetConstPtr<MemSpace>();
        auto wptr0 =
            m_weightMap[m_basisKeys[0]].template GetConstPtr<MemSpace>();
        auto wptr1 =
            m_weightMap[m_basisKeys[1]].template GetConstPtr<MemSpace>();

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            jac_ptr = &(jptr[m_jac_idx]);

            if constexpr (DEFORMED)
            {
                m_jac_idx += nqTot;
            }
            else
            {
                m_jac_idx++;
            }

            // Load and transpose data
            // load_interleave(inptr, nqTot, tmpIn);

            IProduct2DKernel<SHAPE_TYPE, false, false, DEFORMED>(
                nm0, nm1, nq0, nq1, correct, tmpIn, bptr0, bptr1, wptr0, wptr1,
                jac_ptr, wsp0, tmpOut);

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
        const size_t exp_idx = 0;
        const auto expPtr    = this->m_expansionList->GetExp(exp_idx);

        const auto nm0 = expPtr->GetBasisNumModes(0);
        const auto nm1 = expPtr->GetBasisNumModes(1);
        const auto nm2 = expPtr->GetBasisNumModes(2);

        const auto nq0 = expPtr->GetNumPoints(0);
        const auto nq1 = expPtr->GetNumPoints(1);
        const auto nq2 = expPtr->GetNumPoints(2);

        const auto nqTot = nq0 * nq1 * nq2;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
        // const auto nqBlocks = nqTot * vec_t::width;
        // const auto nmBlocks = m_nmTot * vec_t::width;

        // auto *inptr  = &input[0];
        // auto *outptr = &output[0];

        const bool correct =
            (expPtr->GetBasisType(0) == LibUtilities::eModified_A);

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

        auto jptr = m_jac.template GetPtr<MemSpace>();
        auto bptr0 =
            m_basisMap[m_basisKeys[0]].template GetConstPtr<MemSpace>();
        auto bptr1 =
            m_basisMap[m_basisKeys[1]].template GetConstPtr<MemSpace>();
        auto bptr2 =
            m_basisMap[m_basisKeys[2]].template GetConstPtr<MemSpace>();
        auto wptr0 =
            m_weightMap[m_basisKeys[0]].template GetConstPtr<MemSpace>();
        auto wptr1 =
            m_weightMap[m_basisKeys[1]].template GetConstPtr<MemSpace>();
        auto wptr2 =
            m_weightMap[m_basisKeys[2]].template GetConstPtr<MemSpace>();

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            jac_ptr = &(jptr[m_jac_idx]);

            if constexpr (DEFORMED)
            {
                m_jac_idx += nqTot;
            }
            else
            {
                m_jac_idx++;
            }

            // Load and transpose data
            // load_interleave(inptr, nqTot, tmpIn);

            IProduct3DKernel<SHAPE_TYPE, false, false, DEFORMED>(
                nm0, nm1, nm2, nq0, nq1, nq2, correct, tmpIn, bptr0, bptr1,
                bptr2, wptr0, wptr1, wptr2, jac_ptr, wsp0, wsp1, wsp2, tmpOut);

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
        const size_t exp_idx = 0;
        const auto expPtr    = this->m_expansionList->GetExp(exp_idx);

        constexpr auto nqTot = nq0 * nq1 * nq2;
        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
        // constexpr auto nqBlocks = nqTot * vec_t::width;
        // const auto nmBlocks     = m_nmTot * vec_t::width;

        // auto *inptr  = &input[0];
        // auto *outptr = &output[0];

        const bool correct =
            (expPtr->GetBasisType(0) == LibUtilities::eModified_A);

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

        auto jptr = m_jac.template GetPtr<MemSpace>();
        auto bptr0 =
            m_basisMap[m_basisKeys[0]].template GetConstPtr<MemSpace>();
        auto bptr1 =
            m_basisMap[m_basisKeys[1]].template GetConstPtr<MemSpace>();
        auto bptr2 =
            m_basisMap[m_basisKeys[2]].template GetConstPtr<MemSpace>();
        auto wptr0 =
            m_weightMap[m_basisKeys[0]].template GetConstPtr<MemSpace>();
        auto wptr1 =
            m_weightMap[m_basisKeys[1]].template GetConstPtr<MemSpace>();
        auto wptr2 =
            m_weightMap[m_basisKeys[2]].template GetConstPtr<MemSpace>();

        for (size_t e = 0; e < m_nElmtGroup; ++e)
        {
            jac_ptr = &(jptr[m_jac_idx]);

            if constexpr (DEFORMED)
            {
                m_jac_idx += nqTot;
            }
            else
            {
                m_jac_idx++;
            }

            // Load and transpose data
            // load_interleave(inptr, nqTot, tmpIn);

            IProduct3DKernel<SHAPE_TYPE, false, false, DEFORMED>(
                nm0, nm1, nm2, nq0, nq1, nq2, correct, tmpIn, bptr0, bptr1,
                bptr2, wptr0, wptr1, wptr2, jac_ptr, wsp0, wsp1, wsp2, tmpOut);

            // de-interleave and store data
            // deinterleave_store(tmpOut, m_nmTot, outptr);

            tmpIn += nqTot;
            tmpOut += nmTot * vec_t::width;
        }
    }
};

} // namespace Nektar::Operators::detail
