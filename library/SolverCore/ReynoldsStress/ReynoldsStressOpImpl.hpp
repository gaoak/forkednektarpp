///////////////////////////////////////////////////////////////////////////////
//
// File: ReynoldsStressOpImpl.hpp
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
// Description: Device-portable velocity-product operator.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <algorithm>
#include <array>

#include <LibUtilities/BasicUtils/Utils/UtilsKernels.hpp>
#include <SolverCore/ReynoldsStress/ReynoldsStressKernels.hpp>
#include <SolverCore/ReynoldsStress/ReynoldsStressOp.hpp>

namespace Nektar::SolverCore::detail
{

template <typename ExecSpace, typename TData>
class ReynoldsStressOpImpl : public ReynoldsStressOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    ReynoldsStressOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                         const std::vector<std::string> &components)
        : ReynoldsStressOp<TData>(expansionList, components)
    {
        m_dimension = expansionList->GetCoordim(0);
        ASSERTL0(m_dimension == 2 || m_dimension == 3,
                 "ReynoldsStress supports only 2D and 3D expansions.");
        ASSERTL0(expansionList->GetShapeDimension() == m_dimension,
                 "ReynoldsStress does not support embedded-manifold "
                 "expansions.");
        ASSERTL0(components.size() >= m_dimension,
                 "ReynoldsStress requires one velocity component per "
                 "dimension.");
        m_nProducts = m_dimension * (m_dimension + 1) / 2;
    }

    static std::string className;

    static std::unique_ptr<MultiRegions::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<ReynoldsStressOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    std::array<unsigned int, 3> m_velocityComponents{};
    unsigned int m_dimension = 0;
    unsigned int m_nProducts = 0;

    void v_Configure(
        const std::vector<unsigned int> &velocityComponents) override
    {
        ASSERTL0(velocityComponents.size() == m_dimension,
                 "ReynoldsStress velocity selection must match the "
                 "dimension.");
        std::copy(velocityComponents.begin(), velocityComponents.end(),
                  m_velocityComponents.begin());
    }

    template <unsigned int DIMENSION>
    void ApplyProducts(LibUtilities::Field<TData, FieldState::Phys> &velocity,
                       LibUtilities::Field<TData, FieldState::Phys> &products)
    {
        for (unsigned int block = 0; block < velocity.GetBlocks().size();
             ++block)
        {
            const unsigned int streamID = block + 1;
            auto &velocityBlock         = velocity.GetBlocks()[block];
            auto &productBlock          = products.GetBlocks()[block];
            ASSERTL1(productBlock.CompSize() == velocityBlock.CompSize(),
                     "ReynoldsStress velocity-product block layout "
                     "mismatch.");

            const auto velocityPtr =
                velocityBlock.template GetPtr<MemSpace, ReadOnly>(streamID);
            auto productPtr =
                productBlock.template GetPtr<MemSpace, ReadWrite>(streamID);

            // The products are laid out like the velocity, so that one
            // storage index addresses the same point in both; a block at
            // another interleave width is reshaped, keeping its content.
            const unsigned int interleaveWidth =
                velocityBlock.GetInterleaveWidth();
            if (productBlock.GetInterleaveWidth() != interleaveWidth)
            {
                LibUtilities::ReshapeStorage<ExecSpace>(
                    interleaveWidth, productBlock.GetInterleaveWidth(),
                    productBlock.GetNumElementsWithPadding() * m_nProducts,
                    productBlock.GetNumData(), productPtr, streamID);
                productBlock.template SetInterleaveWidth<TData>(
                    interleaveWidth);
            }

            ReynoldsStressKernel<DIMENSION, ExecSpace>(
                velocityBlock.CompSize(), m_velocityComponents[0],
                m_velocityComponents[1], m_velocityComponents[2], this->m_scale,
                velocityPtr, productPtr, streamID);
        }
    }

    void v_Apply(
        LibUtilities::Field<TData, FieldState::Phys> &velocity,
        LibUtilities::Field<TData, FieldState::Phys> &products) override
    {
        ASSERTL1(velocity.GetNumHomoModes() == 1 &&
                     products.GetNumHomoModes() == 1,
                 "ReynoldsStress does not support homogeneous expansions.");
        ASSERTL1(velocity.GetNumComponents() >= m_dimension &&
                     products.GetNumComponents() == m_nProducts,
                 "ReynoldsStress velocity or product component count "
                 "mismatch.");
        ASSERTL1(velocity.GetBlocks().size() == products.GetBlocks().size(),
                 "ReynoldsStress velocity-product block count mismatch.");

        switch (m_dimension)
        {
            case 2:
                ApplyProducts<2>(velocity, products);
                break;
            case 3:
                ApplyProducts<3>(velocity, products);
                break;
            default:
                NEKERROR(ErrorUtil::efatal,
                         "ReynoldsStress supports only dimensions 2 and 3.");
        }
    }
};

} // namespace Nektar::SolverCore::detail
