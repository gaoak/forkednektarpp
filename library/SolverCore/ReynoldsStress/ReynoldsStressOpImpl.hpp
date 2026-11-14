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
        // A 3DH1 expansion is a 2D mesh extended by the homogeneous direction.
        const bool homogeneous =
            expansionList->GetExpType() == MultiRegions::e3DH1D;
        const unsigned int shapeDimension = expansionList->GetShapeDimension();
        m_dimension = shapeDimension + (homogeneous ? 1 : 0);
        ASSERTL0(m_dimension == 2 || m_dimension == 3,
                 "ReynoldsStress supports only 2D, 3D and 3DH1 expansions.");
        ASSERTL0(expansionList->GetCoordim(0) == shapeDimension,
                 "ReynoldsStress does not support embedded-manifold "
                 "expansions.");
        ASSERTL0(components.size() >= m_dimension,
                 "ReynoldsStress requires one velocity component per "
                 "dimension.");
        m_nProducts = m_dimension * (m_dimension + 1) / 2;
        m_numPlanes = homogeneous ? expansionList->GetZIDs().size() : 1;
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
    /// Number of local planes, or 1 without a homogeneous direction.
    unsigned int m_numPlanes = 1;

    void v_Configure(
        const std::vector<unsigned int> &velocityComponents) override
    {
        ASSERTL0(velocityComponents.size() == m_dimension,
                 "ReynoldsStress velocity selection must match the "
                 "dimension.");
        std::copy(velocityComponents.begin(), velocityComponents.end(),
                  m_velocityComponents.begin());
    }

    void v_Apply(
        LibUtilities::Field<TData, FieldState::Phys> &velocity,
        LibUtilities::Field<TData, FieldState::Phys> &products) override
    {
        ASSERTL1(velocity.GetNumHomoModes() == m_numPlanes &&
                     products.GetNumHomoModes() == m_numPlanes,
                 "ReynoldsStress velocity or product plane count mismatch.");
        ASSERTL1(velocity.GetNumComponents() >= m_dimension &&
                     products.GetNumComponents() == m_nProducts,
                 "ReynoldsStress velocity or product component count "
                 "mismatch.");
        ASSERTL1(velocity.GetBlocks().size() == products.GetBlocks().size(),
                 "ReynoldsStress velocity-product block count mismatch.");

        switch (m_dimension)
        {
            case 2:
                OperatorND<2>(velocity, products);
                break;
            case 3:
                OperatorND<3>(velocity, products);
                break;
            default:
                NEKERROR(ErrorUtil::efatal,
                         "ReynoldsStress supports only dimensions 2 and 3.");
        }
    }

    template <unsigned int DIMENSION>
    void OperatorND(LibUtilities::Field<TData, FieldState::Phys> &velocity,
                    LibUtilities::Field<TData, FieldState::Phys> &products)
    {
        // The products are laid out like the velocity, so that one storage
        // index addresses the same point in both; a block at another
        // interleave width is reshaped, keeping its content.
        products.ReshapeStorage(velocity.GetInterleaveWidth(), ExecSpace::name);

        for (unsigned int block_idx = 0;
             block_idx < velocity.GetBlocks().size(); ++block_idx)
        {
            const unsigned int streamID = block_idx + 1;

            auto &velocityBlock = velocity.GetBlocks()[block_idx];
            auto &productBlock  = products.GetBlocks()[block_idx];

            ASSERTL1(productBlock.CompSize() == velocityBlock.CompSize(),
                     "ReynoldsStress velocity-product block layout "
                     "mismatch.");

            const auto velocityPtr =
                velocityBlock.template GetPtr<MemSpace, ReadOnly>(streamID);
            auto productPtr =
                productBlock.template GetPtr<MemSpace, ReadWrite>(streamID);

            ReynoldsStressKernelLauncher<DIMENSION, ExecSpace>(
                velocityBlock.CompSize(), m_numPlanes, m_velocityComponents[0],
                m_velocityComponents[1], m_velocityComponents[2], this->m_scale,
                velocityPtr, productPtr, streamID);
        }
    }
};

} // namespace Nektar::SolverCore::detail
