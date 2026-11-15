///////////////////////////////////////////////////////////////////////////////
//
// File: EnergyOpImpl.hpp
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
// Description: Device-portable kinetic-energy and enstrophy operator.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <algorithm>
#include <array>

#include <IncNavierStokesSolverRedesign/Utilities/Energy/EnergyKernels.hpp>
#include <IncNavierStokesSolverRedesign/Utilities/Energy/EnergyOp.hpp>
#include <LocalRegions/DataWarehouse/GeometricDataWarehouse.hpp>

namespace Nektar::detail
{

template <typename ExecSpace, typename TData>
class EnergyOpImpl : public EnergyOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    EnergyOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                 const std::vector<std::string> &components)
        : EnergyOp<TData>(expansionList, components)
    {
        // A 3DH1 expansion is a 2D mesh extended by the homogeneous direction.
        const bool homogeneous =
            expansionList->GetExpType() == MultiRegions::e3DH1D;
        const unsigned int shapeDimension = expansionList->GetShapeDimension();
        m_dimension = shapeDimension + (homogeneous ? 1 : 0);
        ASSERTL0(m_dimension == 2 || m_dimension == 3,
                 "EnergyOp supports only 2D, 3D and 3DH1 expansions.");
        ASSERTL0(expansionList->GetCoordim(0) == shapeDimension,
                 "EnergyOp does not support embedded-manifold expansions.");
        ASSERTL0(components.size() >= m_dimension,
                 "EnergyOp requires one velocity component per dimension.");

        // The quadrature metric of each block is fetched from the data
        // warehouse at the interleave width of the first velocity read at it.
        const size_t nBlocks =
            MultiRegions::GetCollections(expansionList).size();
        for (auto &metric : m_metric)
        {
            metric.assign(nBlocks, nullptr);
        }

        // z quadrature weight of each local plane, times Lhom / 2, with which
        // the kernel sums the planes; a single unit weight without a
        // homogeneous direction.
        std::vector<TData> planeWeights{TData{1}};
        if (homogeneous)
        {
            const auto w       = expansionList->GetHomogeneousBasis()->GetW();
            const auto zIDs    = expansionList->GetZIDs();
            const auto homoLen = expansionList->GetHomoLen();
            ASSERTL0(
                zIDs.size() > 1,
                "EnergyOp requires the homogeneous derivative, which needs "
                "more than one homogeneous plane on each rank.");
            planeWeights.resize(zIDs.size());
            for (unsigned int n = 0; n < zIDs.size(); ++n)
            {
                planeWeights[n] = w[zIDs[n]] * homoLen * 0.5;
            }
        }
        m_numPlanes = planeWeights.size();
        m_planeWeights =
            LibUtilities::MemoryRegion<TData>::template FromVector<MemSpace>(
                "Energy plane weights", planeWeights);

        this->m_result = LibUtilities::MemoryRegion<TData>(
            "Energy result", EnergyOp<TData>::resultSize, eHostPinned);
    }

    // className - for OperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in OperatorFactory.
    static std::unique_ptr<MultiRegions::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<EnergyOpImpl<ExecSpace, TData>>(expansionList,
                                                                components);
    }

protected:
    /// Interleave width of the execution space: an input block is read at
    /// this width or at width one.
    static constexpr unsigned int m_implInterleaveWidth =
        NektarSpaces::vector_width<ExecSpace, TData>::value;

    /// Quadrature metric of every point of each block, indexed by block, at
    /// interleave width one (first) and at m_implInterleaveWidth (second);
    /// null until a velocity is first read at that width.
    std::array<std::vector<const TData *>, 2> m_metric;
    /// Quadrature weight of each local homogeneous plane, of size
    /// m_numPlanes.
    LibUtilities::MemoryRegion<TData> m_planeWeights;
    std::array<unsigned int, 3> m_velocityComponents{};
    unsigned int m_dimension = 0;
    /// Number of local planes, or 1 without a homogeneous direction.
    unsigned int m_numPlanes = 1;

    void v_Configure(
        const std::vector<unsigned int> &velocityComponents) override
    {
        ASSERTL0(velocityComponents.size() == m_dimension,
                 "EnergyOp velocity selection must match the dimension.");
        std::copy(velocityComponents.begin(), velocityComponents.end(),
                  m_velocityComponents.begin());
    }

    void v_Apply(
        LibUtilities::Field<TData, FieldState::Phys> &velocity,
        LibUtilities::Field<TData, FieldState::Phys> &velocityGradient) override
    {
        ASSERTL1(velocity.GetNumHomoModes() == m_numPlanes &&
                     velocityGradient.GetNumHomoModes() == m_numPlanes,
                 "EnergyOp velocity plane count mismatch.");
        ASSERTL1(velocity.GetNumComponents() >= m_dimension,
                 "EnergyOp velocity component count mismatch.");
        ASSERTL1(velocity.GetBlocks().size() == m_metric[0].size() &&
                     velocityGradient.GetBlocks().size() == m_metric[0].size(),
                 "EnergyOp velocity block count changed after configuration.");

        // Each block is integrated on its own stream, between the zeroing of
        // the result and its reduction on the null stream.
        this->m_result.template Initialize<MemSpace>(TData{0});

        switch (m_dimension)
        {
            case 2:
                OperatorND<2>(velocity, velocityGradient);
                break;
            case 3:
                OperatorND<3>(velocity, velocityGradient);
                break;
            default:
                ASSERTL0(false, "EnergyOp supports only dimensions 2 and 3.");
        }

        // Reduce over the spatial ranks, which exclude those of other time
        // slices, to every one or to rank zero alone.
        auto comm = this->m_expansionList->GetComm()->GetSpaceComm();
        if (this->m_rootOnly)
        {
            comm->template Reduce<MemSpace>(this->m_result,
                                            LibUtilities::ReduceSum, 0);
        }
        else
        {
            comm->template AllReduce<MemSpace>(this->m_result,
                                               LibUtilities::ReduceSum);
        }
    }

    template <unsigned int DIMENSION>
    void OperatorND(
        LibUtilities::Field<TData, FieldState::Phys> &velocity,
        LibUtilities::Field<TData, FieldState::Phys> &velocityGradient)
    {
        static_assert(DIMENSION == 2 || DIMENSION == 3,
                      "EnergyOp supports only dimensions 2 and 3.");

        for (unsigned int block_idx = 0;
             block_idx < velocity.GetBlocks().size(); ++block_idx)
        {
            const unsigned int streamID = block_idx + 1;

            auto &velocityBlock = velocity.GetBlocks()[block_idx];
            auto &gradientBlock = velocityGradient.GetBlocks()[block_idx];
            const unsigned int interleaveWidth =
                velocityBlock.GetInterleaveWidth();

            ASSERTL0(interleaveWidth == 1 ||
                         interleaveWidth == m_implInterleaveWidth,
                     "EnergyOp reads a velocity at interleave width 1 or " +
                         std::to_string(m_implInterleaveWidth) + ", not " +
                         std::to_string(interleaveWidth) + ".");
            ASSERTL1(velocityBlock.CompSize() == gradientBlock.CompSize() &&
                         gradientBlock.GetInterleaveWidth() == interleaveWidth,
                     "EnergyOp velocity-gradient block layout mismatch.");

            // The metric is laid out like the velocity, so one storage index
            // addresses the same point in both.
            const TData *&metric =
                m_metric[interleaveWidth == 1 ? 0 : 1][block_idx];
            if (metric == nullptr)
            {
                metric = this->m_dataWarehouse->template GetData<MemSpace>(
                    LocalRegions::QuadratureMetricKey<TData>(block_idx,
                                                             interleaveWidth));
            }

            const auto velocityPtr =
                velocityBlock.template GetPtr<MemSpace, ReadOnly>(streamID);
            const auto gradientPtr =
                gradientBlock.template GetPtr<MemSpace, ReadOnly>(streamID);
            const auto planeWeights =
                m_planeWeights.template GetPtr<MemSpace, ReadOnly>(streamID);
            auto result =
                this->m_result.template GetPtr<MemSpace, ReadWrite>(streamID);

            EnergyKernelLauncher<DIMENSION, ExecSpace>(
                velocityBlock.CompSize(), m_numPlanes, m_velocityComponents[0],
                m_velocityComponents[1], m_velocityComponents[2], metric,
                planeWeights, velocityPtr, gradientPtr, result, streamID);
        }
    }
};

} // namespace Nektar::detail
