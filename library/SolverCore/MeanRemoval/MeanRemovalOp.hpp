///////////////////////////////////////////////////////////////////////////////
//
// File: MeanRemovalOp.hpp
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

#include <MultiRegions/Common/Operator.hpp>
#include <SolverCore/Mean/MeanBlockOp.hpp>
#include <SolverCore/SolverCore.hpp>

namespace Nektar::SolverCore
{

template <typename TData> class LinearSystemOp;

/**
 * @brief Subtract the domain average of every component of a physical-space
 * field from it, in place.
 *
 * A variable governed by a singular system, such as the pressure under
 * Neumann or periodic boundaries only, is determined up to an additive
 * constant; this operator removes it, so that norms and errors of the field
 * reflect the discretisation rather than the constant. With
 * SetNeumannBalance() it instead drives each component's domain integral to
 * the surface integral of the system's Neumann values. The integrals are
 * accumulated per block in the selected operator execution space by the
 * Mean block operators and reduced once across the ranks. The domain volume
 * is computed by the same block operators on the first Apply() and kept.
 *
 * With a homogeneous direction (3DH1) the block operators sum the local
 * planes of each component with their z quadrature weights, so that one
 * constant, the mean over the 3D domain, is removed from every plane. The
 * volume is the plane area, reduced over the ranks sharing the planes only,
 * times the homogeneous length.
 *
 * Apply() is collective: every rank must call it, and the first Apply()
 * also computes the volume. GetVolume() and GetBalancedMean() are valid on
 * every rank.
 */
template <typename TData>
class MeanRemovalOp : public MultiRegions::Operator<TData>
{
public:
    static std::shared_ptr<MeanRemovalOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "")
    {
        EnsureLinked();

        auto session = expansionList->GetSession();

        std::string execStr0 =
            (execStr == "")
                ? MultiRegions::Operator<TData>::GetOpExecSpace(session)
                : execStr;

        auto op = MultiRegions::Operator<TData>::template Create<MeanRemovalOp>(
            expansionList, components, execStr0);

        // z quadrature weight of each local plane, times Lhom / 2, with which
        // the block ops sum the planes; empty without a homogeneous direction.
        std::vector<TData> planeWeights;
        if (op->m_homogeneous)
        {
            const auto w    = expansionList->GetHomogeneousBasis()->GetW();
            const auto zIDs = expansionList->GetZIDs();
            for (unsigned int n = 0; n < zIDs.size(); ++n)
            {
                planeWeights.push_back(w[zIDs[n]] * op->m_homoLen * 0.5);
            }
            op->m_planeWeights = planeWeights;
        }

        const auto blockAttr =
            MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                expansionList);

        for (unsigned int block_idx = 0; block_idx < blockAttr.size();
             block_idx++)
        {
            const auto exp =
                MultiRegions::GetCollection(expansionList, block_idx)
                    .GetExpVector()[0];

            op->m_blockOp.push_back(MeanBlockOp<TData>::Create(
                block_idx, exp, expansionList->GetDataWarehouseSharedPtr(),
                execStr0));
            op->m_blockOp.back()->SetNumElements(
                blockAttr[block_idx].GetNumElements());
            if (op->m_homogeneous)
            {
                op->m_blockOp.back()->SetPlaneWeights(planeWeights);
            }
        }

        return op;
    }

    static inline const std::string name = "MeanRemoval";

    /// Collective: every rank must call it.
    void Apply(LibUtilities::Field<TData, FieldState::Phys> &inout)
    {
        v_Apply(inout);
    }

    /// Collective: as Apply().
    void operator()(LibUtilities::Field<TData, FieldState::Phys> &inout)
    {
        v_Apply(inout);
    }

    /**
     * @brief Drive the domain integral of each component to the surface
     * integral of the Neumann boundary values of @p system, instead of to
     * zero.
     *
     * @details
     * A singular pressure system is solvable only when the domain integral
     * of its forcing equals the surface integral of its Neumann boundary
     * values. Apply() then subtracts (integral_c - surface_c) / volume from
     * component c, where surface_c is summed by
     * LinearSystemOp::AddNeumannSurfaceIntegral() and reduced across ranks.
     * A nullptr @p system, the default, removes the mean.
     *
     * The operator keeps a non-owning pointer to @p system, which must
     * outlive its use here.
     *
     * @param system The system with one Neumann value per component, or
     * nullptr.
     */
    void SetNeumannBalance(LinearSystemOp<TData> *system)
    {
        m_neumannSystem = system;
    }

    /**
     * @brief The mean that Apply() drives each component to: the surface
     * integral of the Neumann boundary values of the system set by
     * SetNeumannBalance(), surface_c, divided by the volume of the domain,
     * or zero without a system.
     *
     * @details
     * The constant forcing with this value has the domain integral that
     * the system needs, without a field to balance. Collective: surface_c is
     * reduced across ranks and read back to the host once.
     */
    std::vector<TData> GetBalancedMean()
    {
        const auto volume = GetVolume();
        v_ComputeNeumannSurfaceIntegral();
        auto means = m_surfaceIntegral.ToVector();
        for (auto &mean : means)
        {
            mean /= volume;
        }
        return means;
    }

    /// Domain volume, computed by the first Apply() or on the first call,
    /// whichever comes first, and kept; collective on the first call before
    /// any Apply().
    TData GetVolume()
    {
        v_ComputeVolume();
        return m_volume;
    }

protected:
    bool m_volumeComputed = false;
    TData m_volume        = 1.0;
    /// Whether the field has a homogeneous direction (3DH1).
    bool m_homogeneous = false;
    /// Homogeneous length, or 1 without a homogeneous direction.
    TData m_homoLen = 1.0;
    /// z quadrature weight of each local plane, times Lhom / 2; empty
    /// without a homogeneous direction.
    std::vector<TData> m_planeWeights;
    LibUtilities::MemoryRegion<TData> m_domainIntegral;
    LibUtilities::MemoryRegion<TData> m_surfaceIntegral;
    LibUtilities::MemoryRegion<TData> m_surfaceIntegralPlane;
    std::vector<std::shared_ptr<MeanBlockOp<TData>>> m_blockOp;
    LinearSystemOp<TData> *m_neumannSystem = nullptr;

    MeanRemovalOp(const MultiRegions::ExpListSharedPtr &expansionList,
                  const std::vector<std::string> &components)
        : MultiRegions::Operator<TData>(expansionList, components)
    {
        if (expansionList->GetExpType() == MultiRegions::e3DH1D)
        {
            m_homogeneous = true;
            m_homoLen     = expansionList->GetHomoLen();
        }
    }

    ~MeanRemovalOp() override = default;

    virtual void v_Apply(
        LibUtilities::Field<TData, FieldState::Phys> &inout) = 0;
    virtual void v_ComputeNeumannSurfaceIntegral()           = 0;
    virtual void v_ComputeVolume()                           = 0;
};

} // namespace Nektar::SolverCore
