///////////////////////////////////////////////////////////////////////////////
//
// File: HOutflowBndCondOpImpl.hpp
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
// Description: Dong's high-order outflow boundary condition implementation.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <MultiRegions/ContField.h>
#include <SpatialDomains/Conditions.h>

#include <LibUtilities/BasicUtils/StringUtils.hpp>

#include <algorithm>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include <IncNavierStokesSolverRedesign/BoundaryConditions/HOutflowBndCond/HOutflowBndCondKernels.hpp>
#include <IncNavierStokesSolverRedesign/BoundaryConditions/HOutflowBndCond/HOutflowBndCondOp.hpp>

#include <LibUtilities/Backends/Backends.hpp>
#include <LibUtilities/BasicUtils/Math/Math.hpp>
#include <MultiRegions/BndCondOps/Common/CGBndCondKernels.hpp>
#include <MultiRegions/BndCondOps/Common/CGBndCondMapBuilder.hpp>
#include <MultiRegions/BndCondOps/Common/PhysTraceExtractBndCond.hpp>
#include <MultiRegions/BndCondOps/DirBndCond/DirBndCondKernels.hpp>
#include <MultiRegions/BndCondOps/FwdTransBC/FwdTransBCOp.hpp>
#include <MultiRegions/BndCondOps/NeuBndCond/NeuBndCondKernels.hpp>
#include <MultiRegions/ElmtOps/Expression/ExpressionOp.hpp>
#include <MultiRegions/ElmtOps/IProductWRTBase/IProductWRTBaseOp.hpp>
#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivOp.hpp>

namespace Nektar::detail
{

template <typename ExecSpace, typename TData> class HOutflowBndCondOpImpl;

/**
 * @brief The Dirichlet condition of the pressure system, which applies the
 * pressure of the HOutflowBndCondOpImpl set by SetOwner().
 *
 * @details
 * Apply() sets the Dirichlet pressure coefficients of the regions with
 * HOutflowBndCondOp::ApplyPressure(), from the pressure boundary values of
 * the last HOutflowBndCondOp::UpdatePressure().
 */
template <typename ExecSpace, typename TData>
class HOutflowPressureAdapter : public MultiRegions::DirBndCondOp<TData>
{
public:
    HOutflowPressureAdapter(const MultiRegions::ExpListSharedPtr &expansionList,
                            const std::vector<std::string> &components)
        : MultiRegions::DirBndCondOp<TData>(expansionList, components)
    {
    }

    /// Set the operator whose pressure Apply() applies, which must outlive
    /// this one.
    void SetOwner(HOutflowBndCondOpImpl<ExecSpace, TData> *owner)
    {
        m_owner = owner;
    }

protected:
    void v_Apply(LibUtilities::Field<TData, FieldState::Coeff> &out) override
    {
        m_owner->ApplyPressure(out);
    }

    void v_UpdateBndCoeffs([[maybe_unused]] const TData &time) override
    {
        // The pressure is set by UpdatePressure(), before the pressure solve.
    }

    size_t v_GetNumBndDofs() const override
    {
        return m_owner->GetNumPressureBndDofs();
    }

private:
    // The operator owning this one, set by SetOwner().
    HOutflowBndCondOpImpl<ExecSpace, TData> *m_owner = nullptr;
};

/**
 * @brief The Neumann condition of the velocity system, which applies the
 * velocity flux of the HOutflowBndCondOpImpl set by SetOwner().
 *
 * @details
 * Apply() computes the flux with HOutflowBndCondOp::ApplyVelocity(), from
 * the solved pressure set by HOutflowBndCondOp::SetPressure() and the
 * boundary data of the last HOutflowBndCondOp::UpdatePressure(), and adds
 * its inner product to the velocity forcing, so the flux is never older than
 * the pressure.
 */
template <typename ExecSpace, typename TData>
class HOutflowVelocityAdapter : public MultiRegions::NeuBndCondOp<TData>
{
public:
    HOutflowVelocityAdapter(const MultiRegions::ExpListSharedPtr &expansionList,
                            const std::vector<std::string> &components)
        : MultiRegions::NeuBndCondOp<TData>(expansionList, components)
    {
    }

    /// Set the operator whose velocity flux Apply() applies, which must
    /// outlive this one.
    void SetOwner(HOutflowBndCondOpImpl<ExecSpace, TData> *owner)
    {
        m_owner = owner;
    }

protected:
    void v_Apply(LibUtilities::Field<TData, FieldState::Coeff> &inout) override
    {
        m_owner->ApplyVelocity(inout);
    }

    void v_UpdateBndCoeffs([[maybe_unused]] const TData &time) override
    {
        // The flux depends on the solved pressure, not on the time alone:
        // Apply() computes it.
    }

    size_t v_GetNumBndDofs() const override
    {
        return m_owner->GetNumVelocityBndDofs();
    }

    void v_ExcludeUserDefinedType(
        [[maybe_unused]] const std::string &userDefinedType) override
    {
        NEKERROR(ErrorUtil::efatal, "The HOutflow velocity condition applies "
                                    "only its own regions; it has no default "
                                    "regions to exclude.");
    }

private:
    // The operator owning this one, set by SetOwner().
    HOutflowBndCondOpImpl<ExecSpace, TData> *m_owner = nullptr;
};

/**
 * @brief Implementation of HOutflowBndCondOp.
 *
 * @details
 * UpdatePressure() gathers the velocity on the volume elements adjacent to
 * the regions and differentiates it there, extracts the velocity from the
 * volume field and its gradient from those elements onto every region with
 * PhysTraceExtractBndCond, evaluates the pressure condition and projects it
 * onto the boundary basis. ApplyPressure(), the Apply() of the pressure
 * adapter, then sets the Dirichlet pressure coefficients. ApplyVelocity(),
 * the Apply() of the velocity adapter, extracts the solved pressure set by
 * SetPressure() onto every region, evaluates the velocity flux, takes its
 * inner product with the boundary basis and scatters it into the velocity
 * forcing. Every block of the work runs on its own stream.
 */
template <typename ExecSpace, typename TData>
class HOutflowBndCondOpImpl : public HOutflowBndCondOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;
    friend class HOutflowPressureAdapter<ExecSpace, TData>;
    friend class HOutflowVelocityAdapter<ExecSpace, TData>;

public:
    HOutflowBndCondOpImpl(
        const MultiRegions::ExpListSharedPtr &pressureExpansionList,
        const std::vector<std::string> &pressureComponents)
        : HOutflowBndCondOp<TData>(pressureExpansionList, pressureComponents)
    {
        ASSERTL1(pressureComponents.size() == 1,
                 "HOutflow expects one pressure component.");
        ASSERTL1(std::dynamic_pointer_cast<MultiRegions::ContField>(
                     pressureExpansionList),
                 "HOutflow requires a continuous pressure field.");
        m_nDim = pressureExpansionList->GetCoordim(0);

        auto session = pressureExpansionList->GetSession();
        session->LoadParameter("OutflowBC_theta", m_theta, TData(1));
        session->LoadParameter("OutflowBC_alpha1", m_alpha1, TData(0));
        session->LoadParameter("OutflowBC_alpha2", m_alpha2, TData(0));
        session->LoadParameter("U0_HighOrderBC", m_U0, TData(1));
        session->LoadParameter("Delta_HighOrderBC", m_delta, TData(1.0 / 20.0));
        ASSERTL0(m_U0 > TData(0) && m_delta > TData(0),
                 "HOutflow requires positive U0_HighOrderBC and "
                 "Delta_HighOrderBC.");

        BuildPressureBoundaryData();
        m_pressureAdapter =
            std::make_shared<HOutflowPressureAdapter<ExecSpace, TData>>(
                pressureExpansionList, pressureComponents);
        m_pressureAdapter->SetOwner(this);
    }

    static std::string className;

    static std::unique_ptr<MultiRegions::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<HOutflowBndCondOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    // An index into a block at interleave width 1 and one into a block at
    // m_interleaveWidth, each built from host data the first time a block at
    // that width is seen, and kept.
    struct InterleaveWidthIndices
    {
        bool isBuiltIndex            = false;
        bool isBuiltInterleavedIndex = false;
        LibUtilities::MemoryRegion<size_t> index;
        LibUtilities::MemoryRegion<size_t> interleavedIndex;
    };

    // Point of the source block and point of the destination block of every
    // point of a volume gather, for one pair of interleave widths of the two
    // blocks.
    struct GatherIndices
    {
        LibUtilities::MemoryRegion<size_t> srcIndex;
        LibUtilities::MemoryRegion<size_t> dstIndex;
    };

    // The boundary-adjacent elements of one block of the velocity field
    // that land in one block of the restricted field.
    struct VolumeGather
    {
        unsigned int srcBlock = 0;
        unsigned int dstBlock = 0;
        size_t nData          = 0;
        // Element of the source block and element of the destination block
        // of every element gathered, in increasing destination element.
        std::vector<size_t> srcElmt;
        std::vector<size_t> dstElmt;
        // The indices for every pair of interleave widths of the source and
        // the destination block seen, built from srcElmt and dstElmt the
        // first time the pair is seen, and kept.
        std::map<std::pair<unsigned int, unsigned int>, GatherIndices> indices;

        // Move-only, as GatherIndices is: otherwise MSVC instantiates the
        // copy of indices when std::vector<VolumeGather> reallocates.
        VolumeGather()                                = default;
        VolumeGather(const VolumeGather &)            = delete;
        VolumeGather &operator=(const VolumeGather &) = delete;
        VolumeGather(VolumeGather &&)                 = default;
        VolumeGather &operator=(VolumeGather &&)      = default;
    };

    // The coefficients of one domain block a boundary coefficient map
    // writes: e * nCoeffs + c for coefficient c of element e, and the index
    // built from it for the interleave width of the block.
    struct CoeffTargets
    {
        size_t nCoeffs = 0;
        std::vector<size_t> elmtCoeff;
        InterleaveWidthIndices indices;
    };

    // Interleave width of an interleaved field: 1 (Serial), the SIMD width
    // (AVX) or the warp size (Device).
    static constexpr unsigned int m_interleaveWidth =
        NektarSpaces::vector_width<ExecSpace, TData>::value;
    // Stream the compact arrays are taken on, the first block's, through
    // which the block streams writing and reading them wait on one another.
    static constexpr unsigned int m_opStreamID = 1;

    unsigned int m_nDim = 0;
    TData m_theta = 1, m_alpha1 = 0, m_alpha2 = 0, m_U0 = 1,
          m_delta             = TData(1.0 / 20.0);
    bool m_velocityContextSet = false;
    bool m_pressureUpdated    = false;

    MultiRegions::ExpListSharedPtr m_velocityExpansionList;
    MultiRegions::ExpListSharedPtr m_restrictedVelocityExpansionList;
    // The velocity on the boundary-adjacent elements, which m_physDerivOp
    // differentiates into m_restrictedGradient. Both are laid out at the
    // interleave width of the velocity (SetRestrictedInterleaveWidth()).
    LibUtilities::Field<TData, FieldState::Phys> m_restrictedVelocity;
    LibUtilities::Field<TData, FieldState::Phys> m_restrictedGradient;
    std::shared_ptr<MultiRegions::PhysDerivOp<TData>> m_physDerivOp;
    std::shared_ptr<HOutflowPressureAdapter<ExecSpace, TData>>
        m_pressureAdapter;
    std::shared_ptr<HOutflowVelocityAdapter<ExecSpace, TData>>
        m_velocityAdapter;

    std::vector<unsigned int> m_pressureConditionIndices;
    std::vector<unsigned int> m_regionIds;
    std::vector<MultiRegions::ExpListSharedPtr> m_pressureBndExpLists;
    std::vector<MultiRegions::ExpListSharedPtr> m_velocityBndExpLists;
    // Per region, the pressure condition UpdatePressure() evaluates, and
    // then the solved pressure the velocity adapter extracts.
    std::vector<LibUtilities::Field<TData, FieldState::Phys>> m_pressureBndPhys;
    std::vector<LibUtilities::Field<TData, FieldState::Coeff>>
        m_pressureBndCoeffs;
    std::vector<LibUtilities::Field<TData, FieldState::Phys>> m_velocityBndPhys;
    std::vector<LibUtilities::Field<TData, FieldState::Coeff>>
        m_velocityBndCoeffs;
    std::vector<LibUtilities::Field<TData, FieldState::Phys>> m_fbPhys;
    std::vector<LibUtilities::Field<TData, FieldState::Phys>> m_normals;
    std::vector<LibUtilities::Field<TData, FieldState::Phys>> m_energyMinusFb;
    std::vector<LibUtilities::Field<TData, FieldState::Phys>> m_divergence;
    std::vector<std::shared_ptr<MultiRegions::FwdTransBCOp<TData>>>
        m_pressureFwdTransOps;
    std::vector<std::shared_ptr<MultiRegions::IProductWRTBaseOp<TData>>>
        m_velocityIProductOps;
    std::vector<std::shared_ptr<MultiRegions::ExpressionOp<TData>>>
        m_fbExpressionOps;
    std::vector<VolumeGather> m_volumeGathers;

    // Per region, extracts the velocity onto the points of the region from
    // the volume velocity field.
    std::vector<std::shared_ptr<
        MultiRegions::detail::PhysTraceExtractBndCond<ExecSpace, TData>>>
        m_bndVelocityExtractOps;
    std::vector<LibUtilities::Field<TData, FieldState::Phys>> m_bndVelocity;
    // Per region, extracts m_restrictedGradient onto the points of the
    // region.
    std::vector<std::shared_ptr<
        MultiRegions::detail::PhysTraceExtractBndCond<ExecSpace, TData>>>
        m_bndGradientExtractOps;
    std::vector<LibUtilities::Field<TData, FieldState::Phys>> m_bndGradient;
    // Per region, extracts the solved pressure onto the points of the
    // region.
    std::vector<std::shared_ptr<
        MultiRegions::detail::PhysTraceExtractBndCond<ExecSpace, TData>>>
        m_bndPressureExtractOps;

    size_t m_pressureNumBndCoeffCompSize = 0;
    size_t m_velocityNumBndCoeffCompSize = 0;
    std::vector<size_t> m_pressureCoeffOffsets;
    std::vector<size_t> m_velocityCoeffOffsets;
    LibUtilities::MemoryRegion<TData> m_pressureCompactCoeff;
    LibUtilities::MemoryRegion<TData> m_velocityCompactCoeff;
    // Streams of the region and domain blocks, which write the compact
    // arrays and read the values gathered from them.
    std::vector<unsigned int> m_pressureUpdateStreams;
    std::vector<unsigned int> m_velocityUpdateStreams;

    // The targets of the pressure and velocity maps are indexed at the
    // interleave width of the block ApplyPressure() or ApplyVelocity()
    // receives.
    bool m_pressureSignChange = false;
    std::vector<CoeffTargets> m_pressureMap;
    std::vector<LibUtilities::MemoryRegion<TData>> m_pressureSign;
    std::vector<LibUtilities::MemoryRegion<TData>> m_pressureMappedCoeff;
    std::vector<LibUtilities::MemoryRegion<size_t>> m_pressureCoeffSrc;
    // Offset and number of the pressure targets, per block (the pressure
    // has one component).
    std::vector<size_t> m_pressureOffsets;
    std::vector<size_t> m_pressureCounts;

    bool m_velocityAnySignChange = false;
    std::vector<bool> m_velocitySignChange;
    std::vector<CoeffTargets> m_velocityMap;
    std::vector<LibUtilities::MemoryRegion<TData>> m_velocitySign;
    std::vector<LibUtilities::MemoryRegion<TData>> m_velocityMappedCoeff;
    std::vector<LibUtilities::MemoryRegion<size_t>> m_velocityCoeffSrc;
    std::vector<std::vector<size_t>> m_velocityCompOffsets;
    std::vector<std::vector<size_t>> m_velocityCompCounts;
    std::vector<CoeffTargets> m_velocityGroupTarget;
    std::vector<LibUtilities::MemoryRegion<size_t>> m_velocityGroupOffset;
    std::vector<LibUtilities::MemoryRegion<size_t>> m_velocityGroupSrc;
    std::vector<LibUtilities::MemoryRegion<TData>> m_velocityGroupSign;
    std::vector<LibUtilities::MemoryRegion<TData>> m_velocityGroupValue;
    std::vector<std::vector<size_t>> m_velocityCompGroupOffsets;
    std::vector<std::vector<size_t>> m_velocityCompGroupCounts;

    size_t GetNumPressureBndDofs() const
    {
        return m_pressureNumBndCoeffCompSize;
    }

    size_t GetNumVelocityBndDofs() const
    {
        return m_velocityNumBndCoeffCompSize * m_nDim;
    }

    void v_SetVelocityContext(
        const MultiRegions::ExpListSharedPtr &velocityExpansionList,
        const std::vector<std::string> &velocityComponents) override
    {
        ASSERTL1(velocityComponents.size() >= m_nDim,
                 "Insufficient velocity components for HOutflow.");
        const std::vector<std::string> components(
            velocityComponents.begin(), velocityComponents.begin() + m_nDim);
        m_velocityExpansionList = velocityExpansionList;

        ValidateAndBuildVelocityBoundaryData(components);
        BuildBoundaryElementData(components);
        m_velocityAdapter =
            std::make_shared<HOutflowVelocityAdapter<ExecSpace, TData>>(
                velocityExpansionList, components);
        m_velocityAdapter->SetOwner(this);
        nekStreamSynchronize(0);
        m_velocityContextSet = true;
    }

    void v_UpdatePressure(
        LibUtilities::Field<TData, FieldState::Phys> &velocity) override
    {
        ASSERTL1(m_velocityContextSet,
                 "SetVelocityContext must be called before UpdatePressure.");
        ASSERTL1(this->m_kinvis > TData(0),
                 "HOutflow requires a positive kinematic viscosity, set by "
                 "SetKinvis().");
        if (m_pressureNumBndCoeffCompSize == 0)
        {
            m_pressureUpdated = true;
            return;
        }

        // The fields on the boundary-adjacent elements take the interleave
        // width of the velocity.
        const unsigned int interleaveWidth =
            velocity.GetBlocks()[m_volumeGathers[0].srcBlock]
                .GetInterleaveWidth();
        SetRestrictedInterleaveWidth(m_restrictedVelocity, interleaveWidth);
        SetRestrictedInterleaveWidth(m_restrictedGradient, interleaveWidth);
        GatherVelocity(velocity);
        m_physDerivOp->Apply(m_restrictedVelocity, m_restrictedGradient);

        EvaluateFb(this->m_time);
        EvaluatePressure(velocity, this->m_kinvis, this->m_scale);
        ProjectPressure();
        m_pressureUpdated = true;
    }

    void v_ApplyPressure(
        LibUtilities::Field<TData, FieldState::Coeff> &out) override
    {
        if (!m_pressureUpdated || m_pressureNumBndCoeffCompSize == 0)
        {
            return;
        }

        for (unsigned int blk = 0; blk < out.GetBlocks().size(); ++blk)
        {
            const unsigned int streamID = blk + 1;
            // The targets index the block at its own interleave width.
            auto &outBlock                     = out.GetBlocks()[blk];
            const unsigned int interleaveWidth = outBlock.GetInterleaveWidth();
            auto outPtr =
                outBlock.template GetPtr<MemSpace, ReadWrite>(streamID);

            const size_t count = m_pressureCounts[blk];
            if (count > 0)
            {
                const size_t offset = m_pressureOffsets[blk];
                auto mapPtr =
                    TargetsAtInterleaveWidth(m_pressureMap[blk],
                                             interleaveWidth, streamID)
                        .template GetPtr<MemSpace, ReadOnly>(streamID) +
                    offset;
                auto coeffPtr =
                    m_pressureMappedCoeff[blk]
                        .template GetPtr<MemSpace, ReadOnly>(streamID) +
                    offset;
                if (m_pressureSignChange)
                {
                    auto signPtr =
                        m_pressureSign[blk].template GetPtr<MemSpace, ReadOnly>(
                            streamID) +
                        offset;
                    MultiRegions::detail::DirBndCondKernel<ExecSpace>(
                        count, signPtr, mapPtr, coeffPtr, outPtr, streamID);
                }
                else
                {
                    MultiRegions::detail::DirBndCondKernel<ExecSpace>(
                        count, mapPtr, coeffPtr, outPtr, streamID);
                }
            }
        }
    }

    void v_ApplyVelocity(
        LibUtilities::Field<TData, FieldState::Coeff> &inout) override
    {
        // A rank without regions has nothing to apply, and need not have
        // updated the pressure.
        if (m_velocityNumBndCoeffCompSize == 0)
        {
            return;
        }
        ASSERTL1(m_pressureUpdated,
                 "UpdatePressure must precede the velocity solve.");
        ASSERTL1(this->m_pressure,
                 "SetPressure must be called before the velocity solve.");
        ASSERTL1(this->m_kinvis > TData(0),
                 "HOutflow requires a positive kinematic viscosity, set by "
                 "SetKinvis().");
        EvaluateVelocity(*this->m_pressure, this->m_kinvis);
        ProjectVelocity();
        ApplyNeumannMaps(inout);
    }

    std::shared_ptr<MultiRegions::DirBndCondOp<TData>> v_GetPressureDirichletOp()
        override
    {
        return m_pressureAdapter;
    }

    std::shared_ptr<MultiRegions::NeuBndCondOp<TData>> v_GetVelocityNeumannOp()
        override
    {
        ASSERTL1(m_velocityContextSet,
                 "SetVelocityContext must be called before requesting the "
                 "HOutflow velocity adapter.");
        return m_velocityAdapter;
    }

    // Whether a boundary condition is part of the high-order outflow
    // condition: USERDEFINEDTYPE="HOutflow".
    static bool IsHOutflow(
        const SpatialDomains::BoundaryConditionShPtr &condition)
    {
        return StringUtils::IEquals(condition->GetUserDefined(),
                                    HOutflowBndCondOp<TData>::userDefinedType);
    }

    void BuildPressureBoundaryData()
    {
        const auto &bndExpansions =
            this->m_expansionList->GetBndCondExpansions();
        const auto &bndConditions = this->m_expansionList->GetBndConditions();
        const SpatialDomains::BoundaryConditions bcs(
            this->m_expansionList->GetSession(),
            this->m_expansionList->GetGraph());
        const auto &bregions = bcs.GetBoundaryRegions();
        for (const auto &[regionId, region] : bregions)
        {
            m_regionIds.push_back(regionId);
        }
        ASSERTL1(m_regionIds.size() == bndExpansions.size(),
                 "Boundary region and expansion counts differ.");

        std::vector<std::vector<bool>> isTypeByRegion(bndExpansions.size());
        std::vector<size_t> numBcExpCoeffs(bndExpansions.size());
        for (unsigned int n = 0; n < bndExpansions.size(); ++n)
        {
            const bool match = IsHOutflow(bndConditions[n]);
            ASSERTL0(!match || bndConditions[n]->GetBoundaryConditionType() ==
                                   SpatialDomains::eDirichlet,
                     "HOutflow pressure conditions must be Dirichlet.");
            isTypeByRegion[n] = {match};
            numBcExpCoeffs[n] = bndExpansions[n]->GetNcoeffs();
            if (!match)
            {
                continue;
            }

            auto bndExp = bndExpansions[n];
            bndExp->SetDataWarehouse();
            m_pressureConditionIndices.push_back(n);
            m_pressureBndExpLists.push_back(bndExp);
            m_pressureCoeffOffsets.push_back(m_pressureNumBndCoeffCompSize);
            m_pressureNumBndCoeffCompSize += bndExp->GetNcoeffs();

            const auto physBlocks =
                MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                    bndExp);
            const auto coeffBlocks =
                MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                    bndExp);
            m_pressureBndPhys.emplace_back("HOutflow pressure boundary phys",
                                           physBlocks, this->m_components, 1);
            m_pressureBndCoeffs.emplace_back(
                "HOutflow pressure boundary coeffs", coeffBlocks,
                this->m_components, 1);
            m_pressureBndPhys.back().template Initialize<MemSpace>(0.0);
            m_pressureBndCoeffs.back().template Initialize<MemSpace>(0.0);
            m_pressureFwdTransOps.push_back(
                MultiRegions::FwdTransBCOp<TData>::Create(
                    bndExp, this->m_components, ExecSpace::name));
        }

        if (m_pressureNumBndCoeffCompSize == 0)
        {
            return;
        }
        // The kernels are instantiated for two and three dimensions.
        ASSERTL0(m_nDim == 2 || m_nDim == 3,
                 "HOutflow is implemented for two and three dimensions.");
        auto maps =
            MultiRegions::detail::BuildCGBndCondCoeffMaps<ExecSpace, TData>(
                this->m_expansionList, this->m_components, bregions,
                isTypeByRegion, numBcExpCoeffs, m_pressureNumBndCoeffCompSize,
                [](unsigned int, const auto &, const auto &, const auto &) {});
        m_pressureSignChange  = maps.signChange[0];
        m_pressureSign        = std::move(maps.sign);
        m_pressureMappedCoeff = std::move(maps.bndCoeff);
        m_pressureCoeffSrc    = std::move(maps.bndCoeffSrc);
        m_pressureOffsets.resize(maps.compOffsets.size());
        m_pressureCounts.resize(maps.compCounts.size());
        for (size_t blk = 0; blk < m_pressureCounts.size(); ++blk)
        {
            m_pressureOffsets[blk] = maps.compOffsets[blk][0];
            m_pressureCounts[blk]  = maps.compCounts[blk][0];
        }
        m_pressureCompactCoeff =
            LibUtilities::MemoryRegion<TData>(m_pressureNumBndCoeffCompSize);
        // Allocated here, on the null stream, so the region block streams
        // that write it never race the allocation.
        m_pressureCompactCoeff.template GetPtr<MemSpace, WriteOnly>();
        m_pressureUpdateStreams =
            UpdateStreams(m_pressureMappedCoeff.size(), m_pressureBndCoeffs);

        m_pressureMap = BuildCoeffTargets(maps.map, this->m_expansionList);
    }

    void ValidateAndBuildVelocityBoundaryData(
        const std::vector<std::string> &components)
    {
        const SpatialDomains::BoundaryConditions bcs(
            this->m_expansionList->GetSession(),
            this->m_expansionList->GetGraph());
        const auto &bregions   = bcs.GetBoundaryRegions();
        const auto &conditions = bcs.GetBoundaryConditions();
        const auto &velocityExp =
            m_velocityExpansionList->GetBndCondExpansions();
        ASSERTL1(velocityExp.size() == m_regionIds.size(),
                 "Pressure and velocity boundary region counts differ.");

        std::set<unsigned int> selected(m_pressureConditionIndices.begin(),
                                        m_pressureConditionIndices.end());
        std::vector<std::vector<bool>> isTypeByRegion(m_regionIds.size());
        std::vector<size_t> numBcExpCoeffs(m_regionIds.size());

        for (unsigned int n = 0; n < m_regionIds.size(); ++n)
        {
            const auto conditionIt = conditions.find(m_regionIds[n]);
            ASSERTL0(conditionIt != conditions.end(),
                     "Unable to locate HOutflow boundary region.");
            bool anyVelocityTag      = false;
            bool allVelocityHOutflow = true;
            std::vector<LibUtilities::EquationSharedPtr> equations;
            std::vector<bool> componentMatches(m_nDim, false);
            for (unsigned int d = 0; d < m_nDim; ++d)
            {
                const auto componentIt =
                    conditionIt->second->find(components[d]);
                ASSERTL0(componentIt != conditionIt->second->end(),
                         "Unable to locate HOutflow velocity component.");
                const bool tagged = IsHOutflow(componentIt->second);
                const bool valid =
                    tagged && componentIt->second->GetBoundaryConditionType() ==
                                  SpatialDomains::eNeumann;
                anyVelocityTag      = anyVelocityTag || tagged;
                allVelocityHOutflow = allVelocityHOutflow && valid;
                componentMatches[d] = valid;
                equations.push_back(componentIt->second->GetEquation());
            }

            const bool pressureSelected = selected.count(n) != 0;
            ASSERTL0(!anyVelocityTag || pressureSelected,
                     "HOutflow velocity conditions require a paired "
                     "HOutflow pressure Dirichlet condition.");
            ASSERTL0(!pressureSelected || allVelocityHOutflow,
                     "HOutflow pressure conditions require HOutflow Neumann "
                     "conditions on every velocity component.");
            isTypeByRegion[n] = pressureSelected
                                    ? componentMatches
                                    : std::vector<bool>(m_nDim, false);
            numBcExpCoeffs[n] = velocityExp[n]->GetNcoeffs();
            if (!pressureSelected)
            {
                continue;
            }

            const unsigned int selectedId = m_velocityBndExpLists.size();
            auto bndExp                   = velocityExp[n];
            bndExp->SetDataWarehouse();
            auto pressureBndExp = m_pressureBndExpLists[selectedId];
            ASSERTL1(bndExp->GetTotPoints() == pressureBndExp->GetTotPoints(),
                     "HOutflow pressure and velocity boundary quadrature "
                     "must agree.");
            const auto pressurePhysBlocks =
                MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                    pressureBndExp);
            const auto velocityPhysBlocks =
                MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                    bndExp);
            ASSERTL1(pressurePhysBlocks.size() == velocityPhysBlocks.size(),
                     "HOutflow pressure and velocity boundary block layouts "
                     "must agree.");
            for (unsigned int blk = 0; blk < pressurePhysBlocks.size(); ++blk)
            {
                ASSERTL1(pressurePhysBlocks[blk].GetNumData() ==
                                 velocityPhysBlocks[blk].GetNumData() &&
                             pressurePhysBlocks[blk].GetNumElements() ==
                                 velocityPhysBlocks[blk].GetNumElements(),
                         "HOutflow pressure and velocity boundary block "
                         "shapes must agree.");
                const auto pressureExpansions =
                    MultiRegions::GetCollection(pressureBndExp, blk)
                        .GetExpVector();
                const auto velocityExpansions =
                    MultiRegions::GetCollection(bndExp, blk).GetExpVector();
                for (unsigned int e = 0; e < pressureExpansions.size(); ++e)
                {
                    ASSERTL1(
                        pressureExpansions[e]->GetGeom()->GetGlobalID() ==
                            velocityExpansions[e]->GetGeom()->GetGlobalID(),
                        "HOutflow pressure and velocity boundary element "
                        "ordering must agree.");
                }
            }
            m_velocityBndExpLists.push_back(bndExp);
            m_velocityCoeffOffsets.push_back(m_velocityNumBndCoeffCompSize);
            m_velocityNumBndCoeffCompSize += bndExp->GetNcoeffs();

            const auto physBlocks =
                MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                    bndExp);
            const auto coeffBlocks =
                MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                    bndExp);
            m_velocityBndPhys.emplace_back("HOutflow velocity boundary phys",
                                           physBlocks, components, 1);
            m_velocityBndCoeffs.emplace_back(
                "HOutflow velocity boundary coeffs", coeffBlocks, components,
                1);
            m_fbPhys.emplace_back("HOutflow f_b", physBlocks, components, 1);
            m_normals.emplace_back("HOutflow normals", physBlocks, components,
                                   1);
            m_energyMinusFb.emplace_back("HOutflow E minus f_b", physBlocks,
                                         components, 1);
            m_divergence.emplace_back("HOutflow divergence", physBlocks,
                                      this->m_components, 1);
            m_velocityBndPhys.back().template Initialize<MemSpace>(0.0);
            m_velocityBndCoeffs.back().template Initialize<MemSpace>(0.0);
            m_fbPhys.back().template Initialize<MemSpace>(0.0);
            m_normals.back().template Initialize<MemSpace>(0.0);
            m_energyMinusFb.back().template Initialize<MemSpace>(0.0);
            m_divergence.back().template Initialize<MemSpace>(0.0);

            auto expression = MultiRegions::ExpressionOp<TData>::Create(
                bndExp, components, ExecSpace::name);
            expression->SetExpressions(equations);
            m_fbExpressionOps.push_back(expression);
            m_velocityIProductOps.push_back(
                MultiRegions::IProductWRTBaseOp<TData>::Create(
                    bndExp, components, ExecSpace::name));

            Array<OneD, Array<OneD, double>> normals;
            this->m_expansionList->GetBoundaryNormals(n, normals);
            for (unsigned int blk = 0;
                 blk < m_normals.back().GetBlocks().size(); ++blk)
            {
                auto &normalBlock = m_normals.back().GetBlocks()[blk];
                auto ptr =
                    normalBlock
                        .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
                auto collection = MultiRegions::GetCollection(bndExp, blk);
                const auto &expansions = collection.GetExpVector();
                for (unsigned int e = 0; e < expansions.size(); ++e)
                {
                    const auto offset =
                        bndExp->GetPhys_Offset(expansions[e]->GetElmtId());
                    for (unsigned int d = 0; d < m_nDim; ++d)
                    {
                        for (unsigned int p = 0; p < normalBlock.GetNumData();
                             ++p)
                        {
                            ptr[d * normalBlock.CompSize() +
                                e * normalBlock.GetNumData() + p] =
                                static_cast<TData>(normals[d][offset + p]);
                        }
                    }
                }
            }
        }

        ASSERTL1(m_velocityBndExpLists.size() == m_pressureBndExpLists.size(),
                 "HOutflow pressure and velocity region sets differ.");
        // The assembly map of a velocity component other than the one of
        // m_velocityExpansionList is built from a ContField, collectively: all
        // ranks enter BuildCGBndCondCoeffMaps() when any rank has an HOutflow
        // region, even without coefficients of its own.
        size_t hasAnyVelocityBndCoeff = m_velocityNumBndCoeffCompSize;
        this->m_expansionList->GetSession()->GetComm()->GetRowComm()->AllReduce(
            hasAnyVelocityBndCoeff, LibUtilities::ReduceMax);
        if (hasAnyVelocityBndCoeff == 0)
        {
            return;
        }

        auto maps =
            MultiRegions::detail::BuildCGBndCondCoeffMaps<ExecSpace, TData>(
                m_velocityExpansionList, components, bregions, isTypeByRegion,
                numBcExpCoeffs, m_velocityNumBndCoeffCompSize,
                [](unsigned int, const auto &, const auto &, const auto &) {});
        m_velocityAnySignChange    = maps.anySignChange;
        m_velocitySignChange       = std::move(maps.signChange);
        m_velocitySign             = std::move(maps.uniqueSign);
        m_velocityMappedCoeff      = std::move(maps.uniqueBndCoeff);
        m_velocityCoeffSrc         = std::move(maps.uniqueBndCoeffSrc);
        m_velocityCompOffsets      = std::move(maps.compUniqueOffsets);
        m_velocityCompCounts       = std::move(maps.compUniqueCounts);
        m_velocityGroupOffset      = std::move(maps.groupOffset);
        m_velocityGroupSrc         = std::move(maps.groupSrc);
        m_velocityGroupSign        = std::move(maps.groupSign);
        m_velocityGroupValue       = std::move(maps.groupValue);
        m_velocityCompGroupOffsets = std::move(maps.compGroupOffsets);
        m_velocityCompGroupCounts  = std::move(maps.compGroupCounts);
        m_velocityCompactCoeff     = LibUtilities::MemoryRegion<TData>(
            m_nDim * m_velocityNumBndCoeffCompSize);
        // Allocated here, on the null stream, so the region block streams
        // that write it never race the allocation. A rank without velocity
        // coefficients of its own never uses it.
        if (m_velocityNumBndCoeffCompSize > 0)
        {
            m_velocityCompactCoeff.template GetPtr<MemSpace, WriteOnly>();
        }
        m_velocityUpdateStreams =
            UpdateStreams(m_velocityMappedCoeff.size(), m_velocityBndCoeffs);

        m_velocityMap =
            BuildCoeffTargets(maps.uniqueMap, m_velocityExpansionList);
        m_velocityGroupTarget =
            BuildCoeffTargets(maps.groupTarget, m_velocityExpansionList);
    }

    // The streams of @p nDomainBlocks domain blocks and of the blocks of
    // every region field of @p fields.
    static std::vector<unsigned int> UpdateStreams(
        const size_t nDomainBlocks,
        std::vector<LibUtilities::Field<TData, FieldState::Coeff>> &fields)
    {
        size_t nUpdateStreams = nDomainBlocks;
        for (auto &field : fields)
        {
            nUpdateStreams = std::max(nUpdateStreams, field.GetBlocks().size());
        }
        std::vector<unsigned int> streams;
        for (unsigned int id = 1; id <= nUpdateStreams; ++id)
        {
            streams.push_back(id);
        }
        return streams;
    }

    // The targets of every domain block of @p expansionList from the index
    // into the block at interleave width 1 BuildCGBndCondCoeffMaps() returns,
    // which is read back to the host once. The indices into the blocks are
    // built from it on first use.
    std::vector<CoeffTargets> BuildCoeffTargets(
        std::vector<LibUtilities::MemoryRegion<size_t>> &unitWidthIndex,
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        const auto coeffBlocks =
            MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                expansionList);
        std::vector<CoeffTargets> targets(unitWidthIndex.size());
        for (unsigned int blk = 0; blk < unitWidthIndex.size(); ++blk)
        {
            targets[blk].nCoeffs  = coeffBlocks[blk].GetNumData();
            const size_t nTargets = unitWidthIndex[blk].size();
            if (nTargets > 0)
            {
                auto elmtCoeff =
                    unitWidthIndex[blk]
                        .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
                targets[blk].elmtCoeff.assign(elmtCoeff, elmtCoeff + nTargets);
            }
        }
        return targets;
    }

    void BuildBoundaryElementData(const std::vector<std::string> &components)
    {
        if (m_pressureNumBndCoeffCompSize == 0)
        {
            return;
        }
        struct VolumeLocation
        {
            unsigned int block;
            size_t element;
            LocalRegions::ExpansionSharedPtr expansion;
        };

        auto BuildLocations = [](const MultiRegions::ExpListSharedPtr &list,
                                 auto &locations, auto &owners) {
            const auto blocks =
                MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(list);
            for (unsigned int blk = 0; blk < blocks.size(); ++blk)
            {
                auto collection        = MultiRegions::GetCollection(list, blk);
                const auto &expansions = collection.GetExpVector();
                for (unsigned int e = 0; e < expansions.size(); ++e)
                {
                    const int geometry =
                        expansions[e]->GetGeom()->GetGlobalID();
                    locations[geometry] = {blk, e, expansions[e]};
                    for (unsigned int f = 0; f < expansions[e]->GetNtraces();
                         ++f)
                    {
                        owners[expansions[e]->GetGeom()->GetTid(f)] = {geometry,
                                                                       f};
                    }
                }
            }
        };

        std::map<int, VolumeLocation> velocityLocations;
        std::map<int, std::pair<int, unsigned int>> velocityOwners;
        BuildLocations(m_velocityExpansionList, velocityLocations,
                       velocityOwners);

        // The velocity elements adjacent to a region, matched to its
        // elements by geometry id. The pressure and velocity boundary
        // expansion lists agree element for element
        // (ValidateAndBuildVelocityBoundaryData()).
        std::set<int> selectedVelocityGeometry;
        for (unsigned int n = 0; n < m_pressureBndExpLists.size(); ++n)
        {
            const auto blocks =
                MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                    m_pressureBndExpLists[n]);
            for (unsigned int blk = 0; blk < blocks.size(); ++blk)
            {
                auto collection =
                    MultiRegions::GetCollection(m_pressureBndExpLists[n], blk);
                const auto &expansions = collection.GetExpVector();
                for (unsigned int e = 0; e < expansions.size(); ++e)
                {
                    const auto owner = velocityOwners.find(
                        expansions[e]->GetGeom()->GetGlobalID());
                    ASSERTL1(owner != velocityOwners.end(),
                             "Unable to locate an element adjacent to "
                             "HOutflow.");
                    selectedVelocityGeometry.insert(owner->second.first);
                }
            }
        }

        SpatialDomains::ExpansionInfoMap restrictedInfo;
        const auto &velocityInfo =
            m_velocityExpansionList->GetGraph()->GetExpansionInfo(
                components[0]);
        for (const auto geometry : selectedVelocityGeometry)
        {
            const auto it = velocityInfo.find(geometry);
            ASSERTL1(it != velocityInfo.end(),
                     "Unable to locate HOutflow velocity expansion data.");
            restrictedInfo.emplace(*it);
        }
        m_restrictedVelocityExpansionList =
            MemoryManager<MultiRegions::ExpList>::AllocateSharedPtr(
                m_velocityExpansionList->GetSession(), restrictedInfo, false);
        m_restrictedVelocityExpansionList->SetDataWarehouse();
        const auto restrictedBlocks =
            MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                m_restrictedVelocityExpansionList);
        m_restrictedVelocity = LibUtilities::Field<TData, FieldState::Phys>(
            "HOutflow restricted velocity", restrictedBlocks, components, 1);
        std::vector<std::string> gradientComponents(m_nDim * m_nDim);
        for (unsigned int d = 0; d < m_nDim; ++d)
        {
            for (unsigned int j = 0; j < m_nDim; ++j)
            {
                gradientComponents[d * m_nDim + j] =
                    "d" + components[d] + "d" + std::to_string(j);
            }
        }
        m_restrictedGradient = LibUtilities::Field<TData, FieldState::Phys>(
            "HOutflow restricted velocity gradient", restrictedBlocks,
            gradientComponents, 1);
        m_restrictedVelocity.template Initialize<MemSpace>(0.0);
        m_restrictedGradient.template Initialize<MemSpace>(0.0);

        std::map<int, VolumeLocation> restrictedLocations;
        std::map<int, std::pair<int, unsigned int>> unusedOwners;
        BuildLocations(m_restrictedVelocityExpansionList, restrictedLocations,
                       unusedOwners);
        // Per pair of source and destination block, the destination element
        // of every element gathered, mapped to its source element, in
        // increasing destination element.
        std::map<std::pair<unsigned int, unsigned int>,
                 std::map<size_t, size_t>>
            gathers;
        for (const int geometry : selectedVelocityGeometry)
        {
            const auto &src = velocityLocations.at(geometry);
            const auto &dst = restrictedLocations.at(geometry);
            gathers[{src.block, dst.block}][dst.element] = src.element;
        }
        for (auto &[blocks, map] : gathers)
        {
            VolumeGather gather;
            gather.srcBlock = blocks.first;
            gather.dstBlock = blocks.second;
            gather.nData    = restrictedBlocks[blocks.second].GetNumData();
            for (const auto &[dstElmt, srcElmt] : map)
            {
                gather.dstElmt.push_back(dstElmt);
                gather.srcElmt.push_back(srcElmt);
            }
            m_volumeGathers.push_back(std::move(gather));
        }

        m_physDerivOp = MultiRegions::PhysDerivOp<TData>::Create(
            m_restrictedVelocityExpansionList, components, ExecSpace::name);

        // Per region, the extractions of the velocity, of its gradient and
        // of the pressure. The pressure and velocity boundary expansion
        // lists have the same points (ValidateAndBuildVelocityBoundaryData()),
        // so the extracted fields pair with m_normals and m_fbPhys point by
        // point. The velocity is extracted from the volume velocity field,
        // and its gradient from m_restrictedGradient: the expansion list of
        // the latter, built from an ExpansionInfoMap, has no mesh graph, so
        // its extraction takes that of the velocity field.
        for (unsigned int n = 0; n < m_pressureBndExpLists.size(); ++n)
        {
            const auto bndPhysBlocks =
                MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                    m_pressureBndExpLists[n]);

            m_bndVelocityExtractOps.push_back(
                std::make_shared<MultiRegions::detail::PhysTraceExtractBndCond<
                    ExecSpace, TData>>(m_velocityExpansionList, components,
                                       m_pressureBndExpLists[n],
                                       ExecSpace::name));
            m_bndVelocity.emplace_back("HOutflow boundary velocity",
                                       bndPhysBlocks, components, 1);
            m_bndVelocity.back().template Initialize<MemSpace>(0.0);

            m_bndGradientExtractOps.push_back(
                std::make_shared<MultiRegions::detail::PhysTraceExtractBndCond<
                    ExecSpace, TData>>(
                    m_restrictedVelocityExpansionList, gradientComponents,
                    m_pressureBndExpLists[n], ExecSpace::name,
                    m_velocityExpansionList->GetGraph()));
            m_bndGradient.emplace_back("HOutflow boundary velocity gradient",
                                       bndPhysBlocks, gradientComponents, 1);
            m_bndGradient.back().template Initialize<MemSpace>(0.0);

            m_bndPressureExtractOps.push_back(
                std::make_shared<MultiRegions::detail::PhysTraceExtractBndCond<
                    ExecSpace, TData>>(
                    this->m_expansionList, this->m_components,
                    m_pressureBndExpLists[n], ExecSpace::name));
        }
    }

    // Point of a block at interleave width @p interleaveWidth of every point
    // @p gather copies of the elements @p elmts, its source or its
    // destination elements: point q of element e lies at
    // (e / interleaveWidth) * interleaveWidth * nData + q * interleaveWidth
    // + e % interleaveWidth. The points are in the order of the destination
    // points in a destination block at interleave width
    // @p dstInterleaveWidth, the lane varying fastest, so that neighbouring
    // threads write neighbouring points, and read neighbouring points where
    // neighbouring destination elements come from neighbouring source
    // elements.
    static std::vector<size_t> PointIndex(const VolumeGather &gather,
                                          const std::vector<size_t> &elmts,
                                          const unsigned int interleaveWidth,
                                          const unsigned int dstInterleaveWidth)
    {
        std::vector<size_t> index;
        index.reserve(elmts.size() * gather.nData);
        // The destination elements increase, so those of one destination
        // group are consecutive.
        for (size_t first = 0; first < elmts.size();)
        {
            const size_t dstGroup = gather.dstElmt[first] / dstInterleaveWidth;
            size_t last           = first + 1;
            while (last < elmts.size() &&
                   gather.dstElmt[last] / dstInterleaveWidth == dstGroup)
            {
                ++last;
            }
            for (size_t q = 0; q < gather.nData; ++q)
            {
                for (size_t k = first; k < last; ++k)
                {
                    const size_t group = elmts[k] / interleaveWidth;
                    const size_t lane  = elmts[k] % interleaveWidth;
                    index.push_back(group * interleaveWidth * gather.nData +
                                    q * interleaveWidth + lane);
                }
            }
            first = last;
        }
        return index;
    }

    // Coefficient of a domain block at interleave width @p interleaveWidth of
    // every target of @p targets: coefficient c of element e lies at
    // (e / interleaveWidth) * interleaveWidth * nCoeffs + c * interleaveWidth
    // + e % interleaveWidth.
    static std::vector<size_t> TargetIndex(const CoeffTargets &targets,
                                           const unsigned int interleaveWidth)
    {
        std::vector<size_t> index(targets.elmtCoeff.size());
        for (size_t k = 0; k < index.size(); ++k)
        {
            const size_t elmt  = targets.elmtCoeff[k] / targets.nCoeffs;
            const size_t coeff = targets.elmtCoeff[k] % targets.nCoeffs;
            const size_t group = elmt / interleaveWidth;
            const size_t lane  = elmt % interleaveWidth;
            index[k]           = group * interleaveWidth * targets.nCoeffs +
                       coeff * interleaveWidth + lane;
        }
        return index;
    }

    // The one of the indices of @p indices for a block at interleave width
    // @p interleaveWidth, built from the host index @p buildIndex returns for
    // that width, on stream @p streamID, the first time it is requested.
    template <typename BuildIndexFn>
    static LibUtilities::MemoryRegion<size_t> &IndexAtInterleaveWidth(
        InterleaveWidthIndices &indices, const unsigned int interleaveWidth,
        const unsigned int streamID, BuildIndexFn &&buildIndex)
    {
        ASSERTL0(interleaveWidth == 1u || interleaveWidth == m_interleaveWidth,
                 "HOutflow takes fields at interleave width 1 or at the "
                 "interleave width of its execution space.");
        const bool isUnitWidth = interleaveWidth == 1u;
        auto &index   = isUnitWidth ? indices.index : indices.interleavedIndex;
        bool &isBuilt = isUnitWidth ? indices.isBuiltIndex
                                    : indices.isBuiltInterleavedIndex;
        if (!isBuilt)
        {
            index = LibUtilities::MemoryRegion<size_t>::template FromVector<
                MemSpace, size_t>(buildIndex(interleaveWidth), streamID);
            isBuilt = true;
        }
        return index;
    }

    // The indices of @p gather for a source block at interleave width
    // @p srcInterleaveWidth and a destination block at
    // @p dstInterleaveWidth, built on stream @p streamID on first use.
    static GatherIndices &GatherIndicesAtInterleaveWidths(
        VolumeGather &gather, const unsigned int srcInterleaveWidth,
        const unsigned int dstInterleaveWidth, const unsigned int streamID)
    {
        ASSERTL0((srcInterleaveWidth == 1u ||
                  srcInterleaveWidth == m_interleaveWidth) &&
                     (dstInterleaveWidth == 1u ||
                      dstInterleaveWidth == m_interleaveWidth),
                 "HOutflow takes fields at interleave width 1 or at the "
                 "interleave width of its execution space.");
        auto [it, isNew] = gather.indices.try_emplace(
            std::make_pair(srcInterleaveWidth, dstInterleaveWidth));
        if (isNew)
        {
            it->second.srcIndex =
                LibUtilities::MemoryRegion<size_t>::template FromVector<
                    MemSpace, size_t>(PointIndex(gather, gather.srcElmt,
                                                 srcInterleaveWidth,
                                                 dstInterleaveWidth),
                                      streamID);
            it->second.dstIndex =
                LibUtilities::MemoryRegion<size_t>::template FromVector<
                    MemSpace, size_t>(PointIndex(gather, gather.dstElmt,
                                                 dstInterleaveWidth,
                                                 dstInterleaveWidth),
                                      streamID);
        }
        return it->second;
    }

    // Lay @p field, on the boundary-adjacent elements, out at interleave
    // width @p interleaveWidth, that of the velocity, so that the gathers
    // write it and the operators on those elements read it without
    // reshaping. A field changing width is zeroed, so that its padding
    // stays zero: every element of it is written again before it is read.
    static void SetRestrictedInterleaveWidth(
        LibUtilities::Field<TData, FieldState::Phys> &field,
        const unsigned int interleaveWidth)
    {
        ASSERTL0(interleaveWidth == 1u || interleaveWidth == m_interleaveWidth,
                 "HOutflow takes fields at interleave width 1 or at the "
                 "interleave width of its execution space.");
        bool isAtWidth = true;
        for (auto &block : field.GetBlocks())
        {
            isAtWidth =
                isAtWidth && block.GetInterleaveWidth() == interleaveWidth;
        }
        if (!isAtWidth)
        {
            Math::zero<ExecSpace>(field);
            field.SetInterleaveWidth(interleaveWidth);
        }
    }

    // The index of @p targets into a domain block at interleave width
    // @p interleaveWidth, built on stream @p streamID on first use.
    static LibUtilities::MemoryRegion<size_t> &TargetsAtInterleaveWidth(
        CoeffTargets &targets, const unsigned int interleaveWidth,
        const unsigned int streamID)
    {
        return IndexAtInterleaveWidth(
            targets.indices, interleaveWidth, streamID,
            [&targets](const unsigned int blockInterleaveWidth) {
                return TargetIndex(targets, blockInterleaveWidth);
            });
    }

    // Copy the boundary-adjacent elements of the velocity into
    // m_restrictedVelocity, each block at its own interleave width. Each
    // gather runs on the stream of its destination block, after the work
    // queued on the stream of its source block, and the source streams wait
    // on the gathers in turn, without the host waiting on either.
    void GatherVelocity(LibUtilities::Field<TData, FieldState::Phys> &velocity)
    {
        for (auto &gather : m_volumeGathers)
        {
            const unsigned int srcStreamID = gather.srcBlock + 1;
            const unsigned int streamID    = gather.dstBlock + 1;
            auto &srcBlock = velocity.GetBlocks()[gather.srcBlock];
            auto &dstBlock = m_restrictedVelocity.GetBlocks()[gather.dstBlock];

            RecordStreamEvent<ExecSpace>(srcStreamID);
            SetStreamDependencies<ExecSpace>(streamID, srcStreamID);

            auto &indices = GatherIndicesAtInterleaveWidths(
                gather, srcBlock.GetInterleaveWidth(),
                dstBlock.GetInterleaveWidth(), streamID);
            auto src = srcBlock.template GetPtr<MemSpace, ReadOnly>(streamID);
            // The points of the block this gather does not write are
            // padding, which stays zero, or written by another gather.
            auto dst = dstBlock.template GetPtr<MemSpace, ReadWrite>(streamID);
            auto srcIdx =
                indices.srcIndex.template GetPtr<MemSpace, ReadOnly>(streamID);
            auto dstIdx =
                indices.dstIndex.template GetPtr<MemSpace, ReadOnly>(streamID);
            HOutflowVolumeGatherKernel<ExecSpace>(
                gather.srcElmt.size() * gather.nData, m_nDim,
                srcBlock.CompSize(), dstBlock.CompSize(), srcIdx, dstIdx, src,
                dst, streamID);
            RecordStreamEvent<ExecSpace>(streamID);
        }

        for (const auto &gather : m_volumeGathers)
        {
            const unsigned int srcStreamID = gather.srcBlock + 1;
            const unsigned int streamID    = gather.dstBlock + 1;
            SetStreamDependencies<ExecSpace>(srcStreamID, streamID);
        }
    }

    void EvaluateFb(const TData time)
    {
        for (unsigned int n = 0; n < m_fbPhys.size(); ++n)
        {
            m_fbExpressionOps[n]->SetTime(time);
            m_fbExpressionOps[n]->Apply(m_fbPhys[n], m_fbPhys[n]);
        }
    }

    // Evaluate the pressure condition on every region from the velocity
    // field @p velocity and the gradient in m_restrictedGradient.
    void EvaluatePressure(
        LibUtilities::Field<TData, FieldState::Phys> &velocity,
        const TData kinvis, const TData scale)
    {
        for (unsigned int n = 0; n < m_pressureBndPhys.size(); ++n)
        {
            m_bndVelocityExtractOps[n]->Apply(velocity, m_bndVelocity[n]);
            m_bndGradientExtractOps[n]->Apply(m_restrictedGradient,
                                              m_bndGradient[n]);

            for (unsigned int blk = 0;
                 blk < m_pressureBndPhys[n].GetBlocks().size(); ++blk)
            {
                // The extractions, m_normals and m_fbPhys are ordered before
                // the stream of the block.
                const unsigned int streamID = blk + 1;
                auto &velocityBlock         = m_bndVelocity[n].GetBlocks()[blk];
                auto &gradientBlock         = m_bndGradient[n].GetBlocks()[blk];
                auto &normalBlock           = m_normals[n].GetBlocks()[blk];
                auto &fbBlock               = m_fbPhys[n].GetBlocks()[blk];
                auto &pressureBlock = m_pressureBndPhys[n].GetBlocks()[blk];
                auto &energyBlock   = m_energyMinusFb[n].GetBlocks()[blk];
                auto &divBlock      = m_divergence[n].GetBlocks()[blk];
                const size_t nPoints =
                    pressureBlock.GetNumElements() * pressureBlock.GetNumData();
                const size_t compSize = velocityBlock.CompSize();
                auto velocityPtr =
                    velocityBlock.template GetPtr<MemSpace, ReadOnly>(streamID);
                auto gradientPtr =
                    gradientBlock.template GetPtr<MemSpace, ReadOnly>(streamID);
                auto normalPtr =
                    normalBlock.template GetPtr<MemSpace, ReadOnly>(streamID);
                auto fbPtr =
                    fbBlock.template GetPtr<MemSpace, ReadOnly>(streamID);
                auto pressurePtr =
                    pressureBlock.template GetPtr<MemSpace, WriteOnly>(
                        streamID);
                auto energyPtr =
                    energyBlock.template GetPtr<MemSpace, WriteOnly>(streamID);
                auto divPtr =
                    divBlock.template GetPtr<MemSpace, WriteOnly>(streamID);
                if (m_nDim == 2)
                {
                    HOutflowPressureKernel<ExecSpace, 2>(
                        nPoints, compSize, velocityPtr, gradientPtr, normalPtr,
                        fbPtr, kinvis, scale, m_theta, m_alpha1, m_alpha2, m_U0,
                        m_delta, pressurePtr, energyPtr, divPtr, streamID);
                }
                else
                {
                    HOutflowPressureKernel<ExecSpace, 3>(
                        nPoints, compSize, velocityPtr, gradientPtr, normalPtr,
                        fbPtr, kinvis, scale, m_theta, m_alpha1, m_alpha2, m_U0,
                        m_delta, pressurePtr, energyPtr, divPtr, streamID);
                }
            }
        }
    }

    void ProjectPressure()
    {
        ProjectAndGatherCoeffs(
            m_pressureBndCoeffs, m_pressureCoeffOffsets,
            m_pressureNumBndCoeffCompSize, m_pressureCompactCoeff,
            m_pressureCoeffSrc, m_pressureMappedCoeff, nullptr, nullptr,
            m_pressureUpdateStreams, [this](unsigned int n) {
                // The forward transform assumes a zero output, of which
                // it sets the vertex modes first.
                Math::zero<ExecSpace>(m_pressureBndCoeffs[n]);
                m_pressureFwdTransOps[n]->Apply(m_pressureBndPhys[n],
                                                m_pressureBndCoeffs[n]);
            });
    }

    void EvaluateVelocity(
        LibUtilities::Field<TData, FieldState::Phys> &pressure,
        const TData kinvis)
    {
        for (unsigned int n = 0; n < m_velocityBndPhys.size(); ++n)
        {
            m_bndPressureExtractOps[n]->Apply(pressure, m_pressureBndPhys[n]);

            for (unsigned int blk = 0;
                 blk < m_velocityBndPhys[n].GetBlocks().size(); ++blk)
            {
                // The extraction, m_normals, m_energyMinusFb and m_divergence
                // are ordered before the stream of the block.
                const unsigned int streamID = blk + 1;
                auto &pressureBlock = m_pressureBndPhys[n].GetBlocks()[blk];
                auto &normalBlock   = m_normals[n].GetBlocks()[blk];
                auto &energyBlock   = m_energyMinusFb[n].GetBlocks()[blk];
                auto &divBlock      = m_divergence[n].GetBlocks()[blk];
                auto &outBlock      = m_velocityBndPhys[n].GetBlocks()[blk];
                const size_t nPoints =
                    outBlock.GetNumElements() * outBlock.GetNumData();
                const size_t compSize = outBlock.CompSize();
                auto pressurePtr =
                    pressureBlock.template GetPtr<MemSpace, ReadOnly>(streamID);
                auto normalPtr =
                    normalBlock.template GetPtr<MemSpace, ReadOnly>(streamID);
                auto energyPtr =
                    energyBlock.template GetPtr<MemSpace, ReadOnly>(streamID);
                auto divPtr =
                    divBlock.template GetPtr<MemSpace, ReadOnly>(streamID);
                auto outPtr =
                    outBlock.template GetPtr<MemSpace, WriteOnly>(streamID);
                if (m_nDim == 2)
                {
                    HOutflowVelocityKernel<ExecSpace, 2>(
                        nPoints, compSize, pressurePtr, normalPtr, energyPtr,
                        divPtr, kinvis, outPtr, streamID);
                }
                else
                {
                    HOutflowVelocityKernel<ExecSpace, 3>(
                        nPoints, compSize, pressurePtr, normalPtr, energyPtr,
                        divPtr, kinvis, outPtr, streamID);
                }
            }
        }
    }

    void ProjectVelocity()
    {
        ProjectAndGatherCoeffs(
            m_velocityBndCoeffs, m_velocityCoeffOffsets,
            m_velocityNumBndCoeffCompSize, m_velocityCompactCoeff,
            m_velocityCoeffSrc, m_velocityMappedCoeff, &m_velocityGroupSrc,
            &m_velocityGroupValue, m_velocityUpdateStreams,
            [this](unsigned int n) {
                m_velocityIProductOps[n]->Apply(m_velocityBndPhys[n],
                                                m_velocityBndCoeffs[n]);
            });
    }

    // Project every region with project(n), on the streams of its blocks,
    // copying every block into the array compact on the same stream as soon
    // as it is projected. Then gather compact into every domain block on the
    // stream of the block. The streams @p updateStreams of the region and
    // domain blocks wait on one another through m_opStreamID before and
    // after the gathers.
    template <typename ProjectFn>
    void ProjectAndGatherCoeffs(
        std::vector<LibUtilities::Field<TData, FieldState::Coeff>> &fields,
        const std::vector<size_t> &offsets, const size_t compSize,
        LibUtilities::MemoryRegion<TData> &compact,
        std::vector<LibUtilities::MemoryRegion<size_t>> &coefficientSources,
        std::vector<LibUtilities::MemoryRegion<TData>> &mappedCoefficients,
        std::vector<LibUtilities::MemoryRegion<size_t>> *groupSources,
        std::vector<LibUtilities::MemoryRegion<TData>> *groupValues,
        const std::vector<unsigned int> &updateStreams, ProjectFn &&project)
    {
        if (compSize == 0)
        {
            return;
        }
        const unsigned int nComp = fields[0].GetNumComponents();
        auto compactPtr =
            compact.template GetPtr<MemSpace, WriteOnly>(m_opStreamID);

        for (unsigned int n = 0; n < fields.size(); ++n)
        {
            project(n);

            size_t regionOffset = 0;
            for (unsigned int blk = 0; blk < fields[n].GetBlocks().size();
                 ++blk)
            {
                const unsigned int streamID = blk + 1;
                auto &block                 = fields[n].GetBlocks()[blk];
                auto src = block.template GetPtr<MemSpace, ReadOnly>(streamID);
                const size_t blockLength =
                    block.GetNumElements() * block.GetNumData();

                // On the stream that projected the block.
                for (unsigned int c = 0; c < nComp; ++c)
                {
                    Math::copyKernel<ExecSpace>(
                        blockLength, src + c * block.CompSize(),
                        compactPtr + c * compSize + offsets[n] + regionOffset,
                        streamID);
                }
                regionOffset += blockLength;
            }
        }

        // Gather compact into the domain blocks v_Apply() and
        // ApplyNeumannMaps() read, each on the stream of its block. Every
        // gather reads entries written on any region block's stream, so the
        // block streams first wait on one another through m_opStreamID.
        JoinStreams<ExecSpace>(m_opStreamID, updateStreams);
        ReleaseStreams<ExecSpace>(m_opStreamID, updateStreams);

        for (unsigned int blk = 0; blk < mappedCoefficients.size(); ++blk)
        {
            const unsigned int streamID = blk + 1;
            MultiRegions::detail::CGBndCondGatherKernel<ExecSpace>(
                coefficientSources[blk].size(),
                coefficientSources[blk].template GetPtr<MemSpace, ReadOnly>(
                    streamID),
                compactPtr,
                mappedCoefficients[blk].template GetPtr<MemSpace, WriteOnly>(
                    streamID),
                streamID);
            if (groupSources)
            {
                MultiRegions::detail::CGBndCondGatherKernel<ExecSpace>(
                    (*groupSources)[blk].size(),
                    (*groupSources)[blk].template GetPtr<MemSpace, ReadOnly>(
                        streamID),
                    compactPtr,
                    (*groupValues)[blk].template GetPtr<MemSpace, WriteOnly>(
                        streamID),
                    streamID);
            }
        }

        // The next projection overwrites compact on the region block
        // streams, which wait on the gathers in turn.
        JoinStreams<ExecSpace>(m_opStreamID, updateStreams);
        ReleaseStreams<ExecSpace>(m_opStreamID, updateStreams);
    }

    void ApplyNeumannMaps(LibUtilities::Field<TData, FieldState::Coeff> &inout)
    {
        for (unsigned int blk = 0; blk < inout.GetBlocks().size(); ++blk)
        {
            const unsigned int streamID = blk + 1;
            // The targets index the block at its own interleave width.
            auto &block                        = inout.GetBlocks()[blk];
            const unsigned int interleaveWidth = block.GetInterleaveWidth();
            auto ptr = block.template GetPtr<MemSpace, ReadWrite>(streamID);
            auto map = TargetsAtInterleaveWidth(m_velocityMap[blk],
                                                interleaveWidth, streamID)
                           .template GetPtr<MemSpace, ReadOnly>(streamID);
            auto coeff =
                m_velocityMappedCoeff[blk].template GetPtr<MemSpace, ReadOnly>(
                    streamID);
            const TData *sign =
                m_velocityAnySignChange
                    ? m_velocitySign[blk].template GetPtr<MemSpace, ReadOnly>(
                          streamID)
                    : nullptr;
            for (unsigned int d = 0; d < m_nDim; ++d)
            {
                const size_t offset = m_velocityCompOffsets[blk][d];
                const size_t count  = m_velocityCompCounts[blk][d];
                if (count > 0)
                {
                    if (m_velocitySignChange[d])
                    {
                        MultiRegions::detail::NeuBndCondKernel<ExecSpace>(
                            count, sign + offset, map + offset, coeff + offset,
                            ptr + d * block.CompSize(), streamID);
                    }
                    else
                    {
                        MultiRegions::detail::NeuBndCondKernel<ExecSpace>(
                            count, map + offset, coeff + offset,
                            ptr + d * block.CompSize(), streamID);
                    }
                }

                const size_t groupCount = m_velocityCompGroupCounts[blk][d];
                if (groupCount > 0)
                {
                    const size_t groupOffset =
                        m_velocityCompGroupOffsets[blk][d];
                    auto groupOffsets =
                        m_velocityGroupOffset[blk]
                            .template GetPtr<MemSpace, ReadOnly>(streamID) +
                        groupOffset;
                    auto groupTargets =
                        TargetsAtInterleaveWidth(m_velocityGroupTarget[blk],
                                                 interleaveWidth, streamID)
                            .template GetPtr<MemSpace, ReadOnly>(streamID) +
                        groupOffset;
                    auto groupValues =
                        m_velocityGroupValue[blk]
                            .template GetPtr<MemSpace, ReadOnly>(streamID);
                    if (m_velocitySignChange[d])
                    {
                        auto groupSigns =
                            m_velocityGroupSign[blk]
                                .template GetPtr<MemSpace, ReadOnly>(streamID);
                        MultiRegions::detail::NeuBndCondGroupKernel<ExecSpace>(
                            groupCount, groupOffsets, groupTargets, groupSigns,
                            groupValues, ptr + d * block.CompSize(), streamID);
                    }
                    else
                    {
                        MultiRegions::detail::NeuBndCondGroupKernel<ExecSpace>(
                            groupCount, groupOffsets, groupTargets, groupValues,
                            ptr + d * block.CompSize(), streamID);
                    }
                }
            }
        }
    }
};

} // namespace Nektar::detail
