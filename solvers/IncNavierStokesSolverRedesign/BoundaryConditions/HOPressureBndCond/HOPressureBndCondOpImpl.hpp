///////////////////////////////////////////////////////////////////////////////
//
// File: HOPressureBndCondOpImpl.hpp
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
// Description: Higher-order pressure boundary condition implementation.
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

#include <IncNavierStokesSolverRedesign/BoundaryConditions/HOPressureBndCond/HOPressureBndCondKernels.hpp>
#include <IncNavierStokesSolverRedesign/BoundaryConditions/HOPressureBndCond/HOPressureBndCondOp.hpp>

#include <LibUtilities/Backends/Backends.hpp>
#include <LibUtilities/BasicUtils/Math/Math.hpp>
#include <MultiRegions/BndCondOps/Common/CGBndCondKernels.hpp>
#include <MultiRegions/BndCondOps/Common/CGBndCondMapBuilder.hpp>
#include <MultiRegions/BndCondOps/Common/PhysTraceExtractBndCond.hpp>
#include <MultiRegions/BndCondOps/FwdTransBC/FwdTransBCOp.hpp>
#include <MultiRegions/BndCondOps/NeuBndCond/NeuBndCondKernels.hpp>
#include <MultiRegions/ElmtOps/BwdTrans/BwdTransOp.hpp>
#include <MultiRegions/ElmtOps/CurlCurl/CurlCurlOp.hpp>
#include <MultiRegions/ElmtOps/Expression/ExpressionOp.hpp>
#include <MultiRegions/ElmtOps/IProductWRTBase/IProductWRTBaseOp.hpp>

namespace Nektar::detail
{

/**
 * @brief Implementation of HOPressureBndCondOp.
 *
 * @details
 * Update() gathers the velocity and the flux on the volume elements adjacent
 * to the regions, computes the curl-curl of the velocity there, combines it
 * with the flux, extracts the result onto every region with
 * PhysTraceExtractBndCond, projects it onto the normals and takes its inner
 * product with the boundary basis. Apply() scatters that into the pressure
 * forcing. Every block of the work runs on its own stream.
 *
 * UpdateWithAcceleration() subtracts the normal component of the Dirichlet
 * velocity as the velocity solve imposes it: the session expression projected
 * onto the boundary expansion of the velocity, at the points of the region.
 * The predictor carries that velocity on the boundary, so the acceleration
 * vanishes at a steady state; the expression itself would leave the
 * projection error, divided by the implicit step size, in the flux. Its
 * coefficients on that expansion are kept and computed once, unless a
 * Dirichlet velocity condition of the region is time-dependent; it is
 * evaluated from them at every update into the work storage of the flux,
 * which no longer holds the flux by then.
 *
 * On a region with a velocity component that is not Dirichlet,
 * UpdateWithAcceleration() also extracts predictor - rhs and subtracts its
 * projection onto the normal components of those velocity components, whose
 * flux is thus formed from rhs, without acceleration.
 *
 * UpdateWeak() does the same with the predictor left out of the combined
 * flux, which then holds the viscous term alone before the acceleration and
 * predictor - rhs are subtracted.
 */
template <typename ExecSpace, typename TData>
class HOPressureBndCondOpImpl : public HOPressureBndCondOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    HOPressureBndCondOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                            const std::vector<std::string> &components)
        : HOPressureBndCondOp<TData>(expansionList, components)
    {
        m_nDim = expansionList->GetCoordim(0);
        BuildBoundaryData();
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<MultiRegions::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<HOPressureBndCondOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    // The flux an update computes: that of Update(), from its rhs; that of
    // UpdateWithAcceleration(), from the predictor less the acceleration to
    // the boundary velocity; or that of UpdateWeak(), which leaves the
    // predictor out as well.
    enum class FluxForm
    {
        Plain,
        Acceleration,
        Weak
    };

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
    // Stream m_compactBndCoeff is taken on, the first block's, through which
    // the block streams writing and reading it wait on one another.
    static constexpr unsigned int m_opStreamID = 1;

    bool m_isUpdated          = false;
    bool m_velocityContextSet = false;
    // Whether m_bndVelocityCoeffs has been computed, by the first update
    // with the acceleration.
    bool m_isBndVelocityEvaluated = false;
    unsigned int m_nDim           = 0;
    // On the boundary-adjacent elements, the velocity, from which m_curlCurl
    // is computed, and then rhs, or predictor when the acceleration is added,
    // combined in place with kinvis * m_curlCurl; in the weak form, rhs for
    // predictor - rhs, and then -kinvis * m_curlCurl. Both are laid out at
    // the interleave width of the velocity (SetRestrictedInterleaveWidth()).
    LibUtilities::Field<TData, FieldState::Phys> m_bndElmtWork;
    LibUtilities::Field<TData, FieldState::Phys> m_curlCurl;
    // On the boundary-adjacent elements, predictor - rhs, at the interleave
    // width of the velocity. Instantiated only with
    // m_hasNonDirichletVelocity.
    LibUtilities::Field<TData, FieldState::Phys> m_bndElmtHistory;
    std::vector<LibUtilities::Field<TData, FieldState::Phys>> m_bndPhys;
    std::vector<LibUtilities::Field<TData, FieldState::Coeff>> m_bndCoeffs;
    // Per region, the flux extracted onto the points of the region, then
    // predictor - rhs where the region has a velocity component that is not
    // Dirichlet, and then the Dirichlet velocity as the velocity solve
    // imposes it, zero in the velocity components that are not Dirichlet;
    // see EvaluateBoundaryVelocity().
    std::vector<LibUtilities::Field<TData, FieldState::Phys>> m_bndWorkPhys;
    // Per region, the coefficients of that Dirichlet velocity on the
    // boundary expansion of the velocity (uninstantiated for a point region).
    std::vector<LibUtilities::Field<TData, FieldState::Coeff>>
        m_bndVelocityCoeffs;
    // Per region, whether a Dirichlet velocity condition of the region is
    // time-dependent, so that m_bndVelocityCoeffs is computed at every
    // update.
    std::vector<bool> m_bndVelocityIsTimeDependent;
    std::vector<LibUtilities::Field<TData, FieldState::Phys>> m_bndNormals;
    // Per region, the normals with the components of the Dirichlet velocity
    // components zeroed, instantiated only for a region with a velocity
    // component that is not Dirichlet.
    std::vector<LibUtilities::Field<TData, FieldState::Phys>>
        m_bndNonDirichletNormals;
    std::vector<MultiRegions::ExpListSharedPtr> m_bndExpLists;
    std::vector<unsigned int> m_bndRegionIds;
    std::vector<std::shared_ptr<MultiRegions::IProductWRTBaseOp<TData>>>
        m_bndIProductOps;
    // Per region, the operators evaluating the Dirichlet velocity on the
    // boundary expansion of the velocity: the expression of the Dirichlet
    // velocity, its projection onto the expansion and the evaluation of that
    // at its points (nullptr for a point region).
    std::vector<std::shared_ptr<MultiRegions::ExpressionOp<TData>>>
        m_bndVelocityExpressionOps;
    std::vector<std::shared_ptr<MultiRegions::FwdTransBCOp<TData>>>
        m_bndVelocityFwdTransOps;
    std::vector<std::shared_ptr<MultiRegions::BwdTransOp<TData>>>
        m_bndVelocityBwdTransOps;
    std::vector<bool> m_bndIsPoint;
    MultiRegions::ExpListSharedPtr m_bndElmtExpansionList;
    std::shared_ptr<MultiRegions::CurlCurlOp<TData>> m_curlCurlOp;
    std::vector<VolumeGather> m_volumeGather;
    std::vector<unsigned int> m_bndConditionIndices;

    // Per region, extracts the flux in m_bndElmtWork onto the points of the
    // region.
    std::vector<std::shared_ptr<
        MultiRegions::detail::PhysTraceExtractBndCond<ExecSpace, TData>>>
        m_bndFluxExtractOps;

    // Number of boundary coefficients of all the regions on this rank, the
    // size of m_compactBndCoeff.
    size_t m_numBndCoeffCompSize = 0;
    // Offset of every region in m_compactBndCoeff.
    std::vector<size_t> m_bndCoeffOffsets;
    // Boundary coefficients of all the regions, one after the other, which
    // BuildBoundaryCoeffs() gathers into m_bndCoeff and m_groupValue.
    LibUtilities::MemoryRegion<TData> m_compactBndCoeff;
    // Streams of the region and domain blocks, which write
    // m_compactBndCoeff and read the values gathered from it.
    std::vector<unsigned int> m_updateStreams;

    // Maps from the boundary coefficients to the domain coefficients, built
    // by BuildCGBndCondCoeffMaps(): the targets written once (m_map, m_sign,
    // m_bndCoeff, m_bndCoeffSrc), and the targets written by more than one
    // boundary coefficient (m_group*), such as a vertex shared by two faces
    // of one element. The targets are indexed at the interleave width of the
    // block v_Apply() receives.
    bool m_anySignChange = false;

    std::vector<CoeffTargets> m_map;
    std::vector<LibUtilities::MemoryRegion<TData>> m_sign;
    std::vector<LibUtilities::MemoryRegion<TData>> m_bndCoeff;
    std::vector<LibUtilities::MemoryRegion<size_t>> m_bndCoeffSrc;
    // Number of targets written once, per block (the pressure has one
    // component).
    std::vector<size_t> m_uniqueCounts;

    std::vector<CoeffTargets> m_groupTarget;
    std::vector<LibUtilities::MemoryRegion<size_t>> m_groupOffset;
    std::vector<LibUtilities::MemoryRegion<size_t>> m_groupSrc;
    std::vector<LibUtilities::MemoryRegion<TData>> m_groupSign;
    std::vector<LibUtilities::MemoryRegion<TData>> m_groupValue;
    // Number of targets written more than once, per block.
    std::vector<size_t> m_groupCounts;

    void v_SetVelocityContext(
        const MultiRegions::ExpListSharedPtr &velocityExpansionList,
        const std::vector<std::string> &velocityComponents) override
    {
        m_nDim = velocityExpansionList->GetCoordim(0);
        ASSERTL1(velocityComponents.size() >= m_nDim,
                 "Insufficient velocity components for HOPressureBndCond.");
        const std::vector<std::string> velocityComps(
            velocityComponents.begin(), velocityComponents.begin() + m_nDim);

        // The velocity conditions decide whether the boundary-adjacent
        // elements hold predictor - rhs.
        BuildBoundaryVelocityData(velocityExpansionList, velocityComps);
        BuildBoundaryElementData(velocityExpansionList, velocityComps);
        m_velocityContextSet = true;
    }

    void v_Update(
        LibUtilities::Field<TData, FieldState::Phys> &rhs,
        LibUtilities::Field<TData, FieldState::Phys> &velocity) override
    {
        UpdateFlux(FluxForm::Plain, rhs, velocity, nullptr);
        FinaliseUpdate(FluxForm::Plain);
    }

    void v_UpdateWithAcceleration(
        LibUtilities::Field<TData, FieldState::Phys> &rhs,
        LibUtilities::Field<TData, FieldState::Phys> &velocity,
        LibUtilities::Field<TData, FieldState::Phys> &predictor) override
    {
        UpdateFlux(FluxForm::Acceleration, predictor, velocity, &rhs);
        FinaliseUpdate(FluxForm::Acceleration);
    }

    void v_UpdateWeak(
        LibUtilities::Field<TData, FieldState::Phys> &rhs,
        LibUtilities::Field<TData, FieldState::Phys> &velocity,
        LibUtilities::Field<TData, FieldState::Phys> &predictor) override
    {
        UpdateFlux(FluxForm::Weak, predictor, velocity, &rhs);
        FinaliseUpdate(FluxForm::Weak);
    }

    void v_UpdateBndCoeffs([[maybe_unused]] const TData &time) override
    {
        // The flux is set by Update().
    }

    size_t v_GetNumBndDofs() override
    {
        return m_numBndCoeffCompSize;
    }

    void v_ExcludeUserDefinedType(
        [[maybe_unused]] const std::string &userDefinedType) override
    {
        NEKERROR(ErrorUtil::efatal, "HOPressureBndCondOp applies only its own "
                                    "regions; it has no default "
                                    "regions to exclude.");
    }

    // Build the expansion list of the volume elements adjacent to the
    // regions, the maps gathering them from the velocity field, and the
    // operators acting on them.
    void BuildBoundaryElementData(
        const MultiRegions::ExpListSharedPtr &velocityExpansionList,
        const std::vector<std::string> &velocityComponents)
    {
        struct VolumeLocation
        {
            unsigned int block = 0;
            size_t element     = 0;
            LocalRegions::ExpansionSharedPtr expansion;
        };

        const auto velocityBlocks =
            MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                velocityExpansionList);
        std::map<int, VolumeLocation> fullVolumeLocation;
        std::map<int, int> faceOwner;
        for (unsigned int blk = 0; blk < velocityBlocks.size(); ++blk)
        {
            auto collection =
                MultiRegions::GetCollection(velocityExpansionList, blk);
            const auto &expansions = collection.GetExpVector();
            for (unsigned int e = 0; e < expansions.size(); ++e)
            {
                const auto volumeGeometry =
                    expansions[e]->GetGeom()->GetGlobalID();
                fullVolumeLocation[volumeGeometry] = {blk, e, expansions[e]};
                for (unsigned int face = 0; face < expansions[e]->GetNtraces();
                     ++face)
                {
                    faceOwner[expansions[e]->GetGeom()->GetTid(face)] =
                        volumeGeometry;
                }
            }
        }

        // The volume elements adjacent to a region, matched to its
        // elements by geometry id.
        std::set<int> selectedVolumeGeometry;
        for (unsigned int n = 0; n < m_bndExpLists.size(); ++n)
        {
            const auto bndBlocks =
                MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                    m_bndExpLists[n]);
            for (unsigned int blk = 0; blk < bndBlocks.size(); ++blk)
            {
                auto collection =
                    MultiRegions::GetCollection(m_bndExpLists[n], blk);
                const auto &expansions = collection.GetExpVector();
                for (unsigned int e = 0; e < expansions.size(); ++e)
                {
                    const auto owner =
                        faceOwner.find(expansions[e]->GetGeom()->GetGlobalID());
                    if (owner != faceOwner.end())
                    {
                        selectedVolumeGeometry.insert(owner->second);
                    }
                }
            }
        }

        if (selectedVolumeGeometry.empty())
        {
            return;
        }
        // The kernels are instantiated for two and three dimensions.
        ASSERTL0(m_nDim == 2 || m_nDim == 3,
                 "HOPressureBndCond is implemented for two and three "
                 "dimensions.");

        SpatialDomains::ExpansionInfoMap restrictedExpansionInfo;
        const auto &velocityExpansionInfo =
            velocityExpansionList->GetGraph()->GetExpansionInfo(
                velocityComponents[0]);
        for (const auto geometry : selectedVolumeGeometry)
        {
            const auto info = velocityExpansionInfo.find(geometry);
            ASSERTL1(info != velocityExpansionInfo.end(),
                     "Unable to locate boundary-adjacent velocity expansion "
                     "metadata.");
            restrictedExpansionInfo.emplace(*info);
        }

        m_bndElmtExpansionList =
            MemoryManager<MultiRegions::ExpList>::AllocateSharedPtr(
                velocityExpansionList->GetSession(), restrictedExpansionInfo,
                false);
        m_bndElmtExpansionList->SetDataWarehouse();
        const auto restrictedBlocks =
            MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                m_bndElmtExpansionList);

        m_bndElmtWork = LibUtilities::Field<TData, FieldState::Phys>(
            "HOPressureBndCond boundary element velocity and flux",
            restrictedBlocks, velocityComponents, 1);
        m_curlCurl = LibUtilities::Field<TData, FieldState::Phys>(
            "HOPressureBndCond CurlCurl", restrictedBlocks, velocityComponents,
            1);
        m_bndElmtWork.template Initialize<MemSpace>(0.0);
        m_curlCurl.template Initialize<MemSpace>(0.0);
        if (this->m_hasNonDirichletVelocity)
        {
            m_bndElmtHistory = LibUtilities::Field<TData, FieldState::Phys>(
                "HOPressureBndCond boundary element history", restrictedBlocks,
                velocityComponents, 1);
            m_bndElmtHistory.template Initialize<MemSpace>(0.0);
        }

        std::map<int, VolumeLocation> restrictedVolumeLocation;
        for (unsigned int blk = 0; blk < restrictedBlocks.size(); ++blk)
        {
            auto collection =
                MultiRegions::GetCollection(m_bndElmtExpansionList, blk);
            const auto &expansions = collection.GetExpVector();
            for (unsigned int e = 0; e < expansions.size(); ++e)
            {
                restrictedVolumeLocation
                    [expansions[e]->GetGeom()->GetGlobalID()] = {blk, e,
                                                                 expansions[e]};
            }
        }

        // Per pair of source and destination block, the destination element
        // of every element gathered, mapped to its source element, in
        // increasing destination element.
        std::map<std::pair<unsigned int, unsigned int>,
                 std::map<size_t, size_t>>
            volumeMaps;
        for (const auto geometry : selectedVolumeGeometry)
        {
            const auto &src = fullVolumeLocation.at(geometry);
            const auto &dst = restrictedVolumeLocation.at(geometry);
            ASSERTL1(src.expansion->GetTotPoints() ==
                         dst.expansion->GetTotPoints(),
                     "Boundary element quadrature differs after restriction.");
            volumeMaps[{src.block, dst.block}][dst.element] = src.element;
        }
        for (auto &[blocks, maps] : volumeMaps)
        {
            VolumeGather gather;
            gather.srcBlock = blocks.first;
            gather.dstBlock = blocks.second;
            gather.nData    = restrictedBlocks[blocks.second].GetNumData();
            for (const auto &[dstElmt, srcElmt] : maps)
            {
                gather.dstElmt.push_back(dstElmt);
                gather.srcElmt.push_back(srcElmt);
            }
            m_volumeGather.push_back(std::move(gather));
        }

        m_curlCurlOp = MultiRegions::CurlCurlOp<TData>::Create(
            m_bndElmtExpansionList, velocityComponents, ExecSpace::name);

        // An extraction for every region, empty where no element of the
        // region is on this rank. m_bndElmtExpansionList, built from an
        // ExpansionInfoMap, has no mesh graph, so the extraction takes that of
        // the velocity field.
        for (unsigned int n = 0; n < m_bndExpLists.size(); ++n)
        {
            m_bndFluxExtractOps.push_back(
                std::make_shared<MultiRegions::detail::PhysTraceExtractBndCond<
                    ExecSpace, TData>>(m_bndElmtExpansionList,
                                       velocityComponents, m_bndExpLists[n],
                                       ExecSpace::name,
                                       velocityExpansionList->GetGraph()));
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
                 "HOPressureBndCond takes fields at interleave width 1 or at "
                 "the interleave width of its execution space.");
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
                 "HOPressureBndCond takes fields at interleave width 1 or at "
                 "the interleave width of its execution space.");
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
                 "HOPressureBndCond takes fields at interleave width 1 or at "
                 "the interleave width of its execution space.");
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

    // Copy the boundary-adjacent elements of the first m_nDim components of
    // src, a field on the velocity expansion list, into dst, each block at
    // its own interleave width. Each gather runs on the stream of its
    // destination block, after the work queued on the stream of its source
    // block, and the source streams wait on the gathers in turn, without the
    // host waiting on either.
    void GatherBoundaryElements(
        LibUtilities::Field<TData, FieldState::Phys> &src,
        LibUtilities::Field<TData, FieldState::Phys> &dst)
    {
        for (auto &gather : m_volumeGather)
        {
            const unsigned int srcStreamID = gather.srcBlock + 1;
            const unsigned int streamID    = gather.dstBlock + 1;
            auto &srcBlock                 = src.GetBlocks()[gather.srcBlock];
            auto &dstBlock                 = dst.GetBlocks()[gather.dstBlock];

            RecordStreamEvent<ExecSpace>(srcStreamID);
            SetStreamDependencies<ExecSpace>(streamID, srcStreamID);

            auto &indices = GatherIndicesAtInterleaveWidths(
                gather, srcBlock.GetInterleaveWidth(),
                dstBlock.GetInterleaveWidth(), streamID);
            auto srcPtr =
                srcBlock.template GetPtr<MemSpace, ReadOnly>(streamID);
            // The points of the block this gather does not write are
            // padding, which stays zero, or written by another gather.
            auto dstPtr =
                dstBlock.template GetPtr<MemSpace, ReadWrite>(streamID);
            auto srcIdx =
                indices.srcIndex.template GetPtr<MemSpace, ReadOnly>(streamID);
            auto dstIdx =
                indices.dstIndex.template GetPtr<MemSpace, ReadOnly>(streamID);
            HOPressureVolumeGatherKernel<ExecSpace>(
                gather.srcElmt.size() * gather.nData, m_nDim,
                srcBlock.CompSize(), dstBlock.CompSize(), srcIdx, dstIdx,
                srcPtr, dstPtr, streamID);
            RecordStreamEvent<ExecSpace>(streamID);
        }

        for (const auto &gather : m_volumeGather)
        {
            const unsigned int srcStreamID = gather.srcBlock + 1;
            const unsigned int streamID    = gather.dstBlock + 1;
            SetStreamDependencies<ExecSpace>(srcStreamID, streamID);
        }
    }

    // Compute scale * (forcing - kinvis * m_curlCurl) . n on every region,
    // forcing being the rhs of Update() or the predictor of
    // UpdateWithAcceleration(). The weak form of UpdateWeak() leaves forcing
    // out: its flux is -scale * kinvis * m_curlCurl . n. With @p rhs, that
    // of UpdateWithAcceleration() and UpdateWeak(), subtract
    // scale * (predictor - rhs) . n over the velocity components that are
    // not Dirichlet on a region.
    void BuildBoundaryPhys(
        const FluxForm form,
        LibUtilities::Field<TData, FieldState::Phys> &forcing,
        LibUtilities::Field<TData, FieldState::Phys> *rhs, const TData kinvis,
        const TData scale)
    {
        // Combine the flux on the boundary-adjacent elements, before
        // extraction, after forming predictor - rhs from the predictor. The
        // weak form needs the predictor only for predictor - rhs, which it
        // forms in m_bndElmtHistory, using m_bndElmtWork, which the weak
        // combination overwrites, for rhs.
        const bool isWeak            = form == FluxForm::Weak;
        const bool isSubtractHistory = rhs && m_bndElmtHistory;
        if (isWeak)
        {
            if (isSubtractHistory)
            {
                GatherBoundaryElements(forcing, m_bndElmtHistory);
                GatherBoundaryElements(*rhs, m_bndElmtWork);
                Math::sub<ExecSpace>(m_bndElmtHistory, m_bndElmtWork,
                                     m_bndElmtHistory);
            }
        }
        else
        {
            GatherBoundaryElements(forcing, m_bndElmtWork);
            if (isSubtractHistory)
            {
                GatherBoundaryElements(*rhs, m_bndElmtHistory);
                Math::sub<ExecSpace>(m_bndElmtWork, m_bndElmtHistory,
                                     m_bndElmtHistory);
            }
        }
        for (unsigned int blk = 0; blk < m_bndElmtWork.GetBlocks().size();
             ++blk)
        {
            const unsigned int streamID = blk + 1;
            auto &fluxBlk               = m_bndElmtWork.GetBlocks()[blk];
            auto &curlBlk               = m_curlCurl.GetBlocks()[blk];
            auto curlPtr =
                curlBlk.template GetPtr<MemSpace, ReadOnly>(streamID);
            auto fluxPtr =
                fluxBlk.template GetPtr<MemSpace, ReadWrite>(streamID);
            // Every point of the block at its interleave width, padding
            // included, which the combination keeps zero.
            const size_t nPoints =
                fluxBlk.GetNumElementsWithPadding() * fluxBlk.GetNumData();
            if (isWeak)
            {
                if (m_nDim == 2)
                {
                    HOPressureVolumeCombineKernel<ExecSpace, 2, false>(
                        nPoints, fluxBlk.CompSize(), kinvis, scale, curlPtr,
                        fluxPtr, streamID);
                }
                else
                {
                    HOPressureVolumeCombineKernel<ExecSpace, 3, false>(
                        nPoints, fluxBlk.CompSize(), kinvis, scale, curlPtr,
                        fluxPtr, streamID);
                }
            }
            else
            {
                if (m_nDim == 2)
                {
                    HOPressureVolumeCombineKernel<ExecSpace, 2, true>(
                        nPoints, fluxBlk.CompSize(), kinvis, scale, curlPtr,
                        fluxPtr, streamID);
                }
                else
                {
                    HOPressureVolumeCombineKernel<ExecSpace, 3, true>(
                        nPoints, fluxBlk.CompSize(), kinvis, scale, curlPtr,
                        fluxPtr, streamID);
                }
            }
        }

        // Each extraction orders itself after the combine above and the
        // streams of the boundary blocks after itself.
        for (unsigned int n = 0; n < m_bndPhys.size(); ++n)
        {
            m_bndFluxExtractOps[n]->Apply(m_bndElmtWork, m_bndWorkPhys[n]);

            for (unsigned int blk = 0; blk < m_bndPhys[n].GetBlocks().size();
                 ++blk)
            {
                const unsigned int streamID = blk + 1;
                auto &fluxBlk               = m_bndWorkPhys[n].GetBlocks()[blk];
                auto &normalBlk             = m_bndNormals[n].GetBlocks()[blk];
                auto &outBlk                = m_bndPhys[n].GetBlocks()[blk];
                auto fluxPtr =
                    fluxBlk.template GetPtr<MemSpace, ReadOnly>(streamID);
                auto normalPtr =
                    normalBlk.template GetPtr<MemSpace, ReadOnly>(streamID);
                auto outPtr =
                    outBlk.template GetPtr<MemSpace, WriteOnly>(streamID);

                const size_t nPoints =
                    outBlk.GetNumElements() * outBlk.GetNumData();
                if (m_nDim == 2)
                {
                    HOPressureBoundaryCombineKernel<ExecSpace, 2>(
                        nPoints, fluxBlk.CompSize(), fluxPtr, normalPtr, outPtr,
                        streamID);
                }
                else
                {
                    HOPressureBoundaryCombineKernel<ExecSpace, 3>(
                        nPoints, fluxBlk.CompSize(), fluxPtr, normalPtr, outPtr,
                        streamID);
                }
            }

            if (!isSubtractHistory || !m_bndNonDirichletNormals[n])
            {
                continue;
            }

            // The extraction waits on the combine above, which read
            // m_bndWorkPhys[n].
            m_bndFluxExtractOps[n]->Apply(m_bndElmtHistory, m_bndWorkPhys[n]);
            for (unsigned int blk = 0; blk < m_bndPhys[n].GetBlocks().size();
                 ++blk)
            {
                const unsigned int streamID = blk + 1;
                auto &historyBlk            = m_bndWorkPhys[n].GetBlocks()[blk];
                auto &normalBlk = m_bndNonDirichletNormals[n].GetBlocks()[blk];
                auto &outBlk    = m_bndPhys[n].GetBlocks()[blk];
                auto historyPtr =
                    historyBlk.template GetPtr<MemSpace, ReadOnly>(streamID);
                auto normalPtr =
                    normalBlk.template GetPtr<MemSpace, ReadOnly>(streamID);
                auto outPtr =
                    outBlk.template GetPtr<MemSpace, ReadWrite>(streamID);

                const size_t nPoints =
                    outBlk.GetNumElements() * outBlk.GetNumData();
                if (m_nDim == 2)
                {
                    HOPressureSubtractNormalVelocityKernel<ExecSpace, 2>(
                        nPoints, historyBlk.CompSize(), scale, historyPtr,
                        normalPtr, outPtr, streamID);
                }
                else
                {
                    HOPressureSubtractNormalVelocityKernel<ExecSpace, 3>(
                        nPoints, historyBlk.CompSize(), scale, historyPtr,
                        normalPtr, outPtr, streamID);
                }
            }
        }
    }

    void BuildBoundaryVelocityData(
        const MultiRegions::ExpListSharedPtr &velocityExpansionList,
        const std::vector<std::string> &velocityComponents)
    {
        const SpatialDomains::BoundaryConditions bcs(
            this->m_expansionList->GetSession(),
            this->m_expansionList->GetGraph());
        const auto &bndConditions = bcs.GetBoundaryConditions();
        const auto &velocityBndExpansions =
            velocityExpansionList->GetBndCondExpansions();

        ASSERTL1(m_bndExpLists.size() == m_bndRegionIds.size(),
                 "HOPressureBndCond boundary metadata differs.");
        for (unsigned int n = 0; n < m_bndExpLists.size(); ++n)
        {
            auto bndExp            = m_bndExpLists[n];
            const auto conditionIt = bndConditions.find(m_bndRegionIds[n]);
            ASSERTL0(conditionIt != bndConditions.end(),
                     "Unable to locate velocity boundary conditions for "
                     "HOPressureBndCond.");

            // A velocity component that is not Dirichlet has no prescribed
            // boundary velocity and takes no acceleration: its boundary
            // velocity is zero, and BuildBoundaryPhys() forms its flux from
            // rhs through m_bndNonDirichletNormals.
            std::vector<LibUtilities::EquationSharedPtr> equations;
            std::vector<bool> isDirichlet(m_nDim);
            bool isTimeDependent = false;
            for (unsigned int d = 0; d < m_nDim; ++d)
            {
                const auto componentIt =
                    conditionIt->second->find(velocityComponents[d]);
                ASSERTL0(componentIt != conditionIt->second->end(),
                         "Unable to locate a velocity component boundary "
                         "condition for HOPressureBndCond.");
                isDirichlet[d] =
                    componentIt->second->GetBoundaryConditionType() ==
                    SpatialDomains::eDirichlet;
                isTimeDependent =
                    isTimeDependent ||
                    (isDirichlet[d] && componentIt->second->IsTimeDependent());
                equations.push_back(
                    isDirichlet[d] ? componentIt->second->GetEquation()
                                   : MemoryManager<LibUtilities::Equation>::
                                         AllocateSharedPtr(
                                             this->m_expansionList->GetSession()
                                                 ->GetInterpreter(),
                                             "0"));
            }
            const bool hasNonDirichletVelocity =
                std::find(isDirichlet.begin(), isDirichlet.end(), false) !=
                isDirichlet.end();
            this->m_hasNonDirichletVelocity =
                this->m_hasNonDirichletVelocity ||
                (hasNonDirichletVelocity && bndExp->GetExpSize() > 0);

            const auto bndPhysBlocks =
                MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                    bndExp);
            m_bndWorkPhys.emplace_back(
                "HOPressureBndCond boundary flux and velocity", bndPhysBlocks,
                velocityComponents, 1);
            m_bndNormals.emplace_back("HOPressureBndCond boundary normals",
                                      bndPhysBlocks, velocityComponents, 1);
            m_bndWorkPhys.back().template Initialize<MemSpace>(0.0);
            m_bndNormals.back().template Initialize<MemSpace>(0.0);
            m_bndNonDirichletNormals.emplace_back();
            if (hasNonDirichletVelocity)
            {
                m_bndNonDirichletNormals.back() =
                    LibUtilities::Field<TData, FieldState::Phys>(
                        "HOPressureBndCond boundary non-Dirichlet normals",
                        bndPhysBlocks, velocityComponents, 1);
                m_bndNonDirichletNormals.back().template Initialize<MemSpace>(
                    0.0);
            }

            Array<OneD, Array<OneD, double>> normals;
            this->m_expansionList->GetBoundaryNormals(m_bndConditionIndices[n],
                                                      normals);
            for (unsigned int blk = 0;
                 blk < m_bndNormals.back().GetBlocks().size(); ++blk)
            {
                auto &normalBlock = m_bndNormals.back().GetBlocks()[blk];
                auto normalPtr =
                    normalBlock
                        .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
                TData *nonDirichletNormalPtr = nullptr;
                if (hasNonDirichletVelocity)
                {
                    nonDirichletNormalPtr =
                        m_bndNonDirichletNormals.back()
                            .GetBlocks()[blk]
                            .template GetPtr<NektarSpaces::HostSpace,
                                             WriteOnly>();
                }
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
                            const size_t idx = d * normalBlock.CompSize() +
                                               e * normalBlock.GetNumData() + p;
                            normalPtr[idx] =
                                static_cast<TData>(normals[d][offset + p]);
                            if (nonDirichletNormalPtr)
                            {
                                nonDirichletNormalPtr[idx] =
                                    isDirichlet[d] ? TData(0.0)
                                                   : normalPtr[idx];
                            }
                        }
                    }
                }
            }

            // The velocity solve imposes the Dirichlet velocity as its
            // projection onto the boundary expansion of the velocity, on
            // which it is therefore evaluated. The flux and the normals are
            // on the boundary expansion of the pressure, whose blocks must
            // hold the same elements at the same points.
            auto velocityBndExp =
                velocityBndExpansions[m_bndConditionIndices[n]];
            // The operators built on this list below need a data warehouse,
            // which it has not got: the equation system sets the warehouse
            // of a field and of its trace, not of its boundary expansion
            // lists, and the boundary condition operators of the velocity
            // solve build lists of their own. Setting it here cannot affect
            // them, since an operator holds the warehouse it was built with.
            velocityBndExp->SetDataWarehouse();
            ASSERTL0(velocityBndExp->SameQuadratureLayout(bndExp),
                     "HOPressureBndCond requires the velocity and pressure "
                     "boundary expansions to share their quadrature points.");
            m_bndVelocityIsTimeDependent.push_back(isTimeDependent);

            auto expressionOp = MultiRegions::ExpressionOp<TData>::Create(
                velocityBndExp, velocityComponents, ExecSpace::name);
            expressionOp->SetExpressions(equations);
            m_bndVelocityExpressionOps.push_back(expressionOp);

            // A point region takes the prescribed velocity itself.
            m_bndVelocityCoeffs.emplace_back();
            m_bndVelocityFwdTransOps.push_back(nullptr);
            m_bndVelocityBwdTransOps.push_back(nullptr);
            if (m_bndIsPoint[n])
            {
                continue;
            }
            m_bndVelocityCoeffs.back() =
                LibUtilities::Field<TData, FieldState::Coeff>(
                    "HOPressureBndCond boundary velocity coeffs",
                    MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                        velocityBndExp),
                    velocityComponents, 1);
            m_bndVelocityCoeffs.back().template Initialize<MemSpace>(0.0);
            m_bndVelocityFwdTransOps.back() =
                MultiRegions::FwdTransBCOp<TData>::Create(
                    velocityBndExp, velocityComponents, ExecSpace::name);
            m_bndVelocityBwdTransOps.back() =
                MultiRegions::BwdTransOp<TData>::Create(
                    velocityBndExp, velocityComponents, ExecSpace::name);
        }
    }

    // Whether a pressure boundary condition takes the higher-order pressure
    // flux: a Neumann condition with USERDEFINEDTYPE="H". Any other Neumann
    // region keeps the value prescribed in the session.
    static bool IsHOPressure(
        const SpatialDomains::BoundaryConditionShPtr &condition)
    {
        return condition->GetBoundaryConditionType() ==
                   SpatialDomains::eNeumann &&
               StringUtils::IEquals(
                   condition->GetUserDefined(),
                   HOPressureBndCondOp<TData>::userDefinedType);
    }

    void BuildBoundaryData()
    {
        ASSERTL1(this->m_components.size() == 1,
                 "HOPressureBndCond expects one pressure component.");

        auto contField = std::dynamic_pointer_cast<MultiRegions::ContField>(
            this->m_expansionList);
        ASSERTL1(contField,
                 "HOPressureBndCond requires a continuous pressure field.");

        const auto &bndExpansions =
            this->m_expansionList->GetBndCondExpansions();
        const auto &bndConditions = this->m_expansionList->GetBndConditions();
        const SpatialDomains::BoundaryConditions bcs(
            this->m_expansionList->GetSession(),
            this->m_expansionList->GetGraph());
        const auto &bregions = bcs.GetBoundaryRegions();
        std::vector<unsigned int> bndRegionIds;
        for (const auto &[regionId, region] : bregions)
        {
            bndRegionIds.push_back(regionId);
        }
        ASSERTL1(bndRegionIds.size() == bndExpansions.size(),
                 "Boundary region and expansion counts differ.");

        // isTypeByRegion and numBcExpCoeffs have an entry for every region,
        // including those without the higher-order condition, for the
        // offsets of BuildCGBndCondCoeffMaps().
        std::vector<std::vector<bool>> isTypeByRegion(bndExpansions.size());
        std::vector<size_t> numBcExpCoeffs(bndExpansions.size());
        for (unsigned int n = 0; n < bndExpansions.size(); ++n)
        {
            const bool isHOP  = IsHOPressure(bndConditions[n]);
            isTypeByRegion[n] = {isHOP};
            numBcExpCoeffs[n] = bndExpansions[n]->GetNcoeffs();

            if (!isHOP)
            {
                continue;
            }

            auto bndExp = bndExpansions[n];
            bndExp->SetDataWarehouse();

            m_bndCoeffOffsets.push_back(m_numBndCoeffCompSize);
            m_numBndCoeffCompSize += numBcExpCoeffs[n];

            m_bndExpLists.push_back(bndExp);
            m_bndRegionIds.push_back(bndRegionIds[n]);
            m_bndConditionIndices.push_back(n);
            m_bndPhys.emplace_back(
                "HOPressureBndCond boundary phys",
                MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                    bndExp),
                this->m_components, 1);
            m_bndCoeffs.emplace_back(
                "HOPressureBndCond boundary coeffs",
                MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                    bndExp),
                this->m_components, 1);
            m_bndPhys.back().template Initialize<MemSpace>(0.0);
            m_bndCoeffs.back().template Initialize<MemSpace>(0.0);

            const bool isPoint = bndExp->GetShapeDimension() == 0;
            m_bndIsPoint.push_back(isPoint);
            m_bndIProductOps.push_back(
                isPoint ? nullptr
                        : MultiRegions::IProductWRTBaseOp<TData>::Create(
                              bndExp, this->m_components, ExecSpace::name));
        }

        // No higher-order pressure region on this rank. The check is local:
        // BuildCGBndCondCoeffMaps() below uses the assembly map of the
        // pressure field and builds no field collectively.
        if (m_numBndCoeffCompSize == 0)
        {
            return;
        }

        auto maps =
            MultiRegions::detail::BuildCGBndCondCoeffMaps<ExecSpace, TData>(
                this->m_expansionList, this->m_components, bregions,
                isTypeByRegion, numBcExpCoeffs, m_numBndCoeffCompSize,
                MultiRegions::detail::CGBndCondCoeffMapSet::UniqueAndGroups,
                [](unsigned int, const auto &, const auto &, const auto &) {});

        m_anySignChange = maps.anySignChange;
        m_sign          = std::move(maps.uniqueSign);
        m_bndCoeff      = std::move(maps.uniqueBndCoeff);
        m_bndCoeffSrc   = std::move(maps.uniqueBndCoeffSrc);
        m_groupOffset   = std::move(maps.groupOffset);
        m_groupSrc      = std::move(maps.groupSrc);
        m_groupSign     = std::move(maps.groupSign);
        m_groupValue    = std::move(maps.groupValue);
        m_uniqueCounts.resize(maps.compUniqueCounts.size());
        m_groupCounts.resize(maps.compGroupCounts.size());
        for (size_t blk = 0; blk < m_uniqueCounts.size(); ++blk)
        {
            m_uniqueCounts[blk] = maps.compUniqueCounts[blk][0];
            m_groupCounts[blk]  = maps.compGroupCounts[blk][0];
        }
        m_compactBndCoeff =
            LibUtilities::MemoryRegion<TData>(m_numBndCoeffCompSize);
        // Allocated here, on the null stream, so the region block streams
        // that write it never race the allocation.
        m_compactBndCoeff.template GetPtr<MemSpace, WriteOnly>();

        // The region blocks write the compact array and the domain blocks
        // read the values gathered from it.
        size_t nUpdateStreams = m_bndCoeff.size();
        for (auto &field : m_bndCoeffs)
        {
            nUpdateStreams = std::max(nUpdateStreams, field.GetBlocks().size());
        }
        for (unsigned int id = 1; id <= nUpdateStreams; ++id)
        {
            m_updateStreams.push_back(id);
        }

        m_map = BuildCoeffTargets(maps.uniqueMap, this->m_expansionList);
        m_groupTarget =
            BuildCoeffTargets(maps.groupTarget, this->m_expansionList);
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

    void BuildBoundaryCoeffs(const FluxForm form)
    {
        if (m_numBndCoeffCompSize == 0)
        {
            return;
        }

        if (form != FluxForm::Plain)
        {
            SubtractBoundaryVelocity();
        }

        // Copy the boundary coefficients of every region into
        // m_compactBndCoeff, in the layout BuildCGBndCondCoeffMaps() assumed,
        // each block on the stream that computed it.
        auto compactPtr =
            m_compactBndCoeff.template GetPtr<MemSpace, WriteOnly>(
                m_opStreamID);

        for (unsigned int n = 0; n < m_bndPhys.size(); ++n)
        {
            if (!m_bndIsPoint[n])
            {
                m_bndIProductOps[n]->Apply(m_bndPhys[n], m_bndCoeffs[n]);
            }
            else
            {
                // A point region has a single block.
                const unsigned int blk      = 0;
                const unsigned int streamID = blk + 1;
                auto &physBlk               = m_bndPhys[n].GetBlocks()[blk];
                auto &coeffBlk              = m_bndCoeffs[n].GetBlocks()[blk];
                auto src =
                    physBlk.template GetPtr<MemSpace, ReadOnly>(streamID);
                auto dst =
                    coeffBlk.template GetPtr<MemSpace, WriteOnly>(streamID);
                const auto count =
                    coeffBlk.GetNumElements() * coeffBlk.GetNumData();
                Math::copyKernel<ExecSpace>(count, src, dst, streamID);
            }

            const size_t bndOffset = m_bndCoeffOffsets[n];
            size_t regionOffset    = 0;
            for (unsigned int blk = 0; blk < m_bndCoeffs[n].GetBlocks().size();
                 ++blk)
            {
                const unsigned int streamID = blk + 1;
                auto &wspBlk                = m_bndCoeffs[n].GetBlocks()[blk];
                auto srcPtr =
                    wspBlk.template GetPtr<MemSpace, ReadOnly>(streamID);
                const auto blockLen =
                    wspBlk.GetNumElements() * wspBlk.GetNumData();

                // On the stream that computed the coefficients of the block.
                Math::copyKernel<ExecSpace>(
                    blockLen, srcPtr, compactPtr + bndOffset + regionOffset,
                    streamID);
                regionOffset += blockLen;
            }
        }

        // Gather m_compactBndCoeff into the domain blocks v_Apply() reads,
        // each on the stream of its block. Every gather reads entries written
        // on any region block's stream, so the block streams first wait on
        // one another through m_opStreamID.
        JoinStreams<ExecSpace>(m_opStreamID, m_updateStreams);
        ReleaseStreams<ExecSpace>(m_opStreamID, m_updateStreams);

        for (unsigned int blk = 0; blk < m_bndCoeff.size(); ++blk)
        {
            const unsigned int streamID = blk + 1;
            auto idxPtr =
                m_bndCoeffSrc[blk].template GetPtr<MemSpace, ReadOnly>(
                    streamID);
            auto dstPtr =
                m_bndCoeff[blk].template GetPtr<MemSpace, WriteOnly>(streamID);
            MultiRegions::detail::CGBndCondGatherKernel<ExecSpace>(
                m_bndCoeffSrc[blk].size(), idxPtr, compactPtr, dstPtr,
                streamID);

            // Gathered even when empty, so that m_groupValue[blk] is valid
            // when v_Apply() reads it.
            auto groupIdxPtr =
                m_groupSrc[blk].template GetPtr<MemSpace, ReadOnly>(streamID);
            auto groupDstPtr =
                m_groupValue[blk].template GetPtr<MemSpace, WriteOnly>(
                    streamID);
            MultiRegions::detail::CGBndCondGatherKernel<ExecSpace>(
                m_groupSrc[blk].size(), groupIdxPtr, compactPtr, groupDstPtr,
                streamID);
        }

        // The next update overwrites m_compactBndCoeff on the region block
        // streams, which wait on the gathers in turn.
        JoinStreams<ExecSpace>(m_opStreamID, m_updateStreams);
        ReleaseStreams<ExecSpace>(m_opStreamID, m_updateStreams);
    }

    // Compute the flux of @p form from the current fields and its boundary
    // coefficients; see BuildBoundaryPhys() for @p forcing and @p rhs.
    void UpdateFlux(const FluxForm form,
                    LibUtilities::Field<TData, FieldState::Phys> &forcing,
                    LibUtilities::Field<TData, FieldState::Phys> &velocity,
                    LibUtilities::Field<TData, FieldState::Phys> *rhs)
    {
        ASSERTL0(form == FluxForm::Plain || !this->m_hasNonDirichletVelocity ||
                     (rhs && *rhs),
                 "HOPressureBndCond requires the explicit contribution rhs on "
                 "a region with a velocity component that is not Dirichlet.");
        ASSERTL1(m_velocityContextSet,
                 "SetVelocityContext must be called before Update.");
        ASSERTL1(this->m_isSetKinvis,
                 "SetKinvis must be called before Update.");
        if (!m_curlCurlOp)
        {
            return;
        }
        // The fields on the boundary-adjacent elements take the interleave
        // width of the velocity, and the forcing is gathered into them at
        // that width too.
        const unsigned int interleaveWidth =
            velocity.GetBlocks()[m_volumeGather[0].srcBlock]
                .GetInterleaveWidth();
        SetRestrictedInterleaveWidth(m_bndElmtWork, interleaveWidth);
        SetRestrictedInterleaveWidth(m_curlCurl, interleaveWidth);
        if (m_bndElmtHistory)
        {
            SetRestrictedInterleaveWidth(m_bndElmtHistory, interleaveWidth);
        }
        GatherBoundaryElements(velocity, m_bndElmtWork);
        m_curlCurlOp->Apply(m_bndElmtWork, m_curlCurl);
        BuildBoundaryPhys(form, forcing, rhs, this->m_kinvis, this->m_scale);
    }

    // Evaluate on every region, into m_bndWorkPhys, the Dirichlet velocity
    // as the velocity solve imposes it, at the time set by SetTime(): the
    // session expression, projected onto the boundary expansion of the
    // velocity with FwdTransBCOp, as DirBndCondOp sets the Dirichlet
    // coefficients of the velocity, and evaluated at its points with
    // BwdTransOp. The coefficients of a region without a time-dependent
    // Dirichlet velocity condition are computed only once. A point region
    // takes the expression itself. m_bndWorkPhys is free once
    // BuildBoundaryPhys() has combined the flux, and is written on the
    // streams of the boundary blocks, which read it there.
    void EvaluateBoundaryVelocity()
    {
        for (unsigned int n = 0; n < m_bndWorkPhys.size(); ++n)
        {
            const bool isCoeffsCurrent =
                m_isBndVelocityEvaluated && !m_bndVelocityIsTimeDependent[n];
            if (m_bndIsPoint[n] || !isCoeffsCurrent)
            {
                m_bndVelocityExpressionOps[n]->SetTime(this->m_time);
                m_bndVelocityExpressionOps[n]->Apply(m_bndWorkPhys[n],
                                                     m_bndWorkPhys[n]);
            }
            if (m_bndIsPoint[n])
            {
                continue;
            }
            if (!isCoeffsCurrent)
            {
                // The forward transform assumes a zero output, of which it
                // sets the vertex modes first.
                Math::zero<ExecSpace>(m_bndVelocityCoeffs[n]);
                m_bndVelocityFwdTransOps[n]->Apply(m_bndWorkPhys[n],
                                                   m_bndVelocityCoeffs[n]);
            }
            m_bndVelocityBwdTransOps[n]->Apply(m_bndVelocityCoeffs[n],
                                               m_bndWorkPhys[n]);
        }
        m_isBndVelocityEvaluated = true;
    }

    void SubtractBoundaryVelocity()
    {
        EvaluateBoundaryVelocity();
        for (unsigned int n = 0; n < m_bndPhys.size(); ++n)
        {
            for (unsigned int blk = 0;
                 blk < m_bndWorkPhys[n].GetBlocks().size(); ++blk)
            {
                const unsigned int streamID = blk + 1;
                auto &velocityBlk           = m_bndWorkPhys[n].GetBlocks()[blk];
                auto &normalBlk             = m_bndNormals[n].GetBlocks()[blk];
                auto &fluxBlk               = m_bndPhys[n].GetBlocks()[blk];
                auto velocityPtr =
                    velocityBlk.template GetPtr<MemSpace, ReadOnly>(streamID);
                auto normalPtr =
                    normalBlk.template GetPtr<MemSpace, ReadOnly>(streamID);
                auto fluxPtr =
                    fluxBlk.template GetPtr<MemSpace, ReadWrite>(streamID);
                const size_t nPoints =
                    fluxBlk.GetNumElements() * fluxBlk.GetNumData();
                if (m_nDim == 2)
                {
                    HOPressureSubtractNormalVelocityKernel<ExecSpace, 2>(
                        nPoints, velocityBlk.CompSize(), this->m_scale,
                        velocityPtr, normalPtr, fluxPtr, streamID);
                }
                else
                {
                    HOPressureSubtractNormalVelocityKernel<ExecSpace, 3>(
                        nPoints, velocityBlk.CompSize(), this->m_scale,
                        velocityPtr, normalPtr, fluxPtr, streamID);
                }
            }
        }
    }

    void FinaliseUpdate(const FluxForm form)
    {
        BuildBoundaryCoeffs(form);
        m_isUpdated = true;
    }

    void v_Apply(LibUtilities::Field<TData, FieldState::Coeff> &inout) override
    {
        if (!m_isUpdated || m_numBndCoeffCompSize == 0)
        {
            return;
        }

        for (unsigned int blk = 0; blk < inout.GetBlocks().size(); ++blk)
        {
            const unsigned int streamID = blk + 1;

            // The targets index the block at its own interleave width.
            auto &inoutBlk                     = inout.GetBlocks()[blk];
            const unsigned int interleaveWidth = inoutBlk.GetInterleaveWidth();

            auto inoutPtr =
                inoutBlk.template GetPtr<MemSpace, ReadWrite>(streamID);
            auto bndCoeffPtr =
                m_bndCoeff[blk].template GetPtr<MemSpace, ReadOnly>(streamID);

            // Targets written by one boundary coefficient.
            const size_t nUnique = m_uniqueCounts[blk];
            if (nUnique > 0)
            {
                auto mapPtr =
                    TargetsAtInterleaveWidth(m_map[blk], interleaveWidth,
                                             streamID)
                        .template GetPtr<MemSpace, ReadOnly>(streamID);
                if (m_anySignChange)
                {
                    auto signPtr =
                        m_sign[blk].template GetPtr<MemSpace, ReadOnly>(
                            streamID);
                    MultiRegions::detail::NeuBndCondKernel<ExecSpace>(
                        nUnique, signPtr, mapPtr, bndCoeffPtr, inoutPtr,
                        streamID);
                }
                else
                {
                    MultiRegions::detail::NeuBndCondKernel<ExecSpace>(
                        nUnique, mapPtr, bndCoeffPtr, inoutPtr, streamID);
                }
            }

            // Targets written by more than one boundary coefficient, summed
            // before one add.
            const size_t nGroups = m_groupCounts[blk];
            if (nGroups > 0)
            {
                auto groupOffsetPtr =
                    m_groupOffset[blk].template GetPtr<MemSpace, ReadOnly>(
                        streamID);
                auto groupTargetPtr =
                    TargetsAtInterleaveWidth(m_groupTarget[blk],
                                             interleaveWidth, streamID)
                        .template GetPtr<MemSpace, ReadOnly>(streamID);
                auto groupValuePtr =
                    m_groupValue[blk].template GetPtr<MemSpace, ReadOnly>(
                        streamID);

                if (m_anySignChange)
                {
                    auto groupSignPtr =
                        m_groupSign[blk].template GetPtr<MemSpace, ReadOnly>(
                            streamID);
                    MultiRegions::detail::NeuBndCondGroupKernel<ExecSpace>(
                        nGroups, groupOffsetPtr, groupTargetPtr, groupSignPtr,
                        groupValuePtr, inoutPtr, streamID);
                }
                else
                {
                    MultiRegions::detail::NeuBndCondGroupKernel<ExecSpace>(
                        nGroups, groupOffsetPtr, groupTargetPtr, groupValuePtr,
                        inoutPtr, streamID);
                }
            }
        }
    }
};

} // namespace Nektar::detail
