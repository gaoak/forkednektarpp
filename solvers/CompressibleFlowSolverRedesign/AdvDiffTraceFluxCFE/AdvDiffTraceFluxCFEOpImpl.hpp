///////////////////////////////////////////////////////////////////////////////
//
// File: AdvDiffTraceFluxCFEOpImpl.hpp
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
// Description: Trace Flux routines with scalar Riemann solvers
//
///////////////////////////////////////////////////////////////////////////////

#pragma once
#include <boost/algorithm/string/predicate.hpp>

#include "AdvDiffTraceFluxCFE/DiffTraceFluxCFEKernels.hpp"
#include "AdvTraceFluxCFE/AdvTraceFluxCFEOpImpl.hpp"
#include "LibUtilities/BasicUtils/Math/MathKernels.hpp"
#include "SolverCore/RiemannSolver/RiemannSolverKernels.hpp"
#include "SolverCore/TraceFlux/TraceFluxOpImpl.hpp"

namespace Nektar::detail
{

/**
 * @brief Name holder for the generated operator factory boilerplate.
 *
 * @tparam TData Floating-point representation, unused here.
 */
// dummy class definition for CMAKE boiler place code to provide consistent name
// for operator
template <typename TData> class AdvDiffTraceFluxCFEOp
{
public:
    static inline const std::string name = "AdvDiffTraceFluxCFE";
};

/**
 * @brief Combined inviscid and viscous trace flux for the compressible
 * Navier-Stokes equations.
 *
 * Evaluates both halves of the trace flux in a single pass. Each block is
 * gathered once, the Riemann problem is solved exactly as in
 * AdvTraceFluxCFEOpImpl, and DiffuseTraceFluxKernel() negates the result and
 * accumulates the viscous contribution on top of it in the same pass. The
 * negation is what puts the inviscid and viscous terms on the same side of
 * the equation; the two are then scattered back together as one flux.
 *
 * Because the viscous term needs the gradient of the conserved state, this
 * class overrides the three-argument `v_Apply(in, inDeriv, flux)`.
 *
 * ### Rotation matrix caching
 *
 * Identical to AdvTraceFluxCFEOpImpl: #m_intRotMat and #m_dirRotMat hold nine
 * entries per padded trace point in 3D, guarded by #m_updateIntRotMat and
 * #m_updateDirRotMat and invalidated by v_OnTraceNormalsChanged().
 *
 * ### Transport properties
 *
 * Reference viscosity, density and pressure are read from the session, and
 * #m_oneOverTstar is formed from them through the ideal gas law so the
 * temperature ratio Sutherland's law needs can be evaluated pointwise.
 * Viscosity is constant unless `ViscosityType` is `Variable`, in which case
 * Sutherland's law applies with #m_TRatioSutherland. Thermal conductivity is
 * expressed as a Prandtl number: give either `Prandtl` or
 * `thermalConductivity`, not both.
 *
 * @tparam RiemannKernel Flux function, as a class template on execution space,
 *                       equation-of-state parameters and dimension.
 * @tparam EoSParamType  Equation-of-state parameter pack.
 * @tparam ExecSpace     Execution space the operator runs in.
 * @tparam TData         Floating-point representation used by the field data.
 *
 * @see SolverCore::detail::TraceFluxOpImpl for the trace numbering and the
 * meaning of T0/T1.
 * @see RiemannKernelLauncher for the inviscid half.
 * @see DiffuseTraceFluxKernel for the viscous half.
 */
template <template <typename, typename, unsigned int> typename RiemannKernel,
          typename EoSParamType, typename ExecSpace, typename TData>
class AdvDiffTraceFluxCFEOpImpl
    : public SolverCore::detail::TraceFluxOpImpl<ExecSpace, TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Construct the operator, build the equation of state, size the
     * workspaces and read the transport parameters.
     *
     * @param expansionList - Expansion list the operator acts on.
     * @param components    - Names of the conserved variables.
     */
    AdvDiffTraceFluxCFEOpImpl(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : SolverCore::detail::TraceFluxOpImpl<ExecSpace, TData>(expansionList,
                                                                components)
    {
        m_nDim  = expansionList->GetExp(0)->GetShapeDimension();
        m_nComp = components.size();
        LibUtilities::SessionReaderSharedPtr session =
            expansionList->GetSession();

        SetUpEquationOfState(expansionList->GetSession(), m_EoS);

        // inviscid initialisation
        size_t maxNTraceNPts = 0, totNTraceNPts = 0;
        for (unsigned b = 0; b < this->m_intT0.size(); ++b)
        {
            totNTraceNPts += this->m_intT0[b].m_nTraceXnPtsPad;
            maxNTraceNPts =
                std::max(maxNTraceNPts, this->m_intT0[b].m_nTraceXnPtsPad);
        }

        if (m_nDim == 3)
        {
            m_intRotMat = LibUtilities::MemoryRegion<TData>(totNTraceNPts * 9);
        }

        totNTraceNPts = 0;
        for (unsigned b = 0; b < this->m_bndT0.size(); ++b)
        {
            totNTraceNPts += this->m_bndT0[b].m_nTraceXnPtsPad;
            maxNTraceNPts =
                std::max(maxNTraceNPts, this->m_bndT0[b].m_nTraceXnPtsPad);
        }

        if (m_nDim == 3)
        {
            m_dirRotMat = LibUtilities::MemoryRegion<TData>(totNTraceNPts * 9);
        }

        // The parallel blocks keep their own rotation slab, and they must be
        // in maxNTraceNPts before anything below is sized from it: the
        // rotation scratch and, further down, m_wspDeriv - whose arrays
        // m_gloDerivT0/T1 and m_penFactor would otherwise be allocated too
        // small for a parallel block wider than any interior or boundary one.
        totNTraceNPts = 0;
        for (unsigned b = 0; b < this->m_parT0.size(); ++b)
        {
            totNTraceNPts += this->m_parT0[b].m_nTraceXnPtsPad;
            maxNTraceNPts =
                std::max(maxNTraceNPts, this->m_parT0[b].m_nTraceXnPtsPad);
        }

        if (m_nDim == 3 && totNTraceNPts > 0)
        {
            m_parRotMat = LibUtilities::MemoryRegion<TData>(totNTraceNPts * 9);
        }

        // Padded so each rotation scratch starts on a vector boundary: the
        // AVX kernels reinterpret_cast these to tinysimd::simd<TData> and use
        // aligned moves, which fault on a mid-vector address.
        const size_t rotStride =
            this->PadToVectorWidth(maxNTraceNPts * m_nComp);
        m_rotwsp = LibUtilities::MemoryRegion<TData>(3 * rotStride);
        m_rotwsp.template Initialize<MemSpace>(0);
        // set up temporary pointers
        m_rot1Ptr = m_rotwsp.template GetPtr<MemSpace, WriteOnly>();
        m_rot2Ptr = m_rot1Ptr + rotStride;
        m_rot3Ptr = m_rot2Ptr + rotStride;

        // viscous initialisation
        double rhoInf, pInf;

        // reference values could probably be in seperate session section
        session->LoadReferenceValue("viscosity", m_muRef, 1.78e-05);
        session->LoadReferenceValue("density", rhoInf, 1.225);
        session->LoadReferenceValue("pressure", pInf, 101325);

        m_oneOverTstar = (rhoInf * m_EoS.gasConst()) / pInf;

        std::string viscosityType;
        session->LoadSolverInfo("ViscosityType", viscosityType, "Constant");
        if (boost::iequals(viscosityType, "Variable"))
        {
            m_isMuVariable = true;
            double Tref, Tsuth;
            session->LoadReferenceValue("Temperature", Tref, 288.15);
            session->LoadReferenceValue("Tsutherland", Tsuth, 110.0);
            m_TRatioSutherland = Tsuth / Tref;
        }
        else
        {
            m_isMuVariable = false;
        }

        double Cp = m_EoS.gamma() / (m_EoS.gamma() - 1.0) * m_EoS.gasConst();

        if (session->DefinesReferenceValue("thermalConductivity"))
        {
            ASSERTL0(!session->DefinesReferenceValue("Prandtl"),
                     "Cannot define both Prandtl and thermalConductivity.");
            double thermalConductivityRef;
            session->LoadReferenceValue("thermalConductivity",
                                        thermalConductivityRef, 1.0);
            m_Prandtl = Cp * m_muRef / thermalConductivityRef;
        }
        else
        {
            double Prandtl;
            session->LoadReferenceValue("Prandtl", Prandtl, 0.72);
            m_Prandtl = Prandtl;
        }

        // One length per array, summed to size the buffer and then walked to
        // carve the pointers, so the total cannot fall behind the carving. A
        // chunk appended to the carving alone puts the whole of it past the
        // end of the buffer, where it neither faults nor holds its values -
        // it reads back as whatever else owns that memory, which is a long
        // way from the NaNs it eventually produces. The assertion is the
        // cheap way to keep that from being a debugging exercise, so it is
        // ASSERTL0: it runs once per construction and the failure it catches
        // is silent in a release build.
        const size_t nDerivPts  = maxNTraceNPts * m_nDim * m_nComp;
        const size_t nScalarPts = maxNTraceNPts;

        // MemoryRegion rather than std::vector: these arrays reach the AVX
        // kernels, which reinterpret_cast them to tinysimd::simd<TData> and
        // use aligned moves. std::vector only guarantees max_align_t (16 bytes
        // here), which is short of the 32 AVX2 wants and the 64 AVX512 wants,
        // so its base can fault before any offset is even added. MemoryRegion
        // defaults to NektarSpaces::host_memory_alignment for this reason.
        // The strides are padded so each array also starts on a vector.
        const size_t derivStride  = this->PadToVectorWidth(nDerivPts);
        const size_t scalarStride = this->PadToVectorWidth(nScalarPts);

        m_wspDeriv = LibUtilities::MemoryRegion<TData>(2 * derivStride +
                                                       2 * scalarStride);
        m_wspDeriv.template Initialize<MemSpace>(0);

        TData *const base = m_wspDeriv.template GetPtr<MemSpace, WriteOnly>();
        TData *cursor     = base;
        m_gloDerivT0      = cursor;
        cursor += derivStride;
        m_gloDerivT1 = cursor;
        cursor += derivStride;
        m_penFactor = cursor;
        cursor += scalarStride;
        m_bndEnergyWeight = cursor;
        cursor += scalarStride;

        ASSERTL0(cursor == base + m_wspDeriv.size(),
                 "m_wspDeriv is not the exact sum of the arrays carved out of "
                 "it - add the new array's length to the size above.");

        // The exterior state the diffusion sees, which is not always the one
        // the Riemann solver sees - see the boundary loop in v_Apply().
        m_wspGloT1Diff = LibUtilities::MemoryRegion<TData>(
            this->PadToVectorWidth(maxNTraceNPts * m_nComp));
        m_wspGloT1Diff.template Initialize<MemSpace>(0);
        m_gloT1Diff = m_wspGloT1Diff.template GetPtr<MemSpace, WriteOnly>();
    }

    /**
     * @brief Ask each boundary condition what it does to the boundary flux.
     *
     * What kind of exterior state it built, and what weight it puts on the
     * diffusive flux. Both are the condition's own business - this operator
     * only supplies the trace layout to apply them in - so both are asked
     * rather than inferred from the boundary tag.
     *
     * Done on the host once: the mapping from trace to region is fixed after
     * construction, and so is the set of attached conditions.
     */
    void BuildBndEnergyWeights()
    {
        m_bndEnergyWeightPerTrace.assign(this->m_bndTraceIdT1.size(), {});
        m_bndGhostIsReflected.assign(this->m_bndTraceIdT1.size(), {});
        m_bndCondDiffState.assign(this->m_bndTraceIdT1.size(), {});
        m_bndCondDiffStateBlk.assign(this->m_bndTraceIdT1.size(), {});

        for (unsigned b = 0; b < this->m_bndTraceIdT1.size(); ++b)
        {
            const auto numTrace = this->m_bndTraceIdT1[b].size();
            m_bndEnergyWeightPerTrace[b].assign(numTrace, TData(1.0));
            m_bndGhostIsReflected[b].assign(numTrace, false);
            m_bndCondDiffState[b].assign(numTrace, nullptr);
            m_bndCondDiffStateBlk[b].assign(numTrace, 0u);

            if (!this->HaveBndStorage())
            {
                continue;
            }

            for (size_t t = 0; t < numTrace; ++t)
            {
                unsigned blk;
                size_t offset, compOffset;
                this->m_BndCondOp->GetTraceLocation(this->m_bndTraceIdT1[b][t],
                                                    blk, offset, compOffset);

                // A region no condition claims keeps whatever the session
                // evaluated: an ordinary Dirichlet value, imposed and
                // unweighted, which is what the defaults below give.
                const auto *bndCondOp = this->GetBndCondUpdateOp(blk);

                if (bndCondOp)
                {
                    // Reflective only if something has actually built the
                    // mirrored state in the boundary storage, so the operator
                    // that built it is asked rather than the tag being matched
                    // here. An inviscid Wall or a Symmetry plane becomes
                    // reflective by declaring itself so, with no change to
                    // this file.
                    m_bndGhostIsReflected[b][t] = bndCondOp->GhostIsReflected();

                    // The kernel weights the energy component alone, that
                    // being the only one a condition has so far needed to
                    // suppress - an adiabatic wall's q.n. The interface is per
                    // component, so a weight this operator cannot apply is
                    // announced rather than quietly dropped.
                    for (unsigned c = 0; c + 1 < m_nComp; ++c)
                    {
                        ASSERTL0(
                            bndCondOp->BndFluxWeight(blk, c) == TData(1.0),
                            "A boundary condition asked for a diffusive flux "
                            "weight on a component other than the energy, "
                            "which this operator does not apply.");
                    }

                    m_bndEnergyWeightPerTrace[b][t] =
                        bndCondOp->BndFluxWeight(blk, m_nComp - 1);

                    // A condition acting on the diffusion's exterior state is
                    // called back in the boundary loop of v_Apply(), where
                    // that state exists. Recorded per trace so the loop is a
                    // pointer test rather than a virtual call on every trace
                    // of every block.
                    if (bndCondOp->HasDiffusionExteriorState())
                    {
                        m_bndCondDiffState[b][t]    = bndCondOp;
                        m_bndCondDiffStateBlk[b][t] = blk;
                    }
                }
            }
        }
    }

    /// Key this operator is registered under in the OperatorFactory. Defined
    /// by the generated factory boilerplate, not in this header.
    // className - for OperatorFactory
    static std::string className;

    /**
     * @brief Factory creator function.
     *
     * @param expansionList - Expansion list the operator acts on.
     * @param components    - Names of the conserved variables.
     *
     * @return A new operator instance, owned by the caller.
     */
    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<MultiRegions::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<AdvDiffTraceFluxCFEOpImpl<
            RiemannKernel, EoSParamType, ExecSpace, TData>>(expansionList,
                                                            components);
    }

protected:
    /**
     * @brief Synchronise every block of @p field after the first.
     *
     * Memory is synchronised a block at a time, while the kernels below walk
     * a field from its first block's pointer; taking that pointer therefore
     * leaves every later block unsynchronised. Each field is walked to its
     * own length, since they do not share a block count.
     */
    template <typename MemAccess, typename TField>
    NEK_FORCE_INLINE static void SyncTrailingBlocks(TField &field)
    {
        for (unsigned blk = 1; blk < field.GetBlocks().size(); ++blk)
        {
            field.GetBlocks()[blk].template GetPtr<MemSpace, MemAccess>();
        }
    }

    /// Interleave width of an element-local field, which the gathers and
    /// scatters read and write it at; Field reshapes all its blocks together.
    template <typename TField> NEK_FORCE_INLINE static unsigned Width(TField &f)
    {
        return f.GetBlocks()[0].GetInterleaveWidth();
    }

    /// Spatial dimension of the mesh.
    unsigned m_nDim;
    /// Number of conserved variables, equal to #m_nDim plus two.
    unsigned m_nComp;
    /// Equation-of-state parameters.
    EoSParamType m_EoS;

    // inviscid
    /// Rotation scratch and the cached 3D rotation matrices for the interior
    /// and Dirichlet boundary blocks respectively.
    LibUtilities::MemoryRegion<TData> m_rotwsp, m_intRotMat, m_dirRotMat;
    /// Pointers into #m_rotwsp is unused by the apply paths,
    /// which declare their own local of the same name.
    TData *m_rot1Ptr, *m_rot2Ptr, *m_rot3Ptr;
    /// Set while the interior rotation matrices still need generating.
    bool m_updateIntRotMat = true;
    /// Set while the boundary rotation matrices still need generating.
    bool m_updateDirRotMat = true;
    /// Cached rotations for the parallel blocks, and their staleness flag.
    LibUtilities::MemoryRegion<TData> m_parRotMat;
    bool m_updateParRotMat = true;

    // viscosity parameters
    /// True when `ViscosityType` is `Variable`, selecting Sutherland's law.
    bool m_isMuVariable;
    /// Reference dynamic viscosity.
    TData m_muRef;
    /// Prandtl number, derived from `thermalConductivity` when that is given
    /// in place of `Prandtl`.
    TData m_Prandtl;

    // Sutherland's law parameters
    /// Sutherland temperature as a fraction of the reference temperature.
    TData m_TRatioSutherland;
    /// Reciprocal reference temperature, formed from the reference density,
    /// pressure and gas constant.
    TData m_oneOverTstar;

    // workspace
    /// Backing storage for #m_gloDerivT0, #m_gloDerivT1, #m_penFactor and
    /// #m_bndEnergyWeight, carved up in the constructor. Anything added to
    /// that carving needs its length added to the size there too.
    LibUtilities::MemoryRegion<TData> m_wspDeriv;
    LibUtilities::MemoryRegion<TData> m_wspGloT1Diff;
    /// Forward-side gradient of the conserved state, packed on the global
    /// trace with `nDim * nComp` components per point.
    TData *m_gloDerivT0;
    /// Backward-side gradient, same layout as #m_gloDerivT0.
    TData *m_gloDerivT1;
    /// Per-point geometric penalty factor gathered from the data warehouse.
    TData *m_penFactor;
    /// Per trace point weight on the energy component of the viscous boundary
    /// flux: zero on an adiabatic wall, one everywhere else. Filled alongside
    /// the penalty factor in the boundary loop.
    TData *m_bndEnergyWeight;
    /// Per boundary trace, resolved once from the USERDEFINEDTYPE of the
    /// region it belongs to. Indexed as m_bndTraceIdT1.
    std::vector<std::vector<TData>> m_bndEnergyWeightPerTrace;

    /// Whether a boundary trace's exterior state is already a reflection of
    /// the interior about the condition it imposes, as a wall's mirrored
    /// momentum is. Indexed as m_bndTraceIdT1.
    std::vector<std::vector<bool>> m_bndGhostIsReflected;

    /// The condition acting on the diffusion's exterior state at each boundary
    /// trace, or null where none does. Indexed as m_bndTraceIdT1.
    std::vector<std::vector<const SolverCore::BndCondUpdateOp<TData> *>>
        m_bndCondDiffState;

    /// Boundary storage block of each such trace, which the condition needs to
    /// tell its own regions apart. Indexed as m_bndTraceIdT1.
    std::vector<std::vector<unsigned>> m_bndCondDiffStateBlk;

    /**
     * @brief The boundary-only half of a block's gather metadata, resident in
     * the execution space.
     *
     * Where each trace sits in the global trace is common to every trace-flux
     * operator, and the base class mirrors it as
     * TraceFluxOpImpl::m_bndGloTraceDev. What is particular to this operator
     * is the pair below, which follow the boundary conditions rather than the
     * mesh and so are resolved by BuildBndEnergyWeights() rather than by the
     * constructor. Hence the separate, lazy upload guarded by
     * #m_bndGatherDataBuilt.
     */
    struct BndGatherDeviceData
    {
        /// Per-trace energy weight.
        LibUtilities::MemoryRegion<TData> engyWeight;
        /// Per-trace reflected-ghost flag. The region could hold `bool`; it
        /// holds `unsigned` because it is filled from a `std::vector`, and
        /// `std::vector<bool>` has no contiguous storage to copy from.
        LibUtilities::MemoryRegion<unsigned> ghostIsReflected;
    };

    /// Boundary blocks, indexed as m_bndT0.
    std::vector<BndGatherDeviceData> m_bndGather;
    /// Whether the above has been filled.
    bool m_bndGatherDataBuilt{false};

    /**
     * @brief Make the boundary gather metadata available.
     *
     * Called from DirichletOp(). Pulls BuildBndEnergyWeights() forward when it
     * has to: the weights are part of what is uploaded, so they cannot wait
     * for the boundary loop that reads them.
     */
    void EnsureBndGatherData()
    {
        if (m_bndGatherDataBuilt)
        {
            return;
        }

        // An adiabatic wall is a condition on the heat flux, not on the
        // state: with u = 0 at the wall the energy component of the viscous
        // flux is exactly q.n, so suppressing it is the condition. Resolve
        // which traces that applies to once.
        if (m_bndEnergyWeightPerTrace.empty())
        {
            BuildBndEnergyWeights();
        }

        m_bndGather.clear();
        m_bndGather.resize(this->m_bndT0.size());

        for (unsigned b = 0; b < this->m_bndT0.size(); ++b)
        {
            m_bndGather[b].engyWeight =
                LibUtilities::MemoryRegion<TData>::template FromVector<
                    MemSpace>(m_bndEnergyWeightPerTrace[b]);

            // std::vector<bool> is a bitfield, so it is widened on the host
            // before being copied rather than pointed at.
            std::vector<unsigned> reflected(m_bndGhostIsReflected[b].begin(),
                                            m_bndGhostIsReflected[b].end());
            m_bndGather[b].ghostIsReflected = LibUtilities::MemoryRegion<
                unsigned>::template FromVector<MemSpace>(reflected);
        }

        m_bndGatherDataBuilt = true;
    }

    /**
     * @brief Exterior state the *diffusion* sees on a boundary trace.
     *
     * Zhen Guo's thesis, Ch. 2, "Symmetric interior penalty Galerkin method",
     * Eq. (2.17), writes the interior penalty flux with a single averaging
     * rule, {w} = (w+ + w-)/2, and penalty beta_p = (P+1)^2 / h_T. That rule
     * is applied unchanged on a physical boundary, with the exterior state w-
     * constructed from the boundary condition (Ch. 3.2: "on physical
     * boundaries, there is a complex relation between Q+ and Q-, which are
     * implicitly defined by the physical boundary conditions").
     *
     * A half-and-half average only reproduces the imposed state g if w- is the
     * *reflection* 2g - Q+. A wall ghost already is one - mirroring momentum
     * reflects about zero velocity - but a Dirichlet ghost is g itself, since
     * that same array is what the Riemann solver needs for the advective flux.
     * Pairing g with a half average would impose only half of it and halve the
     * penalty jump with it, so the diffusion is given its own exterior state
     * here: the reflection where the condition is Dirichlet, and m_gloT1
     * unchanged where the ghost already reflects.
     *
     * The same numbers could be reached by sharing one exterior state and
     * varying the average weight per region instead, 1 where the value is
     * imposed and 1/2 where the ghost reflects. Both give {w} = g and
     * jump = 2(g - Q+); the difference is only whether the condition-specific
     * part lives in the ghost or in the weight. Keeping it in the ghost leaves
     * the kernel a direct transcription of Eq. (2.17), with one averaging rule.
     */
    TData *m_gloT1Diff;

    /**
     * @brief Invalidate the cached rotations when the normals change.
     *
     * Covers both sources of a change: the mesh-derived normals loaded from
     * the warehouse, and any supplied through SetTraceNormals().
     */
    void v_OnTraceNormalsChanged() override
    {
        m_updateIntRotMat = true;
        m_updateDirRotMat = true;
        m_updateParRotMat = true;
    }

    // The base declares each hook twice, for the two- and three-field forms;
    // overriding one form alone would hide the other.
    using SolverCore::TraceFluxOp<TData>::v_Apply;
    using SolverCore::TraceFluxOp<TData>::v_ApplyParallel;
    /**
     * @brief Evaluate the combined inviscid and viscous trace flux on the
     * interior and boundary traces.
     *
     * Dispatches to InteriorOp() and BoundaryOp() for the trace dimension
     * determined at construction. Neither needs anything from another rank,
     * so a caller may run this between BeginParallelExchange() and
     * EndParallelExchange(); the partition traces are v_ApplyParallel()'s,
     * afterwards.
     *
     * @param in      - Physical-space conserved variables.
     * @param inDeriv - Physical-space gradient of @p in, `nDim` components per
     *                  conserved variable.
     * @param flux    - Trace-space output.
     */
    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
                 LibUtilities::Field<TData, FieldState::Phys> &inDeriv,
                 LibUtilities::Field<TData, FieldState::Phys> &flux) override
    {
        // loop over interior and boundary traces and evaluate flux
        // using InteriorOp and BoundaryOp defined below
        switch (this->m_traceDim)
        {
            case 0:
                this->template InteriorOp<0>(in, inDeriv, flux);
                this->template BoundaryOp<0>(in, inDeriv, flux);
                break;
            case 1:
                this->template InteriorOp<1>(in, inDeriv, flux);
                this->template BoundaryOp<1>(in, inDeriv, flux);
                break;
            case 2:
                this->template InteriorOp<2>(in, inDeriv, flux);
                this->template BoundaryOp<2>(in, inDeriv, flux);
                break;
            default:
                NEKERROR(ErrorUtil::efatal, "Uknown value of m_traceDim");
                break;
        }
    }

    /**
     * @brief Evaluate the combined inviscid and viscous trace flux on the
     * partition-cut traces.
     *
     * Reads the state and gradient the two exchange channels delivered, so it
     * must follow EndParallelExchange(); @p flux must be the field v_Apply()
     * wrote, whose partition slots this fills. Does nothing at all in serial.
     */
    void v_ApplyParallel(
        LibUtilities::Field<TData, FieldState::Phys> &in,
        LibUtilities::Field<TData, FieldState::Phys> &inDeriv,
        LibUtilities::Field<TData, FieldState::Phys> &flux) override
    {
        switch (this->m_traceDim)
        {
            case 0:
                this->template ParallelOp<0>(in, inDeriv, flux);
                break;
            case 1:
                this->template ParallelOp<1>(in, inDeriv, flux);
                break;
            case 2:
                this->template ParallelOp<2>(in, inDeriv, flux);
                break;
            default:
                NEKERROR(ErrorUtil::efatal, "Uknown value of m_traceDim");
                break;
        }
    }

    /**
     * @brief Evaluate the combined flux on every interior trace.
     *
     * For each interior block: gather the conserved state and its gradient on
     * both sides, gather the trace normals and penalty factor, solve the
     * Riemann problem, negate the result and accumulate the viscous flux on
     * top in one pass, and scatter back to both elements.
     *
     * Clears #m_updateIntRotMat on return.
     *
     * @tparam TRACEDIM Trace dimension; must equal `m_traceDim`.
     * @param  in       Physical-space conserved variables.
     * @param  inDeriv  Physical-space gradient of @p in.
     * @param  flux     Trace-space output.
     */
    template <unsigned TRACEDIM>
    NEK_FORCE_INLINE void InteriorOp(
        LibUtilities::Field<TData, FieldState::Phys> &in,
        LibUtilities::Field<TData, FieldState::Phys> &inDeriv,
        LibUtilities::Field<TData, FieldState::Phys> &flux)
    {
        auto numflux       = flux.GetNumComponents();
        const unsigned dim = TRACEDIM + 1;

        this->SetWorkSpace(numflux);

        ASSERTL1(in.GetNumComponents() == numflux,
                 "Assumption that input components are the "
                 "same as the flux components");

        // primarily here to ensure all blocks are intialised since
        // the GetPtr() below will only intiialise first block.
        flux.template Initialize<MemSpace>(0);

        // get field pointers
        auto inPtr = in.GetBlocks()[0].template GetPtr<MemSpace, ReadOnly>();
        auto inDerivPtr =
            inDeriv.GetBlocks()[0].template GetPtr<MemSpace, ReadOnly>();
        auto fluxPtr =
            flux.GetBlocks()[0].template GetPtr<MemSpace, WriteOnly>();
        auto normPtr = this->m_traceNormals.GetBlocks()[0]
                           .template GetPtr<MemSpace, ReadOnly>();

        SyncTrailingBlocks<ReadOnly>(in);
        SyncTrailingBlocks<ReadOnly>(inDeriv);
        SyncTrailingBlocks<WriteOnly>(flux);
        SyncTrailingBlocks<ReadOnly>(this->m_traceNormals);

        // These trace geometry/weight arrays are read-only mesh data, so
        // they live in the data warehouse instead of per-op Field storage.
        // These trace geometry/weight arrays are read-only mesh data, so
        // they live in the data warehouse instead of per-op Field storage.
        // One array spanning every block, one component per block - so it is
        // indexed by the scalar offset below and not by the normals'.
        auto penFactorPtr = this->m_dataWarehouse->template GetData<MemSpace>(
            MultiRegions::IPTraceScalarKey<TData>(
                0, MultiRegions::IPTraceScalarData::IPPenaltyFactor));

        size_t rotMatOffset = 0;
        for (unsigned b = 0; b < this->m_intT0.size(); ++b)
        {
            // Step 1: Get global traces associated with block b and put into
            // m_gloT0 and m_gloT1
            this->template SetInteriorParams<TRACEDIM>(b);
            this->template GetInteriorTraces<TRACEDIM>(
                b, numflux, inPtr, this->m_gloT0, this->m_gloT1, Width(in));
            this->template GetInteriorTraces<TRACEDIM>(
                b, numflux * m_nDim, inDerivPtr, m_gloDerivT0, m_gloDerivT1,
                Width(inDeriv));

            // Step 2: Solve Riemann proble on close packed variables
            // left and right  traces (m_gloT0, m_gloT1),
            auto numBlock = this->m_intT0[b].offset.size();
            auto npTBlock = this->m_intT0[b].m_nTraceXnPtsPad;
            for (unsigned t = 0; t < numBlock; ++t)
            {
                // Single component per block, so not the normals' offset,
                // which is in their coordDim-component layout.
                //
                // Worth checking: the failure this catches - indexing this
                // array with the normals' coordDim-component offset - reads
                // adjacent trace data far more often than anything invalid, so
                // it gives a plausible wrong answer rather than a crash, and
                // only on a mesh with more than one trace block.
                ASSERTL1(
                    this->m_intGloTraceScalarOffset[b][t] + this->m_npTot <=
                        this->m_gloTraceScalarSize,
                    "Single-component trace data is read past the end of "
                    "its block: offset " +
                        std::to_string(this->m_intGloTraceScalarOffset[b][t]) +
                        " + " + std::to_string(this->m_npTot) + " points > " +
                        std::to_string(this->m_gloTraceScalarSize) + ".");
            }

            // gather the normals and the penalty factor
            const auto G =
                this->GetGloTraceOffsetView(this->m_intGloTraceDev[b]);

            SolverCore::detail::GatherGloTraceComponentsKernel<ExecSpace>(
                numBlock, this->m_npTot, npTBlock, dim, normPtr, G,
                this->m_norms);
            SolverCore::detail::GatherGloTraceScalarKernel<ExecSpace>(
                numBlock, this->m_npTot, penFactorPtr, G, m_penFactor);

            // get hold of rotation matrix if TraceDim == 2
            auto rotMatPtr =
                (TRACEDIM == 2)
                    ? (m_updateIntRotMat)
                          ? m_intRotMat.template GetPtr<MemSpace, WriteOnly>() +
                                rotMatOffset * 9
                          : m_intRotMat.template GetPtr<MemSpace, ReadOnly>() +
                                rotMatOffset * 9
                    : this->m_norms;

            // input close packed variables from both traces
            // (m_gloT0, m_gloT1), apply flux Kernel and return flux
            if (m_updateIntRotMat)
            {
                RiemannKernelLauncher<RiemannKernel, EoSParamType, ExecSpace,
                                      dim, true>(
                    m_EoS, npTBlock, this->m_norms, rotMatPtr, m_rot1Ptr,
                    m_rot2Ptr, m_rot3Ptr, this->m_gloT0, this->m_gloT1,
                    this->m_flux);
            }
            else
            {
                RiemannKernelLauncher<RiemannKernel, EoSParamType, ExecSpace,
                                      dim, false>(
                    m_EoS, npTBlock, this->m_norms, rotMatPtr, m_rot1Ptr,
                    m_rot2Ptr, m_rot3Ptr, this->m_gloT0, this->m_gloT1,
                    this->m_flux);
            }

            DiffuseTraceFluxKernel<ExecSpace, true>(
                m_EoS, npTBlock, m_nDim, m_nComp, m_Prandtl, m_muRef,
                m_isMuVariable, m_oneOverTstar, m_TRatioSutherland,
                this->m_norms, m_penFactor, m_bndEnergyWeight, this->m_gloT0,
                this->m_gloT1, m_gloDerivT0, m_gloDerivT1, this->m_flux);
            // Step 3: Replace flux values from (m_gloT0,m_gloT1) into
            // the local flux storage
            if (this->m_append)
            {
                this->template InterpBackInteriorFlux<TRACEDIM, true>(
                    b, numflux, fluxPtr, Width(flux));
            }
            else
            {
                this->template InterpBackInteriorFlux<TRACEDIM, false>(
                    b, numflux, fluxPtr, Width(flux));
            }
            rotMatOffset += this->m_intT0[b].m_nTraceXnPtsPad;
        }
        m_updateIntRotMat = false;
    }

    /**
     * @brief Evaluate the combined flux on every trace cut by the partitioner.
     *
     * InteriorOp() for traces whose backward element is on another rank, and
     * identical to it in every respect the physics can see. A parallel trace
     * is an *interior* trace: its exterior state and gradient are a real
     * neighbouring element's, delivered on exchange channels 0 and 1, so the
     * Riemann solve is the interior one and DiffuseTraceFluxKernel() is
     * instantiated with `IsInterior = true` - average weights of one half,
     * jump weights of one, no boundary energy-flux weight. Routing these
     * through the boundary form would reflect a perfectly good neighbour
     * state about itself and return something finite, smooth and wrong.
     *
     * The result is scattered to the local element alone; the neighbour
     * computes the same flux from the same four traces and scatters it to
     * the other element itself. The 3D rotations are cached in their own
     * slab, #m_parRotMat, and the per-block penalty scalars are fetched
     * through the parallel scalar offsets - partitioning multiplies trace
     * blocks, which is exactly the case the ASSERTL1 below was written for.
     *
     * Returns immediately when there are no such traces, which is every case
     * in serial.
     *
     * @tparam TRACEDIM Trace dimension; must equal `m_traceDim`.
     * @param  in       Physical-space conserved variables.
     * @param  inDeriv  Physical-space gradient of @p in.
     * @param  flux     Trace-space output.
     */
    template <unsigned TRACEDIM>
    NEK_FORCE_INLINE void ParallelOp(
        LibUtilities::Field<TData, FieldState::Phys> &in,
        LibUtilities::Field<TData, FieldState::Phys> &inDeriv,
        LibUtilities::Field<TData, FieldState::Phys> &flux)
    {
        if (this->m_parT0.empty())
        {
            return;
        }

        auto numflux       = flux.GetNumComponents();
        const unsigned dim = TRACEDIM + 1;

        this->SetWorkSpace(numflux);

        ASSERTL1(in.GetNumComponents() == numflux,
                 "Assumption that input components are the "
                 "same as the flux components");

        // get field pointers
        auto inPtr = in.GetBlocks()[0].template GetPtr<MemSpace, ReadOnly>();
        auto inDerivPtr =
            inDeriv.GetBlocks()[0].template GetPtr<MemSpace, ReadOnly>();
        auto fluxPtr =
            flux.GetBlocks()[0].template GetPtr<MemSpace, ReadWrite>();
        auto normPtr = this->m_traceNormals.GetBlocks()[0]
                           .template GetPtr<MemSpace, ReadOnly>();

        SyncTrailingBlocks<ReadOnly>(in);
        SyncTrailingBlocks<ReadOnly>(inDeriv);
        SyncTrailingBlocks<ReadWrite>(flux);
        SyncTrailingBlocks<ReadOnly>(this->m_traceNormals);

        auto penFactorPtr = this->m_dataWarehouse->template GetData<MemSpace>(
            MultiRegions::IPTraceScalarKey<TData>(
                0, MultiRegions::IPTraceScalarData::IPPenaltyFactor));

        size_t rotMatOffset = 0;
        for (unsigned b = 0; b < this->m_parT0.size(); ++b)
        {
            // Step 1: Get global traces associated with block b and put into
            // m_gloT0 and m_gloT1; the backward sides come from the receive
            // buffers of channels 0 and 1.
            this->template SetParallelParams<TRACEDIM>(b);
            this->template GetParallelTraces<TRACEDIM>(
                b, numflux, inPtr, this->m_gloT0, this->m_gloT1, 0, Width(in));
            this->template GetParallelTraces<TRACEDIM>(
                b, numflux * m_nDim, inDerivPtr, m_gloDerivT0, m_gloDerivT1, 1,
                Width(inDeriv));

            // Step 2: gather the normals and penalty factor for these traces
            auto numBlock = this->m_parT0[b].offset.size();
            auto npTBlock = this->m_parT0[b].m_nTraceXnPtsPad;
            for (unsigned t = 0; t < numBlock; ++t)
            {
                // Single component per block, so not the normals' offset; see
                // the note on the same read in InteriorOp().
                ASSERTL1(
                    this->m_parGloTraceScalarOffset[b][t] + this->m_npTot <=
                        this->m_gloTraceScalarSize,
                    "Single-component trace data is read past the end of "
                    "its block: offset " +
                        std::to_string(this->m_parGloTraceScalarOffset[b][t]) +
                        " + " + std::to_string(this->m_npTot) + " points > " +
                        std::to_string(this->m_gloTraceScalarSize) + ".");
            }

            const auto G =
                this->GetGloTraceOffsetView(this->m_parGloTraceDev[b]);

            SolverCore::detail::GatherGloTraceComponentsKernel<ExecSpace>(
                numBlock, this->m_npTot, npTBlock, dim, normPtr, G,
                this->m_norms);
            SolverCore::detail::GatherGloTraceScalarKernel<ExecSpace>(
                numBlock, this->m_npTot, penFactorPtr, G, m_penFactor);

            // get hold of rotation matrix if TraceDim == 2
            auto rotMatPtr =
                (TRACEDIM == 2)
                    ? (m_updateParRotMat)
                          ? m_parRotMat.template GetPtr<MemSpace, WriteOnly>() +
                                rotMatOffset * 9
                          : m_parRotMat.template GetPtr<MemSpace, ReadOnly>() +
                                rotMatOffset * 9
                    : this->m_norms;

            // input close packed variables from both traces
            // (m_gloT0, m_gloT1), apply flux Kernel and return flux
            if (m_updateParRotMat)
            {
                RiemannKernelLauncher<RiemannKernel, EoSParamType, ExecSpace,
                                      dim, true>(
                    m_EoS, npTBlock, this->m_norms, rotMatPtr, m_rot1Ptr,
                    m_rot2Ptr, m_rot3Ptr, this->m_gloT0, this->m_gloT1,
                    this->m_flux);
            }
            else
            {
                RiemannKernelLauncher<RiemannKernel, EoSParamType, ExecSpace,
                                      dim, false>(
                    m_EoS, npTBlock, this->m_norms, rotMatPtr, m_rot1Ptr,
                    m_rot2Ptr, m_rot3Ptr, this->m_gloT0, this->m_gloT1,
                    this->m_flux);
            }

            DiffuseTraceFluxKernel<ExecSpace, true>(
                m_EoS, npTBlock, m_nDim, m_nComp, m_Prandtl, m_muRef,
                m_isMuVariable, m_oneOverTstar, m_TRatioSutherland,
                this->m_norms, m_penFactor, m_bndEnergyWeight, this->m_gloT0,
                this->m_gloT1, m_gloDerivT0, m_gloDerivT1, this->m_flux);

            // Step 3: Replace flux values into the local flux storage, this
            // side only.
            if (this->m_append)
            {
                this->template InterpBackParallelFlux<TRACEDIM, true>(
                    b, numflux, fluxPtr, Width(flux));
            }
            else
            {
                this->template InterpBackParallelFlux<TRACEDIM, false>(
                    b, numflux, fluxPtr, Width(flux));
            }
            rotMatOffset += this->m_parT0[b].m_nTraceXnPtsPad;
        }
        m_updateParRotMat = false;
    }

    /**
     * @brief Evaluate the combined flux on every Dirichlet boundary trace.
     *
     * As InteriorOp(), with the exterior conserved state taken from the
     * boundary condition operator and the exterior gradient taken equal to the
     * interior one, since the boundary condition operator holds no gradient
     * data. DiffuseTraceFluxKernel() is instantiated in its boundary form.
     *
     * Clears #m_updateDirRotMat on return.
     *
     * @tparam TRACEDIM Trace dimension; must equal `m_traceDim`.
     * @param  in       Physical-space conserved variables.
     * @param  inDeriv  Physical-space gradient of @p in.
     * @param  flux     Trace-space output.
     */
    template <unsigned TRACEDIM>
    NEK_FORCE_INLINE void BoundaryOp(
        LibUtilities::Field<TData, FieldState::Phys> &in,
        LibUtilities::Field<TData, FieldState::Phys> &inDeriv,
        LibUtilities::Field<TData, FieldState::Phys> &flux)
    {

        auto numflux       = flux.GetNumComponents();
        const unsigned dim = TRACEDIM + 1;

        this->SetWorkSpace(numflux);

        ASSERTL1(in.GetNumComponents() == numflux,
                 "Assumption that input components are the "
                 "same as the flux components");

        // get field pointers
        auto inPtr = in.GetBlocks()[0].template GetPtr<MemSpace, ReadOnly>();
        auto inDerivPtr =
            inDeriv.GetBlocks()[0].template GetPtr<MemSpace, ReadOnly>();
        auto fluxPtr =
            flux.GetBlocks()[0].template GetPtr<MemSpace, WriteOnly>();
        auto normPtr = this->m_traceNormals.GetBlocks()[0]
                           .template GetPtr<MemSpace, ReadOnly>();

        SyncTrailingBlocks<ReadOnly>(in);
        SyncTrailingBlocks<ReadOnly>(inDeriv);
        SyncTrailingBlocks<WriteOnly>(flux);
        SyncTrailingBlocks<ReadOnly>(this->m_traceNormals);

        // These trace geometry/weight arrays are read-only mesh data, so
        // they live in the data warehouse instead of per-op Field storage.
        // These trace geometry/weight arrays are read-only mesh data, so
        // they live in the data warehouse instead of per-op Field storage.
        // One array spanning every block, one component per block - so it is
        // indexed by the scalar offset below and not by the normals'.
        auto penFactorPtr = this->m_dataWarehouse->template GetData<MemSpace>(
            MultiRegions::IPTraceScalarKey<TData>(
                0, MultiRegions::IPTraceScalarData::IPPenaltyFactor));

        EnsureBndGatherData();

        size_t rotMatOffset = 0;
        for (unsigned b = 0; b < this->m_bndT0.size(); ++b)
        {
            // Step 1: Get global trace 't' and put into m_gloT0 and
            // m_gloT1
            this->template SetBoundaryParams<TRACEDIM>(b);
            this->template GetBoundaryTraces<TRACEDIM>(
                b, numflux, inPtr, this->m_gloT0, this->m_gloT1, Width(in));

            auto numBlock = this->m_bndT0[b].offset.size();
            auto npTBlock = this->m_bndT0[b].m_nTraceXnPtsPad;

            // The normals are gathered before the gradient traces: a Neumann
            // condition - the adiabatic wall's dE/dn - needs them to resolve
            // the normal component of the gradient.
            for (unsigned t = 0; t < numBlock; ++t)
            {
                // Single component per block, so not the normals' offset,
                // which is in their coordDim-component layout.
                //
                // Worth checking: the failure this catches - indexing this
                // array with the normals' coordDim-component offset - reads
                // adjacent trace data far more often than anything invalid, so
                // it gives a plausible wrong answer rather than a crash, and
                // only on a mesh with more than one trace block.
                ASSERTL1(
                    this->m_bndGloTraceScalarOffset[b][t] + this->m_npTot <=
                        this->m_gloTraceScalarSize,
                    "Single-component trace data is read past the end of "
                    "its block: offset " +
                        std::to_string(this->m_bndGloTraceScalarOffset[b][t]) +
                        " + " + std::to_string(this->m_npTot) + " points > " +
                        std::to_string(this->m_gloTraceScalarSize) + ".");
            }

            // gather the normals, the penalty factor, and the energy weight,
            // which is one value per trace rather than one per point
            const auto G =
                this->GetGloTraceOffsetView(this->m_bndGloTraceDev[b]);

            SolverCore::detail::GatherGloTraceComponentsKernel<ExecSpace>(
                numBlock, this->m_npTot, npTBlock, dim, normPtr, G,
                this->m_norms);
            SolverCore::detail::GatherGloTraceScalarKernel<ExecSpace>(
                numBlock, this->m_npTot, penFactorPtr, G, m_penFactor);
            SolverCore::detail::BroadcastGloTraceValueKernel<ExecSpace>(
                numBlock, this->m_npTot,
                m_bndGather[b].engyWeight.template GetPtr<MemSpace, ReadOnly>(),
                m_bndEnergyWeight);

            // Build the exterior state the diffusion sees, which is not always
            // the one the Riemann solver sees.
            //
            // Every average across a trace is the half-and-half {w} =
            // (w+ + w-)/2 of Zhen-Guo Yan, "Efficient implicit spectral/hp
            // element DG techniques for compressible flows", PhD thesis,
            // Imperial College London, 2021, Ch. 2, Eq. (2.17), on boundaries
            // as well as in the interior. That is only correct if the exterior
            // state is a ghost
            // whose average with the interior is the value we mean to impose,
            // and boundary regions come in two kinds:
            //
            //   reflective  - a wall or symmetry plane, where BndCondWallCFEOp
            //                 has already built the ghost by negating momentum
            //                 in the boundary storage. m_gloT1 is (rho+,
            //                 -rhou+, E+), the average is the no-slip state,
            //                 and the ghost stands as it is for both operators.
            //
            //   imposed     - an ordinary session Dirichlet region, where
            //                 m_gloT1 holds the target value g. The Riemann
            //                 solver wants g, so m_gloT1 must keep it; the
            //                 diffusion wants the viscous flux evaluated at g,
            //                 which under a half-and-half average means the
            //                 ghost 2g - Q+. Hence the separate array.
            //
            // The same distinction could be carried as a per-region average
            // weight, 1/2 on reflective regions and 1 elsewhere, applied to the
            // unmodified boundary value: reflecting the ghost and averaging is
            // the same as not reflecting it and taking the boundary value
            // alone. Carrying it in the state keeps one averaging rule in the
            // kernels, so a new boundary condition declares which kind of
            // exterior state it builds instead of threading a weight through
            // the flux evaluation.
            //
            // Seed the whole padded block, not only the points in use: the
            // kernel walks every point of it.
            Nektar::Math::copyKernel<ExecSpace>(numflux * npTBlock,
                                                this->m_gloT1, m_gloT1Diff, 0);

            ReflectDiffusionGhostStateKernel<ExecSpace>(
                numBlock, this->m_npTot, npTBlock, numflux,
                m_bndGather[b]
                    .ghostIsReflected.template GetPtr<MemSpace, ReadOnly>(),
                this->m_gloT0, this->m_gloT1, m_gloT1Diff);

            // Any condition that acts on the diffusion's exterior state
            // and on nothing else - an isothermal wall's temperature, which
            // belongs in the viscous flux and must not reach the Riemann
            // solver. The condition owns the physics; this
            // operator owns only the layout, so it hands over a window onto
            // one trace and the stride between components.
            for (unsigned t = 0; t < numBlock; ++t)
            {
                auto *bndCondOp = m_bndCondDiffState[b][t];

                if (!bndCondOp)
                {
                    continue;
                }

                const size_t o = this->m_npTot * t;

                bndCondOp->ApplyDiffusionExteriorState(
                    m_bndCondDiffStateBlk[b][t], m_gloT1Diff + o,
                    this->m_gloT0 + o, this->m_npTot, npTBlock);
            }

            // The gradient carries any Neumann conditions - for a compressible
            // wall that is dE/dn, which with u = 0 at the wall is the
            // adiabatic dT/dn. Components with any other condition keep the
            // exterior gradient equal to the interior, as before.
            this->template GetNeumannBoundaryTraces<TRACEDIM>(
                b, numflux, m_nDim, inDerivPtr, this->m_norms, m_gloDerivT0,
                m_gloDerivT1, Width(inDeriv));

            // Step 2: Solve Riemann proble on  close packed variables
            // left and right  traces (m_gloT0, m_gloT1),

            // get hold of rotation matrix
            auto rotMatPtr =
                (TRACEDIM == 2)
                    ? (m_updateDirRotMat)
                          ? m_dirRotMat.template GetPtr<MemSpace, WriteOnly>() +
                                rotMatOffset * 9
                          : m_dirRotMat.template GetPtr<MemSpace, ReadOnly>() +
                                rotMatOffset * 9
                    : this->m_norms;

            // input close packed variables from both traces
            // (m_gloT0, m_gloT1), apply flux Kernel and
            // return flux
            if (m_updateDirRotMat)
            {
                RiemannKernelLauncher<RiemannKernel, EoSParamType, ExecSpace,
                                      dim, true>(
                    m_EoS, npTBlock, this->m_norms, rotMatPtr, m_rot1Ptr,
                    m_rot2Ptr, m_rot3Ptr, this->m_gloT0, this->m_gloT1,
                    this->m_flux);
            }
            else
            {
                RiemannKernelLauncher<RiemannKernel, EoSParamType, ExecSpace,
                                      dim, false>(
                    m_EoS, npTBlock, this->m_norms, rotMatPtr, m_rot1Ptr,
                    m_rot2Ptr, m_rot3Ptr, this->m_gloT0, this->m_gloT1,
                    this->m_flux);
            }
            DiffuseTraceFluxKernel<ExecSpace, false>(
                m_EoS, npTBlock, m_nDim, m_nComp, m_Prandtl, m_muRef,
                m_isMuVariable, m_oneOverTstar, m_TRatioSutherland,
                this->m_norms, m_penFactor, m_bndEnergyWeight, this->m_gloT0,
                m_gloT1Diff, m_gloDerivT0, m_gloDerivT1, this->m_flux);

            // Step 3: Replace flux values from (m_gloT0,m_gloT1) into
            // the local flux storage
            if (this->m_append)
            {
                this->template InterpBackDirichletFlux<TRACEDIM, true>(
                    b, numflux, fluxPtr, Width(flux));
            }
            else
            {
                this->template InterpBackDirichletFlux<TRACEDIM, false>(
                    b, numflux, fluxPtr, Width(flux));
            }
            rotMatOffset += this->m_bndT0[b].m_nTraceXnPtsPad;
        }
        m_updateDirRotMat = false;
    }
};

} // namespace Nektar::detail
