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
#include <LibUtilities/BasicUtils/StringUtils.hpp>

#include <algorithm>

#include <AdvDiffTraceFluxCFE/DiffTraceFluxCFEKernels.hpp>
#include <AdvTraceFluxCFE/AdvTraceFluxCFEKernels.hpp>
#include <DiffusionCFEVolFlux/CFEViscosityParams.hpp>
#include <LibUtilities/BasicUtils/Math/MathKernels.hpp>
#include <SolverCore/RiemannSolver/RiemannSolverKernels.hpp>
#include <SolverCore/TraceFlux/TraceFluxOpImpl.hpp>

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
 * AdvTraceFluxCFEOpImpl, and DiffuseTraceFluxKernelLauncher() negates the
 * result and accumulates the viscous contribution on top of it in the same
 * pass. The negation is what puts the inviscid and viscous terms on the same
 * side of the equation; the two are then scattered back together as one flux. A
 * block whose two sides are both collocated with the global trace skips the
 * gathers and scatters: the base's in-place kernels read and write the
 * element-local traces through the trace point maps, with ViscousPointFlux
 * as the point flux.
 *
 * Because the viscous term needs the gradient of the conserved state, this
 * class overrides the three-argument `v_Apply(in, inDeriv, flux)`.
 *
 * The Riemann solvers work in Cartesian components through the trace
 * normal, as in AdvTraceFluxCFEOpImpl: nothing is rotated or cached.
 *
 * ### Transport properties
 *
 * The viscosity law and the Prandtl number are read from the session as
 * every viscous flux of the solver reads them (#m_viscParams,
 * CFEViscosityParams): constant viscosity unless `ViscosityType` is `Variable`,
 * in which case Sutherland's law applies, and either `Prandtl` or
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
 * @see DiffuseTraceFluxKernelLauncher for the viscous half.
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

        SetUpEquationOfState(expansionList->GetSession(), m_viscParams.EoS);

        // The widest block of the interior, boundary and partition lists,
        // which sizes m_wspDeriv below: its arrays m_gloDerivT0/T1 and
        // m_penFactor would otherwise be too small for a partition block wider
        // than any interior or boundary one.
        size_t maxNTraceNPts = 0;
        for (const auto *list :
             {&this->m_intT0, &this->m_bndT0, &this->m_parT0})
        {
            for (const auto &blk : *list)
            {
                maxNTraceNPts = std::max(maxNTraceNPts, blk.m_nTraceXnPtsPad);
            }
        }

        // viscous initialisation
        m_viscParams.visc =
            CFEViscosityParams<TData>::Read(session, m_viscParams.EoS);
        ASSERTL0(m_nComp == m_nDim + 2,
                 "The viscous trace flux needs the dimension plus two "
                 "components.");

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
    /// Spatial dimension of the mesh.
    unsigned m_nDim;
    /// Number of conserved variables, equal to #m_nDim plus two.
    unsigned m_nComp;
    /// The equation of state and the viscosity law, as the kernels take
    /// them.
    ViscousTraceParams<EoSParamType, TData> m_viscParams;
    /// The flux field zero-initialised last, and the width it was zeroed
    /// at: every path writes the points it owns, so a field is zeroed when it
    /// is first handed in or its layout changes, and its padding stays zero.
    const LibUtilities::Field<TData, FieldState::Phys> *m_fluxZeroed = nullptr;
    std::vector<unsigned> m_fluxZeroedWidths;

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
        /// Per point of the padded block: non-zero where the diffusion's
        /// ghost is to be reflected, on the points of a trace whose ghost does
        /// not already reflect (see DiffusionGhostStateKernelLauncher()).
        /// `unsigned` rather than `bool`, since it is filled from a
        /// `std::vector`.
        LibUtilities::MemoryRegion<unsigned> reflectGhost;
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

            // Spread over the points of the padded block on the host, so that
            // the kernel needs no division to find a point's trace.
            size_t npTot = 1;
            for (unsigned d = 0; d < this->m_traceDim; ++d)
            {
                npTot *= this->m_bndNptsT1[b][d];
            }
            std::vector<unsigned> reflect(this->m_bndT0[b].m_nTraceXnPtsPad, 0);
            for (size_t trace = 0; trace < m_bndGhostIsReflected[b].size();
                 ++trace)
            {
                if (!m_bndGhostIsReflected[b][trace])
                {
                    std::fill_n(reflect.begin() + trace * npTot, npTot, 1u);
                }
            }
            m_bndGather[b].reflectGhost = LibUtilities::MemoryRegion<
                unsigned>::template FromVector<MemSpace>(reflect);
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

    /// Whether the viscous trace flux takes the average-of-normal-fluxes
    /// form, in which the second input of Apply() is each element's outward
    /// normal viscous flux on its traces rather than the gradient.
    bool UseAvgNormalFlux() const
    {
        return this->GetViscousTraceForm() ==
               SolverCore::TraceFluxOp<
                   TData>::ViscousTraceForm::AverageOfNormalFluxes;
    }

    /// The gradient trace the boundary traces read: the second input under
    /// the flux-of-averages form, the field set through
    /// SetBoundaryGradientTrace() otherwise.
    LibUtilities::Field<TData, FieldState::Phys> &BoundaryGradientTrace(
        LibUtilities::Field<TData, FieldState::Phys> &inDeriv)
    {
        if (!UseAvgNormalFlux())
        {
            return inDeriv;
        }
        auto *grad = this->GetBoundaryGradientTrace();
        ASSERTL0(grad || !this->HasBoundaryTraces(),
                 "The average-of-normal-fluxes form needs the gradient trace "
                 "for the boundary traces; see SetBoundaryGradientTrace().");
        return grad ? *grad : inDeriv;
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
     *                  conserved variable; under the average-of-normal-fluxes
     *                  form each element's outward normal viscous flux, one
     *                  component per conserved variable.
     * @param flux    - Trace-space output.
     */
    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
                 LibUtilities::Field<TData, FieldState::Phys> &inDeriv,
                 LibUtilities::Field<TData, FieldState::Phys> &flux) override
    {
        this->JoinBlockStreams();

        // The flux is written at the width of the state traces.
        flux.SetInterleaveWidth(in);

        // Initialises every block's storage the first time a field is
        // handed in at a width, on the trace stream the kernels writing the
        // flux run on (the GetPtr() calls below touch the first block only);
        // afterwards each path writes the points it owns, and the padding,
        // which none writes, stays zero. Appending, the flux handed in is
        // the caller's, and is added to as it stands.
        if (!this->m_append &&
            (m_fluxZeroed != &flux ||
             m_fluxZeroedWidths != flux.GetInterleaveWidth()))
        {
            flux.template Initialize<MemSpace>(0, this->m_opStreamID);
            m_fluxZeroed       = &flux;
            m_fluxZeroedWidths = flux.GetInterleaveWidth();
        }

        // The boundary pass calls the conditions' diffusion hook.
        this->SetBndCondStreams();

        auto &bndDeriv = BoundaryGradientTrace(inDeriv);

        // loop over interior and boundary traces and evaluate flux
        // using InteriorOp and BoundaryOp defined below
        switch (this->m_traceDim)
        {
            case 0:
                this->template InteriorOp<0>(in, inDeriv, flux);
                this->template BoundaryOp<0>(in, bndDeriv, flux);
                break;
            case 1:
                this->template InteriorOp<1>(in, inDeriv, flux);
                this->template BoundaryOp<1>(in, bndDeriv, flux);
                break;
            case 2:
                this->template InteriorOp<2>(in, inDeriv, flux);
                this->template BoundaryOp<2>(in, bndDeriv, flux);
                break;
            default:
                NEKERROR(ErrorUtil::efatal, "Uknown value of m_traceDim");
                break;
        }

        this->ResetBndCondStreams();

        this->ReleaseBlockStreams();
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
        if (this->m_parT0.empty())
        {
            return;
        }

        this->JoinBlockStreams();

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

        this->ReleaseBlockStreams();
    }

    /**
     * @brief Evaluate the combined flux on every interior trace.
     *
     * For each interior block: gather the conserved state and its gradient on
     * both sides, gather the trace normals and penalty factor, solve the
     * Riemann problem, negate the result and accumulate the viscous flux on
     * top in one pass, and scatter back to both elements.
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
        auto numflux             = flux.GetNumComponents();
        const unsigned dim       = TRACEDIM + 1;
        const bool avgNormalFlux = UseAvgNormalFlux();
        // Components of the second input: the gradient, or under the
        // average-of-normal-fluxes form the normal viscous flux.
        const unsigned numSecond = avgNormalFlux ? numflux : numflux * m_nDim;

        // The gather scratch is sliced per (trace, component), so it is
        // sized for the larger of the two gathers.
        this->SetWorkSpace(numSecond);

        ASSERTL1(in.GetNumComponents() == numflux,
                 "Assumption that input components are the "
                 "same as the flux components");
        ASSERTL1(inDeriv.GetNumComponents() ==
                     (avgNormalFlux ? numflux : numflux * m_nDim),
                 "The second input has the wrong number of components for "
                 "the viscous trace flux form in use");

        // One interleave width per field, every block's.
        // The layouts of the three fields, and the one the in-place kernels
        // read and write all three through, where they agree.
        auto &inLayout    = this->LayoutFor(in);
        auto &derivLayout = this->LayoutFor(inDeriv);
        auto &fluxLayout  = this->LayoutFor(flux);

        // get field pointers
        auto inPtr = in.GetBlocks()[0].template GetPtr<MemSpace, ReadOnly>(
            this->m_opStreamID);
        auto inDerivPtr =
            inDeriv.GetBlocks()[0].template GetPtr<MemSpace, ReadOnly>(
                this->m_opStreamID);
        auto fluxPtr = flux.GetBlocks()[0].template GetPtr<MemSpace, WriteOnly>(
            this->m_opStreamID);
        auto normPtr = this->m_traceNormals;

        this->template SyncTrailingBlocks<ReadOnly>(in);
        this->template SyncTrailingBlocks<ReadOnly>(inDeriv);
        this->template SyncTrailingBlocks<WriteOnly>(flux);

        // These trace geometry/weight arrays are read-only mesh data, so
        // they live in the data warehouse instead of per-op Field storage.
        // One array spanning every block, one component per block - so it is
        // indexed by the scalar offset below and not by the normals'.
        auto penFactorPtr = this->m_dataWarehouse->template GetData<MemSpace>(
            MultiRegions::IPTraceScalarKey<TData>(
                0, MultiRegions::IPTraceScalarData::IPPenaltyFactor));

        // The width the in-place kernels read the fields at, 0 for none, and
        // the point flux of each viscous trace form.
        auto *sharedLayout = this->SharedLayout(in, inDeriv, flux);
        const ViscousPointFlux<true, RiemannKernel, ExecSpace, EoSParamType,
                               dim, TData>
            avgPointFlux{m_viscParams};
        const ViscousPointFlux<false, RiemannKernel, ExecSpace, EoSParamType,
                               dim, TData>
            ipPointFlux{m_viscParams};

        for (unsigned b = 0; b < this->m_intT0.size(); ++b)
        {
            // Step 1: Get global traces associated with block b and put into
            // m_gloT0 and m_gloT1
            this->template SetInteriorParams<TRACEDIM>(b);

            // In-place path: both sides collocated, the fields at a width
            // the kernels read.
            if (avgNormalFlux ? this->template TakeBlockInPlace<dim, false>(
                                    b, avgPointFlux, sharedLayout, numflux,
                                    inPtr, inDerivPtr, fluxPtr, normPtr)
                              : this->template TakeBlockInPlace<dim, false>(
                                    b, ipPointFlux, sharedLayout, numflux,
                                    inPtr, inDerivPtr, fluxPtr, normPtr))
            {
                continue;
            }

            this->template GetInteriorTraces<TRACEDIM>(
                b, numflux, inPtr, this->m_gloT0, this->m_gloT1, inLayout);
            this->template GetInteriorTraces<TRACEDIM>(
                b, numSecond, inDerivPtr, m_gloDerivT0, m_gloDerivT1,
                derivLayout);

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
                this->m_norms, this->m_opStreamID);
            SolverCore::detail::GatherGloTraceScalarKernel<ExecSpace>(
                numBlock, this->m_npTot, penFactorPtr, G, m_penFactor,
                this->m_opStreamID);

            // Solve the Riemann problem through the gathered normals.
            RiemannKernelLauncher<RiemannKernel, EoSParamType, ExecSpace, dim>(
                m_viscParams.EoS, npTBlock, this->m_norms, this->m_gloT0,
                this->m_gloT1, this->m_flux, this->m_opStreamID);

            if (avgNormalFlux)
            {
                // The second input holds each side's outward normal viscous
                // flux; the kernel averages them and adds the penalty.
                AvgNormalFluxTraceKernelLauncher<ExecSpace>(
                    m_viscParams.EoS, npTBlock, m_nDim,
                    m_viscParams.visc.prandtl, m_viscParams.visc.muRef,
                    m_viscParams.visc.isMuVariable,
                    m_viscParams.visc.oneOverTstar,
                    m_viscParams.visc.tRatioSutherland, this->m_norms,
                    m_penFactor, this->m_gloT0, this->m_gloT1, m_gloDerivT0,
                    m_gloDerivT1, this->m_flux, this->m_opStreamID);
            }
            else
            {
                DiffuseTraceFluxKernelLauncher<ExecSpace, true>(
                    m_viscParams.EoS, npTBlock, m_nDim,
                    m_viscParams.visc.prandtl, m_viscParams.visc.muRef,
                    m_viscParams.visc.isMuVariable,
                    m_viscParams.visc.oneOverTstar,
                    m_viscParams.visc.tRatioSutherland, this->m_norms,
                    m_penFactor, m_bndEnergyWeight, this->m_gloT0,
                    this->m_gloT1, m_gloDerivT0, m_gloDerivT1, this->m_flux,
                    this->m_opStreamID);
            }
            // Step 3: Replace flux values from (m_gloT0,m_gloT1) into
            // the local flux storage
            if (this->m_append)
            {
                this->template InterpBackInteriorFlux<TRACEDIM, true>(
                    b, numflux, fluxPtr, fluxLayout);
            }
            else
            {
                this->template InterpBackInteriorFlux<TRACEDIM, false>(
                    b, numflux, fluxPtr, fluxLayout);
            }
        }
    }

    /**
     * @brief Evaluate the combined flux on every trace cut by the partitioner.
     *
     * InteriorOp() for traces whose backward element is on another rank, and
     * identical to it in every respect the physics can see. A parallel trace
     * is an *interior* trace: its exterior state and gradient are a real
     * neighbouring element's, delivered on exchange channels 0 and 1, so the
     * Riemann solve is the interior one and DiffuseTraceFluxKernelLauncher() is
     * instantiated with `IsInterior = true` - average weights of one half,
     * jump weights of one, no boundary energy-flux weight. Routing these
     * through the boundary form would reflect a perfectly good neighbour
     * state about itself and return something finite, smooth and wrong.
     *
     * The result is scattered to the local element alone; the neighbour
     * computes the same flux from the same four traces and scatters it to
     * the other element itself. The per-block penalty scalars are fetched
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

        auto numflux             = flux.GetNumComponents();
        const unsigned dim       = TRACEDIM + 1;
        const bool avgNormalFlux = UseAvgNormalFlux();
        // Components of the second input: the gradient, or under the
        // average-of-normal-fluxes form the normal viscous flux.
        const unsigned numSecond = avgNormalFlux ? numflux : numflux * m_nDim;

        // The gather scratch is sliced per (trace, component), so it is
        // sized for the larger of the two gathers.
        this->SetWorkSpace(numSecond);

        ASSERTL1(in.GetNumComponents() == numflux,
                 "Assumption that input components are the "
                 "same as the flux components");

        // One interleave width per field, every block's.
        // The layouts of the three fields, and the one the in-place kernels
        // read and write all three through, where they agree.
        auto &inLayout    = this->LayoutFor(in);
        auto &derivLayout = this->LayoutFor(inDeriv);
        auto &fluxLayout  = this->LayoutFor(flux);

        // get field pointers
        auto inPtr = in.GetBlocks()[0].template GetPtr<MemSpace, ReadOnly>(
            this->m_opStreamID);
        auto inDerivPtr =
            inDeriv.GetBlocks()[0].template GetPtr<MemSpace, ReadOnly>(
                this->m_opStreamID);
        auto fluxPtr = flux.GetBlocks()[0].template GetPtr<MemSpace, ReadWrite>(
            this->m_opStreamID);
        auto normPtr = this->m_traceNormals;

        this->template SyncTrailingBlocks<ReadOnly>(in);
        this->template SyncTrailingBlocks<ReadOnly>(inDeriv);
        this->template SyncTrailingBlocks<ReadWrite>(flux);

        auto penFactorPtr = this->m_dataWarehouse->template GetData<MemSpace>(
            MultiRegions::IPTraceScalarKey<TData>(
                0, MultiRegions::IPTraceScalarData::IPPenaltyFactor));

        // The width the in-place kernels read the fields at, 0 for none, and
        // the point flux of each viscous trace form.
        auto *sharedLayout = this->SharedLayout(in, inDeriv, flux);
        const ViscousPointFlux<true, RiemannKernel, ExecSpace, EoSParamType,
                               dim, TData>
            avgPointFlux{m_viscParams};
        const ViscousPointFlux<false, RiemannKernel, ExecSpace, EoSParamType,
                               dim, TData>
            ipPointFlux{m_viscParams};

        for (unsigned b = 0; b < this->m_parT0.size(); ++b)
        {
            // Step 1: Get global traces associated with block b and put into
            // m_gloT0 and m_gloT1; the backward sides come from the receive
            // buffers of channels 0 and 1.
            this->template SetParallelParams<TRACEDIM>(b);

            // In-place path, as InteriorOp(): the local side read through
            // the point maps, the neighbour's side from the receive buffers,
            // which channel 1 fills with the second input of either viscous
            // form.
            if (avgNormalFlux ? this->template TakeBlockInPlace<dim, true>(
                                    b, avgPointFlux, sharedLayout, numflux,
                                    inPtr, inDerivPtr, fluxPtr, normPtr)
                              : this->template TakeBlockInPlace<dim, true>(
                                    b, ipPointFlux, sharedLayout, numflux,
                                    inPtr, inDerivPtr, fluxPtr, normPtr))
            {
                continue;
            }

            this->template GetParallelTraces<TRACEDIM>(
                b, numflux, inPtr, this->m_gloT0, this->m_gloT1, inLayout, 0);
            this->template GetParallelTraces<TRACEDIM>(
                b, numSecond, inDerivPtr, m_gloDerivT0, m_gloDerivT1,
                derivLayout, 1);

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
                this->m_norms, this->m_opStreamID);
            SolverCore::detail::GatherGloTraceScalarKernel<ExecSpace>(
                numBlock, this->m_npTot, penFactorPtr, G, m_penFactor,
                this->m_opStreamID);

            // Solve the Riemann problem through the gathered normals.
            RiemannKernelLauncher<RiemannKernel, EoSParamType, ExecSpace, dim>(
                m_viscParams.EoS, npTBlock, this->m_norms, this->m_gloT0,
                this->m_gloT1, this->m_flux, this->m_opStreamID);

            if (avgNormalFlux)
            {
                // The second input holds each side's outward normal viscous
                // flux; the kernel averages them and adds the penalty.
                AvgNormalFluxTraceKernelLauncher<ExecSpace>(
                    m_viscParams.EoS, npTBlock, m_nDim,
                    m_viscParams.visc.prandtl, m_viscParams.visc.muRef,
                    m_viscParams.visc.isMuVariable,
                    m_viscParams.visc.oneOverTstar,
                    m_viscParams.visc.tRatioSutherland, this->m_norms,
                    m_penFactor, this->m_gloT0, this->m_gloT1, m_gloDerivT0,
                    m_gloDerivT1, this->m_flux, this->m_opStreamID);
            }
            else
            {
                DiffuseTraceFluxKernelLauncher<ExecSpace, true>(
                    m_viscParams.EoS, npTBlock, m_nDim,
                    m_viscParams.visc.prandtl, m_viscParams.visc.muRef,
                    m_viscParams.visc.isMuVariable,
                    m_viscParams.visc.oneOverTstar,
                    m_viscParams.visc.tRatioSutherland, this->m_norms,
                    m_penFactor, m_bndEnergyWeight, this->m_gloT0,
                    this->m_gloT1, m_gloDerivT0, m_gloDerivT1, this->m_flux,
                    this->m_opStreamID);
            }

            // Step 3: Replace flux values into the local flux storage, this
            // side only.
            if (this->m_append)
            {
                this->template InterpBackParallelFlux<TRACEDIM, true>(
                    b, numflux, fluxPtr, fluxLayout);
            }
            else
            {
                this->template InterpBackParallelFlux<TRACEDIM, false>(
                    b, numflux, fluxPtr, fluxLayout);
            }
        }
    }

    /**
     * @brief Evaluate the combined flux on every Dirichlet boundary trace.
     *
     * As InteriorOp(), with the exterior conserved state taken from the
     * boundary condition operator and the exterior gradient taken equal to the
     * interior one, since the boundary condition operator holds no gradient
     * data. DiffuseTraceFluxKernelLauncher() is instantiated in its boundary
     * form.
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
        // Nothing to read or write without a boundary block; the gradient
        // trace in particular may never have been formed on this rank.
        if (this->m_bndT0.empty())
        {
            return;
        }

        auto numflux       = flux.GetNumComponents();
        const unsigned dim = TRACEDIM + 1;

        // The gather scratch is sliced per (trace, component), so it is
        // sized for the gradient gather, the boundary traces reading the
        // gradient under either viscous trace form.
        this->SetWorkSpace(numflux * m_nDim);

        ASSERTL1(in.GetNumComponents() == numflux,
                 "Assumption that input components are the "
                 "same as the flux components");

        // One interleave width per field, every block's.
        // The layouts of the three fields, and the one the in-place kernels
        // read and write all three through, where they agree.
        auto &inLayout    = this->LayoutFor(in);
        auto &derivLayout = this->LayoutFor(inDeriv);
        auto &fluxLayout  = this->LayoutFor(flux);

        // get field pointers
        auto inPtr = in.GetBlocks()[0].template GetPtr<MemSpace, ReadOnly>(
            this->m_opStreamID);
        auto inDerivPtr =
            inDeriv.GetBlocks()[0].template GetPtr<MemSpace, ReadOnly>(
                this->m_opStreamID);
        auto fluxPtr = flux.GetBlocks()[0].template GetPtr<MemSpace, WriteOnly>(
            this->m_opStreamID);
        auto normPtr = this->m_traceNormals;

        this->template SyncTrailingBlocks<ReadOnly>(in);
        this->template SyncTrailingBlocks<ReadOnly>(inDeriv);
        this->template SyncTrailingBlocks<WriteOnly>(flux);

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

        for (unsigned b = 0; b < this->m_bndT0.size(); ++b)
        {
            // Step 1: Get global trace 't' and put into m_gloT0 and
            // m_gloT1
            this->template SetBoundaryParams<TRACEDIM>(b);
            this->template GetBoundaryTraces<TRACEDIM>(
                b, numflux, inPtr, this->m_gloT0, this->m_gloT1, inLayout);

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
                this->m_norms, this->m_opStreamID);
            SolverCore::detail::GatherGloTraceScalarKernel<ExecSpace>(
                numBlock, this->m_npTot, penFactorPtr, G, m_penFactor,
                this->m_opStreamID);
            SolverCore::detail::BroadcastGloTraceValueKernel<ExecSpace>(
                numBlock, this->m_npTot,
                m_bndGather[b].engyWeight.template GetPtr<MemSpace, ReadOnly>(
                    this->m_opStreamID),
                m_bndEnergyWeight, this->m_opStreamID);

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
            // The whole padded block, not only the points in use: the
            // kernel walks every point of it.
            DiffusionGhostStateKernelLauncher<ExecSpace>(
                npTBlock, numflux,
                m_bndGather[b].reflectGhost.template GetPtr<MemSpace, ReadOnly>(
                    this->m_opStreamID),
                this->m_gloT0, this->m_gloT1, m_gloT1Diff, this->m_opStreamID);

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
                m_gloDerivT1, derivLayout);

            // Step 2: Solve Riemann proble on  close packed variables
            // left and right  traces (m_gloT0, m_gloT1),

            // Solve the Riemann problem through the gathered normals.
            RiemannKernelLauncher<RiemannKernel, EoSParamType, ExecSpace, dim>(
                m_viscParams.EoS, npTBlock, this->m_norms, this->m_gloT0,
                this->m_gloT1, this->m_flux, this->m_opStreamID);
            DiffuseTraceFluxKernelLauncher<ExecSpace, false>(
                m_viscParams.EoS, npTBlock, m_nDim, m_viscParams.visc.prandtl,
                m_viscParams.visc.muRef, m_viscParams.visc.isMuVariable,
                m_viscParams.visc.oneOverTstar,
                m_viscParams.visc.tRatioSutherland, this->m_norms, m_penFactor,
                m_bndEnergyWeight, this->m_gloT0, m_gloT1Diff, m_gloDerivT0,
                m_gloDerivT1, this->m_flux, this->m_opStreamID);

            // Step 3: Replace flux values from (m_gloT0,m_gloT1) into
            // the local flux storage
            if (this->m_append)
            {
                this->template InterpBackDirichletFlux<TRACEDIM, true>(
                    b, numflux, fluxPtr, fluxLayout);
            }
            else
            {
                this->template InterpBackDirichletFlux<TRACEDIM, false>(
                    b, numflux, fluxPtr, fluxLayout);
            }
        }
    }
};

} // namespace Nektar::detail
