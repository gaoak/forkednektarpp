///////////////////////////////////////////////////////////////////////////////
//
// File: TestFusedElmt.hpp
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
// Description: Test fixture of the fused element-local operators against the
// separate operators they replace.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <UnitTests/MultiRegions/TestOp.hpp>

#include <MultiRegions/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseOp.hpp>
#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivOp.hpp>
#include <MultiRegions/ElmtOps/PhysTraceExtract/PhysTraceExtractOp.hpp>
#include <SolverCore/AdvDiffusion/AdvDiffFusedElmt/AdvDiffFusedElmtOp.hpp>
#include <SolverCore/Advection/AdvectionFusedElmt/AdvectionFusedElmtOp.hpp>
#include <UnitTests/SolverCore/TestFusedElmtPolicies.hpp>

#include <cmath>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::MultiRegions;

/**
 * @brief A fused element-local operator against the separate operators it
 * replaces, for the flux policy @p Policy, on the mesh of the derived fixture
 * with its expansions replaced: every element of the given expansion type,
 * number of modes and number of points per direction.
 *
 * The operator is the advection one (SolverCore::AdvectionFusedElmtOp) or,
 * for @p VISCOUS, the advection-diffusion one (SolverCore::AdvDiffFusedElmtOp).
 * The separate chain is PhysTraceExtract for the state trace; for the
 * advection-diffusion operator PhysDeriv for the gradient; the policy
 * evaluated point by point for the fluxes; IProductWRTDerivBase with scale -1
 * for the derivative inner products; and for the advection-diffusion
 * operator PhysTraceExtract::ApplyNormalComponent() of the viscous flux for
 * the normal viscous flux on the traces, and PhysTraceExtract of the gradient
 * for the gradient trace. Each case is run with the fields at width 1 and,
 * where the execution space interleaves, at its vector width, and for the
 * advection-diffusion operator with a gradient trace on every other element
 * and on every element.
 *
 * @tparam Policy   The flux policy (TestFusedElmtPolicies.hpp).
 * @tparam VISCOUS  Test the advection-diffusion operator rather than the
 *                  advection one.
 */
template <typename Policy, bool VISCOUS>
class TestFusedElmt : public TestOp<double, FieldState::Phys, FieldState::Phys,
                                    MultiRegions::DisContField>
{
public:
    using TData     = double;
    using PhysField = Field<TData, FieldState::Phys>;

    static constexpr unsigned int NDIM = Policy::NDIM;
    static constexpr unsigned int NVAR = Policy::NVAR;
    static constexpr unsigned int NAUX = Policy::NAUX;

    using FusedElmtOp =
        std::conditional_t<VISCOUS, SolverCore::AdvDiffFusedElmtOp<TData>,
                           SolverCore::AdvectionFusedElmtOp<TData>>;

    TestFusedElmt() = default;

    /**
     * @brief Run one expansion: @p type with @p nmodes modes and, when
     * positive, @p npts points per direction (the type's own otherwise).
     *
     * @return Whether every variant matched the chain within @p tol,
     *         relative to the largest entry of each reference.
     */
    bool RunExpansion(const SpatialDomains::ExpansionType type,
                      const int nmodes, const int npts, const TData tol)
    {
        if (!this->session)
        {
            this->SetSession();
        }
        SetExpansions(type, nmodes, npts);
        this->fixt_explist->SetDataWarehouse();
        Reference();

        const std::string exec = GlobalConfiguration::ExecStr();
        auto op =
            FusedElmtOp::Create(this->fixt_explist, Components(), Policy::name,
                                exec, GlobalConfiguration::ImplStr());
        if (!op)
        {
            // Refused rightly only when an element has more points than a
            // fused thread block may have threads.
            std::cout << "  no fused operator for this mesh" << std::endl;
            return !FitsDeviceBlock(exec);
        }

        const bool interleaves = NektarSpaces::GetVectorWidth<TData>(exec) > 1;
        bool match = RunVariant(*op, false, GradTraceList::None, tol);
        if (interleaves)
        {
            match = RunVariant(*op, true, GradTraceList::None, tol) && match;
        }
        if constexpr (VISCOUS)
        {
            match =
                RunVariant(*op, interleaves, GradTraceList::EveryOther, tol) &&
                match;
            match = RunVariant(*op, false, GradTraceList::All, tol) && match;
            // Without a gradient trace again, after one was written.
            match = RunVariant(*op, false, GradTraceList::None, tol) && match;
        }
        return match;
    }

protected:
    /// Which elements of each block a variant writes the gradient trace of.
    enum class GradTraceList
    {
        None,
        EveryOther,
        All
    };

    /// The inputs, laid out as Field::ToVector() returns them.
    std::vector<TData> m_stateValues, m_auxValues;
    /// The chain's outputs, laid out likewise.
    std::vector<TData> m_traceRef, m_normViscRef, m_outRef, m_gradTraceRef;

    static std::vector<std::string> Components()
    {
        std::vector<std::string> comps;
        for (unsigned int n = 0; n < NVAR; ++n)
        {
            comps.push_back("q" + std::to_string(n));
        }
        return comps;
    }

    /**
     * @brief Whether the mesh has a deformed quadrilateral element.
     *
     * Order 0 skips such meshes: with one Gauss point per direction the
     * trace normals of a deformed quadrilateral are wrong in
     * LocalRegions::QuadExp::v_ComputeTraceNormal, which reads past its edge
     * array.
     */
    bool HasDeformedQuad()
    {
        if (!this->session)
        {
            this->SetSession();
        }
        const SpatialDomains::MeshGraphSharedPtr graph =
            SpatialDomains::MeshGraphIO::Read(this->session);
        for (auto [id, quad] : graph->GetGeomMap<SpatialDomains::QuadGeom>())
        {
            if (graph->GetMeshDimension() == 2 &&
                quad->CalcGeomType() == SpatialDomains::eDeformed)
            {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Whether the operator, created on a session that names no
     * implementation, takes the default of each block
     * (MultiRegions::ElmtOp::GetOpImpl()): it exists exactly where the
     * default of every block, read from the default implementation table,
     * has a block operator of this family, which is SumFac on Serial and
     * AVX and SumFacTOP on Device.
     *
     * @param   type    The expansion type of every element.
     * @param   nmodes  The number of modes of every element.
     */
    bool DefaultImplementation(const SpatialDomains::ExpansionType type,
                               const int nmodes)
    {
        const std::string exec = GlobalConfiguration::ExecStr();

        // The fixture's session names the implementation on the command
        // line, which overrides the default: build one that does not.
        int argc      = 3;
        char **argv   = new char *[argc];
        argv[0]       = strdup("exe_name");
        argv[1]       = strdup(this->meshName.data());
        argv[2]       = strdup(("--opExecSpace=" + exec).c_str());
        this->session = LibUtilities::SessionReader::CreateInstance(argc, argv);
        for (int i = 0; i < argc; ++i)
        {
            free(argv[i]);
        }
        delete[] argv;

        SetExpansions(type, nmodes, -1);
        this->fixt_explist->SetDataWarehouse();

        const std::string familyImpl =
            exec == "Device" ? "SumFacTOP" : "SumFac";
        bool expected = FitsDeviceBlock(exec);
        for (auto &coll : MultiRegions::GetCollections(this->fixt_explist))
        {
            expected =
                expected && MultiRegions::Operator<TData>::GetDefaultOpImpl(
                                coll.GetExpVector()[0], exec) == familyImpl;
        }
        const bool created = static_cast<bool>(FusedElmtOp::Create(
            this->fixt_explist, Components(), Policy::name, exec));
        if (created != expected)
        {
            std::cout << "  modes " << nmodes << ": operator "
                      << (created ? "created" : "not created")
                      << " where the default implementation "
                      << (expected ? "has" : "has no") << " block operator"
                      << std::endl;
        }
        return created == expected;
    }

    /// Whether every element has no more points than a fused thread block
    /// may have threads on @p exec (SolverCore::fusedElmtMaxBlockSize).
    bool FitsDeviceBlock(const std::string &exec) const
    {
        bool fits = true;
        if (exec == "Device")
        {
            for (size_t e = 0; e < this->fixt_explist->GetExpSize(); ++e)
            {
                fits = fits &&
                       static_cast<unsigned int>(
                           this->fixt_explist->GetExp(e)->GetTotPoints()) <=
                           SolverCore::fusedElmtMaxBlockSize;
            }
        }
        return fits;
    }

    /// Read the mesh and give every element @p type with @p nmodes modes
    /// and @p npts points, then build the expansion list.
    void SetExpansions(const SpatialDomains::ExpansionType type,
                       const int nmodes, const int npts)
    {
        auto graph = SpatialDomains::MeshGraphIO::Read(this->session);
        for (const auto &var : this->session->GetVariables())
        {
            for (const auto &expInfo : graph->GetExpansionInfo(var))
            {
                expInfo.second->m_basisKeyVector =
                    SpatialDomains::MeshGraph::DefineBasisKeyFromExpansionType(
                        expInfo.second->m_geomPtr, type, nmodes, npts);
            }
        }
        const auto &vars = this->session->GetVariables();
        this->fixt_explist =
            MemoryManager<MultiRegions::DisContField>::AllocateSharedPtr(
                this->session, graph, vars[0], true, true,
                Collections::eNoCollection);
    }

    /**
     * @brief The state and the auxiliary fields, every entry different: a
     * positive density and pressure, a velocity of order one and an
     * auxiliary field near one, so that the fluxes are those of a physical
     * state. @p npts entries per component.
     */
    void StateValues(const size_t npts)
    {
        const TData gamma = 1.4;
        m_stateValues.assign(NVAR * npts, 0.0);
        m_auxValues.assign(NAUX * npts, 0.0);
        for (size_t q = 0; q < npts; ++q)
        {
            const TData x        = TData(0.37) * q;
            const TData rho      = 1.0 + 0.2 * std::sin(x + 0.3);
            const TData pressure = 1.0 + 0.1 * std::sin(0.7 * x + 1.1);
            TData kinetic        = 0.0;
            m_stateValues[q]     = rho;
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                const TData velocity              = 0.3 * std::cos(1.3 * x + d);
                m_stateValues[(1 + d) * npts + q] = rho * velocity;
                kinetic += 0.5 * rho * velocity * velocity;
            }
            m_stateValues[(NDIM + 1) * npts + q] =
                pressure / (gamma - 1.0) + kinetic;
            for (unsigned int a = 0; a < NAUX; ++a)
            {
                m_auxValues[a * npts + q] = 1.0 + 0.3 * std::sin(0.9 * x + a);
            }
        }
    }

    /// The separate chain, in the execution space under test, from the
    /// inputs StateValues() makes.
    void Reference()
    {
        const auto &explist = this->fixt_explist;
        const auto comps    = Components();
        const auto physAttr =
            GetBlockAttributes<TData, FieldState::Phys>(explist);
        const auto traceAttr =
            GetLocTraceBlockAttributes<TData, FieldState::Phys>(explist);
        const std::string exec = GlobalConfiguration::ExecStr();

        PhysField state("state", physAttr, NVAR, 1);
        PhysField flux("flux", physAttr, NVAR * NDIM, 1);
        PhysField trace("trace", traceAttr, NVAR, 1);
        PhysField out("out", physAttr, NVAR, 1);
        size_t npts = 0;
        for (const auto &attr : physAttr)
        {
            npts += attr.GetNumElements() * attr.GetNumData();
        }
        state.template Initialize<NektarSpaces::HostSpace>(0.0);
        flux.template Initialize<NektarSpaces::HostSpace>(0.0);
        trace.template Initialize<NektarSpaces::HostSpace>(0.0);
        out.template Initialize<NektarSpaces::HostSpace>(0.0);
        StateValues(npts);
        state.template CopyVector<NektarSpaces::HostSpace>(m_stateValues);

        auto traceOp =
            PhysTraceExtractOp<TData>::Create(explist, comps, exec, "SumFac");
        traceOp->Apply(state, trace);
        m_traceRef = trace.template ToVector<TData>();

        // The gradient, direction-major per variable as PhysDeriv lays it
        // out, component n * NDIM + d.
        std::vector<TData> gradValues(NVAR * NDIM * npts, 0.0);
        if constexpr (VISCOUS)
        {
            PhysField grad("grad", physAttr, NVAR * NDIM, 1);
            grad.template Initialize<NektarSpaces::HostSpace>(0.0);
            PhysDerivOp<TData>::Create(explist, comps, exec, "SumFac")
                ->Apply(state, grad);
            gradValues = grad.template ToVector<TData>();

            std::vector<std::string> gradComps(NVAR * NDIM, "g");
            PhysField gradTrace("gradTrace", traceAttr, NVAR * NDIM, 1);
            gradTrace.template Initialize<NektarSpaces::HostSpace>(0.0);
            PhysTraceExtractOp<TData>::Create(explist, gradComps, exec,
                                              "SumFac")
                ->Apply(grad, gradTrace);
            m_gradTraceRef = gradTrace.template ToVector<TData>();
        }

        // The policy point by point.
        Policy policy(this->session);
        std::vector<TData> fluxValues(NVAR * NDIM * npts, 0.0);
        std::vector<TData> viscValues(NVAR * NDIM * npts, 0.0);
        for (size_t q = 0; q < npts; ++q)
        {
            // The advection operator has no viscous flux, and reads no
            // gradient.
            TData statePt[NVAR], fluxPt[NDIM][NVAR],
                viscFluxPt[NDIM][NVAR] = {}, auxPt[NAUX > 0 ? NAUX : 1];
            [[maybe_unused]] TData gradPt[NDIM][NVAR];
            for (unsigned int n = 0; n < NVAR; ++n)
            {
                statePt[n] = m_stateValues[n * npts + q];
                for (unsigned int d = 0; d < NDIM; ++d)
                {
                    gradPt[d][n] = gradValues[(n * NDIM + d) * npts + q];
                }
            }
            for (unsigned int a = 0; a < NAUX; ++a)
            {
                auxPt[a] = m_auxValues[a * npts + q];
            }
            // The policy's fluxes, combined here rather than through the
            // operator's own wrappers: flux = viscFlux - advFlux.
            const TData *auxArg = NAUX > 0 ? auxPt : nullptr;
            policy.AdvFlux(statePt, auxArg, fluxPt);
            if constexpr (VISCOUS)
            {
                policy.ViscFlux(statePt, gradPt, auxArg, viscFluxPt);
            }
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                for (unsigned int n = 0; n < NVAR; ++n)
                {
                    fluxPt[d][n] =
                        (VISCOUS ? viscFluxPt[d][n] : TData(0)) - fluxPt[d][n];
                }
            }
            for (unsigned int n = 0; n < NVAR; ++n)
            {
                for (unsigned int d = 0; d < NDIM; ++d)
                {
                    fluxValues[(n * NDIM + d) * npts + q] = fluxPt[d][n];
                    viscValues[(n * NDIM + d) * npts + q] = viscFluxPt[d][n];
                }
            }
        }

        flux.template CopyVector<NektarSpaces::HostSpace>(fluxValues);
        auto iprodOp = IProductWRTDerivBaseOp<FieldState::Phys, TData>::Create(
            explist, comps, exec, "SumFac");
        iprodOp->SetScale(-1.0);
        iprodOp->Apply(flux, out);
        m_outRef = out.template ToVector<TData>();

        if constexpr (VISCOUS)
        {
            PhysField visc("visc", physAttr, NVAR * NDIM, 1);
            PhysField normVisc("normVisc", traceAttr, NVAR, 1);
            visc.template Initialize<NektarSpaces::HostSpace>(0.0);
            normVisc.template Initialize<NektarSpaces::HostSpace>(0.0);
            visc.template CopyVector<NektarSpaces::HostSpace>(viscValues);
            traceOp->ApplyNormalComponent(visc, normVisc);
            m_normViscRef = normVisc.template ToVector<TData>();
        }
    }

    /// The listed elements of each block for @p list.
    std::vector<std::vector<unsigned int>> GradTraceElements(
        const GradTraceList list) const
    {
        const auto physAttr =
            GetBlockAttributes<TData, FieldState::Phys>(this->fixt_explist);
        std::vector<std::vector<unsigned int>> elements(physAttr.size());
        for (size_t blk = 0; blk < physAttr.size(); ++blk)
        {
            for (unsigned int e = 0; e < physAttr[blk].GetNumElements(); ++e)
            {
                if (list == GradTraceList::All || e % 2 == 0)
                {
                    elements[blk].push_back(e);
                }
            }
        }
        return elements;
    }

    /// One variant: the fused operator from the inputs the chain took.
    bool RunVariant(FusedElmtOp &op, const bool interleaved,
                    const GradTraceList list, const TData tol)
    {
        const auto &explist = this->fixt_explist;
        const auto physAttr =
            GetBlockAttributes<TData, FieldState::Phys>(explist);
        const auto traceAttr =
            GetLocTraceBlockAttributes<TData, FieldState::Phys>(explist);
        const std::string exec = GlobalConfiguration::ExecStr();

        PhysField state("state", physAttr, NVAR, 1);
        PhysField trace("trace", traceAttr, NVAR, 1);
        PhysField out("out", physAttr, NVAR, 1);
        state.template Initialize<NektarSpaces::HostSpace>(0.0);
        trace.template Initialize<NektarSpaces::HostSpace>(0.0);
        out.template Initialize<NektarSpaces::HostSpace>(0.0);
        state.template CopyVector<NektarSpaces::HostSpace>(m_stateValues);
        PhysField aux;
        if constexpr (NAUX > 0)
        {
            aux = PhysField("aux", physAttr, NAUX, 1);
            aux.template Initialize<NektarSpaces::HostSpace>(0.0);
            aux.template CopyVector<NektarSpaces::HostSpace>(m_auxValues);
        }
        PhysField normVisc;
        if constexpr (VISCOUS)
        {
            normVisc = PhysField("normVisc", traceAttr, NVAR, 1);
            normVisc.template Initialize<NektarSpaces::HostSpace>(0.0);
        }

        // The gradient trace on the listed elements only, in list order.
        const auto elements = GradTraceElements(list);
        PhysField gradTrace;
        if constexpr (VISCOUS)
        {
            if (list != GradTraceList::None)
            {
                const auto vectorWidth =
                    NektarSpaces::max_vector_width<TData>::value;
                std::vector<BlockAttributes<FieldState::Phys>> gradTraceAttr;
                for (size_t blk = 0; blk < traceAttr.size(); ++blk)
                {
                    const size_t nelmt = elements[blk].size();
                    gradTraceAttr.push_back(
                        {nelmt,
                         ((nelmt + vectorWidth - 1) / vectorWidth) *
                             vectorWidth,
                         traceAttr[blk].GetNumData(), 1});
                }
                gradTrace =
                    PhysField("gradTrace", gradTraceAttr, NVAR * NDIM, 1);
                gradTrace.template Initialize<NektarSpaces::HostSpace>(0.0);
                op.SetGradientTrace(&gradTrace, &elements);
            }
            else
            {
                op.SetGradientTrace(nullptr, nullptr);
            }
        }

        if (interleaved)
        {
            // Interleaved at the execution space's vector width: the SIMD
            // width, the host operators' own, or the warp size.
            const unsigned int interleaveWidth =
                NektarSpaces::GetVectorWidth<TData>(exec);
            state.ReshapeStorage(interleaveWidth, exec);
            if constexpr (NAUX > 0)
            {
                aux.ReshapeStorage(interleaveWidth, exec);
            }
        }
        if constexpr (VISCOUS && NAUX > 0)
        {
            op.Apply(state, aux, trace, normVisc, out);
        }
        else if constexpr (VISCOUS)
        {
            op.Apply(state, trace, normVisc, out);
        }
        else if constexpr (NAUX > 0)
        {
            op.Apply(state, aux, trace, out);
        }
        else
        {
            op.Apply(state, trace, out);
        }

        bool match =
            Matches("out", out.template ToVector<TData>(), m_outRef, tol);
        match = Matches("trace", trace.template ToVector<TData>(), m_traceRef,
                        tol) &&
                match;
        if constexpr (VISCOUS)
        {
            match =
                Matches("traceNormVisc", normVisc.template ToVector<TData>(),
                        m_normViscRef, tol) &&
                match;
        }
        if (list != GradTraceList::None)
        {
            match = Matches("gradTrace", gradTrace.template ToVector<TData>(),
                            GradTraceExpected(elements, traceAttr, gradTrace),
                            tol) &&
                    match;
        }
        match = Matches("state", state.template ToVector<TData>(),
                        m_stateValues, TData(0)) &&
                match;
        if (!match)
        {
            std::cout << "  mismatch: interleaved " << interleaved
                      << ", gradient trace list " << static_cast<int>(list)
                      << std::endl;
        }
        return match;
    }

    /// The chain's gradient trace of the listed elements, laid out as the
    /// compact gradient trace field's ToVector() returns it.
    std::vector<TData> GradTraceExpected(
        const std::vector<std::vector<unsigned int>> &elements,
        const std::vector<BlockAttributes<FieldState::Phys>> &traceAttr,
        PhysField &gradTrace) const
    {
        size_t fullCompSize = 0;
        for (const auto &attr : traceAttr)
        {
            fullCompSize += attr.GetNumElements() * attr.GetNumData();
        }
        const size_t compactCompSize =
            gradTrace.template ToVector<TData>().size() / (NVAR * NDIM);

        std::vector<TData> expected(NVAR * NDIM * compactCompSize, 0.0);
        for (unsigned int c = 0; c < NVAR * NDIM; ++c)
        {
            size_t fullOffset    = c * fullCompSize;
            size_t compactOffset = c * compactCompSize;
            for (size_t blk = 0; blk < traceAttr.size(); ++blk)
            {
                const unsigned int ndata = traceAttr[blk].GetNumData();
                for (size_t k = 0; k < elements[blk].size(); ++k)
                {
                    for (unsigned int i = 0; i < ndata; ++i)
                    {
                        expected[compactOffset + k * ndata + i] =
                            m_gradTraceRef[fullOffset +
                                           elements[blk][k] * ndata + i];
                    }
                }
                fullOffset += traceAttr[blk].GetNumElements() * ndata;
                compactOffset += elements[blk].size() * ndata;
            }
        }
        return expected;
    }

    /// Whether @p out matches @p expected within @p tol times the largest
    /// entry of @p expected, or times machine epsilon when that is smaller:
    /// the inputs are of order one, so an output below epsilon is zero up
    /// to round-off (the gradient of a one-mode expansion on a collapsed
    /// shape).
    static bool Matches(const std::string &what, const std::vector<TData> &out,
                        const std::vector<TData> &expected, const TData tol)
    {
        if (out.size() != expected.size())
        {
            std::cout << "  " << what << ": " << out.size()
                      << " entries against " << expected.size() << std::endl;
            return false;
        }
        TData norm = std::numeric_limits<TData>::epsilon(), err = 0;
        for (size_t i = 0; i < out.size(); ++i)
        {
            norm = std::max(norm, std::abs(expected[i]));
            err  = std::max(err, std::isfinite(out[i])
                                     ? std::abs(out[i] - expected[i])
                                     : std::numeric_limits<TData>::max());
        }
        if (err > tol * norm)
        {
            std::cout << "  " << what << ": max error " << err
                      << " against max entry " << norm << std::endl;
            return false;
        }
        return true;
    }
};

// clang-format off
/// A fixture for @p policy on the mesh @p filename, testing the
/// advection-diffusion operator for @p viscous and the advection one
/// otherwise.
#define TEST_FUSEDELMT_FIXTURE(type, policy, viscous, filename)                \
    class type                                                                 \
        : public TestFusedElmt<Nektar::UnitTests::policy, viscous>             \
    {                                                                          \
    public:                                                                    \
        type()                                                                 \
        {                                                                      \
            meshName = filename;                                               \
        }                                                                      \
    };

// Orders 1 to 8, each with the expansion type's own points and with one
// point more per direction, on the modified and the orthogonal bases, and
// order 0 on the orthogonal basis (a one-mode modified basis is not a
// constant) unless the mesh has a deformed quadrilateral (see
// HasDeformedQuad()): the compile-time and the runtime sizes of the size
// switch, more than the eight points per direction the line-blocked kernels
// hold in registers, and trace rules that are and are not the volume rules.
#define TEST_FUSEDELMT_CASE(test_name, test, tol)                              \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        const bool orderZero = !HasDeformedQuad();                             \
        for (const auto type :                                                 \
             {SpatialDomains::eModified, SpatialDomains::eOrthogonal})         \
        {                                                                      \
            const int minModes =                                               \
                type == SpatialDomains::eOrthogonal && orderZero ? 1 : 2;      \
            for (int nmodes = minModes; nmodes <= 9; ++nmodes)                 \
            {                                                                  \
                for (const int npts : {-1, nmodes + 1})                        \
                {                                                              \
                    const bool match = RunExpansion(type, nmodes, npts, tol);  \
                    if (!match)                                                \
                    {                                                          \
                        std::cout << "Failed: type " << type << ", modes "     \
                                  << nmodes << ", points " << npts             \
                                  << std::endl;                                \
                    }                                                          \
                    BOOST_TEST(match);                                         \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }

// Orders 1 to 8 on a session that names no implementation: the operator
// takes the default of each block (DefaultImplementation()).
#define TEST_FUSEDELMT_DEFAULT_CASE(test_name, test)                          \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        for (int nmodes = 2; nmodes <= 9; ++nmodes)                            \
        {                                                                      \
            BOOST_TEST(                                                        \
                DefaultImplementation(SpatialDomains::eModified, nmodes));     \
        }                                                                      \
    }
// clang-format on
