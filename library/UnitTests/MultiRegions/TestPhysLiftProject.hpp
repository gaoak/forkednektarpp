///////////////////////////////////////////////////////////////////////////////
//
// File: TestPhysLiftProject.hpp
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
// Description: Fixture of the PhysLiftProject operator unit test: the operator
// against the four-operator chain it replaces.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <UnitTests/MultiRegions/TestOp.hpp>

#include <LibUtilities/BasicUtils/Math/Math.hpp>
#include <MultiRegions/ElmtOps/BwdTrans/BwdTransOp.hpp>
#include <MultiRegions/ElmtOps/IProductWRTBase/IProductWRTBaseOp.hpp>
#include <MultiRegions/ElmtOps/IProductWRTPhysTrace/IProductWRTPhysTraceOp.hpp>
#include <MultiRegions/ElmtOps/MultiplyByElmtInvMass/MultiplyByElmtInvMassOp.hpp>
#include <MultiRegions/ElmtOps/PhysLiftProject/PhysLiftProjectOp.hpp>

#include <cmath>
#include <set>
#include <string>
#include <vector>

using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::MultiRegions;

/**
 * @brief The lift-and-project operator against the chain it replaces,
 * IProductWRTPhysTrace (appending) + IProductWRTBase without integration +
 * MultiplyByElmtInvMass + BwdTrans, on the mesh of the derived fixture with
 * its expansions replaced: every element of the given expansion type,
 * number of modes and number of points per direction.
 *
 * Each case is run with and without the trace, with the scale left at one
 * and overwriting, and with a scale and appending onto a field that already
 * holds values. Where the execution space interleaves, the case is run
 * again with the inputs handed in at the operator's width, which the weak
 * DG right-hand side does, and the inputs are checked to come back intact.
 */
template <typename TData>
class TestPhysLiftProject
    : public TestOp<TData, FieldState::Phys, FieldState::Phys,
                    MultiRegions::DisContField>
{
public:
    TestPhysLiftProject() = default;

    /**
     * @brief Run one expansion: @p type with @p nmodes modes and, when
     * positive, @p npts points per direction (the type's own otherwise).
     *
     * @return Whether every variant matched the chain within @p tol,
     *         relative to the largest entry of the reference.
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
        RecordForms();

        bool match = true;
        for (const bool lift : {true, false})
        {
            for (const bool append : {false, true})
            {
                const TData scale = append ? TData(-0.7) : TData(1.0);
                match = RunVariant(lift, append, scale, false, tol) && match;
                if (GlobalConfiguration::ExecStr() == "AVX")
                {
                    match = RunVariant(lift, append, scale, true, tol) && match;
                }
            }
        }
        return match;
    }

    /// The forms and geometry types the cases run so far reached, as
    /// "Pointwise/affine" and so on.
    std::set<std::string> m_forms;

protected:
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

    /// Note the form and geometry type of every block.
    void RecordForms()
    {
        const auto nblocks =
            GetBlockAttributes<TData, FieldState::Phys>(this->fixt_explist)
                .size();
        for (unsigned int blk = 0; blk < nblocks; ++blk)
        {
            const auto exp =
                GetCollection(this->fixt_explist, blk).GetExpVector()[0];
            const auto form = MultiRegions::detail::PhysLiftProjectFormOf(exp);
            const std::string name =
                form == MultiRegions::detail::PhysLiftProjectForm::Pointwise
                    ? "Pointwise"
                : form == MultiRegions::detail::PhysLiftProjectForm::StdInvMass
                    ? "StdInvMass"
                    : "ElmtInvMass";
            const bool deformed =
                exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
            m_forms.insert(name + (deformed ? "/deformed" : "/affine"));
        }
    }

    /// Values for @p field, element-local and different in every entry and
    /// component.
    static std::vector<TData> Values(Field<TData, FieldState::Phys> &field,
                                     const TData seed)
    {
        std::vector<TData> values = field.template ToVector<TData>();
        for (size_t i = 0; i < values.size(); ++i)
        {
            values[i] = TData(0.25) + std::sin(seed + TData(0.37) * i) +
                        TData(0.5) * std::cos(TData(1.3) * seed + 0.11 * i);
        }
        return values;
    }

    /// One variant: the operator and the chain from the same inputs.
    bool RunVariant(const bool lift, const bool append, const TData scale,
                    const bool interleaved, const TData tol)
    {
        const auto &explist = this->fixt_explist;
        const auto ncomp    = this->session->GetVariables().size();
        const std::vector<std::string> comps(this->session->GetVariables());
        const auto physAttr =
            GetBlockAttributes<TData, FieldState::Phys>(explist);
        const auto coefAttr =
            GetBlockAttributes<TData, FieldState::Coeff>(explist);
        const auto traceAttr =
            GetLocTraceBlockAttributes<TData, FieldState::Phys>(explist);

        Field<TData, FieldState::Phys> in("in", physAttr, ncomp, 1);
        Field<TData, FieldState::Phys> trace("trace", traceAttr, ncomp, 1);
        Field<TData, FieldState::Phys> out("out", physAttr, ncomp, 1);
        Field<TData, FieldState::Phys> expected("expected", physAttr, ncomp, 1);
        Field<TData, FieldState::Phys> sum("sum", physAttr, ncomp, 1);
        Field<TData, FieldState::Coeff> coef("coef", coefAttr, ncomp, 1);
        Field<TData, FieldState::Coeff> coef2("coef2", coefAttr, ncomp, 1);
        in.template Initialize<NektarSpaces::HostSpace>(0.0);
        trace.template Initialize<NektarSpaces::HostSpace>(0.0);
        out.template Initialize<NektarSpaces::HostSpace>(0.0);
        expected.template Initialize<NektarSpaces::HostSpace>(0.0);

        const auto inValues    = Values(in, 0.1);
        const auto traceValues = Values(trace, 0.7);
        const auto outValues   = Values(out, 1.9);
        in.template CopyVector<NektarSpaces::HostSpace>(inValues);
        trace.template CopyVector<NektarSpaces::HostSpace>(traceValues);
        sum.template CopyVector<NektarSpaces::HostSpace>(inValues);
        if (append)
        {
            out.template CopyVector<NektarSpaces::HostSpace>(outValues);
            expected.template CopyVector<NektarSpaces::HostSpace>(outValues);
        }

        // The chain, in the execution space under test.
        const std::string exec = GlobalConfiguration::ExecStr();
        auto liftOp = IProductWRTPhysTraceOp<TData>::Create(explist, comps,
                                                            exec, "SumFac");
        auto bTransOp =
            IProductWRTBaseOp<TData>::Create(explist, comps, exec, "SumFac");
        auto invMassOp = MultiplyByElmtInvMassOp<TData>::Create(
            explist, comps, exec, "Generic");
        auto bwdTransOp =
            BwdTransOp<TData>::Create(explist, comps, exec, "SumFac");
        liftOp->SetAppend(true);
        bTransOp->SetIntegration(false);
        bTransOp->SetScale(scale);
        bwdTransOp->SetAppend(append);
        if (lift)
        {
            liftOp->Apply(trace, sum);
        }
        bTransOp->Apply(sum, coef);
        invMassOp->Apply(coef, coef2);
        bwdTransOp->Apply(coef2, expected);

        // The operator.
        auto op = PhysLiftProjectOp<TData>::Create(explist, comps, exec);
        op->SetScale(scale);
        op->SetAppend(append);
        if (interleaved)
        {
            // Interleaved at the SIMD width, the host operators' own.
            const unsigned int interleaveWidth = tinysimd::simd<TData>::width;
            in.ReshapeStorage(interleaveWidth, exec);
            trace.ReshapeStorage(interleaveWidth, exec);
        }
        if (lift)
        {
            op->Apply(in, trace, out);
        }
        else
        {
            op->Apply(in, out);
        }

        bool match = Matches(out.template ToVector<TData>(),
                             expected.template ToVector<TData>(), tol);
        match =
            Matches(in.template ToVector<TData>(), inValues, TData(0)) && match;
        match =
            Matches(trace.template ToVector<TData>(), traceValues, TData(0)) &&
            match;
        if (!match)
        {
            std::cout << "  mismatch: lift " << lift << ", append " << append
                      << ", interleaved " << interleaved << std::endl;
        }
        return match;
    }

    /// Whether @p out matches @p expected within @p tol times the largest
    /// entry of @p expected.
    static bool Matches(const std::vector<TData> &out,
                        const std::vector<TData> &expected, const TData tol)
    {
        TData norm = 0, err = 0;
        for (size_t i = 0; i < out.size(); ++i)
        {
            norm = std::max(norm, std::abs(expected[i]));
            err  = std::max(err, std::isfinite(out[i])
                                     ? std::abs(out[i] - expected[i])
                                     : std::numeric_limits<TData>::max());
        }
        if (err > tol * norm)
        {
            std::cout << "  max error " << err << " against max entry " << norm
                      << std::endl;
            return false;
        }
        return true;
    }
};

// clang-format off
#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
#define TEST(type, filename)                                                   \
    class type : public TestPhysLiftProject<double>                            \
    {                                                                          \
    public:                                                                    \
        type()                                                                 \
        {                                                                      \
            meshName = filename;                                               \
        }                                                                      \
    };
#else
#define TEST(type, filename)
#endif
// clang-format on

TEST(Seg, "run/segment.xml")
TEST(SegCurved, "run/segment_curved.xml")
TEST(QuadTri, "run/square_all_elements.xml")
TEST(QuadTriCurved, "run/square_all_elements_curved.xml")
TEST(Hex, "run/hex.xml")
TEST(HexCurved, "run/hex_curved.xml")
TEST(Prism, "run/prism.xml")
TEST(PrismCurved, "run/prism_curved.xml")
TEST(Pyr, "run/pyr.xml")
TEST(PyrCurved, "run/pyr_curved.xml")
TEST(Tet, "run/tet.xml")
TEST(TetCurved, "run/tet_curved.xml")
