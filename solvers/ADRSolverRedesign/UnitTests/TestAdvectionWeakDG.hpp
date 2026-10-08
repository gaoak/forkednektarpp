///////////////////////////////////////////////////////////////////////////////
//
// File: TestAdvectionWeakDG.hpp
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

#include "ADRSolverRedesign/LinearAdvVolumeFlux/LinearAdvVolumeFluxOp.hpp"
#include "SolverCore/Advection/AdvectionWeakDG/AdvectionWeakDGOp.hpp"
#include "SolverCore/TraceFlux/TraceFluxOp.hpp"
#include <MultiRegions/ElmtOps/PhysTraceExtract/PhysTraceExtractOp.hpp>

#include <ADRSolver/EquationSystems/UnsteadyAdvection.h>
#include <SolverUtils/RiemannSolvers/UpwindSolver.h>

#include "UnitTests/Operators/TestOp.hpp"

using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::MultiRegions;
using namespace Nektar::SolverCore;

template <typename TData>
class TestAdvectionWeakDG
    : public TestOp<TData, FieldState::Phys, FieldState::Phys,
                    MultiRegions::DisContField>
{
public:
    TestAdvectionWeakDG() = default;

    void SetTestCase()
    {

        // Set initial conditions from the quadrature point coordinates. The
        // coordinates do not move when the mesh is partitioned, so serial and
        // parallel runs start from the same field and must finish with the
        // same one; a value taken from the storage index instead would differ
        // between rank counts, and its magnitude would grow with the element
        // count, which is what forced the loose tolerances this test used to
        // carry.
        for (unsigned blk = 0, eid = 0; blk < this->fixt_in->GetBlocks().size();
             ++blk)
        {
            auto &block = this->fixt_in->GetBlocks()[blk];
            auto inptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

            const auto cdim = this->fixt_explist->GetCoordim(0);

            for (unsigned n = 0; n < this->fixt_in->GetNumComponents() *
                                         this->fixt_in->GetNumHomoModes();
                 ++n)
            {
                for (size_t el = 0, cnt = 0; el < block.GetNumElements(); ++el)
                {
                    auto coords =
                        this->fixt_explist->GetExp(eid + el)->GetCoords();

                    for (unsigned phys = 0; phys < block.GetNumData();
                         ++phys, ++cnt)
                    {
                        TData v = 1.0 + TData(n);
                        for (unsigned d = 0; d < cdim; ++d)
                        {
                            v += TData(d + 1) * coords[d][phys];
                        }
                        inptr[cnt] = v;
                    }
                }
                inptr += block.CompSize();
            }

            eid += block.GetNumElements();
        }

        size_t ndim  = this->fixt_explist->GetCoordim(0);
        size_t nphys = this->fixt_explist->GetTotPoints();

        auto blocks_in =
            GetBlockAttributes<TData, FieldState::Phys>(this->fixt_explist);
        auto f_advectVel = LibUtilities::Field<TData, FieldState::Phys>(
            "f_advectVel", blocks_in, ndim, 1);
        fixt_advectVel = new LibUtilities::Field<TData, FieldState::Phys>(
            std::move(f_advectVel));

        for (unsigned blk = 0; blk < fixt_advectVel->GetBlocks().size(); ++blk)
        {

            auto &block = fixt_advectVel->GetBlocks()[blk];
            auto inptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            for (unsigned nc = 0; nc < block.GetNumComponents(); ++nc)
            {
                for (size_t el = 0, cnt = 0; el < block.GetNumElements(); ++el)
                {
                    for (unsigned phys = 0; phys < block.GetNumData();
                         ++phys, ++cnt)
                    {
                        // inptr[cnt] = phys;
                        inptr[cnt] = 1 + nc;
                    }
                }
                inptr += block.CompSize();
            }
        }

        // offset advection velocities
        Array<OneD, TData> tmp = fixt_advectVel->ToArray();
        m_advectVel            = Array<OneD, Array<OneD, TData>>(ndim);
        for (int i = 0; i < ndim; ++i)
        {
            m_advectVel[i] = tmp + i * nphys;
        }

        auto blocks_trace = GetBlockAttributes<TData, FieldState::Phys>(
            this->fixt_explist->GetTrace());

        auto f_traceAdvVel = LibUtilities::Field<TData, FieldState::Phys>(
            "f_traceAdvVel", blocks_trace, ndim, 1);
        fixt_traceAdvVel = new LibUtilities::Field<TData, FieldState::Phys>(
            std::move(f_traceAdvVel));

        for (unsigned blk = 0; blk < fixt_traceAdvVel->GetBlocks().size();
             ++blk)
        {

            auto &block = fixt_traceAdvVel->GetBlocks()[blk];
            auto inptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            for (unsigned nc = 0; nc < block.GetNumComponents(); ++nc)
            {
                for (size_t el = 0, cnt = 0; el < block.GetNumElements(); ++el)
                {
                    for (unsigned phys = 0; phys < block.GetNumData();
                         ++phys, ++cnt)
                    {
                        // inptr[cnt] = phys;
                        inptr[cnt] = 1 + nc;
                    }
                }
                inptr += block.CompSize();
            }
        }

        m_traceAdvVel = fixt_traceAdvVel->ToArray();

        // Compute expected solution.
        ExpectedSolution();
    }

    void RunTestCase()
    {
        this->fixt_explist->SetDataWarehouse();
        this->fixt_explist->GetTrace()->SetDataWarehouse();

        std::string riemName;
        this->session->LoadSolverInfo("UpwindType", riemName, "Upwind");

        auto AdvectOp = AdvectionWeakDGOp<TData>::Create(
            this->fixt_explist, this->session->GetVariables());
        // add prefix to name to match operator definition
        riemName       = "ScalarTraceFlux" + riemName;
        auto TraceFlux = SolverCore::TraceFluxOp<TData>::Create(
            this->fixt_explist, this->session->GetVariables(), riemName);

        // Set up trace advection velocity on trace for upwind
        // solver
        size_t ncoordim = this->fixt_explist->GetCoordim(0);
        LibUtilities::Field<double, FieldState::Phys> traceAdvVel(
            "traceAdvVel",
            MultiRegions::GetLocTraceBlockAttributes<double, FieldState::Phys>(
                this->fixt_explist),
            ncoordim, 1);

        // extract advection trace
        auto physTraceExtractOp = PhysTraceExtractOp<double>::Create(
            this->fixt_explist, this->fixt_advectVel->GetComponentNames());
        physTraceExtractOp->Apply(*this->fixt_advectVel, traceAdvVel);
        // set up average value along trace
        TraceFlux->SetTraceAdvVel(traceAdvVel);

        // finally append TraceFlux to AdvectOp
        AdvectOp->SetTraceFlux(TraceFlux);

        auto VolumeFluxOp = LinearAdvVolumeFluxOp<TData>::Create(
            this->fixt_explist, this->session->GetVariables());
        VolumeFluxOp->SetAdvVel(*this->fixt_advectVel);
        AdvectOp->SetVolumeFluxOp(VolumeFluxOp);

        AdvectOp->Apply(*this->fixt_in, *this->fixt_out);
    }

    void ExpectedSolution()
    {
        unsigned numComp = this->fixt_in->GetNumComponents();
        size_t nphys     = this->fixt_explist->GetTotPoints();

        std::string advName;
        std::string riemName;
        this->session->LoadSolverInfo("AdvectionType", advName, "WeakDG");
        auto advObject =
            SolverUtils::GetAdvectionFactory().CreateInstance(advName, advName);
        advObject->SetFluxVector(&TestAdvectionWeakDG::GetFluxVector, this);

        this->session->LoadSolverInfo("UpwindType", riemName, "Upwind");
        auto riemannSolver =
            SolverUtils::GetRiemannSolverFactory().CreateInstance(
                riemName, this->session);
        riemannSolver->SetScalar("Vn", &TestAdvectionWeakDG::GetNormalVel,
                                 this);

        Nektar::Array<Nektar::OneD, Nektar::MultiRegions::ExpListSharedPtr> exp(
            numComp);

        std::vector<std::string> variables = this->session->GetVariables();
        auto graph = SpatialDomains::MeshGraphIO::Read(this->session);
        ASSERTL0(variables.size() >= numComp,
                 "numComp is larger than the number of variales");

        for (unsigned i = 0; i < numComp; ++i)
        {
            // in case of orthogonal expansions reset expansion info for legacy
            // code to work
            if (!this->fixt_explist->GetExp(0)->IsBoundaryInteriorExpansion())
            {
                graph->ResetExpansionInfoToModified(variables[i]);
            }
            exp[i] =
                MemoryManager<MultiRegions::DisContField>::AllocateSharedPtr(
                    this->session, graph, variables[i], true, true,
                    Collections::eNoCollection);
        }

        advObject->SetRiemannSolver(riemannSolver);
        advObject->InitObject(this->session, exp);

        Array<OneD, TData> in = this->fixt_in->ToArray();
        Array<OneD, TData> out(nphys * numComp, 0.0);

        Array<OneD, Array<OneD, TData>> inarray(numComp);
        Array<OneD, Array<OneD, TData>> outarray(numComp);

        for (unsigned i = 0; i < numComp; ++i)
        {
            inarray[i] = Array<OneD, TData>(nphys, 0.0);
            for (size_t j = 0; j < nphys; j++)
            {
                inarray[i][j] = in[i * nphys + j];
            }
            outarray[i] = Array<OneD, TData>(nphys, 0.0);
        }

        advObject->Advect(numComp, exp, m_advectVel, inarray, outarray, 0);

        for (unsigned i = 0; i < numComp; ++i)
        {
            for (size_t j = 0; j < nphys; ++j)
            {
                out[i * nphys + j] = outarray[i][j];
            }
        }

        this->fixt_expected->template CopyArray<NektarSpaces::HostSpace>(out);
    }

    const Array<OneD, TData> &GetNormalVel()
    {
        const unsigned coordDim = this->fixt_explist->GetCoordim(0);
        const size_t nTracePts = this->fixt_explist->GetTrace()->GetTotPoints();

        if (m_normalVel.size() == 0)
        {
            m_normalVel = Array<OneD, TData>(nTracePts, 0.0);
            // Gather normals
            Array<OneD, Array<OneD, TData>> traceNormals(coordDim);
            for (unsigned i = 0; i < coordDim; ++i)
            {
                traceNormals[i] = Array<OneD, TData>(nTracePts, 0.0);
            }

            this->fixt_explist->GetTrace()->GetNormals(traceNormals);

            // v = (1,1,...), so Vn = n · v = sum_k n_k
            for (unsigned i = 0; i < coordDim; ++i)
            {
                for (size_t j = 0; j < nTracePts; ++j)
                {
                    m_normalVel[j] +=
                        traceNormals[i][j] * m_traceAdvVel[i * nTracePts + j];
                }
            }
        }

        return m_normalVel;
    }

    void GetFluxVector(const Array<OneD, Array<OneD, TData>> &physfield,
                       Array<OneD, Array<OneD, Array<OneD, TData>>> &flux)
    {
        ASSERTL1(flux[0].size() == m_advectVel.size(),
                 "Dimension of flux array and velocity array do not match");

        const int nq = this->fixt_explist->GetNpoints();

        for (int i = 0; i < flux.size(); ++i)
        {
            for (int j = 0; j < flux[0].size(); ++j)
            {
                for (int k = 0; k < nq; ++k)
                {
                    flux[i][j][k] = physfield[i][k] * m_advectVel[j][k];
                }
            }
        }
    }

public:
    LibUtilities::Field<TData, FieldState::Phys> *fixt_advectVel   = nullptr;
    LibUtilities::Field<TData, FieldState::Phys> *fixt_traceAdvVel = nullptr;
    Array<OneD, TData> m_traceAdvVel;
    Array<OneD, TData> m_normalVel;
    Array<OneD, Array<OneD, TData>> m_advectVel;
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TESTFLOAT(type, filename)                                              \
    class type##float : public TestAdvectionWeakDG<float>                      \
    {                                                                          \
    public:                                                                    \
        type##float()                                                          \
        {                                                                      \
            meshName = filename;                                               \
        }                                                                      \
    };
#else
#define TESTFLOAT(type, filename)
#endif
#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
#define TESTDOUBLE(type, filename)                                             \
    class type : public TestAdvectionWeakDG<double>                            \
    {                                                                          \
    public:                                                                    \
        type()                                                                 \
        {                                                                      \
            meshName = filename;                                               \
        }                                                                      \
    };
#else
#define TESTDOUBLE(type, filename)
#endif
#define TEST(type, filename)                                                   \
    TESTFLOAT(type, filename)                                                  \
    TESTDOUBLE(type, filename)
// clang-format on

TEST(Seg, "run/Seg_Dir.xml")

TEST(SegOrtho, "run/Seg_Ortho_Dir.xml")

TEST(Tri, "run/Tri_Dir.xml")

TEST(TriOrtho, "run/Tri_Ortho_Dir.xml")

TEST(Quad, "run/Quad_Dir.xml")

TEST(Quad_Per, "run/Quad_Per.xml")

TEST(QuadOrtho_Per, "run/Quad_Ortho_Per.xml")

TEST(TriQuad_VarP, "run/TriQuad_varP_Dir.xml")

TEST(Prism_VarP, "run/Prism_varP_Dir.xml")

TEST(Hex, "run/Hex_Dir.xml")

TEST(HexOrtho, "run/Hex_Ortho_Dir.xml")

TEST(Hex_Per, "run/Hex_Per.xml")

TEST(Prism, "run/Prism_Dir.xml")

TEST(PrismOrtho, "run/Prism_Ortho_Dir.xml")

TEST(Pyr, "run/Pyr_Dir.xml")

TEST(PyrOrtho, "run/Pyr_Ortho_Dir.xml")

TEST(Tet, "run/Tet_Dir.xml")

TEST(TetOrtho, "run/Tet_Ortho_Dir.xml")

TEST(CubeAllElements, "run/cube_all_elements_dir.xml")
