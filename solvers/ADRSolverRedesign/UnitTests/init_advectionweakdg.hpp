///////////////////////////////////////////////////////////////////////////////
//
// File: init_advectionweakdg.hpp
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
#include "SolverCore/RiemannSolvers/RiemannSolverOp.hpp"
#include <ADRSolver/EquationSystems/UnsteadyAdvection.h>
#include <SolverUtils/RiemannSolvers/RiemannSolver.h>
#include <SolverUtils/RiemannSolvers/UpwindSolver.h>

#include "UnitTests/Operators/init_fields.hpp"

using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::MultiRegions;
using namespace Nektar::Operators;
using namespace Nektar::SolverCore;

template <typename TData>
class AdvectionWeakDG
    : public InitFields<TData, FieldState::Phys, FieldState::Phys,
                        MultiRegions::DisContField>
{
public:
    AdvectionWeakDG()
        : InitFields<TData, FieldState::Phys, FieldState::Phys,
                     MultiRegions::DisContField>()
    {
    }

    void SetTestCase()
    {

        // Set initial conditions.
        for (unsigned int blk = 0; blk < this->fixt_in->GetBlocks().size();
             ++blk)
        {
            auto &block = this->fixt_in->GetBlocks()[blk];
            auto inptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            for (unsigned int n = 0; n < this->fixt_in->GetNumComponents() *
                                             this->fixt_in->GetNumHomoModes();
                 ++n)
            {
                for (size_t el = 0, cnt = 0; el < block.GetNumElements(); ++el)
                {
                    for (unsigned int phys = 0; phys < block.GetNumData();
                         ++phys, ++cnt)
                    {
                        inptr[cnt] = phys + el;
                    }
                }
                inptr += block.CompSize();
            }
        }

        size_t ndim  = this->fixt_explist->GetCoordim(0);
        size_t nphys = this->fixt_explist->GetTotPoints();

        auto blocks_in =
            GetBlockAttributes<TData, FieldState::Phys>(this->fixt_explist);
        auto f_advectVel = LibUtilities::Field<TData, FieldState::Phys>(
            "f_advectVel", blocks_in, ndim, 1);
        fixt_advectVel = new LibUtilities::Field<TData, FieldState::Phys>(
            std::move(f_advectVel));

        for (unsigned int blk = 0; blk < fixt_advectVel->GetBlocks().size();
             ++blk)
        {

            auto &block = fixt_advectVel->GetBlocks()[blk];
            auto inptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            for (unsigned int nc = 0; nc < block.GetNumComponents(); ++nc)
            {
                for (size_t el = 0, cnt = 0; el < block.GetNumElements(); ++el)
                {
                    for (unsigned int phys = 0; phys < block.GetNumData();
                         ++phys, ++cnt)
                    {
                        inptr[cnt] = phys;
                    }
                }
                inptr += block.CompSize();
            }
        }

        Array<OneD, TData> tmp = fixt_advectVel->ToArray();
        m_advectVel            = Array<OneD, Array<OneD, TData>>(ndim);
        for (int i = 0; i < ndim; ++i)
        {
            m_advectVel[i] = Array<OneD, TData>(nphys, 0.0);
            m_advectVel[i] = tmp + i * nphys;
        }

        auto blocks_trace = GetBlockAttributes<TData, FieldState::Phys>(
            this->fixt_explist->GetTrace());
        auto f_traceAdvVel = LibUtilities::Field<TData, FieldState::Phys>(
            "f_traceAdvVel", blocks_trace, ndim, 1);
        fixt_traceAdvVel = new LibUtilities::Field<TData, FieldState::Phys>(
            std::move(f_traceAdvVel));

        for (unsigned int blk = 0; blk < fixt_traceAdvVel->GetBlocks().size();
             ++blk)
        {

            auto &block = fixt_traceAdvVel->GetBlocks()[blk];
            auto inptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            for (unsigned int nc = 0; nc < block.GetNumComponents(); ++nc)
            {
                for (size_t el = 0, cnt = 0; el < block.GetNumElements(); ++el)
                {
                    for (unsigned int phys = 0; phys < block.GetNumData();
                         ++phys, ++cnt)
                    {
                        inptr[cnt] = phys;
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
        auto RiemannOp = RiemannSolverOp<TData>::Create(
            this->fixt_explist, this->session->GetVariables(), riemName);
        RiemannOp->SetTraceAdvVel(*this->fixt_traceAdvVel);

        auto VolumeFluxOp = LinearAdvVolumeFluxOp<TData>::Create(
            this->fixt_explist, this->session->GetVariables());
        VolumeFluxOp->SetAdvectVel(*this->fixt_advectVel);

        auto AdvectOp = AdvectionWeakDGOp<TData>::Create(
            this->fixt_explist, this->session->GetVariables());
        AdvectOp->SetRiemannSolver(RiemannOp);
        AdvectOp->SetVolumeFluxOp(VolumeFluxOp);
        AdvectOp->Apply(*this->fixt_in, *this->fixt_out);
    }

    void ExpectedSolution()
    {
        unsigned int numComp = this->fixt_in->GetNumComponents();
        size_t nphys         = this->fixt_explist->GetTotPoints();

        std::string advName;
        std::string riemName;
        this->session->LoadSolverInfo("AdvectionType", advName, "WeakDG");
        auto advObject =
            SolverUtils::GetAdvectionFactory().CreateInstance(advName, advName);
        advObject->SetFluxVector(&AdvectionWeakDG::GetFluxVector, this);
        this->session->LoadSolverInfo("UpwindType", riemName, "Upwind");
        auto riemannSolver =
            SolverUtils::GetRiemannSolverFactory().CreateInstance(
                riemName, this->session);
        riemannSolver->SetScalar("Vn", &AdvectionWeakDG::GetNormalVel, this);

        Nektar::Array<Nektar::OneD, Nektar::MultiRegions::ExpListSharedPtr> exp(
            numComp);
        for (unsigned int i = 0; i < numComp; ++i)
        {
            exp[i] = this->fixt_explist;
        }

        advObject->SetRiemannSolver(riemannSolver);
        advObject->InitObject(this->session, exp);

        Array<OneD, TData> in = this->fixt_in->ToArray();
        Array<OneD, TData> out(nphys * numComp, 0.0);

        Array<OneD, Array<OneD, TData>> inarray(numComp);
        Array<OneD, Array<OneD, TData>> outarray(numComp);

        for (unsigned int i = 0; i < numComp; ++i)
        {
            inarray[i] = Array<OneD, TData>(nphys, 0.0);
            for (size_t j = 0; j < nphys; j++)
            {
                inarray[i][j] = in[i * nphys + j];
            }
            outarray[i] = Array<OneD, TData>(nphys, 0.0);
        }

        advObject->Advect(numComp, exp, m_advectVel, inarray, outarray, 0);

        for (unsigned int i = 0; i < numComp; ++i)
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
        const unsigned int coordDim = this->fixt_explist->GetCoordim(0);
        const size_t nTracePts = this->fixt_explist->GetTrace()->GetTotPoints();

        if (m_normalVel.size() == 0)
        {
            m_normalVel = Array<OneD, TData>(nTracePts, 0.0);
            // Gather normals
            Array<OneD, Array<OneD, TData>> traceNormals(coordDim);
            for (unsigned int i = 0; i < coordDim; ++i)
            {
                traceNormals[i] = Array<OneD, TData>(nTracePts, 0.0);
            }

            this->fixt_explist->GetTrace()->GetNormals(traceNormals);

            // v = (1,1,...), so Vn = n · v = sum_k n_k
            for (unsigned int i = 0; i < coordDim; ++i)
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
    class type##float : public AdvectionWeakDG<float>                          \
    {                                                                          \
    public:                                                                    \
        type##float()                                                          \
        {                                                                      \
            std::string dir = "../../../library/UnitTests/Operators/";         \
            meshName = dir + filename;                                         \
        }                                                                      \
    };
#else
#define TESTFLOAT(type, filename)
#endif
#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
#define TESTDOUBLE(type, filename)                                             \
    class type : public AdvectionWeakDG<double>                                \
    {                                                                          \
    public:                                                                    \
        type()                                                                 \
        {                                                                      \
            std::string dir = "../../../library/UnitTests/Operators/";         \
            meshName = dir + filename;                                         \
        }                                                                      \
    };
#else
#define TESTDOUBLE(type, filename)
#endif
#define TEST(type, filename)                                                   \
    TESTFLOAT(type, filename)                                                  \
    TESTDOUBLE(type, filename)
// clang-format on

TEST(Seg, "run/segment.xml")

TEST(SegSEM, "run/line_sem.xml")

TEST(Quad, "run/square.xml")

TEST(QuadVarP, "run/square_varp.xml")

TEST(QuadSEM, "run/square_sem.xml")

TEST(Tri, "run/tri.xml")

TEST(TriVarP, "run/tri_varp.xml")

TEST(SquareAllElements, "run/square_all_elements.xml")

TEST(Hex, "run/hex.xml")

TEST(HexVarP, "run/hex_varp.xml")

TEST(HexSEM, "run/hex_sem.xml")

TEST(Prism, "run/prism.xml")

TEST(PrismVarP, "run/prism_varp.xml")

TEST(Pyr, "run/pyr.xml")

TEST(PyrVarP, "run/pyr_varp.xml")

TEST(Tet, "run/tet.xml")

TEST(TetVarP, "run/tet_varp.xml")

TEST(CubePrismHex, "run/cube_prismhex.xml")

TEST(CubeAllElements, "run/cube_all_elements.xml")
