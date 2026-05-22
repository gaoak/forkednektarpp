///////////////////////////////////////////////////////////////////////////////
//
// File: init_linearadrsolve_fields.hpp
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

#include "init_fields.hpp"

#include "Operators/GlobalLinSysOps/LinearSolvers/LinearSolverOp.hpp"
#include "Operators/GlobalLinSysOps/LinearSystems/LinearADRSolve/LinearADRSolveOp.hpp"
#include "Operators/PreconOps/PreconOp.hpp"

#include <memory>

using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

class ContFieldMatrixAccessor : public MultiRegions::ContField
{
public:
    using MultiRegions::ContField::ContField;
    using MultiRegions::ExpList::GenGlobalMatrixFull;
};

template <typename TData>
class LinearADRSolveField
    : public InitFields<TData, FieldState::Phys, FieldState::Coeff,
                        MultiRegions::ContField>
{
public:
    LinearADRSolveField()
        : InitFields<TData, FieldState::Phys, FieldState::Coeff,
                     MultiRegions::ContField>()
    {
    }

    // Custom configure method to avoid non-symmetric system warning in
    // GlobalLinsSysIterative
    void Configure(void)
    {
        this->SetSession();
        auto graph = SpatialDomains::MeshGraphIO::Read(this->session);
        ConfigureExpectedSolverInfo();
        this->fixt_explist =
            MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
                this->session, graph, "u", true, false,
                Collections::eNoCollection);
        this->fixt_explist->SetDataWarehouse();
        this->SetFixture(1);
    }

    void SetTestCase()
    {
        // Set initial conditions.
        const unsigned int numComp = this->fixt_in->GetNumComponents() *
                                     this->fixt_in->GetNumHomoModes();
        const size_t nphys = this->fixt_explist->GetTotPoints();
        m_coordDim         = this->fixt_explist->GetCoordim(0);

        Array<OneD, TData> x(nphys);
        Array<OneD, TData> y(nphys);
        Array<OneD, TData> z(nphys);
        Array<OneD, TData> fce(numComp * nphys, 0.0);
        this->fixt_explist->GetCoords(x, y, z);
        this->fixt_in->template Initialize<NektarSpaces::HostSpace>(0.0);

        if (this->session->DefinesFunction("Forcing"))
        {
            for (unsigned int n = 0; n < numComp; ++n)
            {
                auto func = this->session->GetFunction(
                    "Forcing", n % this->session->GetVariables().size());
                Array<OneD, TData> fceVar = fce + n * nphys;
                func->Evaluate(x, y, z, fceVar);
            }
        }

        for (unsigned int blk = 0; blk < this->fixt_in->GetBlocks().size();
             ++blk)
        {
            auto &block = this->fixt_in->GetBlocks()[blk];
            auto inptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            for (unsigned int n = 0; n < numComp; ++n)
            {
                auto xptr   = x.data();
                auto yptr   = y.data();
                auto zptr   = z.data();
                auto fceptr = fce.data() + n * nphys;

                for (size_t el = 0, cnt = 0; el < block.GetNumElements(); ++el)
                {
                    for (unsigned int phys = 0; phys < block.GetNumData();
                         ++phys, ++cnt)
                    {
                        if (this->session->DefinesFunction("Forcing"))
                        {
                            inptr[cnt] = *(fceptr++);
                        }
                        else
                        {
                            inptr[cnt] = 1.0 + n;
                            if (m_coordDim == 1)
                            {
                                for (unsigned int p = 1; p < 4; ++p)
                                {
                                    inptr[cnt] +=
                                        (n + 1) * p * std::pow(*xptr, p);
                                }
                                xptr++;
                            }
                            else if (m_coordDim == 2)
                            {
                                for (unsigned int p = 1; p < 4; ++p)
                                {
                                    inptr[cnt] += (n + 1) * p *
                                                  std::pow(*xptr, p) *
                                                  std::pow(*yptr, p);
                                }
                                xptr++;
                                yptr++;
                            }
                            else
                            {
                                for (unsigned int p = 1; p < 4; ++p)
                                {
                                    inptr[cnt] +=
                                        (n + 1) * p * std::pow(*xptr, p) *
                                        std::pow(*yptr, p) * std::pow(*zptr, p);
                                }
                                xptr++;
                                yptr++;
                                zptr++;
                            }
                        }
                    }
                }

                inptr += block.CompSize();
            }
        }

        this->fixt_out->template Initialize<NektarSpaces::HostSpace>(0.0);

        // Get lambda from this->session or default to 10.0
        m_lambda = this->session->DefinesParameter("Lambda")
                       ? this->session->GetParameter("Lambda")
                       : 10.0;

        // Set up diffusion coefficient.
        const auto diffCoeffSize = m_coordDim * (m_coordDim + 1) / 2;
        m_diffCoeff.resize(diffCoeffSize);

        // Set up (isotropic) diffusion coefficient.
        if (m_coordDim == 1)
        {
            m_diffCoeff[0] = 1.0; // D00
        }
        else if (m_coordDim == 2)
        {
            m_diffCoeff[0] = 1.0; // D00
            m_diffCoeff[2] = 1.0; // D11
        }
        else
        {
            m_diffCoeff[0] = 1.0; // D00
            m_diffCoeff[2] = 1.0; // D11
            m_diffCoeff[5] = 1.0; // D22
        }

        // Set advection velocity
        m_vel = Array<OneD, TData>(nphys * m_coordDim, 1.0);
        Array<OneD, TData> tmp;
        for (unsigned int d = 1; d < m_coordDim; ++d)
        {
            Vmath::Fill(nphys, d + 1.0, tmp = m_vel + d * nphys, 1);
        }

        // Compute expected solution.
        ExpectedSolution();
    }

    void RunTestCase(const std::string &method)
    {
        auto advelblockAttr =
            GetBlockAttributes<TData, FieldState::Phys>(this->fixt_explist);
        auto vel = Field<TData, FieldState::Phys>("vel", advelblockAttr,
                                                  m_coordDim, 1);
        vel.template CopyArray<NektarSpaces::HostSpace>(m_vel);

        std::string execStr(
            boost::unit_test::framework::master_test_suite().argv[1]);
        std::string implStr(
            boost::unit_test::framework::master_test_suite().argv[2]);

        // reshape this->fixt_in
        if (implStr == "SumFac" || execStr == "AVX")
        {
            for (auto &block : this->fixt_in->GetBlocks())
            {
                auto ptr =
                    block.template GetPtr<NektarSpaces::HostSpace, ReadWrite>();
                for (unsigned int nc = 0; nc < block.GetNumComponents(); ++nc)
                {
                    // reshuffle data into simd_t width for AVX check
                    ReshapeStorage<NektarSpaces::Serial>(
                        NektarSpaces::GetVectorWidth<TData>(execStr),
                        block.GetInterleaveWidth(),
                        block.GetNumElementsWithPadding(), block.GetNumData(),
                        ptr);
                    ptr += block.CompSize();
                }

                block.template SetInterleaveWidth<TData>(
                    NektarSpaces::GetVectorWidth<TData>(execStr));
            }
            this->fixt_out->SetInterleaveWidth(
                NektarSpaces::GetVectorWidth<TData>(execStr));
        }

        auto op = LinearADRSolveOp<TData>::Create(
            this->fixt_explist, this->session->GetVariables());
        auto precon = PreconOp<TData>::Create(
            this->fixt_explist, this->session->GetVariables(), "Diagonal");
        auto linsolve = LinearSolverOp<TData>::Create(
            this->fixt_explist, this->session->GetVariables(), method);
        op->SetLambda(m_lambda);
        op->SetDiffCoeff(m_diffCoeff);
        op->SetAdvVel(vel);
        op->SetLinearSolver(linsolve);
        op->SetPrecon(precon);
        op->UpdatePrecon();
        op->Apply(*this->fixt_in, *this->fixt_out);
    }

    void ExpectedSolution()
    {
        // Calculate expected result from Nektar++.
        const unsigned int numComp = this->fixt_in->GetNumComponents() *
                                     this->fixt_in->GetNumHomoModes();
        const size_t nphys    = this->fixt_explist->GetTotPoints();
        const size_t ncoeffs  = this->fixt_explist->GetNcoeffs();
        const auto &variables = this->session->GetVariables();

        // Setup input/output arrays
        Array<OneD, TData> inphys = this->fixt_in->ToArray();
        Array<OneD, TData> outcoeffs(numComp * ncoeffs, 0.0);

        // Set lambda as constant coefficient
        StdRegions::ConstFactorMap factors;
        factors[StdRegions::eFactorLambda] = -m_lambda;

        // Set up diffusion coefficient.
        if (m_coordDim == 2)
        {
            factors[StdRegions::eFactorCoeffD00] = m_diffCoeff[0];
            factors[StdRegions::eFactorCoeffD01] = m_diffCoeff[1];
            factors[StdRegions::eFactorCoeffD11] = m_diffCoeff[2];
        }
        else if (m_coordDim == 3)
        {
            factors[StdRegions::eFactorCoeffD00] = m_diffCoeff[0];
            factors[StdRegions::eFactorCoeffD01] = m_diffCoeff[1];
            factors[StdRegions::eFactorCoeffD11] = m_diffCoeff[2];
            factors[StdRegions::eFactorCoeffD02] = m_diffCoeff[3];
            factors[StdRegions::eFactorCoeffD12] = m_diffCoeff[4];
            factors[StdRegions::eFactorCoeffD22] = m_diffCoeff[5];
        }

        // Set advection velocities via varcoeffs
        StdRegions::VarCoeffMap varcoeffs;
        std::vector<StdRegions::VarCoeffType> velCoeffType = {
            StdRegions::eVarCoeffVelX, StdRegions::eVarCoeffVelY,
            StdRegions::eVarCoeffVelZ};
        for (unsigned int d = 0; d < m_coordDim; ++d)
        {
            Array<OneD, TData> tmp;
            Array<OneD, TData> tmpVel(nphys);
            Vmath::Vcopy(nphys, m_vel + d * nphys, 1, tmpVel, 1);
            varcoeffs[velCoeffType[d]] = tmpVel;
        }

        auto graph = SpatialDomains::MeshGraphIO::Read(this->session);
        ConfigureExpectedSolverInfo();

        // Solve each component separately.
        for (unsigned int n = 0; n < numComp; ++n)
        {
            auto expListVar =
                MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
                    this->session, graph, variables[n % variables.size()], true,
                    false, Collections::eNoCollection);
            Array<OneD, TData> inphysVar    = inphys + n * nphys;
            Array<OneD, TData> outcoeffsVar = outcoeffs + n * ncoeffs;
            expListVar->LinearAdvectionDiffusionReactionSolve(
                inphysVar, outcoeffsVar, factors, varcoeffs);
        }

        // Copy solution back
        this->fixt_expected->template CopyArray<NektarSpaces::HostSpace>(
            outcoeffs);
    }

    StdRegions::ConstFactorMap BuildFactors() const
    {
        StdRegions::ConstFactorMap factors;
        factors[StdRegions::eFactorLambda] = -m_lambda;

        if (m_coordDim == 2)
        {
            factors[StdRegions::eFactorCoeffD00] = m_diffCoeff[0];
            factors[StdRegions::eFactorCoeffD01] = m_diffCoeff[1];
            factors[StdRegions::eFactorCoeffD11] = m_diffCoeff[2];
        }
        else if (m_coordDim == 3)
        {
            factors[StdRegions::eFactorCoeffD00] = m_diffCoeff[0];
            factors[StdRegions::eFactorCoeffD01] = m_diffCoeff[1];
            factors[StdRegions::eFactorCoeffD11] = m_diffCoeff[2];
            factors[StdRegions::eFactorCoeffD02] = m_diffCoeff[3];
            factors[StdRegions::eFactorCoeffD12] = m_diffCoeff[4];
            factors[StdRegions::eFactorCoeffD22] = m_diffCoeff[5];
        }

        return factors;
    }

    StdRegions::VarCoeffMap BuildVarCoeffs(const Array<OneD, TData> &vel) const
    {
        const size_t nphys = this->fixt_explist->GetTotPoints();
        StdRegions::VarCoeffMap varcoeffs;
        std::vector<StdRegions::VarCoeffType> velCoeffType = {
            StdRegions::eVarCoeffVelX, StdRegions::eVarCoeffVelY,
            StdRegions::eVarCoeffVelZ};

        for (unsigned int d = 0; d < m_coordDim; ++d)
        {
            Array<OneD, TData> tmpVel(nphys);
            Vmath::Vcopy(nphys, vel + d * nphys, 1, tmpVel, 1);
            varcoeffs[velCoeffType[d]] = tmpVel;
        }

        return varcoeffs;
    }

    std::shared_ptr<ContFieldMatrixAccessor> BuildExpectedExpList(
        const std::string &variable)
    {
        auto graph = SpatialDomains::MeshGraphIO::Read(this->session);
        ConfigureExpectedSolverInfo();

        return MemoryManager<ContFieldMatrixAccessor>::AllocateSharedPtr(
            this->session, graph, variable, true, false,
            Collections::eNoCollection);
    }

    DNekMatSharedPtr BuildGlobalADRMatrix(const std::string &variable,
                                          const Array<OneD, TData> &vel)
    {
        auto expListVar      = BuildExpectedExpList(variable);
        const size_t nphys   = expListVar->GetTotPoints();
        const size_t ncoeffs = expListVar->GetNcoeffs();
        Array<OneD, TData> inphys(nphys, 0.0);
        Array<OneD, TData> outcoeffs(ncoeffs, 0.0);

        auto key = expListVar->LinearAdvectionDiffusionReactionSolve(
            inphys, outcoeffs, BuildFactors(), BuildVarCoeffs(vel));

        return expListVar->GenGlobalMatrixFull(
            key, expListVar->GetLocalToGlobalMap());
    }

protected:
    void ConfigureExpectedSolverInfo()
    {
        const auto &variables = this->session->GetVariables();
        std::string iterSolver =
            this->session->DefinesSolverInfo("LinSysIterSolver")
                ? this->session->GetSolverInfo("LinSysIterSolver")
                : "ConjugateGradient";
        std::string gmresSolver =
            iterSolver.find("Loc") != std::string::npos ? "GMRESLoc" : "GMRES";

        if (this->session->DefinesSolverInfo("LinSysIterSolver"))
        {
            this->session->SetSolverInfo("LinSysIterSolver", gmresSolver);
        }
        for (const auto &var : variables)
        {
            this->session->SetGlobalSysSolnInfo(var, "GlobalSysSoln",
                                                "IterativeFull");
            this->session->SetGlobalSysSolnInfo(var, "LinSysIterSolver",
                                                gmresSolver);
        }
    }

    unsigned int m_coordDim;
    TData m_lambda;
    std::vector<TData> m_diffCoeff;
    Array<OneD, TData> m_vel;
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TESTFLOAT(type, filename)                                              \
    class type##float : public LinearADRSolveField<float>                      \
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
    class type : public LinearADRSolveField<double>                            \
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

TEST(Helmholtz1D_Seg, "run/Helmholtz1D_P8.xml")
TEST(Helmholtz1D_Seg_3C, "run/Helmholtz1D_3C.xml")

TEST(Helmholtz2D_Tri_Quad, "run/Helmholtz2D_varP.xml")
TEST(Helmholtz2D_Tri_Quad_3C, "run/Helmholtz2D_3C.xml")

TEST(Helmholtz2D_AllBCs, "run/Helmholtz2D_P7_AllBCs.xml")

TEST(Helmholtz3D_Hex, "run/Helmholtz3D_Hex_Heterogeneous.xml")
TEST(Helmholtz3D_Hex_3C, "run/Helmholtz3D_Hex_3C.xml")
TEST(Helmholtz3D_Hex_3C_Single, "run/Helmholtz3D_Hex_3C_Single.xml")

TEST(Helmholtz3D_Prism, "run/Helmholtz3D_Prism_VarP.xml")

TEST(Helmholtz3D_Pyr, "run/Helmholtz3D_Pyr_VarP.xml")

TEST(Helmholtz3D_Tet, "run/Helmholtz3D_Tet_VarP.xml")
