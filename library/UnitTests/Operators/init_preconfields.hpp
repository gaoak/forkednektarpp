///////////////////////////////////////////////////////////////////////////////
//
// File: init_preconfields.hpp
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

#include "Operators/AssmbScatr/AssmbScatrZeroDirOp.hpp"
#include "Operators/ElmtOps/Helmholtz/HelmholtzOp.hpp"
#include "Operators/PreconOps/PreconOp.hpp"

#include <MultiRegions/GlobalLinSys.h>
#include <MultiRegions/Preconditioner.h>

using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar::MultiRegions;
using namespace Nektar;

template <typename TData>
class PreconField
    : public InitFields<TData, FieldState::Coeff, FieldState::Coeff,
                        MultiRegions::ContField>
{
public:
    PreconField()
        : InitFields<TData, FieldState::Coeff, FieldState::Coeff,
                     MultiRegions::ContField>()
    {
    }

    void SetTestCase(const std::string &method)
    {
        // Set initial conditions.
        for (unsigned int blk = 0; blk < this->fixt_in->GetBlocks().size();
             ++blk)
        {
            auto &block = this->fixt_in->GetBlocks()[blk];
            auto inptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            for (size_t el = 0, cnt = 0; el < block.GetNumElements(); ++el)
            {
                for (unsigned int coeff = 0; coeff < block.GetNumData();
                     ++coeff, ++cnt)
                {
                    inptr[cnt] = 1.0;
                }
            }
        }

        // Get lambda from this->session or default to 10.0
        m_lambda = this->session->DefinesParameter("Lambda")
                       ? this->session->GetParameter("Lambda")
                       : 10.0;

        // Set up diffusion coefficient.
        const auto coordDim      = this->fixt_explist->GetCoordim(0);
        const auto diffCoeffSize = coordDim * (coordDim + 1) / 2;
        m_diffCoeff.resize(diffCoeffSize);

        if (coordDim == 1)
        {
            m_diffCoeff[0] = 1.0; // D00
        }
        else if (coordDim == 2)
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

        // Compute expected solution.
        ExpectedSolution(method);
    }

    void RunTestCase(const std::string &method)
    {
        std::string execStr = Operator<TData>::GetOpExecSpace(this->session);

        auto assmb = AssmbScatrZeroDirOp<TData>::Create(
            this->fixt_explist, this->session->GetVariables());
        auto op     = HelmholtzOp<TData>::Create(this->fixt_explist,
                                                 this->session->GetVariables());
        auto precon = PreconOp<TData>::Create(
            this->fixt_explist, this->session->GetVariables(), method);
        op->SetLambda(m_lambda);
        op->SetDiffCoeff(m_diffCoeff);
        precon->Configure(op);
        assmb->Apply(*this->fixt_in, *this->fixt_out);

        // reshape this->fixt_out
        if (execStr == "AVX")
        {
            this->fixt_out->ReshapeStorage(
                NektarSpaces::GetVectorWidth<TData>(execStr), execStr);
        }

        precon->Apply(*this->fixt_out, *this->fixt_out);
    }

    void ExpectedSolution(const std::string &method)
    {
        auto nin    = this->session->GetVariables().size();
        auto ncoeff = this->fixt_explist->GetNcoeffs();

        // Calculate expected result from Nektar++
        Array<OneD, TData> incoeffs = this->fixt_in->ToArray();
        Array<OneD, TData> outcoeffs(nin * ncoeff, 0.0);

        StdRegions::ConstFactorMap factors;
        factors[StdRegions::eFactorLambda] = m_lambda;

        auto map = std::dynamic_pointer_cast<MultiRegions::ContField>(
                       this->fixt_explist)
                       ->GetLocalToGlobalMap();
        GlobalLinSysKey key(StdRegions::eHelmholtz, map, factors);
        auto globalSys = GetGlobalLinSysFactory().CreateInstance(
            "IterativeFull", key, this->fixt_explist, map);
        auto precond =
            GetPreconFactory().CreateInstance(method, globalSys, map);
        precond->BuildPreconditioner();
        for (unsigned int n = 0; n < nin; n++)
        {
            Array<OneD, TData> tmp;
            precond->DoPreconditioner(incoeffs + n * ncoeff,
                                      tmp = outcoeffs + n * ncoeff, true);
        }
        this->fixt_expected->template CopyArray<NektarSpaces::HostSpace>(
            outcoeffs);
    }

private:
    TData m_lambda;
    std::vector<TData> m_diffCoeff;
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TESTFLOAT(type, filename)                                              \
    class type##float : public PreconField<float>                              \
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
    class type : public PreconField<double>                                    \
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

TEST(Helmholtz1D_3C, "run/Helmholtz1D_3C.xml")

TEST(Helmholtz2D_Tri_Quad, "run/Helmholtz2D_varP.xml")

TEST(Helmholtz2D_AllBCs, "run/Helmholtz2D_P7_AllBCs.xml")

TEST(Helmholtz2D_3C, "run/Helmholtz2D_3C.xml")

TEST(Helmholtz3D_Hex, "run/Helmholtz3D_Hex_Heterogeneous.xml")

TEST(Helmholtz3D_Hex_3C, "run/Helmholtz3D_Hex_3C.xml")

TEST(Helmholtz3D_Prism, "run/Helmholtz3D_Prism_VarP.xml")

TEST(Helmholtz3D_Pyr, "run/Helmholtz3D_Pyr_VarP.xml")

TEST(Helmholtz3D_Tet, "run/Helmholtz3D_Tet_VarP.xml")
