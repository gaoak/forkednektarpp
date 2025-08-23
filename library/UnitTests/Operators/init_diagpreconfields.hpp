///////////////////////////////////////////////////////////////////////////////
//
// File: init_diagpreconfields.hpp
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

#include "Operators/ElmtOps/Helmholtz/HelmholtzOp.hpp"
#include "Operators/PreconOps/DiagPrecon/DiagPreconOp.hpp"

#include <MultiRegions/GlobalLinSys.h>
#include <MultiRegions/Preconditioner.h>

using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar::MultiRegions;
using namespace Nektar;

class DiagPreconField
    : public InitFields<double, FieldState::Coeff, FieldState::Coeff,
                        MultiRegions::ContField>
{
public:
    DiagPreconField()
        : InitFields<double, FieldState::Coeff, FieldState::Coeff,
                     MultiRegions::ContField>()
    {
    }

    void SetTestCase()
    {

        // Set initial conditions.
        for (unsigned int blk = 0; blk < fixt_in->GetBlocks().size(); ++blk)
        {
            auto &block = fixt_in->GetBlocks()[blk];
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

        // Get lambda from session or default to 10.0
        m_lambda = session->DefinesParameter("Lambda")
                       ? session->GetParameter("Lambda")
                       : 10.0;

        // Compute expected solution.
        ExpectedSolution();
    }

    void RunTestCase()
    {
        auto op     = HelmholtzOp<double>::Create(fixt_explist);
        auto precon = DiagPreconOp<double>::Create(fixt_explist);
        op->SetLambda(m_lambda);
        precon->Configure(op);
        precon->Apply(*fixt_in, *fixt_out);
    }

    void ExpectedSolution()
    {
        // Calculate expected result from Nektar++
        Array<OneD, double> incoeffs = fixt_in->ToArray();
        Array<OneD, double> outcoeffs(fixt_explist->GetNcoeffs(), 0.0);

        StdRegions::ConstFactorMap factors;
        factors[StdRegions::eFactorLambda] = m_lambda;

        auto map =
            std::dynamic_pointer_cast<MultiRegions::ContField>(fixt_explist)
                ->GetLocalToGlobalMap();
        GlobalLinSysKey key(StdRegions::eHelmholtz, map, factors);
        auto globalSys = GetGlobalLinSysFactory().CreateInstance(
            "IterativeFull", key, fixt_explist, map);
        auto precond =
            GetPreconFactory().CreateInstance("Diagonal", globalSys, map);
        precond->BuildPreconditioner();
        precond->DoPreconditioner(incoeffs, outcoeffs, true);
        fixt_expected->CopyArray<NektarSpaces::HostSpace>(outcoeffs);
    }

private:
    double m_lambda;
};

#define TEST(type, filename)                                                   \
    class type : public DiagPreconField                                        \
    {                                                                          \
    public:                                                                    \
        type()                                                                 \
        {                                                                      \
            meshName = filename;                                               \
        }                                                                      \
    };

TEST(Helmholtz1D_Seg, "run/Helmholtz1D_P8.xml")

TEST(Helmholtz2D_Tri_Quad, "run/Helmholtz2D_varP.xml")

TEST(Helmholtz2D_AllBCs, "run/Helmholtz2D_P7_AllBCs.xml")

TEST(Helmholtz3D_Hex, "run/Helmholtz3D_Hex_Heterogeneous.xml")

TEST(Helmholtz3D_Prism, "run/Helmholtz3D_Prism_VarP.xml")

TEST(Helmholtz3D_Pyr, "run/Helmholtz3D_Pyr_VarP.xml")

TEST(Helmholtz3D_Tet, "run/Helmholtz3D_Tet_VarP.xml")
