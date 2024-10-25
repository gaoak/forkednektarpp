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

#include "Operators/PreconOps/OperatorDiagPrecon.hpp"

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
        double *inptr =
            fixt_in->template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        for (auto const &block : fixt_in->GetBlocks())
        {
            for (size_t el = 0, cnt = 0; el < block.num_elements; ++el)
            {
                for (size_t coeff = 0; coeff < block.num_pts; ++coeff, ++cnt)
                {
                    inptr[cnt] = 1.0;
                }
            }
            inptr += block.block_size;
        }
        ExpectedSolution();
    }

    template <typename ExecSpace, typename Impl> void RunTestCase()
    {
        auto HelmholtzOp =
            Helmholtz<>::template create<ExecSpace, Impl>(fixt_explist);
        auto DiagPreconOp =
            DiagPrecon<>::template create<ExecSpace, Impl>(fixt_explist);
        HelmholtzOp->setLambda(1.0);
        DiagPreconOp->configure(HelmholtzOp);
        DiagPreconOp->apply(*fixt_in, *fixt_out);
    }

    void ExpectedSolution()
    {
        // Calculate expected result from Nektar++
        Array<OneD, double> incoeffs = fixt_in->toArray();
        Array<OneD, double> outcoeffs(fixt_explist->GetNcoeffs(), 0.0);
        StdRegions::ConstFactorMap factors;
        factors[StdRegions::eFactorLambda] =
            fixt_explist->GetSession()->DefinesParameter("Lambda")
                ? fixt_explist->GetSession()->GetParameter("Lambda")
                : 1.0;
        auto map = fixt_explist->GetLocalToGlobalMap();
        GlobalLinSysKey key(StdRegions::eHelmholtz, map, factors);
        auto globalSys = GetGlobalLinSysFactory().CreateInstance(
            "IterativeFull", key, fixt_explist, map);
        auto precond =
            GetPreconFactory().CreateInstance("Diagonal", globalSys, map);
        precond->BuildPreconditioner();
        precond->DoPreconditioner(incoeffs, outcoeffs, true);

        // Copy expected result from Array to pointer
        double *coeffptr = outcoeffs.get();
        double *expptr =
            fixt_expected
                ->template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        for (auto const &block : fixt_expected->GetBlocks())
        {
            size_t cnt = 0;
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                for (size_t coeff = 0; coeff < block.num_pts; ++coeff, ++cnt)
                {
                    expptr[cnt] = (*coeffptr++);
                }
            }
            expptr += block.block_size;
        }
    }
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
