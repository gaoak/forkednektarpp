///////////////////////////////////////////////////////////////////////////////
//
// File: init_expressionfields.hpp
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

#include "Operators/ElmtOps/Expression/ExpressionOp.hpp"

using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

class ExpressionField
    : public InitFields<double, FieldState::Phys, FieldState::Phys,
                        MultiRegions::ContField>
{
public:
    ExpressionField()
        : InitFields<double, FieldState::Phys, FieldState::Phys,
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
            for (unsigned int n = 0;
                 n < fixt_in->GetNumComponents() * fixt_in->GetNumHomoModes();
                 ++n)
            {
                for (size_t el = 0, cnt = 0; el < block.GetNumElements(); ++el)
                {
                    for (unsigned int phys = 0; phys < block.GetNumData();
                         ++phys, ++cnt)
                    {
                        inptr[cnt] = 1.0 + n;
                    }
                }
                inptr += block.size();
            }
        }
        // Set both to zero such that adding the forcing gives the same result
        fixt_out->Initialize<NektarSpaces::HostSpace>(0.0);
        fixt_expected->Initialize<NektarSpaces::HostSpace>(0.0);

        // Define time and scalar
        m_time  = 0.1;
        m_scale = 2.0;

        // Compute expected solution.
        ExpectedSolution();
    }

    void RunTestCase()
    {
        auto op = ExpressionOp<double>::Create(fixt_explist, "Forcing");
        op->SetTime(m_time);
        op->SetScale(m_scale);
        op->Apply(*fixt_in, *fixt_out);
    }

    void ExpectedSolution()
    {
        // Get number of variables and quadrature points
        const unsigned int nVariables = session->GetVariables().size();
        const size_t nphys            = fixt_explist->GetTotPoints();

        // Initialise array storage for forcing evaluation
        Array<OneD, double> fce(nVariables * nphys);

        // Get coordinate data for function evaluation
        Array<OneD, double> x(nphys);
        Array<OneD, double> y(nphys);
        Array<OneD, double> z(nphys);
        fixt_explist->GetCoords(x, y, z);

        // Copy fixt_in into NektarArray
        Array<OneD, double> inphys = fixt_in->ToArray();

        // Gather fielddata (coordinates, time and EVARS)
        std::vector<Array<OneD, const double>> fielddata;
        fielddata.push_back(x);
        fielddata.push_back(y);
        fielddata.push_back(z);
        fielddata.push_back(Array<OneD, double>(nphys, m_time));
        for (unsigned int i = 0; i < nVariables; ++i)
        {
            fielddata.push_back(inphys + i * nphys);
        }

        // Evaluate function from session file
        if (session->DefinesFunction("Forcing"))
        {
            // Evaluate function for each component
            for (unsigned int i = 0; i < nVariables; ++i)
            {
                // Create reference to storage per variable
                Array<OneD, double> fce_var = fce + i * nphys;

                // Get and evaluate function
                auto func = session->GetFunction("Forcing", i);
                func->Evaluate(fielddata, fce_var);

                // Multiply by scalar and add to input data
                Vmath::Smul(nphys, m_scale, fce_var, 1, fce_var, 1);
            }
            fixt_expected->CopyArray<NektarSpaces::HostSpace>(fce);
        }
    }

protected:
    double m_time;
    double m_scale;
};

#define TEST(type, filename)                                                   \
    class type : public ExpressionField                                        \
    {                                                                          \
    public:                                                                    \
        type()                                                                 \
        {                                                                      \
            meshName = filename;                                               \
        }                                                                      \
    };

TEST(Helmholtz1D_Seg, "run/Helmholtz1D_P8.xml")

TEST(Helmholtz2D_Tri_Quad, "run/Helmholtz2D_P7_AllBCs.xml")

TEST(Helmholtz3D_Hex, "run/Helmholtz3D_Hex_Heterogeneous.xml")

TEST(Helmholtz3D_Prism, "run/Helmholtz3D_Prism_VarP.xml")

TEST(Helmholtz3D_Pyr, "run/Helmholtz3D_Pyr_VarP.xml")

TEST(Helmholtz3D_Tet, "run/Helmholtz3D_Tet_VarP.xml")

TEST(Helmholtz3D_3C, "run/Helmholtz3D_Hex_multicomponent.xml")

TEST(Seg_3C_Evars, "run/segment_multicomponent_evars.xml")

TEST(QuadTri_2C_Evars, "run/quadtri_multicomponent_evars.xml")

TEST(Hex_3C_Evars, "run/hex_multicomponent_evars.xml")
