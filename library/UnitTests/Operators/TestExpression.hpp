///////////////////////////////////////////////////////////////////////////////
//
// File: TestExpression.hpp
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

#include "TestOp.hpp"

#include "Operators/ElmtOps/Expression/ExpressionOp.hpp"
#include <LibUtilities/BasicUtils/Utils/UtilsKernels.hpp>

using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::Operators;

template <typename TData>
class TestExpression : public TestOp<TData, FieldState::Phys, FieldState::Phys,
                                     MultiRegions::ContField>
{
public:
    TestExpression() = default;

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
                        inptr[cnt] = 1.0 + n;
                    }
                }
                inptr += block.CompSize();
            }
        }
        // Set both to zero such that adding the forcing gives the same result
        this->fixt_out->template Initialize<NektarSpaces::HostSpace>(0.0);
        this->fixt_expected->template Initialize<NektarSpaces::HostSpace>(0.0);

        // Define time and scalar
        m_time  = 0.1;
        m_scale = 2.0;

        // Compute expected solution.
        ExpectedSolution();
    }

    void RunTestCase()
    {
        std::string execStr = Operator<TData>::GetOpExecSpace(this->session);

        // reshape this->fixt_in
        if (execStr == "AVX")
        {
            this->fixt_in->ReshapeStorage(
                NektarSpaces::GetVectorWidth<TData>(execStr), execStr);
        }

        auto op = ExpressionOp<TData>::Create(this->fixt_explist,
                                              this->session->GetVariables());
        op->SetTime(m_time);
        op->SetScale(m_scale);
        std::vector<LibUtilities::EquationSharedPtr> expressions;
        const auto numfields = this->session->GetVariables().size();
        for (int nf = 0; nf < numfields; ++nf)
        {
            expressions.push_back(this->session->GetFunction("Forcing", nf));
        }
        op->SetExpressions(expressions);
        op->Apply(*this->fixt_in, *this->fixt_out);
    }

    void ExpectedSolution()
    {
        // Get number of variables and quadrature points
        const unsigned int nVariables = this->session->GetVariables().size();
        const size_t nphys            = this->fixt_explist->GetTotPoints();

        // Initialise array storage for forcing evaluation
        Array<OneD, TData> fce(nVariables * nphys);

        // Get coordinate data for function evaluation
        Array<OneD, TData> x(nphys);
        Array<OneD, TData> y(nphys);
        Array<OneD, TData> z(nphys);
        this->fixt_explist->GetCoords(x, y, z);

        // Copy this->fixt_in into NektarArray
        Array<OneD, TData> inphys = this->fixt_in->ToArray();

        // Gather fielddata (coordinates, time and EVARS)
        std::vector<Array<OneD, const TData>> fielddata;
        fielddata.push_back(x);
        fielddata.push_back(y);
        fielddata.push_back(z);
        fielddata.push_back(Array<OneD, TData>(nphys, m_time));
        for (unsigned int i = 0; i < nVariables; ++i)
        {
            fielddata.push_back(inphys + i * nphys);
        }

        // Evaluate function from this->session file
        if (this->session->DefinesFunction("Forcing"))
        {
            // Evaluate function for each component
            for (unsigned int i = 0; i < nVariables; ++i)
            {
                // Create reference to storage per variable
                Array<OneD, TData> fce_var = fce + i * nphys;

                // Get and evaluate function
                auto func = this->session->GetFunction("Forcing", i);
                func->Evaluate(fielddata, fce_var);

                // Multiply by scalar and add to input data
                Vmath::Smul(nphys, m_scale, fce_var, 1, fce_var, 1);
            }
            this->fixt_expected->template CopyArray<NektarSpaces::HostSpace>(
                fce);
        }
    }

protected:
    TData m_time;
    TData m_scale;
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TESTFLOAT(type, filename)                                              \
    class type##float : public TestExpression<float>                           \
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
    class type : public TestExpression<double>                                 \
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

TEST(Helmholtz2D_Tri_Quad, "run/Helmholtz2D_P7_AllBCs.xml")

TEST(Helmholtz3D_Hex, "run/Helmholtz3D_Hex_Heterogeneous.xml")

TEST(Helmholtz3D_Prism, "run/Helmholtz3D_Prism_VarP.xml")

TEST(Helmholtz3D_Pyr, "run/Helmholtz3D_Pyr_VarP.xml")

TEST(Helmholtz3D_Tet, "run/Helmholtz3D_Tet_VarP.xml")

TEST(Helmholtz3D_3C, "run/Helmholtz3D_Hex_3C.xml")

TEST(Seg_3C_Evars, "run/segment_multicomponent_evars.xml")

TEST(QuadTri_2C_Evars, "run/quadtri_multicomponent_evars.xml")

TEST(Hex_3C_Evars, "run/hex_multicomponent_evars.xml")
