///////////////////////////////////////////////////////////////////////////////
//
// File: TestRobin.hpp
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

#include "Operators/BndCondOps/RobBndCond/RobBndCondOp.hpp"

using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::Operators;

template <typename TData>
class TestRobin : public TestOp<TData, FieldState::Coeff, FieldState::Coeff,
                                MultiRegions::ContField>
{
public:
    TestRobin() = default;

    void SetTestCase()
    {
        // Set initial conditions.
        this->fixt_in->template Initialize<NektarSpaces::HostSpace>(1.0);
        this->fixt_out->template Initialize<NektarSpaces::HostSpace>(0.0);
        this->fixt_expected->template Initialize<NektarSpaces::HostSpace>(0.0);

        // Compute expected solution.
        ExpectedSolution();
    }

    void RunTestCase()
    {
        std::string execStr = Operator<TData>::GetOpExecSpace(this->session);

        if (execStr == "AVX")
        {
            for (auto &block : this->fixt_in->GetBlocks())
            {
                block.template SetInterleaveWidth<TData>(
                    NektarSpaces::GetVectorWidth<TData>(execStr));
            }
        }

        auto op = RobBndCondOp<TData>::Create(this->fixt_explist,
                                              this->session->GetVariables());
        op->Apply(*this->fixt_in, *this->fixt_out);
    }

    void ExpectedSolution()
    {
        // Get variables
        auto variables     = this->fixt_explist->GetSession()->GetVariables();
        unsigned int nvars = variables.size();

        // Get number of components
        unsigned int ncomp   = this->fixt_in->GetNumComponents();
        unsigned int ncoeffs = this->fixt_explist->GetNcoeffs();
        Array<OneD, double> outcoeffs(ncomp * ncoeffs, 0.0);

        ASSERTL0(nvars == ncomp,
                 "Number of components and number fields in "
                 "session file must be equal. Instead they are Ncomp = " +
                     std::to_string(ncomp) + " and Nfields = " +
                     std::to_string(nvars) + ", respectively.")

        auto incoeffs = this->fixt_in->template ToArray<double>();

        // Impose Robin BCs for each component
        double time = 0.0;
        Array<OneD, double> tmpIn;
        Array<OneD, double> tmpOut;
        for (unsigned int nc = 0; nc < ncomp; ++nc)
        {
            // Setup dedicated explist for each variable to evaluate correct BCs
            MultiRegions::ContField bcfield(this->fixt_explist->GetSession(),
                                            this->fixt_explist->GetGraph(),
                                            variables[nc], true, false,
                                            Collections::eNoCollection);

            // Fill boundary expansions and impose
            bcfield.EvaluateBoundaryConditions(time);
            // TODO impose (local/elemental) robin mass matrix
            bcfield.ImposeRobinConditions(tmpOut = outcoeffs + nc * ncoeffs);
        }
        this->fixt_expected->template CopyArray<NektarSpaces::HostSpace>(
            outcoeffs);
    }
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TESTFLOAT(type, filename)                                              \
    class type##float : public TestRobin<float>                                \
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
    class type : public TestRobin<double>                                      \
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

TEST(Helmholtz2D_Tri_Quad_AllBCs, "run/Helmholtz2D_P7_AllBCs.xml")
