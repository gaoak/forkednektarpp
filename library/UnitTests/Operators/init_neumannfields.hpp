///////////////////////////////////////////////////////////////////////////////
//
// File: init_neumannfields.hpp
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

#include "Operators/BndCondOps/NeuBndCond/NeuBndCondOp.hpp"

using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

template <typename TData>
class NeumannField
    : public InitFields<TData, FieldState::Coeff, FieldState::Coeff,
                        MultiRegions::ContField>
{
public:
    NeumannField()
        : InitFields<TData, FieldState::Coeff, FieldState::Coeff,
                     MultiRegions::ContField>()
    {
    }

    void SetTestCase(const double time = 0.0)
    {
        // Set initial conditions.
        this->fixt_out->template Initialize<NektarSpaces::HostSpace>(0.0);

        // Compute expected solution.
        ExpectedSolution(time);
    }

    void RunTestCase(const double time = 0.0)
    {
        std::string execStr = Operator<TData>::GetOpExecSpace(this->session);

        if (execStr == "AVX")
        {
            for (auto &block : this->fixt_out->GetBlocks())
            {
                block.template SetInterleaveWidth<TData>(
                    NektarSpaces::GetVectorWidth<TData>(execStr));
            }
        }

        auto op = NeuBndCondOp<TData>::Create(this->fixt_explist,
                                              this->session->GetVariables());
        op->UpdateBndCoeffs(time);
        op->Apply(*this->fixt_out);
    }

    void ExpectedSolution(const double time = 0.0)
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

        // Impose Neumann BCs for each component
        Array<OneD, double> tmp;
        for (unsigned int nc = 0; nc < ncomp; ++nc)
        {
            // Setup dedicated explist for each variable to evaluate correct BCs
            MultiRegions::ContField bcfield(this->fixt_explist->GetSession(),
                                            this->fixt_explist->GetGraph(),
                                            variables[nc], true, false,
                                            Collections::eNoCollection);

            // Update BC for this variable and impose
            bcfield.EvaluateBoundaryConditions(time, variables[nc]);
            bcfield.ImposeNeumannConditions(tmp = outcoeffs + nc * ncoeffs);
        }
        this->fixt_expected->template CopyArray<NektarSpaces::HostSpace>(
            outcoeffs);
    }
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TESTFLOAT(type, filename)                                              \
    class type##float : public NeumannField<float>                             \
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
    class type : public NeumannField<double>                                   \
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

TEST(Helmholtz1D_Seg_TimeDependentNeumann,
     "run/Helmholtz1D_P8_TimeDependentDirichlet.xml")

TEST(Helmholtz1D_Seg_3C, "run/Helmholtz1D_3C.xml")

TEST(Helmholtz1D_Seg_3C_mixedBC, "run/Helmholtz1D_3C_mixedBC.xml")

TEST(Helmholtz2D_Quad, "run/Helmholtz2D_Quad.xml")

TEST(Helmholtz2D_Quad_3C, "run/Helmholtz2D_Quad_3C.xml")

TEST(Helmholtz2D_Tri, "run/Helmholtz2D_Tri.xml")

TEST(Helmholtz2D_Tri_3C, "run/Helmholtz2D_Tri_3C.xml")

TEST(Helmholtz2D_Tri_Quad, "run/Helmholtz2D_P7_AllBCs.xml")

TEST(Helmholtz2D_Tri_Quad_TimeDependentBC,
     "run/Helmholtz2D_P7_AllBCs_TimeDependentDirichlet.xml")

TEST(Helmholtz2D_Tri_Quad_3C, "run/Helmholtz2D_3C.xml")

TEST(Helmholtz2D_Tri_Quad_3C_mixedBC, "run/Helmholtz2D_3C_mixedBC.xml")

TEST(Helmholtz3D_Hex, "run/Helmholtz3D_Hex_Heterogeneous.xml")

TEST(Helmholtz3D_Prism, "run/Helmholtz3D_Prism_VarP.xml")

TEST(Helmholtz3D_Prism_TimeDependentBC,
     "run/Helmholtz3D_Prism_VarP_TimeDependentBC.xml")

TEST(Helmholtz3D_Pyr, "run/Helmholtz3D_Pyr_VarP.xml")

TEST(Helmholtz3D_Tet, "run/Helmholtz3D_Tet_VarP.xml")

TEST(Helmholtz3D_Tet_TimeDependentBC,
     "run/Helmholtz3D_Tet_VarP_TimeDependentBC.xml")

TEST(Helmholtz3D_Hex_AllBCs, "run/Helmholtz3D_Hex_AllBCs_P6.xml")

TEST(Helmholtz3D_Hex_AllBCs_TimeDependentBC,
     "run/Helmholtz3D_Hex_AllBCs_P6_TimeDependentBC.xml")

TEST(Helmholtz3D_Hex_3C, "run/Helmholtz3D_Hex_3C.xml")

TEST(Helmholtz3D_Hex_3C_mixedBC, "run/Helmholtz3D_Hex_3C_mixedBC.xml")
