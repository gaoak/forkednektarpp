///////////////////////////////////////////////////////////////////////////////
//
// File: TestAvgAssmbScatr.hpp
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
// Description: Fixture of the AvgAssmbScatr unit tests, which compare the
// operator with AvgAssemble() and GlobalToLocal() of the legacy assembly map.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "TestOp.hpp"

#include <MultiRegions/AvgAssmbScatr/AvgAssmbScatrOp.hpp>

#include <MultiRegions/ContField.h>

#include <LibUtilities/Backends/Backends.hpp>

using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::MultiRegions;

/// Input of the tests at a local coefficient: a function of the centroid of
/// its element, which makes the field discontinuous across every element
/// boundary, plus a function of the mode and the component.
template <typename TData>
TData AvgAssmbScatrTestFn(const double cx, const double cy, const double cz,
                          const unsigned int mode, const unsigned int comp)
{
    return static_cast<TData>(1.0 + std::sin(1.7 * cx + 0.3) +
                              std::cos(2.3 * cy) + 0.5 * cz +
                              0.25 * std::cos(1.1 * mode) + 0.1 * comp);
}

template <typename TData>
class TestAvgAssmbScatr
    : public TestOp<TData, FieldState::Coeff, FieldState::Coeff, ContField>
{
public:
    TestAvgAssmbScatr() = default;

public:
    /// Operator under test, created once per test case.
    std::shared_ptr<AvgAssmbScatrOp<TData>> fixt_op;

    /// Fill the input from the element centroids, which a partition does
    /// not change, set the expected solution to the legacy average and
    /// create the operator.
    void SetTestCase()
    {
        for (unsigned int blk = 0; blk < this->fixt_in->GetBlocks().size();
             ++blk)
        {
            auto &block = this->fixt_in->GetBlocks()[blk];
            auto collection =
                MultiRegions::GetCollection(this->fixt_explist, blk);
            const auto &expansions = collection.GetExpVector();
            auto inptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            for (unsigned int nc = 0; nc < this->fixt_in->GetNumComponents();
                 ++nc)
            {
                for (size_t el = 0, cnt = 0; el < expansions.size(); ++el)
                {
                    const auto npts = expansions[el]->GetTotPoints();
                    Array<OneD, NekDouble> x(npts, 0.0), y(npts, 0.0),
                        z(npts, 0.0);
                    expansions[el]->GetCoords(x, y, z);
                    double cx = 0.0, cy = 0.0, cz = 0.0;
                    for (int q = 0; q < npts; ++q)
                    {
                        cx += x[q];
                        cy += y[q];
                        cz += z[q];
                    }
                    cx /= npts;
                    cy /= npts;
                    cz /= npts;
                    for (unsigned int coeff = 0; coeff < block.GetNumData();
                         ++coeff, ++cnt)
                    {
                        inptr[cnt] =
                            AvgAssmbScatrTestFn<TData>(cx, cy, cz, coeff, nc);
                    }
                }
                inptr += block.CompSize();
            }
        }

        ExpectedSolution();

        fixt_op = AvgAssmbScatrOp<TData>::Create(this->fixt_explist,
                                                 this->session->GetVariables());
    }

    /// Reshape the input to the interleave width of the execution space.
    void ReshapeInput()
    {
        std::string execStr = Operator<TData>::GetOpExecSpace(this->session);
        if (execStr == "AVX")
        {
            this->fixt_in->ReshapeStorage(
                NektarSpaces::GetVectorWidth<TData>(execStr), execStr);
        }
    }

    /// Interleave width of every block of @p field.
    std::vector<unsigned int> GetInterleaveWidths(
        Field<TData, FieldState::Coeff> &field)
    {
        std::vector<unsigned int> widths;
        for (auto &block : field.GetBlocks())
        {
            widths.push_back(block.GetInterleaveWidth());
        }
        return widths;
    }

    /// Average the input in place into the output, and return whether the
    /// average keeps the interleave width of the input.
    bool RunTestCase()
    {
        ReshapeInput();
        const auto inWidths = GetInterleaveWidths(*this->fixt_in);

        fixt_op->Apply(*this->fixt_in);
        this->fixt_out->template Copy<NektarSpaces::HostSpace>(*this->fixt_in);

        return GetInterleaveWidths(*this->fixt_in) == inWidths;
    }

    /// Average the input once into the expected solution and that once more
    /// into the output, which must then equal the expected solution.
    void RunTestCaseTwice()
    {
        ReshapeInput();
        fixt_op->Apply(*this->fixt_in);
        this->fixt_expected->template Copy<NektarSpaces::HostSpace>(
            *this->fixt_in);
        fixt_op->Apply(*this->fixt_in);
        this->fixt_out->template Copy<NektarSpaces::HostSpace>(*this->fixt_in);
    }

    /// Average the continuous legacy average, which must be left unchanged.
    void RunTestCaseContinuous()
    {
        this->fixt_in->template Copy<NektarSpaces::HostSpace>(
            *this->fixt_expected);
        ReshapeInput();
        fixt_op->Apply(*this->fixt_in);
        this->fixt_out->template Copy<NektarSpaces::HostSpace>(*this->fixt_in);
    }

    /// First of two inputs to one operator: the continuous legacy average
    /// at interleave width 1, averaged in place in the output, which must
    /// be left unchanged.
    void RunTestCaseReuseFirst()
    {
        this->fixt_out->template Copy<NektarSpaces::HostSpace>(
            *this->fixt_expected);
        fixt_op->Apply(*this->fixt_out);
    }

    /// Second of two inputs to the same operator: the discontinuous input at
    /// the interleave width of the execution space, averaged into the
    /// output, which must equal the legacy average.
    void RunTestCaseReuseSecond()
    {
        ReshapeInput();
        fixt_op->Apply(*this->fixt_in, *this->fixt_out);
    }

    /// Set the expected solution to AvgAssemble() then GlobalToLocal() of
    /// the input, per component.
    void ExpectedSolution()
    {
        std::vector<std::string> variables = this->session->GetVariables();
        std::vector<std::shared_ptr<MultiRegions::ContField>> contfields;
        auto graph           = SpatialDomains::MeshGraphIO::Read(this->session);
        unsigned int numComp = this->fixt_in->GetNumComponents();

        for (auto &variable : variables)
        {
            contfields.push_back(
                MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
                    this->session, graph, variable, true, false,
                    Collections::eNoCollection));
        }

        size_t ncoeffs               = this->fixt_explist->GetNcoeffs();
        Array<OneD, double> incoeffs = this->fixt_in->ToArray();
        Array<OneD, double> outcoeffs(numComp * ncoeffs);

        for (unsigned int i = 0; i < variables.size(); ++i)
        {
            Array<OneD, double> tmp;
            auto map = contfields[i]->GetLocalToGlobalMap();
            map->AvgAssemble(incoeffs + i * ncoeffs,
                             tmp = outcoeffs + i * ncoeffs);
            map->GlobalToLocal(outcoeffs + i * ncoeffs,
                               tmp = outcoeffs + i * ncoeffs);
        }
        this->fixt_expected->template CopyArray<NektarSpaces::HostSpace>(
            outcoeffs);
    }
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TESTFLOAT(type, filename)                                              \
    class type##float : public TestAvgAssmbScatr<float>                        \
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
    class type : public TestAvgAssmbScatr<double>                              \
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

TEST(Seg, "run/segment.xml")

TEST(Quad, "run/square.xml")

TEST(QuadVarP, "run/square_varp.xml")

TEST(QuadSEM, "run/square_sem.xml")

TEST(QuadPer, "run/Quad_Per.xml")

TEST(Tri, "run/tri.xml")

TEST(TriVarP, "run/tri_varp.xml")

TEST(SquareAllElements, "run/square_all_elements.xml")

TEST(Hex, "run/hex.xml")

TEST(HexVarP, "run/hex_varp.xml")

TEST(HexSEM, "run/hex_sem.xml")

TEST(HexPer, "run/Hex_Per.xml")

TEST(Prism, "run/prism.xml")

TEST(PrismVarP, "run/prism_varp.xml")

TEST(Pyr, "run/pyr.xml")

TEST(PyrVarP, "run/pyr_varp.xml")

TEST(Tet, "run/tet.xml")

TEST(TetVarP, "run/tet_varp.xml")

TEST(CubePrismHex, "run/cube_prismhex.xml")

TEST(CubeAllElements, "run/cube_all_elements.xml")
