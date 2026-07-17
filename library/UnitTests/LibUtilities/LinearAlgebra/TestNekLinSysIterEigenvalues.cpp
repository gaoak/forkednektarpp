///////////////////////////////////////////////////////////////////////////////
//
// File: TestNekLinSysIterEigenvalues.cpp
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
// Description: Unit tests for iterative linear solver eigenvalue estimates.
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/LinearAlgebra/Lapack.hpp>
#include <LibUtilities/LinearAlgebra/NekLinSysIterCG.h>
#include <LibUtilities/LinearAlgebra/NekLinSysIterGMRES.h>

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

namespace Nektar::NekLinSysIterEigenvalueUnitTests
{
using namespace LibUtilities;

class BandedMatrixOperator
{
public:
    BandedMatrixOperator(const std::vector<NekDouble> &diagonal,
                         const std::vector<NekDouble> &offDiagonal)
        : m_diagonal(diagonal), m_offDiagonal(offDiagonal)
    {
    }

    void Apply(const Array<OneD, const NekDouble> &input,
               Array<OneD, NekDouble> &output, const bool &flag)
    {
        boost::ignore_unused(flag);

        std::fill(output.data(), output.data() + output.size(), 0.0);
        for (size_t i = 0; i < m_diagonal.size(); ++i)
        {
            output[i] += m_diagonal[i] * input[i];
            if (i > 0)
            {
                output[i] += m_offDiagonal[i - 1] * input[i - 1];
            }
            if (i + 1 < m_diagonal.size())
            {
                output[i] += m_offDiagonal[i] * input[i + 1];
            }
        }
    }

private:
    std::vector<NekDouble> m_diagonal;
    std::vector<NekDouble> m_offDiagonal;
};

// create a fake session file for the linear solver
SessionReaderSharedPtr CreateTestSession()
{
    const std::string filename =
        "/tmp/nektar_linsys_iter_eigenvalue_session.xml";
    std::ofstream sessionFile(filename);
    sessionFile << "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
                << "<NEKTAR>\n"
                << "  <CONDITIONS />\n"
                << "</NEKTAR>\n";
    sessionFile.close();

    char arg0[]                        = "LinearAlgebraUnitTests";
    char *argv[]                       = {arg0, nullptr};
    std::vector<std::string> filenames = {filename};
    CommSharedPtr comm = GetCommFactory().CreateInstance("Serial", 0, nullptr);
    return SessionReader::CreateInstance(1, argv, filenames, comm);
}

std::vector<NekDouble> CreateBandedMatrix(
    const std::vector<NekDouble> &diagonal,
    const std::vector<NekDouble> &offDiagonal)
{
    const int n = diagonal.size();
    std::vector<NekDouble> matrix(n * n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        matrix[i + i * n] = diagonal[i];
        if (i + 1 < n)
        {
            matrix[i + (i + 1) * n] = offDiagonal[i];
            matrix[i + 1 + i * n]   = offDiagonal[i];
        }
    }
    return matrix;
}

NekDouble MaxRealEigenvalueFromLapack(const std::vector<NekDouble> &matrix,
                                      const int n)
{
    std::vector<NekDouble> a = matrix;
    Array<OneD, NekDouble> wr(n, 0.0);
    Array<OneD, NekDouble> wi(n, 0.0);
    Array<OneD, NekDouble> work(3 * n, 0.0);

    const char jobvl = 'N';
    const char jobvr = 'N';
    NekDouble dum    = 0.0;
    int info         = 0;

    Lapack::Dgeev(jobvl, jobvr, n, a.data(), n, wr.data(), wi.data(), &dum, 1,
                  &dum, 1, work.data(), work.size(), info);
    BOOST_REQUIRE_EQUAL(info, 0);

    return *std::max_element(wr.data(), wr.data() + n);
}

NekDouble MaxValue(const Array<OneD, const NekDouble> &values)
{
    return *std::max_element(values.data(), values.data() + values.size());
}

BOOST_AUTO_TEST_CASE(TestCGComputesMaximumEigenvalue)
{
    const std::vector<NekDouble> diagonal    = {4.0, 5.0, 6.0, 7.0, 30.0};
    const std::vector<NekDouble> offDiagonal = {0.5, -0.35, 0.4, -0.2};
    const int n                              = diagonal.size();
    const NekDouble reference                = MaxRealEigenvalueFromLapack(
        CreateBandedMatrix(diagonal, offDiagonal), n);

    auto session = CreateTestSession();

    NekSysKey key;
    key.m_NekLinSysMaxIterations = 10;
    key.m_NekLinSysTolerance     = 1.0e-13;

    NekLinSysIterCG solver(session, session->GetComm(), n, key);
    solver.InitObject();
    solver.SetEigenValueFlags(true, false);

    BandedMatrixOperator matrixOperator(diagonal, offDiagonal);
    NekSysOperators operators;
    operators.DefineNekSysLhsEval(&BandedMatrixOperator::Apply,
                                  &matrixOperator);
    solver.SetSysOperators(operators);

    Array<OneD, NekDouble> rhs(n, 1.0);
    Array<OneD, NekDouble> solution(n, 0.0);
    solver.SolveSystem(n, rhs, solution);

    const Array<OneD, NekDouble> &computed = solver.GetEigenValues();
    BOOST_REQUIRE_EQUAL(computed.size(), n);
    BOOST_CHECK_SMALL(std::abs(MaxValue(computed) - reference) / reference,
                      1.0e-10);
}

BOOST_AUTO_TEST_CASE(TestGMRESComputesMaximumEigenvalue)
{
    const std::vector<NekDouble> diagonal    = {4.0, 5.0, 6.0, 7.0, 30.0};
    const std::vector<NekDouble> offDiagonal = {0.5, -0.35, 0.4, -0.2};
    const int n                              = diagonal.size();
    const NekDouble reference                = MaxRealEigenvalueFromLapack(
        CreateBandedMatrix(diagonal, offDiagonal), n);

    auto session = CreateTestSession();

    NekSysKey key;
    key.m_NekLinSysMaxIterations = n - 1;
    key.m_NekLinSysTolerance     = 1.0e-13;
    key.m_LinSysMaxStorage       = n - 1;
    key.m_KrylovMaxHessMatBand   = n;

    NekLinSysIterGMRES solver(session, session->GetComm(), n, key);
    solver.InitObject();
    solver.SetFlagWarnings(false);
    solver.SetEigenValueFlags(true, false);

    BandedMatrixOperator matrixOperator(diagonal, offDiagonal);
    NekSysOperators operators;
    operators.DefineNekSysLhsEval(&BandedMatrixOperator::Apply,
                                  &matrixOperator);
    solver.SetSysOperators(operators);

    Array<OneD, NekDouble> rhs(n, 1.0);
    Array<OneD, NekDouble> solution(n, 0.0);
    solver.SolveSystem(n, rhs, solution);

    const Array<OneD, NekDouble> &computed = solver.GetEigenValues();
    BOOST_REQUIRE_EQUAL(computed.size(), n);
    BOOST_CHECK_SMALL(std::abs(MaxValue(computed) - reference) / reference,
                      1.0e-6);
}
} // namespace Nektar::NekLinSysIterEigenvalueUnitTests
