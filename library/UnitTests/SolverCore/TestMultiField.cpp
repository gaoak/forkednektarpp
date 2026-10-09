///////////////////////////////////////////////////////////////////////////////
//
// File: TestMultiField.cpp
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
// Description: Unit tests of the MultiField products against the Math
// functions, on host execution spaces.
//
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_MODULE TestMultiField

#include <LibUtilities/BasicUtils/Field/Field.hpp>
#include <LibUtilities/BasicUtils/Math/Math.hpp>

#include <SolverCore/GlobalLinSysOps/MultiFieldHelper/MultiFieldHelper.hpp>

#include <UnitTests/TestBoostSetup.hpp>
#include <UnitTests/TestGlobalConfiguration.hpp>

#include <cmath>
#include <limits>

using namespace Nektar;

NEKTAR_TEST_GLOBAL_CONFIGURATION(Nektar::UnitTests::TestArgs::Exec);

#include <UnitTests/TestBoostTeardown.hpp>

namespace
{

using CoeffField = LibUtilities::Field<double, FieldState::Coeff>;
using MaskField  = LibUtilities::Field<std::uint8_t, FieldState::Coeff>;
using NektarSpaces::HostSpace;

// Two padded blocks: 5 elements padded to 8 with 3 points, and 2 elements
// padded to 4 with 6 points.
const std::vector<LibUtilities::BlockAttributes<FieldState::Coeff>> blockAttr =
    {{5, 8, 3, 1}, {2, 4, 6, 1}};
const std::vector<std::string> components = {"u", "v"};

/// Set each entry of @p field to value(g, padding): g counts the entries,
/// padding is true in padded elements.
template <typename TData, typename F>
void Fill(LibUtilities::Field<TData, FieldState::Coeff> &field, F value)
{
    size_t g = 0;
    for (auto &block : field.GetBlocks())
    {
        auto ptr = block.template GetPtr<HostSpace, WriteOnly>();
        for (unsigned int n = 0; n < block.GetNumComponents(); ++n)
        {
            for (size_t e = 0; e < block.GetNumElementsWithPadding(); ++e)
            {
                for (unsigned int p = 0; p < block.GetNumData(); ++p, ++g)
                {
                    const bool padding = e >= block.GetNumElements();
                    *(ptr++)           = value(g, padding);
                }
            }
        }
    }
}

void FillColumn(CoeffField &field, const double seed)
{
    Fill(field, [seed](const size_t g, const bool padding) {
        return padding ? 7.0 : std::sin(0.3 * seed + 0.01 * g + 0.1);
    });
}

bool Close(CoeffField &a, CoeffField &b, const double tol)
{
    bool close = true;
    for (unsigned int blk = 0; blk < a.GetBlocks().size(); ++blk)
    {
        auto &ablk = a.GetBlocks()[blk];
        auto aptr  = ablk.GetPtr<HostSpace, ReadOnly>();
        auto bptr  = b.GetBlocks()[blk].GetPtr<HostSpace, ReadOnly>();
        for (size_t i = 0; i < ablk.CompSize() * ablk.GetNumComponents(); ++i)
        {
            close = close && std::abs(aptr[i] - bptr[i]) <=
                                 tol * (1.0 + std::abs(bptr[i]));
        }
    }
    return close;
}

/// Columns 1 to 4 of 5, over segments of 2 columns, so that the products
/// span three segments.
template <typename ExecSpace> void RunMultiDot()
{
    LibUtilities::MultiField<double, FieldState::Coeff> X("X", blockAttr,
                                                          components, 2);
    X.ResizeNumField(5);
    for (unsigned int j = 0; j < X.GetNumField(); ++j)
    {
        FillColumn(X[j], j);
    }

    CoeffField w("w", blockAttr, components, 1);
    FillColumn(w, 11.0);

    MaskField mask("mask", blockAttr, components, 1);
    Fill(mask, [](const size_t g, const bool) { return g % 3 != 0; });

    std::vector<double> dots(4), maskedDots(4);
    SolverCore::MultiDot<ExecSpace>(X, 1, 5, w, dots.data());
    SolverCore::MultiDot<ExecSpace>(mask, X, 1, 5, w, maskedDots.data());

    for (size_t j = 1; j < 5; ++j)
    {
        double ref, maskedRef;
        Math::ddot<ExecSpace>(X[j], w, &ref);
        Math::ddot<ExecSpace>(mask, X[j], w, &maskedRef);
        BOOST_CHECK_CLOSE(dots[j - 1], ref, 1.0E-10);
        BOOST_CHECK_CLOSE(maskedDots[j - 1], maskedRef, 1.0E-10);
    }
}

template <typename ExecSpace> void RunMultiAxpy()
{
    LibUtilities::MultiField<double, FieldState::Coeff> X("X", blockAttr,
                                                          components, 2);
    X.ResizeNumField(5);
    for (unsigned int j = 0; j < X.GetNumField(); ++j)
    {
        FillColumn(X[j], j);
    }

    const std::vector<double> coeffs = {0.5, -1.25, 2.0, 0.75};

    // z = -0.5 z + 2 X[1:5] coeffs.
    CoeffField z("z", blockAttr, components, 1);
    CoeffField zRef("zRef", blockAttr, components, 1);
    FillColumn(z, 13.0);
    FillColumn(zRef, 13.0);

    SolverCore::MultiAxpy<ExecSpace>(2.0, X, 1, 5, coeffs.data(), -0.5, z);

    Math::mul<ExecSpace>(-0.5, zRef, zRef);
    for (size_t j = 1; j < 5; ++j)
    {
        Math::daxpy<ExecSpace>(2.0 * coeffs[j - 1], X[j], zRef, zRef);
    }
    BOOST_CHECK(Close(z, zRef, 1.0E-13));

    // z = X[0:3] coeffs, without reading z.
    Fill(z, [](const size_t, const bool) {
        return std::numeric_limits<double>::quiet_NaN();
    });

    SolverCore::MultiAxpy<ExecSpace>(1.0, X, 0, 3, coeffs.data(), 0.0, z);

    Math::mul<ExecSpace>(coeffs[0], X[0], zRef);
    for (size_t j = 1; j < 3; ++j)
    {
        Math::daxpy<ExecSpace>(coeffs[j], X[j], zRef, zRef);
    }
    BOOST_CHECK(Close(z, zRef, 1.0E-13));
}

/// Gram matrix of columns 1 to 3 of 4, in one segment.
template <typename ExecSpace> void RunGram()
{
    LibUtilities::MultiField<double, FieldState::Coeff> X("X", blockAttr,
                                                          components, 4);
    X.ResizeNumField(4);
    for (unsigned int j = 0; j < X.GetNumField(); ++j)
    {
        FillColumn(X[j], j);
    }

    std::vector<double> gram(9);
    SolverCore::Gram<ExecSpace>(X, 1, 4, gram.data());

    for (size_t a = 0; a < 3; ++a)
    {
        for (size_t b = 0; b < 3; ++b)
        {
            double ref;
            Math::ddot<ExecSpace>(X[1 + a], X[1 + b], &ref);
            BOOST_CHECK_CLOSE(gram[a + 3 * b], ref, 1.0E-10);
        }
    }
}

} // namespace

BOOST_AUTO_TEST_SUITE(TestSuiteMultiField)

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
BOOST_AUTO_TEST_CASE(MultiField_MultiDot)
{
    const std::string &execStr = GlobalConfiguration::ExecStr();
    if (execStr == "Serial")
    {
        RunMultiDot<NektarSpaces::Serial>();
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execStr == "AVX")
    {
        RunMultiDot<NektarSpaces::AVX>();
    }
#endif
    else
    {
        BOOST_FAIL("Unsupported execution space: " + execStr);
    }
}

BOOST_AUTO_TEST_CASE(MultiField_MultiAxpy)
{
    const std::string &execStr = GlobalConfiguration::ExecStr();
    if (execStr == "Serial")
    {
        RunMultiAxpy<NektarSpaces::Serial>();
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execStr == "AVX")
    {
        RunMultiAxpy<NektarSpaces::AVX>();
    }
#endif
    else
    {
        BOOST_FAIL("Unsupported execution space: " + execStr);
    }
}

BOOST_AUTO_TEST_CASE(MultiField_Gram)
{
    const std::string &execStr = GlobalConfiguration::ExecStr();
    if (execStr == "Serial")
    {
        RunGram<NektarSpaces::Serial>();
    }
#if defined(NEKTAR_ENABLE_SIMD)
    else if (execStr == "AVX")
    {
        RunGram<NektarSpaces::AVX>();
    }
#endif
    else
    {
        BOOST_FAIL("Unsupported execution space: " + execStr);
    }
}
#endif

BOOST_AUTO_TEST_SUITE_END()
