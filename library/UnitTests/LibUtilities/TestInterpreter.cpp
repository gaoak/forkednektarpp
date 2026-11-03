///////////////////////////////////////////////////////////////////////////////
//
// File: TestInterpreter.cpp
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

#include <LibUtilities/Interpreter/Interpreter.h>

#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>

#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace Nektar::InterpreterUnitTests
{

BOOST_AUTO_TEST_CASE(TestConstant)
{
    LibUtilities::Interpreter interp;
    int func1 = interp.DefineFunction("x", "-2");
    Array<OneD, NekDouble> in(1, 2.0), out(1);

    interp.Evaluate(func1, {in}, out);
    BOOST_CHECK_EQUAL(out[0], -2);
}

BOOST_AUTO_TEST_CASE(TestPowOperator)
{
    LibUtilities::Interpreter interp;
    int func1 = interp.DefineFunction("x", "5*(-(2^x)^4)");
    int func2 = interp.DefineFunction("x", "-x^2");
    int func3 = interp.DefineFunction("x", "2^2^4");
    Array<OneD, NekDouble> in(1, 2.0), out(1);

    interp.Evaluate(func1, {in}, out);
    BOOST_CHECK_EQUAL(out[0], -1280);

    interp.Evaluate(func2, {in}, out);
    BOOST_CHECK_EQUAL(out[0], -4);

    interp.Evaluate(func3, {in}, out);
    BOOST_CHECK_EQUAL(out[0], 65536);
}

BOOST_AUTO_TEST_CASE(TestFmodOperator)
{
    LibUtilities::Interpreter interp;
    int func1 = interp.DefineFunction("x", "x % 3.2");
    int func2 = interp.DefineFunction("x", "fmod(x,3.2)");
    Array<OneD, NekDouble> in(1, 5.2), out(1);

    NekDouble epsilon = std::numeric_limits<NekDouble>::epsilon();

    interp.Evaluate(func1, {in}, out);
    BOOST_CHECK_CLOSE(out[0], 2.0, epsilon);
    interp.Evaluate(func2, {in}, out);
    BOOST_CHECK_CLOSE(out[0], 2.0, epsilon);
}

BOOST_AUTO_TEST_CASE(TestBesselFunction)
{
    LibUtilities::Interpreter interp;
    int func1 = interp.DefineFunction("x", "bessel(1, x)");
    int func2 = interp.DefineFunction("x", "bessel(0, 0)");
    Array<OneD, NekDouble> in(1, 2.0), out(1);

    // J_1(2) and J_0(0).
    interp.Evaluate(func1, {in}, out);
    BOOST_CHECK_CLOSE(out[0], 0.57672480775687338720, 1.0e-12);
    interp.Evaluate(func2, {in}, out);
    BOOST_CHECK_CLOSE(out[0], 1.0, 1.0e-12);
}

BOOST_AUTO_TEST_CASE(TestTwoArgumentFunctionOneConstant)
{
    // A fresh interpreter per expression, so that no earlier evaluation has
    // left the constant argument in its state.
    const std::vector<std::pair<std::string, NekDouble>> cases = {
        {"atan2(1, x)", std::atan2(1.0, 2.0)},
        {"atan2(x, 1)", std::atan2(2.0, 1.0)},
        {"fmod(7.5, x)", 1.5},
        {"fmod(x, 1.5)", 0.5},
        {"bessel(1, x)", 0.57672480775687338720}};
    Array<OneD, NekDouble> in(1, 2.0), out(1);

    for (auto &c : cases)
    {
        LibUtilities::Interpreter interp;
        int func = interp.DefineFunction("x", c.first);
        interp.Evaluate(func, {in}, out);
        BOOST_CHECK_CLOSE(out[0], c.second, 1.0e-12);
    }
}

} // namespace Nektar::InterpreterUnitTests
