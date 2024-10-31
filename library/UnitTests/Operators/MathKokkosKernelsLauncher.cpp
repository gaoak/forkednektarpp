///////////////////////////////////////////////////////////////////////////////
//
// File: MathKokkosKernelsLauncher.cpp
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

#include "Operators/MathKernels/MathKernels.hpp"

#include "MathKernelsLauncher.hpp"

using namespace Nektar;
using ExecSpace = NektarSpaces::KOKKOS;

void negKernelLauncher(Field<double, FieldState::Phys> &x,
                       Field<double, FieldState::Phys> &y)
{
    neg<ExecSpace, double>(x, y);
}

void addKernelLauncher(Field<double, FieldState::Phys> &x,
                       Field<double, FieldState::Phys> &y,
                       Field<double, FieldState::Phys> &z)
{
    add<ExecSpace, double>(x, y, z);
}

void subKernelLauncher(Field<double, FieldState::Phys> &x,
                       Field<double, FieldState::Phys> &y,
                       Field<double, FieldState::Phys> &z)
{
    sub<ExecSpace, double>(x, y, z);
}

void daxpyKernelLauncher(const double alpha, Field<double, FieldState::Phys> &x,
                         Field<double, FieldState::Phys> &y,
                         Field<double, FieldState::Phys> &z)
{
    daxpy<ExecSpace, double>(alpha, x, y, z);
}

void divKernelLauncher(Field<double, FieldState::Phys> &x,
                       Field<double, FieldState::Phys> &y,
                       Field<double, FieldState::Phys> &z)
{
    div<ExecSpace, double>(x, y, z);
}

double sumKernelLauncher(Field<double, FieldState::Phys> &x)
{
    double ans;
    reduceSum<ExecSpace, double>(x, &ans);
    return ans;
}

double maxKernelLauncher(Field<double, FieldState::Phys> &x)
{
    double ans;
    reduceMax<ExecSpace, double>(x, &ans);
    return ans;
}

double minKernelLauncher(Field<double, FieldState::Phys> &x)
{
    double ans;
    reduceMin<ExecSpace, double>(x, &ans);
    return ans;
}

double innerproductKernelLauncher(Field<double, FieldState::Phys> &x,
                                  Field<double, FieldState::Phys> &y)
{
    double ans;
    ddot<ExecSpace, double>(x, y, &ans);
    return ans;
}

double l1normKernelLauncher(Field<double, FieldState::Phys> &x)
{
    double ans;
    l1norm<ExecSpace, double>(x, &ans);
    return ans;
}

double l2normKernelLauncher(Field<double, FieldState::Phys> &x)
{
    double ans;
    l2norm<ExecSpace, double>(x, &ans);
    return ans;
}

double lpnormKernelLauncher(const unsigned int p,
                            Field<double, FieldState::Phys> &x)
{
    double ans;
    lpnorm<ExecSpace, double>(p, x, &ans);
    return ans;
}

double linfnormKernelLauncher(Field<double, FieldState::Phys> &x)
{
    double ans;
    linfnorm<ExecSpace, double>(x, &ans);
    return ans;
}
