///////////////////////////////////////////////////////////////////////////////
//
// File: MathKernelsLauncher.hpp
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

#include <Operators/Field/Field.hpp>

void negKernelLauncher(Field<double, FieldState::Phys> &,
                       Field<double, FieldState::Phys> &);
void addKernelLauncher(Field<double, FieldState::Phys> &,
                       Field<double, FieldState::Phys> &,
                       Field<double, FieldState::Phys> &);
void subKernelLauncher(Field<double, FieldState::Phys> &,
                       Field<double, FieldState::Phys> &,
                       Field<double, FieldState::Phys> &);
void mulKernelLauncher(Field<double, FieldState::Phys> &,
                       Field<double, FieldState::Phys> &,
                       Field<double, FieldState::Phys> &);
void divKernelLauncher(Field<double, FieldState::Phys> &,
                       Field<double, FieldState::Phys> &,
                       Field<double, FieldState::Phys> &);
void daxpyKernelLauncher(const double alpha, Field<double, FieldState::Phys> &,
                         Field<double, FieldState::Phys> &,
                         Field<double, FieldState::Phys> &);
double sumKernelLauncher(Field<double, FieldState::Phys> &);
double maxKernelLauncher(Field<double, FieldState::Phys> &);
double minKernelLauncher(Field<double, FieldState::Phys> &);
double innerproductKernelLauncher(Field<double, FieldState::Phys> &,
                                  Field<double, FieldState::Phys> &);
double l1normKernelLauncher(Field<double, FieldState::Phys> &);
double l2normKernelLauncher(Field<double, FieldState::Phys> &);
double lpnormKernelLauncher(const unsigned int p,
                            Field<double, FieldState::Phys> &);
double linfnormKernelLauncher(Field<double, FieldState::Phys> &);
