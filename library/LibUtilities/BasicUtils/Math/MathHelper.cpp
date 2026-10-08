///////////////////////////////////////////////////////////////////////////////
//
// File: MathHelper.cpp
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

#include "LibUtilities/BasicUtils/Math/MathHelperDef.hpp"

// MathHelper explicit instantiation for MemoryRegion based functions. Provide
// functionalities to dynamically select the Execution space by using a string
// parameter.
namespace Nektar::Math
{

// zero template specialization.
template void MathHelper::zero<LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<double> &x, const std::string &execSpace);
template void MathHelper::zero<LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<float> &x, const std::string &execSpace);
template void MathHelper::zero<LibUtilities::Field<double, FieldState::Phys>>(
    LibUtilities::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template void MathHelper::zero<LibUtilities::Field<float, FieldState::Phys>>(
    LibUtilities::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template void MathHelper::zero<LibUtilities::Field<double, FieldState::Coeff>>(
    LibUtilities::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template void MathHelper::zero<LibUtilities::Field<float, FieldState::Coeff>>(
    LibUtilities::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);

// fill template specialization.
template void MathHelper::fill<double, LibUtilities::MemoryRegion<double>>(
    const double &val, LibUtilities::MemoryRegion<double> &x,
    const std::string &execSpace);
template void MathHelper::fill<float, LibUtilities::MemoryRegion<float>>(
    const float &val, LibUtilities::MemoryRegion<float> &x,
    const std::string &execSpace);
template void MathHelper::fill<double,
                               LibUtilities::Field<double, FieldState::Phys>>(
    const double &val, LibUtilities::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template void MathHelper::fill<float,
                               LibUtilities::Field<float, FieldState::Phys>>(
    const float &val, LibUtilities::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template void MathHelper::fill<double,
                               LibUtilities::Field<double, FieldState::Coeff>>(
    const double &val, LibUtilities::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template void MathHelper::fill<float,
                               LibUtilities::Field<float, FieldState::Coeff>>(
    const float &val, LibUtilities::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);

// copy template specialization.
template void MathHelper::copy<LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<double> &x,
    LibUtilities::MemoryRegion<double> &y, const std::string &execSpace);
template void MathHelper::copy<LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<float> &x, LibUtilities::MemoryRegion<float> &y,
    const std::string &execSpace);
template void MathHelper::copy<LibUtilities::Field<double, FieldState::Phys>>(
    LibUtilities::Field<double, FieldState::Phys> &x,
    LibUtilities::Field<double, FieldState::Phys> &y,
    const std::string &execSpace);
template void MathHelper::copy<LibUtilities::Field<float, FieldState::Phys>>(
    LibUtilities::Field<float, FieldState::Phys> &x,
    LibUtilities::Field<float, FieldState::Phys> &y,
    const std::string &execSpace);
template void MathHelper::copy<LibUtilities::Field<double, FieldState::Coeff>>(
    LibUtilities::Field<double, FieldState::Coeff> &x,
    LibUtilities::Field<double, FieldState::Coeff> &y,
    const std::string &execSpace);
template void MathHelper::copy<LibUtilities::Field<float, FieldState::Coeff>>(
    LibUtilities::Field<float, FieldState::Coeff> &x,
    LibUtilities::Field<float, FieldState::Coeff> &y,
    const std::string &execSpace);

// abs template specialization.
template void MathHelper::abs<LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<double> &x,
    LibUtilities::MemoryRegion<double> &y, const std::string &execSpace);
template void MathHelper::abs<LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<float> &x, LibUtilities::MemoryRegion<float> &y,
    const std::string &execSpace);
template void MathHelper::abs<LibUtilities::Field<double, FieldState::Phys>>(
    LibUtilities::Field<double, FieldState::Phys> &x,
    LibUtilities::Field<double, FieldState::Phys> &y,
    const std::string &execSpace);
template void MathHelper::abs<LibUtilities::Field<float, FieldState::Phys>>(
    LibUtilities::Field<float, FieldState::Phys> &x,
    LibUtilities::Field<float, FieldState::Phys> &y,
    const std::string &execSpace);
template void MathHelper::abs<LibUtilities::Field<double, FieldState::Coeff>>(
    LibUtilities::Field<double, FieldState::Coeff> &x,
    LibUtilities::Field<double, FieldState::Coeff> &y,
    const std::string &execSpace);
template void MathHelper::abs<LibUtilities::Field<float, FieldState::Coeff>>(
    LibUtilities::Field<float, FieldState::Coeff> &x,
    LibUtilities::Field<float, FieldState::Coeff> &y,
    const std::string &execSpace);

// neg template specialization.
template void MathHelper::neg<LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<double> &x,
    LibUtilities::MemoryRegion<double> &y, const std::string &execSpace);
template void MathHelper::neg<LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<float> &x, LibUtilities::MemoryRegion<float> &y,
    const std::string &execSpace);
template void MathHelper::neg<LibUtilities::Field<double, FieldState::Phys>>(
    LibUtilities::Field<double, FieldState::Phys> &x,
    LibUtilities::Field<double, FieldState::Phys> &y,
    const std::string &execSpace);
template void MathHelper::neg<LibUtilities::Field<float, FieldState::Phys>>(
    LibUtilities::Field<float, FieldState::Phys> &x,
    LibUtilities::Field<float, FieldState::Phys> &y,
    const std::string &execSpace);
template void MathHelper::neg<LibUtilities::Field<double, FieldState::Coeff>>(
    LibUtilities::Field<double, FieldState::Coeff> &x,
    LibUtilities::Field<double, FieldState::Coeff> &y,
    const std::string &execSpace);
template void MathHelper::neg<LibUtilities::Field<float, FieldState::Coeff>>(
    LibUtilities::Field<float, FieldState::Coeff> &x,
    LibUtilities::Field<float, FieldState::Coeff> &y,
    const std::string &execSpace);

// sqrt template specialization.
template void MathHelper::sqrt<LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<double> &x,
    LibUtilities::MemoryRegion<double> &y, const std::string &execSpace);
template void MathHelper::sqrt<LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<float> &x, LibUtilities::MemoryRegion<float> &y,
    const std::string &execSpace);
template void MathHelper::sqrt<LibUtilities::Field<double, FieldState::Phys>>(
    LibUtilities::Field<double, FieldState::Phys> &x,
    LibUtilities::Field<double, FieldState::Phys> &y,
    const std::string &execSpace);
template void MathHelper::sqrt<LibUtilities::Field<float, FieldState::Phys>>(
    LibUtilities::Field<float, FieldState::Phys> &x,
    LibUtilities::Field<float, FieldState::Phys> &y,
    const std::string &execSpace);
template void MathHelper::sqrt<LibUtilities::Field<double, FieldState::Coeff>>(
    LibUtilities::Field<double, FieldState::Coeff> &x,
    LibUtilities::Field<double, FieldState::Coeff> &y,
    const std::string &execSpace);
template void MathHelper::sqrt<LibUtilities::Field<float, FieldState::Coeff>>(
    LibUtilities::Field<float, FieldState::Coeff> &x,
    LibUtilities::Field<float, FieldState::Coeff> &y,
    const std::string &execSpace);

// add template specialization.
template void MathHelper::add<LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<double> &x,
    LibUtilities::MemoryRegion<double> &y,
    LibUtilities::MemoryRegion<double> &z, const std::string &execSpace);
template void MathHelper::add<LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<float> &x, LibUtilities::MemoryRegion<float> &y,
    LibUtilities::MemoryRegion<float> &z, const std::string &execSpace);
template void MathHelper::add<LibUtilities::Field<double, FieldState::Phys>>(
    LibUtilities::Field<double, FieldState::Phys> &x,
    LibUtilities::Field<double, FieldState::Phys> &y,
    LibUtilities::Field<double, FieldState::Phys> &z,
    const std::string &execSpace);
template void MathHelper::add<LibUtilities::Field<float, FieldState::Phys>>(
    LibUtilities::Field<float, FieldState::Phys> &x,
    LibUtilities::Field<float, FieldState::Phys> &y,
    LibUtilities::Field<float, FieldState::Phys> &z,
    const std::string &execSpace);
template void MathHelper::add<LibUtilities::Field<double, FieldState::Coeff>>(
    LibUtilities::Field<double, FieldState::Coeff> &x,
    LibUtilities::Field<double, FieldState::Coeff> &y,
    LibUtilities::Field<double, FieldState::Coeff> &z,
    const std::string &execSpace);
template void MathHelper::add<LibUtilities::Field<float, FieldState::Coeff>>(
    LibUtilities::Field<float, FieldState::Coeff> &x,
    LibUtilities::Field<float, FieldState::Coeff> &y,
    LibUtilities::Field<float, FieldState::Coeff> &z,
    const std::string &execSpace);

// sub template specialization.
template void MathHelper::sub<LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<double> &x,
    LibUtilities::MemoryRegion<double> &y,
    LibUtilities::MemoryRegion<double> &z, const std::string &execSpace);
template void MathHelper::sub<LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<float> &x, LibUtilities::MemoryRegion<float> &y,
    LibUtilities::MemoryRegion<float> &z, const std::string &execSpace);
template void MathHelper::sub<LibUtilities::Field<double, FieldState::Phys>>(
    LibUtilities::Field<double, FieldState::Phys> &x,
    LibUtilities::Field<double, FieldState::Phys> &y,
    LibUtilities::Field<double, FieldState::Phys> &z,
    const std::string &execSpace);
template void MathHelper::sub<LibUtilities::Field<float, FieldState::Phys>>(
    LibUtilities::Field<float, FieldState::Phys> &x,
    LibUtilities::Field<float, FieldState::Phys> &y,
    LibUtilities::Field<float, FieldState::Phys> &z,
    const std::string &execSpace);
template void MathHelper::sub<LibUtilities::Field<double, FieldState::Coeff>>(
    LibUtilities::Field<double, FieldState::Coeff> &x,
    LibUtilities::Field<double, FieldState::Coeff> &y,
    LibUtilities::Field<double, FieldState::Coeff> &z,
    const std::string &execSpace);
template void MathHelper::sub<LibUtilities::Field<float, FieldState::Coeff>>(
    LibUtilities::Field<float, FieldState::Coeff> &x,
    LibUtilities::Field<float, FieldState::Coeff> &y,
    LibUtilities::Field<float, FieldState::Coeff> &z,
    const std::string &execSpace);

// mul template specialization.
template void MathHelper::mul<LibUtilities::MemoryRegion<double>>(
    const double alpha, LibUtilities::MemoryRegion<double> &x,
    LibUtilities::MemoryRegion<double> &y, const std::string &execSpace);
template void MathHelper::mul<LibUtilities::MemoryRegion<float>>(
    const float alpha, LibUtilities::MemoryRegion<float> &x,
    LibUtilities::MemoryRegion<float> &y, const std::string &execSpace);
template void MathHelper::mul<LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<double> &x,
    LibUtilities::MemoryRegion<double> &y,
    LibUtilities::MemoryRegion<double> &z, const std::string &execSpace);
template void MathHelper::mul<LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<float> &x, LibUtilities::MemoryRegion<float> &y,
    LibUtilities::MemoryRegion<float> &z, const std::string &execSpace);
template void MathHelper::mul<LibUtilities::Field<double, FieldState::Phys>>(
    const double alpha, LibUtilities::Field<double, FieldState::Phys> &x,
    LibUtilities::Field<double, FieldState::Phys> &y,
    const std::string &execSpace);
template void MathHelper::mul<LibUtilities::Field<float, FieldState::Phys>>(
    const float alpha, LibUtilities::Field<float, FieldState::Phys> &x,
    LibUtilities::Field<float, FieldState::Phys> &y,
    const std::string &execSpace);
template void MathHelper::mul<LibUtilities::Field<double, FieldState::Coeff>>(
    const double alpha, LibUtilities::Field<double, FieldState::Coeff> &x,
    LibUtilities::Field<double, FieldState::Coeff> &y,
    const std::string &execSpace);
template void MathHelper::mul<LibUtilities::Field<float, FieldState::Coeff>>(
    const float alpha, LibUtilities::Field<float, FieldState::Coeff> &x,
    LibUtilities::Field<float, FieldState::Coeff> &y,
    const std::string &execSpace);
template void MathHelper::mul<LibUtilities::Field<double, FieldState::Phys>>(
    LibUtilities::Field<double, FieldState::Phys> &x,
    LibUtilities::Field<double, FieldState::Phys> &y,
    LibUtilities::Field<double, FieldState::Phys> &z,
    const std::string &execSpace);
template void MathHelper::mul<LibUtilities::Field<float, FieldState::Phys>>(
    LibUtilities::Field<float, FieldState::Phys> &x,
    LibUtilities::Field<float, FieldState::Phys> &y,
    LibUtilities::Field<float, FieldState::Phys> &z,
    const std::string &execSpace);
template void MathHelper::mul<LibUtilities::Field<double, FieldState::Coeff>>(
    LibUtilities::Field<double, FieldState::Coeff> &x,
    LibUtilities::Field<double, FieldState::Coeff> &y,
    LibUtilities::Field<double, FieldState::Coeff> &z,
    const std::string &execSpace);
template void MathHelper::mul<LibUtilities::Field<float, FieldState::Coeff>>(
    LibUtilities::Field<float, FieldState::Coeff> &x,
    LibUtilities::Field<float, FieldState::Coeff> &y,
    LibUtilities::Field<float, FieldState::Coeff> &z,
    const std::string &execSpace);

// div template specialization.
template void MathHelper::div<LibUtilities::MemoryRegion<double>>(
    const double alpha, LibUtilities::MemoryRegion<double> &x,
    LibUtilities::MemoryRegion<double> &y, const std::string &execSpace);
template void MathHelper::div<LibUtilities::MemoryRegion<float>>(
    const float alpha, LibUtilities::MemoryRegion<float> &x,
    LibUtilities::MemoryRegion<float> &y, const std::string &execSpace);
template void MathHelper::div<LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<double> &x,
    LibUtilities::MemoryRegion<double> &y,
    LibUtilities::MemoryRegion<double> &z, const std::string &execSpace);
template void MathHelper::div<LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<float> &x, LibUtilities::MemoryRegion<float> &y,
    LibUtilities::MemoryRegion<float> &z, const std::string &execSpace);
template void MathHelper::div<LibUtilities::Field<double, FieldState::Phys>>(
    const double alpha, LibUtilities::Field<double, FieldState::Phys> &x,
    LibUtilities::Field<double, FieldState::Phys> &y,
    const std::string &execSpace);
template void MathHelper::div<LibUtilities::Field<float, FieldState::Phys>>(
    const float alpha, LibUtilities::Field<float, FieldState::Phys> &x,
    LibUtilities::Field<float, FieldState::Phys> &y,
    const std::string &execSpace);
template void MathHelper::div<LibUtilities::Field<double, FieldState::Coeff>>(
    const double alpha, LibUtilities::Field<double, FieldState::Coeff> &x,
    LibUtilities::Field<double, FieldState::Coeff> &y,
    const std::string &execSpace);
template void MathHelper::div<LibUtilities::Field<float, FieldState::Coeff>>(
    const float alpha, LibUtilities::Field<float, FieldState::Coeff> &x,
    LibUtilities::Field<float, FieldState::Coeff> &y,
    const std::string &execSpace);
template void MathHelper::div<LibUtilities::Field<double, FieldState::Phys>>(
    LibUtilities::Field<double, FieldState::Phys> &x,
    LibUtilities::Field<double, FieldState::Phys> &y,
    LibUtilities::Field<double, FieldState::Phys> &z,
    const std::string &execSpace);
template void MathHelper::div<LibUtilities::Field<float, FieldState::Phys>>(
    LibUtilities::Field<float, FieldState::Phys> &x,
    LibUtilities::Field<float, FieldState::Phys> &y,
    LibUtilities::Field<float, FieldState::Phys> &z,
    const std::string &execSpace);
template void MathHelper::div<LibUtilities::Field<double, FieldState::Coeff>>(
    LibUtilities::Field<double, FieldState::Coeff> &x,
    LibUtilities::Field<double, FieldState::Coeff> &y,
    LibUtilities::Field<double, FieldState::Coeff> &z,
    const std::string &execSpace);
template void MathHelper::div<LibUtilities::Field<float, FieldState::Coeff>>(
    LibUtilities::Field<float, FieldState::Coeff> &x,
    LibUtilities::Field<float, FieldState::Coeff> &y,
    LibUtilities::Field<float, FieldState::Coeff> &z,
    const std::string &execSpace);

// daxpy template specialization.
template void MathHelper::daxpy<LibUtilities::MemoryRegion<double>>(
    const double alpha, LibUtilities::MemoryRegion<double> &x,
    LibUtilities::MemoryRegion<double> &y,
    LibUtilities::MemoryRegion<double> &z, const std::string &execSpace);
template void MathHelper::daxpy<LibUtilities::MemoryRegion<float>>(
    const float alpha, LibUtilities::MemoryRegion<float> &x,
    LibUtilities::MemoryRegion<float> &y, LibUtilities::MemoryRegion<float> &z,
    const std::string &execSpace);
template void MathHelper::daxpy<LibUtilities::Field<double, FieldState::Phys>>(
    const double alpha, LibUtilities::Field<double, FieldState::Phys> &x,
    LibUtilities::Field<double, FieldState::Phys> &y,
    LibUtilities::Field<double, FieldState::Phys> &z,
    const std::string &execSpace);
template void MathHelper::daxpy<LibUtilities::Field<float, FieldState::Phys>>(
    const float alpha, LibUtilities::Field<float, FieldState::Phys> &x,
    LibUtilities::Field<float, FieldState::Phys> &y,
    LibUtilities::Field<float, FieldState::Phys> &z,
    const std::string &execSpace);
template void MathHelper::daxpy<LibUtilities::Field<double, FieldState::Coeff>>(
    const double alpha, LibUtilities::Field<double, FieldState::Coeff> &x,
    LibUtilities::Field<double, FieldState::Coeff> &y,
    LibUtilities::Field<double, FieldState::Coeff> &z,
    const std::string &execSpace);
template void MathHelper::daxpy<LibUtilities::Field<float, FieldState::Coeff>>(
    const float alpha, LibUtilities::Field<float, FieldState::Coeff> &x,
    LibUtilities::Field<float, FieldState::Coeff> &y,
    LibUtilities::Field<float, FieldState::Coeff> &z,
    const std::string &execSpace);

// reduceSum template specialization.
template double MathHelper::reduceSum<LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<double> &x, const std::string &execSpace);
template float MathHelper::reduceSum<LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<float> &x, const std::string &execSpace);
template double MathHelper::reduceSum<LibUtilities::MemoryRegion<uint8_t>,
                                      LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<uint8_t> &mask,
    LibUtilities::MemoryRegion<double> &x, const std::string &execSpace);
template float MathHelper::reduceSum<LibUtilities::MemoryRegion<uint8_t>,
                                     LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<uint8_t> &mask,
    LibUtilities::MemoryRegion<float> &x, const std::string &execSpace);
template double MathHelper::reduceSum<
    LibUtilities::Field<double, FieldState::Phys>>(
    LibUtilities::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float MathHelper::reduceSum<
    LibUtilities::Field<float, FieldState::Phys>>(
    LibUtilities::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double MathHelper::reduceSum<
    LibUtilities::Field<double, FieldState::Coeff>>(
    LibUtilities::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float MathHelper::reduceSum<
    LibUtilities::Field<float, FieldState::Coeff>>(
    LibUtilities::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double MathHelper::reduceSum<
    LibUtilities::Field<uint8_t, FieldState::Phys>,
    LibUtilities::Field<double, FieldState::Phys>>(
    LibUtilities::Field<uint8_t, FieldState::Phys> &mask,
    LibUtilities::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float MathHelper::reduceSum<
    LibUtilities::Field<uint8_t, FieldState::Phys>,
    LibUtilities::Field<float, FieldState::Phys>>(
    LibUtilities::Field<uint8_t, FieldState::Phys> &mask,
    LibUtilities::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double MathHelper::reduceSum<
    LibUtilities::Field<uint8_t, FieldState::Coeff>,
    LibUtilities::Field<double, FieldState::Coeff>>(
    LibUtilities::Field<uint8_t, FieldState::Coeff> &mask,
    LibUtilities::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float MathHelper::reduceSum<
    LibUtilities::Field<uint8_t, FieldState::Coeff>,
    LibUtilities::Field<float, FieldState::Coeff>>(
    LibUtilities::Field<uint8_t, FieldState::Coeff> &mask,
    LibUtilities::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);

// reduceMax template specialization.
template double MathHelper::reduceMax<LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<double> &x, const std::string &execSpace);
template float MathHelper::reduceMax<LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<float> &x, const std::string &execSpace);
template double MathHelper::reduceMax<LibUtilities::MemoryRegion<uint8_t>,
                                      LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<uint8_t> &mask,
    LibUtilities::MemoryRegion<double> &x, const std::string &execSpace);
template float MathHelper::reduceMax<LibUtilities::MemoryRegion<uint8_t>,
                                     LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<uint8_t> &mask,
    LibUtilities::MemoryRegion<float> &x, const std::string &execSpace);
template double MathHelper::reduceMax<
    LibUtilities::Field<double, FieldState::Phys>>(
    LibUtilities::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float MathHelper::reduceMax<
    LibUtilities::Field<float, FieldState::Phys>>(
    LibUtilities::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double MathHelper::reduceMax<
    LibUtilities::Field<double, FieldState::Coeff>>(
    LibUtilities::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float MathHelper::reduceMax<
    LibUtilities::Field<float, FieldState::Coeff>>(
    LibUtilities::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double MathHelper::reduceMax<
    LibUtilities::Field<uint8_t, FieldState::Phys>,
    LibUtilities::Field<double, FieldState::Phys>>(
    LibUtilities::Field<uint8_t, FieldState::Phys> &mask,
    LibUtilities::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float MathHelper::reduceMax<
    LibUtilities::Field<uint8_t, FieldState::Phys>,
    LibUtilities::Field<float, FieldState::Phys>>(
    LibUtilities::Field<uint8_t, FieldState::Phys> &mask,
    LibUtilities::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double MathHelper::reduceMax<
    LibUtilities::Field<uint8_t, FieldState::Coeff>,
    LibUtilities::Field<double, FieldState::Coeff>>(
    LibUtilities::Field<uint8_t, FieldState::Coeff> &mask,
    LibUtilities::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float MathHelper::reduceMax<
    LibUtilities::Field<uint8_t, FieldState::Coeff>,
    LibUtilities::Field<float, FieldState::Coeff>>(
    LibUtilities::Field<uint8_t, FieldState::Coeff> &mask,
    LibUtilities::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);

// reduceMin template specialization.
template double MathHelper::reduceMin<LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<double> &x, const std::string &execSpace);
template float MathHelper::reduceMin<LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<float> &x, const std::string &execSpace);
template double MathHelper::reduceMin<LibUtilities::MemoryRegion<uint8_t>,
                                      LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<uint8_t> &mask,
    LibUtilities::MemoryRegion<double> &x, const std::string &execSpace);
template float MathHelper::reduceMin<LibUtilities::MemoryRegion<uint8_t>,
                                     LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<uint8_t> &mask,
    LibUtilities::MemoryRegion<float> &x, const std::string &execSpace);
template double MathHelper::reduceMin<
    LibUtilities::Field<double, FieldState::Phys>>(
    LibUtilities::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float MathHelper::reduceMin<
    LibUtilities::Field<float, FieldState::Phys>>(
    LibUtilities::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double MathHelper::reduceMin<
    LibUtilities::Field<double, FieldState::Coeff>>(
    LibUtilities::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float MathHelper::reduceMin<
    LibUtilities::Field<float, FieldState::Coeff>>(
    LibUtilities::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double MathHelper::reduceMin<
    LibUtilities::Field<uint8_t, FieldState::Phys>,
    LibUtilities::Field<double, FieldState::Phys>>(
    LibUtilities::Field<uint8_t, FieldState::Phys> &mask,
    LibUtilities::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float MathHelper::reduceMin<
    LibUtilities::Field<uint8_t, FieldState::Phys>,
    LibUtilities::Field<float, FieldState::Phys>>(
    LibUtilities::Field<uint8_t, FieldState::Phys> &mask,
    LibUtilities::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double MathHelper::reduceMin<
    LibUtilities::Field<uint8_t, FieldState::Coeff>,
    LibUtilities::Field<double, FieldState::Coeff>>(
    LibUtilities::Field<uint8_t, FieldState::Coeff> &mask,
    LibUtilities::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float MathHelper::reduceMin<
    LibUtilities::Field<uint8_t, FieldState::Coeff>,
    LibUtilities::Field<float, FieldState::Coeff>>(
    LibUtilities::Field<uint8_t, FieldState::Coeff> &mask,
    LibUtilities::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);

// ddot template specialization.
template double MathHelper::ddot<LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<double> &x,
    LibUtilities::MemoryRegion<double> &y, const std::string &execSpace);
template float MathHelper::ddot<LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<float> &x, LibUtilities::MemoryRegion<float> &y,
    const std::string &execSpace);
template double MathHelper::ddot<LibUtilities::MemoryRegion<uint8_t>,
                                 LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<uint8_t> &mask,
    LibUtilities::MemoryRegion<double> &x,
    LibUtilities::MemoryRegion<double> &y, const std::string &execSpace);
template float MathHelper::ddot<LibUtilities::MemoryRegion<uint8_t>,
                                LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<uint8_t> &mask,
    LibUtilities::MemoryRegion<float> &x, LibUtilities::MemoryRegion<float> &y,
    const std::string &execSpace);
template double MathHelper::ddot<LibUtilities::Field<double, FieldState::Phys>>(
    LibUtilities::Field<double, FieldState::Phys> &x,
    LibUtilities::Field<double, FieldState::Phys> &y,
    const std::string &execSpace);
template float MathHelper::ddot<LibUtilities::Field<float, FieldState::Phys>>(
    LibUtilities::Field<float, FieldState::Phys> &x,
    LibUtilities::Field<float, FieldState::Phys> &y,
    const std::string &execSpace);
template double MathHelper::ddot<
    LibUtilities::Field<double, FieldState::Coeff>>(
    LibUtilities::Field<double, FieldState::Coeff> &x,
    LibUtilities::Field<double, FieldState::Coeff> &y,
    const std::string &execSpace);
template float MathHelper::ddot<LibUtilities::Field<float, FieldState::Coeff>>(
    LibUtilities::Field<float, FieldState::Coeff> &x,
    LibUtilities::Field<float, FieldState::Coeff> &y,
    const std::string &execSpace);
template double MathHelper::ddot<LibUtilities::Field<uint8_t, FieldState::Phys>,
                                 LibUtilities::Field<double, FieldState::Phys>>(
    LibUtilities::Field<uint8_t, FieldState::Phys> &mask,
    LibUtilities::Field<double, FieldState::Phys> &x,
    LibUtilities::Field<double, FieldState::Phys> &y,
    const std::string &execSpace);
template float MathHelper::ddot<LibUtilities::Field<uint8_t, FieldState::Phys>,
                                LibUtilities::Field<float, FieldState::Phys>>(
    LibUtilities::Field<uint8_t, FieldState::Phys> &mask,
    LibUtilities::Field<float, FieldState::Phys> &x,
    LibUtilities::Field<float, FieldState::Phys> &y,
    const std::string &execSpace);
template double MathHelper::ddot<
    LibUtilities::Field<uint8_t, FieldState::Coeff>,
    LibUtilities::Field<double, FieldState::Coeff>>(
    LibUtilities::Field<uint8_t, FieldState::Coeff> &mask,
    LibUtilities::Field<double, FieldState::Coeff> &x,
    LibUtilities::Field<double, FieldState::Coeff> &y,
    const std::string &execSpace);
template float MathHelper::ddot<LibUtilities::Field<uint8_t, FieldState::Coeff>,
                                LibUtilities::Field<float, FieldState::Coeff>>(
    LibUtilities::Field<uint8_t, FieldState::Coeff> &mask,
    LibUtilities::Field<float, FieldState::Coeff> &x,
    LibUtilities::Field<float, FieldState::Coeff> &y,
    const std::string &execSpace);

// l1norm template specialization.
template double MathHelper::l1norm<LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<double> &x, const std::string &execSpace);
template float MathHelper::l1norm<LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<float> &x, const std::string &execSpace);
template double MathHelper::l1norm<LibUtilities::MemoryRegion<uint8_t>,
                                   LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<uint8_t> &mask,
    LibUtilities::MemoryRegion<double> &x, const std::string &execSpace);
template float MathHelper::l1norm<LibUtilities::MemoryRegion<uint8_t>,
                                  LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<uint8_t> &mask,
    LibUtilities::MemoryRegion<float> &x, const std::string &execSpace);
template double MathHelper::l1norm<
    LibUtilities::Field<double, FieldState::Phys>>(
    LibUtilities::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float MathHelper::l1norm<LibUtilities::Field<float, FieldState::Phys>>(
    LibUtilities::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double MathHelper::l1norm<
    LibUtilities::Field<double, FieldState::Coeff>>(
    LibUtilities::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float MathHelper::l1norm<
    LibUtilities::Field<float, FieldState::Coeff>>(
    LibUtilities::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double MathHelper::l1norm<
    LibUtilities::Field<uint8_t, FieldState::Phys>,
    LibUtilities::Field<double, FieldState::Phys>>(
    LibUtilities::Field<uint8_t, FieldState::Phys> &mask,
    LibUtilities::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float MathHelper::l1norm<
    LibUtilities::Field<uint8_t, FieldState::Phys>,
    LibUtilities::Field<float, FieldState::Phys>>(
    LibUtilities::Field<uint8_t, FieldState::Phys> &mask,
    LibUtilities::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double MathHelper::l1norm<
    LibUtilities::Field<uint8_t, FieldState::Coeff>,
    LibUtilities::Field<double, FieldState::Coeff>>(
    LibUtilities::Field<uint8_t, FieldState::Coeff> &mask,
    LibUtilities::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float MathHelper::l1norm<
    LibUtilities::Field<uint8_t, FieldState::Coeff>,
    LibUtilities::Field<float, FieldState::Coeff>>(
    LibUtilities::Field<uint8_t, FieldState::Coeff> &mask,
    LibUtilities::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);

// l2norm template specialization.
template double MathHelper::l2norm<LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<double> &x, const std::string &execSpace);
template float MathHelper::l2norm<LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<float> &x, const std::string &execSpace);
template double MathHelper::l2norm<LibUtilities::MemoryRegion<uint8_t>,
                                   LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<uint8_t> &mask,
    LibUtilities::MemoryRegion<double> &x, const std::string &execSpace);
template float MathHelper::l2norm<LibUtilities::MemoryRegion<uint8_t>,
                                  LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<uint8_t> &mask,
    LibUtilities::MemoryRegion<float> &x, const std::string &execSpace);
template double MathHelper::l2norm<
    LibUtilities::Field<double, FieldState::Phys>>(
    LibUtilities::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float MathHelper::l2norm<LibUtilities::Field<float, FieldState::Phys>>(
    LibUtilities::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double MathHelper::l2norm<
    LibUtilities::Field<double, FieldState::Coeff>>(
    LibUtilities::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float MathHelper::l2norm<
    LibUtilities::Field<float, FieldState::Coeff>>(
    LibUtilities::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double MathHelper::l2norm<
    LibUtilities::Field<uint8_t, FieldState::Phys>,
    LibUtilities::Field<double, FieldState::Phys>>(
    LibUtilities::Field<uint8_t, FieldState::Phys> &mask,
    LibUtilities::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float MathHelper::l2norm<
    LibUtilities::Field<uint8_t, FieldState::Phys>,
    LibUtilities::Field<float, FieldState::Phys>>(
    LibUtilities::Field<uint8_t, FieldState::Phys> &mask,
    LibUtilities::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double MathHelper::l2norm<
    LibUtilities::Field<uint8_t, FieldState::Coeff>,
    LibUtilities::Field<double, FieldState::Coeff>>(
    LibUtilities::Field<uint8_t, FieldState::Coeff> &mask,
    LibUtilities::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float MathHelper::l2norm<
    LibUtilities::Field<uint8_t, FieldState::Coeff>,
    LibUtilities::Field<float, FieldState::Coeff>>(
    LibUtilities::Field<uint8_t, FieldState::Coeff> &mask,
    LibUtilities::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);

// lpnorm template specialization.
template double MathHelper::lpnorm<LibUtilities::MemoryRegion<double>>(
    const unsigned int p, LibUtilities::MemoryRegion<double> &x,
    const std::string &execSpace);
template float MathHelper::lpnorm<LibUtilities::MemoryRegion<float>>(
    const unsigned int p, LibUtilities::MemoryRegion<float> &x,
    const std::string &execSpace);
template double MathHelper::lpnorm<LibUtilities::MemoryRegion<uint8_t>,
                                   LibUtilities::MemoryRegion<double>>(
    const unsigned int p, LibUtilities::MemoryRegion<uint8_t> &mask,
    LibUtilities::MemoryRegion<double> &x, const std::string &execSpace);
template float MathHelper::lpnorm<LibUtilities::MemoryRegion<uint8_t>,
                                  LibUtilities::MemoryRegion<float>>(
    const unsigned int p, LibUtilities::MemoryRegion<uint8_t> &mask,
    LibUtilities::MemoryRegion<float> &x, const std::string &execSpace);
template double MathHelper::lpnorm<
    LibUtilities::Field<double, FieldState::Phys>>(
    const unsigned int p, LibUtilities::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float MathHelper::lpnorm<LibUtilities::Field<float, FieldState::Phys>>(
    const unsigned int p, LibUtilities::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double MathHelper::lpnorm<
    LibUtilities::Field<double, FieldState::Coeff>>(
    const unsigned int p, LibUtilities::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float MathHelper::lpnorm<
    LibUtilities::Field<float, FieldState::Coeff>>(
    const unsigned int p, LibUtilities::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double MathHelper::lpnorm<
    LibUtilities::Field<uint8_t, FieldState::Phys>,
    LibUtilities::Field<double, FieldState::Phys>>(
    const unsigned int p, LibUtilities::Field<uint8_t, FieldState::Phys> &mask,
    LibUtilities::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float MathHelper::lpnorm<
    LibUtilities::Field<uint8_t, FieldState::Phys>,
    LibUtilities::Field<float, FieldState::Phys>>(
    const unsigned int p, LibUtilities::Field<uint8_t, FieldState::Phys> &mask,
    LibUtilities::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double MathHelper::lpnorm<
    LibUtilities::Field<uint8_t, FieldState::Coeff>,
    LibUtilities::Field<double, FieldState::Coeff>>(
    const unsigned int p, LibUtilities::Field<uint8_t, FieldState::Coeff> &mask,
    LibUtilities::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float MathHelper::lpnorm<
    LibUtilities::Field<uint8_t, FieldState::Coeff>,
    LibUtilities::Field<float, FieldState::Coeff>>(
    const unsigned int p, LibUtilities::Field<uint8_t, FieldState::Coeff> &mask,
    LibUtilities::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);

// linfnorm template specialization.
template double MathHelper::linfnorm<LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<double> &x, const std::string &execSpace);
template float MathHelper::linfnorm<LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<float> &x, const std::string &execSpace);
template double MathHelper::linfnorm<LibUtilities::MemoryRegion<uint8_t>,
                                     LibUtilities::MemoryRegion<double>>(
    LibUtilities::MemoryRegion<uint8_t> &mask,
    LibUtilities::MemoryRegion<double> &x, const std::string &execSpace);
template float MathHelper::linfnorm<LibUtilities::MemoryRegion<uint8_t>,
                                    LibUtilities::MemoryRegion<float>>(
    LibUtilities::MemoryRegion<uint8_t> &mask,
    LibUtilities::MemoryRegion<float> &x, const std::string &execSpace);
template double MathHelper::linfnorm<
    LibUtilities::Field<double, FieldState::Phys>>(
    LibUtilities::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float MathHelper::linfnorm<
    LibUtilities::Field<float, FieldState::Phys>>(
    LibUtilities::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double MathHelper::linfnorm<
    LibUtilities::Field<double, FieldState::Coeff>>(
    LibUtilities::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float MathHelper::linfnorm<
    LibUtilities::Field<float, FieldState::Coeff>>(
    LibUtilities::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double MathHelper::linfnorm<
    LibUtilities::Field<uint8_t, FieldState::Phys>,
    LibUtilities::Field<double, FieldState::Phys>>(
    LibUtilities::Field<uint8_t, FieldState::Phys> &mask,
    LibUtilities::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float MathHelper::linfnorm<
    LibUtilities::Field<uint8_t, FieldState::Phys>,
    LibUtilities::Field<float, FieldState::Phys>>(
    LibUtilities::Field<uint8_t, FieldState::Phys> &mask,
    LibUtilities::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double MathHelper::linfnorm<
    LibUtilities::Field<uint8_t, FieldState::Coeff>,
    LibUtilities::Field<double, FieldState::Coeff>>(
    LibUtilities::Field<uint8_t, FieldState::Coeff> &mask,
    LibUtilities::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float MathHelper::linfnorm<
    LibUtilities::Field<uint8_t, FieldState::Coeff>,
    LibUtilities::Field<float, FieldState::Coeff>>(
    LibUtilities::Field<uint8_t, FieldState::Coeff> &mask,
    LibUtilities::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);

} // namespace Nektar::Math
