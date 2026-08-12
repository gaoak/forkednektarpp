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

#include "MultiRegions/Field/Math.hpp"

#include "LibUtilities/BasicUtils/Math/MathHelperDef.hpp"

// MathHelper explicit instantiation for Field based functions. Provide
// functionalities to dynamically select the Execution space by using a string
// parameter.
namespace Nektar::Math
{

// zero template specialization.
template void MathHelper::zero<MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template void MathHelper::zero<MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template void MathHelper::zero<MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template void MathHelper::zero<MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);

// fill template specialization.
template void MathHelper::fill<double,
                               MultiRegions::Field<double, FieldState::Phys>>(
    const double &val, MultiRegions::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template void MathHelper::fill<float,
                               MultiRegions::Field<float, FieldState::Phys>>(
    const float &val, MultiRegions::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template void MathHelper::fill<double,
                               MultiRegions::Field<double, FieldState::Coeff>>(
    const double &val, MultiRegions::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template void MathHelper::fill<float,
                               MultiRegions::Field<float, FieldState::Coeff>>(
    const float &val, MultiRegions::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);

// copy template specialization.
template void MathHelper::copy<MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<double, FieldState::Phys> &x,
    MultiRegions::Field<double, FieldState::Phys> &y,
    const std::string &execSpace);
template void MathHelper::copy<MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<float, FieldState::Phys> &x,
    MultiRegions::Field<float, FieldState::Phys> &y,
    const std::string &execSpace);
template void MathHelper::copy<MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<double, FieldState::Coeff> &x,
    MultiRegions::Field<double, FieldState::Coeff> &y,
    const std::string &execSpace);
template void MathHelper::copy<MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<float, FieldState::Coeff> &x,
    MultiRegions::Field<float, FieldState::Coeff> &y,
    const std::string &execSpace);

// abs template specialization.
template void MathHelper::abs<MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<double, FieldState::Phys> &x,
    MultiRegions::Field<double, FieldState::Phys> &y,
    const std::string &execSpace);
template void MathHelper::abs<MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<float, FieldState::Phys> &x,
    MultiRegions::Field<float, FieldState::Phys> &y,
    const std::string &execSpace);
template void MathHelper::abs<MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<double, FieldState::Coeff> &x,
    MultiRegions::Field<double, FieldState::Coeff> &y,
    const std::string &execSpace);
template void MathHelper::abs<MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<float, FieldState::Coeff> &x,
    MultiRegions::Field<float, FieldState::Coeff> &y,
    const std::string &execSpace);

// neg template specialization.
template void MathHelper::neg<MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<double, FieldState::Phys> &x,
    MultiRegions::Field<double, FieldState::Phys> &y,
    const std::string &execSpace);
template void MathHelper::neg<MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<float, FieldState::Phys> &x,
    MultiRegions::Field<float, FieldState::Phys> &y,
    const std::string &execSpace);
template void MathHelper::neg<MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<double, FieldState::Coeff> &x,
    MultiRegions::Field<double, FieldState::Coeff> &y,
    const std::string &execSpace);
template void MathHelper::neg<MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<float, FieldState::Coeff> &x,
    MultiRegions::Field<float, FieldState::Coeff> &y,
    const std::string &execSpace);

// sqrt template specialization.
template void MathHelper::sqrt<MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<double, FieldState::Phys> &x,
    MultiRegions::Field<double, FieldState::Phys> &y,
    const std::string &execSpace);
template void MathHelper::sqrt<MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<float, FieldState::Phys> &x,
    MultiRegions::Field<float, FieldState::Phys> &y,
    const std::string &execSpace);
template void MathHelper::sqrt<MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<double, FieldState::Coeff> &x,
    MultiRegions::Field<double, FieldState::Coeff> &y,
    const std::string &execSpace);
template void MathHelper::sqrt<MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<float, FieldState::Coeff> &x,
    MultiRegions::Field<float, FieldState::Coeff> &y,
    const std::string &execSpace);

// add template specialization.
template void MathHelper::add<MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<double, FieldState::Phys> &x,
    MultiRegions::Field<double, FieldState::Phys> &y,
    MultiRegions::Field<double, FieldState::Phys> &z,
    const std::string &execSpace);
template void MathHelper::add<MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<float, FieldState::Phys> &x,
    MultiRegions::Field<float, FieldState::Phys> &y,
    MultiRegions::Field<float, FieldState::Phys> &z,
    const std::string &execSpace);
template void MathHelper::add<MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<double, FieldState::Coeff> &x,
    MultiRegions::Field<double, FieldState::Coeff> &y,
    MultiRegions::Field<double, FieldState::Coeff> &z,
    const std::string &execSpace);
template void MathHelper::add<MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<float, FieldState::Coeff> &x,
    MultiRegions::Field<float, FieldState::Coeff> &y,
    MultiRegions::Field<float, FieldState::Coeff> &z,
    const std::string &execSpace);

// sub template specialization.
template void MathHelper::sub<MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<double, FieldState::Phys> &x,
    MultiRegions::Field<double, FieldState::Phys> &y,
    MultiRegions::Field<double, FieldState::Phys> &z,
    const std::string &execSpace);
template void MathHelper::sub<MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<float, FieldState::Phys> &x,
    MultiRegions::Field<float, FieldState::Phys> &y,
    MultiRegions::Field<float, FieldState::Phys> &z,
    const std::string &execSpace);
template void MathHelper::sub<MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<double, FieldState::Coeff> &x,
    MultiRegions::Field<double, FieldState::Coeff> &y,
    MultiRegions::Field<double, FieldState::Coeff> &z,
    const std::string &execSpace);
template void MathHelper::sub<MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<float, FieldState::Coeff> &x,
    MultiRegions::Field<float, FieldState::Coeff> &y,
    MultiRegions::Field<float, FieldState::Coeff> &z,
    const std::string &execSpace);

// mul template specialization.
template void MathHelper::mul<MultiRegions::Field<double, FieldState::Phys>>(
    const double alpha, MultiRegions::Field<double, FieldState::Phys> &x,
    MultiRegions::Field<double, FieldState::Phys> &y,
    const std::string &execSpace);
template void MathHelper::mul<MultiRegions::Field<float, FieldState::Phys>>(
    const float alpha, MultiRegions::Field<float, FieldState::Phys> &x,
    MultiRegions::Field<float, FieldState::Phys> &y,
    const std::string &execSpace);
template void MathHelper::mul<MultiRegions::Field<double, FieldState::Coeff>>(
    const double alpha, MultiRegions::Field<double, FieldState::Coeff> &x,
    MultiRegions::Field<double, FieldState::Coeff> &y,
    const std::string &execSpace);
template void MathHelper::mul<MultiRegions::Field<float, FieldState::Coeff>>(
    const float alpha, MultiRegions::Field<float, FieldState::Coeff> &x,
    MultiRegions::Field<float, FieldState::Coeff> &y,
    const std::string &execSpace);
template void MathHelper::mul<MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<double, FieldState::Phys> &x,
    MultiRegions::Field<double, FieldState::Phys> &y,
    MultiRegions::Field<double, FieldState::Phys> &z,
    const std::string &execSpace);
template void MathHelper::mul<MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<float, FieldState::Phys> &x,
    MultiRegions::Field<float, FieldState::Phys> &y,
    MultiRegions::Field<float, FieldState::Phys> &z,
    const std::string &execSpace);
template void MathHelper::mul<MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<double, FieldState::Coeff> &x,
    MultiRegions::Field<double, FieldState::Coeff> &y,
    MultiRegions::Field<double, FieldState::Coeff> &z,
    const std::string &execSpace);
template void MathHelper::mul<MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<float, FieldState::Coeff> &x,
    MultiRegions::Field<float, FieldState::Coeff> &y,
    MultiRegions::Field<float, FieldState::Coeff> &z,
    const std::string &execSpace);

// div template specialization.
template void MathHelper::div<MultiRegions::Field<double, FieldState::Phys>>(
    const double alpha, MultiRegions::Field<double, FieldState::Phys> &x,
    MultiRegions::Field<double, FieldState::Phys> &y,
    const std::string &execSpace);
template void MathHelper::div<MultiRegions::Field<float, FieldState::Phys>>(
    const float alpha, MultiRegions::Field<float, FieldState::Phys> &x,
    MultiRegions::Field<float, FieldState::Phys> &y,
    const std::string &execSpace);
template void MathHelper::div<MultiRegions::Field<double, FieldState::Coeff>>(
    const double alpha, MultiRegions::Field<double, FieldState::Coeff> &x,
    MultiRegions::Field<double, FieldState::Coeff> &y,
    const std::string &execSpace);
template void MathHelper::div<MultiRegions::Field<float, FieldState::Coeff>>(
    const float alpha, MultiRegions::Field<float, FieldState::Coeff> &x,
    MultiRegions::Field<float, FieldState::Coeff> &y,
    const std::string &execSpace);
template void MathHelper::div<MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<double, FieldState::Phys> &x,
    MultiRegions::Field<double, FieldState::Phys> &y,
    MultiRegions::Field<double, FieldState::Phys> &z,
    const std::string &execSpace);
template void MathHelper::div<MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<float, FieldState::Phys> &x,
    MultiRegions::Field<float, FieldState::Phys> &y,
    MultiRegions::Field<float, FieldState::Phys> &z,
    const std::string &execSpace);
template void MathHelper::div<MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<double, FieldState::Coeff> &x,
    MultiRegions::Field<double, FieldState::Coeff> &y,
    MultiRegions::Field<double, FieldState::Coeff> &z,
    const std::string &execSpace);
template void MathHelper::div<MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<float, FieldState::Coeff> &x,
    MultiRegions::Field<float, FieldState::Coeff> &y,
    MultiRegions::Field<float, FieldState::Coeff> &z,
    const std::string &execSpace);

// daxpy template specialization.
template void MathHelper::daxpy<MultiRegions::Field<double, FieldState::Phys>>(
    const double alpha, MultiRegions::Field<double, FieldState::Phys> &x,
    MultiRegions::Field<double, FieldState::Phys> &y,
    MultiRegions::Field<double, FieldState::Phys> &z,
    const std::string &execSpace);
template void MathHelper::daxpy<MultiRegions::Field<float, FieldState::Phys>>(
    const float alpha, MultiRegions::Field<float, FieldState::Phys> &x,
    MultiRegions::Field<float, FieldState::Phys> &y,
    MultiRegions::Field<float, FieldState::Phys> &z,
    const std::string &execSpace);
template void MathHelper::daxpy<MultiRegions::Field<double, FieldState::Coeff>>(
    const double alpha, MultiRegions::Field<double, FieldState::Coeff> &x,
    MultiRegions::Field<double, FieldState::Coeff> &y,
    MultiRegions::Field<double, FieldState::Coeff> &z,
    const std::string &execSpace);
template void MathHelper::daxpy<MultiRegions::Field<float, FieldState::Coeff>>(
    const float alpha, MultiRegions::Field<float, FieldState::Coeff> &x,
    MultiRegions::Field<float, FieldState::Coeff> &y,
    MultiRegions::Field<float, FieldState::Coeff> &z,
    const std::string &execSpace);

// reduceSum template specialization.
template double MathHelper::reduceSum<
    MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float MathHelper::reduceSum<
    MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double MathHelper::reduceSum<
    MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float MathHelper::reduceSum<
    MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double MathHelper::reduceSum<
    MultiRegions::Field<uint8_t, FieldState::Phys>,
    MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<uint8_t, FieldState::Phys> &mask,
    MultiRegions::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float MathHelper::reduceSum<
    MultiRegions::Field<uint8_t, FieldState::Phys>,
    MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<uint8_t, FieldState::Phys> &mask,
    MultiRegions::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double MathHelper::reduceSum<
    MultiRegions::Field<uint8_t, FieldState::Coeff>,
    MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<uint8_t, FieldState::Coeff> &mask,
    MultiRegions::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float MathHelper::reduceSum<
    MultiRegions::Field<uint8_t, FieldState::Coeff>,
    MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<uint8_t, FieldState::Coeff> &mask,
    MultiRegions::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);

// reduceMax template specialization.
template double MathHelper::reduceMax<
    MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float MathHelper::reduceMax<
    MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double MathHelper::reduceMax<
    MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float MathHelper::reduceMax<
    MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double MathHelper::reduceMax<
    MultiRegions::Field<uint8_t, FieldState::Phys>,
    MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<uint8_t, FieldState::Phys> &mask,
    MultiRegions::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float MathHelper::reduceMax<
    MultiRegions::Field<uint8_t, FieldState::Phys>,
    MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<uint8_t, FieldState::Phys> &mask,
    MultiRegions::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double MathHelper::reduceMax<
    MultiRegions::Field<uint8_t, FieldState::Coeff>,
    MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<uint8_t, FieldState::Coeff> &mask,
    MultiRegions::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float MathHelper::reduceMax<
    MultiRegions::Field<uint8_t, FieldState::Coeff>,
    MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<uint8_t, FieldState::Coeff> &mask,
    MultiRegions::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);

// reduceMin template specialization.
template double MathHelper::reduceMin<
    MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float MathHelper::reduceMin<
    MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double MathHelper::reduceMin<
    MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float MathHelper::reduceMin<
    MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double MathHelper::reduceMin<
    MultiRegions::Field<uint8_t, FieldState::Phys>,
    MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<uint8_t, FieldState::Phys> &mask,
    MultiRegions::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float MathHelper::reduceMin<
    MultiRegions::Field<uint8_t, FieldState::Phys>,
    MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<uint8_t, FieldState::Phys> &mask,
    MultiRegions::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double MathHelper::reduceMin<
    MultiRegions::Field<uint8_t, FieldState::Coeff>,
    MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<uint8_t, FieldState::Coeff> &mask,
    MultiRegions::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float MathHelper::reduceMin<
    MultiRegions::Field<uint8_t, FieldState::Coeff>,
    MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<uint8_t, FieldState::Coeff> &mask,
    MultiRegions::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);

// ddot template specialization.
template double MathHelper::ddot<MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<double, FieldState::Phys> &x,
    MultiRegions::Field<double, FieldState::Phys> &y,
    const std::string &execSpace);
template float MathHelper::ddot<MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<float, FieldState::Phys> &x,
    MultiRegions::Field<float, FieldState::Phys> &y,
    const std::string &execSpace);
template double MathHelper::ddot<
    MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<double, FieldState::Coeff> &x,
    MultiRegions::Field<double, FieldState::Coeff> &y,
    const std::string &execSpace);
template float MathHelper::ddot<MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<float, FieldState::Coeff> &x,
    MultiRegions::Field<float, FieldState::Coeff> &y,
    const std::string &execSpace);

template double MathHelper::ddot<MultiRegions::Field<uint8_t, FieldState::Phys>,
                                 MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<uint8_t, FieldState::Phys> &mask,
    MultiRegions::Field<double, FieldState::Phys> &x,
    MultiRegions::Field<double, FieldState::Phys> &y,
    const std::string &execSpace);
template float MathHelper::ddot<MultiRegions::Field<uint8_t, FieldState::Phys>,
                                MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<uint8_t, FieldState::Phys> &mask,
    MultiRegions::Field<float, FieldState::Phys> &x,
    MultiRegions::Field<float, FieldState::Phys> &y,
    const std::string &execSpace);
template double MathHelper::ddot<
    MultiRegions::Field<uint8_t, FieldState::Coeff>,
    MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<uint8_t, FieldState::Coeff> &mask,
    MultiRegions::Field<double, FieldState::Coeff> &x,
    MultiRegions::Field<double, FieldState::Coeff> &y,
    const std::string &execSpace);
template float MathHelper::ddot<MultiRegions::Field<uint8_t, FieldState::Coeff>,
                                MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<uint8_t, FieldState::Coeff> &mask,
    MultiRegions::Field<float, FieldState::Coeff> &x,
    MultiRegions::Field<float, FieldState::Coeff> &y,
    const std::string &execSpace);

// l1norm template specialization.
template double MathHelper::l1norm<
    MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float MathHelper::l1norm<MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double MathHelper::l1norm<
    MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float MathHelper::l1norm<
    MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double MathHelper::l1norm<
    MultiRegions::Field<uint8_t, FieldState::Phys>,
    MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<uint8_t, FieldState::Phys> &mask,
    MultiRegions::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float MathHelper::l1norm<
    MultiRegions::Field<uint8_t, FieldState::Phys>,
    MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<uint8_t, FieldState::Phys> &mask,
    MultiRegions::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double MathHelper::l1norm<
    MultiRegions::Field<uint8_t, FieldState::Coeff>,
    MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<uint8_t, FieldState::Coeff> &mask,
    MultiRegions::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float MathHelper::l1norm<
    MultiRegions::Field<uint8_t, FieldState::Coeff>,
    MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<uint8_t, FieldState::Coeff> &mask,
    MultiRegions::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);

// l2norm template specialization.
template double MathHelper::l2norm<
    MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float MathHelper::l2norm<MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double MathHelper::l2norm<
    MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float MathHelper::l2norm<
    MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double MathHelper::l2norm<
    MultiRegions::Field<uint8_t, FieldState::Phys>,
    MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<uint8_t, FieldState::Phys> &mask,
    MultiRegions::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float MathHelper::l2norm<
    MultiRegions::Field<uint8_t, FieldState::Phys>,
    MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<uint8_t, FieldState::Phys> &mask,
    MultiRegions::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double MathHelper::l2norm<
    MultiRegions::Field<uint8_t, FieldState::Coeff>,
    MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<uint8_t, FieldState::Coeff> &mask,
    MultiRegions::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float MathHelper::l2norm<
    MultiRegions::Field<uint8_t, FieldState::Coeff>,
    MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<uint8_t, FieldState::Coeff> &mask,
    MultiRegions::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);

// lpnorm template specialization.
template double MathHelper::lpnorm<
    MultiRegions::Field<double, FieldState::Phys>>(
    const unsigned int p, MultiRegions::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float MathHelper::lpnorm<MultiRegions::Field<float, FieldState::Phys>>(
    const unsigned int p, MultiRegions::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double MathHelper::lpnorm<
    MultiRegions::Field<double, FieldState::Coeff>>(
    const unsigned int p, MultiRegions::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float MathHelper::lpnorm<
    MultiRegions::Field<float, FieldState::Coeff>>(
    const unsigned int p, MultiRegions::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double MathHelper::lpnorm<
    MultiRegions::Field<uint8_t, FieldState::Phys>,
    MultiRegions::Field<double, FieldState::Phys>>(
    const unsigned int p, MultiRegions::Field<uint8_t, FieldState::Phys> &mask,
    MultiRegions::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float MathHelper::lpnorm<
    MultiRegions::Field<uint8_t, FieldState::Phys>,
    MultiRegions::Field<float, FieldState::Phys>>(
    const unsigned int p, MultiRegions::Field<uint8_t, FieldState::Phys> &mask,
    MultiRegions::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double MathHelper::lpnorm<
    MultiRegions::Field<uint8_t, FieldState::Coeff>,
    MultiRegions::Field<double, FieldState::Coeff>>(
    const unsigned int p, MultiRegions::Field<uint8_t, FieldState::Coeff> &mask,
    MultiRegions::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float MathHelper::lpnorm<
    MultiRegions::Field<uint8_t, FieldState::Coeff>,
    MultiRegions::Field<float, FieldState::Coeff>>(
    const unsigned int p, MultiRegions::Field<uint8_t, FieldState::Coeff> &mask,
    MultiRegions::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);

// linfnorm template specialization.
template double MathHelper::linfnorm<
    MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float MathHelper::linfnorm<
    MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double MathHelper::linfnorm<
    MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float MathHelper::linfnorm<
    MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);
template double MathHelper::linfnorm<
    MultiRegions::Field<uint8_t, FieldState::Phys>,
    MultiRegions::Field<double, FieldState::Phys>>(
    MultiRegions::Field<uint8_t, FieldState::Phys> &mask,
    MultiRegions::Field<double, FieldState::Phys> &x,
    const std::string &execSpace);
template float MathHelper::linfnorm<
    MultiRegions::Field<uint8_t, FieldState::Phys>,
    MultiRegions::Field<float, FieldState::Phys>>(
    MultiRegions::Field<uint8_t, FieldState::Phys> &mask,
    MultiRegions::Field<float, FieldState::Phys> &x,
    const std::string &execSpace);
template double MathHelper::linfnorm<
    MultiRegions::Field<uint8_t, FieldState::Coeff>,
    MultiRegions::Field<double, FieldState::Coeff>>(
    MultiRegions::Field<uint8_t, FieldState::Coeff> &mask,
    MultiRegions::Field<double, FieldState::Coeff> &x,
    const std::string &execSpace);
template float MathHelper::linfnorm<
    MultiRegions::Field<uint8_t, FieldState::Coeff>,
    MultiRegions::Field<float, FieldState::Coeff>>(
    MultiRegions::Field<uint8_t, FieldState::Coeff> &mask,
    MultiRegions::Field<float, FieldState::Coeff> &x,
    const std::string &execSpace);

} // namespace Nektar::Math
