///////////////////////////////////////////////////////////////////////////////
//
// File: OperatorIdentity.cpp
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

#include "OperatorIdentity.hpp"

namespace Nektar::Operators
{

// define static variables for Identity operator descriptor for the default fp
// type config For coeff -> coeff
template <>
const std::string Identity<FieldState::Coeff, default_fp_type>::key =
    "IdentityCoeff";
template <>
const std::string Identity<FieldState::Coeff, default_fp_type>::default_impl =
    "StdMat";

// For phys -> phys
template <>
const std::string Identity<FieldState::Phys, default_fp_type>::key =
    "IdentityPhys";
template <>
const std::string Identity<FieldState::Phys, default_fp_type>::default_impl =
    "StdMat";

} // namespace Nektar::Operators
