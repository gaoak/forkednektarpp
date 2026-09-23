///////////////////////////////////////////////////////////////////////////////
//
// File: DerivZOpImpl.hpp
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
// Description: Declaration of the homogeneous z-derivative backend.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/FFT/DerivZDeviceFFT.h>

namespace Nektar::Operators::detail
{

/// \brief Which way round a homogeneous z-derivative maps its components.
///
/// Two of these are the same mapping read in opposite directions and the
/// third leaves the components alone, so one parameter covers all three.
enum class DerivZLayout
{
    /// Component n of a scalar field to slot 3n + 2 of a vector, as PhysDeriv
    /// needs for the z entry of its gradient.
    ScalarToVectorZ,
    /// Component 3n + 2 of a vector, the z direction of variable n, to
    /// component n of a scalar, as IProductWRTDerivBase and Divergence need.
    VectorZToScalar,
    /// Component n to component n, both sides scalar, as Laplacian and
    /// Helmholtz need for the weak z-Laplacian.
    Identity,
};

/// Which z-derivative to take. It belongs to the transform, so it is defined
/// once in LibUtilities and named here for the operators that pick one.
using LibUtilities::DerivZOrder;

/// \brief The homogeneous z-derivative, specialised per execution space in
/// DerivZOpSerialAVX.hpp and DerivZOpDevice.hpp.
///
/// LAYOUT says which components are read and written and DERIVORDER which
/// derivative is taken; both orders give the derivative itself, so an
/// operator wanting minus the second one -- the form the weak z-Laplacian
/// takes -- subtracts at its own level.
///
/// APPEND says whether Launch() adds its result to the output or replaces
/// it: the operators whose xy pass has already written a partial result --
/// Divergence, which leaves du/dx + dv/dy there -- accumulate on top, while
/// those writing into a slot of their own overwrite.
template <typename ExecSpace, typename TData, DerivZLayout LAYOUT,
          DerivZOrder DERIVORDER, bool APPEND, typename Enable = void>
class DerivZOpImpl;
} // namespace Nektar::Operators::detail

#include "Operators/ElmtOps/DerivZOpDevice.hpp"
#include "Operators/ElmtOps/DerivZOpSerialAVX.hpp"
