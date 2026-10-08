///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionZOpImpl.hpp
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
// Description: Declaration of the homogeneous z-advection backend.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

namespace Nektar::MultiRegions::detail
{

/// \brief The advection along the homogeneous direction, w du/dz, added on
/// top of whatever the xy pass has already left in the output.
///
/// It sits here rather than with one operator because two of them take it:
/// Advection itself, and LinAdvDiffReaction, whose weak z advection is this
/// term formed at the quadrature points and taken back through the inner
/// product. Specialised per execution space in AdvectionZOpSerialAVX.hpp and
/// AdvectionZOpDevice.hpp.
template <typename ExecSpace, typename TData, typename Enable = void>
class AdvectionZOpImpl;
} // namespace Nektar::MultiRegions::detail

#include <MultiRegions/ElmtOps/AdvectionZOpDevice.hpp>
#include <MultiRegions/ElmtOps/AdvectionZOpSerialAVX.hpp>
