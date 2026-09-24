///////////////////////////////////////////////////////////////////////////////
//
// File: SolverCore.hpp
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

#pragma once

#include <SolverCore/SolverCoreDeclspec.h>

namespace Nektar::SolverCore
{

/**
 * @brief Forces any translation unit that calls this function to carry a
 * genuine, real cross-library symbol reference into libSolverCore.
 *
 * SolverCore's operators (Advection/Diffusion/RiemannSolvers) register
 * themselves with the Operators factory purely as a side effect of static
 * initialisation performed in generated *OpImpl translation units (see
 * e.g. RiemannOperatorFactoryDec.cpp.in). Most consumers only ever reach
 * SolverCore through header-only templates and never call a symbol that is
 * actually *defined* in the compiled SolverCore library, so a
 * linker/loader that only keeps what is referenced -- e.g. GNU ld's
 * default "--as-needed" behaviour on Debian/Ubuntu, or the Windows PE
 * loader simply not generating an import entry for an unused DLL -- can
 * silently drop SolverCore from the final binary. In that case the
 * library never gets loaded, and none of its operators end up registered
 * (e.g. "UpwindSerial"/"UpwindDevice"), even though everything compiled
 * and linked without any error.
 *
 * Calling this (empty) function from operator entry points that are
 * always compiled directly into consumers -- SolverCore::TraceFluxOp,
 * AdvectionWeakDGOp, DiffusionIPOp and AdvWeakDGDiffusionIPOp all call it
 * from Create() -- guarantees a real reference exists, forcing every linker
 * to keep SolverCore linked and every loader to load it, on any platform,
 * without relying on any linker-specific flags.
 *
 * Anything added to SolverCore that consumers reach only through header-only
 * templates needs the same call, or its registrations can vanish.
 */
SOLVER_CORE_EXPORT void EnsureLinked();

} // namespace Nektar::SolverCore
