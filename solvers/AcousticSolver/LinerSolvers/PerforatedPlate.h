///////////////////////////////////////////////////////////////////////////////
//
// File: PerforatedPlate.h
//
// For more information, please see: http://www.nektar.info
//
// The MIT License
//
// Copyright (c) 2026 Nicolas de Jong
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
// Description: PerforatedPlate base class
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_SOLVERS_ACOUSTICSOLVER_LINERSOLVERS_PERFORATEDPLATE_H
#define NEKTAR_SOLVERS_ACOUSTICSOLVER_LINERSOLVERS_PERFORATEDPLATE_H

#include <LibUtilities/TimeIntegration/TimeIntegrationScheme.h>
#include <LibUtilities/TimeIntegration/TimeIntegrationSchemeOperators.h>

#include <LibUtilities/LinearAlgebra/NekVector.hpp>

using namespace Nektar::LibUtilities;

namespace Nektar
{

class PerforatedPlate
{
public:
    PerforatedPlate(int linerType, NekDouble w0, NekDouble wI, NekDouble u0,
                    NekDouble uI);

    void InitObject(Array<OneD, NekDouble> &A, Array<OneD, NekDouble> &B,
                    short sizeF, int nBC, Array<OneD, int> &nPnt, int nExp);

    // Pressure jump across the perforated plate
    Array<OneD, NekDouble> LinerOutput(int bound, int point, const NekDouble *u,
                                       const NekDouble intTime,
                                       const NekDouble simdt);

    // Time integration
    TimeIntegrationSchemeOperators m_timeInt;
    void LinerOperator(const Array<OneD, const Array<OneD, NekDouble>> &inx,
                       Array<OneD, Array<OneD, NekDouble>> &outx,
                       const NekDouble time);
    void LinerProjection(const Array<OneD, Array<OneD, NekDouble>> &inx,
                         Array<OneD, Array<OneD, NekDouble>> &outx,
                         const NekDouble time);

private:
    // Data inputs
    const short m_modelType = 0; // Unused, for future development
    const NekDouble m_w0;        // Initial frequency used for liner 0 state
    const NekDouble m_wI; // Imaginary component of initial frequency for liner
    const NekDouble m_u0; // Real component of initial liner input
    const NekDouble m_uI; // Imaginary component of initial liner input

    // Rational fraction coefficients
    NekVector<NekDouble> m_A;
    NekVector<NekDouble> m_B;

    // Variable for time integration m_U[BC][Exp][var][Point]
    Array<OneD, Array<OneD, Array<OneD, Array<OneD, NekDouble>>>> m_U;
    Array<OneD, Array<OneD, Array<OneD, Array<OneD, NekDouble>>>> m_y;
    NekDouble m_currentTime = 0.0;
    NekDouble m_lastTime    = 0.0;
    NekDouble m_dt          = 0.0;

    // Time integration
    LibUtilities::TimeIntegrationSchemeSharedPtr m_timeScheme;
    LibUtilities::TimeIntegrationSchemeOperators m_ODE;
};

} // namespace Nektar

#endif
