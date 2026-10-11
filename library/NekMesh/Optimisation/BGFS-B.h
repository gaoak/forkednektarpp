////////////////////////////////////////////////////////////////////////////////
//
//  File: BGFS-B.h
//
//  For more information, please see: http://www.nektar.info/
//
//  The MIT License
//
//  Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
//  Department of Aeronautics, Imperial College London (UK), and Scientific
//  Computing and Imaging Institute, University of Utah (USA).
//
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//
//  Description: bounded BFGS minimisation.
////////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_MESHUTILS_OPTIMISATION_BGFSB_H
#define NEKTAR_MESHUTILS_OPTIMISATION_BGFSB_H

#include <NekMesh/Optimisation/OptimiseObj.h>

namespace Nektar::NekMesh
{

/// Where a minimisation ended up, and whether it got there.
struct BGFSResult
{
    /// The best point found.
    std::vector<NekDouble> x;

    /// True if the scaled gradient norm fell below the tolerance. False
    /// means the iteration limit was reached or the line search gave up, and
    /// #x is then wherever it had got to, which the caller may prefer to
    /// discard.
    bool converged = false;

    /// Gradient norm at #x, scaled by Objective::scale.
    NekDouble norm = 0.0;

    /// How many steps were taken.
    int iterations = 0;
};

/**
 * @brief Minimise @p obj subject to its bounds, starting from @p x.
 *
 * Bounded BFGS: a projected gradient step onto the feasible box, then a
 * quasi-Newton step within the variables the box leaves free, with a line
 * search on the Wolfe conditions.
 *
 * The optimiser owns the iterate and its two Hessian approximations for the
 * duration of the call; @p obj is only read.
 */
BGFSResult BGFSOptimise(const Objective &obj, std::vector<NekDouble> x,
                        NekDouble tolerance, int maxIterations);

} // namespace Nektar::NekMesh

#endif
