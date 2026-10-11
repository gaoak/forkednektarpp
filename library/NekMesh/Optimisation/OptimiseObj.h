////////////////////////////////////////////////////////////////////////////////
//
//  File: OptimiseObj.h
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
//  Description: the problem a bounded BFGS optimisation is given.
////////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_MESHUTILS_OPTIMISATION_OPTIMISEOBJ_H
#define NEKTAR_MESHUTILS_OPTIMISATION_OPTIMISEOBJ_H

#include <functional>
#include <vector>

#include <LibUtilities/LinearAlgebra/NekTypeDefs.hpp>

namespace Nektar::NekMesh
{

/// The value of an objective at a point.
using ObjectiveValue = std::function<NekDouble(const std::vector<NekDouble> &)>;

/// The gradient of an objective at a point, one entry per variable.
using ObjectiveGradient =
    std::function<std::vector<NekDouble>(const std::vector<NekDouble> &)>;

/**
 * @brief Everything a bounded minimisation needs to know about its problem.
 *
 * This is the problem, not the solver and not the iterate: handing it to
 * BGFSOptimise() does not change it, and the same one can be solved from
 * several starting points. What the optimiser carries -- where it currently
 * is, and its approximations to the Hessian -- belongs to the optimiser.
 *
 * Objectives are built by free functions returning one of these, so a caller
 * writes down the quantity it wants minimised and nothing else; see
 * NekMesh/SurfaceMeshing/OptimiseFunctions.h for the two the high-order
 * surface mesher uses.
 */
struct Objective
{
    /// The quantity to minimise.
    ObjectiveValue value;

    /// Its gradient. Must agree with #value, which nothing checks.
    ObjectiveGradient gradient;

    /// Lower and upper bound on each variable; the search stays inside them.
    std::vector<NekDouble> lower, upper;

    /**
     * @brief The size of each variable, used only to judge convergence.
     *
     * The gradient is tested against a tolerance after dividing through by
     * these, so that the test means the same thing whatever the variables are
     * measured in -- a curve parameter running over tens and one running over
     * thousandths should not need different tolerances. Set them to one to
     * test the gradient as it stands.
     */
    std::vector<NekDouble> scale;
};

} // namespace Nektar::NekMesh

#endif
