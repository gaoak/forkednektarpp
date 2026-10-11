////////////////////////////////////////////////////////////////////////////////
//
//  File: OptimiseFunctions.h
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
//  Description: the objectives the high-order surface mesher minimises.
////////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_MESHUTILS_SURFACEMESH_OPTIMISEFUNCTIONS_H
#define NEKTAR_MESHUTILS_SURFACEMESH_OPTIMISEFUNCTIONS_H

#include <NekMesh/Optimisation/OptimiseObj.h>
#include <SpatialDomains/CADSystem/CADCurve.h>
#include <SpatialDomains/CADSystem/CADSurf.h>

namespace Nektar::NekMesh
{

/**
 * @brief Spreading the interior nodes of an edge that lies along a CAD curve.
 *
 * The nodes sit at curve parameters running from @p tBegin to @p tEnd, the
 * two ends fixed and the interior ones free. What is minimised is the sum
 * over consecutive pairs of the squared distance between them divided by the
 * gap between the reference points @p z, which is smallest when the nodes are
 * spread along the curve in the same proportions as @p z is along the
 * reference element.
 */
Objective CurveEdgeObjective(const SpatialDomains::CADCurveSharedPtr &curve,
                             NekDouble tBegin, NekDouble tEnd,
                             const std::vector<NekDouble> &z);

/**
 * @brief The same, for an edge lying on a CAD surface rather than a curve.
 *
 * Here each node has two free variables rather than one, and they are
 * interleaved: the vector is @f$ (u_1, v_1, u_2, v_2, \ldots) @f$.
 */
Objective SurfaceEdgeObjective(const SpatialDomains::CADSurfSharedPtr &surf,
                               std::array<NekDouble, 2> uvBegin,
                               std::array<NekDouble, 2> uvEnd,
                               const std::vector<NekDouble> &z);

} // namespace Nektar::NekMesh
#endif
