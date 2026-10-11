////////////////////////////////////////////////////////////////////////////////
//
//  File: BGFS-B.cpp
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

#include <cmath>
#include <limits>
#include <set>

#include <LibUtilities/BasicUtils/Vmath.hpp>
#include <NekMesh/Optimisation/BGFS-B.h>

using namespace std;

namespace Nektar::NekMesh
{

namespace
{

DNekMat AsColumn(const vector<NekDouble> &v)
{
    DNekMat m(v.size(), 1, 0.0);
    for (size_t i = 0; i < v.size(); ++i)
    {
        m(i, 0) = v[i];
    }
    return m;
}

/**
 * @brief One step: project onto the box, take a quasi-Newton step in what is
 * left free, line search it, and update the Hessian approximations.
 *
 * @p x and @p J are advanced to the new point, @p B and @p H updated for it.
 * Returns false if the line search did not find a point satisfying the Wolfe
 * conditions, in which case nothing has moved.
 */
bool Step(const Objective &obj, vector<NekDouble> &x, DNekMat &J, DNekMat &B,
          DNekMat &H)
{
    const size_t n              = x.size();
    const vector<NekDouble> &li = obj.lower;
    const vector<NekDouble> &ui = obj.upper;

    // Variables the projected gradient does not drive straight into a bound;
    // the quasi-Newton step below moves only these.
    set<int> freeSet;
    vector<NekDouble> ti(n);
    for (size_t i = 0; i < n; i++)
    {
        if (J(i, 0) < 0)
        {
            ti[i] = (x[i] - ui[i]) / J(i, 0);
        }
        else if (J(i, 0) > 0)
        {
            ti[i] = (x[i] - li[i]) / J(i, 0);
        }
        else
        {
            ti[i] = numeric_limits<double>::max();
        }

        if (ti[i] > 0)
        {
            freeSet.insert(i);
        }
    }

    DNekMat d(n, 1);
    for (size_t i = 0; i < n; i++)
    {
        d(i, 0) = fabs(ti[i]) < 1E-10 ? 0.0 : -1.0 * J(i, 0);
    }

    // The Cauchy point: the steepest descent step, clipped to the box.
    vector<NekDouble> xci(n);
    for (size_t i = 0; i < n; i++)
    {
        if (x[i] + d(i, 0) < li[i])
        {
            xci[i] = li[i];
            freeSet.erase(i);
            continue;
        }
        else
        {
            xci[i] = x[i] + d(i, 0);
        }

        if (x[i] + d(i, 0) > ui[i])
        {
            xci[i] = ui[i];
            freeSet.erase(i);
            continue;
        }
        else
        {
            xci[i] = x[i] + d(i, 0);
        }
    }

    DNekMat Z(n, n, 0.0);
    for (size_t i = 0; i < n; i++)
    {
        if (freeSet.find(i) != freeSet.end())
        {
            Z(i, i) = 1.0;
        }
    }

    DNekMat dx(n, 1, 0.0);
    for (size_t i = 0; i < n; i++)
    {
        dx(i, 0) = xci[i] - x[i];
    }

    DNekMat rg = Z * (J + B * dx);
    DNekMat du = -1.0 * H * rg;

    // Shorten the quasi-Newton step so it too stays inside the box.
    NekDouble alpha = 1.0;
    for (auto it = freeSet.begin(); it != freeSet.end(); it++)
    {
        int i = (*it);
        if (li[i] - xci[i] > alpha * du(i, 0))
        {
            alpha = min(alpha, (li[i] - xci[i]) / du(i, 0));
        }
        else if (ui[i] - xci[i] < alpha * du(i, 0))
        {
            alpha = min(alpha, (ui[i] - xci[i]) / du(i, 0));
        }
    }

    DNekMat grad = alpha * du;

    vector<NekDouble> dk(n), xibar(n);
    for (size_t i = 0; i < n; i++)
    {
        xibar[i] =
            freeSet.find(i) != freeSet.end() ? xci[i] + grad(i, 0) : xci[i];
    }

    Vmath::Vsub(n, &xibar[0], 1, &x[0], 1, &dk[0], 1);

    NekDouble c = 0.0;
    NekDouble r = 0.0;
    NekDouble l = 0.0;
    for (size_t i = 0; i < n; i++)
    {
        c += 1E-4 * J(i, 0) * dk[i];
        r += J(i, 0) * dk[i];
    }

    // Backtracking line search on the Wolfe conditions.
    NekDouble lam   = 2.0;
    int iterationCt = 0;
    NekDouble fo    = obj.value(x);
    NekDouble fn    = 0.0;
    vector<NekDouble> tst(n);
    DNekMat Jn;

    do
    {
        if (iterationCt > 100)
        {
            return false;
        }
        iterationCt++;

        lam *= 0.5;

        for (size_t i = 0; i < n; i++)
        {
            tst[i] = x[i] + lam * dk[i];
        }

        fn = obj.value(tst);

        // Kept rather than recomputed once the loop exits: the gradient at
        // the accepted point is this same one, and it is the dearer of the
        // two evaluations.
        Jn = AsColumn(obj.gradient(tst));

        l = 0.0;
        for (size_t i = 0; i < n; i++)
        {
            l += Jn(i, 0) * dk[i];
        }
    } while (fn > fo + c || fabs(l) > 1.0 * fabs(r));

    DNekMat y  = Jn - J;
    DNekMat yT = Jn - J;
    yT.Transpose();

    DNekMat s(n, 1, 0.0);
    for (size_t i = 0; i < n; i++)
    {
        s(i, 0) = lam * dk[i];
    }
    DNekMat sT = s;
    sT.Transpose();

    DNekMat d1 = yT * s;
    DNekMat d2 = sT * B * s;
    DNekMat d3 = sT * y;
    DNekMat n1 = yT * H * y;

    NekDouble ynorm = 0.0;
    for (size_t i = 0; i < n; i++)
    {
        ynorm += y(i, 0) * y(i, 0);
    }

    // Skip the update when the curvature condition is too weak to trust it.
    if (d3(0, 0) > 2.2E-16 * ynorm)
    {
        B = B + y * yT * (1.0 / d1(0, 0)) - B * s * sT * B * (1.0 / d2(0, 0));
        H = H + (d3(0, 0) + n1(0, 0)) / d3(0, 0) / d3(0, 0) * s * sT -
            1.0 / d3(0, 0) * (H * y * sT + s * yT * H);
    }

    J = Jn;
    x = tst;

    return true;
}

} // namespace

BGFSResult BGFSOptimise(const Objective &obj, vector<NekDouble> x,
                        NekDouble tolerance, int maxIterations)
{
    const size_t n = x.size();

    ASSERTL0(obj.lower.size() == n && obj.upper.size() == n &&
                 obj.scale.size() == n,
             "the objective's bounds and scales must match the number of "
             "variables being optimised");

    // Both start at the identity, which is to say the first step is steepest
    // descent and the curvature is learned from there.
    DNekMat B(n, n, 0.0), H(n, n, 0.0);
    for (size_t i = 0; i < n; i++)
    {
        B(i, i) = 1.0;
        H(i, i) = 1.0;
    }

    DNekMat J = AsColumn(obj.gradient(x));

    BGFSResult result;

    for (int iteration = 0;; iteration++)
    {
        NekDouble norm = 0.0;
        for (size_t i = 0; i < n; i++)
        {
            norm += J(i, 0) * J(i, 0) / obj.scale[i] / obj.scale[i];
        }
        norm = sqrt(norm);

        result.x          = x;
        result.norm       = norm;
        result.iterations = iteration;

        if (norm < tolerance)
        {
            result.converged = true;
            return result;
        }

        if (iteration > maxIterations || !Step(obj, x, J, B, H))
        {
            result.converged = false;
            return result;
        }
    }
}

} // namespace Nektar::NekMesh
