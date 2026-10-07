////////////////////////////////////////////////////////////////////////////////
//
//  File: NodeOpti.cpp
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
//  Description: Calculate jacobians of elements.
//
////////////////////////////////////////////////////////////////////////////////

#include <boost/multi_array.hpp>

#include <limits>

#include "Evaluator.hxx"
#include "Hessian.hxx"
#include "NodeOpti.h"

using namespace std;
using namespace Nektar::NekMesh;

namespace Nektar::NekMesh
{

NodeOptiFactory &GetNodeOptiFactory()
{
    static NodeOptiFactory asd;
    return asd;
}

std::mutex NodeOpti::mtx;

int NodeOpti2D2D::m_type = GetNodeOptiFactory().RegisterCreatorFunction(
    22, NodeOpti2D2D::create, "2D2D");

void NodeOpti2D2D::Optimise()
{
    NekDouble minJacNew;
    NekDouble currentW = GetFunctional<2>(minJacNew);
    NekDouble newVal   = currentW;

    // Gradient already zero
    if (m_grad[0] * m_grad[0] + m_grad[1] * m_grad[1] > gradTol())
    {
        // needs to optimise
        NekDouble xc = (*m_node)[0];
        NekDouble yc = (*m_node)[1];

        NekDouble sk[2];
        NekDouble val;

        // Calculate minimum eigenvalue
        MinEigen<2>(val);

        if (val < 1e-6)
        {
            // Add constant identity to Hessian matrix.
            m_grad[2] += 1e-6 - val;
            m_grad[4] += 1e-6 - val;
        }

        NewtonDirection<2>(m_grad.data(), sk);

        bool found   = false;
        NekDouble pg = (m_grad[0] * sk[0] + m_grad[1] * sk[1]);

        // A direction that does not point downhill lets the Wolfe condition
        // below accept a step that raises the energy, since c1*alpha*pg is
        // then positive. Fall back on steepest descent.
        if (pg >= 0.0)
        {
            sk[0] = -m_grad[0];
            sk[1] = -m_grad[1];
            pg    = -(m_grad[0] * m_grad[0] + m_grad[1] * m_grad[1]);
        }

        // normal gradient line Search. The node is left where it is and the
        // step offered to the functional instead, so that a rejected step
        // costs nothing to undo and the derivatives of the mapping, which do
        // not depend on the step, are not recomputed for each one.
        NekDouble alpha = 1.0;

        while (alpha > alphaTol())
        {
            NekDouble offset[2] = {alpha * sk[0], alpha * sk[1]};

            newVal = GetFunctionalAt<2>(offset, minJacNew);

            // Wolfe conditions
            if (newVal <= currentW + c1() * (alpha * pg))
            {
                MoveNode<2>(offset);
                found = true;
                break;
            }

            alpha /= 2.0;
        }

        if (!found)
        {
            mtx.lock();
            m_res->nReset[2]++;
            mtx.unlock();
        }
        else
        {
            mtx.lock();
            if (alpha < 1.0)
            {
                m_res->alphaI++;
            }
            mtx.unlock();
        }

        mtx.lock();
        m_res->val = max(sqrt(((*m_node)[0] - xc) * ((*m_node)[0] - xc) +
                              ((*m_node)[1] - yc) * ((*m_node)[1] - yc)),
                         m_res->val);
        mtx.unlock();
    }

    mtx.lock();
    m_res->func += newVal;
    mtx.unlock();
}

int NodeOpti3D3D::m_type = GetNodeOptiFactory().RegisterCreatorFunction(
    33, NodeOpti3D3D::create, "3D3D");

void NodeOpti3D3D::Optimise()
{
    NekDouble minJacNew;
    NekDouble currentW = GetFunctional<3>(minJacNew);
    NekDouble newVal   = currentW;

    if (m_grad[0] * m_grad[0] + m_grad[1] * m_grad[1] + m_grad[2] * m_grad[2] >
        gradTol())
    {
        // needs to optimise
        NekDouble xc = (*m_node)[0];
        NekDouble yc = (*m_node)[1];
        NekDouble zc = (*m_node)[2];

        NekDouble sk[3];
        NekDouble val;

        // Calculate minimum eigenvalue
        MinEigen<3>(val);

        if (val < 1e-6)
        {
            // Add constant identity to Hessian matrix.
            m_grad[3] += 1e-6 - val;
            m_grad[6] += 1e-6 - val;
            m_grad[8] += 1e-6 - val;
        }

        NewtonDirection<3>(m_grad.data(), sk);

        bool found = false;

        NekDouble pg =
            (m_grad[0] * sk[0] + m_grad[1] * sk[1] + m_grad[2] * sk[2]);

        // See the note in NodeOpti2D2D::Optimise.
        if (pg >= 0.0)
        {
            sk[0] = -m_grad[0];
            sk[1] = -m_grad[1];
            sk[2] = -m_grad[2];
            pg    = -(m_grad[0] * m_grad[0] + m_grad[1] * m_grad[1] +
                   m_grad[2] * m_grad[2]);
        }

        // normal gradient line Search; see the note in NodeOpti2D2D.
        NekDouble alpha = 1.0;

        while (alpha > alphaTol())
        {
            NekDouble offset[3] = {alpha * sk[0], alpha * sk[1], alpha * sk[2]};

            newVal = GetFunctionalAt<3>(offset, minJacNew);

            // Wolfe conditions
            if (newVal <= currentW + c1() * alpha * pg)
            {
                MoveNode<3>(offset);
                found = true;
                break;
            }

            alpha /= 2.0;
        }

        if (!found)
        {
            mtx.lock();
            m_res->nReset[2]++;
            mtx.unlock();
        }
        else
        {
            mtx.lock();
            if (alpha < 1.0)
            {
                m_res->alphaI++;
            }
            mtx.unlock();
        }

        mtx.lock();
        m_res->val = max(sqrt(((*m_node)[0] - xc) * ((*m_node)[0] - xc) +
                              ((*m_node)[1] - yc) * ((*m_node)[1] - yc) +
                              ((*m_node)[2] - zc) * ((*m_node)[2] - zc)),
                         m_res->val);
        mtx.unlock();
    }
    mtx.lock();
    m_res->func += newVal;
    mtx.unlock();
}

} // namespace Nektar::NekMesh
