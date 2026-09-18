////////////////////////////////////////////////////////////////////////////////
//
//  File: BooleanOperators.cpp
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
//  Description: Boolean operators for comparison of mesh objects.
//
////////////////////////////////////////////////////////////////////////////////

#include <NekMesh/MeshElements/HOAlignment.h>
#include <NekMesh/MeshElements/Mesh.h>

using namespace std;

namespace Nektar::NekMesh
{

/**
 * @brief Compares two element config structs
 */
bool operator==(ElmtConfig const &c1, ElmtConfig const &c2)
{
    return (c1.m_e == c2.m_e && c1.m_order == c2.m_order);
}

/**
 * @brief Test equality of two conditions - i.e. compare types, fields
 * and values but _not_ composite ids.
 */
bool operator==(ConditionSharedPtr const &c1, ConditionSharedPtr const &c2)
{
    int i, n = c1->type.size();

    if (n != c2->type.size())
    {
        return false;
    }

    for (i = 0; i < n; ++i)
    {
        if (c1->type[i] != c2->type[i])
        {
            return false;
        }

        if (c1->field[i] != c2->field[i] || c1->value[i] != c2->value[i])
        {
            return false;
        }
    }

    return true;
}

} // namespace Nektar::NekMesh
