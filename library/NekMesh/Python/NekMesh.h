///////////////////////////////////////////////////////////////////////////////
//
// File: NekMesh.h
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
// Description: NekPy configuration for NekMesh to defined opaque types.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_NEKMESH_PYTHON_NEKMESH_H
#define NEKTAR_NEKMESH_PYTHON_NEKMESH_H

#include <NekMesh/MeshElements/Element.h>
#include <NekMesh/MeshElements/Mesh.h>

using namespace Nektar;
using namespace Nektar::NekMesh;

// The mesh is held in a MeshGraph now: its vertices and elements are
// SpatialDomains geometry owned by the graph, exposed through NekPy's
// SpatialDomains module, so there are no NekMesh-side container types left to
// make opaque. Element tags are bound as an opaque map so that Python sees the
// mesh's own map rather than a copy.
PYBIND11_MAKE_OPAQUE(GeomTagMap);

#endif
