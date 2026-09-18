###############################################################################
##
## File: test_nekmesh_node.py
##
## For more information, please see: http://www.nektar.info
##
## The MIT License
##
## Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
## Department of Aeronautics, Imperial College London (UK), and Scientific
## Computing and Imaging Institute, University of Utah (USA).
##
## Permission is hereby granted, free of charge, to any person obtaining a
## copy of this software and associated documentation files (the "Software"),
## to deal in the Software without restriction, including without limitation
## the rights to use, copy, modify, merge, publish, distribute, sublicense,
## and/or sell copies of the Software, and to permit persons to whom the
## Software is furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included
## in all copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
## OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
## THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
## FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
## DEALINGS IN THE SOFTWARE.
##
## Description: Unit tests for mesh vertices.
##
###############################################################################

from NekPy.NekMesh import Mesh
import unittest

class TestVertex(unittest.TestCase):
    """A mesh vertex is a SpatialDomains PointGeom owned by the MeshGraph,
    so it is not seperately built as old NekMesh Node."""

    def setUp(self):
        self.mesh = Mesh()
        self.mesh.expDim   = 3
        self.mesh.spaceDim = 3
        self.id = 1
        self.x, self.y, self.z = 1.1, 2.2, 3.3
        self.vert = self.mesh.CreateVertex(self.id, self.x, self.y, self.z)

    def testVertexCoordinates(self):
        x, y, z = self.vert.GetCoordinates()
        self.assertEqual(x, self.x)
        self.assertEqual(y, self.y)
        self.assertEqual(z, self.z)

    def testVertexGlobalID(self):
        self.assertEqual(self.vert.GetGlobalID(), self.id)
        self.vert.SetGlobalID(2)
        self.assertEqual(self.vert.GetGlobalID(), 2)

    def testVertexCoordim(self):
        self.assertEqual(self.vert.GetCoordim(), 3)

    def testVertexAutomaticID(self):
        # An id of -1 asks the mesh for the next unused one.
        other = self.mesh.CreateVertex(-1, 0.0, 0.0, 0.0)
        self.assertNotEqual(other.GetGlobalID(), self.vert.GetGlobalID())

    def testMeshOwnsVertices(self):
        verts = self.mesh.GetVertices()
        self.assertEqual(len(verts), 1)
        self.assertEqual(verts[0].GetGlobalID(), self.vert.GetGlobalID())

        for i in range(9):
            self.mesh.CreateVertex(-1, float(i), float(i), float(i))
        self.assertEqual(len(self.mesh.GetVertices()), 10)

if __name__ == '__main__':
    unittest.main()
