###############################################################################
##
## File: test_nekmesh_mesh.py
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
## Description: Unit tests for the Mesh class.
##
###############################################################################

from NekPy.NekMesh import Mesh, ElmtConfig
from NekPy.LibUtilities import ShapeType
import unittest
import numpy as np

class TestMesh(unittest.TestCase):

    def _initialize_static_values(self):
        self.coord_1x, self.coord_2x = 0.0, 3.0
        self.coord_1y, self.coord_2y = 1.0, 2.0
        self.nx, self.ny = 11, 7
        self.comp_ID     = 2
        self.x_points    = np.linspace(self.coord_1x, self.coord_2x, self.nx)
        self.y_points    = np.linspace(self.coord_1y, self.coord_2y, self.ny)
        self.expDim      = 2
        self.spaceDim    = 2

    def _create_mesh(self):
        self.mesh = Mesh()
        self.mesh.expDim   = self.expDim
        self.mesh.spaceDim = self.spaceDim

    def _initialize_nodes(self):
        self.nodes = []
        id_cnt = 0
        for y in range(self.ny):
            tmp = []
            for x in range(self.nx):
                tmp.append(self.mesh.CreateVertex(
                    id_cnt, self.x_points[x], self.y_points[y], 0.0))
                id_cnt += 1
            self.nodes.append(tmp)

    def _create_triangular_elements(self):
        config = ElmtConfig(ShapeType.Triangle, 1, False, False)
        for y in range(self.ny - 1):
            for x in range(self.nx - 1):
                self.mesh.CreateElement(
                    config,
                    [self.nodes[y][x], self.nodes[y+1][x+1],
                     self.nodes[y+1][x]],
                    self.comp_ID)
                self.mesh.CreateElement(
                    config,
                    [self.nodes[y][x], self.nodes[y][x+1],
                     self.nodes[y+1][x+1]],
                    self.comp_ID)

    def setUp(self):
        self._initialize_static_values()
        self._create_mesh()
        self._initialize_nodes()
        self._create_triangular_elements()

    def testMeshDimensions(self):
        self.assertEqual(self.mesh.expDim,   self.expDim)
        self.assertEqual(self.mesh.spaceDim, self.spaceDim)

    def testMeshDimensionsAreTheGraphs(self):
        # expDim and spaceDim live on the MeshGraph now; the Mesh forwards.
        self.mesh.expDim   = 3
        self.mesh.spaceDim = 3
        self.assertEqual(self.mesh.graph.GetMeshDimension(), 3)
        self.assertEqual(self.mesh.graph.GetSpaceDimension(), 3)

    def testMeshVertices(self):
        self.assertEqual(len(self.mesh.GetVertices()), self.nx * self.ny)

    def testMeshElements(self):
        expected = 2 * (self.nx - 1) * (self.ny - 1)
        self.assertEqual(len(self.mesh.GetElements(2)), expected)
        self.assertEqual(self.mesh.GetNumElements(), expected)

        # Every element went into the composite it was given.
        for el in self.mesh.GetElements(2):
            self.assertEqual(self.mesh.GetTag(el), self.comp_ID)

    def testMeshElementsBadDimension(self):
        with self.assertRaises(IndexError):
            self.mesh.GetElements(4)

    def testMeshElementTags(self):
        tags = self.mesh.GetAllElementTags(2)
        self.assertEqual(len(tags), 2 * (self.nx - 1) * (self.ny - 1))

if __name__ == '__main__':
    unittest.main()
