###############################################################################
##
## File: test_nekmesh_element.py
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
## Description: Unit tests for mesh elements.
##
###############################################################################

from NekPy.NekMesh import ElmtConfig, Mesh
from NekPy.LibUtilities import ShapeType, PointsType

import unittest

class TestElmtConfig(unittest.TestCase):
    def testElmtConfigConstructor(self):
        ElmtConfig(ShapeType.Triangle, 2, True, True, True,
                   PointsType.GaussLobattoLegendre,
                   PointsType.GaussGaussLegendre)

    def testNoDefaultConstructor(self):
        with self.assertRaises(TypeError):
            ElmtConfig()

class TestElement(unittest.TestCase):
    """An element is a SpatialDomains Geometry that is holded by MeshGraph. 
    The geom constructor creates it and the child edges and faces if unique.
    Last, associates it to a composite."""

    def setUp(self):
        self.mesh = Mesh()
        self.mesh.expDim   = 2
        self.mesh.spaceDim = 2
        self.x, self.y = 1.1, 2.2
        self.nodes = [
            self.mesh.CreateVertex(1, self.x,       self.y,       0.0),
            self.mesh.CreateVertex(2, self.x + 1.0, self.y,       0.0),
            self.mesh.CreateVertex(3, self.x,       self.y + 1.0, 0.0),
        ]
        self.config  = ElmtConfig(ShapeType.Triangle, 1, False, False)
        self.comp_ID = 2
        self.element = self.mesh.CreateElement(self.config, self.nodes,
                                               self.comp_ID)

    def testElementGetGlobalID(self):
        self.element.GetGlobalID()

    def testElementGetShapeDim(self):
        self.assertEqual(self.element.GetShapeDim(), 2)

    def testElementGetShapeType(self):
        self.assertEqual(self.element.GetShapeType(), ShapeType.Triangle)

    def testElementCounts(self):
        self.assertEqual(self.element.GetNumVerts(), 3)
        self.assertEqual(self.element.GetNumEdges(), 3)

    def testElementTag(self):
        self.assertEqual(self.mesh.GetTag(self.element), self.comp_ID)
        self.mesh.SetTag(self.element, 5)
        self.assertEqual(self.mesh.GetTag(self.element), 5)

    def testElementInMesh(self):
        elements = self.mesh.GetElements(2)
        self.assertEqual(len(elements), 1)
        self.assertEqual(elements[0].GetGlobalID(),
                         self.element.GetGlobalID())

    def testElementRemove(self):
        self.mesh.RemoveElement(self.element)
        self.assertEqual(len(self.mesh.GetElements(2)), 0)
        with self.assertRaises(KeyError):
            self.mesh.GetTag(self.element)

    def testElementSharesEdges(self):
        # A second triangle along the first one's edge must reuse it rather
        # than create a duplicate.
        fourth = self.mesh.CreateVertex(4, self.x + 1.0, self.y + 1.0, 0.0)
        self.mesh.CreateElement(
            self.config, [self.nodes[1], fourth, self.nodes[2]], self.comp_ID)
        self.assertEqual(len(self.mesh.GetElements(2)), 2)
        self.assertEqual(self.mesh.GetNumElements(), 2)

if __name__ == '__main__':
    unittest.main()
