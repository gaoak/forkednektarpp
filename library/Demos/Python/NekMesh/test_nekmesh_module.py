###############################################################################
##
## File: test_nekmesh_module.py
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
## Description: Unit tests for the Module class.
##
###############################################################################

from NekPy.LibUtilities import ShapeType, NekError
from NekPy.NekMesh import ElmtConfig, Mesh, Module, ProcessModule, \
                          ModuleType, InputModule, OutputModule
import numpy as np
import unittest
import sys


class InheritFromInputModuleTest(InputModule):
    def __init__(self, mesh):
        InputModule.__init__(self, mesh)

    def Process(self):
        # The elements, edges and faces were created by the CreateGeom(),
        # so only the renumberinga and the composites are left.
        self.ProcessVertices()
        self.ProcessElements()
        self.ProcessComposites()

class InheritFromProcessModuleTest(ProcessModule):
    def __init__(self, mesh):
        ProcessModule.__init__(self, mesh)

    def Process(self):
        self.ProcessVertices()
        self.ProcessElements()
        self.ProcessComposites()

class InheritFromOutputModuleTest(OutputModule):
    def __init__(self, mesh):
        OutputModule.__init__(self, mesh)

    def Process(self):
        self.ProcessVertices()
        self.ProcessElements()
        self.ProcessComposites()


class TestModule(unittest.TestCase):

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

    def testModuleProcessRuntimeError(self):
        mod = ProcessModule(self.mesh)
        with self.assertRaises(RuntimeError):
            mod.Process()

    def testInheritFromInputModuleTest(self):
        InheritFromInputModuleTest(self.mesh).Process()

    def testInheritFromProcessModuleTest(self):
        InheritFromProcessModuleTest(self.mesh).Process()

    def testInheritFromOutputModuleTest(self):
        InheritFromOutputModuleTest(self.mesh).Process()

    def testRemoveOrphanedEntities(self):
        # Untagging an element leaves its geometry in the graph until the
        # orphan sweep, which is a Module operation.
        class Sweeper(ProcessModule):
            def __init__(self, mesh):
                ProcessModule.__init__(self, mesh)
            def Process(self):
                self.RemoveOrphanedEntities()

        el = self.mesh.GetElements(2)[0]
        before = len(self.mesh.GetElements(2))
        self.mesh.RemoveElement(el)
        self.assertEqual(len(self.mesh.GetElements(2)), before - 1)
        Sweeper(self.mesh).Process()

    def testCreateModuleUnknownParameter(self):
        ProcessModule.Create("jac", self.mesh,
                             unknown_parameter=False).Process()

    def testCreateExceptionUnknownModule(self):
        with self.assertRaises(NekError):
            ProcessModule.Create("unknown_module", self.mesh)

    def testCreateExceptionWrongArgs(self):
        with self.assertRaises(NekError):
            InputModule.Create("xml", self.mesh)

    def testExceptionNoMesh(self):
        with self.assertRaises(NekError):
            ProcessModule.Create("jac", "wrong_argument")

if __name__ == '__main__':
    unittest.main()
