<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> 3D Homogeneous 1D Poisson with singular non-zero Neumann BCs at P=4 with corrections for non-singular solution, in parallel </description>
    <executable>ADRSolver</executable>
    <parameters>-P IterativeSolverTolerance=1e-12 Poisson_singular_3DHomo1D.xml</parameters>
    <processes>2</processes>
    <files>
        <file description="Session File">Poisson_singular_3DHomo1D.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-12">  1.95397e-16 </value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-12"> 4.44089e-16 </value>
        </metric>
    </metrics>
</test>
