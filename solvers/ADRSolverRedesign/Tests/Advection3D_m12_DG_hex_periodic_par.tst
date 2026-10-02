<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>3D unsteady DG advection, hexahedra, order 1, P=12,periodic bcs. Two ranks, the same reference as the serial run: a discontinuous-Galerkin trace crossing a partition takes its neighbour state through the exchange, and a device run must refresh its copy of the received buffer after every exchange.</description>
    <executable>ADRSolverRedesign</executable>
    <parameters>Advection3D_m12_DG_hex_periodic.xml</parameters>
    <processes>2</processes>
    <files>
        <file description="Session File">Advection3D_m12_DG_hex_periodic.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-12">0.000497955</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-03">0.000612751</value>
        </metric>
    </metrics>
</test>
