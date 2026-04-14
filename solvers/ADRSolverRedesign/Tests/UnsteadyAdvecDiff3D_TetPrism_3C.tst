<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>3D unsteady advection-diffusion on a tet-prism mesh with 3 components using a scaled polynomial MMS</description>
    <executable>ADRSolverRedesign</executable>
    <parameters>UnsteadyAdvecDiff3D_TetPrism_3C.xml</parameters>
    <files>
        <file description="Session File">UnsteadyAdvecDiff3D_TetPrism_3C.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="5e-6">0.000493713</value>
            <value variable="v" tolerance="3e-6">0.000246856</value>
            <value variable="w" tolerance="2e-6">0.000123428</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="5e-6">0.00172477</value>
            <value variable="v" tolerance="3e-6">0.000862385</value>
            <value variable="w" tolerance="2e-6">0.000431192</value>
        </metric>
    </metrics>
</test>
