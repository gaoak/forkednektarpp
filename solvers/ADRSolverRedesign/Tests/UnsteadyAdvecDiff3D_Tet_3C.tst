<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>3D unsteady advection-diffusion on a tet mesh with 3 components using a scaled polynomial MMS</description>
    <executable>ADRSolverRedesign</executable>
    <parameters>UnsteadyAdvecDiff3D_Tet_3C.xml</parameters>
    <files>
        <file description="Session File">UnsteadyAdvecDiff3D_Tet_3C.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="5e-5">0.0211474</value>
            <value variable="v" tolerance="3e-5">0.0105737</value>
            <value variable="w" tolerance="2e-5">0.00528686</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="5e-5">0.0265346</value>
            <value variable="v" tolerance="3e-5">0.0132673</value>
            <value variable="w" tolerance="2e-5">0.00663366</value>
        </metric>
    </metrics>
</test>
