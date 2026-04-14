<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>3D unsteady advection-diffusion on a prism mesh with 3 components using a scaled polynomial MMS</description>
    <executable>ADRSolverRedesign</executable>
    <parameters>UnsteadyAdvecDiff3D_Prism_3C.xml</parameters>
    <files>
        <file description="Session File">UnsteadyAdvecDiff3D_Prism_3C.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="5e-6">0.000736358</value>
            <value variable="v" tolerance="3e-6">0.000368179</value>
            <value variable="w" tolerance="2e-6">0.000184089</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="5e-6">0.0010974</value>
            <value variable="v" tolerance="3e-6">0.000548702</value>
            <value variable="w" tolerance="2e-6">0.000274351</value>
        </metric>
    </metrics>
</test>
