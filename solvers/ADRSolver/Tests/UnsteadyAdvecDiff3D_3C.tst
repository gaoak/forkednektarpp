<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>3D shared IMEX advection-diffusion with 3 components</description>
    <executable>ADRSolver</executable>
    <parameters>UnsteadyAdvecDiff3D_3C.xml</parameters>
    <files>
        <file description="Session File">UnsteadyAdvecDiff3D_3C.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-8">8.50686e-07</value>
            <value variable="v" tolerance="1e-8">4.25343e-07</value>
            <value variable="w" tolerance="1e-8">2.12672e-07</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-8">1.37646e-06</value>
            <value variable="v" tolerance="1e-8">6.8823e-07</value>
            <value variable="w" tolerance="1e-8">3.44115e-07</value>
        </metric>
    </metrics>
</test>
