<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>3D shared IMEX advection-diffusion with 3 components</description>
    <executable>ADRSolverRedesign</executable>
    <parameters>UnsteadyAdvecDiff3D_3C.xml</parameters>
    <files>
        <file description="Session File">UnsteadyAdvecDiff3D_3C.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-8">6.91534e-07</value>
            <value variable="v" tolerance="1e-8">3.45767e-07</value>
            <value variable="w" tolerance="1e-8">1.72884e-07</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="2e-8">9.34967e-07</value>
            <value variable="v" tolerance="1e-8">4.67484e-07</value>
            <value variable="w" tolerance="1e-8">2.33742e-07</value>
        </metric>
    </metrics>
</test>
