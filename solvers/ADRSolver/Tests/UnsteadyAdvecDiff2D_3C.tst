<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>2D shared IMEX advection-diffusion with 3 components</description>
    <executable>ADRSolver</executable>
    <parameters>UnsteadyAdvecDiff2D_3C.xml</parameters>
    <files>
        <file description="Session File">UnsteadyAdvecDiff2D_3C.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-7">2.55964e-05</value>
            <value variable="v" tolerance="5e-8">1.27982e-05</value>
            <value variable="w" tolerance="3e-8">6.3991e-06</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-7">3.31402e-05</value>
            <value variable="v" tolerance="5e-8">1.65701e-05</value>
            <value variable="w" tolerance="3e-8">8.28505e-06</value>
        </metric>
    </metrics>
</test>
