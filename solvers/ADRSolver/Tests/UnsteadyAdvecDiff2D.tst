<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>2D shared IMEX advection-diffusion</description>
    <executable>ADRSolver</executable>
    <parameters>UnsteadyAdvecDiff2D.xml</parameters>
    <files>
        <file description="Session File">UnsteadyAdvecDiff2D.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-7">2.55964e-05</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-7">3.31402e-05</value>
        </metric>
    </metrics>
</test>
