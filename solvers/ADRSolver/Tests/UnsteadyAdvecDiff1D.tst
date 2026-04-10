<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>1D shared IMEX advection-diffusion</description>
    <executable>ADRSolver</executable>
    <parameters>UnsteadyAdvecDiff1D.xml</parameters>
    <files>
        <file description="Session File">UnsteadyAdvecDiff1D.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-10">4.44963e-08</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-10">6.29279e-08</value>
        </metric>
    </metrics>
</test>
