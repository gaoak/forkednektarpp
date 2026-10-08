<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>2D shared IMEX advection-diffusion with 3 components</description>
    <executable>ADRSolverRedesign</executable>
    <parameters>UnsteadyAdvecDiff2D_3C.xml</parameters>
    <files>
        <file description="Session File">UnsteadyAdvecDiff2D_3C.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-7">1.46583e-05</value>
            <value variable="v" tolerance="5e-8">7.32916e-06</value>
            <value variable="w" tolerance="3e-8">3.66458e-06</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-7">1.74303e-05</value>
            <value variable="v" tolerance="5e-8">8.71515e-06</value>
            <value variable="w" tolerance="3e-8">4.35758e-06</value>
        </metric>
    </metrics>
</test>
