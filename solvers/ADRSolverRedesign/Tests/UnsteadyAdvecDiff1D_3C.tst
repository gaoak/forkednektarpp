<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>1D shared IMEX advection-diffusion with 3 components</description>
    <executable>ADRSolverRedesign</executable>
    <parameters>UnsteadyAdvecDiff1D_3C.xml</parameters>
    <files>
        <file description="Session File">UnsteadyAdvecDiff1D_3C.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-10">2.89649e-10</value>
            <value variable="v" tolerance="5e-11">1.44824e-10</value>
            <value variable="w" tolerance="3e-11">7.24122e-11</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-10">4.09715e-10</value>
            <value variable="v" tolerance="5e-11">2.04858e-10</value>
            <value variable="w" tolerance="3e-11">1.02429e-10</value>
        </metric>
    </metrics>
</test>
