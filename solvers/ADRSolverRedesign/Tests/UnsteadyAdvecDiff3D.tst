<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> 3D shared IMEX advection-diffusion with Redesign operators </description>
    <executable>ADRSolverRedesign</executable>
    <parameters> UnsteadyAdvecDiff3D.xml</parameters>
    <files>
        <file description="Session File"> UnsteadyAdvecDiff3D.xml </file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-7">6.90773e-07</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-7">9.32914e-07</value>
        </metric>
    </metrics>
</test>
