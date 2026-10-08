<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> 3D SteadyADR CG implicit diffusion with Redesign operators </description>
    <executable>ADRSolverRedesign</executable>
    <parameters> SteadyAdvecDiff3D.xml</parameters>
    <files>
        <file description="Session File"> SteadyAdvecDiff3D.xml </file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="2e-11"> 9.6e-10 </value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-09"> 3.6e-08 </value>
        </metric>
    </metrics>
</test>
