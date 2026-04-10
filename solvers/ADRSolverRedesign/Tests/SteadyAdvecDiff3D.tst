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
            <value variable="u" tolerance="2e-10"> 9.61368e-10 </value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1.5e-08"> 4.3645e-08 </value>
        </metric>
    </metrics>
</test>
