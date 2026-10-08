<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> 2D SteadyADR CG implicit diffusion with Redesign operators </description>
    <executable>ADRSolverRedesign</executable>
    <parameters> SteadyAdvecDiff2D.xml</parameters>
    <files>
        <file description="Session File"> SteadyAdvecDiff2D.xml </file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-11"> 2.3e-10 </value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-10"> 1.7e-09 </value>
        </metric>
    </metrics>
</test>
