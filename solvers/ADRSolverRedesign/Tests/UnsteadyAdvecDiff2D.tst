<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> 2D UnsteadyADRRedesign CG implicit diffusion with Redesign operators </description>
    <executable>ADRSolverRedesign</executable>
    <parameters> UnsteadyAdvecDiff2D.xml</parameters>
    <files>
        <file description="Session File"> UnsteadyAdvecDiff2D.xml </file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-9"> 1.14177e-09 </value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-9"> 9.09538e-09 </value>
        </metric>
    </metrics>
</test>
