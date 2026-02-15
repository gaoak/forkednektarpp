<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> 3D UnsteadyADRRedesign CG implicit diffusion with Redesign operators </description>
    <executable>RedesignSolver</executable>
    <parameters> UnsteadyAdvecDiff3D.xml</parameters>
    <files>
        <file description="Session File"> UnsteadyAdvecDiff3D.xml </file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-9"> 8.00357e-08 </value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-9"> 8.98344e-06 </value>
        </metric>
    </metrics>
</test>
