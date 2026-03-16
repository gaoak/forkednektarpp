<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> 3D unsteady implicit diffusion with explicit reaction and three fields </description>
    <executable>ADRSolverRedesign</executable>
    <parameters> ReactionDiffusion3D_3C.xml</parameters>
    <files>
        <file description="Session File"> ReactionDiffusion3D_3C.xml </file>
    </files>
    <metrics>
        <metric type="L2" id="1">
	        <value variable="u" tolerance="1e-8"> 5.51e-08 </value>
            <value variable="v" tolerance="6e-9"> 2.76e-08 </value>
            <value variable="w" tolerance="3e-9"> 1.38e-08 </value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-7"> 5.77e-08 </value>
            <value variable="v" tolerance="6e-8"> 2.89e-08 </value>
            <value variable="w" tolerance="4e-8"> 1.45e-08 </value>
        </metric>
    </metrics>
</test>
