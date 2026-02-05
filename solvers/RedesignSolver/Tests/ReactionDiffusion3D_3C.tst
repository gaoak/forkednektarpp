<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> 3D unsteady implicit diffusion with explicit reaction and three fields </description>
    <executable>RedesignSolver</executable>
    <parameters> ReactionDiffusion3D_3C.xml</parameters>
    <files>
        <file description="Session File"> ReactionDiffusion3D_3C.xml </file>
    </files>
    <metrics>
        <metric type="L2" id="1">
	        <value variable="u" tolerance="8e-10"> 8.27564e-09 </value>
            <value variable="v" tolerance="8e-10"> 8.27564e-09 </value>
            <value variable="w" tolerance="8e-10"> 8.27564e-09 </value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="2e-9"> 2.88769e-08 </value>
            <value variable="v" tolerance="2e-9"> 2.88769e-08 </value>
            <value variable="w" tolerance="2e-9"> 2.88769e-08 </value>
        </metric>
    </metrics>
</test>
