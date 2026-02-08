<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> 3D unsteady CG implicit diffusion with Redesign operators </description>
    <executable>RedesignSolver</executable>
    <parameters> ImplicitDiffusion3D.xml</parameters>
    <files>
        <file description="Session File"> ImplicitDiffusion3D.xml </file>
    </files>
    <metrics>
        <metric type="L2" id="1">
	        <value variable="u" tolerance="2e-8"> 1.2e-07 </value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="2e-8"> 1.1e-07 </value>
        </metric>
    </metrics>
</test>
