<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> 1D unsteady implicit diffusion with explicit reaction using Redesign operators </description>
    <executable>RedesignSolver</executable>
    <parameters> ReactionDiffusion1D.xml</parameters>
    <files>
        <file description="Session File"> ReactionDiffusion1D.xml </file>
    </files>
    <metrics>
        <metric type="L2" id="1">
	        <value variable="u" tolerance="2e-11"> 4.18007e-10 </value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="2e-11"> 4.21551e-10 </value>
        </metric>
    </metrics>
</test>
