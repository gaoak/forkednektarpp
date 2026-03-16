<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> 1D unsteady CG implicit diffusion with Redesign operators </description>
    <executable>RedesignSolver</executable>
    <parameters> ImplicitDiffusion1D.xml</parameters>
    <files>
        <file description="Session File"> ImplicitDiffusion1D.xml </file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="5e-11"> 3.49e-10 </value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="5e-11"> 3.49e-10 </value>
        </metric>
    </metrics>
</test>
