<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> 1D unsteady CG implicit diffusion with 3 components </description>
    <executable>ADRSolverRedesign</executable>
    <parameters> UnsteadyDiffusion1D_3C.xml</parameters>
    <files>
        <file description="Session File"> UnsteadyDiffusion1D_3C.xml </file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="4e-10"> 3.46e-10 </value>
            <value variable="v" tolerance="2e-10"> 1.73e-10 </value>
            <value variable="w" tolerance="1e-10"> 8.65e-11 </value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="4e-10"> 3.46e-10 </value>
            <value variable="v" tolerance="2e-10"> 1.73e-10 </value>
            <value variable="w" tolerance="1e-10"> 8.65e-11 </value>
        </metric>
    </metrics>
</test>
