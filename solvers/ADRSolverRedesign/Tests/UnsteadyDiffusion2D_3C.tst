<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> 2D unsteady CG implicit diffusion with 3 components </description>
    <executable>ADRSolverRedesign</executable>
    <parameters> UnsteadyDiffusion2D_3C.xml</parameters>
    <files>
        <file description="Session File"> UnsteadyDiffusion2D_3C.xml </file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-8"> 1.77e-08 </value>
            <value variable="v" tolerance="5e-9"> 8.85e-09 </value>
            <value variable="w" tolerance="3e-9"> 4.43e-09 </value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="2e-8"> 5.81e-08 </value>
            <value variable="v" tolerance="1e-8"> 2.91e-08 </value>
            <value variable="w" tolerance="6e-9"> 1.46e-08 </value>
        </metric>
    </metrics>
</test>
