<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> 2D unsteady DG explicit advection</description>
    <executable>ADRSolverRedesign</executable>
    <parameters>UnsteadyAdvection2D_ExplicitSDCGaussLobattoLegendre.xml</parameters>
    <processes>1</processes>
    <files>
        <file description="Session File"> UnsteadyAdvection2D_ExplicitSDCGaussLobattoLegendre.xml </file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="5.5e-13">0.0</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="7.0e-13">0.0</value>
        </metric>
    </metrics>
</test>
