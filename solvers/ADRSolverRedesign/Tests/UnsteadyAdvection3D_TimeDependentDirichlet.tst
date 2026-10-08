<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> 3D continuous unsteady advection with time-dependent Dirichlet BCs </description>
    <executable>ADRSolverRedesign</executable>
    <parameters>UnsteadyAdvection3D_TimeDependentDirichlet.xml</parameters>
    <files>
        <file description="Session File">UnsteadyAdvection3D_TimeDependentDirichlet.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-8">0.0</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-8">0.0</value>
        </metric>
    </metrics>
</test>
