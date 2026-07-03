<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> 2D continuous unsteady advection with time-dependent Dirichlet BCs </description>
    <executable>ADRSolverRedesign</executable>
    <parameters>UnsteadyAdvection2D_TimeDependentDirichlet.xml</parameters>
    <files>
        <file description="Session File">UnsteadyAdvection2D_TimeDependentDirichlet.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="3e-7">6.0e-7</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-6">1.0e-6</value>
        </metric>
    </metrics>
</test>
