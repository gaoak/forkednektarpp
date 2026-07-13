<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> 1D continuous unsteady advection with time-dependent Dirichlet BCs </description>
    <executable>ADRSolverRedesign</executable>
    <parameters>UnsteadyAdvection1D_TimeDependentDirichlet.xml</parameters>
    <files>
        <file description="Session File">UnsteadyAdvection1D_TimeDependentDirichlet.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-7">8.1e-7</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="5e-7">3.25e-6</value>
        </metric>
    </metrics>
</test>
