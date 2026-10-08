<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>1D unsteady advection, DG, time-dependent Dirichlet inflow. The
    exact solution u = x - advx*t is linear and so is represented exactly; the
    error only stays at round-off if the boundary values track time, which is
    what makes this the DG counterpart of the Continuous-projection case of the
    same name.</description>
    <executable>ADRSolverRedesign</executable>
    <parameters>UnsteadyAdvection1D_TimeDependentDirichlet_DG.xml</parameters>
    <files>
        <file description="Session File">UnsteadyAdvection1D_TimeDependentDirichlet_DG.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-12">0.0</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-11">0.0</value>
        </metric>
    </metrics>
</test>
