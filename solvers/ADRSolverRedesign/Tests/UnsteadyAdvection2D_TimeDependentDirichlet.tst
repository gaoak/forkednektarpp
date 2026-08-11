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
            <!-- Band spans both implementations: StdMat gives 7.23e-7 and
                 SumFac 2.39e-6. Deliberately excludes zero so that a solver
                 returning no error is not silently accepted. -->
            <value variable="u" tolerance="1.2e-6">1.6e-6</value>
        </metric>
    </metrics>
</test>
