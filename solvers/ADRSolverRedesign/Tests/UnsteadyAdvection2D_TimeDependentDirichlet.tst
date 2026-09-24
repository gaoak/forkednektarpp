<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> 2D continuous unsteady advection with time-dependent Dirichlet BCs </description>
    <executable>ADRSolverRedesign</executable>
    <parameters>UnsteadyAdvection2D_TimeDependentDirichlet.xml</parameters>
    <files>
        <file description="Session File">UnsteadyAdvection2D_TimeDependentDirichlet.xml</file>
    </files>
    <!--
        The CG projection between Runge-Kutta stages is a global iterative
        solve, so the reported error tracks IterativeSolverTolerance rather
        than the discretisation: it comes out around a thousand times the
        tolerance at every level tried, which is why the session file now
        pins the tolerance instead of taking the 1e-9 default.

        Which iterate the solve stops on depends on round-off, so SumFac and
        StdMat, and -O3 and -O0, land on answers that differ by a factor of
        two or three. The windows below span that spread with headroom; do
        not tighten them without also tightening the solver tolerance.
    -->
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="5.5e-10">5.5e-10</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1.5e-9">1.5e-9</value>
        </metric>
    </metrics>
</test>
