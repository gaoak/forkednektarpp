<?xml version="1.0" encoding="utf-8"?>
<!--
    KovasznayFlowHOPBC2D at P=6 for twenty steps of dt=1e-4 with the
    second-order IMEX scheme. The high-order pressure boundary acceleration
    must subtract the projected Dirichlet velocity, which the predictor trace
    carries, rather than the analytic one: their difference, divided by
    lambda*dt, otherwise raises the pressure error to 9e-4 in L2 and 6e-2 in
    Linf, growing as dt decreases. The bounds cover the observed Serial,
    AVX and Device error of both formulations, 1.7e-4 in L2 and 1.1e-3 in
    Linf for the pressure, which decays further towards the steady state,
    and 9.2e-6 in L2 for v, with a margin.
-->
<test>
    <description>KovasznayFlowHOPBC2D_SmallTimeStep: high-order pressure boundary acceleration of a non-representable Dirichlet velocity at a small time step</description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>-I ProjectInitialConditions=Galerkin KovasznayFlowHOPBC2D.xml KovasznayFlowHOPBC2D_SmallTimeStep.xml</parameters>
    <files>
        <file description="Mesh and base session">KovasznayFlowHOPBC2D.xml</file>
        <file description="Expansions and small time step conditions">KovasznayFlowHOPBC2D_SmallTimeStep.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="2e-4">0</value>
            <value variable="v" tolerance="2e-5">0</value>
            <value variable="p" tolerance="3e-4">0</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="3e-4">0</value>
            <value variable="v" tolerance="1e-4">0</value>
            <value variable="p" tolerance="3e-3">0</value>
        </metric>
    </metrics>
</test>
