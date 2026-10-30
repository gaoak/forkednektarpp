<?xml version="1.0" encoding="utf-8"?>
<!--
    KovasznayFlowHOPBC2D_SmallTimeStep with a P=3 velocity and a P=2
    pressure that share their quadrature points, for twenty steps of dt=1e-4
    with the second-order IMEX scheme. The errors are spatial: 2.3e-3, 8.5e-3
    and 1.1e-2 in L2 and 1.1e-2, 1.0e-2 and 2.8e-2 in Linf for u, v and p,
    the same to six digits over Serial, AVX, StdMat and Device in both
    formulations. The bounds cover them with a margin of about 1.3. The
    Dirichlet velocity is representable in neither boundary space;
    subtracting its projection onto the pressure boundary expansion instead
    of the velocity one raises the errors to 0.11, 0.094 and 0.19 in L2 and
    0.25, 0.22 and 0.55 in Linf.
-->
<test>
    <description>KovasznayFlowHOPBC2D_SmallTimeStep_MixedOrder: high-order pressure boundary acceleration of a non-representable Dirichlet velocity at a small time step with a pressure one order below the velocity</description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>-I ProjectInitialConditions=Galerkin KovasznayFlowHOPBC2D.xml KovasznayFlowHOPBC2D_SmallTimeStep.xml KovasznayFlowHOPBC2D_MixedOrder.xml</parameters>
    <files>
        <file description="Mesh and base session">KovasznayFlowHOPBC2D.xml</file>
        <file description="Expansions and small time step conditions">KovasznayFlowHOPBC2D_SmallTimeStep.xml</file>
        <file description="Mixed-order expansions">KovasznayFlowHOPBC2D_MixedOrder.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="3e-3">0</value>
            <value variable="v" tolerance="1.1e-2">0</value>
            <value variable="p" tolerance="1.5e-2">0</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1.4e-2">0</value>
            <value variable="v" tolerance="1.4e-2">0</value>
            <value variable="p" tolerance="4e-2">0</value>
        </metric>
    </metrics>
</test>
