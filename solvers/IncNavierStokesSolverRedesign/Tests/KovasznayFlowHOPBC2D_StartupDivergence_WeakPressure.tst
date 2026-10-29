<?xml version="1.0" encoding="utf-8"?>
<!--
    KovasznayFlowHOPBC2D_StartupDivergence in the weak pressure form, which
    removes the weighted divergence of the stage's initial velocity as a
    volume forcing, not as its inner product with the gradient of the test
    functions, so that the flux of that velocity through the Neumann pressure
    boundaries is kept. The values cover both formulations in the Serial, AVX
    and Device execution spaces.
-->
<test>
    <description>KovasznayFlowHOPBC2D_StartupDivergence_WeakPressure: multistep weight of the divergence-cleaning pressure in the IMEX start-up stages in the weak pressure form</description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>-I PressureFormulation=Weak KovasznayFlowHOPBC2D.xml KovasznayFlowHOPBC2D_StartupDivergenceConditions.xml</parameters>
    <files>
        <file description="Mesh and base session">KovasznayFlowHOPBC2D.xml</file>
        <file description="Perturbed start-up conditions">KovasznayFlowHOPBC2D_StartupDivergenceConditions.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-5">0.003993</value>
            <value variable="v" tolerance="5e-6">0.00088</value>
            <value variable="p" tolerance="3e-3">0.8155</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="2e-5">0.00491</value>
            <value variable="v" tolerance="4e-5">0.00125</value>
            <value variable="p" tolerance="4e-3">0.8315</value>
        </metric>
    </metrics>
</test>
