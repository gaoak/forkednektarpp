<?xml version="1.0" encoding="utf-8"?>
<!--
    One step of the second-order IMEX scheme, taken by its IMEXdirk start-up,
    from the Kovasznay solution with a perturbation that is not
    divergence-free. The start-up stages keep the multistep weight of the
    divergence of the solution they start from, and the pressure error is
    that of the projection of the perturbation. The strong pressure form is
    pinned; KovasznayFlowHOPBC2D_StartupDivergence_WeakPressure covers the
    weak one, the default of the linear-implicit formulation. The values
    cover both formulations in the Serial, AVX and Device execution spaces.
-->
<test>
    <description>KovasznayFlowHOPBC2D_StartupDivergence: multistep weight of the divergence-cleaning pressure in the IMEX start-up stages</description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>-I PressureFormulation=Strong KovasznayFlowHOPBC2D.xml KovasznayFlowHOPBC2D_StartupDivergenceConditions.xml</parameters>
    <files>
        <file description="Mesh and base session">KovasznayFlowHOPBC2D.xml</file>
        <file description="Perturbed start-up conditions">KovasznayFlowHOPBC2D_StartupDivergenceConditions.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-5">0.003995</value>
            <value variable="v" tolerance="5e-6">0.000879</value>
            <value variable="p" tolerance="5e-3">0.823</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-5">0.00486</value>
            <value variable="v" tolerance="1e-5">0.00122</value>
            <value variable="p" tolerance="2e-3">0.833</value>
        </metric>
    </metrics>
</test>
