<?xml version="1.0" encoding="utf-8"?>
<!--
    Dong HOutflow Kovasznay flow with the second-order IMEX scheme for two
    steps, the first taken by its IMEXdirk startup. The velocity of the
    outflow condition is an extrapolation with weights summing to one in
    every implicit solve; a velocity scaled by the implicit coefficient
    instead gives a pressure error of order 1e-2. The linear-implicit
    advection carries the weight of the explicit contribution it replaces.
    The initial condition is the Galerkin projection of the exact solution:
    the averaged default, which is not, raises the pressure error after the
    two steps to 1.8e-7 in L2. The bounds cover the observed Serial and AVX
    error, 1.5e-8 in L2 and 4e-8 in Linf, with a margin.
-->
<test>
    <description>HOutflowKovasznay2D_IMEXOrder2: Dong HOutflow Kovasznay flow through the second-order IMEX startup</description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>-I ProjectInitialConditions=Galerkin KovasznayFlow2D.xml HOutflowKovasznay2D_IMEXOrder2Conditions.xml</parameters>
    <files>
        <file description="Mesh and base session">KovasznayFlow2D.xml</file>
        <file description="HOutflow second-order IMEX conditions">HOutflowKovasznay2D_IMEXOrder2Conditions.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-7">0</value>
            <value variable="v" tolerance="1e-7">0</value>
            <value variable="p" tolerance="1e-7">0</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-6">0</value>
            <value variable="v" tolerance="1e-6">0</value>
            <value variable="p" tolerance="1e-6">0</value>
        </metric>
    </metrics>
</test>
