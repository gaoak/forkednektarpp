<?xml version="1.0" encoding="utf-8"?>
<!--
    HOutflowKovasznay2D in the weak pressure form. The outflow pressure is a
    Dirichlet value, on which the weak form's test functions vanish, so the
    outflow condition is the same as in the strong form. The bounds cover
    the observed Serial, AVX and Device error, 1.2e-8 in L2 and 1.7e-8 in
    Linf, with a margin.
-->
<test>
    <description>Dong HOutflow Kovasznay flow with mixed boundary regions in the weak pressure form</description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>-I ProjectInitialConditions=Galerkin -I PressureFormulation=Weak KovasznayFlow2D.xml HOutflowKovasznay2DConditions.xml</parameters>
    <files>
        <file description="Mesh and base session">KovasznayFlow2D.xml</file>
        <file description="HOutflow conditions">HOutflowKovasznay2DConditions.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="3e-8">0</value>
            <value variable="v" tolerance="3e-8">0</value>
            <value variable="p" tolerance="3e-8">0</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="5e-8">0</value>
            <value variable="v" tolerance="5e-8">0</value>
            <value variable="p" tolerance="5e-8">0</value>
        </metric>
    </metrics>
</test>
