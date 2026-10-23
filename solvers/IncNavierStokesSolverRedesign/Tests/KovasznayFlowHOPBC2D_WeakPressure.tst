<?xml version="1.0" encoding="utf-8"?>
<!--
    KovasznayFlowHOPBC2D in the weak pressure form. The predictor, advection
    and forcing enter the pressure only through the volume term, and the
    high-order pressure boundary flux holds the viscous curl-curl term and
    the boundary acceleration alone. The bounds are an order of magnitude
    tighter than those of KovasznayFlowHOPBC2D.tst, below the strong-form
    pressure error, and cover the observed Serial and Device error with a
    margin.
-->
<test>
    <description>KovasznayFlowHOPBC2D_WeakPressure: exact flow with high-order pressure boundary data in the weak pressure form</description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>-I ProjectInitialConditions=Galerkin -I PressureFormulation=Weak KovasznayFlowHOPBC2D.xml</parameters>
    <files>
        <file description="Session File">KovasznayFlowHOPBC2D.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="5e-8">0</value>
            <value variable="v" tolerance="5e-8">0</value>
            <value variable="p" tolerance="5e-8">0</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="5e-7">0</value>
            <value variable="v" tolerance="5e-7">0</value>
            <value variable="p" tolerance="5e-7">0</value>
        </metric>
    </metrics>
</test>
