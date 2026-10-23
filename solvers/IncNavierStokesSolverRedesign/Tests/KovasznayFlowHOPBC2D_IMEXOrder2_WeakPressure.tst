<?xml version="1.0" encoding="utf-8"?>
<!--
    KovasznayFlowHOPBC2D_IMEXOrder2 in the weak pressure form, where the
    high-order pressure boundary flux holds the viscous curl-curl term and
    the boundary acceleration alone. The bounds cover the observed Serial
    and AVX error with a margin.
-->
<test>
    <description>KovasznayFlowHOPBC2D_IMEXOrder2_WeakPressure: high-order pressure boundary data through the second-order IMEX startup in the weak pressure form</description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>-I PressureFormulation=Weak KovasznayFlowHOPBC2D.xml KovasznayFlowHOPBC2D_IMEXOrder2Conditions.xml</parameters>
    <files>
        <file description="Mesh and base session">KovasznayFlowHOPBC2D.xml</file>
        <file description="Second-order IMEX conditions">KovasznayFlowHOPBC2D_IMEXOrder2Conditions.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="5e-6">0</value>
            <value variable="v" tolerance="5e-6">0</value>
            <value variable="p" tolerance="5e-6">0</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="2e-5">0</value>
            <value variable="v" tolerance="2e-5">0</value>
            <value variable="p" tolerance="2e-5">0</value>
        </metric>
    </metrics>
</test>
