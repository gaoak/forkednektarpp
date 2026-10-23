<?xml version="1.0" encoding="utf-8"?>
<!--
    KovasznayFlowHOPBC2D with the IMEXdirk 2_3_2 scheme for two steps. The
    velocity of the curl-curl term is an extrapolation with weights summing
    to one at every stage, and the linear-implicit advection carries the
    weight of the explicit contribution it replaces; a velocity scaled by
    the implicit coefficient instead gives a pressure error of order one.
    The bounds cover the observed Serial and AVX error with a margin.
-->
<test>
    <description>KovasznayFlowHOPBC2D_IMEXdirk: high-order pressure boundary data through IMEXdirk 2_3_2 stages</description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>KovasznayFlowHOPBC2D.xml KovasznayFlowHOPBC2D_IMEXdirkConditions.xml</parameters>
    <files>
        <file description="Mesh and base session">KovasznayFlowHOPBC2D.xml</file>
        <file description="IMEXdirk conditions">KovasznayFlowHOPBC2D_IMEXdirkConditions.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="5e-5">0</value>
            <value variable="v" tolerance="5e-5">0</value>
            <value variable="p" tolerance="5e-5">0</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-4">0</value>
            <value variable="v" tolerance="1e-4">0</value>
            <value variable="p" tolerance="1e-4">0</value>
        </metric>
    </metrics>
</test>
