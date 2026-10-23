<?xml version="1.0" encoding="utf-8"?>
<!--
    KovasznayFlowHOPBC2D with the second-order IMEX scheme for two steps,
    the first taken by its IMEXdirk startup. The velocity of the curl-curl
    term is an extrapolation with weights summing to one in every implicit
    solve, and the linear-implicit advection carries the weight of the
    explicit contribution it replaces; a velocity scaled by the implicit
    coefficient instead gives a pressure error of order one. The bounds
    cover the observed Serial and AVX error with a margin.
-->
<test>
    <description>KovasznayFlowHOPBC2D_IMEXOrder2: high-order pressure boundary data through the second-order IMEX startup</description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>KovasznayFlowHOPBC2D.xml KovasznayFlowHOPBC2D_IMEXOrder2Conditions.xml</parameters>
    <files>
        <file description="Mesh and base session">KovasznayFlowHOPBC2D.xml</file>
        <file description="Second-order IMEX conditions">KovasznayFlowHOPBC2D_IMEXOrder2Conditions.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-5">0</value>
            <value variable="v" tolerance="1e-5">0</value>
            <value variable="p" tolerance="1e-5">0</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="5e-5">0</value>
            <value variable="v" tolerance="5e-5">0</value>
            <value variable="p" tolerance="5e-5">0</value>
        </metric>
    </metrics>
</test>
