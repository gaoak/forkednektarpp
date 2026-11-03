<?xml version="1.0" encoding="utf-8"?>
<!--
    HOutflowBackflow2D with a P=4 velocity and a P=3 pressure that share
    their quadrature points. The linear shear flow and the zero pressure are
    exact in both spaces, and the errors stay below 3.3e-13 over Serial,
    StdMat and Device (semi-implicit); the bounds are those of
    HOutflowBackflow2D.
-->
<test>
    <description>Dong HOutflow with sign-changing outlet-normal velocity and a pressure one order below the velocity</description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>KovasznayFlow2D.xml HOutflowBackflow2DConditions.xml HOutflowBackflow2D_MixedOrder.xml</parameters>
    <files>
        <file description="Mesh and base session">KovasznayFlow2D.xml</file>
        <file description="Backflow conditions">HOutflowBackflow2DConditions.xml</file>
        <file description="Mixed-order expansions">HOutflowBackflow2D_MixedOrder.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-8">0</value>
            <value variable="v" tolerance="1e-8">0</value>
            <value variable="p" tolerance="1e-8">0</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-8">0</value>
            <value variable="v" tolerance="1e-8">0</value>
            <value variable="p" tolerance="1e-8">0</value>
        </metric>
    </metrics>
</test>
