<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>Dong HOutflow with sign-changing outlet-normal velocity</description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>KovasznayFlow2D.xml HOutflowBackflow2DConditions.xml</parameters>
    <files>
        <file description="Mesh and base session">KovasznayFlow2D.xml</file>
        <file description="Backflow conditions">HOutflowBackflow2DConditions.xml</file>
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
