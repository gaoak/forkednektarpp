<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>Dong HOutflow Kovasznay flow with mixed boundary regions</description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>KovasznayFlow2D.xml HOutflowKovasznay2DConditions.xml</parameters>
    <files>
        <file description="Mesh and base session">KovasznayFlow2D.xml</file>
        <file description="HOutflow conditions">HOutflowKovasznay2DConditions.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-7">2.30665e-08</value>
            <value variable="v" tolerance="1e-7">5.67848e-09</value>
            <value variable="p" tolerance="1e-7">1.07686e-08</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-6">7.11625e-08</value>
            <value variable="v" tolerance="1e-6">4.5929e-08</value>
            <value variable="p" tolerance="1e-6">2.91919e-07</value>
        </metric>
    </metrics>
</test>
