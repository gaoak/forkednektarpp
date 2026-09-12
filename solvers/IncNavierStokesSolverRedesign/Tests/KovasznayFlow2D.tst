<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>KovasznayFlow2D</description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>KovasznayFlow2D.xml</parameters>
    <files>
        <file description="Session File">KovasznayFlow2D.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="2e-10">1.15000e-09</value>
            <value variable="v" tolerance="5e-11">1.17629e-08</value>
            <value variable="p" tolerance="2e-11">8.51900e-09</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="2e-10">2.79370e-09</value>
            <value variable="v" tolerance="1e-10">1.63561e-08</value>
            <value variable="p" tolerance="1.5e-10">2.03660e-08</value>
        </metric>
    </metrics>
</test>
