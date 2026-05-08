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
            <value variable="u" tolerance="1e-10">1.25335e-09</value>
            <value variable="v" tolerance="1e-9">1.17538e-08</value>
            <value variable="p" tolerance="1e-10">8.81536e-09</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-10">2.50594e-09</value>
            <value variable="v" tolerance="1e-9">1.63561e-08</value>
            <value variable="p" tolerance="1e-9">1.62395e-08</value>
        </metric>
    </metrics>
</test>
