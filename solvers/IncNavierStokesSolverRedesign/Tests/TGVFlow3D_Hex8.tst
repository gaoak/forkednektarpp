<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>TGVFlow3D Hex8 10-step redesign reference</description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>-P NumSteps=10 TGVFlow3D_Hex8.xml</parameters>
    <files>
        <file description="Session File">TGVFlow3D_Hex8.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-06">0.00119764</value>
            <value variable="v" tolerance="1e-06">0.00119764</value>
            <value variable="w" tolerance="1e-06">0.00139181</value>
            <value variable="p" tolerance="1e-03">5.907</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-06">0.00038162</value>
            <value variable="v" tolerance="1e-06">0.00038162</value>
            <value variable="w" tolerance="1e-06">0.000239257</value>
            <value variable="p" tolerance="1e-03">0.377823</value>
        </metric>
    </metrics>
</test>
