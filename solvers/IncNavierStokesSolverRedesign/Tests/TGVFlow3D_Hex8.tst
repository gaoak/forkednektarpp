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
            <value variable="u" tolerance="1e-06">0.0011981</value>
            <value variable="v" tolerance="1e-06">0.0011981</value>
            <value variable="w" tolerance="1e-06">0.0013911</value>
            <value variable="p" tolerance="3e-04">0.0071000</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="3e-05">0.00038100</value>
            <value variable="v" tolerance="3e-05">0.00038100</value>
            <value variable="w" tolerance="3e-05">0.00023500</value>
            <value variable="p" tolerance="4e-04">0.00330000</value>
        </metric>
    </metrics>
</test>
