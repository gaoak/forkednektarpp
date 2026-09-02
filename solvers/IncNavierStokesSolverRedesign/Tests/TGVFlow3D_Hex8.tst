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
            <value variable="u" tolerance="1e-06">0.0011977</value>
            <value variable="v" tolerance="1e-06">0.0011977</value>
            <value variable="w" tolerance="1e-06">0.0013918</value>
            <value variable="p" tolerance="6e-05">0.0065090</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="3e-06">0.00038092</value>
            <value variable="v" tolerance="3e-06">0.00038092</value>
            <value variable="w" tolerance="1e-05">0.00023589</value>
            <value variable="p" tolerance="2e-04">0.0028622</value>
        </metric>
    </metrics>
</test>
