<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>Kovasznay Flow P=3 with zero input to GMRES for higher modes</description>
    <executable>IncNavierStokesSolver</executable>
    <parameters>KovaFlow_m3_gmresZeroInputCheck.xml</parameters>
    <files>
        <file description="Session File">KovaFlow_m3_gmresZeroInputCheck.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-06">0.229610</value>
            <value variable="v" tolerance="1e-07">0.0263935</value>
            <value variable="w" tolerance="1e-12">0.0000000</value>
            <value variable="p" tolerance="1e-06">0.143617</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-06">0.3220390</value>
            <value variable="v" tolerance="1e-07">0.0638195</value>
            <value variable="w" tolerance="1e-12">0.0000000</value>
            <value variable="p" tolerance="1e-06">0.3691630</value>
        </metric>
    </metrics>
</test>
