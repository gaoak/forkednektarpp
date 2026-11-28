<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>3D Kovasznay flow on domain with all elements and some non-regular elements with Semi-Implicit GJP stabilisation</description>
    <executable>IncNavierStokesSolver</executable>
    <parameters>Kovasznay3D_AllElmts_GJP.xml</parameters>
    <files>
        <file description="Session File">Kovasznay3D_AllElmts_GJP.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-7">0.026792</value>
            <value variable="v" tolerance="1e-7">0.00985833</value>
            <value variable="w" tolerance="1e-7">0.00826654</value>
            <value variable="p" tolerance="1e-7">0.0677857</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-7">0.1858</value>
            <value variable="v" tolerance="1e-7">0.0359404</value>
            <value variable="w" tolerance="1e-7">0.0355933</value>
            <value variable="p" tolerance="1e-7">0.332075</value>
        </metric>
    </metrics>
</test>
