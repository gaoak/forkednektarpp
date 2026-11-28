<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>3D Tet Kovasnay solution using GJP Stabilisation and dealiasing</description>
    <executable>IncNavierStokesSolver</executable>
    <parameters>Tet_Kovasnay_GJP.xml</parameters>
    <files>
        <file description="Session File">Tet_Kovasnay_GJP.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-8">0.00115597</value>
            <value variable="v" tolerance="1e-8">0.000535565</value>
            <value variable="w" tolerance="1e-10">0.000233246</value>
            <value variable="p" tolerance="1e-8">0.00223087</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-8">0.00900795</value>
            <value variable="v" tolerance="1e-8">0.00346088</value>
            <value variable="w" tolerance="1e-8">0.00115298</value>
            <value variable="p" tolerance="1e-8">0.0193707</value>
        </metric>
    </metrics>
</test>
