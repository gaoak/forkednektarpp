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
            <value variable="u" tolerance="1e-6">0.000711812</value>
            <value variable="v" tolerance="1e-6">0.000148912</value>
            <value variable="w" tolerance="1e-6">0.000145902</value>
            <value variable="p" tolerance="1e-6">0.000449915</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-6">0.00221263</value>
            <value variable="v" tolerance="1e-6">0.000589802</value>
            <value variable="w" tolerance="1e-6">0.000643662</value>
            <value variable="p" tolerance="1e-6">0.00257026</value>
        </metric>
    </metrics>
</test>
