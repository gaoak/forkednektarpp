<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>3D Tet Kovasnay solution using High Order Outflow BCsd</description>
    <executable>IncNavierStokesSolver</executable>
    <parameters>Tet_Kovasnay_HOBC.xml</parameters>
    <files>
        <file description="Session File">Tet_Kovasnay_HOBC.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-6">0.000711064</value>
            <value variable="v" tolerance="1e-6">0.000147126</value>
            <value variable="w" tolerance="1e-6">0.000145532</value>
            <value variable="p" tolerance="1e-6">0.000432245</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-6">0.00228863</value>
            <value variable="v" tolerance="1e-6">0.000601019</value>
            <value variable="w" tolerance="1e-6">0.000649875</value>
            <value variable="p" tolerance="1e-6">0.0025611</value>
        </metric>
    </metrics>
</test>
