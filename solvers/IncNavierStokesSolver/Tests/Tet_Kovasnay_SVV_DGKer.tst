<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>3D Tet Kovasnay solution using DG SVV Kerneal and dealiasing</description>
    <executable>IncNavierStokesSolver</executable>
    <parameters>Tet_Kovasnay_SVV_DGKer.xml</parameters>
    <files>
        <file description="Session File">Tet_Kovasnay_SVV_DGKer.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-6">0.00465653</value>
            <value variable="v" tolerance="1e-6">0.000895037</value>
            <value variable="w" tolerance="1e-6">0.000611923</value>
            <value variable="p" tolerance="1e-6">0.00382918</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-6">0.0438729</value>
            <value variable="v" tolerance="1e-6">0.00392143</value>
            <value variable="w" tolerance="1e-6">0.00356926</value>
            <value variable="p" tolerance="1e-6">0.0249578</value>
        </metric>
    </metrics>
</test>
