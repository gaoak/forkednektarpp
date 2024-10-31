<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>3D Kovasznay flow on curved Tetrahedrons with Semi-Implicit GJP stabilisation</description>
    <executable>IncNavierStokesSolver</executable>
    <parameters>KovaFlow_Tet_Curved.xml -I GJPStabilisation=SemiImplicit</parameters>
    <files>
        <file description="Session File">KovaFlow_Tet_Curved.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-14">0.000661101</value>
            <value variable="v" tolerance="1e-14">0.000269562</value>
            <value variable="w" tolerance="1e-14">0.000144203</value>
            <value variable="p" tolerance="1e-14">0.00211955</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-14">0.00490722</value>
            <value variable="v" tolerance="1e-14">0.00273921</value>
            <value variable="w" tolerance="1e-14">0.000692364</value>
            <value variable="p" tolerance="1e-14">0.0108018</value>
        </metric>
    </metrics>
</test>
