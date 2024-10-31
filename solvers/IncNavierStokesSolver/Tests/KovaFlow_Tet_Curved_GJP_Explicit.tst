<?xml version="1.0" encoding="utf-8"?>
<test>
3   <description>3D Kovasznay flow on curved Tetrahedrons with explicit GJP stabilisation</description>
    <executable>IncNavierStokesSolver</executable>
    <parameters>KovaFlow_Tet_Curved.xml -I GJPStabilisation=Explicit -I GJPNormalVelocity=True</parameters>
    <files>
        <file description="Session File">KovaFlow_Tet_Curved.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-14">0.000310725</value>
            <value variable="v" tolerance="1e-14">0.000193532</value>
            <value variable="w" tolerance="1e-14">0.000106357</value>
            <value variable="p" tolerance="1e-14">0.000774736</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-14">0.00084534</value>
            <value variable="v" tolerance="1e-14">0.00060458</value>
            <value variable="w" tolerance="1e-14">0.000341788</value>
            <value variable="p" tolerance="1e-14">0.00744588</value>
        </metric>
    </metrics>
</test>
