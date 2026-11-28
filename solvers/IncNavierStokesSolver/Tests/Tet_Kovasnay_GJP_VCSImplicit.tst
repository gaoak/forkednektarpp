<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>3D Tet Kovasnay solution using (Semi-Implicit) GJP Stabilisation and dealiasing with the Implicit VCS</description>
    <executable>IncNavierStokesSolver</executable>
    <parameters>Tet_Kovasnay_GJP.xml -I SolverType=VCSImplicit</parameters>
    <files>
        <file description="Session File">Tet_Kovasnay_GJP.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-4">0.00110661</value>
            <value variable="v" tolerance="1e-5">0.000510343</value>
            <value variable="w" tolerance="1e-5">0.000219011</value>
            <value variable="p" tolerance="1e-4">0.00225489</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-4">0.00900795</value>
            <value variable="v" tolerance="1e-4">0.00348158</value>
            <value variable="w" tolerance="1e-4">0.00112694</value>
            <value variable="p" tolerance="1e-4">0.0188701</value>
        </metric>
    </metrics>
</test>
