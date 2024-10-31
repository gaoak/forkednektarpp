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
            <value variable="u" tolerance="1e-4">0.000710463</value>
            <value variable="v" tolerance="1e-5">0.000150538</value>
            <value variable="w" tolerance="1e-5">0.000146027</value>
            <value variable="p" tolerance="1e-4">0.000458513</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-4">0.00222969</value>
            <value variable="v" tolerance="1e-4">0.00059224</value>
            <value variable="w" tolerance="1e-4">0.000655577</value>
            <value variable="p" tolerance="1e-4">0.00271295</value>
        </metric>
    </metrics>
</test>
