<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>desc P=300</description>
    <executable>AcousticSolver</executable>
    <parameters>APE_LinerBC_test.xml</parameters>
    <files>
        <file description="Session File">APE_LinerBC_test.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="p" tolerance="1e-12">857.388</value>
            <value variable="u" tolerance="1e-12">0.115898</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="p" tolerance="1e-12">919.925</value>
            <value variable="u" tolerance="1e-12">0.187885</value>
        </metric>
    </metrics>
</test>

