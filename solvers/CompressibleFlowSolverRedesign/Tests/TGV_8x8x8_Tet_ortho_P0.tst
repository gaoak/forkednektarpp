<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>TGVFlow3D Test 8x8xx domain at P=0,  10 step redesign reference</description>
    <executable>CompressibleFlowSolverRedesign</executable>
    <parameters>TGV_8x8x8_Tet_ortho_P0.xml</parameters>
    <files>
        <file description="Session File">TGV_8x8x8_Tet_ortho_P0.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho" tolerance="1e-7">0.00612899</value>
            <value variable="rhou" tolerance="1e-7">0.0577146</value>
            <value variable="rhov" tolerance="1e-7">0.0571406</value>
            <value variable="rhow" tolerance="1e-7">0.0301863</value>
            <value variable="E" tolerance="1e-7">1.53622</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-7">0.00356394</value>
            <value variable="rhou" tolerance="1e-7">0.0362757</value>
            <value variable="rhov" tolerance="1e-7">0.0419989</value>
            <value variable="rhow" tolerance="1e-7">0.0117296</value>
            <value variable="E" tolerance="1e-7">0.902268</value>
        </metric>
    </metrics>
</test>

