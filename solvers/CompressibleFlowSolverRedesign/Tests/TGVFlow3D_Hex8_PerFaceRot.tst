<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>TGVFlow3D Hex8 with the C[6] periodic faces' edge loops
    rotated one position, so the periodic pairs meet as the transposing
    Dir1BwdDir2_Dir2FwdDir1 / Dir1FwdDir2_Dir2BwdDir1 orientations. The
    geometry is untouched - only the numbering rotates - so the expected
    values are TGVFlow3D_Hex8's own: face numbering is not physics, and this
    test asserts the periodic orientation composition preserves that. The
    same property holds partitioned; the 90-degree orientations are the only
    ones that are not self-inverse, and the only witnesses to an
    inversion-sense mistake in the cross-rank composition.</description>
    <executable>CompressibleFlowSolverRedesign</executable>
    <parameters>TGVFlow3D_Hex8_PerFaceRot.xml</parameters>
    <files>
        <file description="Session File">TGVFlow3D_Hex8_PerFaceRot.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho" tolerance="1e-12">3.14864e-07</value>
            <value variable="rhou" tolerance="1e-12">0.000985134</value>
            <value variable="rhov" tolerance="1e-12">0.000985134</value>
            <value variable="rhow" tolerance="1e-12">0.00139239</value>
            <value variable="E" tolerance="1e-12">0.00286098</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-12">5.59255e-08</value>
            <value variable="rhou" tolerance="1e-12">0.000119842</value>
            <value variable="rhov" tolerance="1e-12">0.000119842</value>
            <value variable="rhow" tolerance="1e-12">0.000239681</value>
            <value variable="E" tolerance="1e-12">0.000623629</value>
        </metric>
    </metrics>
</test>
