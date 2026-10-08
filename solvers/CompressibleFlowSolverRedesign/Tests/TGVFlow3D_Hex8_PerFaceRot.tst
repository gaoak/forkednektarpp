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
            <value variable="rho" tolerance="1e-12">1.28857e-08</value>
            <value variable="rhou" tolerance="1e-12">0.000984287</value>
            <value variable="rhov" tolerance="1e-12">0.000984287</value>
            <value variable="rhow" tolerance="1e-12">0.00139196</value>
            <value variable="E" tolerance="1e-12">0.00283008</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-12">8.33904e-09</value>
            <value variable="rhou" tolerance="1e-12">0.000121752</value>
            <value variable="rhov" tolerance="1e-12">0.000121752</value>
            <value variable="rhow" tolerance="1e-12">0.000241088</value>
            <value variable="E" tolerance="1e-12">0.000444461</value>
        </metric>
    </metrics>
</test>
