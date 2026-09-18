<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Generating the same mesh twice gives the same mesh</description>
    <segment type="sequential">
        <executable>NekMesh</executable>
        <parameters>-f 3d_bl_cyl.mcf 3d_bl_cyl_det_a.xml:xml:uncompress</parameters>
    </segment>
    <segment type="sequential">
        <executable>NekMesh</executable>
        <parameters>-f 3d_bl_cyl.mcf 3d_bl_cyl_det_b.xml:xml:uncompress</parameters>
    </segment>
    <files>
        <file description="Input File">3d_bl_cyl.mcf</file>
        <file description="Input File 2">3d_bl_cyl.stp</file>
    </files>
    <metrics>
        <metric type="filesmatch" id="1">
            <compare>
                <file>3d_bl_cyl_det_a.xml</file>
                <file>3d_bl_cyl_det_b.xml</file>
            </compare>
        </metric>
    </metrics>
</test>
