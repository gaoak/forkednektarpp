<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Generating the same 2D boundary layer twice gives the same mesh</description>
    <segment type="sequential">
        <executable>NekMesh</executable>
        <parameters>-f 2d_bl_aerofoil.mcf 2d_bl_aerofoil_det_a.xml:xml:uncompress</parameters>
    </segment>
    <segment type="sequential">
        <executable>NekMesh</executable>
        <parameters>-f 2d_bl_aerofoil.mcf 2d_bl_aerofoil_det_b.xml:xml:uncompress</parameters>
    </segment>
    <files>
        <file description="Input File">2d_bl_aerofoil.mcf</file>
        <file description="Input File 2">2d_bl_aerofoil.stp</file>
    </files>
    <metrics>
        <metric type="filesmatch" id="1">
            <compare>
                <file>2d_bl_aerofoil_det_a.xml</file>
                <file>2d_bl_aerofoil_det_b.xml</file>
            </compare>
        </metric>
    </metrics>
</test>
