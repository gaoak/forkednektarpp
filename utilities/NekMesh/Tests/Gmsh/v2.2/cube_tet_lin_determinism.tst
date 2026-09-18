<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Converting the same mesh twice gives the same mesh</description>
    <segment type="sequential">
        <executable>NekMesh</executable>
        <parameters>-f cube_tet_lin.msh cube_tet_lin_det_a.xml:xml:uncompress</parameters>
    </segment>
    <segment type="sequential">
        <executable>NekMesh</executable>
        <parameters>-f cube_tet_lin.msh cube_tet_lin_det_b.xml:xml:uncompress</parameters>
    </segment>
    <files>
        <file description="Input File">cube_tet_lin.msh</file>
    </files>
    <metrics>
        <metric type="filesmatch" id="1">
            <compare>
                <file>cube_tet_lin_det_a.xml</file>
                <file>cube_tet_lin_det_b.xml</file>
            </compare>
        </metric>
    </metrics>
</test>
