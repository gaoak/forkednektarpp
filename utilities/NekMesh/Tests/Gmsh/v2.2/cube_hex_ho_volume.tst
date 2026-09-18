<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>A 27-node hexahedron keeps its interior node through a Gmsh round trip</description>
    <segment type="sequential">
        <executable>NekMesh</executable>
        <parameters>-f cube_hex_ho_volume.msh cube_hex_ho_volume_a.msh</parameters>
    </segment>
    <segment type="sequential">
        <executable>NekMesh</executable>
        <parameters>-f -m jac:list cube_hex_ho_volume_a.msh cube_hex_ho_volume_b.msh</parameters>
    </segment>
    <files>
        <file description="Input File">cube_hex_ho_volume.msh</file>
    </files>
    <metrics>
        <!-- An interior node written in the wrong place, or dropped so that
             the element is written as a type gmsh does not have, shows up
             here. -->
        <metric type="regex" id="1">
            <regex>.*Total negative Jacobians: (\d+)</regex>
            <matches>
                <match>
                    <field id="0">0</field>
                </match>
            </matches>
        </metric>
        <!-- Reading back what was written has to give the same file again. -->
        <metric type="filesmatch" id="2">
            <compare>
                <file>cube_hex_ho_volume_a.msh</file>
                <file>cube_hex_ho_volume_b.msh</file>
            </compare>
        </metric>
    </metrics>
</test>
