<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Read a legacy VTK PolyData triangulation</description>
    <executable>NekMesh</executable>
    <parameters>-m jac:list tri_square.vtk tri_square-out.xml:xml:test:stats</parameters>
    <files>
        <file description="Input File">tri_square.vtk</file>
    </files>
    <metrics>
        <metric type="regex" id="1">
            <regex>.*Total negative Jacobians: (\d+)</regex>
            <matches>
                <match>
                    <field id="0">0</field>
                </match>
            </matches>
        </metric>
        <!-- The edge count is the point: the eight triangles are created
             independently, so their 24 half-edges must collapse onto the
             grid's 16 distinct edges for the mesh to be connected. -->
        <metric type="regex" id="2">
            <regex>^.*Node count *: (\d+)</regex>
            <matches>
                <match>
                    <field id="0">9</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="3">
            <regex>^.*Edge count *: (\d+)</regex>
            <matches>
                <match>
                    <field id="0">16</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="4">
            <regex>^.*Elements *: (\d+)</regex>
            <matches>
                <match>
                    <field id="0">8</field>
                </match>
            </matches>
        </metric>
    </metrics>
</test>
