<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Pyramids keep their interior nodes through a Gmsh round trip, order 4</description>
    <segment type="sequential">
        <executable>NekMesh</executable>
        <parameters>-f channel_pyr.msh channel_pyr_a.msh</parameters>
    </segment>
    <segment type="sequential">
        <executable>NekMesh</executable>
        <parameters>-f -m jac:list channel_pyr_a.msh channel_pyr_b.msh</parameters>
    </segment>
    <files>
        <file description="Input File">channel_pyr.msh</file>
    </files>
    <metrics>
        <!-- The pyramid is the shape the writer got wrong: GetEdgeOrient had a
             case for every other shape, so pyramids fell through to the default
             and kept eNoOrientation, writing four of their eight edges the
             wrong way round. Nothing noticed, because until this point no test
             had ever written a pyramid back out. Reading the written file gives
             six negative Jacobians if that case is removed again. -->
        <metric type="regex" id="1">
            <regex>.*Total negative Jacobians: (\d+)</regex>
            <matches>
                <match>
                    <field id="0">0</field>
                </match>
            </matches>
        </metric>
    </metrics>
</test>
