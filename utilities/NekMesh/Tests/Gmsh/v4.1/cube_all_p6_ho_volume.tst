<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Every element shape keeps its interior nodes through a Gmsh round trip, order 6</description>
    <segment type="sequential">
        <executable>NekMesh</executable>
        <parameters>-f cube_all_p6.msh cube_all_p6_a.msh</parameters>
    </segment>
    <segment type="sequential">
        <executable>NekMesh</executable>
        <parameters>-f -m jac:list cube_all_p6_a.msh cube_all_p6_b.msh</parameters>
    </segment>
    <files>
        <file description="Input File">cube_all_p6.msh</file>
    </files>
    <metrics>
        <!-- Reading back what was written is what tests the writer. Interior
             nodes put in the wrong place tangle the element, so this catches an
             ordering that is wrong without being short of nodes: the writer had
             no pyramid case in GetEdgeOrient, which wrote four of a pyramid's
             eight edges unreversed and showed up only here.

             The mesh carries all four shapes with volume curvature, so hexes,
             prisms, pyramids and tetrahedra are all covered. Note that the file
             written is not compared against the input: elements are reoriented
             on the way through, so the round trip is not the identity, and at
             this order the coordinates drift in their last printed digit from
             the projection alone. -->
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
