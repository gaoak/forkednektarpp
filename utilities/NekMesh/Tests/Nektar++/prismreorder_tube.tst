<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Reorder lines of prisms on input</description>
    <executable>NekMesh</executable>
    <parameters>-m jac:list tube.xml:xml:prismreorder prismreorder_tube-out.xml:xml:test:stats</parameters>
    <files>
        <file description="Input File">tube.xml</file>
    </files>
    <metrics>
        <!-- The jac module has to run *after* the reorder, in the same
             pipeline, for this to test anything: reordering renumbers vertices
             and rebuilds the prisms, so any geometry left holding the
             coefficients FillGeom() produced for the old arrangement reports a
             negative Jacobian. Four of these 394 prisms did. -->
        <metric type="regex" id="1">
            <regex>.*Total negative Jacobians: (\d+)</regex>
            <matches>
                <match>
                    <field id="0">0</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="2">
            <regex>^.*Elements *: (\d+)</regex>
            <matches>
                <match>
                    <field id="0">991</field>
                </match>
            </matches>
        </metric>
    </metrics>
</test>
