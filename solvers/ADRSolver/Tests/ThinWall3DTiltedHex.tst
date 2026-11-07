<?xml version='1.0' encoding='utf-8'?>
<test>
    <description>Full 3D thin-wall surface stiffness, ThinWall3DTiltedHex</description>
    <executable>ADRSolver</executable>
    <parameters>ThinWall3DTiltedHexMesh.xml ThinWall3DTiltedHex.xml</parameters>
    <files>
        <file description="Input">ThinWall3DTiltedHexMesh.xml</file>
        <file description="Input">ThinWall3DTiltedHex.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="0">
            <value variable="phi" tolerance="1e-10">0</value>
        </metric>
        <metric type="Linf" id="1">
            <value variable="phi" tolerance="1e-10">0</value>
        </metric>
    </metrics>
</test>