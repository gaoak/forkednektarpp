<?xml version='1.0' encoding='utf-8'?>
<test>
    <description>Full 3D thin-wall surface stiffness, ThinWall3DHexFull</description>
    <executable>ADRSolver</executable>
    <parameters>ThinWall3DHexMesh.xml ThinWall3DHexFull.xml</parameters>
    <files>
        <file description="Input">ThinWall3DHexMesh.xml</file>
        <file description="Input">ThinWall3DHexFull.xml</file>
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