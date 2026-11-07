<?xml version='1.0' encoding='utf-8'?>
<test>
    <description>Full 3D thin-wall hex, XxtMultiLevelStaticCond</description>
    <executable>ADRSolver</executable>
    <parameters>ThinWall3DHexMesh.xml ThinWall3DHexParallel.xml</parameters>
    <files>
        <file description="Input">ThinWall3DHexMesh.xml</file>
        <file description="Input">ThinWall3DHexParallel.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="0">
            <value variable="phi" tolerance="1e-9">0</value>
        </metric>
        <metric type="Linf" id="1">
            <value variable="phi" tolerance="1e-9">0</value>
        </metric>
    </metrics>
    <processes>2</processes>
</test>