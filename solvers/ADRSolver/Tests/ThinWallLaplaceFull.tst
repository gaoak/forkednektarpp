<?xml version='1.0' encoding='utf-8'?>
<test>
    <description>Thin conducting wall surface stiffness: exact harmonic polynomial, DirectFull</description>
    <executable>ADRSolver</executable>
    <parameters>ThinWallLaplaceFull.xml</parameters>
    <files>
        <file description="Session File">ThinWallLaplaceFull.xml</file>
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