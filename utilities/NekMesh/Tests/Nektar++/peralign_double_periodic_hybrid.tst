<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Test PerAlign module from XML file</description>
    <segment type="sequential">
        <executable>NekMesh</executable>
        <parameters>-m peralign:surf1=3,8:surf2=5,9:dir=y\;0.0,0.0,1.0:orient peralign_hybrid.xml peralign_double_periodic_hybrid.xml </parameters>
        <processes> 1 </processes>
    </segment>
    <segment type="sequential">
        <executable>../../solvers/IncNavierStokesSolver/IncNavierStokesSolver</executable>
        <parameters>peralign_double_periodic_hybrid.xml peralign_hybrid_session.xml</parameters>
        <processes> 1 </processes>
    </segment>
    <files>
        <file description="Input File">peralign_hybrid.xml</file>
        <file description="Input File">peralign_hybrid_session.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-1">0.55</value>
        </metric>
    </metrics>
</test>
