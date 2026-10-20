<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>2D Helmholtz with Dirichlet, Neumann and Robin conditions read from file and multiplied by an expression, parallel</description>
    <executable>ADRSolver</executable>
    <parameters>-I GlobalSysSoln=XxtStaticCond Helmholtz2D_BndFile.xml</parameters>
    <processes>2</processes>
    <files>
        <file description="Session File">Helmholtz2D_BndFile.xml</file>
        <file description="Session File">Helmholtz2D_BndFile.bc</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-12">2.15317e-15</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-12">7.10543e-15</value>
        </metric>
    </metrics>
</test>
