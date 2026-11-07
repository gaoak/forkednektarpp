<?xml version='1.0' encoding='utf-8'?>
<test>
    <description>Internal weak electric RHS with insulating, Dirichlet and Neumann boundaries; unequal phi/velocity orders</description>
    <executable>IncNavierStokesSolver</executable>
    <parameters>MHDInternalWeakNeumann.xml</parameters>
    <files>
        <file description="Session File">MHDInternalWeakNeumann.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-10">0</value>
            <value variable="v" tolerance="1e-10">0</value>
            <value variable="phi" tolerance="1e-10">0</value>
            <value variable="p" tolerance="1e-10">0</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-10">0</value>
            <value variable="v" tolerance="1e-10">0</value>
            <value variable="phi" tolerance="1e-10">0</value>
            <value variable="p" tolerance="1e-10">0</value>
        </metric>
    </metrics>
</test>