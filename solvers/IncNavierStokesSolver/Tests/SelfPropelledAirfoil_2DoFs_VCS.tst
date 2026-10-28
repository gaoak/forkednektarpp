<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>Formulation of a velocity correction scheme for a 2-DoFs self-propelled NACA0012 airfoil</description>
    <executable>IncNavierStokesSolver</executable>
    <parameters>SelfPropelledAirfoil_2DoFs_VCS.xml</parameters>
    <files>
        <file description="Session File">SelfPropelledAirfoil_2DoFs_VCS.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="5e-4">0.45858</value>
            <value variable="v" tolerance="5e-4">0.454873</value>
            <value variable="p" tolerance="5e+0">21423.6</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="5e-4">5.1711</value>
            <value variable="v" tolerance="5e-4">5.51151</value>
            <value variable="p" tolerance="5e+0">14158.2</value>
        </metric>
    </metrics>
</test>
