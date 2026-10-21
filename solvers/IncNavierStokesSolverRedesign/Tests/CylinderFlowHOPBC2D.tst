<?xml version="1.0" encoding="utf-8"?>
<!--
    The session defines a steady manufactured solution, so these metrics are
    errors rather than raw field norms. The 1e-3 limit applies independently
    to every velocity and pressure L2/Linf error.
-->
<test>
    <description>CylinderFlowHOPBC2D: manufactured flow on a faceted-cylinder mesh with non-trivial volume forcing and high-order pressure BCs</description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>CylinderFlowHOPBC2D.xml</parameters>
    <files>
        <file description="Session File">CylinderFlowHOPBC2D.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-3">0</value>
            <value variable="v" tolerance="1e-3">0</value>
            <value variable="p" tolerance="1e-3">0</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-3">0</value>
            <value variable="v" tolerance="1e-3">0</value>
            <value variable="p" tolerance="1e-3">0</value>
        </metric>
    </metrics>
</test>
