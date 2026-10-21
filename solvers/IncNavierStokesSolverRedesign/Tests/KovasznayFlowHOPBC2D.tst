<?xml version="1.0" encoding="utf-8"?>
<!--
    This steady exact solution exercises spatially varying advection and
    CurlCurl contributions on Neumann pressure boundaries. One step is enough
    to detect inconsistent pressure-boundary data while keeping the backend
    matrix inexpensive. The bounds cover the observed Serial and Device
    discretisation/iterative-solver error with a small margin.
-->
<test>
    <description>KovasznayFlowHOPBC2D: exact flow with spatially varying high-order pressure boundary data</description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>KovasznayFlowHOPBC2D.xml</parameters>
    <files>
        <file description="Session File">KovasznayFlowHOPBC2D.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-5">0</value>
            <value variable="v" tolerance="1e-5">0</value>
            <value variable="p" tolerance="1e-5">0</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-5">0</value>
            <value variable="v" tolerance="1e-5">0</value>
            <value variable="p" tolerance="1e-5">0</value>
        </metric>
    </metrics>
</test>
