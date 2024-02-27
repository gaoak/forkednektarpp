<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>NS, Couette flow with non-conformal circle rotating, exact solution, parallel</description>
    <executable>CompressibleFlowSolver</executable>
    <parameters>Movement_rotate_couette.xml</parameters>
    <processes>6</processes>
    <files>
        <file description="Session File">Movement_rotate_couette.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho" tolerance="1e-6">3.94376e-08</value>
            <value variable="rhou" tolerance="1e-4">4.63213e-05</value>
            <value variable="rhov" tolerance="1e-4">3.00717e-05</value>
            <value variable="E" tolerance="1e-2">0.011813</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-6">2.47069e-07</value>
            <value variable="rhou" tolerance="1e-3">0.00188217</value>
            <value variable="rhov" tolerance="1e-4">0.000642017</value>
            <value variable="E" tolerance="2e-2">0.0673843</value>
        </metric>
    </metrics>
</test>
