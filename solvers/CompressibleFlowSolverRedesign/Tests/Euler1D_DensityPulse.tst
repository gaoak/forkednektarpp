<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>Euler equations in one dimension: a Gaussian density pulse carried at uniform velocity and pressure across a periodic domain of 20 elements at P=4, compared with its exact translation after 200 steps. Density and momentum share one error and the energy carries half of it, as the exact solution requires. The only one-dimensional case in the suite: the rotation into the trace-normal frame has its own kernel in 1D, which nothing else exercises, and a defect in it once left the momentum and energy of the rotated state unset.</description>
    <executable>CompressibleFlowSolverRedesign</executable>
    <parameters>Euler1D_DensityPulse.xml</parameters>
    <files>
        <file description="Session File">Euler1D_DensityPulse.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho" tolerance="1e-7">7.10818e-07</value>
            <value variable="rhou" tolerance="1e-7">7.10818e-07</value>
            <value variable="E" tolerance="1e-7">3.55409e-07</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-7">2.08388e-06</value>
            <value variable="rhou" tolerance="1e-7">2.08388e-06</value>
            <value variable="E" tolerance="1e-7">1.04194e-06</value>
        </metric>
    </metrics>
</test>
