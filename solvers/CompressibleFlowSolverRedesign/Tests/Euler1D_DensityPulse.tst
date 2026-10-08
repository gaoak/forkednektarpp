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
            <value variable="rho" tolerance="1e-7">2.30432e-06</value>
            <value variable="rhou" tolerance="1e-7">2.30432e-06</value>
            <value variable="E" tolerance="1e-7">1.15216e-06</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-7">1.81487e-05</value>
            <value variable="rhou" tolerance="1e-7">1.81487e-05</value>
            <value variable="E" tolerance="1e-7">9.07435e-06</value>
        </metric>
    </metrics>
</test>
