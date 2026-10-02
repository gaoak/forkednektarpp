<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>Euler equations in one dimension, WeakDG with the Roe solver at P=2: a uniform state with a step in the inflow velocity imposed through Dirichlet conditions at both ends, 40 steps. The legacy solver's Euler1D case with the equation-of-state block the redesign reads; the values are the legacy reference values, which the redesign reproduces to every printed digit. With the periodic density-pulse case this covers the one-dimensional trace flux on both its interior and its boundary path.</description>
    <executable>CompressibleFlowSolverRedesign</executable>
    <parameters>Euler1D.xml</parameters>
    <files>
        <file description="Session File">Euler1D.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho" tolerance="1e-7">1.98838e-06</value>
            <value variable="rhou" tolerance="1e-7">0.00067684</value>
            <value variable="E" tolerance="1e-7">0.575708</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-7">1.98524e-05</value>
            <value variable="rhou" tolerance="1e-7">0.0067577</value>
            <value variable="E" tolerance="1e-7">5.74799</value>
        </metric>
    </metrics>
</test>
