<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
    Stagnation inflow, slip walls and a pressure outflow driving a square
    domain to uniform axial flow at Mach 0.8.

    The first legacy compressible case to port across unchanged apart from the
    time integration, and the one that makes the stagnation inflow worth having:
    the flow starts at a hundredth of its final speed and is accelerated to
    Mach 0.8 entirely by the boundary conditions. The exact solution is
    therefore a statement about the conditions rather than about the initial
    data, and the metrics assert it directly - every error zero to 1e-12 -
    rather than reproducing numbers from a reference run.

    Ported from SquareDomain_Euler_2D_AxialFlow with one change, forced. Legacy
    drives it with a CFL condition and stops on SteadyStateTol; the redesign
    supports neither, and a session naming them would have them silently
    ignored, which is the trap section 4b of the porting notes is about. A fixed
    timestep of 0.002 over 80000 steps reaches the same steady state, in about
    two seconds on one element at NUMMODES=2.

    Legacy runs this same file and converges to the same place - both reach
    1e-13 and below on every variable. The two are not compared digit for digit
    because at that level there is nothing left to compare: the values are
    round-off about zero, and the assertion that matters is that both are zero.

    Three conditions are under test at once, and the case would not run without
    all three: StagnationInflow on the inlet, Wall on the two sides,
    PressureOutflow on the outlet. The inlet also exercises the zero-direction
    convention, where the momentum entries are all zero and the flow is taken
    normal to the boundary.
    </description>
    <executable>CompressibleFlowSolverRedesign</executable>
    <parameters>SquareDomain_Euler_2D_AxialFlow.xml</parameters>
    <files>
        <file description="Session File">SquareDomain_Euler_2D_AxialFlow.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho"  tolerance="1e-12">0.0</value>
            <value variable="rhou" tolerance="1e-12">0.0</value>
            <value variable="rhov" tolerance="1e-12">0.0</value>
            <value variable="E"    tolerance="1e-12">0.0</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho"  tolerance="1e-12">0.0</value>
            <value variable="rhou" tolerance="1e-12">0.0</value>
            <value variable="rhov" tolerance="1e-12">0.0</value>
            <value variable="E"    tolerance="1e-12">0.0</value>
        </metric>
    </metrics>
</test>
