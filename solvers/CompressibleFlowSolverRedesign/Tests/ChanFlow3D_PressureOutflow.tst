<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
    3D channel with a subsonic pressure outflow and adiabatic walls.

    Ported from the legacy CompressibleFlowSolver case ChanFlow3D_infTurbExpl
    with the synthetic eddy forcing removed, and re-meshed. The original mesh
    was 432 elements, sized for the turbulent inflow that forcing drove; with
    the forcing gone the boundary conditions this case now tests need only
    enough elements to have an interior, so it carries a structured 64 element
    mesh - 32 prisms in the near-wall rows, 32 hexes through the core, keeping
    the element mix and the periodic z direction of the original. That forcing
    (CFSSyntheticTurbulence, solvers/CompressibleFlowSolver/Forcing/
    ForcingCFSSyntheticEddy.cpp) has no counterpart in the redesign, which
    silently ignores the FORCING block rather than failing - so leaving it in
    the session would claim support that does not exist, and would make the
    two solvers disagree for a reason that has nothing to do with the boundary
    conditions under test. With it in place the transverse components diverge
    completely: legacy develops rhow ~ 6e-5 where the redesign leaves it at
    round-off, while rho, rhou and E still agree to five figures.

    Without it, the redesign reproduces legacy exactly - every L2 and Linf
    value to all printed digits - so this file carries legacy's own metrics and
    tolerances, and any movement is a regression rather than a discretisation
    difference. rhow is zero to round-off in both solvers; its tolerances are
    set to accept that rather than to assert a value.

    The case covers more than the outflow. It carries PressureOutflow and
    WallAdiabatic regions at once, so it also exercises attaching two boundary
    condition operators to one solver, the assertion that they claim disjoint
    regions, the normal Mach test that selects the subsonic branch, and the
    prescribed pressure being taken before the storage is seeded.
    </description>
    <executable>CompressibleFlowSolverRedesign</executable>
    <parameters>ChanFlow3D_PressureOutflow.xml</parameters>
    <files>
        <file description="Session File">ChanFlow3D_PressureOutflow.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho"  tolerance="1e-7">0.0496251</value>
            <value variable="rhou" tolerance="1e-7">0.0510133</value>
            <value variable="rhov" tolerance="1e-9">0.000194346</value>
            <value variable="rhow" tolerance="1e-10">0.0</value>
            <value variable="E"    tolerance="1e-5">8.88761</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho"  tolerance="1e-7">0.0140143</value>
            <value variable="rhou" tolerance="1e-7">0.0164827</value>
            <value variable="rhov" tolerance="1e-9">0.000793451</value>
            <value variable="rhow" tolerance="1e-10">0.0</value>
            <value variable="E"    tolerance="1e-5">2.50991</value>
        </metric>
    </metrics>
</test>
