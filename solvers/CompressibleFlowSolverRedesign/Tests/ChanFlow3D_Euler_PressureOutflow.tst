<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
    3D inviscid channel with a subsonic pressure outflow.

    The Euler counterpart of ChanFlow3D_PressureOutflow, and the reason the
    advection operator needed a boundary condition hook of its own. EulerCFE
    assembled an AdvectionWeakDGOp, which had no way to attach a
    BndCondUpdateOp, so a region tagged PressureOutflow on the inviscid path
    got the raw session values instead - VALUE="0" on density and momentum,
    which is what the legacy sessions carry. With the hook removed this case
    is NaN by the first step; it is therefore a direct test of the hook and
    not only of the condition.

    Same mesh and same inflow profile as the viscous case. The walls are plain
    Dirichlet holding that profile rather than a wall condition: there is no
    inviscid Wall or Symmetry operator in the redesign yet, and the profile is
    within 3.3e-4 of zero velocity at y = +-1, so a Dirichlet wall is both
    well posed here and identical in the two solvers. That keeps the outflow
    the only thing under test.

    Nothing viscous is left in the session. Carrying mu, Pr, Twall, the
    interior penalty coefficient and the preconditioner settings would imply a
    diffusion term and a linear solve this case does not have, which is the
    same trap as section 4b of the porting notes in a different guise.

    That leaves a session both solvers accept unchanged, so the verification is
    literally the same file through each: every L2 and Linf value agrees to all
    printed digits. This file therefore carries legacy's own metrics and any
    movement is a regression. rhow is zero to round-off in both - the mesh and
    the data are invariant along z - and its tolerances accept that rather than
    assert a value.
    </description>
    <executable>CompressibleFlowSolverRedesign</executable>
    <parameters>ChanFlow3D_Euler_PressureOutflow.xml</parameters>
    <files>
        <file description="Session File">ChanFlow3D_Euler_PressureOutflow.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho"  tolerance="1e-7">0.0496258</value>
            <value variable="rhou" tolerance="1e-7">0.0510245</value>
            <value variable="rhov" tolerance="1e-9">0.000186752</value>
            <value variable="rhow" tolerance="1e-10">0.0</value>
            <value variable="E"    tolerance="1e-5">8.88778</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho"  tolerance="1e-7">0.0140138</value>
            <value variable="rhou" tolerance="1e-7">0.0164762</value>
            <value variable="rhov" tolerance="1e-9">0.00075768</value>
            <value variable="rhow" tolerance="1e-10">0.0</value>
            <value variable="E"    tolerance="1e-5">2.50974</value>
        </metric>
    </metrics>
</test>
