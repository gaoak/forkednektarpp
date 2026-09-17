<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
    ShockTube_2D_mixedMesh_AV_Phys_Mod_IP restated in scaled units, to pin the
    artificial viscosity's independence of the units a session is written in.

    The flow, the mesh and the discretisation are those of the dimensional
    case, Ducros sensor and C0 smoothing included. Only the units differ:
    density is measured in rhoStar = 1.2e5, speed in the inflow sound speed,
    and length is left alone, so the gas constant becomes 1/Gamma and the
    timestep grows by the speed scale. Every value below is the dimensional
    case's own reference value divided by the scale of its variable, rhoStar
    for rho, rhoStar times uStar for the momenta and rhoStar times uStar
    squared for the energy, and each is reproduced to all printed digits. The
    tolerances are the dimensional case's, divided the same way.

    What the case guards is the floor under the element mean density in
    GetMuAv. That floor is a fraction of the reference density rhoInf, which
    the session states in its own units, so it follows the scaling. Held at
    an absolute 1e-4 instead, the density here, around 1e-5, falls below it
    and every element is raised towards the floor: the viscosity comes out
    more than two hundred thousand times too large once rescaled, and this
    case does not merely drift but fails outright with a NaN. The dimensional
    cases cannot see any of this, because their density sits far above the
    floor wherever the floor is measured from.
    </description>
    <executable>CompressibleFlowSolver</executable>
    <parameters>ShockTube_2D_mixedMesh.xml ShockTube_2D_mixedMesh_AV_Phys_Mod_IP_Scaled.xml</parameters>
    <files>
        <file description="Mesh File">ShockTube_2D_mixedMesh.xml</file>
        <file description="Session File">ShockTube_2D_mixedMesh_AV_Phys_Mod_IP_Scaled.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho"  tolerance="9e-15">2.5196e-10</value>
            <value variable="rhou" tolerance="3e-13">5.96632e-08</value>
            <value variable="rhov" tolerance="3e-14">2.99964e-09</value>
            <value variable="E"    tolerance="7e-12">8.86078e-08</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho"  tolerance="9e-13">3.53979e-08</value>
            <value variable="rhou" tolerance="3e-12">1.89823e-06</value>
            <value variable="rhov" tolerance="3e-12">2.05252e-07</value>
            <value variable="E"    tolerance="7e-12">1.68486e-06</value>
        </metric>
    </metrics>
</test>
