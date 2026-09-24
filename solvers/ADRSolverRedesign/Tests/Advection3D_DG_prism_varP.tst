<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>3D unsteady DG advection, prism elements, checking when faces
    meet as eDir1FwdDir2_Dir2BwdDir1. The expected values are the redesign's
    own, not legacy's: legacy reaches machine zero on this manufactured
    solution and the redesign does not, because it integrates the trace flux
    on each element's own trace quadrature where legacy integrates on the
    global trace's, and on these deformed prisms the integrand is not a
    polynomial. Accepted as a design consequence - the difference converges
    away geometrically under over-integration. See
    CompressibleFlowSolverRedesign/Notes/SerialFixesToBackport.md.</description>
    <executable>ADRSolverRedesign</executable>
    <parameters>Advection3D_DG_prism_varP.xml</parameters>
    <files>
        <file description="Session and Mesh File">Advection3D_DG_prism_varP.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-11">5.96441e-07</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-10">1.08936e-04</value>
        </metric>
    </metrics>
</test>
