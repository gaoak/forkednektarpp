<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
    N-S 3D stationary shock on a tetrahedral mesh, physical artificial
    viscosity with the modal sensor and interior penalty diffusion.

    Every other shock-capturing test in this solver is two-dimensional, on
    quadrilaterals and triangles at isotropic order. That is how the
    tetrahedral bug in StdTetExp::v_ReduceOrderCoeffs survived: the modal
    sensor's truncation was misaligned for p >= 1, so the tetrahedral sensor
    was wrong and nothing noticed. This case covers that path end to end.

    The mesh is a box, ten cells along the shock normal and two across each of
    the other directions, every cell cut into six tetrahedra; see
    MakeShockTube3DTetMesh.py, which generates it and asserts both that no
    element is left-handed and that the boundary triangulation is what it
    intends. The flow is purely axial, so the slip walls in y and z are exact
    and the solution stays one-dimensional. They are slip walls rather than
    periodic faces on purpose: a physical boundary is where the artificial
    viscosity's trace value is averaged against zero, and periodic laterals
    would make those faces interior traces and hide it, as they do in the
    two-dimensional case. See the mesh generator for the periodic route,
    which needs NekMesh's peralign module.

    What the metrics separate, all measured at P5 on this mesh:

      shock capturing on        rhou 0.416008   E 234.876   rhov 0.181719
      shock capturing off       rhou 0.220162   E  79.9064  rhov 0.000435

    so the feature moves rhov by a factor of four hundred and E by a factor of
    three. The rhou tolerance below is also tight enough to see the
    ReduceOrderCoeffs fix itself: before it, this case gave rhou 0.416034 and
    E 234.884, a difference of 2.6e-05 in rhou. That margin is thin, and the
    exhaustive check of the truncation masks is the unit test
    TestReduceOrderCoeffs rather than this file; what this case adds is
    three-dimensional coverage that did not exist.
    </description>
    <executable>CompressibleFlowSolver</executable>
    <parameters>ShockTube_3D_tet.xml ShockTube_3D_tet_AV_Phys_Mod_IP.xml</parameters>
    <files>
        <file description="Mesh File">ShockTube_3D_tet.xml</file>
        <file description="Session File">ShockTube_3D_tet_AV_Phys_Mod_IP.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho" tolerance="1e-9">4.72615e-06</value>
            <value variable="rhou" tolerance="1e-5">0.416008</value>
            <value variable="rhov" tolerance="1e-6">0.181719</value>
            <value variable="rhow" tolerance="1e-6">0.181719</value>
            <value variable="E" tolerance="1e-2">234.876</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-7">0.00193434</value>
            <value variable="rhou" tolerance="1e-4">28.1872</value>
            <value variable="rhov" tolerance="1e-4">18.374</value>
            <value variable="rhow" tolerance="1e-4">18.374</value>
            <value variable="E" tolerance="1e-1">25858.4</value>
        </metric>
    </metrics>
</test>
