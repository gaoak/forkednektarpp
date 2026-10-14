<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
        3D Helmholtz/Steady Diffusion Reaction P=3, partitioned over 5
        processes, where every composite and boundary region is named and the
        conditions - including the periodic pair - refer to them by name
    </description>
    <executable>ADRSolver</executable>
    <parameters>--use-scotch Helmholtz3D_CubePeriodic_names.xml</parameters>
    <processes>5</processes>
    <files>
      <file description="Session File">Helmholtz3D_CubePeriodic_names.xml</file>
    </files>
    <!-- This session is Helmholtz3D_CubePeriodic.xml with names put on its
         composites and boundary regions, so the values below are that test's
         to the digit: naming a thing must not change the answer, in parallel
         any more than in serial.

         Partitioning is what makes the parallel run a different test rather
         than the same one repeated. A partition keeps only the boundary
         regions it holds part of, so on most of these five processes the
         periodic pair is split: the region is there and its partner is not.
         A name resolved when the file is read would have nothing to resolve
         against, so both the region a condition applies to and the partner of
         a periodic pair are resolved to their ids when the mesh is
         partitioned. Before that, naming a region sent every condition to
         region 0 and the run died with "Boundary region '0' appears multiple
         times". -->
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="5e-5">4.73887e-05</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="4e-3">0.00876237 </value>
        </metric>
    </metrics>
</test>
