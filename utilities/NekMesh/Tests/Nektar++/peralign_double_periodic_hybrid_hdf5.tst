<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Test PerAlign module from XML file, naming its surfaces, written to HDF5</description>
    <segment type="sequential">
        <executable>NekMesh</executable>
        <parameters>-m peralign:surf1=Down,Symmetry1:surf2=Up,Symmetry2:dir=y\;0.0,0.0,1.0:orient peralign_hybrid.xml peralign_double_periodic_hybrid_hdf5.nekg </parameters>
        <processes> 1 </processes>
    </segment>
    <segment type="sequential">
        <executable>../../solvers/ADRSolver/ADRSolver</executable>
        <parameters> --use-opt-file peralign_hybrid.opt peralign_double_periodic_hybrid_hdf5.xml peralign_hybrid_session.xml</parameters>
        <processes> 1 </processes>
    </segment>
    <files>
      <file description="Opt File">peralign_hybrid.opt</file>
      <file description="Input File">peralign_hybrid.xml</file>
      <file description="Input File">peralign_hybrid_session.xml</file>
    </files>
    <!-- The companion of peralign_double_periodic_hybrid, differing in the
         format the aligned mesh is written to and in naming the surfaces
         rather than numbering them -- surf1=3,8 is surf1=Down,Symmetry1, as
         the composites of this mesh are named -- and asserting the same
         answer: what peralign arranges is the order the two sides of a
         periodic pair are listed in, and that has to survive being written
         out whichever format it is written in.

         It did not. The HDF5 writer sorted each composite's entity IDs, and
         the pairs are made by position, so this run ended at "Unable to
         determine triangle orientation" - the geometric match of a periodic
         pair failing, with no hint that the mesh had been reordered
         underneath it. -->
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-2">0.183094</value>
        </metric>
    </metrics>
</test>
