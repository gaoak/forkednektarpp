<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
        A module given composites can be given their names instead of their
        IDs, and picks out exactly what the IDs pick out
    </description>
    <segment type="sequential">
        <executable>NekMesh</executable>
        <parameters>-m extract:surf=Body1.Wall linear_tet_prism_pyra_mesh.cgns extract_composite_name-byname.msh</parameters>
        <processes>1</processes>
    </segment>
    <segment type="sequential">
        <executable>NekMesh</executable>
        <parameters>-m extract:surf=5 linear_tet_prism_pyra_mesh.cgns extract_composite_name-byid.msh</parameters>
        <processes>1</processes>
    </segment>
    <files>
        <file description="Input File">linear_tet_prism_pyra_mesh.cgns</file>
    </files>
    <!-- Body1.Wall is composite 5 of this mesh, which is named from its CGNS
         boundary conditions, so the two runs above differ only in how the
         module was told which composite to work on and must produce the same
         surface. They are written as Gmsh rather than Nektar++ files because
         a Nektar++ file records the command line that produced it, which
         necessarily differs.

         The two forms are resolved in one place, Module::GetCompositeIds, so
         this stands for the rest of the modules taking composites: bl,
         detect, peralign, scalar and spherigon. peralign has its own test in
         Nektar++/peralign_double_periodic_hybrid_hdf5, where naming the
         surfaces has to give the same solution as numbering them. -->
    <metrics>
        <metric type="filesmatch" id="1">
            <compare>
                <file>extract_composite_name-byname.msh</file>
                <file>extract_composite_name-byid.msh</file>
            </compare>
        </metric>
        <metric type="nowarning" id="2"/>
    </metrics>
</test>
