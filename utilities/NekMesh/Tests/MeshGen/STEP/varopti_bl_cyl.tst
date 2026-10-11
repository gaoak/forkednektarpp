<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Variational optimiser on a boundary-layer mesh of prisms and tetrahedra generated from CAD. The count of invalid elements is what is checked: the boundary layer's pseudo surface used to carry the CAD association of the wall it grew from, and MakeOrder projected it back down on to that wall, folding every prism stack flat. The count is not pinned, since the mesh a generator produces is not identical from one compiler to the next. This case used to check that the thousands of nodes no volume element uses were left alone, but the boundary layer no longer leaves any behind: it now takes down the edges and nodes of the surface triangulations it discards, so there are none here to leave alone. The optimiser still has to cope with them in a mesh read from a file, and nothing covers that</description>
    <executable>NekMesh</executable>
    <parameters>-v -f -m varopti:hyperelastic:maxiter=1:numthreads=4 3d_bl_cyl.mcf varopti_bl_cyl-out.xml:xml:test</parameters>
    <files>
        <file description="Input File">3d_bl_cyl.mcf</file>
        <file description="Input File">3d_bl_cyl.stp</file>
    </files>
    <metrics>
        <metric type="regex" id="1">
            <regex>^.*# invalid elements\s*:\s*(\d+).*</regex>
            <matches>
                <match>
                    <field id="0">0</field>
                </match>
            </matches>
        </metric>
    </metrics>
</test>
