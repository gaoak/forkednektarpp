<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Variational optimiser on a boundary-layer mesh of prisms and tetrahedra generated from CAD. The mesh graph of such a case holds thousands of nodes that no volume element uses, left behind by the surface mesh; reaching for the elements around one of those is what used to end the run, so this checks there are some and that they were left alone. The count of invalid elements is checked too: the boundary layer's pseudo surface used to carry the CAD association of the wall it grew from, and MakeOrder projected it back down on to that wall, folding every prism stack flat. Neither count is pinned, since the mesh a generator produces is not identical from one compiler to the next</description>
    <executable>NekMesh</executable>
    <parameters>-v -f -m varopti:hyperelastic:maxiter=1:numthreads=4 3d_bl_cyl.mcf varopti_bl_cyl-out.xml:xml:test</parameters>
    <files>
        <file description="Input File">3d_bl_cyl.mcf</file>
        <file description="Input File">3d_bl_cyl.stp</file>
    </files>
    <metrics>
        <metric type="regex" id="0">
            <regex>^.*- (Nodes in no element):\s*[1-9][0-9]*\s*\(left alone\)\s*$</regex>
            <matches>
                <match>
                    <field id="0">Nodes in no element</field>
                </match>
            </matches>
        </metric>
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
