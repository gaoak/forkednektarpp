<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
        Boundary conditions addressed as vertex point lists pick out the same
        faces as the element-addressed form of the same mesh
    </description>
    <executable>NekMesh</executable>
    <parameters> -v -m jac:list bc_vertex_pointlist.cgns bc_vertex_pointlist-out.xml:xml:test </parameters>
    <files>
        <file description="Input File">bc_vertex_pointlist.cgns</file>
    </files>
    <metrics>
        <metric type="regex" id="1">
            <regex>.*Total negative Jacobians: (\d+)</regex>
            <matches>
                <match>
                    <field id="0">0</field>
                </match>
            </matches>
        </metric>
        <!-- This is linear_tet_prism_pyra_mesh with its boundary conditions
             rewritten over vertices rather than elements, which is the CGNS
             default addressing. The counts are those the element-addressed
             original gives, so a face is only counted here if it was selected
             by having all of its corners listed. -->
        <metric type="regex" id="2">
            <regex>.*# of bndry elmts in Body1\.Inlet\s*:\s*(\d+)</regex>
            <matches>
                <match>
                    <field id="0">104</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="3">
            <regex>.*# of bndry elmts in Body1\.Outlet\s*:\s*(\d+)</regex>
            <matches>
                <match>
                    <field id="0">100</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="4">
            <regex>.*# of bndry elmts in Body1\.Wall\s*:\s*(\d+)</regex>
            <matches>
                <match>
                    <field id="0">420</field>
                </match>
            </matches>
        </metric>
        <metric type="nowarning" id="5"/>
    </metrics>
</test>
