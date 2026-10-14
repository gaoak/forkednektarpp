<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
        CGNS boundary conditions split a single mixed section into one named
        boundary composite each, and those names reach the conditions file
    </description>
    <executable>NekMesh</executable>
    <parameters> -v -m jac:list linear_tet_prism_pyra_mesh.cgns bc_labels-out.xml:xml:test:cond=bc_labels-cond.xml </parameters>
    <files>
        <file description="Input File">linear_tet_prism_pyra_mesh.cgns</file>
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
        <!-- The three boundary conditions of this file all live in one
             section, so they are only separable through ZoneBC_t. Counts are
             asserted so that a regression which merged them, or matched the
             wrong faces, cannot pass. -->
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
        <metric type="regex" id="5">
            <regex>.*Writing conditions file '.*' with (\d+) boundary regions.*</regex>
            <matches>
                <match>
                    <field id="0">3</field>
                </match>
            </matches>
        </metric>
        <!-- The conditions written out refer to their regions by name, not
             by number, which is the whole point of carrying the names: the
             session says which boundary it means without anyone having to
             look up what number it was given. -->
        <metric type="regex" id="8">
            <regex>.*- region \d+ REF="(Body1\.Inlet)": C\[\d+\].*</regex>
            <matches>
                <match>
                    <field id="0">Body1.Inlet</field>
                </match>
            </matches>
        </metric>
        <metric type="fileexists" id="6">
            <file pattern="^.*bc_labels-cond\.xml$">1</file>
        </metric>
        <metric type="nowarning" id="7"/>
    </metrics>
</test>
