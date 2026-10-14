<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
        A CGNS family names the composite in preference to the boundary
        condition, and boundary conditions sharing a family are merged
    </description>
    <executable>NekMesh</executable>
    <parameters> -v -m jac:list bc_family_names.cgns bc_family_names-out.xml:xml:test </parameters>
    <files>
        <file description="Input File">bc_family_names.cgns</file>
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
        <!-- The boundary conditions in this file are named Body1.Wall,
             Body1.Inlet and Body1.Outlet, with 420, 104 and 100 faces. Family
             PID_3001 covers the wall and PID_3002 the other two, so these
             counts hold only if the family name wins over the boundary
             condition's own and the two sharing PID_3002 end up in one
             composite. -->
        <metric type="regex" id="2">
            <regex>.*# of bndry elmts in PID_3001\s*:\s*(\d+)</regex>
            <matches>
                <match>
                    <field id="0">420</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="3">
            <regex>.*# of bndry elmts in PID_3002\s*:\s*(\d+)</regex>
            <matches>
                <match>
                    <field id="0">204</field>
                </match>
            </matches>
        </metric>
        <metric type="nowarning" id="4"/>
    </metrics>
</test>
