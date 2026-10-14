<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
        A CGNS mesh in the shape a mesher writes one: a MIXED section per part
        ID, each named by its family, and one part with no boundary condition
    </description>
    <executable>NekMesh</executable>
    <parameters> -v -m jac:list mixed_pid_sections.cgns mixed_pid_sections-out.xml:xml:test </parameters>
    <files>
        <file description="Input File">mixed_pid_sections.cgns</file>
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
        <!-- Every section in this file is of type MIXED, which is how at
             least one mesher writes a per-part-ID export. Deciding whether a
             section holds boundary faces from its declared type recognised
             none of them, and the mesh converted with its volume intact and
             every inlet, outlet and wall missing: the counts below were all
             zero, and there were no boundary composites to name.

             inlet, outlet and wall are named from their Family_t rather than
             from the BC_t that points at it, which is what distinguishes
             'inlet' from the boundary condition's own name of 'inlet_1'. -->
        <metric type="regex" id="2">
            <regex>.*# of bndry elmts in inlet\s*:\s*(\d+)</regex>
            <matches>
                <match>
                    <field id="0">104</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="3">
            <regex>.*# of bndry elmts in outlet\s*:\s*(\d+)</regex>
            <matches>
                <match>
                    <field id="0">100</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="4">
            <regex>.*# of bndry elmts in wall\s*:\s*(\d+)</regex>
            <matches>
                <match>
                    <field id="0">320</field>
                </match>
            </matches>
        </metric>
        <!-- No boundary condition covers this part, so it is grouped and
             named by its section instead. A mesher leaving its periodic
             patches this way is why the section fallback has to stay. -->
        <metric type="regex" id="5">
            <regex>.*# of bndry elmts in periodic1\s*:\s*(\d+)</regex>
            <matches>
                <match>
                    <field id="0">100</field>
                </match>
            </matches>
        </metric>
        <!-- Sums to the full boundary of the mesh this was derived from, so
             neither a dropped part nor two parts merging can pass. -->
        <metric type="regex" id="6">
            <regex>.*No. of boundary elements\s*:\s*(\d+)</regex>
            <matches>
                <match>
                    <field id="0">624</field>
                </match>
            </matches>
        </metric>
        <metric type="nowarning" id="7"/>
    </metrics>
</test>
