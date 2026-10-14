<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
        Boundary faces written as a MIXED section still form a boundary
        composite, named after the boundary condition covering them
    </description>
    <executable>NekMesh</executable>
    <parameters> -v -m jac:list HiOrderMesh_tet_hex_pyr_order4_midvol.cgns bc_labels_mixed_section-out.xml:xml:test </parameters>
    <files>
        <file description="Input File">HiOrderMesh_tet_hex_pyr_order4_midvol.cgns</file>
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
        <!-- Both of this file's sections are MIXED. Grouping boundary
             composites by section type alone put none of these 1218 faces in
             a composite at all, so the count guards the recovery as much as
             the name does. -->
        <metric type="regex" id="2">
            <regex>.*# of bndry elmts in Body1\s*:\s*(\d+)</regex>
            <matches>
                <match>
                    <field id="0">1218</field>
                </match>
            </matches>
        </metric>
        <metric type="nowarning" id="3"/>
    </metrics>
</test>
