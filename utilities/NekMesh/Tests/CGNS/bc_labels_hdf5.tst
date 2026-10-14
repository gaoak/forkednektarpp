<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
        Composite names read from CGNS survive a round trip through the HDF5
        geometry format
    </description>
    <segment type="sequential">
        <executable>NekMesh</executable>
        <parameters>linear_tet_prism_pyra_mesh.cgns bc_labels_hdf5-out.nekg</parameters>
        <processes>1</processes>
    </segment>
    <!-- Reading back the companion session of the .nekg is what exercises the
         COMPOSITE_NAME datasets; the names below can only come from there. -->
    <segment type="sequential">
        <executable>NekMesh</executable>
        <parameters>-v bc_labels_hdf5-out.xml bc_labels_hdf5-rt.xml:xml:cond=bc_labels_hdf5-cond.xml</parameters>
        <processes>1</processes>
    </segment>
    <files>
        <file description="Input File">linear_tet_prism_pyra_mesh.cgns</file>
    </files>
    <metrics>
        <metric type="regex" id="1">
            <regex>.*region (\d+) REF="Body1\.Inlet": C\[\d+\]</regex>
            <matches>
                <match>
                    <field id="0">0</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="2">
            <regex>.*region (\d+) REF="Body1\.Outlet": C\[\d+\]</regex>
            <matches>
                <match>
                    <field id="0">1</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="3">
            <regex>.*region (\d+) REF="Body1\.Wall": C\[\d+\]</regex>
            <matches>
                <match>
                    <field id="0">2</field>
                </match>
            </matches>
        </metric>
        <metric type="nowarning" id="4"/>
    </metrics>
</test>
