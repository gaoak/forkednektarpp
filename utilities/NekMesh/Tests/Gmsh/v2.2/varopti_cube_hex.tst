<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Variational optimiser on a hexahedral mesh. The mesh is straight-sided so every element is already perfect and nothing should move; the ideal mapping for a hexahedron was never reachable before, and had one vertex written twice and another not at all, which made it singular</description>
    <executable>NekMesh</executable>
    <parameters>cube_hex_lin.msh varopti_cube_hex-out.xml:xml:test -v -m varopti:hyperelastic:maxiter=3:nq=4:overint=2</parameters>
    <files>
        <file description="Input File">cube_hex_lin.msh</file>
    </files>
    <metrics>
        <metric type="regex" id="0">
            <regex>^.*Worst at end\s*:\s*(-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)</regex>
            <matches>
                <match>
                    <field id="0" tolerance="1e-6">1.000000e+00</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="1">
            <regex>^.*Invalid at end\s*:\s*(\d+)</regex>
            <matches>
                <match>
                    <field id="0">0</field>
                </match>
            </matches>
        </metric>
    </metrics>
</test>
