<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Variational optimiser straightens the interior of a curved hexahedron, taking it from a scaled Jacobian of 0.35 to 1</description>
    <executable>NekMesh</executable>
    <parameters>cube_hex_ho_volume.msh varopti_cube_hex_ho-out.xml:xml:test -v -m varopti:hyperelastic:maxiter=3:overint=2</parameters>
    <files>
        <file description="Input File">cube_hex_ho_volume.msh</file>
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
