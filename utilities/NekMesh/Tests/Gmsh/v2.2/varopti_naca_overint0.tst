<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Variational optimiser with no over-integration. The integration rule then has fewer points than the element has nodes, which every triangle and tetrahedron mesh used to read past the end of its stored mappings for, and segfault on The worst Jacobian is matched loosely: it is the quality after a fixed number of iterations, so it follows the path the optimiser took to get there, and that is not bit-identical from one architecture to the next. The count of invalid elements is the part checked exactly</description>
    <executable>NekMesh</executable>
    <parameters>varopti_naca.msh varopti_naca_overint0-out.xml:xml:test -v -m varopti:hyperelastic:maxiter=5:overint=0</parameters>
    <files>
        <file description="Input File">varopti_naca.msh</file>
    </files>
    <metrics>
        <metric type="regex" id="0">
            <regex>^.*Worst at end\s*:\s*(-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)</regex>
            <matches>
                <match>
                    <field id="0" tolerance="5e-2">9.238265e-01</field>
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
