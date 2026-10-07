<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Variational optimiser on a prismatic mesh. The mesh is straight-sided, so every element is perfect and nothing should move; the optimiser used to report all sixteen as inverted, because the node list it was handed had two of each element's vertices transposed</description>
    <executable>NekMesh</executable>
    <parameters>cube_prism.msh varopti_cube_prism-out.xml:xml:test -v -m varopti:hyperelastic:maxiter=2:nq=5</parameters>
    <files>
        <file description="Input File">cube_prism.msh</file>
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
