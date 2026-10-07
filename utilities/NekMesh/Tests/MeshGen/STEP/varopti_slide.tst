<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Nodes on the CAD boundary slide along it during variational optimisation rather than being held fixed. A node confined to a CAD curve or surface is counted here; it used to be none of them, which left the three CAD-constrained optimisers unreachable. How many there are follows the surface mesh, which is not identical from one compiler to the next, so the regex matches any non-zero count and captures the label instead</description>
    <executable>NekMesh</executable>
    <parameters>-v -m varopti:hyperelastic:maxiter=5 2d_circle_square.mcf varopti_slide-out.xml:xml:test</parameters>
    <files>
        <file description="Input File">2d_circle_square.mcf</file>
        <file description="Input File">2d_circle_square.stp</file>
    </files>
    <metrics>
        <metric type="regex" id="0">
            <regex>^.*- (# sliding on CAD)\s*:\s*[1-9][0-9]*\s*$</regex>
            <matches>
                <match>
                    <field id="0"># sliding on CAD</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="1">
            <regex>^.*# invalid elements\s*:\s*(\d+).*</regex>
            <matches>
                <match>
                    <field id="0">0</field>
                </match>
            </matches>
        </metric>
    </metrics>
</test>
