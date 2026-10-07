<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Variational optimiser, Winslow functional, on a 2D all-triangle NACA0012 case. In two dimensions the Winslow and distortion energies differ only by a constant factor, so this has to reach the same mesh as varopti_naca_distortion The worst Jacobian is matched loosely: it is the quality after a fixed number of iterations, so it follows the path the optimiser took to get there, and that is not bit-identical from one architecture to the next. The count of invalid elements is the part checked exactly</description>
    <executable>NekMesh</executable>
    <parameters>varopti_naca.msh varopti_naca_winslow-out.xml:xml:test -v -m varopti:winslow:maxiter=5</parameters>
    <files>
        <file description="Input File">varopti_naca.msh</file>
    </files>
    <metrics>
        <metric type="regex" id="0">
            <regex>^.*Worst at end\s*:\s*(-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)</regex>
            <matches>
                <match>
                    <field id="0" tolerance="5e-2">7.055189e-01</field>
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
