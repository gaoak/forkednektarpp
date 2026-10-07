<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Variational optimiser untangles an invalid 2D quadrilateral mesh with the Winslow functional. In two dimensions the Winslow and distortion energies differ by a constant factor, so this has to reach the same mesh as varopti_untangle_quad_distortion -- which it only does if both are regularised against the same mesh-wide minimum Jacobian The worst Jacobian is matched loosely: it is the quality after a fixed number of iterations, so it follows the path the optimiser took to get there, and that is not bit-identical from one architecture to the next. The count of invalid elements is the part checked exactly</description>
    <executable>NekMesh</executable>
    <parameters>linearise_invalid_quad.xml varopti_untangle_quad_winslow-out.xml:xml:test -v -m varopti:winslow:maxiter=40</parameters>
    <files>
        <file description="Input File">linearise_invalid_quad.xml</file>
    </files>
    <metrics>
        <metric type="regex" id="1">
            <regex>^.*Invalid at end\s*:\s*(\d+)</regex>
            <matches>
                <match>
                    <field id="0">0</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="0">
            <regex>^.*Worst at end\s*:\s*(-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)</regex>
            <matches>
                <match>
                    <field id="0" tolerance="5e-2">9.517119e-01</field>
                </match>
            </matches>
        </metric>
    </metrics>
</test>
