<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Variational optimiser untangles an invalid 2D quadrilateral mesh with the distortion functional. Untangling is where the Jacobian regularisation is active and the Hessian hardest to get right: with a misweighted one this case used to drive a node to NaN and abort The optimiser is run to convergence, not stopped at a fixed iteration count: part-way through, the worst Jacobian depends on the path the optimiser took, which is not bit-identical from one architecture to the next, whereas the converged value is. The count of invalid elements is checked exactly</description>
    <executable>NekMesh</executable>
    <parameters>linearise_invalid_quad.xml varopti_untangle_quad_distortion-out.xml:xml:test -v -m varopti:roca:maxiter=200</parameters>
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
                    <field id="0" tolerance="1e-3">9.999500e-01</field>
                </match>
            </matches>
        </metric>
    </metrics>
</test>
