<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Variational optimiser, linear elastic functional, on an all-tetrahedron cube/sphere. The strain tensor this uses is built nowhere else in the module, and three dimensions is the only place an error in it shows, so the tolerance here is tighter than the others The worst Jacobian is matched loosely: it is the quality after a fixed number of iterations, so it follows the path the optimiser took to get there, and that is not bit-identical from one architecture to the next. The count of invalid elements is the part checked exactly</description>
    <executable>NekMesh</executable>
    <parameters>varopti_cube_sphere.msh varopti_cube_sphere_linearelastic-out.xml:xml:test -v -m varopti:linearelastic:maxiter=5:nq=4</parameters>
    <files>
        <file description="Input File">varopti_cube_sphere.msh</file>
    </files>
    <metrics>
        <metric type="regex" id="0">
            <regex>^.*Worst at end\s*:\s*(-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)</regex>
            <matches>
                <match>
                    <field id="0" tolerance="5e-2">6.154222e-01</field>
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
