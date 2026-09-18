<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Insert a high-order surface into a linear mesh of the same geometry</description>
    <executable>NekMesh</executable>
    <parameters>-m insertsurface:mesh=insertsurface_curved.xml -m jac:list:quality square_tri_lin.msh insertsurface_square-out.xml:xml:test</parameters>
    <files>
        <file description="Input File">square_tri_lin.msh</file>
        <file description="Surface to insert">insertsurface_curved.xml</file>
    </files>
    <metrics>
        <metric type="regex" id="1">
            <regex>.*Total negative Jacobians: (\d+)</regex>
            <matches>
                <match>
                    <field id="0">0</field>
                </match>
            </matches>
        </metric>
        <!-- insertsurface_curved.xml is this same mesh with its y=0 boundary
             curved onto 0.05*sin(PI*(x+1)/2) by the curve module, so the
             vertices coincide and only edge interiors carry curvature. The
             straight mesh integrates to 100%, so anything less than that is
             evidence the curvature actually transferred; only the integer
             part is matched to stay clear of last-digit rounding. -->
        <metric type="regex" id="2">
            <regex>.*Integration of Jacobian: (\d+)\.\d+%</regex>
            <matches>
                <match>
                    <field id="0">93</field>
                </match>
            </matches>
        </metric>
    </metrics>
</test>
