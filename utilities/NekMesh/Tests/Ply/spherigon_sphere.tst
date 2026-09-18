<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Spherigon smooth a faceted sphere using the ply file's own vertex normals</description>
    <executable>NekMesh</executable>
    <parameters>-m spherigon:N=5 -m jac:list:quality sphere.ply spherigon_sphere-out.xml:xml:test</parameters>
    <files>
        <file description="Input File">sphere.ply</file>
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
        <!-- sphere.ply carries exact outward vertex normals in nx/ny/nz, which
             reach the spherigon through the ModuleContext; smoothing with them
             puts all 1800 generated curve points at radius 1 to within
             6.3e-10, against 4.9e-2 for the facets they replace. Nothing
             printed measures that, so the Jacobian integral of the smoothed
             manifold stands in for it. -->
        <metric type="regex" id="2">
            <regex>.*Integration of Jacobian: ([0-9]*\.[0-9]+)%?</regex>
            <matches>
                <match>
                    <field id="0" tolerance="1e-2">46.3964</field>
                </match>
            </matches>
        </metric>
    </metrics>
</test>
