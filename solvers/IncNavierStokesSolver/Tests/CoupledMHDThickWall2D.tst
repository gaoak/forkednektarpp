<?xml version='1.0' encoding='utf-8'?>
<tests>
    <test>
        <description>CWIPI thick-wall MHD analytical solution, 1 rank(s) per solver</description>
        <segment type="parallel">
            <executable>IncNavierStokesSolver</executable>
            <parameters>--cwipi fluid fluid-mesh.xml fluid.xml</parameters>
            <processes>1</processes>
        </segment>
        <segment type="parallel">
            <executable>../ADRSolver/ADRSolver</executable>
            <parameters>--cwipi electric electric-mesh.xml electric.xml</parameters>
            <processes>1</processes>
        </segment>
        <files>
            <file description="fluid-mesh.xml">../Examples/CoupledMHDThickWall2D/fluid-mesh.xml</file>
            <file description="fluid.xml">../Examples/CoupledMHDThickWall2D/fluid.xml</file>
            <file description="electric-mesh.xml">../Examples/CoupledMHDThickWall2D/electric-mesh.xml</file>
            <file description="electric.xml">../Examples/CoupledMHDThickWall2D/electric.xml</file>
        </files>
        <metrics>
            <metric type="regex" id="0">
                <regex>^L 2 error.*:\s*([+-]?[0-9.eE+-]+).*</regex>
                <matches>
                    <match>
                        <field variable="AnalyticalError0" tolerance="1e-8">0</field>
                    </match>
                    <match>
                        <field variable="AnalyticalError1" tolerance="1e-8">0</field>
                    </match>
                    <match>
                        <field variable="AnalyticalError2" tolerance="1e-8">0</field>
                    </match>
                    <match>
                        <field variable="AnalyticalError3" tolerance="1e-8">0</field>
                    </match>
                    <match>
                        <field variable="AnalyticalError4" tolerance="1e-8">0</field>
                    </match>
                </matches>
            </metric>
            <metric type="regex" id="1">
                <regex>^Electric potential solves:\s*(\d+).*</regex>
                <matches>
                    <match>
                        <field variable="SynchronousRequests">6</field>
                    </match>
                </matches>
            </metric>
        </metrics>
    </test>
    <test>
        <description>CWIPI thick-wall MHD analytical solution, 2 rank(s) per solver</description>
        <segment type="parallel">
            <executable>IncNavierStokesSolver</executable>
            <parameters>--cwipi fluid fluid-mesh.xml fluid.xml</parameters>
            <processes>2</processes>
        </segment>
        <segment type="parallel">
            <executable>../ADRSolver/ADRSolver</executable>
            <parameters>--cwipi electric electric-mesh.xml electric.xml</parameters>
            <processes>2</processes>
        </segment>
        <files>
            <file description="fluid-mesh.xml">../Examples/CoupledMHDThickWall2D/fluid-mesh.xml</file>
            <file description="fluid.xml">../Examples/CoupledMHDThickWall2D/fluid.xml</file>
            <file description="electric-mesh.xml">../Examples/CoupledMHDThickWall2D/electric-mesh.xml</file>
            <file description="electric.xml">../Examples/CoupledMHDThickWall2D/electric.xml</file>
        </files>
        <metrics>
            <metric type="regex" id="0">
                <regex>^L 2 error.*:\s*([+-]?[0-9.eE+-]+).*</regex>
                <matches>
                    <match>
                        <field variable="AnalyticalError0" tolerance="1e-8">0</field>
                    </match>
                    <match>
                        <field variable="AnalyticalError1" tolerance="1e-8">0</field>
                    </match>
                    <match>
                        <field variable="AnalyticalError2" tolerance="1e-8">0</field>
                    </match>
                    <match>
                        <field variable="AnalyticalError3" tolerance="1e-8">0</field>
                    </match>
                    <match>
                        <field variable="AnalyticalError4" tolerance="1e-8">0</field>
                    </match>
                </matches>
            </metric>
            <metric type="regex" id="1">
                <regex>^Electric potential solves:\s*(\d+).*</regex>
                <matches>
                    <match>
                        <field variable="SynchronousRequests">6</field>
                    </match>
                </matches>
            </metric>
        </metrics>
    </test>
</tests>