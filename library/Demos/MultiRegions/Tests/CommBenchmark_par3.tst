<?xml version="1.0" encoding="utf-8" ?>
<tests>
    <test>
        <description>CommBenchmark: SharedIdPlan vs gslib cross-validation, CG workload, 3 ranks</description>
        <executable>CommBenchmark</executable>
        <parameters>Helmholtz3D_Hex_P6.xml --bench-workload cg --bench-repeat 2 --bench-warmup 0</parameters>
        <processes>3</processes>
        <files>
            <file description="Session File">Helmholtz3D_Hex_P6.xml</file>
        </files>
        <metrics>
            <metric type="regex" id="1">
                <regex>^CommBenchmark: np=(\d+)\s+workload=(\w+)\s+validation=(\w+)</regex>
                <matches>
                    <match>
                        <field id="0">3</field>
                        <field id="1">cg</field>
                        <field id="2">PASS</field>
                    </match>
                </matches>
            </metric>
        </metrics>
    </test>
    <test>
        <description>CommBenchmark: SharedIdPlan vs gslib cross-validation, mesh-entity workload, 3 ranks</description>
        <executable>CommBenchmark</executable>
        <parameters>Helmholtz3D_Hex_P6.xml --bench-workload entity --bench-repeat 2 --bench-warmup 0</parameters>
        <processes>3</processes>
        <files>
            <file description="Session File">Helmholtz3D_Hex_P6.xml</file>
        </files>
        <metrics>
            <metric type="regex" id="1">
                <regex>^CommBenchmark: np=(\d+)\s+workload=(\w+)\s+validation=(\w+)</regex>
                <matches>
                    <match>
                        <field id="0">3</field>
                        <field id="1">entity</field>
                        <field id="2">PASS</field>
                    </match>
                </matches>
            </metric>
        </metrics>
    </test>
</tests>
