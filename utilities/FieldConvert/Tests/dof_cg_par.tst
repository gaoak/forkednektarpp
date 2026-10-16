<?xml version='1.0' encoding='utf-8'?>
<test>
    <description>CG global DOF with mixed variable orders, shared boundary vertices and sparse region IDs</description>
    <executable>FieldConvert</executable>
    <parameters>-m dof dof_cg.xml out.stdout</parameters>
    <processes>2</processes>
    <files>
        <file description="Session File">dof_cg.xml</file>
    </files>
    <metrics>
        <metric type="Regex" id="1">
            <regex>Number of DOF for variable (\w+): (\d+)</regex>
            <matches>
                <match>
                    <field id="0">u</field>
                    <field id="1">28</field>
                </match>
                <match>
                    <field id="0">p</field>
                    <field id="1">15</field>
                </match>
            </matches>
        </metric>
        <metric type="Regex" id="2">
            <regex>Number of DOF for variable (\w+) on boundary (\d+): (\d+)</regex>
            <matches>
                <match>
                    <field id="0">u</field>
                    <field id="1">2</field>
                    <field id="2">4</field>
                </match>
                <match>
                    <field id="0">u</field>
                    <field id="1">7</field>
                    <field id="2">4</field>
                </match>
                <match>
                    <field id="0">u</field>
                    <field id="1">11</field>
                    <field id="2">7</field>
                </match>
                <match>
                    <field id="0">u</field>
                    <field id="1">20</field>
                    <field id="2">4</field>
                </match>
                <match>
                    <field id="0">u</field>
                    <field id="1">31</field>
                    <field id="2">4</field>
                </match>
                <match>
                    <field id="0">p</field>
                    <field id="1">2</field>
                    <field id="2">3</field>
                </match>
                <match>
                    <field id="0">p</field>
                    <field id="1">7</field>
                    <field id="2">3</field>
                </match>
                <match>
                    <field id="0">p</field>
                    <field id="1">11</field>
                    <field id="2">5</field>
                </match>
                <match>
                    <field id="0">p</field>
                    <field id="1">20</field>
                    <field id="2">3</field>
                </match>
                <match>
                    <field id="0">p</field>
                    <field id="1">31</field>
                    <field id="2">3</field>
                </match>
            </matches>
        </metric>
        <metric type="Regex" id="3">
            <regex>Total number of DOF on boundary (\d+): (\d+)</regex>
            <matches>
                <match>
                    <field id="0">2</field>
                    <field id="1">7</field>
                </match>
                <match>
                    <field id="0">7</field>
                    <field id="1">7</field>
                </match>
                <match>
                    <field id="0">11</field>
                    <field id="1">12</field>
                </match>
                <match>
                    <field id="0">20</field>
                    <field id="1">7</field>
                </match>
                <match>
                    <field id="0">31</field>
                    <field id="1">7</field>
                </match>
            </matches>
        </metric>
        <metric type="Regex" id="4">
            <regex>Total number of DOF: (\d+)</regex>
            <matches>
                <match>
                    <field id="0">43</field>
                </match>
            </matches>
        </metric>
    </metrics>
</test>