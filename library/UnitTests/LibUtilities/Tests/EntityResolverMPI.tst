<?xml version="1.0" encoding="utf-8" ?>
<tests>
    <test>
        <description>EntityResolver: rendezvous resolve over MPI on 2 ranks</description>
        <executable>EntityResolverMPITest</executable>
        <parameters />
        <processes>2</processes>
        <metrics>
            <metric type="regex" id="1">
                <regex>^np=(\d+)\s+(\w+)</regex>
                <matches>
                    <match>
                        <field id="0">2</field>
                        <field id="1">PASS</field>
                    </match>
                </matches>
            </metric>
        </metrics>
    </test>
    <test>
        <description>EntityResolver: rendezvous resolve over MPI on 3 ranks</description>
        <executable>EntityResolverMPITest</executable>
        <parameters />
        <processes>3</processes>
        <metrics>
            <metric type="regex" id="1">
                <regex>^np=(\d+)\s+(\w+)</regex>
                <matches>
                    <match>
                        <field id="0">3</field>
                        <field id="1">PASS</field>
                    </match>
                </matches>
            </metric>
        </metrics>
    </test>
    <test>
        <description>EntityResolver: rendezvous resolve over MPI on 5 ranks</description>
        <executable>EntityResolverMPITest</executable>
        <parameters />
        <processes>5</processes>
        <metrics>
            <metric type="regex" id="1">
                <regex>^np=(\d+)\s+(\w+)</regex>
                <matches>
                    <match>
                        <field id="0">5</field>
                        <field id="1">PASS</field>
                    </match>
                </matches>
            </metric>
        </metrics>
    </test>
    <test>
        <description>EntityResolver: rendezvous resolve over MPI on 8 ranks</description>
        <executable>EntityResolverMPITest</executable>
        <parameters />
        <processes>8</processes>
        <metrics>
            <metric type="regex" id="1">
                <regex>^np=(\d+)\s+(\w+)</regex>
                <matches>
                    <match>
                        <field id="0">8</field>
                        <field id="1">PASS</field>
                    </match>
                </matches>
            </metric>
        </metrics>
    </test>
</tests>
