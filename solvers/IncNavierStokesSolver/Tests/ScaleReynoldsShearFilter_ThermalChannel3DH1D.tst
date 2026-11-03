<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>Scaling the number of modes in the Reynolds stress filter</description>
        <segment type="sequential">
            <executable>IncNavierStokesSolver</executable>
            <parameters>ScaleReynoldsShearFilter_ThermalChannel3DH1D.xml</parameters>
            <processes> 1 </processes>
        </segment>
        <segment type="sequential">
            <executable>../../utilities/FieldConvert/FieldConvert</executable>
            <parameters>-m printfldnorms ScaleReynoldsShearFilter_ThermalChannel3DH1D.xml ScaleReynoldsShearFilter_ThermalChannel3DH1D_stress.fld stdout</parameters>
            <processes> 1 </processes>
        </segment>
    <files>
        <file description="Session File">ScaleReynoldsShearFilter_ThermalChannel3DH1D.xml</file>
        <file description="Session File">ScaleReynoldsShearFilter_ThermalChannel3DH1D.rst</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-12">1.83108</value>
            <value variable="v" tolerance="1e-12">0.0137982</value>
            <value variable="w" tolerance="1e-12">0</value>
            <value variable="theta" tolerance="1e-12">1.43044</value>
            <value variable="p" tolerance="1e-12">285.26</value>
            <value variable="uu" tolerance="1e-12">2.28719</value>
            <value variable="uv" tolerance="1e-12">0.0127671</value>
            <value variable="uw" tolerance="1e-12">0</value>
            <value variable="utheta" tolerance="1e-12">0.00243827</value>
            <value variable="vv" tolerance="1e-9">3.35665</value>
            <value variable="vw" tolerance="1e-12">0</value>
            <value variable="vtheta" tolerance="1e-9">0.227887</value>
            <value variable="ww" tolerance="1e-12">0</value>
            <value variable="wtheta" tolerance="1e-12">0</value>
            <value variable="thetatheta" tolerance="1e-9">0.0156323</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-12">1.00174</value>
            <value variable="v" tolerance="1e-12">0.0139092</value>
            <value variable="w" tolerance="1e-12">0</value>
            <value variable="theta" tolerance="1e-12">1</value>
            <value variable="p" tolerance="1e-12">-8.22399e-14</value>
            <value variable="uu" tolerance="1e-12">1.46423</value>
            <value variable="uv" tolerance="1e-12">0.0132339</value>
            <value variable="uw" tolerance="1e-12">0</value>
            <value variable="utheta" tolerance="1e-12">0.00201541</value>
            <value variable="vv" tolerance="1e-9">2.43928</value>
            <value variable="vw" tolerance="1e-12">0</value>
            <value variable="vtheta" tolerance="1e-9">0.158707</value>
            <value variable="ww" tolerance="1e-12">0</value>
            <value variable="wtheta" tolerance="1e-12">0</value>
            <value variable="thetatheta" tolerance="1e-9">0.0103396</value>
        </metric>
    </metrics>
</test>
