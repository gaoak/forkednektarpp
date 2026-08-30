<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>gas bubble rising in liquid with surface tension</description>
    <executable>IncNavierStokesSolver</executable>
    <parameters>InterfaceAdvectionMesh2D_VCSLevelSet.xml InterfaceAdvection2D_VCSLevelSet.xml --set-start-time 0 --set-start-chknumber 0</parameters>
    <files>
        <file description="Mesh File">InterfaceAdvectionMesh2D_VCSLevelSet.xml</file>
	<file description="Session File">InterfaceAdvection2D_VCSLevelSet.xml</file>
    </files>
     <metrics>
       <metric type="L2" id="1">
            <value variable="u" tolerance="1e-6">4.99525e-06</value>
            <value variable="v" tolerance="1e-6"> 0.000644632</value>
            <value variable="phi" tolerance="1e-6">0.00562163</value>
	    <value variable="rho" tolerance="1e-2">5.61601</value>
            <value variable="visc" tolerance="1e-2">5.05947e-08</value>
            <value variable="p" tolerance="1e-4">0.0859125</value>
        </metric>
	<metric type="Linf" id="2">
            <value variable="u" tolerance="1e-6">1.54659e-05</value>
            <value variable="v" tolerance="1e-6">0.000755598</value>
            <value variable="phi" tolerance="1e-6">0.0257809</value>
            <value variable="rho" tolerance="1e-2">25.7552</value>
            <value variable="visc" tolerance="1e-2">2.32028e-07</value>
            <value variable="p" tolerance="1e-4">0.899056</value>
        </metric>
     </metrics>
</test>


