<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>1D shared IMEX advection-diffusion with 3 components</description>
    <executable>ADRSolver</executable>
    <parameters>UnsteadyAdvecDiff1D_3C.xml</parameters>
    <files>
        <file description="Session File">UnsteadyAdvecDiff1D_3C.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-10">4.44963e-08</value>
            <value variable="v" tolerance="5e-11">2.22481e-08</value>
            <value variable="w" tolerance="3e-11">1.11241e-08</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-10">6.29279e-08</value>
            <value variable="v" tolerance="5e-11">3.1464e-08</value>
            <value variable="w" tolerance="3e-11">1.5732e-08</value>
        </metric>
    </metrics>
</test>
