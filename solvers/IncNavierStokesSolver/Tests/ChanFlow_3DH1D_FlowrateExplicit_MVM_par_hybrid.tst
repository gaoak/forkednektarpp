<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Laminar Channel Flow 3DH1D, P=4, 12 Fourier modes (MVM) with flowrate control and forcing direction explicitly defined</description>
    <executable>IncNavierStokesSolver</executable>
    <parameters>--npz 2 -P IterativeSolverTolerance=1e-12 ChanFlow_3DH1D_FlowrateExplicit_MVM.xml</parameters>
    <processes>4</processes>
    <files>
        <file description="Session File">ChanFlow_3DH1D_FlowrateExplicit_MVM.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-11">3.65972e-08</value>
            <value variable="v" tolerance="1e-12">5.29172e-14</value>
            <value variable="w" tolerance="1e-12">0</value>
            <value variable="p" tolerance="2e-10">7.18894e-11</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-11">4.57522e-08</value>
            <value variable="v" tolerance="1e-12">9.14753e-14</value>
            <value variable="w" tolerance="1e-12">0</value>
            <value variable="p" tolerance="3e-10">1.0018e-10</value>
        </metric>
    </metrics>
</test>
