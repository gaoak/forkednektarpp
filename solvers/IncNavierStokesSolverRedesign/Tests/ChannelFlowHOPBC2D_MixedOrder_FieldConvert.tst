<?xml version="1.0" encoding="utf-8"?>
<!--
    ChannelFlowHOPBC2D_MixedOrder writes its final state with a P=2 velocity
    and a P=1 pressure on triangles and quadrilaterals. The pressure is
    written through the velocity expansion, its P=2 modes zero, so that
    every variable shares one set of element definitions. FieldConvert then
    reads all three variables and reports their norms, the only output the
    metrics see. A file with the pressure in definitions of its own loses
    the pressure in FieldConvert, which then reports no p norm. The solver
    options are given to the solver segment alone, because Tester passes
    its own to every segment and FieldConvert does not accept them.
-->
<test>
    <description>ChannelFlowHOPBC2D_MixedOrder_FieldConvert: FieldConvert reads the velocity and the lower-order pressure from the field file the solver writes</description>
    <segment type="sequential">
        <executable>IncNavierStokesSolverRedesign</executable>
        <parameters>-I Formulation=semiimplicit --opExecSpace=Serial --opImpl=SumFac ChannelFlowHOPBC2D.xml ChannelFlowHOPBC2D_MixedOrder.xml</parameters>
        <processes>1</processes>
    </segment>
    <segment type="sequential">
        <executable>../../utilities/FieldConvert/FieldConvert</executable>
        <parameters>-m printfldnorms ChannelFlowHOPBC2D.xml ChannelFlowHOPBC2D.fld stdout</parameters>
        <processes>1</processes>
    </segment>
    <files>
        <file description="Mesh and base session">ChannelFlowHOPBC2D.xml</file>
        <file description="Mixed-order expansions">ChannelFlowHOPBC2D_MixedOrder.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-9">4.9295</value>
            <value variable="v" tolerance="1e-9">0</value>
            <value variable="p" tolerance="1e-8">10.3923</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-9">2.2</value>
            <value variable="v" tolerance="1e-9">0</value>
            <value variable="p" tolerance="1e-8">6</value>
        </metric>
    </metrics>
</test>
