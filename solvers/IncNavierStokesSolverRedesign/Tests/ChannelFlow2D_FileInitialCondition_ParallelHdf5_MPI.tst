<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
        ChannelFlow2D_FileInitialCondition_ParallelFile_MPI with the
        initial condition written in the HDF5 format: each process reads
        only the data of its elements, in an import all processes take part
        in. The start state, and the metrics, are those of the serial run.
    </description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>ChannelFlow2D_FileInitialCondition.xml ChannelFlow2D_FileInitialCondition_ParallelHdf5.xml</parameters>
    <processes>3</processes>
    <files>
        <file description="Session File">ChannelFlow2D_FileInitialCondition.xml</file>
        <file description="Conditions">ChannelFlow2D_FileInitialCondition_ParallelHdf5.xml</file>
        <file description="Initial Condition">ChannelFlow2D_FileInitialCondition_ini_hdf5.fld</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-5">0.581426</value>
            <value variable="v" tolerance="1e-5">0.196632</value>
            <value variable="p" tolerance="1e-5">2.66526</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-5">0.339267</value>
            <value variable="v" tolerance="1e-5">0.142453</value>
            <value variable="p" tolerance="1e-5">1.9813</value>
        </metric>
    </metrics>
</test>
