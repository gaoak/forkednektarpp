<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>1D unsteady WeakDG advection, history points in HDF5 format</description>
    <executable>ADRSolver</executable>
    <parameters>Advection1D_HistoryPoints_hdf5.xml</parameters>
    <files>
        <file description="Session File">Advection1D_HistoryPoints_hdf5.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-12">0.00960004</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-12">0.0177832</value>
        </metric>
        <metric type="FileExists" id="3">
            <file pattern=".*/Advection1D_HistoryPoints_hdf5\.h5">1</file>
            <file pattern=".*/Advection1D_HistoryPoints_hdf5\.his">1</file>
        </metric>
    </metrics>
</test>
