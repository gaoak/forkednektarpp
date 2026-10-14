<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
        3D advection with 3 zones and 2 non-conformal interfaces with periodic
        BCs on hexes on 4 processes, where the composites and boundary regions
        are named and the interfaces and conditions refer to them by name
    </description>
    <executable>ADRSolver</executable>
    <parameters>Movement_fixed_3D_Hex_3zones_Periodic_names_par.xml</parameters>
    <processes>4</processes>
    <files>
        <file description="Session File">Movement_fixed_3D_Hex_3zones_Periodic_names_par.xml</file>
    </files>
    <!-- Movement_fixed_3D_Hex_3zones_Periodic_par.xml with names, so the
         values below are that test's to the digit.

         What this adds to the other named parallel test is the MOVEMENT
         section. It sits outside CONDITIONS, so a partition file keeps it
         verbatim and the names in it are resolved when that file is read
         back: the composite names have to reach the partition for an
         interface to still find its side. The periodic conditions here cover
         all three directions, so every partition has a pair split across it. -->
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-6">1.58599e-06</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-5">3.03816e-05</value>
        </metric>
    </metrics>
</test>
