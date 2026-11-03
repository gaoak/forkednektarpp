<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> 3D Poisson with singular non-zero Neumann BCs at P=4 with corrections for non-singular solution  </description>
    <executable>ADRSolver</executable>
    <parameters>Poisson3D_singular_hex.xml</parameters>
    <files>
        <file description="Session File">Poisson3D_singular_hex.xml</file>
    </files>
    <metrics>
      <metric type="L2" id="1">
            <value variable="w" tolerance="1e-12">  3.13907e-15 </value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="w" tolerance="1e-12"> 4.66294e-15 </value>
        </metric>
    </metrics>
</test>
