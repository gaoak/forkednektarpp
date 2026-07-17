<?xml version="1.0" encoding="utf-8"?>
<test>
  <description>FieldConvert output HDF5 in parallel</description>
  <segment type="sequential">
    <executable>FieldConvert</executable>
    <parameters>tube.xml.gz tube-hdf5.nekg:nekg:writedefaultexp</parameters>
    <processes> 6 </processes>
  </segment>
  <segment type="sequential">
    <executable>../NekMesh/NekMesh</executable>
    <parameters>-m jac tube-hdf5.xml test.stdout </parameters>
    <processes> 1 </processes>
  </segment>
  <files>
    <file description="Input File">tube.xml.gz</file>
  </files>
  <metrics>
    <metric type="regex" id="1">
      <regex>.*Total negative Jacobians: (\d+)</regex>
      <matches>
        <match>
          <field id="0">0</field>
        </match>
      </matches>
    </metric>
  </metrics>
</test>
