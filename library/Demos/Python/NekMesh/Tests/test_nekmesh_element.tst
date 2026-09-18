<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Unit test of the Python interface for mesh elements.</description>
    <executable python="true">test_nekmesh_element.py</executable>
    <parameters>-v</parameters>
    <metrics>
        <metric type="pyunittest" id="1">
            <function>testElementCounts</function>
            <function>testElementGetGlobalID</function>
            <function>testElementGetShapeDim</function>
            <function>testElementGetShapeType</function>
            <function>testElementInMesh</function>
            <function>testElementRemove</function>
            <function>testElementSharesEdges</function>
            <function>testElementTag</function>
            <function>testElmtConfigConstructor</function>
            <function>testNoDefaultConstructor</function>
        </metric>
    </metrics>
</test>
