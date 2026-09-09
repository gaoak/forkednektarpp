<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Nodal pyramid derivative, evenly spaced points, P = 8</description>
    <executable>NodalDemo</executable>
    <parameters>--order 8 --type 36 --deriv</parameters>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-12">0.000274048</value>
            <value variable="v" tolerance="1e-12">0.000428794</value>
            <value variable="w" tolerance="1e-11">0.00267056</value>
        </metric>
    </metrics>
</test>
