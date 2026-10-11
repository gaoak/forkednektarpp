///////////////////////////////////////////////////////////////////////////////
//
// File: SessionNodeXml.cpp
//
// For more information, please see: http://www.nektar.info
//
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
// Department of Aeronautics, Imperial College London (UK), and Scientific
// Computing and Imaging Institute, University of Utah (USA).
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// Description: Construction of SessionNode trees from XML.
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/SessionNodeXml.h>

#include <tinyxml.h>

namespace Nektar::LibUtilities
{

/**
 *
 */
SessionNode SessionNodeFromXml(const TiXmlElement *elmt,
                               const std::string &name)
{
    if (!elmt)
    {
        return SessionNode(name);
    }

    SessionNode node(name.empty() ? std::string(elmt->Value()) : name);

    for (const TiXmlAttribute *attr = elmt->FirstAttribute(); attr;
         attr                       = attr->Next())
    {
        node.SetAttribute(attr->Name(), attr->Value());
    }

    // An element carries text if any of its immediate children is a text
    // node; there may be more than one if the text is interrupted by a child
    // element, in which case they are concatenated in document order. This
    // matches what TiXmlElement::GetText() reports for the common case of a
    // single text node, and is better defined than it for the rest.
    std::string text;
    for (const TiXmlNode *child = elmt->FirstChild(); child;
         child                  = child->NextSibling())
    {
        if (const TiXmlText *textNode = child->ToText())
        {
            text += textNode->Value();
        }
        else if (const TiXmlElement *childElmt = child->ToElement())
        {
            node.AddChild(SessionNodeFromXml(childElmt));
        }
    }
    node.SetText(std::move(text));

    return node;
}

} // namespace Nektar::LibUtilities
