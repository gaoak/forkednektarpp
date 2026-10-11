///////////////////////////////////////////////////////////////////////////////
//
// File: SessionNodeXml.h
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

#ifndef NEKTAR_LIB_UTILITIES_BASICUTILS_SESSIONNODEXML_H
#define NEKTAR_LIB_UTILITIES_BASICUTILS_SESSIONNODEXML_H

#include <string>

#include <LibUtilities/BasicUtils/SessionNode.h>
#include <LibUtilities/LibUtilitiesDeclspec.h>

class TiXmlElement;

namespace Nektar::LibUtilities
{

/**
 * @brief Builds a format-neutral SessionNode tree from the TinyXML element
 * @p elmt and its descendants.
 *
 * Element names become node names, element attributes become node
 * attributes, and the text of an element becomes the node's text. As
 * TinyXML is configured to condense whitespace, the text of an element is
 * already stripped of leading and trailing whitespace by the time it
 * reaches here, and internal runs of whitespace have been collapsed to
 * single spaces.
 *
 * Comments, processing instructions and declarations are dropped: they
 * carry no configuration.
 *
 * @param elmt  Element to convert. May be nullptr, in which case an empty
 *              node is returned.
 * @param name  Name to give the resulting root node. If empty, the name of
 *              @p elmt is used. This exists so that a section can be given
 *              its logical name -- "FORCING" rather than whatever the
 *              document happened to call it.
 */
LIB_UTILITIES_EXPORT SessionNode
SessionNodeFromXml(const TiXmlElement *elmt, const std::string &name = "");

} // namespace Nektar::LibUtilities

#endif
