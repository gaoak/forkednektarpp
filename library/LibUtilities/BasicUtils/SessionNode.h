///////////////////////////////////////////////////////////////////////////////
//
// File: SessionNode.h
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
// Description: Format-neutral representation of a session configuration
// section.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_LIB_UTILITIES_BASICUTILS_SESSIONNODE_H
#define NEKTAR_LIB_UTILITIES_BASICUTILS_SESSIONNODE_H

#include <string>
#include <utility>
#include <vector>

#include <LibUtilities/BasicConst/NektarUnivTypeDefs.hpp>
#include <LibUtilities/LibUtilitiesDeclspec.h>

namespace Nektar::LibUtilities
{

namespace SessionNodeConvert
{
/**
 * @brief Convert the string @p str into @p value.
 *
 * Leading and trailing whitespace is ignored for all types except
 * `std::string`, and the whole of the remaining string must be consumed:
 * unlike `std::stoi` and friends, `"3abc"` is @b not accepted as the integer
 * 3. Booleans accept `true`/`false`, `yes`/`no`, `on`/`off` and `1`/`0`, in
 * any combination of upper and lower case.
 *
 * @return `true` if the conversion succeeded, in which case @p value holds
 *         the converted result; `false` otherwise, leaving @p value
 *         untouched.
 */
LIB_UTILITIES_EXPORT bool FromString(const std::string &str,
                                     std::string &value);
LIB_UTILITIES_EXPORT bool FromString(const std::string &str, bool &value);
LIB_UTILITIES_EXPORT bool FromString(const std::string &str, int &value);
LIB_UTILITIES_EXPORT bool FromString(const std::string &str, long &value);
LIB_UTILITIES_EXPORT bool FromString(const std::string &str, long long &value);
LIB_UTILITIES_EXPORT bool FromString(const std::string &str,
                                     unsigned int &value);
LIB_UTILITIES_EXPORT bool FromString(const std::string &str,
                                     unsigned long &value);
LIB_UTILITIES_EXPORT bool FromString(const std::string &str,
                                     unsigned long long &value);
LIB_UTILITIES_EXPORT bool FromString(const std::string &str, float &value);
LIB_UTILITIES_EXPORT bool FromString(const std::string &str, NekDouble &value);
} // namespace SessionNodeConvert

/**
 * @class SessionNode
 *
 * @brief A single node of a format-neutral session configuration tree.
 *
 * A SessionNode is a named node carrying an optional block of text, an
 * ordered list of named string attributes, and an ordered list of child
 * nodes. Child names need not be unique: repeated names are how repeated
 * entries (such as the individual `FORCE` entries of a `FORCING` section) are
 * represented, and document order is preserved.
 *
 * The point of the class is that it carries no knowledge of the format the
 * configuration was written in. Solver components that need to read their own
 * bespoke configuration section -- forcings, mappings, couplings and so on --
 * take a SessionNode rather than reaching into the session's XML document,
 * so that the same components work unchanged whether the configuration
 * arrived from an XML file, some other file format, or was assembled
 * programmatically by an embedding application.
 *
 * Trees are cheap to build and are expected to be small (configuration, not
 * data), so nodes own their children by value and are freely copyable.
 *
 * Reading a section typically looks like:
 *
 * @code
 * // <FORCE TYPE="Absorption">
 * //     <COEFF> SpongeCoefficient </COEFF>
 * //     <BOUNDARYREGIONS> 0 </BOUNDARYREGIONS>
 * // </FORCE>
 * std::string type  = force.GetAttribute("TYPE");
 * std::string coeff = force.GetValue("COEFF");
 * std::string regions = force.GetValue("BOUNDARYREGIONS", "");
 * @endcode
 *
 * Accessors come in three flavours: a `Defines...` predicate; a form taking a
 * default, which is used when the entry is absent; and a form taking no
 * default, which raises a fatal error naming the node's path if the entry is
 * absent. The `...As<T>` forms additionally convert the string, raising a
 * fatal error if the text is not a valid @p T.
 */
class SessionNode
{
public:
    /// Ordered list of (name, value) attribute pairs.
    typedef std::vector<std::pair<std::string, std::string>> AttributeMap;
    /// Ordered list of child nodes.
    typedef std::vector<SessionNode> ChildList;

    /// Construct an empty, unnamed node.
    SessionNode() = default;

    /// Construct a node called @p name, optionally carrying @p text.
    LIB_UTILITIES_EXPORT explicit SessionNode(std::string name,
                                              std::string text = "");

    /* ------ IDENTITY ------ */

    /// Returns the name of this node, e.g. "FORCE".
    LIB_UTILITIES_EXPORT const std::string &GetName() const;

    /**
     * @brief Returns the path of this node from the root of the tree it
     * belongs to, e.g. "FORCING/FORCE/COEFF".
     *
     * The path is maintained by AddChild() and is used to make error messages
     * intelligible; it carries no other meaning.
     */
    LIB_UTILITIES_EXPORT const std::string &GetPath() const;

    /// Returns the text carried by this node, or an empty string if it
    /// carries none.
    LIB_UTILITIES_EXPORT const std::string &GetText() const;

    /// Returns true if this node carries a non-empty block of text.
    LIB_UTILITIES_EXPORT bool DefinesText() const;

    /**
     * @brief Returns a short human-readable rendering of this node, naming
     * its path and its attributes, for use in diagnostics.
     *
     * For example: `COUPLING/I [PROPERTY = "RECEIVESTEPS", VALUE = "1"]`.
     * This is deliberately not XML: it describes the node as the rest of the
     * code sees it, whatever format it was read from.
     */
    LIB_UTILITIES_EXPORT std::string Describe() const;

    /* ------ ATTRIBUTES ------ */

    /// Returns true if this node has an attribute called @p name.
    LIB_UTILITIES_EXPORT bool DefinesAttribute(const std::string &name) const;

    /// Returns the value of attribute @p name; a fatal error is raised if the
    /// attribute is not defined.
    LIB_UTILITIES_EXPORT const std::string &GetAttribute(
        const std::string &name) const;

    /// Returns the value of attribute @p name, or @p def if it is not
    /// defined.
    LIB_UTILITIES_EXPORT std::string GetAttribute(const std::string &name,
                                                  const std::string &def) const;

    /// Returns all of this node's attributes, in the order they were
    /// supplied.
    LIB_UTILITIES_EXPORT const AttributeMap &GetAttributes() const;

    /// Returns attribute @p name converted to a @p T; a fatal error is raised
    /// if the attribute is not defined or does not convert.
    template <typename T> T GetAttributeAs(const std::string &name) const
    {
        T result{};
        const std::string &str = GetAttribute(name);
        if (!SessionNodeConvert::FromString(str, result))
        {
            ThrowConversionError("attribute", name, str);
        }
        return result;
    }

    /// Returns attribute @p name converted to a @p T, or @p def if the
    /// attribute is not defined; a fatal error is raised if it is defined but
    /// does not convert.
    template <typename T>
    T GetAttributeAs(const std::string &name, const T &def) const
    {
        if (!DefinesAttribute(name))
        {
            return def;
        }
        return GetAttributeAs<T>(name);
    }

    /* ------ CHILDREN ------ */

    /// Returns true if this node has at least one child called @p name.
    LIB_UTILITIES_EXPORT bool DefinesChild(const std::string &name) const;

    /// Returns the first child called @p name, or nullptr if there is none.
    LIB_UTILITIES_EXPORT const SessionNode *GetChildPtr(
        const std::string &name) const;

    /// Returns the first child called @p name; a fatal error is raised if
    /// there is none.
    LIB_UTILITIES_EXPORT const SessionNode &GetChild(
        const std::string &name) const;

    /// Returns every child of this node, in document order.
    LIB_UTILITIES_EXPORT const ChildList &GetChildren() const;

    /// Returns every child called @p name, in document order.
    LIB_UTILITIES_EXPORT std::vector<const SessionNode *> GetChildren(
        const std::string &name) const;

    /* ------ CHILD VALUES ------ */

    /**
     * @brief Returns the text of the first child called @p name.
     *
     * This is the dominant way of reading a configuration section, in which
     * each setting is a child node carrying a single value, as in
     * `<COEFF> SpongeCoefficient </COEFF>`. A fatal error is raised if there
     * is no such child.
     */
    LIB_UTILITIES_EXPORT const std::string &GetValue(
        const std::string &name) const;

    /// Returns the text of the first child called @p name, or @p def if there
    /// is no such child.
    LIB_UTILITIES_EXPORT std::string GetValue(const std::string &name,
                                              const std::string &def) const;

    /// Returns the text of the first child called @p name converted to a
    /// @p T; a fatal error is raised if there is no such child or it does not
    /// convert.
    template <typename T> T GetValueAs(const std::string &name) const
    {
        T result{};
        const std::string &str = GetValue(name);
        if (!SessionNodeConvert::FromString(str, result))
        {
            ThrowConversionError("value", name, str);
        }
        return result;
    }

    /// Returns the text of the first child called @p name converted to a
    /// @p T, or @p def if there is no such child; a fatal error is raised if
    /// the child exists but does not convert.
    template <typename T>
    T GetValueAs(const std::string &name, const T &def) const
    {
        if (!DefinesChild(name))
        {
            return def;
        }
        return GetValueAs<T>(name);
    }

    /* ------ TIME LEVEL (PARALLEL-IN-TIME) ------ */

    /**
     * @brief Resolves the `TIMELEVEL` indirection used by parallel-in-time
     * runs.
     *
     * A section may replace its contents with a series of `TIMELEVEL`
     * children, each carrying a `VALUE` attribute, so that different time
     * levels see different settings.
     *
     * If this node has no `TIMELEVEL` children then it is itself returned, so
     * that the common case of a section which does not vary with time level
     * needs no special handling by the caller. Otherwise the child matching
     * @p timeLevel is returned, or nullptr if none matches -- meaning that
     * this section simply does not apply at @p timeLevel, and the caller
     * should skip it.
     *
     * @param timeLevel   The time level to select.
     * @param enableCheck If true, a fatal error is raised, rather than
     *                    nullptr returned, when this node has `TIMELEVEL`
     *                    children but none matches @p timeLevel.
     */
    LIB_UTILITIES_EXPORT const SessionNode *GetTimeLevel(
        const size_t timeLevel, const bool enableCheck = true) const;

    /* ------ CONSTRUCTION ------ */

    /**
     * @brief Appends @p child to this node's list of children.
     *
     * Intended for the readers that build trees and for applications
     * assembling configuration programmatically. Note that this rewrites the
     * paths of @p child and its descendants, so any pointer or reference
     * previously obtained from @p child is invalidated.
     */
    LIB_UTILITIES_EXPORT void AddChild(SessionNode child);

    /// Sets attribute @p name to @p value, replacing any existing value.
    LIB_UTILITIES_EXPORT void SetAttribute(std::string name, std::string value);

    /// Sets the text carried by this node.
    LIB_UTILITIES_EXPORT void SetText(std::string text);

private:
    /// Name of this node.
    std::string m_name;
    /// Path of this node from the root of its tree; see GetPath().
    std::string m_path;
    /// Text carried by this node, if any.
    std::string m_text;
    /// Attributes of this node, in the order supplied.
    AttributeMap m_attributes;
    /// Children of this node, in the order supplied.
    ChildList m_children;

    /// Recomputes #m_path from @p parentPath, recursing into #m_children.
    void UpdatePath(const std::string &parentPath);

    /// Raises a fatal error describing a failed string conversion.
    LIB_UTILITIES_EXPORT void ThrowConversionError(
        const std::string &kind, const std::string &name,
        const std::string &str) const;
};

} // namespace Nektar::LibUtilities

#endif
