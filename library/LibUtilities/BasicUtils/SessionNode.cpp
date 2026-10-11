///////////////////////////////////////////////////////////////////////////////
//
// File: SessionNode.cpp
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

#include <LibUtilities/BasicUtils/SessionNode.h>

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>

#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <limits>

namespace Nektar::LibUtilities
{

namespace SessionNodeConvert
{

namespace
{

/// Returns @p str with leading and trailing whitespace removed.
std::string Trim(const std::string &str)
{
    auto isSpace = [](char c) {
        return std::isspace(static_cast<unsigned char>(c)) != 0;
    };

    size_t begin = 0, end = str.size();
    while (begin < end && isSpace(str[begin]))
    {
        ++begin;
    }
    while (end > begin && isSpace(str[end - 1]))
    {
        --end;
    }
    return str.substr(begin, end - begin);
}

/// Returns @p str converted to lower case.
std::string ToLower(const std::string &str)
{
    std::string result = str;
    for (char &c : result)
    {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return result;
}

/**
 * @brief Convert @p str to a signed integer, checking that the whole of the
 * string is consumed and that the result is representable as a @p T.
 */
template <typename T> bool ToSigned(const std::string &str, T &value)
{
    const std::string trimmed = Trim(str);
    if (trimmed.empty())
    {
        return false;
    }

    char *end     = nullptr;
    errno         = 0;
    long long tmp = std::strtoll(trimmed.c_str(), &end, 10);

    if (errno == ERANGE || end != trimmed.c_str() + trimmed.size())
    {
        return false;
    }
    if (tmp < static_cast<long long>(std::numeric_limits<T>::min()) ||
        tmp > static_cast<long long>(std::numeric_limits<T>::max()))
    {
        return false;
    }

    value = static_cast<T>(tmp);
    return true;
}

/**
 * @brief Convert @p str to an unsigned integer, checking that the whole of
 * the string is consumed and that the result is representable as a @p T.
 *
 * Note that, unlike `strtoull`, a leading minus sign is rejected rather than
 * silently wrapping around.
 */
template <typename T> bool ToUnsigned(const std::string &str, T &value)
{
    const std::string trimmed = Trim(str);
    if (trimmed.empty() || trimmed[0] == '-')
    {
        return false;
    }

    char *end              = nullptr;
    errno                  = 0;
    unsigned long long tmp = std::strtoull(trimmed.c_str(), &end, 10);

    if (errno == ERANGE || end != trimmed.c_str() + trimmed.size())
    {
        return false;
    }
    if (tmp > static_cast<unsigned long long>(std::numeric_limits<T>::max()))
    {
        return false;
    }

    value = static_cast<T>(tmp);
    return true;
}

/**
 * @brief Convert @p str to a floating-point value, checking that the whole of
 * the string is consumed.
 */
template <typename T> bool ToFloat(const std::string &str, T &value)
{
    const std::string trimmed = Trim(str);
    if (trimmed.empty())
    {
        return false;
    }

    char *end  = nullptr;
    errno      = 0;
    double tmp = std::strtod(trimmed.c_str(), &end);

    if (errno == ERANGE || end != trimmed.c_str() + trimmed.size())
    {
        return false;
    }

    value = static_cast<T>(tmp);
    return true;
}

} // namespace

bool FromString(const std::string &str, std::string &value)
{
    value = str;
    return true;
}

bool FromString(const std::string &str, bool &value)
{
    const std::string lower = ToLower(Trim(str));

    if (lower == "true" || lower == "yes" || lower == "on" || lower == "1")
    {
        value = true;
        return true;
    }
    if (lower == "false" || lower == "no" || lower == "off" || lower == "0")
    {
        value = false;
        return true;
    }
    return false;
}

bool FromString(const std::string &str, int &value)
{
    return ToSigned(str, value);
}

bool FromString(const std::string &str, long &value)
{
    return ToSigned(str, value);
}

bool FromString(const std::string &str, long long &value)
{
    return ToSigned(str, value);
}

bool FromString(const std::string &str, unsigned int &value)
{
    return ToUnsigned(str, value);
}

bool FromString(const std::string &str, unsigned long &value)
{
    return ToUnsigned(str, value);
}

bool FromString(const std::string &str, unsigned long long &value)
{
    return ToUnsigned(str, value);
}

bool FromString(const std::string &str, float &value)
{
    return ToFloat(str, value);
}

bool FromString(const std::string &str, NekDouble &value)
{
    return ToFloat(str, value);
}

} // namespace SessionNodeConvert

/**
 *
 */
SessionNode::SessionNode(std::string name, std::string text)
    : m_name(std::move(name)), m_text(std::move(text))
{
    m_path = m_name;
}

/**
 *
 */
const std::string &SessionNode::GetName() const
{
    return m_name;
}

/**
 *
 */
const std::string &SessionNode::GetPath() const
{
    return m_path;
}

/**
 *
 */
const std::string &SessionNode::GetText() const
{
    return m_text;
}

/**
 *
 */
bool SessionNode::DefinesText() const
{
    return !m_text.empty();
}

/**
 *
 */
std::string SessionNode::Describe() const
{
    std::string result = m_path;

    if (!m_attributes.empty())
    {
        result += " [";
        for (size_t i = 0; i < m_attributes.size(); ++i)
        {
            result += (i ? ", " : "") + m_attributes[i].first + " = \"" +
                      m_attributes[i].second + "\"";
        }
        result += "]";
    }

    return result;
}

/**
 *
 */
bool SessionNode::DefinesAttribute(const std::string &name) const
{
    for (const auto &attr : m_attributes)
    {
        if (attr.first == name)
        {
            return true;
        }
    }
    return false;
}

/**
 *
 */
const std::string &SessionNode::GetAttribute(const std::string &name) const
{
    for (const auto &attr : m_attributes)
    {
        if (attr.first == name)
        {
            return attr.second;
        }
    }

    NEKERROR(ErrorUtil::efatal, "Session configuration node '" + m_path +
                                    "' has no attribute '" + name + "'.");

    static const std::string empty;
    return empty;
}

/**
 *
 */
std::string SessionNode::GetAttribute(const std::string &name,
                                      const std::string &def) const
{
    for (const auto &attr : m_attributes)
    {
        if (attr.first == name)
        {
            return attr.second;
        }
    }
    return def;
}

/**
 *
 */
const SessionNode::AttributeMap &SessionNode::GetAttributes() const
{
    return m_attributes;
}

/**
 *
 */
bool SessionNode::DefinesChild(const std::string &name) const
{
    return GetChildPtr(name) != nullptr;
}

/**
 *
 */
const SessionNode *SessionNode::GetChildPtr(const std::string &name) const
{
    for (const auto &child : m_children)
    {
        if (child.m_name == name)
        {
            return &child;
        }
    }
    return nullptr;
}

/**
 *
 */
const SessionNode &SessionNode::GetChild(const std::string &name) const
{
    const SessionNode *child = GetChildPtr(name);

    if (!child)
    {
        NEKERROR(ErrorUtil::efatal, "Session configuration node '" + m_path +
                                        "' has no child '" + name + "'.");

        static const SessionNode empty;
        return empty;
    }

    return *child;
}

/**
 *
 */
const SessionNode::ChildList &SessionNode::GetChildren() const
{
    return m_children;
}

/**
 *
 */
std::vector<const SessionNode *> SessionNode::GetChildren(
    const std::string &name) const
{
    std::vector<const SessionNode *> result;

    for (const auto &child : m_children)
    {
        if (child.m_name == name)
        {
            result.push_back(&child);
        }
    }

    return result;
}

/**
 *
 */
const std::string &SessionNode::GetValue(const std::string &name) const
{
    return GetChild(name).m_text;
}

/**
 *
 */
std::string SessionNode::GetValue(const std::string &name,
                                  const std::string &def) const
{
    const SessionNode *child = GetChildPtr(name);
    return child ? child->m_text : def;
}

/**
 *
 */
const SessionNode *SessionNode::GetTimeLevel(const size_t timeLevel,
                                             const bool enableCheck) const
{
    std::vector<const SessionNode *> levels = GetChildren("TIMELEVEL");

    if (levels.empty())
    {
        return this;
    }

    for (const SessionNode *level : levels)
    {
        if (level->GetAttributeAs<size_t>("VALUE") == timeLevel)
        {
            return level;
        }
    }

    if (enableCheck)
    {
        NEKERROR(ErrorUtil::efatal,
                 "TIMELEVEL value " + std::to_string(timeLevel) +
                     " not found in session configuration node '" + m_path +
                     "'.");
    }

    return nullptr;
}

/**
 *
 */
void SessionNode::AddChild(SessionNode child)
{
    m_children.push_back(std::move(child));
    m_children.back().UpdatePath(m_path);
}

/**
 *
 */
void SessionNode::SetAttribute(std::string name, std::string value)
{
    for (auto &attr : m_attributes)
    {
        if (attr.first == name)
        {
            attr.second = std::move(value);
            return;
        }
    }

    m_attributes.emplace_back(std::move(name), std::move(value));
}

/**
 *
 */
void SessionNode::SetText(std::string text)
{
    m_text = std::move(text);
}

/**
 *
 */
void SessionNode::UpdatePath(const std::string &parentPath)
{
    m_path = parentPath.empty() ? m_name : parentPath + "/" + m_name;

    for (auto &child : m_children)
    {
        child.UpdatePath(m_path);
    }
}

/**
 *
 */
void SessionNode::ThrowConversionError(const std::string &kind,
                                       const std::string &name,
                                       const std::string &str) const
{
    NEKERROR(ErrorUtil::efatal, "Unable to interpret " + kind + " '" + name +
                                    "' of session configuration node '" +
                                    m_path + "': got '" + str + "'.");
}

} // namespace Nektar::LibUtilities
