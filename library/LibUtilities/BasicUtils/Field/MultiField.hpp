///////////////////////////////////////////////////////////////////////////////
//
// File: MultiField.hpp
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
// Description: A set of fields stored as the columns of a matrix.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/BasicUtils/Field/Field.hpp>
#include <LibUtilities/BasicUtils/Field/SharedFieldStorage.hpp>

#include <memory>
#include <string>
#include <vector>

namespace Nektar::LibUtilities
{

/**
 * @brief A set of fields stored as the columns of a column-major matrix.
 *
 * Each column is an ordinary Field, so operators act on it as on any other.
 * Columns are stored in segments of a fixed number of columns, each segment
 * one contiguous allocation with a fixed distance between columns, so that
 * products over the columns of a segment are matrix products. Segments are
 * added as columns are, so storage grows with the number of columns used, and
 * existing columns never move. All columns are expected to share one
 * interleave width.
 *
 * @tparam TData  The floating-point representation of the fields.
 * @tparam TState A FieldState value representing the state of the fields.
 */
template <typename TData, FieldState TState> class MultiField
{
public:
    using value_type = TData;
    using FieldType  = Field<TData, TState>;

    /**
     * @brief Construct an empty set of fields.
     *
     * @param name               - Name of the set; columns are named after it.
     * @param blockAttr          - Block attributes of each column.
     * @param components         - Names of the components of each column.
     * @param numFieldPerSegment - Number of fields per storage segment.
     */
    MultiField(const std::string &name,
               const std::vector<BlockAttributes<TState>> &blockAttr,
               const std::vector<std::string> &components,
               const unsigned int numFieldPerSegment = 16)
        : m_name(name), m_blockAttr(blockAttr), m_components(components),
          m_numFieldPerSegment(numFieldPerSegment)
    {
        for (auto &attr : blockAttr)
        {
            m_fieldSize += attr.CompSize() * components.size();
        }

        // Columns lie back to back. The blocks are padded to a whole number of
        // SIMD vectors, so each column starts aligned, as a Field's storage
        // does.
        const size_t columnBytes = m_fieldSize * sizeof(TData);
        if (columnBytes % NektarSpaces::host_memory_alignment != 0)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MultiField - Block attributes are not padded to the host "
                     "memory alignment.");
        }
    }

    FieldType &operator[](const unsigned int j)
    {
        return *m_columns[j];
    }

    /**
     * @brief Number of fields, i.e. columns.
     */
    unsigned int GetNumField() const
    {
        return m_columns.size();
    }

    /**
     * @brief Add fields, i.e. columns, and the segments to store them, up to
     * @p n fields. Existing fields are kept; none are removed.
     */
    void ResizeNumField(const unsigned int n)
    {
        while (m_columns.size() < n)
        {
            const unsigned int j = m_columns.size();
            if (j % m_numFieldPerSegment == 0)
            {
                m_segments.push_back(
                    std::make_shared<SharedFieldStorage<TData>>(
                        m_fieldSize * m_numFieldPerSegment));
            }

            // Not make_unique: the shared-storage constructor of Field is
            // private to MultiField.
            m_columns.push_back(std::unique_ptr<FieldType>(new FieldType(
                m_name + " " + std::to_string(j), m_blockAttr, m_components, 1,
                m_segments.back(), (j % m_numFieldPerSegment) * m_fieldSize)));
        }
    }

    /**
     * @brief Number of values in each column, which is also the distance
     * between consecutive columns of a segment.
     */
    size_t GetFieldSize() const
    {
        return m_fieldSize;
    }

    /**
     * @brief Number of fields per segment.
     */
    unsigned int GetNumFieldPerSegment() const
    {
        return m_numFieldPerSegment;
    }

private:
    std::string m_name;
    std::vector<BlockAttributes<TState>> m_blockAttr;
    std::vector<std::string> m_components;
    unsigned int m_numFieldPerSegment;
    size_t m_fieldSize = 0;
    std::vector<std::shared_ptr<SharedFieldStorage<TData>>> m_segments;
    std::vector<std::unique_ptr<FieldType>> m_columns;
};

} // namespace Nektar::LibUtilities
