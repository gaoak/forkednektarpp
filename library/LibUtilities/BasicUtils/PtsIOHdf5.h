////////////////////////////////////////////////////////////////////////////////
//
//  File: PtsIOHdf5.h
//
//  For more information, please see: http://www.nektar.info/
//
//  The MIT License
//
//  Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
//  Department of Aeronautics, Imperial College London (UK), and Scientific
//  Computing and Imaging Institute, University of Utah (USA).
//
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//
//  Description: I/O routines for points in HDF5 format
//
////////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_LIB_UTILITIES_BASIC_UTILS_PTSIOHDF5_H
#define NEKTAR_LIB_UTILITIES_BASIC_UTILS_PTSIOHDF5_H

#include <LibUtilities/BasicUtils/H5.h>
#include <LibUtilities/BasicUtils/PtsField.h>
#include <LibUtilities/Communication/Comm.h>

#include <string>
#include <vector>

namespace Nektar::LibUtilities
{

class PtsIOHdf5;

typedef std::shared_ptr<PtsIOHdf5> PtsIOHdf5SharedPtr;

/**
 * @brief Writes a time series of point data to a HDF5 file.
 *
 * The file is laid out as
 *
 *     NEKTAR/
 *       COORDINATES/
 *         x          - dataset, [npoints]
 *         y          - dataset, [npoints]
 *         z          - dataset, [npoints]
 *       TIME-DATA/
 *         <field1>   - dataset, [ntimes x (npoints + 1)]
 *         ...
 *         <fieldN>   - dataset, [ntimes x (npoints + 1)]
 *
 * The coordinates are fixed for the lifetime of the file and are written
 * once. Each field is a single two-dimensional dataset, extensible along the
 * time dimension, whose first column holds the time and whose remaining
 * columns hold that field's value at each point. Storing the time in the
 * datasets themselves means a row is self-describing and the whole time
 * history of a field can be read with a single call.
 */
class PtsIOHdf5
{
public:
    /// Opens @p fileName and prepares it to receive the fields described by
    /// @p ptsField, discarding any data at or after @p time. @p expectedRows
    /// is an optional hint at how many times will be written, used only to
    /// choose a chunk size; zero means it is not known.
    LIB_UTILITIES_EXPORT PtsIOHdf5(LibUtilities::CommSharedPtr pComm,
                                   const std::string &fileName,
                                   const NekDouble time,
                                   const PtsFieldSharedPtr &ptsField,
                                   const size_t expectedRows = 0);

    LIB_UTILITIES_EXPORT ~PtsIOHdf5();

    /// Appends the field values held by @p ptsField at time @p time.
    LIB_UTILITIES_EXPORT void Write(const PtsFieldSharedPtr &ptsField,
                                    const NekDouble time);

    /// Flushes and closes the file. Called by the destructor.
    LIB_UTILITIES_EXPORT void Close();

private:
    /// Target size of a single HDF5 chunk, in bytes. Chunks much smaller than
    /// this make the chunk index dominate the file; chunks much larger risk
    /// exceeding the default chunk cache.
    static constexpr size_t TargetChunkBytes = 64 * 1024;

    /// Fewest time rows in a chunk. A chunk holding only a handful of rows
    /// costs more in index entries than it saves in allocated space.
    static constexpr size_t MinChunkRows = 64;

    size_t ChunkRows(const size_t rowLength, const size_t expectedRows) const;
    NekDouble ReadTime(const H5::DataSetSharedPtr &dataSet,
                       const hsize_t row) const;
    hsize_t FindRestartRow(const H5::DataSetSharedPtr &dataSet,
                           const hsize_t nRows, const NekDouble time) const;

    LibUtilities::CommSharedPtr m_comm;

    /// Name of file
    const std::string m_fileName;

    /// True on the process that owns the file
    bool m_isRoot = false;

    /// Number of points, fixed when the file is opened
    size_t m_nPts = 0;

    /// Index of the next row to be written
    hsize_t m_row = 0;

    H5::FileSharedPtr m_file;
    H5::GroupSharedPtr m_timeGroup;
    std::vector<H5::DataSetSharedPtr> m_fieldData;

    /// Scratch space for one row: time followed by one value per point
    std::vector<NekDouble> m_rowBuffer;
};

} // namespace Nektar::LibUtilities

#endif
