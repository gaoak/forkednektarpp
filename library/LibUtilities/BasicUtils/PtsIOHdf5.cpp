////////////////////////////////////////////////////////////////////////////////
//
//  File: PtsIOHdf5.cpp
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

#include <LibUtilities/BasicUtils/PtsIOHdf5.h>

#include <LibUtilities/BasicUtils/Filesystem.hpp>
#include <LibUtilities/BasicUtils/RealComparison.hpp>

#include <algorithm>

namespace Nektar::LibUtilities
{

/**
 * @brief Opens the file and prepares it to receive data.
 *
 * If @p fileName does not exist it is created. If it does exist, the
 * coordinates it holds are checked against @p ptsField and any rows at or
 * after @p time are discarded, so that a simulation restarted from an earlier
 * time overwrites the history it is about to recompute rather than appending
 * to it.
 *
 * Only the root process touches the file; see Write().
 *
 * @param pComm     Communicator
 * @param fileName  Name of file
 * @param time      Physical time at which output will start
 * @param ptsField  Supplies the point coordinates and the field names. The
 *                  field values it holds are not used.
 * @param expectedRows  Hint at the number of times that will be written, or
 *                  zero if unknown. Only affects the chunk size.
 */
PtsIOHdf5::PtsIOHdf5(LibUtilities::CommSharedPtr pComm,
                     const std::string &fileName, const NekDouble time,
                     const PtsFieldSharedPtr &ptsField,
                     const size_t expectedRows)
    : m_comm(pComm), m_fileName(fileName)
{
    m_isRoot = m_comm->TreatAsRankZero();

    if (!m_isRoot)
    {
        return;
    }

    m_nPts                 = ptsField->GetNpoints();
    const size_t nFields   = ptsField->GetNFields();
    const size_t nDim      = ptsField->GetDim();
    const hsize_t rowLen   = m_nPts + 1;
    const hsize_t chunkLen = ChunkRows(rowLen, expectedRows);

    m_rowBuffer.resize(rowLen, 0.0);

    // Open an existing file, or create a new one
    if (fs::exists(fs::path(m_fileName)))
    {
        m_file = H5::File::Open(m_fileName, H5F_ACC_RDWR);
    }
    else
    {
        m_file = H5::File::Create(m_fileName, H5F_ACC_TRUNC);
    }

    // ContainsDataSet tests for a link of this name, so it answers for a
    // group just as well as for a dataset
    H5::GroupSharedPtr root = m_file->ContainsDataSet("NEKTAR")
                                  ? m_file->OpenGroup("NEKTAR")
                                  : m_file->CreateGroup("NEKTAR");

    // Coordinates are fixed for the lifetime of the file, so write them on
    // the first visit and check them for consistency on any later one
    const std::vector<std::string> dimName{"x", "y", "z"};
    H5::GroupSharedPtr coordGroup = root->ContainsDataSet("COORDINATES")
                                        ? root->OpenGroup("COORDINATES")
                                        : root->CreateGroup("COORDINATES");

    for (size_t i = 0; i < nDim; ++i)
    {
        if (coordGroup->ContainsDataSet(dimName[i]))
        {
            H5::DataSetSharedPtr coordData =
                coordGroup->OpenDataSet(dimName[i]);
            ASSERTL0(coordData->GetSpace()->GetSize() == m_nPts,
                     "The number of points in dataset '" + dimName[i] +
                         "' of file '" + m_fileName +
                         "' does not match the number of history points.");
            continue;
        }

        H5::DataSpaceSharedPtr space   = H5::DataSpace::OneD(m_nPts);
        H5::DataSetSharedPtr coordData = coordGroup->CreateDataSet(
            dimName[i], H5::DataType::OfObject(NekDouble{0.}), space);

        std::vector<NekDouble> coords(m_nPts, 0.0);
        for (size_t j = 0; j < m_nPts; ++j)
        {
            coords[j] = ptsField->GetPointVal(i, j);
        }
        space->SelectRange(0, m_nPts);
        coordData->Write(coords, space);
    }

    m_timeGroup = root->ContainsDataSet("TIME-DATA")
                      ? root->OpenGroup("TIME-DATA")
                      : root->CreateGroup("TIME-DATA");

    // Open or create one extensible dataset per field. The number of rows
    // already present is taken from the first field that exists; a field that
    // has only just been added to the session starts empty and is grown to
    // match, with HDF5 filling the rows it missed with zeros.
    bool haveRows     = false;
    hsize_t nRowsFile = 0;

    for (size_t i = 0; i < nFields; ++i)
    {
        const std::string fieldName = ptsField->GetFieldName(i);

        if (m_timeGroup->ContainsDataSet(fieldName))
        {
            H5::DataSetSharedPtr fieldData =
                m_timeGroup->OpenDataSet(fieldName);
            std::vector<hsize_t> dims = fieldData->GetSpace()->GetDims();

            ASSERTL0(dims.size() == 2 && dims[1] == rowLen,
                     "Dataset '" + fieldName + "' of file '" + m_fileName +
                         "' does not hold the expected number of history "
                         "points.");

            if (!haveRows)
            {
                nRowsFile = dims[0];
                haveRows  = true;
            }

            m_fieldData.push_back(fieldData);
            continue;
        }

        // Unlimited along time, fixed along the row
        std::vector<hsize_t> dims{0, rowLen};
        std::vector<hsize_t> maxDims{H5S_UNLIMITED, rowLen};
        std::vector<hsize_t> chunk{chunkLen, rowLen};

        H5::PListSharedPtr createPL = H5::PList::DatasetCreate();
        createPL->SetChunk(chunk);

        m_fieldData.push_back(m_timeGroup->CreateDataSet(
            fieldName, H5::DataType::OfObject(NekDouble{0.}),
            std::make_shared<H5::DataSpace>(dims, maxDims), createPL));
    }

    // Drop any rows at or after the restart time
    m_row = haveRows ? FindRestartRow(m_fieldData[0], nRowsFile, time) : 0;

    for (auto &fieldData : m_fieldData)
    {
        fieldData->SetExtent(std::vector<hsize_t>{m_row, rowLen});
    }

    m_file->Flush();
}

/**
 *
 */
PtsIOHdf5::~PtsIOHdf5()
{
    Close();
}

/**
 * @brief Number of time rows in a single chunk.
 *
 * HDF5 allocates a chunk as soon as any part of it is written, so a chunk
 * sized for a long run would be mostly empty after a short one. The chunk is
 * therefore capped at PtsIOHdf5::TargetChunkBytes, which keeps the chunk
 * index small, and then reduced to @p expectedRows where that is known to be
 * smaller, subject to a floor of PtsIOHdf5::MinChunkRows.
 */
size_t PtsIOHdf5::ChunkRows(const size_t rowLength,
                            const size_t expectedRows) const
{
    const size_t maxRows =
        std::max<size_t>(1, TargetChunkBytes / (rowLength * sizeof(NekDouble)));

    if (expectedRows == 0 || expectedRows >= maxRows)
    {
        return maxRows;
    }

    return std::min(maxRows, std::max(expectedRows, MinChunkRows));
}

/**
 * @brief Reads the time stored in the first column of row @p row.
 */
NekDouble PtsIOHdf5::ReadTime(const H5::DataSetSharedPtr &dataSet,
                              const hsize_t row) const
{
    H5::DataSpaceSharedPtr space = dataSet->GetSpace();
    space->SelectRange(std::vector<hsize_t>{row, 0},
                       std::vector<hsize_t>{1, 1});

    std::vector<NekDouble> value;
    dataSet->Read(value, space);

    return value[0];
}

/**
 * @brief Index of the first row whose time is at or after @p time.
 *
 * Times increase down the dataset, so this is a binary search; only the rows
 * it visits are read, rather than the whole time column.
 */
hsize_t PtsIOHdf5::FindRestartRow(const H5::DataSetSharedPtr &dataSet,
                                  const hsize_t nRows,
                                  const NekDouble time) const
{
    hsize_t lo = 0, hi = nRows;

    while (lo < hi)
    {
        const hsize_t mid     = lo + (hi - lo) / 2;
        const NekDouble tTest = ReadTime(dataSet, mid);

        // A relative comparison, so that the row written at exactly the
        // restart time is recognised whatever the magnitude of the time
        if (tTest > time || IsRealEqual(tTest, time))
        {
            hi = mid;
        }
        else
        {
            lo = mid + 1;
        }
    }

    return lo;
}

/**
 * @brief Appends one row of field values.
 *
 * Each field's dataset is extended by a single row, which is then filled with
 * @p time followed by that field's value at each point.
 *
 * Note that the file is written by the root process, so distributed data must
 * be communicated to it before calling this function. This limitation might
 * be relaxed in the future: because each field is one dataset, each process
 * could write the points it owns directly as a collective operation.
 *
 * @param ptsField  PtsField holding the values for this time
 * @param time      Physical time
 */
void PtsIOHdf5::Write(const PtsFieldSharedPtr &ptsField, const NekDouble time)
{
    if (!m_isRoot)
    {
        return;
    }

    ASSERTL1(m_file, "PtsIOHdf5 has been closed.");
    ASSERTL0(ptsField->GetNpoints() == m_nPts &&
                 ptsField->GetNFields() == m_fieldData.size(),
             "PtsField does not match the file opened by PtsIOHdf5.");

    const hsize_t rowLen = m_nPts + 1;
    const size_t nDim    = ptsField->GetDim();

    const std::vector<hsize_t> dims{m_row + 1, rowLen};
    const std::vector<hsize_t> start{m_row, 0};
    const std::vector<hsize_t> count{1, rowLen};

    m_rowBuffer[0] = time;

    for (size_t i = 0; i < m_fieldData.size(); ++i)
    {
        const Array<OneD, NekDouble> field = ptsField->GetPts(i + nDim);
        for (size_t j = 0; j < m_nPts; ++j)
        {
            m_rowBuffer[j + 1] = field[j];
        }

        m_fieldData[i]->SetExtent(dims);

        // The dataspace must be re-read after the extent changes
        H5::DataSpaceSharedPtr space = m_fieldData[i]->GetSpace();
        space->SelectRange(start, count);
        m_fieldData[i]->Write(m_rowBuffer, space);
    }

    ++m_row;

    // Keep the file usable, and the data safe, if the run is interrupted
    m_file->Flush();
}

/**
 *
 */
void PtsIOHdf5::Close()
{
    m_fieldData.clear();
    m_timeGroup.reset();
    m_file.reset();
}

} // namespace Nektar::LibUtilities
