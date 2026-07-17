////////////////////////////////////////////////////////////////////////////////
//
//  File: MeshGraphIOHDF5.cpp
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
//  Description: HDF5-based mesh format for Nektar++.
//
////////////////////////////////////////////////////////////////////////////////

#include <boost/algorithm/string.hpp>
#include <tinyxml.h>
#include <type_traits>

#include <LibUtilities/BasicUtils/CppCommandLine.hpp>
#include <LibUtilities/BasicUtils/Filesystem.hpp>
#include <LibUtilities/BasicUtils/ParseUtils.h>
#include <LibUtilities/BasicUtils/Timer.h>
#include <SpatialDomains/MeshGraphIOHDF5.h>
#include <SpatialDomains/MeshPartition.h>
#include <SpatialDomains/Movement/Movement.h>

#define TIME_RESULT(verb, msg, timer)                                          \
    if (verb)                                                                  \
    {                                                                          \
        std::cout << "  - " << msg << ": " << timer.TimePerTest(1) << "\n"     \
                  << std::endl;                                                \
    }

using namespace Nektar::LibUtilities;

namespace Nektar::SpatialDomains
{

/**
 * @class MeshGraphIOHDF5
 *
 * @brief Reader and writer for the Nektar++ HDF5 mesh format.
 *
 * @par On-disk layout
 *
 * The geometry is stored in a single HDF5 file (conventionally `*.nekg`)
 * under the following group hierarchy:
 *
 * @code
 *   NEKTAR/                         (group)
 *   └── GEOMETRY/                   (group; attribute FORMAT_VERSION : uint)
 *       ├── MESH/                   (group: the geometric/topological payload)
 *       │   ├── VERT                vertex coordinates
 *       │   ├── SEG                 segment  -> vertex IDs
 *       │   ├── TRI, QUAD           2D faces -> edge (segment) IDs
 *       │   ├── TET, PYR, PRISM, HEX  3D elements -> face IDs
 *       │   ├── CURVE_EDGE          curved-edge descriptors
 *       │   ├── CURVE_FACE          curved-face descriptors
 *       │   ├── CURVE_NODES         curve point coordinates
 *       │   ├── COMPOSITE           composite definition strings
 *       │   └── DOMAIN              domain definition strings
 *       └── MAPS/                   (group: row -> global ID lookups)
 *           ├── VERT, SEG, TRI, ...   one int ID per row of the MESH dataset
 *           ├── CURVE_EDGE, CURVE_FACE
 *           ├── COMPOSITE
 *           └── DOMAIN
 * @endcode
 *
 * The central design idea is that every dataset in `MESH/` is keyed purely by
 * its *row index*, and the matching dataset in `MAPS/` (with the same name)
 * supplies the global Nektar++ ID for each row. Datasets therefore never need
 * to be stored in ID order, which is what allows each MPI rank to write a
 * disjoint, contiguous hyperslab of rows independently (see
 * GetGeomWriteLayout() and WriteGeometryMap()). References *between* datasets
 * (e.g. a `SEG` row naming its two vertices) are always by global ID, resolved
 * on read through the `MAPS/` tables.
 *
 * @par Geometry datasets (`MESH/VERT`, `SEG`, `TRI`, ...)
 *
 * Each is a 2D array of shape `[nEntitiesGlobal, nGeomData]`. The column count
 * @c nGeomData is fixed per shape type and given by GetGeomDataDim():
 *   - `VERT`  : 3 doubles  — the (x, y, z) coordinates of the point;
 *   - `SEG`   : 2 ints     — the global IDs of the two bounding vertices;
 *   - `TRI`   : 3 ints, `QUAD` : 4 ints — global IDs of the bounding edges;
 *   - `TET`   : 4 ints, `PYR`/`PRISM` : 5 ints, `HEX` : 6 ints — global IDs of
 *     the bounding faces.
 *
 * The companion `MAPS/<shape>` dataset is a 1D `[nEntitiesGlobal]` int array
 * giving the global ID of the entity in each row.
 *
 * @par Curve datasets (`MESH/CURVE_EDGE`, `CURVE_FACE`, `CURVE_NODES`)
 *
 * `CURVE_EDGE` and `CURVE_FACE` are 2D `[nCurvesGlobal, 3]` int arrays. Each
 * row describes one curved entity:
 *   - col 0: number of points defining the curve;
 *   - col 1: the LibUtilities::PointsType distribution of those points;
 *   - col 2: the row offset into `CURVE_NODES` at which this curve's points
 *     begin.
 *
 * The matching `MAPS/CURVE_EDGE` / `MAPS/CURVE_FACE` 1D int arrays give the
 * global edge/face ID owning each curve. `CURVE_NODES` is a single shared 2D
 * `[nPointsGlobal, 3]` double array holding the (x, y, z) coordinates of every
 * curve point in the mesh, concatenated; individual curves slice into it using
 * the offset in column 2 and the count in column 0.
 *
 * @par Composite and domain datasets (`MESH/COMPOSITE`, `DOMAIN`)
 *
 * Both are 1D variable-length string datasets, with companion `MAPS/`
 * int-ID arrays. A `COMPOSITE` entry is a string such as `" T[0-5,7] "`: a
 * single-character shape tag followed by a bracketed, run-length-compressed
 * sequence of geometry IDs. The recognised tags are
 *   `V` (vertex), `S`/`E` (segment/edge), `Q` (quad), `T` (triangle),
 *   `F` (mixed face), `A` (tet), `P` (pyramid), `R` (prism), `H` (hex).
 * A `DOMAIN` entry (format version 2) is a compressed sequence string of the
 * composite IDs forming that domain; in format version 1 a single entry held
 * the entire domain. Because variable-length strings cannot be written
 * collectively, these two datasets are gathered to and written by rank 0 only
 * (see WriteComposites() and WriteDomain()).
 */

/**
 * @brief Version of the Nektar++ HDF5 geometry format.
 *
 * This is embedded into the main `NEKTAR/GEOMETRY` group as the
 * `FORMAT_VERSION` attribute when a mesh is written, and checked on read so
 * that newer files are not silently misinterpreted by an older binary
 * (v_PartitionMesh() asserts that the file version does not exceed this
 * value). The reader retains backwards compatibility with version 1, which
 * differs only in the layout of the `DOMAIN` dataset (see ReadDomain()).
 */
const unsigned int MeshGraphIOHDF5::FORMAT_VERSION = 2;

std::string MeshGraphIOHDF5::className =
    GetMeshGraphIOFactory().RegisterCreatorFunction(
        "HDF5", MeshGraphIOHDF5::create, "IO with HDF5 geometry");

/**
 * @brief Read the complete geometry from the HDF5 file into the #MeshGraph.
 *
 * This is the serial/post-partition entry point used once the mesh has already
 * been partitioned: it reads the `COMPOSITE` and `DOMAIN` datasets via
 * ReadComposites() and ReadDomain(), pulls expansion information from the
 * accompanying XML, then closes the HDF5 file and (optionally) materialises the
 * full graph connectivity.
 *
 * @param fillGraph  If true, call MeshGraph::FillGraph() once reading is
 *                   complete to build the derived connectivity maps.
 */
void MeshGraphIOHDF5::v_ReadGeometry(bool fillGraph)
{
    ReadComposites();
    ReadDomain();

    m_meshGraph->ReadExpansionInfo(m_session->GetElement("NEKTAR/EXPANSIONS"));

    // Close up shop.
    m_mesh->Close();
    m_maps->Close();
    m_file->Close();

    if (fillGraph)
    {
        m_meshGraph->FillGraph();
    }
}

/**
 * @brief Utility function to split a vector equally amongst a number of
 * processors.
 *
 * @param vecsize  Size of the total amount of work
 * @param rank     Rank of this process
 * @param nprocs   Number of processors in the group
 *
 * @return A pair with the offset this process should occupy, along with the
 *         count for the amount of work.
 */
std::pair<size_t, size_t> SplitWork(size_t vecsize, int rank, int nprocs)
{
    size_t div = vecsize / nprocs;
    size_t rem = vecsize % nprocs;
    if (rank < rem)
    {
        return std::make_pair(rank * (div + 1), div + 1);
    }
    else
    {
        return std::make_pair((rank - rem) * div + rem * (div + 1), div);
    }
}

/**
 * @brief Return the number of columns (`nGeomData`) used to store a geometry of
 * type @tparam T in its `MESH/<shape>` dataset.
 *
 * This is the per-row width of the geometry datasets described in the file
 * overview, and is selected at compile time on the entity's topological
 * dimension `T::kDim`:
 *   - 0D (points): 3, the (x, y, z) coordinates;
 *   - 1D (segments): `T::kNverts`, the bounding vertex count;
 *   - 2D (faces): `T::kNedges`, the bounding edge count;
 *   - 3D (elements): `T::kNfaces`, the bounding face count.
 *
 * @tparam T  Geometry type (e.g. PointGeom, SegGeom, TriGeom, HexGeom).
 * @return The fixed number of data columns for that geometry type.
 */
template <class T, typename std::enable_if<T::kDim == 0, int>::type = 0>
inline int GetGeomDataDim([[maybe_unused]] GeomMapView<T> &geomMap)
{
    return 3;
}

/// @copydoc GetGeomDataDim
template <class T, typename std::enable_if<T::kDim == 1, int>::type = 0>
inline int GetGeomDataDim([[maybe_unused]] GeomMapView<T> &geomMap)
{
    return T::kNverts;
}

/// @copydoc GetGeomDataDim
template <class T, typename std::enable_if<T::kDim == 2, int>::type = 0>
inline int GetGeomDataDim([[maybe_unused]] GeomMapView<T> &geomMap)
{
    return T::kNedges;
}

/// @copydoc GetGeomDataDim
template <class T, typename std::enable_if<T::kDim == 3, int>::type = 0>
inline int GetGeomDataDim([[maybe_unused]] GeomMapView<T> &geomMap)
{
    return T::kNfaces;
}

/**
 * @brief Recursion terminator for UniqueValues().
 *
 * @see UniqueValues(std::unordered_set<int>&, const std::vector<int>&, T&...)
 */
template <class... T>
inline void UniqueValues([[maybe_unused]] std::unordered_set<int> &unique)
{
}

/**
 * @brief Accumulate the union of one or more integer vectors into a set.
 *
 * Used when recursing down the geometry hierarchy to collect, without
 * duplicates, the set of sub-entity IDs referenced by a batch of just-read
 * entities (e.g. the unique face IDs named by all local 3D elements), which
 * then becomes the read list for the next dimension down.
 *
 * @param unique  Set accumulating the distinct values (modified in place).
 * @param input   The next vector whose entries are inserted into @p unique.
 * @param args    Any further vectors, processed by recursion.
 */
template <class... T>
inline void UniqueValues(std::unordered_set<int> &unique,
                         const std::vector<int> &input, T &...args)
{
    for (auto i : input)
    {
        unique.insert(i);
    }

    UniqueValues(unique, args...);
}

std::string MeshGraphIOHDF5::cmdSwitch =
    LibUtilities::SessionReader::RegisterCmdLineFlag(
        "use-hdf5-node-comm", "",
        "Use a per-node communicator for HDF5 partitioning.");

/**
 * @brief Read and partition the mesh in parallel directly from HDF5.
 *
 * This is the parallel read path. It opens the HDF5 file (collectively, using
 * MPI-IO when built with parallel HDF5), reads the `FORMAT_VERSION` attribute
 * and the top-dimensional element connectivity needed to build the dual graph,
 * and hands that graph to a parallel partitioner (PtScotch, or ParMetis if
 * available). Each rank then determines the set of element rows it owns and
 * recurses *down* the dimensional hierarchy — elements to faces to edges to
 * vertices — using ReadGeometryData() to read only the rows it needs, and
 * ReadCurveMap() for the associated curvature, before constructing the local
 * Geometry objects with FillGeomMap().
 *
 * An optional two-level partitioning scheme is available via the
 * `--use-hdf5-node-comm` command-line flag: the global communicator is split
 * into a per-node (inner) and inter-node (outer) communicator so that the
 * expensive graph partitioning is performed once per node and the result
 * scattered to the node-local ranks, reducing the partitioner's communication
 * cost on many-core nodes.
 *
 * Unlike the XML reader, every rank parses the session file here, since the
 * cost is negligible compared with avoiding repeated metadata broadcasts.
 *
 * @param session  The session reader providing the communicator, command-line
 *                 options and the `NEKTAR/GEOMETRY` XML stub that names the
 *                 HDF5 file and its mesh/space dimensions.
 */
void MeshGraphIOHDF5::v_PartitionMesh(
    LibUtilities::SessionReaderSharedPtr session)
{
    LibUtilities::Timer all;
    all.Start();
    int err;
    LibUtilities::CommSharedPtr comm     = session->GetComm();
    LibUtilities::CommSharedPtr commMesh = comm->GetRowComm();
    const bool isRoot                    = comm->TreatAsRankZero();

    // By default, only the root process will have read the session file, which
    // is done to avoid every process needing to read the XML file. For HDF5, we
    // don't care about this, so just have every process parse the session file.
    if (!isRoot)
    {
        session->InitSession();
    }

    // We use the XML geometry to find information about the HDF5 file.
    m_session            = session;
    m_xmlGeom            = session->GetElement("NEKTAR/GEOMETRY");
    TiXmlAttribute *attr = m_xmlGeom->FirstAttribute();
    int meshDimension    = 3;
    int spaceDimension   = 3;

    while (attr)
    {
        std::string attrName(attr->Name());
        if (attrName == "DIM")
        {
            err = attr->QueryIntValue(&meshDimension);
            ASSERTL0(err == TIXML_SUCCESS, "Unable to read mesh dimension.");
        }
        else if (attrName == "SPACE")
        {
            err = attr->QueryIntValue(&spaceDimension);
            ASSERTL0(err == TIXML_SUCCESS, "Unable to read space dimension.");
        }
        else if (attrName == "PARTITION")
        {
            ASSERTL0(false,
                     "PARTITION parameter should only be used in XML meshes");
        }
        else if (attrName == "HDF5FILE")
        {
            m_hdf5Name = attr->Value();
        }
        else if (attrName == "PARTITIONED")
        {
            ASSERTL0(false,
                     "PARTITIONED parameter should only be used in XML meshes");
        }
        else
        {
            std::string errstr("Unknown attribute: ");
            errstr += attrName;
            ASSERTL1(false, errstr.c_str());
        }
        // Get the next attribute.
        attr = attr->Next();
    }

    ASSERTL0(m_hdf5Name.size() > 0, "Unable to obtain mesh file name.");
    ASSERTL0(meshDimension <= spaceDimension,
             "Mesh dimension greater than space dimension.");

    m_meshGraph->SetMeshDimension(meshDimension);
    m_meshGraph->SetSpaceDimension(spaceDimension);

    // Open handle to the HDF5 mesh
    LibUtilities::H5::PListSharedPtr parallelProps = H5::PList::Default();
    m_readPL                                       = H5::PList::Default();

#if 0
#if defined(NEKTAR_USE_MPI) && defined(NEKTAR_HDF5_PARALLEL)
    if (commMesh->GetSize() > 1)
    {
        // Use MPI/O to access the file
        parallelProps = H5::PList::FileAccess();
        parallelProps->SetMpio(commMesh);
        // Use collective IO
        m_readPL = H5::PList::DatasetXfer();
        m_readPL->SetDxMpioCollective();
    }
#endif
#endif

    m_file = H5::File::Open(m_hdf5Name, H5F_ACC_RDONLY, parallelProps);

    auto root = m_file->OpenGroup("NEKTAR");
    ASSERTL0(root, "Cannot find NEKTAR group in HDF5 file.");

    auto root2 = root->OpenGroup("GEOMETRY");
    ASSERTL0(root2, "Cannot find NEKTAR/GEOMETRY group in HDF5 file.");

    // Check format version
    H5::Group::AttrIterator attrIt  = root2->attr_begin();
    H5::Group::AttrIterator attrEnd = root2->attr_end();
    for (; attrIt != attrEnd; ++attrIt)
    {
        if (*attrIt == "FORMAT_VERSION")
        {
            break;
        }
    }
    ASSERTL0(attrIt != attrEnd,
             "Unable to determine Nektar++ geometry HDF5 file version.");
    root2->GetAttribute("FORMAT_VERSION", m_inFormatVersion);

    ASSERTL0(m_inFormatVersion <= FORMAT_VERSION,
             "File format in " + m_hdf5Name +
                 " is higher than supported in "
                 "this version of Nektar++");

    m_mesh = root2->OpenGroup("MESH");
    ASSERTL0(m_mesh, "Cannot find NEKTAR/GEOMETRY/MESH group in HDF5 file.");
    m_maps = root2->OpenGroup("MAPS");
    ASSERTL0(m_mesh, "Cannot find NEKTAR/GEOMETRY/MAPS group in HDF5 file.");

    if (m_meshPartitioned)
    {
        return;
    }

    if (m_meshGraph->GetDomainRange() &&
        m_meshGraph->GetDomainRange()->m_compElmts)
    {
        m_meshGraph->GetDomainRange()->m_compElmts = meshDimension;
    }

    SetupCompositeRange(m_meshGraph->GetDomainRange());

    m_meshPartitioned = true;
    m_meshGraph->SetMeshPartitioned(true);

    // Depending on dimension, read element IDs.
    std::map<int,
             std::vector<std::tuple<std::string, int, LibUtilities::ShapeType>>>
        dataSets;

    dataSets[1] = {{"SEG", 2, LibUtilities::eSegment}};
    dataSets[2] = {{"TRI", 3, LibUtilities::eTriangle},
                   {"QUAD", 4, LibUtilities::eQuadrilateral}};
    dataSets[3] = {{"TET", 4, LibUtilities::eTetrahedron},
                   {"PYR", 5, LibUtilities::ePyramid},
                   {"PRISM", 5, LibUtilities::ePrism},
                   {"HEX", 6, LibUtilities::eHexahedron}};

    const bool verbRoot = isRoot && session->DefinesCmdLineArgument("verbose");

    if (verbRoot)
    {
        std::cout << "Reading HDF5 geometry..." << std::endl;
    }

    // If we want to use an multi-level communicator, then split the
    // communicator at this point. We set by default the inter-node communicator
    // to be the normal communicator: this way if the multi-level partitioning
    // is disabled we proceed as 'normal'.
    LibUtilities::CommSharedPtr innerComm, interComm = comm;
    int innerRank = 0, innerSize = 1,
        interRank = interComm->GetRowComm()->GetRank(),
        interSize = interComm->GetRowComm()->GetSize();

    if (session->DefinesCmdLineArgument("use-hdf5-node-comm"))
    {
        ASSERTL0(comm->GetSize() == commMesh->GetSize(),
                 "--use-hdf5-node-comm not available with Parallel-in-Time")

        auto splitComm = comm->SplitCommNode();
        innerComm      = splitComm.first;
        interComm      = splitComm.second;
        innerRank      = innerComm->GetRank();
        innerSize      = innerComm->GetSize();

        if (innerRank == 0)
        {
            interRank = interComm->GetRank();
            interSize = interComm->GetSize();
        }
    }

    // Unordered set of rows of the dataset this process needs to read for the
    // elements of dimension meshDimension.
    std::unordered_set<int> toRead;

    // Calculate reasonably even distribution of processors for calling
    // ptScotch. We'll do this work on only one process per node.
    LibUtilities::Timer t;
    t.Start();

    if (verbRoot)
    {
        std::cout << "  - beginning partitioning" << std::endl;
    }

    // Perform initial read if either (a) we are on all ranks and multi-level
    // partitioning is not enabled; or (b) rank 0 of all nodes if it is.
    if (innerRank == 0)
    {
        LibUtilities::Timer t2;
        t2.Start();

        const bool verbRoot2 =
            isRoot && session->DefinesCmdLineArgument("verbose");

        // Read IDs for partitioning purposes
        std::vector<int> ids;

        // Map from element ID to 'row' which is a contiguous ordering required
        // for parallel partitioning.
        std::vector<MeshEntity> elmts;
        std::unordered_map<int, int> row2id, id2row;

        LibUtilities::H5::FileSharedPtr file    = m_file;
        LibUtilities::H5::PListSharedPtr readPL = m_readPL;
        LibUtilities::H5::GroupSharedPtr mesh = m_mesh, maps = m_maps;

        if (innerComm)
        {
            // For per-node partitioning, create a temporary reader (otherwise
            // communicators are inconsistent).
            auto parallelProps = H5::PList::FileAccess();
            parallelProps->SetMpio(interComm);

            // Use collective IO
#if 0
#if defined(NEKTAR_USE_MPI) && defined(NEKTAR_HDF5_PARALLEL)
            readPL = H5::PList::DatasetXfer();
            readPL->SetDxMpioCollective();
#endif
#endif
            file = H5::File::Open(m_hdf5Name, H5F_ACC_RDONLY, parallelProps);

            auto root  = file->OpenGroup("NEKTAR");
            auto root2 = root->OpenGroup("GEOMETRY");
            mesh       = root2->OpenGroup("MESH");
            maps       = root2->OpenGroup("MAPS");
        }

        int rowCount = 0;
        for (auto &it : dataSets[meshDimension])
        {
            std::string ds = std::get<0>(it);

            if (!mesh->ContainsDataSet(ds))
            {
                continue;
            }

            // Open metadata dataset
            H5::DataSetSharedPtr data    = mesh->OpenDataSet(ds);
            H5::DataSpaceSharedPtr space = data->GetSpace();
            std::vector<hsize_t> dims    = space->GetDims();

            H5::DataSetSharedPtr mdata    = maps->OpenDataSet(ds);
            H5::DataSpaceSharedPtr mspace = mdata->GetSpace();
            std::vector<hsize_t> mdims    = mspace->GetDims();

            // TODO: This could perhaps be done more intelligently; reads all
            // IDs for the top-level elements so that we can construct the dual
            // graph of the mesh.
            std::vector<int> tmpElmts, tmpIds;
            mdata->Read(tmpIds, mspace, readPL);
            data->Read(tmpElmts, space, readPL);

            const int nGeomData = std::get<1>(it);
            auto tmpIt          = tmpElmts.begin();

            if (m_meshGraph->GetDomainRange() ==
                LibUtilities::NullDomainRangeShPtr)
            {
                // avoid range checking on larger meshes if not required
                for (int i = 0; i < tmpIds.size();
                     ++i, ++rowCount, tmpIt += nGeomData)
                {
                    MeshEntity e;
                    row2id[rowCount]  = tmpIds[i];
                    id2row[tmpIds[i]] = row2id[rowCount];
                    e.id              = rowCount;
                    e.origId          = tmpIds[i];
                    e.ghost           = false;
                    e.list =
                        std::vector<unsigned int>(tmpIt, tmpIt + nGeomData);
                    elmts.push_back(e);
                }
            }
            else
            {
                // avoid range checking on larger meshes if not required
                for (int i = 0; i < tmpIds.size();
                     ++i, ++rowCount, tmpIt += nGeomData)
                {
                    MeshEntity e;
                    row2id[rowCount]  = tmpIds[i];
                    id2row[tmpIds[i]] = row2id[rowCount];
                    e.id              = rowCount;
                    e.origId          = tmpIds[i];
                    e.ghost           = false;
                    e.list =
                        std::vector<unsigned int>(tmpIt, tmpIt + nGeomData);
                    if (m_meshGraph->CheckRange(e))
                    {
                        elmts.push_back(e);
                    }
                }
            }
        }

        interComm->GetRowComm()->Block();

        t2.Stop();
        TIME_RESULT(verbRoot2, "  - initial read", t2);
        t2.Start();

        // Do not partition in serial since postprocessing may not
        // lead to very suitable element distribution that then leads
        // to challenges with Scotch
        if (commMesh->GetSize() > 1)
        {
            // Check to see we have at least as many processors as elements.
            size_t numElmt = elmts.size();
            ASSERTL0(commMesh->GetSize() <= numElmt,
                     "This mesh has more processors than elements!");

            auto elRange = SplitWork(numElmt, interRank, interSize);

            // Construct map of element entities for partitioner.
            std::map<int, MeshEntity> partElmts;
            std::unordered_set<int> facetIDs;

            int vcnt = 0;

            for (int el = elRange.first; el < elRange.first + elRange.second;
                 ++el, ++vcnt)
            {
                MeshEntity elmt    = elmts[el];
                elmt.ghost         = false;
                partElmts[elmt.id] = elmt;

                for (auto &facet : elmt.list)
                {
                    facetIDs.insert(facet);
                }
            }

            // Now identify ghost vertices for the graph. This could also
            // probably be improved.
            int nLocal = vcnt;
            for (int i = 0; i < numElmt; ++i)
            {
                // Ignore anything we already read.
                if (i >= elRange.first && i < elRange.first + elRange.second)
                {
                    continue;
                }

                MeshEntity elmt = elmts[i];
                bool insert     = false;

                // Check for connections to local elements.
                for (auto &eId : elmt.list)
                {
                    if (facetIDs.find(eId) != facetIDs.end())
                    {
                        insert = true;
                        break;
                    }
                }

                if (insert)
                {
                    elmt.ghost         = true;
                    partElmts[elmt.id] = elmt;
                }
            }

            // Create partitioner. Default partitioner to use is PtScotch. Use
            // ParMetis as default if it is installed. Override default with
            // command-line flags if they are set.
            std::string partitionerName =
                commMesh->GetSize() > 1 ? "PtScotch" : "Scotch";
            if (GetMeshPartitionFactory().ModuleExists("ParMetis"))
            {
                partitionerName = "ParMetis";
            }
            if (session->DefinesCmdLineArgument("use-parmetis"))
            {
                partitionerName = "ParMetis";
            }
            if (session->DefinesCmdLineArgument("use-ptscotch"))
            {
                partitionerName = "PtScotch";
            }

            MeshPartitionSharedPtr partitioner =
                GetMeshPartitionFactory().CreateInstance(
                    partitionerName, session, interComm, meshDimension,
                    partElmts, CreateCompositeDescriptor(id2row));

            t2.Stop();
            TIME_RESULT(verbRoot2, "  - partitioner setup", t2);
            t2.Start();

            partitioner->PartitionMesh(interSize, true, false, nLocal);
            t2.Stop();
            TIME_RESULT(verbRoot2, "  - partitioning", t2);
            t2.Start();

            // Now construct a second graph that is partitioned in serial by
            // this rank.
            std::vector<unsigned int> nodeElmts;
            partitioner->GetElementIDs(interRank, nodeElmts);

            if (innerSize > 1)
            {
                // Construct map of element entities for partitioner.
                std::map<int, MeshEntity> partElmts;
                std::unordered_map<int, int> row2elmtid, elmtid2row;

                int vcnt = 0;

                // We need to keep track of which elements in the new partition
                // correspond to elemental IDs for later (in a similar manner to
                // row2id).
                for (auto &elmtRow : nodeElmts)
                {
                    row2elmtid[vcnt]                  = elmts[elmtRow].origId;
                    elmtid2row[elmts[elmtRow].origId] = vcnt;
                    MeshEntity elmt                   = elmts[elmtRow];
                    elmt.ghost                        = false;
                    partElmts[vcnt++]                 = elmt;
                }

                // Create temporary serial communicator for serial partitioning.
                auto tmpComm = LibUtilities::GetCommFactory().CreateInstance(
                    "Serial", 0, 0);

                MeshPartitionSharedPtr partitioner =
                    GetMeshPartitionFactory().CreateInstance(
                        "Scotch", session, tmpComm, meshDimension, partElmts,
                        CreateCompositeDescriptor(elmtid2row));

                t2.Stop();
                TIME_RESULT(verbRoot2, "  - inner partition setup", t2);
                t2.Start();

                partitioner->PartitionMesh(innerSize, true, false, 0);

                t2.Stop();
                TIME_RESULT(verbRoot2, "  - inner partitioning", t2);
                t2.Start();

                // Send contributions to remaining processors.
                for (int i = 1; i < innerSize; ++i)
                {
                    std::vector<unsigned int> tmp;
                    partitioner->GetElementIDs(i, tmp);
                    size_t tmpsize = tmp.size();
                    for (int j = 0; j < tmpsize; ++j)
                    {
                        tmp[j] = row2elmtid[tmp[j]];
                    }
                    innerComm->Send(i, tmpsize);
                    innerComm->Send(i, tmp);
                }

                t2.Stop();
                TIME_RESULT(verbRoot2, "  - inner partition scatter", t2);

                std::vector<unsigned int> tmp;
                partitioner->GetElementIDs(0, tmp);

                for (auto &tmpId : tmp)
                {
                    toRead.insert(row2elmtid[tmpId]);
                }
            }
            else
            {
                for (auto &tmpId : nodeElmts)
                {
                    toRead.insert(row2id[tmpId]);
                }
            }
        }
        else // Serial: Fill toRead with Elmt.origId
        {
            for (auto &tmpId : elmts)
            {
                toRead.insert(tmpId.origId);
            }
        }
    }
    else
    {
        // For multi-level partitioning, the innermost rank receives its
        // partitions from rank 0 on each node.
        size_t tmpSize;
        innerComm->Recv(0, tmpSize);
        std::vector<unsigned int> tmp(tmpSize);
        innerComm->Recv(0, tmp);

        for (auto &tmpId : tmp)
        {
            toRead.insert(tmpId);
        }
    }

    t.Stop();
    TIME_RESULT(verbRoot, "partitioning total", t);

    // Since objects are going to be constructed starting from vertices, we
    // now need to recurse down the geometry facet dimensions to figure out
    // which rows to read from each dataset.
    std::vector<int> vertIDs, segIDs, triIDs, quadIDs;
    std::vector<int> tetIDs, prismIDs, pyrIDs, hexIDs;
    std::vector<int> segData, triData, quadData, tetData;
    std::vector<int> prismData, pyrData, hexData;
    std::vector<NekDouble> vertData;

    auto &vertSet     = m_meshGraph->GetGeomMap<PointGeom>();
    auto &segGeoms    = m_meshGraph->GetGeomMap<SegGeom>();
    auto &triGeoms    = m_meshGraph->GetGeomMap<TriGeom>();
    auto &quadGeoms   = m_meshGraph->GetGeomMap<QuadGeom>();
    auto &hexGeoms    = m_meshGraph->GetGeomMap<HexGeom>();
    auto &pyrGeoms    = m_meshGraph->GetGeomMap<PyrGeom>();
    auto &prismGeoms  = m_meshGraph->GetGeomMap<PrismGeom>();
    auto &tetGeoms    = m_meshGraph->GetGeomMap<TetGeom>();
    auto &curvedEdges = m_meshGraph->GetCurvedEdges();
    auto &curvedFaces = m_meshGraph->GetCurvedFaces();

    if (meshDimension == 3)
    {
        t.Start();
        // Read 3D data
        ReadGeometryData(hexGeoms, "HEX", toRead, hexIDs, hexData);
        ReadGeometryData(pyrGeoms, "PYR", toRead, pyrIDs, pyrData);
        ReadGeometryData(prismGeoms, "PRISM", toRead, prismIDs, prismData);
        ReadGeometryData(tetGeoms, "TET", toRead, tetIDs, tetData);

        toRead.clear();
        UniqueValues(toRead, hexData, pyrData, prismData, tetData);
        t.Stop();
        TIME_RESULT(verbRoot, "read 3D elements", t);
    }

    if (meshDimension >= 2)
    {
        t.Start();
        // Read 2D data
        ReadGeometryData(triGeoms, "TRI", toRead, triIDs, triData);
        ReadGeometryData(quadGeoms, "QUAD", toRead, quadIDs, quadData);

        toRead.clear();
        UniqueValues(toRead, triData, quadData);
        t.Stop();
        TIME_RESULT(verbRoot, "read 2D elements", t);
    }

    if (meshDimension >= 1)
    {
        t.Start();
        // Read 1D data
        ReadGeometryData(segGeoms, "SEG", toRead, segIDs, segData);

        toRead.clear();
        UniqueValues(toRead, segData);
        t.Stop();
        TIME_RESULT(verbRoot, "read 1D elements", t);
    }

    t.Start();
    ReadGeometryData(vertSet, "VERT", toRead, vertIDs, vertData);
    t.Stop();
    TIME_RESULT(verbRoot, "read 0D elements", t);

    // Now start to construct geometry objects, starting from vertices
    // upwards.
    t.Start();
    FillGeomMap(vertSet, CurveMap(), vertIDs, vertData);
    t.Stop();
    TIME_RESULT(verbRoot, "construct 0D elements", t);

    if (meshDimension >= 1)
    {
        // Read curves
        toRead.clear();
        for (auto &edge : segIDs)
        {
            toRead.insert(edge);
        }
        ReadCurveMap(curvedEdges, "CURVE_EDGE", toRead);

        t.Start();
        FillGeomMap(segGeoms, curvedEdges, segIDs, segData);
        t.Stop();
        TIME_RESULT(verbRoot, "construct 1D elements", t);
    }

    if (meshDimension >= 2)
    {
        // Read curves
        toRead.clear();
        for (auto &face : triIDs)
        {
            toRead.insert(face);
        }
        for (auto &face : quadIDs)
        {
            toRead.insert(face);
        }
        ReadCurveMap(curvedFaces, "CURVE_FACE", toRead);

        t.Start();
        FillGeomMap(triGeoms, curvedFaces, triIDs, triData);
        FillGeomMap(quadGeoms, curvedFaces, quadIDs, quadData);
        t.Stop();
        TIME_RESULT(verbRoot, "construct 2D elements", t);
    }

    if (meshDimension >= 3)
    {
        t.Start();
        FillGeomMap(hexGeoms, CurveMap(), hexIDs, hexData);
        FillGeomMap(prismGeoms, CurveMap(), prismIDs, prismData);
        FillGeomMap(pyrGeoms, CurveMap(), pyrIDs, pyrData);
        FillGeomMap(tetGeoms, CurveMap(), tetIDs, tetData);
        t.Stop();
        TIME_RESULT(verbRoot, "construct 3D elements", t);
    }

    // Populate m_bndRegOrder.
    if (session->DefinesElement("NEKTAR/CONDITIONS"))
    {
        std::set<int> vBndRegionIdList;
        TiXmlElement *vConditions =
            new TiXmlElement(*session->GetElement("Nektar/Conditions"));
        TiXmlElement *vBndRegions =
            vConditions->FirstChildElement("BOUNDARYREGIONS");
        // Use fine-level for mesh partition (Parallel-in-Time)
        LibUtilities::SessionReader::GetXMLElementTimeLevel(vBndRegions, 0);
        TiXmlElement *vItem;

        if (vBndRegions)
        {
            auto &graph_bndRegOrder = m_meshGraph->GetBndRegionOrdering();
            vItem                   = vBndRegions->FirstChildElement();
            while (vItem)
            {
                std::string vSeqStr = vItem->FirstChild()->ToText()->Value();
                std::string::size_type indxBeg = vSeqStr.find_first_of('[') + 1;
                std::string::size_type indxEnd = vSeqStr.find_last_of(']') - 1;
                vSeqStr = vSeqStr.substr(indxBeg, indxEnd - indxBeg + 1);

                std::vector<unsigned int> vSeq;
                ParseUtils::GenerateSeqVector(vSeqStr.c_str(), vSeq);

                int p                = atoi(vItem->Attribute("ID"));
                m_bndRegOrder[p]     = vSeq;
                graph_bndRegOrder[p] = vSeq;
                vItem                = vItem->NextSiblingElement();
            }
        }
    }

    all.Stop();
    TIME_RESULT(verbRoot, "total time", all);
}

/**
 * @brief Construct a single Geometry object of type #T from one row of
 * raw dataset data and register it with the #MeshGraph.
 *
 * One explicit specialisation exists per geometry type. Each interprets the
 * `nGeomData` values of a `MESH/<shape>` row according to the format described
 * in the file overview — coordinates for points, bounding vertex IDs for
 * segments, bounding edge IDs for 2D faces, bounding face IDs for 3D elements —
 * resolving the referenced sub-entity IDs against the already-constructed
 * lower-dimensional geometries. The unspecialised primary template is a no-op.
 *
 * @tparam T         Geometry type to construct.
 * @tparam DataType  Element type of @p data (NekDouble for points, int
 *                   otherwise).
 * @param  geomMap   View of the map this geometry belongs to (unused; present
 *                   for template deduction).
 * @param  id        Global ID to assign to the new geometry.
 * @param  data      Pointer to this entity's row of dataset values.
 * @param  curve     Associated curvature for the entity, or nullptr if straight
 *                   sided (used only by the 1D/2D specialisations).
 */
template <class T, typename DataType>
void MeshGraphIOHDF5::ConstructGeomObject(
    [[maybe_unused]] GeomMapView<T> &geomMap, [[maybe_unused]] int id,
    [[maybe_unused]] DataType *data, [[maybe_unused]] Curve *curve)
{
}

/// @copydoc MeshGraphIOHDF5::ConstructGeomObject
template <>
void MeshGraphIOHDF5::ConstructGeomObject(
    [[maybe_unused]] GeomMapView<PointGeom> &geomMap, int id, NekDouble *data,
    [[maybe_unused]] Curve *curve)
{
    m_meshGraph->CreatePointGeom(m_meshGraph->GetSpaceDimension(), id, data[0],
                                 data[1], data[2]);
}

/// @copydoc MeshGraphIOHDF5::ConstructGeomObject
template <>
void MeshGraphIOHDF5::ConstructGeomObject(
    [[maybe_unused]] GeomMapView<SegGeom> &geomMap, int id, int *data,
    Curve *curve)
{
    std::array<PointGeom *, 2> pts = {m_meshGraph->GetPointGeom(data[0]),
                                      m_meshGraph->GetPointGeom(data[1])};

    m_meshGraph->CreateSegGeom(id, m_meshGraph->GetSpaceDimension(), pts,
                               curve);
}

/// @copydoc MeshGraphIOHDF5::ConstructGeomObject
template <>
void MeshGraphIOHDF5::ConstructGeomObject(
    [[maybe_unused]] GeomMapView<TriGeom> &geomMap, int id, int *data,
    Curve *curve)
{
    std::array<SegGeom *, 3> segs = {m_meshGraph->GetSegGeom(data[0]),
                                     m_meshGraph->GetSegGeom(data[1]),
                                     m_meshGraph->GetSegGeom(data[2])};
    m_meshGraph->CreateTriGeom(id, segs, curve);
}

/// @copydoc MeshGraphIOHDF5::ConstructGeomObject
template <>
void MeshGraphIOHDF5::ConstructGeomObject(
    [[maybe_unused]] GeomMapView<QuadGeom> &geomMap, int id, int *data,
    Curve *curve)
{
    std::array<SegGeom *, 4> segs = {
        m_meshGraph->GetSegGeom(data[0]), m_meshGraph->GetSegGeom(data[1]),
        m_meshGraph->GetSegGeom(data[2]), m_meshGraph->GetSegGeom(data[3])};
    m_meshGraph->CreateQuadGeom(id, segs, curve);
}

/// @copydoc MeshGraphIOHDF5::ConstructGeomObject
template <>
void MeshGraphIOHDF5::ConstructGeomObject(
    [[maybe_unused]] GeomMapView<TetGeom> &geomMap, int id, int *data,
    [[maybe_unused]] Curve *curve)
{
    std::array<TriGeom *, 4> faces = {
        m_meshGraph->GetTriGeom(data[0]), m_meshGraph->GetTriGeom(data[1]),
        m_meshGraph->GetTriGeom(data[2]), m_meshGraph->GetTriGeom(data[3])};

    auto geom = m_meshGraph->CreateTetGeom(id, faces);
    m_meshGraph->PopulateFaceToElMap(geom, TetGeom::kNfaces);
}

/// @copydoc MeshGraphIOHDF5::ConstructGeomObject
template <>
void MeshGraphIOHDF5::ConstructGeomObject(
    [[maybe_unused]] GeomMapView<PyrGeom> &geomMap, int id, int *data,
    [[maybe_unused]] Curve *curve)
{
    std::array<Geometry2D *, 5> faces = {m_meshGraph->GetGeometry2D(data[0]),
                                         m_meshGraph->GetGeometry2D(data[1]),
                                         m_meshGraph->GetGeometry2D(data[2]),
                                         m_meshGraph->GetGeometry2D(data[3]),
                                         m_meshGraph->GetGeometry2D(data[4])};

    auto geom = m_meshGraph->CreatePyrGeom(id, faces);
    m_meshGraph->PopulateFaceToElMap(geom, PyrGeom::kNfaces);
}

/// @copydoc MeshGraphIOHDF5::ConstructGeomObject
template <>
void MeshGraphIOHDF5::ConstructGeomObject(
    [[maybe_unused]] GeomMapView<PrismGeom> &geomMap, int id, int *data,
    [[maybe_unused]] Curve *curve)
{
    std::array<Geometry2D *, 5> faces = {m_meshGraph->GetGeometry2D(data[0]),
                                         m_meshGraph->GetGeometry2D(data[1]),
                                         m_meshGraph->GetGeometry2D(data[2]),
                                         m_meshGraph->GetGeometry2D(data[3]),
                                         m_meshGraph->GetGeometry2D(data[4])};

    auto geom = m_meshGraph->CreatePrismGeom(id, faces);
    m_meshGraph->PopulateFaceToElMap(geom, PrismGeom::kNfaces);
}

/// @copydoc MeshGraphIOHDF5::ConstructGeomObject
template <>
void MeshGraphIOHDF5::ConstructGeomObject(
    [[maybe_unused]] GeomMapView<HexGeom> &geomMap, int id, int *data,
    [[maybe_unused]] Curve *curve)
{
    std::array<QuadGeom *, 6> faces = {
        m_meshGraph->GetQuadGeom(data[0]), m_meshGraph->GetQuadGeom(data[1]),
        m_meshGraph->GetQuadGeom(data[2]), m_meshGraph->GetQuadGeom(data[3]),
        m_meshGraph->GetQuadGeom(data[4]), m_meshGraph->GetQuadGeom(data[5])};

    auto geom = m_meshGraph->CreateHexGeom(id, faces);
    m_meshGraph->PopulateFaceToElMap(geom, HexGeom::kNfaces);
}

/**
 * @brief Construct every Geometry object for one shape type from the raw data
 * read out of its `MESH/<shape>` dataset.
 *
 * Walks the flat @p geomData buffer in strides of `nGeomData` (one stride per
 * entity), pairing each row with its global ID from @p ids and any matching
 * curvature from @p curveMap, and delegates per-entity construction to
 * ConstructGeomObject(). A fast path is taken when no curvature is present so
 * that the per-row map lookup is skipped.
 *
 * @tparam T         Geometry type being constructed.
 * @tparam DataType  Element type of @p geomData (NekDouble for points, int
 *                   otherwise).
 * @param  geomMap   Destination geometry map for this shape type.
 * @param  curveMap  Curvature for the entities, keyed by global ID; may be
 *                   empty.
 * @param  ids       Global IDs, one per entity, in row order.
 * @param  geomData  Flattened dataset values, `nGeomData` per entity.
 */
template <class T, typename DataType>
void MeshGraphIOHDF5::FillGeomMap(GeomMapView<T> &geomMap,
                                  const CurveMap &curveMap,
                                  std::vector<int> &ids,
                                  std::vector<DataType> &geomData)
{
    const int nGeomData = GetGeomDataDim(geomMap);
    const int nRows     = geomData.size() / nGeomData;
    Curve *empty        = nullptr;

    // Construct geometry object.
    if (curveMap.size() > 0)
    {
        for (int i = 0, cnt = 0; i < nRows; i++, cnt += nGeomData)
        {
            auto cIt = curveMap.find(ids[i]);
            ConstructGeomObject(geomMap, ids[i], &geomData[cnt],
                                cIt == curveMap.end() ? empty
                                                      : cIt->second.get());
        }
    }
    else
    {
        for (int i = 0, cnt = 0; i < nRows; i++, cnt += nGeomData)
        {
            ConstructGeomObject(geomMap, ids[i], &geomData[cnt], empty);
        }
    }
}

/**
 * @brief Selectively read the rows of a `MESH/<shape>` dataset whose global IDs
 * are required by this rank.
 *
 * Reads the full `MAPS/<shape>` ID list (which is small — one int per entity),
 * scans it for IDs present in @p readIds, and builds an HDF5 point selection
 * (a list of `(row, col)` coordinate pairs) so that only the matching rows of
 * the 2D `MESH/<shape>` data array are actually read off disk. The collected
 * global IDs are returned in @p ids and the corresponding flattened geometry
 * values in @p geomData, in matching row order. If the dataset is absent (e.g.
 * a shape type not present in this mesh) the function returns immediately.
 *
 * @tparam T         Geometry type associated with @p dataSet.
 * @tparam DataType  Element type read back (NekDouble for `VERT`, int
 *                   otherwise).
 * @param  geomMap   View used to determine the per-row column count.
 * @param  dataSet   Dataset name (`"VERT"`, `"SEG"`, `"TRI"`, ...).
 * @param  readIds   Set of global IDs this rank needs.
 * @param[out] ids       Global IDs actually read, in row order.
 * @param[out] geomData  Flattened values for the read rows.
 */
template <class T, typename DataType>
void MeshGraphIOHDF5::ReadGeometryData(GeomMapView<T> &geomMap,
                                       std::string dataSet,
                                       const std::unordered_set<int> &readIds,
                                       std::vector<int> &ids,
                                       std::vector<DataType> &geomData)
{
    if (!m_mesh->ContainsDataSet(dataSet))
    {
        return;
    }

    // Open mesh dataset
    H5::DataSetSharedPtr data    = m_mesh->OpenDataSet(dataSet);
    H5::DataSpaceSharedPtr space = data->GetSpace();
    std::vector<hsize_t> dims    = space->GetDims();

    // Open metadata dataset
    H5::DataSetSharedPtr mdata    = m_maps->OpenDataSet(dataSet);
    H5::DataSpaceSharedPtr mspace = mdata->GetSpace();
    std::vector<hsize_t> mdims    = mspace->GetDims();

    ASSERTL0(mdims[0] == dims[0], "map and data set lengths do not match");

    const int nGeomData = GetGeomDataDim(geomMap);

    // Read all IDs
    std::vector<int> allIds;
    mdata->Read(allIds, mspace);

    // Selective reading; clear data space range so that we can select
    // certain rows from the datasets.
    space->ClearRange();

    int i = 0;
    std::vector<hsize_t> coords;
    for (auto &id : allIds)
    {
        if (readIds.find(id) != readIds.end())
        {
            for (int j = 0; j < nGeomData; ++j)
            {
                coords.push_back(i);
                coords.push_back(j);
            }
            ids.push_back(id);
        }
        ++i;
    }

    space->SetSelection(coords.size() / 2, coords);

    // Read selected data.
    data->Read(geomData, space, m_readPL);
}

/**
 * @brief Read curvature for a selected set of edges or faces from a
 * `CURVE_EDGE`/`CURVE_FACE` dataset and its shared `CURVE_NODES` store.
 *
 * Proceeds in two selective reads. First the `MAPS/<dsName>` IDs are scanned
 * against @p readIds to pick out the descriptor rows of interest; each such row
 * yields a point count, a LibUtilities::PointsType and an offset into
 * `CURVE_NODES` (the three columns described in the file overview). The point
 * counts and offsets are then turned into a second HDF5 point selection that
 * reads exactly the required coordinate rows from the shared `CURVE_NODES`
 * dataset, which are used to populate the Curve objects' point lists.
 *
 * A collective reduction over the number of curves to be read is performed so
 * that, under collective MPI-IO, all ranks agree on whether any read takes
 * place; if no rank requires any curve the function returns early. Absence of
 * the dataset (an uncurved mesh) is likewise handled by an early return.
 *
 * @param[out] curveMap  Destination map of global ID -> Curve, populated here.
 * @param  dsName    Curve descriptor dataset name (`"CURVE_EDGE"` or
 *                   `"CURVE_FACE"`).
 * @param  readIds   Global edge/face IDs whose curvature is required.
 */
void MeshGraphIOHDF5::ReadCurveMap(CurveMap &curveMap, std::string dsName,
                                   const std::unordered_set<int> &readIds)
{
    auto &curveNodes = m_meshGraph->GetAllCurveNodes();

    // If dataset does not exist, exit.
    if (!m_mesh->ContainsDataSet(dsName))
    {
        return;
    }

    // Open up curve map data.
    H5::DataSetSharedPtr curveData    = m_mesh->OpenDataSet(dsName);
    H5::DataSpaceSharedPtr curveSpace = curveData->GetSpace();

    // Open up ID data set.
    H5::DataSetSharedPtr idData    = m_maps->OpenDataSet(dsName);
    H5::DataSpaceSharedPtr idSpace = idData->GetSpace();

    // Read all IDs and clear data space.
    std::vector<int> ids, newIds;
    idData->Read(ids, idSpace);
    curveSpace->ClearRange();

    // Search IDs to figure out which curves to read.
    std::vector<hsize_t> curveSel;

    int cnt = 0;
    for (auto &id : ids)
    {
        if (readIds.find(id) != readIds.end())
        {
            curveSel.push_back(cnt);
            curveSel.push_back(0);
            curveSel.push_back(cnt);
            curveSel.push_back(1);
            curveSel.push_back(cnt);
            curveSel.push_back(2);
            newIds.push_back(id);
        }

        ++cnt;
    }

    // Check to see whether any processor will read anything
    auto toRead = newIds.size();
    m_session->GetComm()->GetRowComm()->AllReduce(toRead,
                                                  LibUtilities::ReduceSum);

    if (toRead == 0)
    {
        return;
    }

    // Now read curve map and read data.
    std::vector<int> curveInfo;
    curveSpace->SetSelection(curveSel.size() / 2, curveSel);
    curveData->Read(curveInfo, curveSpace, m_readPL);

    curveSel.clear();

    std::unordered_map<int, int> curvePtOffset;

    // Construct curves. We'll populate nodes in a minute!
    for (int i = 0, cnt = 0, cnt2 = 0; i < curveInfo.size() / 3; ++i, cnt += 3)
    {
        CurveUniquePtr curve = ObjPoolManager<Curve>::AllocateUniquePtr(
            newIds[i], (LibUtilities::PointsType)curveInfo[cnt + 1]);

        curve->m_points.resize(curveInfo[cnt]);

        const int ptOffset = curveInfo[cnt + 2];

        for (int j = 0; j < curveInfo[cnt]; ++j)
        {
            // ptoffset gives us the row, multiply by 3 for number of
            // coordinates.
            curveSel.push_back(ptOffset + j);
            curveSel.push_back(0);
            curveSel.push_back(ptOffset + j);
            curveSel.push_back(1);
            curveSel.push_back(ptOffset + j);
            curveSel.push_back(2);
        }

        // Store the offset so we know to come back later on to fill in
        // these points.
        curvePtOffset[newIds[i]] = 3 * cnt2;
        cnt2 += curveInfo[cnt];

        curveMap[newIds[i]] = std::move(curve);
    }

    curveInfo.clear();

    // Open node data spacee.
    H5::DataSetSharedPtr nodeData    = m_mesh->OpenDataSet("CURVE_NODES");
    H5::DataSpaceSharedPtr nodeSpace = nodeData->GetSpace();

    nodeSpace->ClearRange();
    nodeSpace->SetSelection(curveSel.size() / 2, curveSel);

    std::vector<NekDouble> nodeRawData;
    nodeData->Read(nodeRawData, nodeSpace, m_readPL);

    // Go back and populate data from nodes.
    for (auto &cIt : curvePtOffset)
    {
        Curve *curve = curveMap[cIt.first].get();

        // Create nodes.
        int cnt = cIt.second;
        for (int i = 0; i < curve->m_points.size(); ++i, cnt += 3)
        {
            curveNodes.emplace_back(
                ObjPoolManager<PointGeom>::AllocateUniquePtr(
                    0, m_meshGraph->GetSpaceDimension(), nodeRawData[cnt],
                    nodeRawData[cnt + 1], nodeRawData[cnt + 2]));
            curve->m_points[i] = curveNodes.back().get();
        }
    }
}

/**
 * @brief Read the `DOMAIN` dataset and rebuild the #MeshGraph domain map.
 *
 * Each entry of the variable-length string `MESH/DOMAIN` dataset is expanded
 * (via MeshGraph::GetCompositeList()) into the set of composites forming a
 * domain, and keyed by the domain ID taken from the companion `MAPS/DOMAIN`
 * dataset.
 *
 * Two on-disk layouts are supported for backwards compatibility:
 *   - format version 1: a single string describing the entire domain, stored
 *     under domain ID 0;
 *   - format version 2 (current): one string per domain, each a compressed
 *     sequence of composite IDs, with explicit domain IDs in `MAPS/DOMAIN`.
 */
void MeshGraphIOHDF5::ReadDomain()
{
    auto &domain = m_meshGraph->GetDomain();

    if (m_inFormatVersion == 1)
    {
        std::map<int, CompositeSharedPtr> fullDomain;
        H5::DataSetSharedPtr dst     = m_mesh->OpenDataSet("DOMAIN");
        H5::DataSpaceSharedPtr space = dst->GetSpace();

        std::vector<std::string> data;
        dst->ReadVectorString(data, space, m_readPL);
        m_meshGraph->GetCompositeList(data[0], fullDomain);
        domain[0] = fullDomain;

        return;
    }

    std::vector<CompositeMap> fullDomain;
    H5::DataSetSharedPtr dst     = m_mesh->OpenDataSet("DOMAIN");
    H5::DataSpaceSharedPtr space = dst->GetSpace();

    std::vector<std::string> data;
    dst->ReadVectorString(data, space, m_readPL);
    for (auto &dIt : data)
    {
        fullDomain.push_back(CompositeMap());
        m_meshGraph->GetCompositeList(dIt, fullDomain.back());
    }

    H5::DataSetSharedPtr mdata    = m_maps->OpenDataSet("DOMAIN");
    H5::DataSpaceSharedPtr mspace = mdata->GetSpace();

    std::vector<int> ids;
    mdata->Read(ids, mspace);

    for (int i = 0; i < ids.size(); ++i)
    {
        domain[ids[i]] = fullDomain[i];
    }
}

/**
 * @brief Populate the trace ID set of a composite-based domain range from the
 * `COMPOSITE` dataset.
 *
 * When a domain range restricts the read to a list of composites
 * (`rng->m_compElmts`), this reads the `COMPOSITE` strings and their IDs, and
 * for every requested composite expands its bracketed ID sequence into
 * @c rng->m_traceIDs so that the range can later be applied during element
 * construction. Returns immediately if no composite range is in use.
 *
 * @param rng  The domain range to populate (modified in place).
 */
void MeshGraphIOHDF5::SetupCompositeRange(LibUtilities::DomainRangeShPtr &rng)
{
    if (!rng || rng->m_compElmts == false)
    {
        return; // composite range not being used.
    }

    std::string nm = "COMPOSITE";

    H5::DataSetSharedPtr data    = m_mesh->OpenDataSet(nm);
    H5::DataSpaceSharedPtr space = data->GetSpace();
    std::vector<hsize_t> dims    = space->GetDims();

    std::vector<std::string> comps;
    data->ReadVectorString(comps, space);

    H5::DataSetSharedPtr mdata    = m_maps->OpenDataSet(nm);
    H5::DataSpaceSharedPtr mspace = mdata->GetSpace();
    std::vector<hsize_t> mdims    = mspace->GetDims();

    std::vector<int> ids;
    mdata->Read(ids, mspace);

    for (int i = 0; i < dims[0]; i++)
    {

        if (rng->m_comps.count(ids[i]))
        {

            std::string compStr = comps[i];

            char type;
            std::istringstream strm(compStr);

            strm >> type;

            CompositeSharedPtr comp =
                MemoryManager<Composite>::AllocateSharedPtr();

            std::string::size_type indxBeg = compStr.find_first_of('[') + 1;
            std::string::size_type indxEnd = compStr.find_last_of(']') - 1;

            std::string indxStr =
                compStr.substr(indxBeg, indxEnd - indxBeg + 1);
            std::vector<unsigned int> seqVector;

            ParseUtils::GenerateSeqVector(indxStr, seqVector);

            for (auto it : seqVector) // add ids to Traceid
            {
                rng->m_traceIDs.insert(it);
            }
        }
    }
}

/**
 * @brief Read the `COMPOSITE` dataset and build the #MeshGraph composite map.
 *
 * Reads the variable-length composite strings from `MESH/COMPOSITE` and their
 * IDs from `MAPS/COMPOSITE`. Each string is parsed into its shape tag and
 * bracketed, run-length-encoded ID sequence (see the file overview for the tag
 * meanings). The sequence is recorded in the composite ordering and then
 * resolved against the already-constructed geometry maps, gathering the matched
 * Geometry pointers into a Composite. Range checking via
 * MeshGraph::CheckRange() is applied for the 2D/3D shapes so that a composite
 * spanning a clipped region only retains entities actually present locally.
 * Empty composites are dropped.
 *
 * Note the `'F'` (mixed face) tag is resolved against both the triangle and
 * quadrilateral maps, and `'S'`/`'E'` are treated identically as segments.
 */
void MeshGraphIOHDF5::ReadComposites()
{
    auto &vertSet                = m_meshGraph->GetGeomMap<PointGeom>();
    auto &segGeoms               = m_meshGraph->GetGeomMap<SegGeom>();
    auto &triGeoms               = m_meshGraph->GetGeomMap<TriGeom>();
    auto &quadGeoms              = m_meshGraph->GetGeomMap<QuadGeom>();
    auto &tetGeoms               = m_meshGraph->GetGeomMap<TetGeom>();
    auto &pyrGeoms               = m_meshGraph->GetGeomMap<PyrGeom>();
    auto &prismGeoms             = m_meshGraph->GetGeomMap<PrismGeom>();
    auto &hexGeoms               = m_meshGraph->GetGeomMap<HexGeom>();
    CompositeMap &meshComposites = m_meshGraph->GetComposites();

    std::string nm = "COMPOSITE";

    H5::DataSetSharedPtr data    = m_mesh->OpenDataSet(nm);
    H5::DataSpaceSharedPtr space = data->GetSpace();
    std::vector<hsize_t> dims    = space->GetDims();

    std::vector<std::string> comps;
    data->ReadVectorString(comps, space);

    H5::DataSetSharedPtr mdata    = m_maps->OpenDataSet(nm);
    H5::DataSpaceSharedPtr mspace = mdata->GetSpace();
    std::vector<hsize_t> mdims    = mspace->GetDims();

    std::vector<int> ids;
    mdata->Read(ids, mspace);

    auto &graph_compOrder = m_meshGraph->GetCompositeOrdering();
    for (int i = 0; i < dims[0]; i++)
    {
        std::string compStr = comps[i];

        char type;
        std::istringstream strm(compStr);

        strm >> type;

        CompositeSharedPtr comp = MemoryManager<Composite>::AllocateSharedPtr();

        std::string::size_type indxBeg = compStr.find_first_of('[') + 1;
        std::string::size_type indxEnd = compStr.find_last_of(']') - 1;

        std::string indxStr = compStr.substr(indxBeg, indxEnd - indxBeg + 1);
        std::vector<unsigned int> seqVector;

        ParseUtils::GenerateSeqVector(indxStr, seqVector);
        m_compOrder[ids[i]]     = seqVector;
        graph_compOrder[ids[i]] = seqVector;

        switch (type)
        {
            case 'V':
                for (auto &i : seqVector)
                {
                    auto it = vertSet.find(i);
                    if (it != vertSet.end())
                    {
                        comp->m_geomVec.push_back((*it).second);
                    }
                }
                break;
            case 'S':
            case 'E':
                for (auto &i : seqVector)
                {
                    auto it = segGeoms.find(i);
                    if (it != segGeoms.end())
                    {
                        comp->m_geomVec.push_back((*it).second);
                    }
                }
                break;
            case 'Q':
                for (auto &i : seqVector)
                {
                    auto it = quadGeoms.find(i);
                    if (it != quadGeoms.end())
                    {
                        if (m_meshGraph->CheckRange(*(*it).second))
                        {
                            comp->m_geomVec.push_back((*it).second);
                        }
                    }
                }
                break;
            case 'T':
                for (auto &i : seqVector)
                {
                    auto it = triGeoms.find(i);
                    if (it != triGeoms.end())
                    {
                        if (m_meshGraph->CheckRange(*(*it).second))
                        {
                            comp->m_geomVec.push_back((*it).second);
                        }
                    }
                }
                break;
            case 'F':
                for (auto &i : seqVector)
                {
                    auto it1 = quadGeoms.find(i);
                    if (it1 != quadGeoms.end())
                    {
                        if (m_meshGraph->CheckRange(*(*it1).second))
                        {
                            comp->m_geomVec.push_back((*it1).second);
                        }
                    }
                    auto it2 = triGeoms.find(i);
                    if (it2 != triGeoms.end())
                    {
                        if (m_meshGraph->CheckRange(*(*it2).second))
                        {
                            comp->m_geomVec.push_back((*it2).second);
                        }
                    }
                }
                break;
            case 'A':
                for (auto &i : seqVector)
                {
                    auto it = tetGeoms.find(i);
                    if (it != tetGeoms.end())
                    {
                        if (m_meshGraph->CheckRange(*(*it).second))
                        {
                            comp->m_geomVec.push_back((*it).second);
                        }
                    }
                }
                break;
            case 'P':
                for (auto &i : seqVector)
                {
                    auto it = pyrGeoms.find(i);
                    if (it != pyrGeoms.end())
                    {
                        if (m_meshGraph->CheckRange(*(*it).second))
                        {
                            comp->m_geomVec.push_back((*it).second);
                        }
                    }
                }
                break;
            case 'R':
                for (auto &i : seqVector)
                {
                    auto it = prismGeoms.find(i);
                    if (it != prismGeoms.end())
                    {
                        if (m_meshGraph->CheckRange(*(*it).second))
                        {
                            comp->m_geomVec.push_back((*it).second);
                        }
                    }
                }
                break;
            case 'H':
                for (auto &i : seqVector)
                {
                    auto it = hexGeoms.find(i);
                    if (it != hexGeoms.end())
                    {
                        if (m_meshGraph->CheckRange(*(*it).second))
                        {
                            comp->m_geomVec.push_back((*it).second);
                        }
                    }
                }
                break;
        }

        if (comp->m_geomVec.size() > 0)
        {
            meshComposites[ids[i]] = comp;
        }
    }
}

/**
 * @brief Build a lightweight composite descriptor from the `COMPOSITE` dataset
 * for use by the mesh partitioner.
 *
 * Reads and parses the composite strings exactly as ReadComposites() does, but
 * instead of constructing geometry it produces, per composite ID, a
 * `(ShapeType, [entity IDs])` pair. The entity ID list is filtered through
 * @p id2row so that only entities visible in the current contiguous partition
 * ordering are retained; composites left empty after filtering are skipped.
 * Because this descriptor is only used for partitioning, the distinction
 * between the `'Q'` and `'F'` face tags is unimportant and both map to
 * LibUtilities::eQuadrilateral.
 *
 * @param id2row  Map from global entity ID to partition row index; entities not
 *                present are excluded.
 * @return Map from composite ID to its shape type and filtered entity IDs.
 */
CompositeDescriptor MeshGraphIOHDF5::CreateCompositeDescriptor(
    std::unordered_map<int, int> &id2row)
{
    CompositeDescriptor ret;

    std::string nm = "COMPOSITE";

    H5::DataSetSharedPtr data    = m_mesh->OpenDataSet(nm);
    H5::DataSpaceSharedPtr space = data->GetSpace();
    std::vector<hsize_t> dims    = space->GetDims();

    std::vector<std::string> comps;
    data->ReadVectorString(comps, space);

    H5::DataSetSharedPtr mdata    = m_maps->OpenDataSet(nm);
    H5::DataSpaceSharedPtr mspace = mdata->GetSpace();
    std::vector<hsize_t> mdims    = mspace->GetDims();

    std::vector<int> ids;
    mdata->Read(ids, mspace);

    for (int i = 0; i < dims[0]; i++)
    {
        std::string compStr = comps[i];

        char type;
        std::istringstream strm(compStr);

        strm >> type;

        std::string::size_type indxBeg = compStr.find_first_of('[') + 1;
        std::string::size_type indxEnd = compStr.find_last_of(']') - 1;

        std::string indxStr = compStr.substr(indxBeg, indxEnd - indxBeg + 1);
        std::vector<unsigned int> seqVector;
        ParseUtils::GenerateSeqVector(indxStr, seqVector);

        LibUtilities::ShapeType shapeType = eNoShapeType;

        switch (type)
        {
            case 'V':
                shapeType = LibUtilities::ePoint;
                break;
            case 'S':
            case 'E':
                shapeType = LibUtilities::eSegment;
                break;
            case 'Q':
            case 'F':
                // Note that for HDF5, the composite descriptor is only used
                // for partitioning purposes so 'F' tag is not really going
                // to be critical in this context.
                shapeType = LibUtilities::eQuadrilateral;
                break;
            case 'T':
                shapeType = LibUtilities::eTriangle;
                break;
            case 'A':
                shapeType = LibUtilities::eTetrahedron;
                break;
            case 'P':
                shapeType = LibUtilities::ePyramid;
                break;
            case 'R':
                shapeType = LibUtilities::ePrism;
                break;
            case 'H':
                shapeType = LibUtilities::eHexahedron;
                break;
        }

        ASSERTL0(shapeType != eNoShapeType, "Invalid shape.");

        std::vector<int> filteredVector;
        for (auto &compElmt : seqVector)
        {
            if (id2row.find(compElmt) == id2row.end())
            {
                continue;
            }

            filteredVector.push_back(compElmt);
        }

        if (filteredVector.size() == 0)
        {
            continue;
        }

        ret[ids[i]] = std::make_pair(shapeType, filteredVector);
    }

    return ret;
}

/**
 * @brief Return the @p i-th data value to be written for a geometry @p geom of
 * type @tparam T.
 *
 * This is the write-side counterpart of the `MESH/<shape>` row layout: it
 * yields, in order, the values that fill one row of the dataset. The
 * overload is selected on `T::kDim`:
 *   - 0D: the @p i-th coordinate of the point (NekDouble);
 *   - 1D: the global ID of the @p i-th bounding vertex;
 *   - 2D: the global ID of the @p i-th bounding edge;
 *   - 3D: the global ID of the @p i-th bounding face.
 *
 * @tparam T  Geometry type.
 * @param geom  The geometry whose data is being serialised.
 * @param i     Column index within the row, in `[0, nGeomData)`.
 * @return The value for column @p i (NekDouble for points, int otherwise).
 */
template <class T, typename std::enable_if<T::kDim == 0, int>::type = 0>
inline NekDouble GetGeomData(T *geom, int i)
{
    return (*geom)(i);
}

/// @copydoc GetGeomData
template <class T, typename std::enable_if<T::kDim == 1, int>::type = 0>
inline int GetGeomData(T *geom, int i)
{
    return geom->GetVid(i);
}

/// @copydoc GetGeomData
template <class T, typename std::enable_if<T::kDim == 2, int>::type = 0>
inline int GetGeomData(T *geom, int i)
{
    return geom->GetEid(i);
}

/// @copydoc GetGeomData
template <class T, typename std::enable_if<T::kDim == 3, int>::type = 0>
inline int GetGeomData(T *geom, int i)
{
    return geom->GetFid(i);
}

/**
 * @brief Assign a unique owner to each entity and compute this rank's
 * contiguous write offset for a collective dataset write.
 *
 * In parallel a geometry (especially a shared vertex, edge or face) may be held
 * by several ranks, but each must be written to the dataset exactly once. This
 * routine establishes a globally consistent ownership: when @p checkUnique is
 * set it uses a gslib unique reduction (Gs::Unique) over the entity IDs so that
 * for every shared ID exactly one rank is left as owner. It then forms the
 * exclusive prefix sum of per-rank owned counts to give @p writeOffset, the
 * first global row this rank writes, so that all ranks' owned rows tile the
 * dataset contiguously without gaps or overlap.
 *
 * @param  idMap        Global IDs of the entities held locally.
 * @param  checkUnique  If true, deduplicate shared entities across ranks; pass
 *                      false for entities known to be rank-exclusive.
 * @param  comm         Communicator over which ownership is resolved.
 * @param[out] owned       Per-local-entity flag: true if this rank owns it.
 * @param[out] writeOffset This rank's first global row index in the dataset.
 * @return The total (global) number of owned rows, i.e. the dataset length.
 */
hsize_t MeshGraphIOHDF5::GetGeomWriteLayout(
    const std::vector<int> &idMap, bool checkUnique,
    const LibUtilities::CommSharedPtr &comm, std::vector<bool> &owned,
    hsize_t &writeOffset)
{
    const size_t nLocal = idMap.size();
    owned.assign(nLocal, true);

    if (checkUnique)
    {
        // +1 so genuine IDs are strictly positive: gslib ignores 0 and treats
        // sign specially.
        Array<OneD, long> gsId(nLocal);
        for (size_t i = 0; i < nLocal; ++i)
        {
            gsId[i] = static_cast<long>(idMap[i]) + 1;
        }

        // Negates all but one occurrence of each shared ID across ranks;
        // the rank left positive is the unique owner (globally consistent).
        Gs::Unique(gsId, comm);

        // If the ID is positive then it is owned by this rank, mark the entry
        // as true
        for (size_t i = 0; i < nLocal; ++i)
        {
            owned[i] = gsId[i] > 0;
        }
    }

    // Compute number of owned ids on this rank
    const unsigned long nOwned = std::count(owned.begin(), owned.end(), true);

    // Exclusive prefix sum of owned counts -> this rank's first global
    // row. Could be replaced by MPI_Exscan if we had a wrapper
    const int nProc = comm->GetSize();
    const int rank  = comm->GetRank();
    Array<OneD, unsigned long> counts(nProc, 0ul);
    counts[rank] = nOwned;
    comm->AllReduce(counts, LibUtilities::ReduceSum);

    writeOffset   = 0;
    hsize_t total = 0;
    for (int p = 0; p < nProc; ++p)
    {
        if (p < rank)
        {
            writeOffset += counts[p];
        }
        total += counts[p];
    }
    return total;
}

/**
 * @brief Collectively write the `MESH/<datasetName>` and `MAPS/<datasetName>`
 * datasets for one geometry shape type.
 *
 * Gathers the local entities of type @tparam T into a flat data buffer (using
 * GetGeomData() for the per-row values) and an ID list, resolves single
 * ownership and the contiguous per-rank write offset via GetGeomWriteLayout(),
 * then writes each rank's owned rows into its own hyperslab of the two
 * datasets:
 *   - `MESH/<datasetName>` : 2D `[nGlobal, nGeomData]` of coordinates (points)
 *     or sub-entity IDs (everything else);
 *   - `MAPS/<datasetName>` : 1D `[nGlobal]` of the entities' global IDs.
 *
 * Ranks owning no rows still participate in the (collective) dataset creation
 * and write with an empty selection. When @p checkUnique is set, the resolved
 * ownership is cached in #m_geomOwners so the subsequent curve write can reuse
 * the same owner for each edge/face. If no rank holds any entity of this type
 * the datasets are not created.
 *
 * @tparam T           Geometry type being written.
 * @param  geomMap     Local map of entities of this type.
 * @param  datasetName Dataset base name (`"VERT"`, `"SEG"`, ...), shared by the
 *                     `MESH/` and `MAPS/` groups.
 * @param  checkUnique Whether entities may be shared across ranks and so need
 *                     ownership deduplication (true for facets, false for
 *                     top-level elements).
 * @param  comm        Communicator over which the collective write is made.
 */
template <class T>
void MeshGraphIOHDF5::WriteGeometryMap(GeomMapView<T> &geomMap,
                                       std::string datasetName,
                                       bool checkUnique,
                                       LibUtilities::CommSharedPtr &comm)
{
    typedef typename std::conditional<std::is_same_v<T, PointGeom>, NekDouble,
                                      int>::type DataType;

    const int nGeomData = GetGeomDataDim(geomMap);
    const size_t nLocal = geomMap.size();

    // Compute number of geometries of this type across all processors
    size_t nGlobal = nLocal;
    comm->AllReduce(nGlobal, LibUtilities::ReduceSum);

    // If we have no geometries to write across all processors, then return.
    if (nGlobal == 0)
    {
        return;
    }

    // Construct maps for storage of geometry IDs & geometry data.
    std::vector<int> idMap(nLocal);
    std::vector<DataType> data(nLocal * nGeomData);

    int cnt1 = 0, cnt2 = 0;
    for (auto [id, geom] : geomMap)
    {
        idMap[cnt1++] = id;

        for (int j = 0; j < nGeomData; ++j)
        {
            data[cnt2 + j] = GetGeomData(geom, j);
        }

        cnt2 += nGeomData;
    }

    std::vector<bool> owned;
    hsize_t writeOffset = 0;
    hsize_t nRowsGlobal =
        GetGeomWriteLayout(idMap, checkUnique, comm, owned, writeOffset);

    // Compact to owned-only, data and idMap in lockstep.
    std::vector<int> ownedIds;
    std::vector<DataType> ownedData;
    for (size_t i = 0; i < nLocal; ++i)
    {
        if (!owned[i])
        {
            continue;
        }
        ownedIds.push_back(idMap[i]);
        ownedData.insert(ownedData.end(), data.begin() + i * nGeomData,
                         data.begin() + (i + 1) * nGeomData);
    }

    // Store ownership for later potential use in curve data
    if (checkUnique)
    {
        auto &s = m_geomOwners[datasetName];
        for (size_t i = 0; i < nLocal; ++i)
        {
            if (owned[i])
            {
                s.insert(idMap[i]);
            }
        }
    }

    const hsize_t nOwned = ownedIds.size();

    // Write out maps dataset
    {
        std::vector<hsize_t> dims = {nRowsGlobal, (hsize_t)nGeomData};
        H5::DataTypeSharedPtr tp  = H5::DataType::OfObject(DataType{});
        H5::DataSpaceSharedPtr ds = std::make_shared<H5::DataSpace>(dims);
        H5::DataSetSharedPtr dst  = m_mesh->CreateDataSet(datasetName, tp, ds);

        H5::DataSpaceSharedPtr fs = dst->GetSpace();
        if (nOwned)
        {
            fs->SelectRange({writeOffset, 0}, {nOwned, (hsize_t)nGeomData});
        }
        else
        {
            // still join the collective call even if nothing to write
            fs->ClearRange();
        }
        dst->Write(ownedData, fs);
    }

    // Write out ids dataset
    {
        std::vector<hsize_t> dims = {nRowsGlobal};
        H5::DataTypeSharedPtr tp  = H5::DataType::OfObject(nGeomData);
        H5::DataSpaceSharedPtr ds = std::make_shared<H5::DataSpace>(dims);
        H5::DataSetSharedPtr dst  = m_maps->CreateDataSet(datasetName, tp, ds);

        H5::DataSpaceSharedPtr fs = dst->GetSpace();
        if (nOwned)
        {
            fs->SelectRange(writeOffset, nOwned);
        }
        else
        {
            // still join the collective call even if nothing to write
            fs->ClearRange();
        }
        dst->Write(ownedIds, fs);
    }
}

/**
 * @brief Collectively write one curve descriptor dataset (`CURVE_EDGE` or
 * `CURVE_FACE`) and stage its points for the shared `CURVE_NODES` write.
 *
 * Iterates the locally held curves, keeping only those whose owning edge/face
 * this rank owns (@p owned, as cached by WriteGeometryMap()), so each curve is
 * written exactly once. For each owned curve it appends a descriptor row
 * `[npoints, PointsType, CURVE_NODES offset]` and accumulates the curve's
 * point coordinates into @p curvedPts with globally consistent indices. The
 * descriptor rows are written into this rank's contiguous hyperslab
 * `[rowOffset, nRowsGlobal)` of `MESH/<dsName>`, and the owning IDs into the
 * matching `MAPS/<dsName>` dataset. The actual coordinates are written later by
 * WriteCurvePoints(). Returns immediately if no curves exist globally.
 *
 * @param  curves       Local curve map (global ID -> Curve).
 * @param  dsName       Descriptor dataset name (`"CURVE_EDGE"`/`"CURVE_FACE"`).
 * @param[in,out] curvedPts  Accumulator of point coordinates and their global
 *                           `CURVE_NODES` indices, shared across calls.
 * @param[in,out] ptOffset   Running global `CURVE_NODES` row offset, advanced
 *                           by the number of points written here.
 * @param[in,out] newIdx     Running global `CURVE_NODES` index assigned to each
 *                           point.
 * @param  owned        Global IDs of edges/faces this rank owns.
 * @param  rowOffset    This rank's first descriptor row in `MESH/<dsName>`.
 * @param  nRowsGlobal  Global descriptor row count (the dataset length).
 */
void MeshGraphIOHDF5::WriteCurveMap(CurveMap &curves, std::string dsName,
                                    MeshCurvedPts &curvedPts, int &ptOffset,
                                    int &newIdx,
                                    const std::unordered_set<int> &owned,
                                    hsize_t rowOffset, hsize_t nRowsGlobal)
{
    // No curves to write (globally), return immediately
    if (nRowsGlobal == 0)
    {
        return;
    }

    std::vector<int> data, map;
    for (auto &c : curves)
    {
        // Ensure curve owner writes once
        if (!owned.count(c.first))
        {
            continue;
        }
        map.push_back(c.first);
        data.push_back((int)c.second->m_points.size());
        data.push_back(c.second->m_ptype);
        data.push_back(ptOffset); // this is the offset into CURVE_NODES
        ptOffset += (int)c.second->m_points.size();

        for (auto &pt : c.second->m_points)
        {
            MeshVertex v;

            // v.id is the global CURVE_NODES row across all ranks
            v.id = newIdx;
            pt->GetCoords(v.x, v.y, v.z);
            curvedPts.pts.push_back(v);
            curvedPts.index.push_back(newIdx++);
        }
    }

    const hsize_t nLocalRows = map.size();

    // records: nRowsGlobal x 3
    {
        std::vector<hsize_t> dims = {nRowsGlobal, 3};
        H5::DataTypeSharedPtr tp  = H5::DataType::OfObject(int{});
        H5::DataSpaceSharedPtr ds = std::make_shared<H5::DataSpace>(dims);
        H5::DataSetSharedPtr dst  = m_mesh->CreateDataSet(dsName, tp, ds);
        H5::DataSpaceSharedPtr fs = dst->GetSpace();

        if (nLocalRows)
        {
            fs->SelectRange({rowOffset, 0}, {nLocalRows, 3});
        }
        else
        {
            fs->ClearRange();
        }

        dst->Write(data, fs, m_writePL);
    }
    // id map: nRowsGlobal
    {
        std::vector<hsize_t> dims = {nRowsGlobal};
        H5::DataTypeSharedPtr tp  = H5::DataType::OfObject(int{});
        H5::DataSpaceSharedPtr ds = std::make_shared<H5::DataSpace>(dims);
        H5::DataSetSharedPtr dst  = m_maps->CreateDataSet(dsName, tp, ds);
        H5::DataSpaceSharedPtr fs = dst->GetSpace();

        if (nLocalRows)
        {
            fs->SelectRange(rowOffset, nLocalRows);
        }
        else
        {
            fs->ClearRange();
        }

        dst->Write(map, fs, m_writePL);
    }
}

/**
 * @brief Collectively write the shared `CURVE_NODES` coordinate dataset.
 *
 * Flattens the point coordinates accumulated by WriteCurveMap() (for both
 * curved edges and curved faces) into an `[nPoints, 3]` double buffer and
 * writes this rank's points into its contiguous hyperslab
 * `[ptBase, ptBase + myPts)` of the global `MESH/CURVE_NODES` dataset. The
 * offsets stored in column 2 of the `CURVE_EDGE`/`CURVE_FACE` descriptors index
 * into this dataset. Ranks with no points still join the collective write with
 * an empty selection; if there are no curve points globally the dataset is not
 * created.
 *
 * @param  curvedPts  The staged curve points for this rank.
 * @param  ptBase     This rank's first row in `CURVE_NODES`.
 * @param  totalPts   Global number of curve points (the dataset length).
 */
void MeshGraphIOHDF5::WriteCurvePoints(MeshCurvedPts &curvedPts, hsize_t ptBase,
                                       hsize_t totalPts)
{
    if (totalPts == 0)
    {
        // No points to write, return immediately.
        return;
    }

    // Create a buffer that will store all of the curved point information
    // (since this MeshCurvedPts stores non-POD representation of each point).
    const hsize_t myPts = curvedPts.pts.size();
    std::vector<double> vertData(myPts * 3);
    int cnt = 0;
    for (auto &pt : curvedPts.pts)
    {
        vertData[cnt++] = pt.x;
        vertData[cnt++] = pt.y;
        vertData[cnt++] = pt.z;
    }

    // Write out curved points.
    std::vector<hsize_t> dims = {totalPts, 3};
    H5::DataTypeSharedPtr tp  = H5::DataType::OfObject(double{});
    H5::DataSpaceSharedPtr ds = std::make_shared<H5::DataSpace>(dims);
    H5::DataSetSharedPtr dst  = m_mesh->CreateDataSet("CURVE_NODES", tp, ds);

    if (totalPts)
    {
        H5::DataSpaceSharedPtr fs = dst->GetSpace();

        // Ensure each rank participates in the write, even if they have no data
        // to write.
        if (myPts)
        {
            fs->SelectRange({ptBase, 0}, {myPts, 3});
        }
        else
        {
            fs->ClearRange();
        }
        dst->Write(vertData, fs, m_writePL);
    }
}

/**
 * @brief Write the `COMPOSITE` dataset describing every composite in the mesh.
 *
 * Composites are stored as variable-length strings, which HDF5 cannot write
 * collectively, so the data is gathered to rank 0 which performs the write.
 * Each rank packs its locally held composites into a flat int buffer of the
 * form `[nComps, {compId, tag, nIds, id0, id1, ...}, ...]`, where `tag` is the
 * shape character (`'V'`, `'S'`, `'T'`, `'F'`, ...). These buffers are
 * length-exchanged and gathered to every rank; rank 0 then merges the
 * contributions per composite ID (a single composite may be split across
 * ranks), deduplicates and sorts the entity IDs, and emits one
 * run-length-compressed string such as `" T[0-5,7] "` per composite.
 *
 * Rank 0 writes:
 *   - `MESH/COMPOSITE` : 1D variable-length string array, one per composite;
 *   - `MAPS/COMPOSITE` : 1D int array of the corresponding composite IDs.
 *
 * The merged composite tags/strings are also cached in #m_globalComps for the
 * subsequent default-expansion XML generation.
 *
 * @note Rank 0 ends up holding every composite, which is a potential
 *       scalability bottleneck; a Gatherv to root (rather than the current
 *       AllReduce-based gather) would avoid replicating the data onto every
 *       rank.
 *
 * @param composites  This rank's local composite map.
 * @param comm        Communicator over which composites are gathered.
 */
void MeshGraphIOHDF5::WriteComposites(CompositeMap &composites,
                                      LibUtilities::CommSharedPtr &comm)
{
    const int nProc = comm->GetSize(), rank = comm->GetRank();

    // In parallel we need to tell every process which geometry IDs are included
    // in each composite. Most of this code therefore communicates this
    // information between ranks. Note that rank 0 is the only process to
    // actually write data to the dataset, so it will contain information about
    // every composite in the mesh which is a potential scalability issue.

    // First pack data about each composite into an array.
    //
    // layout: [ nComps, {compId, tag, nIds, id0, id1, ...}, ... ]
    // where 'tag' is e.g. 'T' for triangles, 'F' for faces, etc.
    std::vector<int> local;
    {
        int nComps = 0;
        std::vector<int> body;
        for (auto &cIt : composites)
        {
            auto &gv = cIt.second->m_geomVec;
            if (gv.empty())
            {
                // nothing in this composite for some reason
                continue;
            }

            ++nComps;
            body.push_back(cIt.first);
            body.push_back(static_cast<int>(GetCompositeTag(gv[0])));
            body.push_back(static_cast<int>(gv.size()));
            for (Geometry *g : gv)
            {
                body.push_back(g->GetGlobalID());
            }
        }
        local.push_back(nComps);
        local.insert(local.end(), body.begin(), body.end());
    }

    // Communicate lengths of local packing array.
    Array<OneD, int> lens(nProc, 0);
    lens[rank] = static_cast<int>(local.size());
    comm->AllReduce(lens, LibUtilities::ReduceSum);

    // Compute offsets for next step of communicating data.
    Array<OneD, int> offs(nProc, 0);
    int total = 0;
    for (int p = 0; p < nProc; ++p)
    {
        offs[p] = total;
        total += lens[p];
    }

    // TODO: replace with Gatherv to root so only rank 0 holds information.
    Array<OneD, int> gathered(total, 0);
    for (int i = 0; i < static_cast<int>(local.size()); ++i)
    {
        gathered[offs[rank] + i] = local[i];
    }
    comm->AllReduce(gathered, LibUtilities::ReduceSum);

    // Now rank 0 will deduplicate composites (since >1 rank may hold a geometry
    // ID in its composite) and sort/compress.
    std::vector<std::string> comps;
    std::vector<int> c_map;

    if (rank == 0)
    {
        std::map<int, std::pair<char, std::set<unsigned int>>> merged;
        for (int p = 0; p < nProc; ++p)
        {
            if (lens[p] == 0)
            {
                continue;
            }
            int pos    = offs[p];
            int nComps = gathered[pos++];
            for (int c = 0; c < nComps; ++c)
            {
                int cid    = gathered[pos++];
                char tag   = static_cast<char>(gathered[pos++]);
                int nIds   = gathered[pos++];
                auto &slot = merged[cid];
                slot.first = tag;
                for (int k = 0; k < nIds; ++k)
                {
                    slot.second.insert((unsigned int)gathered[pos++]);
                }
            }
        }

        // Now that this is deduplicated we compress and construct the composite
        // string.
        for (auto &m : merged)
        {
            std::vector<unsigned int> ids(m.second.second.begin(),
                                          m.second.second.end());

            // construct composite string
            std::stringstream ss;
            ss << " " << m.second.first << "["
               << ParseUtils::GenerateSeqString(ids) << "] ";

            comps.push_back(ss.str());
            c_map.push_back(m.first);

            m_globalComps[m.first] = std::make_pair(m.second.first, ss.str());
        }
    }

    // We're done with collective part, rank 0 will write out the composite
    // strings.
    if (rank != 0)
    {
        return;
    }

    int nGlobal = (int)comps.size();

    H5::DataTypeSharedPtr tp  = H5::DataType::String();
    H5::DataSpaceSharedPtr ds = H5::DataSpace::OneD(nGlobal);
    H5::DataSetSharedPtr dst  = m_mesh->CreateDataSet("COMPOSITE", tp, ds);
    dst->WriteVectorString(comps, ds, tp);

    tp  = H5::DataType::OfObject(int{});
    ds  = H5::DataSpace::OneD(nGlobal);
    dst = m_maps->CreateDataSet("COMPOSITE", tp, ds);
    dst->Write(c_map, ds);
}

/**
 * @brief Write the `DOMAIN` dataset describing the composites forming each
 * domain.
 *
 * Like WriteComposites(), domains are variable-length strings written by rank 0
 * only. Each rank packs its local domains as
 * `[nDom, {domId, nComp, c0, c1, ...}, ...]`. Because a domain's composites may
 * be distributed across ranks (e.g. domain 0 on rank 0, domain 1 on rank 1),
 * the per-rank buffers are gathered with an `AllGatherv` (sizes exchanged
 * first) and rank 0 unions the composite IDs per domain ID before writing. The
 * gather is plain MPI, so this runs safely in the serial phase after the
 * collective HDF5 file has been closed and reopened by rank 0.
 *
 * Rank 0 writes:
 *   - `MESH/DOMAIN` : 1D variable-length string array, one compressed composite
 *     sequence per domain;
 *   - `MAPS/DOMAIN` : 1D int array of the corresponding domain IDs.
 *
 * @note As with WriteComposites(), `AllGatherv` replicates the (small) payload
 *       onto every rank; a root-only Gatherv would be preferable if this idiom
 *       were ever reused for element-scale data.
 *
 * @param domain  This rank's local map of domain ID -> composite map.
 * @param comm    Communicator over which domains are gathered.
 */
void MeshGraphIOHDF5::WriteDomain(std::map<int, CompositeMap> &domain,
                                  LibUtilities::CommSharedPtr &comm)
{
    // Called by ALL ranks in the serial phase: the AllReduce gather is MPI, not
    // HDF5, so it's fine after the collective close/reopen. Only rank 0 holds
    // the file handle and writes. Domains can partition across ranks (domain 0
    // on rank 0, domain 1 on rank 1), so rank 0's local map is incomplete - we
    // union per domain ID across all ranks first.

    const int nProc = comm->GetSize();
    const int rank  = comm->GetRank();

    // Pack: [ nDom, {domId, nComp, c0, c1, ...}, ... ]
    std::vector<int> local;
    {
        int nDom = 0;
        std::vector<int> body;
        for (auto &dIt : domain)
        {
            ++nDom;
            body.push_back(dIt.first);
            body.push_back(static_cast<int>(dIt.second.size()));
            for (auto &cIt : dIt.second)
            {
                body.push_back(static_cast<int>(cIt.first));
            }
        }
        local.push_back(nDom);
        local.insert(local.end(), body.begin(), body.end());
    }

    // --- size handshake: nProc ints, the only collective on metadata-sized
    //     data. Everyone needs all lengths to build the offset map. ---
    Array<OneD, int> recvSizes(nProc, 0);
    recvSizes[rank] = static_cast<int>(local.size());
    comm->AllReduce(recvSizes, LibUtilities::ReduceSum); // sizes only: tiny

    Array<OneD, int> recvOffsets(nProc, 0);
    int total = 0;
    for (int p = 0; p < nProc; ++p)
    {
        recvOffsets[p] = total;
        total += recvSizes[p];
    }

    // --- Gatherv: concatenate payloads, no zero-pad, no reduction. ---
    // NB AllGatherv replicates onto every rank. For the small composite/domain
    // payload that's fine. If a root-only Gatherv exists in the wrapper, prefer
    // it here so non-root ranks don't hold `total` — matters if this idiom is
    // ever reused for element-scale data.
    Array<OneD, int> sendData(local.size());
    for (int i = 0; i < (int)local.size(); ++i)
    {
        sendData[i] = local[i];
    }

    Array<OneD, int> gathered(total, 0);
    comm->AllGatherv(sendData, gathered, recvSizes, recvOffsets);

    if (rank != 0)
    {
        return;
    }

    // Merge: union composite IDs per domain ID (set => dedup + sort).
    std::map<int, std::set<unsigned int>> merged;
    for (int p = 0; p < nProc; ++p)
    {
        if (recvSizes[p] == 0)
        {
            continue;
        }
        int pos  = recvOffsets[p];
        int nDom = gathered[pos++];
        for (int d = 0; d < nDom; ++d)
        {
            int domId  = gathered[pos++];
            int nComp  = gathered[pos++];
            auto &slot = merged[domId];
            for (int k = 0; k < nComp; ++k)
            {
                slot.insert((unsigned int)gathered[pos++]);
            }
        }
    }

    if (merged.empty())
    {
        return;
    }

    std::vector<int> d_map;
    std::vector<std::string> doms;
    for (auto &dIt : merged)
    {
        std::vector<unsigned int> ids(dIt.second.begin(), dIt.second.end());
        doms.push_back(ParseUtils::GenerateSeqString(ids));
        d_map.push_back(dIt.first);
    }

    H5::DataTypeSharedPtr tp  = H5::DataType::String();
    H5::DataSpaceSharedPtr ds = H5::DataSpace::OneD(doms.size());
    H5::DataSetSharedPtr dst  = m_mesh->CreateDataSet("DOMAIN", tp, ds);
    dst->WriteVectorString(doms, ds, tp);

    tp  = H5::DataType::OfObject(d_map[0]);
    ds  = H5::DataSpace::OneD(d_map.size());
    dst = m_maps->CreateDataSet("DOMAIN", tp, ds);
    dst->Write(d_map, ds);
}

/**
 * @brief Write the full mesh geometry to an HDF5 file (plus a companion XML
 * stub).
 *
 * This is the top-level write entry point. The output base name yields a
 * `.nekg` HDF5 file and a `.xml` stub. The HDF5 file is created collectively
 * (with MPI-IO under parallel HDF5) and the `NEKTAR/GEOMETRY` group is built
 * with its `FORMAT_VERSION` attribute and the `MESH` and `MAPS` subgroups. The
 * write then proceeds in two phases:
 *
 *   1. *Collective phase.* The geometry maps are written with
 *      WriteGeometryMap() (vertices, segments, then 2D/3D shapes as dictated by
 *      the mesh dimension), which also records facet ownership. Owned
 *      edges/faces are tallied so that the `CURVE_EDGE`, `CURVE_FACE` and
 *      `CURVE_NODES` datasets can be laid out with per-rank contiguous offsets
 *      and written collectively by WriteCurveMap() and WriteCurvePoints().
 *
 *   2. *Serial phase.* Composites and domains are variable-length strings that
 *      cannot be written collectively, so all ranks close the file, rank 0
 *      reopens it, and WriteComposites()/WriteDomain() gather the data to rank
 *      0 for writing. (These two are still called on all ranks because the
 *      gather is collective.)
 *
 * Finally, rank 0 writes/updates the XML stub: it records the mesh/space
 * dimensions and the HDF5 file name, optionally emits a default `EXPANSIONS`
 * block (one modified C0 expansion per top-dimensional composite) when
 * @p defaultExp is set and no expansions already exist, and appends any
 * non-conformal movement information.
 *
 * @param outfilename  Output base name; the extension is replaced to form the
 *                     `.nekg` and `.xml` paths.
 * @param defaultExp   If true, generate a default expansion definition in the
 *                     XML for meshes that do not already define one.
 * @param metadata     Field metadata map (currently unused for geometry write).
 */
void MeshGraphIOHDF5::v_WriteGeometry(
    const std::string &outfilename, bool defaultExp,
    [[maybe_unused]] const LibUtilities::FieldMetaDataMap &metadata)
{
    // It is possible that we were given an empty graph without a session set;
    // in this case construct a communicator. Note we have to construct an MPI
    // communicator if we are using MPI-enabled HDF5, otherwise dataset writes
    // will not work (since it expects MPI to be initialised).
    LibUtilities::CommSharedPtr comm, commMesh;

    if (!m_meshGraph->GetSession())
    {
        LibUtilities::CppCommandLine cmd({"MeshGraphIO"});
        std::string commType = "Serial";
        if (LibUtilities::GetCommFactory().ModuleExists("ParallelMPI"))
        {
            commType = "ParallelMPI";
        }

        comm = LibUtilities::GetCommFactory().CreateInstance(
            commType, cmd.GetArgc(), cmd.GetArgv());
        commMesh = comm;
    }
    else
    {
        comm     = m_meshGraph->GetSession()->GetComm();
        commMesh = comm->GetRowComm();
    }

    const bool isRoot = comm->TreatAsRankZero();

    auto &vertSet                = m_meshGraph->GetGeomMap<PointGeom>();
    auto &segGeoms               = m_meshGraph->GetGeomMap<SegGeom>();
    auto &triGeoms               = m_meshGraph->GetGeomMap<TriGeom>();
    auto &quadGeoms              = m_meshGraph->GetGeomMap<QuadGeom>();
    auto &hexGeoms               = m_meshGraph->GetGeomMap<HexGeom>();
    auto &pyrGeoms               = m_meshGraph->GetGeomMap<PyrGeom>();
    auto &prismGeoms             = m_meshGraph->GetGeomMap<PrismGeom>();
    auto &tetGeoms               = m_meshGraph->GetGeomMap<TetGeom>();
    CurveMap &curvedEdges        = m_meshGraph->GetCurvedEdges();
    CurveMap &curvedFaces        = m_meshGraph->GetCurvedFaces();
    CompositeMap &meshComposites = m_meshGraph->GetComposites();
    auto domain                  = m_meshGraph->GetDomain();

    std::vector<std::string> tmp;
    boost::split(tmp, outfilename, boost::is_any_of("."));
    std::string filenameXml  = tmp[0] + ".xml";
    std::string filenameHdf5 = tmp[0] + ".nekg";

    ///////////////////////
    // HDF5 part (parallel)
    ///////////////////////

    LibUtilities::H5::PListSharedPtr parallelProps = H5::PList::Default();
    m_writePL                                      = H5::PList::Default();

#if defined(NEKTAR_USE_MPI) && defined(NEKTAR_HDF5_PARALLEL)
    if (commMesh->GetSize() > 1)
    {
        // Use MPI/O to access the file
        parallelProps = H5::PList::FileAccess();
        parallelProps->SetMpio(commMesh);
        // Use collective IO
        m_writePL = H5::PList::DatasetXfer();
        m_writePL->SetDxMpioCollective();
    }
#endif

#if !defined(NEKTAR_HDF5_PARALLEL)
    ASSERTL0(commMesh->GetSize() == 1,
             "Parallel HDF5 mesh output only supported when using a parallel "
             "version of HDF5");
#endif

    m_file = H5::File::Create(filenameHdf5, H5F_ACC_TRUNC, H5::PList::Default(),
                              parallelProps);
    auto hdfRoot  = m_file->CreateGroup("NEKTAR");
    auto hdfRoot2 = hdfRoot->CreateGroup("GEOMETRY");

    // Write format version.
    hdfRoot2->SetAttribute("FORMAT_VERSION", FORMAT_VERSION);

    // Create main groups.
    m_mesh = hdfRoot2->CreateGroup("MESH");
    m_maps = hdfRoot2->CreateGroup("MAPS");

    int meshDim = m_meshGraph->GetMeshDimension();

    WriteGeometryMap(vertSet, "VERT", meshDim > 1, commMesh);
    WriteGeometryMap(segGeoms, "SEG", meshDim > 1, commMesh);
    if (meshDim > 1)
    {
        WriteGeometryMap(triGeoms, "TRI", meshDim > 1, commMesh);
        WriteGeometryMap(quadGeoms, "QUAD", meshDim > 1, commMesh);
    }
    if (meshDim > 2)
    {
        WriteGeometryMap(tetGeoms, "TET", meshDim > 1, commMesh);
        WriteGeometryMap(pyrGeoms, "PYR", meshDim > 1, commMesh);
        WriteGeometryMap(prismGeoms, "PRISM", meshDim > 1, commMesh);
        WriteGeometryMap(hexGeoms, "HEX", meshDim > 1, commMesh);
    }

    // Write curve output. We'll use the stored data from WriteGeometryMap which
    // determines which rank 'owns' an edge in order to dump out curvature
    // collectively.

    // First assemble some sets of owned edges (can use directly the stored
    // information) and owned faces (need to distinguish between TRI/QUAD
    // datasets).
    const std::unordered_set<int> &ownedEdges = m_geomOwners["SEG"];
    std::unordered_set<int> ownedFaces;
    for (const std::string n : {"TRI", "QUAD"})
    {
        auto it = m_geomOwners.find(n);
        if (it != m_geomOwners.end())
        {
            ownedFaces.insert(it->second.begin(), it->second.end());
        }
    }

    // Small functor that takes our curved map and figures out how many curves
    // and points we need to store.
    auto tally = [](CurveMap &cm, const std::unordered_set<int> &owned,
                    int &nCurves, int &nPts) {
        nCurves = 0;
        nPts    = 0;
        for (auto &c : cm)
        {
            if (owned.count(c.first))
            {
                ++nCurves;
                nPts += (int)c.second->m_points.size();
            }
        }
    };

    // Compute number of curved edges, faces and points locally
    int neC, nePts, nfC, nfPts;
    tally(curvedEdges, ownedEdges, neC, nePts);
    tally(curvedFaces, ownedFaces, nfC, nfPts);

    // Total number of curve points to store.
    const int myPts = nePts + nfPts;

    // Tell every rank how many edges/faces/points we own.
    const int nProc = commMesh->GetSize();
    const int rank  = commMesh->GetRank();
    Array<OneD, int> tallies(nProc * 3, 0);
    tallies[rank * 3 + 0] = neC;
    tallies[rank * 3 + 1] = nfC;
    tallies[rank * 3 + 2] = myPts;
    commMesh->AllReduce(tallies, LibUtilities::ReduceSum);

    // Compute the offsets where we need to put data for CURVE_EDGE, CURVE_FACE
    // and CURVE_POINTS datasets.
    hsize_t edgeRowOffset = 0, edgeRowsGlobal = 0;
    hsize_t faceRowOffset = 0, faceRowsGlobal = 0;
    hsize_t ptBase = 0, totalPts = 0;
    for (int p = 0; p < nProc; ++p)
    {
        if (p < rank)
        {
            edgeRowOffset += tallies[p * 3 + 0];
        }
        if (p < rank)
        {
            faceRowOffset += tallies[p * 3 + 1];
        }
        if (p < rank)
        {
            ptBase += tallies[p * 3 + 2];
        }

        edgeRowsGlobal += tallies[p * 3 + 0];
        faceRowsGlobal += tallies[p * 3 + 1];
        totalPts += tallies[p * 3 + 2];
    }

    // Now write out the data into chunks inside the datasets.
    int ptOffset = (int)ptBase, newIdx = (int)ptBase;
    MeshCurvedPts curvePts;
    WriteCurveMap(curvedEdges, "CURVE_EDGE", curvePts, ptOffset, newIdx,
                  ownedEdges, edgeRowOffset, edgeRowsGlobal);
    WriteCurveMap(curvedFaces, "CURVE_FACE", curvePts, ptOffset, newIdx,
                  ownedFaces, faceRowOffset, faceRowsGlobal);
    WriteCurvePoints(curvePts, ptBase, totalPts);

    // At this point, what we can do collectively is finished. We need to write
    // composites and domains, but these are stored as variable-length strings
    // which can't be done collectively. So now we close the file on all
    // processors, but rank 0 will reopen it so that we can add the COMPOSITES
    // and DOMAIN datasets. Note that functions still need to be called
    // collectively so that we can assemble the required information.

    // Release our hold on the groups we opened.
    hdfRoot->Close();
    hdfRoot2->Close();

    // Note creation of empty shared_ptrs below avoids double-closing files in
    // rank != 0 when MeshGraphIOHDF5 destructor is called.
    m_writePL = LibUtilities::H5::PListSharedPtr();
    m_mesh    = LibUtilities::H5::GroupSharedPtr();
    m_maps    = LibUtilities::H5::GroupSharedPtr();
    m_file    = LibUtilities::H5::FileSharedPtr();

    // Root now reopens the file.
    if (isRoot)
    {
        m_file =
            H5::File::Open(filenameHdf5, H5F_ACC_RDWR, H5::PList::Default());

        hdfRoot = m_file->OpenGroup("NEKTAR");
        ASSERTL0(hdfRoot, "Cannot find NEKTAR group in HDF5 file.");

        hdfRoot2 = hdfRoot->OpenGroup("GEOMETRY");
        ASSERTL0(hdfRoot2, "Cannot find NEKTAR/GEOMETRY group in HDF5 file.");

        m_mesh = hdfRoot2->OpenGroup("MESH");
        ASSERTL0(m_mesh,
                 "Cannot find NEKTAR/GEOMETRY/MESH group in HDF5 file.");

        m_maps = hdfRoot2->OpenGroup("MAPS");
        ASSERTL0(m_maps,
                 "Cannot find NEKTAR/GEOMETRY/MAPS group in HDF5 file.");
    }

    // Write composites and domain. Called collectively because we'll need to
    // send composite information to rank 0.
    WriteComposites(meshComposites, commMesh);
    WriteDomain(domain, commMesh);

    //////////////////
    // XML part
    //////////////////

    if (!isRoot)
    {
        return;
    }

    // Check to see if a xml of the same name exists
    // if might have boundary conditions etc, we will just alter the
    // geometry tag if needed
    TiXmlDocument *doc = new TiXmlDocument;
    TiXmlElement *root;
    TiXmlElement *geomTag;

    if (fs::exists(filenameXml.c_str()))
    {
        std::ifstream file(filenameXml.c_str());
        file >> (*doc);
        TiXmlHandle docHandle(doc);
        root = docHandle.FirstChildElement("NEKTAR").Element();
        ASSERTL0(root, "Unable to find NEKTAR tag in file.");
        geomTag    = root->FirstChildElement("GEOMETRY");
        defaultExp = false;
    }
    else
    {
        TiXmlDeclaration *decl = new TiXmlDeclaration("1.0", "utf-8", "");
        doc->LinkEndChild(decl);
        root = new TiXmlElement("NEKTAR");
        doc->LinkEndChild(root);

        geomTag = new TiXmlElement("GEOMETRY");
        root->LinkEndChild(geomTag);
    }

    // Update attributes with dimensions.
    geomTag->SetAttribute("DIM", m_meshGraph->GetMeshDimension());
    geomTag->SetAttribute("SPACE", m_meshGraph->GetSpaceDimension());
    geomTag->SetAttribute("HDF5FILE", filenameHdf5);

    geomTag->Clear();

    std::map<char, int> tagToShapeDim = {{'V', 0}, {'S', 1}, {'Q', 2},
                                         {'T', 2}, {'F', 2}, {'A', 3},
                                         {'R', 3}, {'P', 3}, {'H', 3}};

    if (defaultExp)
    {
        TiXmlElement *expTag = new TiXmlElement("EXPANSIONS");

        for (auto &cIt : m_globalComps)
        {
            if (tagToShapeDim[cIt.second.first] ==
                m_meshGraph->GetMeshDimension())
            {
                TiXmlElement *exp = new TiXmlElement("E");
                exp->SetAttribute("COMPOSITE",
                                  "C[" + std::to_string(cIt.first) + "]");
                exp->SetAttribute("NUMMODES", 4);
                exp->SetAttribute("TYPE", "MODIFIED");
                exp->SetAttribute("FIELDS", "u");

                expTag->LinkEndChild(exp);
            }
        }
        root->LinkEndChild(expTag);
    }

    auto movement = m_meshGraph->GetMovement();
    if (movement)
    {
        movement->WriteMovement(root);
    }

    doc->SaveFile(filenameXml);
}

} // namespace Nektar::SpatialDomains
