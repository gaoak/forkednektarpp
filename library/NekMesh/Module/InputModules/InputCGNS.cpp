////////////////////////////////////////////////////////////////////////////////
//
//  File: InputCGNS.cpp
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
//  Description: CGNS converter.
//
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "InputCGNS.h"
#include <NekMesh/MeshElements/Element.h>
#include <SpatialDomains/CompositeNames.h>
#include <SpatialDomains/MeshGraphIO.h>

using namespace std;

namespace Nektar::NekMesh
{

using namespace Nektar::NekMesh;
using SpatialDomains::SanitiseCompositeName;

ModuleKey InputCGNS::className = GetModuleFactory().RegisterCreatorFunction(
    ModuleKey(eInputModule, "cgns"), InputCGNS::create, "Reads CGNS file.");
/**
 * @brief Set up InputCGNS object.
 */
InputCGNS::InputCGNS(MeshSharedPtr m) : InputModule(m)
{
    m_config["processall"] =
        ConfigOption(true, "0", "Process edges, faces as well as composites");
    m_config["nobcgroup"] = ConfigOption(
        true, "0",
        "Ignore the file's boundary conditions when forming boundary "
        "composites; group and name them by element section instead");
    m_config["prismlines"] = ConfigOption(
        false, "",
        "Write the prisms and the lines running through them to this file, as "
        "a legacy VTK unstructured grid, for looking at where a mesh cannot "
        "be numbered");
}

InputCGNS::~InputCGNS()
{
}

/**
 * @brief Add node to m_node and m_vertexSet
 */
void InputCGNS::SaveNode(int id, NekDouble x, NekDouble y, NekDouble z)
{
    m_loadedNodes[id] =
        ObjPoolManager<SpatialDomains::PointGeom>::AllocateUniquePtr(3, id, x,
                                                                     y, z);
    m_node.push_back(m_loadedNodes[id].get());
}

/**
 * CGNS file can be read using the CGNS Mid-Level Library
 * https://cgns.github.io/CGNS_docs_current/midlevel/index.html
 * Process() converts this information into .xml format for use in Nektar++
 */
void InputCGNS::Process()
{
    SetupElements();

    ProcessElements();
    ProcessComposites();

    PrintSummary();
}

void InputCGNS::SetupElements()
{
    string fname = m_config["infile"].as<string>();

    // 9 is the max number required (for a 3D structured mesh): [NVertexI,
    // NVertexJ, NVertexK, NCellI, NCellJ, NCellK, NBoundVertexI, NBoundVertexJ,
    // NBoundVertexK]).
    // https://cgns.github.io/CGNS_docs_current/midlevel/structural.html#zone
    cgsize_t sizes[9];

    // 33 is the max, set by convention
    char zoneName[33];

    if (cg_open(fname.c_str(), CG_MODE_READ, &m_fileIndex))
    {
        m_log(FATAL) << "Error opening file: '" << fname << "'" << endl;
    }
    else
    {
        m_log(VERBOSE) << "Reading CGNS file: '" << fname << "'" << endl;
    }

    char baseName[33];
    int expDim, spaceDim;

    // Read the base information
    if (cg_base_read(m_fileIndex, m_baseIndex, baseName, &expDim, &spaceDim))
    {
        m_log(FATAL) << "Error reading base information" << endl;
        cg_close(m_fileIndex);
    }
    else
    {
        m_mesh->m_meshGraph->SetMeshDimension(expDim);
        m_mesh->m_meshGraph->SetSpaceDimension(spaceDim);
        m_log(VERBOSE) << "Read base information" << endl;
        m_log(VERBOSE) << "Base name    : " << baseName << endl;
        m_log(VERBOSE) << "Element dim  : " << expDim << endl;
        m_log(VERBOSE) << "Physical dim : " << spaceDim << endl;
    }

    int nZones = 0;
    if (cg_nzones(m_fileIndex, m_baseIndex, &nZones) || nZones < 1)
    {
        m_log(FATAL) << "Error reading the number of zones." << endl;
        cg_close(m_fileIndex);
    }

    m_log(VERBOSE) << "Number of zones: " << nZones << endl;

    // Every zone is read into one mesh. A zone is a part of the model rather
    // than a model of its own; a mesher splits a volume from the boundary layer
    // grown on it this way, so the zones abut, and the faces they share are
    // stitched below rather than left as a boundary.
    int nCoords = 0, n3DElems = 0;

    // Each pair in elemInfo contains the element type and the element's node
    // indices
    vector<pair<ElementType_t, vector<int>>> elemInfo;
    vector<CGNSSection> sections;

    for (m_zoneIndex = 1; m_zoneIndex <= nZones; ++m_zoneIndex)
    {
        ZoneType_t zoneType;

        // Read the zone type
        if (cg_zone_type(m_fileIndex, m_baseIndex, m_zoneIndex, &zoneType))
        {
            cg_close(m_fileIndex);
            m_log(FATAL) << "Error reading zone type for zone " << m_zoneIndex
                         << endl;
        }

        if (zoneType != CGNS_ENUMV(Unstructured))
        {
            m_log(FATAL) << "Only unstructured meshes supported" << endl;
        }

        // Read the zone information
        if (cg_zone_read(m_fileIndex, m_baseIndex, m_zoneIndex, zoneName,
                         sizes))
        {
            cg_close(m_fileIndex);
            m_log(FATAL) << "Error reading zone information for zone "
                         << m_zoneIndex << endl;
        }

        // Nodes of this zone are appended to those already read, so a node
        // index from here on is into the whole file rather than into a zone.
        int nodeOffset = nCoords;
        int zoneCoords = sizes[0];

        m_zoneNames[m_zoneIndex] = SanitiseCompositeName(zoneName);

        m_log(VERBOSE) << "Read zone information" << endl;
        m_log(VERBOSE) << "Zone name   : " << zoneName << endl;
        m_log(VERBOSE) << "Grid Points : " << zoneCoords << endl;
        m_log(VERBOSE) << "No of elmts : " << sizes[1] << endl;

        nCoords += zoneCoords;
        n3DElems += sizes[1];

        // Read in coordinates from the mesh
        m_x.resize(zoneCoords);
        m_y.resize(zoneCoords);
        m_z.resize(zoneCoords);

        cgsize_t rmin[3] = {1, 1, 1};
        cgsize_t rmax[3] = {zoneCoords, 1, 1}; // NOTE: for unstructured meshes

        if (cg_coord_read(m_fileIndex, m_baseIndex, m_zoneIndex, "CoordinateX",
                          RealDouble, rmin, rmax, m_x.data()) ||
            cg_coord_read(m_fileIndex, m_baseIndex, m_zoneIndex, "CoordinateY",
                          RealDouble, rmin, rmax, m_y.data()) ||
            cg_coord_read(m_fileIndex, m_baseIndex, m_zoneIndex, "CoordinateZ",
                          RealDouble, rmin, rmax, m_z.data()))
        {
            cg_close(m_fileIndex);
            m_log(FATAL) << "Error reading coordinates" << endl;
        }

        m_log(VERBOSE) << "Read " << zoneCoords << " coordinates" << endl;

        // Create a node for each coord
        for (int i = 0; i < zoneCoords; i++)
        {
            SaveNode(nodeOffset + i, m_x[i], m_y[i], m_z[i]);
        }

        // Read number of sections
        int nSections;
        if (cg_nsections(m_fileIndex, m_baseIndex, m_zoneIndex, &nSections))
        {
            cg_close(m_fileIndex);
            m_log(FATAL) << "Error reading number of sections." << endl;
        }

        m_log(VERBOSE) << "Number of sections: " << nSections << endl;

        // Get the element vertices for each section
        for (int sectionInd = 1; sectionInd <= nSections; ++sectionInd)
        {
            char sectionName[33];
            ElementType_t elemType;
            cgsize_t start, end;
            int nbndry, parentFlag;

            if (cg_section_read(m_fileIndex, m_baseIndex, m_zoneIndex,
                                sectionInd, sectionName, &elemType, &start,
                                &end, &nbndry, &parentFlag))
            {
                cg_close(m_fileIndex);
                m_log(FATAL)
                    << "Error reading section information." << std::endl;
            }

            int elemSize = end - start + 1;
            m_log(VERBOSE) << "Reading section data: " << endl;
            m_log(VERBOSE) << "        section name: " << sectionName << endl;
            m_log(VERBOSE) << "        section type: "
                           << ElementTypeName[elemType] << endl;
            m_log(VERBOSE) << "        section count: " << elemSize << endl;

            cgsize_t elemDataSize;

            cg_ElementDataSize(m_fileIndex, m_baseIndex, m_zoneIndex,
                               sectionInd, &elemDataSize);

            // Record the section so that boundary composites can be formed from
            // it later. Both ranges are needed: boundary conditions name
            // elements by the file's own numbering, which restarts in each
            // zone, and everything after this works in positions within
            // elemInfo, which does not.
            CGNSSection sec;
            sec.zone       = m_zoneIndex;
            sec.name       = SanitiseCompositeName(sectionName);
            sec.family     = ReadSectionFamilyName(sectionInd);
            sec.cgStart    = start;
            sec.cgEnd      = end;
            sec.infoBegin  = (int)elemInfo.size();
            sec.isBoundary = false;

            if (elemType == CGNS_ENUMV(ElementTypeNull) ||
                elemType == CGNS_ENUMV(ElementTypeUserDefined) ||
                elemType == CGNS_ENUMV(NGON_n) ||
                elemType == CGNS_ENUMV(NFACE_n))
            {
                m_log(FATAL) << "Section type " << ElementTypeName[elemType]
                             << " not supported" << endl;
            }

            if (elemType == CGNS_ENUMV(MIXED))
            {
                ExtractMixedElemInfo(elemDataSize, elemSize, sectionInd,
                                     nodeOffset, elemInfo);
            }
            else
            {
                ExtractSeparatedElemInfo(elemDataSize, elemSize, elemType,
                                         sectionInd, nodeOffset, elemInfo);
            }

            // Whether a section holds boundary faces is decided per element
            // rather than from the section's own type, so that a MIXED section
            // holding faces is recognised as well as one declared TRI_n or
            // QUAD_n.
            sec.infoEnd = (int)elemInfo.size();
            for (int i = sec.infoBegin; i < sec.infoEnd; ++i)
            {
                if (IsBoundaryFace(elemInfo[i].first))
                {
                    sec.isBoundary = true;
                    break;
                }
            }

            sections.push_back(sec);
        }
    }

    // We now read connectivities, which say which faces the zones share and
    // which close a periodic pair. We read both before the faces are built: the
    // shared ones are stitched rather than kept, and a periodic side needs its
    // own composite.
    std::map<std::string, std::vector<int>> connGroups;
    std::unordered_set<int> interfaceFaces;
    std::vector<std::pair<int, int>> interfacePairs;
    ReadConnectivities(nZones, sections, elemInfo, connGroups, interfaceFaces,
                       interfacePairs);

    // For more than one zone, we use connectivity to eliminate the duplicate
    // vertices across the interface.
    if (nZones > 1)
    {
        MergeZoneVertices(elemInfo, interfacePairs, nCoords);
    }

    // create FaceNodes, ElementFaces and BndElemenFaces from elemInfo
    std::unordered_map<int, vector<int>> FaceNodes;
    Array<OneD, vector<int>> ElementFaces(n3DElems);
    vector<int> BoundaryElems;
    vector<int> VolumeElems;
    ReadFaces(elemInfo, FaceNodes, ElementFaces, BoundaryElems, VolumeElems);

    // Decide how the boundary faces are grouped into composites, and what
    // each group is called. This has to happen while the file is still open.
    vector<CGNSBoundaryGroup> bndGroups = BuildBoundaryGroups(
        sections, elemInfo, nZones, connGroups, interfaceFaces);

    // We've now read all the information we require, so close the CGNS file:
    // the next steps will reorder elemental nodes to ensure C^0 connectivity
    // with the modified basis functions as required in Nektar++.
    if (cg_close(m_fileIndex))
    {
        m_log(FATAL) << "Error closing CGNS file." << std::endl;
    }
    else
    {
        m_log(VERBOSE) << "Closing CGNS file." << std::endl;
    }

    // Reset node ordering so that all prism faces have consistent numbering for
    // singular vertex re-ordering. A highly non-trivial function!
    ResetNodes(m_node, ElementFaces, FaceNodes, elemInfo, VolumeElems);

    int nComposite = 0, i = 0;
    int nelements = ElementFaces.size();
    int cnt;

    // Move to element creation: starting with prisms.
    cnt = 0;
    for (i = 0; i < nelements; ++i)
    {
        Array<OneD, int> Nodes =
            SortFaceNodes(m_node, ElementFaces[i], FaceNodes);
        if (ElementFaces[i].size() == 5 && Nodes.size() == 6) // is a prism
        {
            GenElement3D(m_node, elemInfo[VolumeElems[i]].second,
                         ElementFaces[i], FaceNodes,
                         elemInfo[VolumeElems[i]].first, nComposite, true);
            ++cnt;
        }
    }
    m_log(VERBOSE) << "  - # of prisms: " << cnt << endl;

    if (cnt)
    {
        nComposite++;
    }

    // create Pyras second
    cnt = 0;
    for (i = 0; i < nelements; ++i)
    {
        Array<OneD, int> Nodes =
            SortFaceNodes(m_node, ElementFaces[i], FaceNodes);
        if (ElementFaces[i].size() == 5 && Nodes.size() == 5) // is a pyra
        {
            GenElement3D(m_node, elemInfo[VolumeElems[i]].second,
                         ElementFaces[i], FaceNodes,
                         elemInfo[VolumeElems[i]].first, nComposite, true);
            ++cnt;
        }
    }
    m_log(VERBOSE) << "  - # of pyramids: " << cnt << endl;

    if (cnt)
    {
        nComposite++;
    }

    // create Tets third
    cnt = 0;
    for (i = 0; i < nelements; ++i)
    {
        if (ElementFaces[i].size() == 4) // is a tetra
        {
            GenElement3D(m_node, elemInfo[VolumeElems[i]].second,
                         ElementFaces[i], FaceNodes,
                         elemInfo[VolumeElems[i]].first, nComposite, true);
            ++cnt;
        }
    }
    m_log(VERBOSE) << "  - # of tetrahedra: " << cnt << endl;

    if (cnt)
    {
        nComposite++;
    }

    // create Hexes fourth
    cnt = 0;
    for (i = 0; i < nelements; ++i)
    {
        if (ElementFaces[i].size() == 6) // is a hexa
        {
            GenElement3D(m_node, elemInfo[VolumeElems[i]].second,
                         ElementFaces[i], FaceNodes,
                         elemInfo[VolumeElems[i]].first, nComposite, true);
            ++cnt;
        }
    }
    m_log(VERBOSE) << "  - # of hexa: " << cnt << endl;
    if (cnt)
    {
        nComposite++;
    }

    // create Boundary elements last
    auto &compositesLabels = m_mesh->m_meshGraph->GetCompositesLabels();

    for (auto &grp : bndGroups)
    {
        cnt = 0;
        for (int i : grp.elemIdx)
        {
            GenElement2D(m_node, elemInfo[i].second, elemInfo[i].first,
                         nComposite, true);
            ++cnt;
        }

        if (cnt)
        {
            m_log(VERBOSE) << "  - # of bndry elmts in " + grp.name + " :"
                           << cnt << endl;
            if (!grp.name.empty())
            {
                compositesLabels[nComposite] = grp.name;
            }
            nComposite++;
        }
    }

    // We need to give MeshGraph the ownership of the vertices. We also renumber
    // them.
    std::vector<std::pair<int, SpatialDomains::PointGeomUniquePtr>> movedVerts;
    movedVerts.reserve(m_vertIDs.size());
    int contigVertID = 0;
    for (int node : m_vertIDs)
    {
        auto vert = std::move(m_loadedNodes[node]);
        vert->SetGlobalID(contigVertID);
        movedVerts.emplace_back(contigVertID, std::move(vert));
        contigVertID++;
    }
    m_mesh->m_meshGraph->BulkAddGeom<SpatialDomains::PointGeom>(movedVerts);

    // We need to give MeshGraph the ownership of the high-order nodes. Give
    // them -1 ID.
    for (int node : m_curveNodeIDs)
    {
        if (m_vertIDs.find(node) == m_vertIDs.end())
        {
            m_mesh->m_meshGraph->GetAllCurveNodes().push_back(
                std::move(m_loadedNodes[node]));
        }
    }
    for (auto &node : m_mesh->m_meshGraph->GetAllCurveNodes())
    {
        node->SetGlobalID(-1);
    }
}

void InputCGNS::ExtractSeparatedElemInfo(
    cgsize_t elemDataSize, int elemSize, ElementType_t elemType, int sectionInd,
    int nodeOffset, vector<pair<ElementType_t, vector<int>>> &elemInfo)
{
    cgsize_t *ielem = new cgsize_t[elemDataSize];
    cgsize_t iparentdata;

    int nElemNodes;

    if (cg_elements_read(m_fileIndex, m_baseIndex, m_zoneIndex, sectionInd,
                         ielem, &iparentdata))
    {
        m_log(FATAL) << "Error reading element type " << elemType << endl;
    }
    else
    {
        m_log(VERBOSE) << "Read element type "
                       << ElementTypeName[m_ind2ElemType[elemType]] << endl;
    }

    // Get the number of nodes
    cg_npe(elemType, &nElemNodes);

    for (uint sectionElemIndex = 0; sectionElemIndex < elemSize;
         sectionElemIndex++)
    {
        elemInfo.push_back(pair<ElementType_t, vector<int>>(elemType, {}));

        for (uint vertIndex = sectionElemIndex * nElemNodes;
             vertIndex < (sectionElemIndex + 1) * nElemNodes; vertIndex++)
        {
            // -1 is to convert between index conventions (CGNS starts at 1);
            // the offset carries a zone-local index into the whole file.
            elemInfo[elemInfo.size() - 1].second.push_back(
                nodeOffset + ielem[vertIndex] - 1);
        }
    }

    delete[] ielem;
}

void InputCGNS::ExtractMixedElemInfo(
    cgsize_t elemDataSize, int elemSize, int sectionInd, int nodeOffset,
    vector<pair<ElementType_t, vector<int>>> &elemInfo)
{

    if (elemSize == 0)
    {
        return;
    }

    cgsize_t *ielem      = new cgsize_t[elemDataSize];
    cgsize_t *offsetData = new cgsize_t[elemSize + 1];
    cgsize_t iparentdata;

    // The data array ElementStartOffset contains the starting positions of each
    // element in the ElementConnectivity data array and its last value
    // corresponds to the ElementConnectivity total size. See:
    // https://cgns.github.io/CGNS_docs_current/sids/gridflow.html#Elements
    if (cg_poly_elements_read(m_fileIndex, m_baseIndex, m_zoneIndex, sectionInd,
                              ielem, offsetData, &iparentdata))
    {
        m_log(FATAL) << "Error reading element type MIXED" << endl;
    }
    else
    {
        m_log(VERBOSE) << "Read element type MIXED" << endl;
    }

    ElementType_t elemType;

    for (uint sectionElemIndex = 0; sectionElemIndex < elemSize;
         sectionElemIndex++)
    {
        elemType = m_ind2ElemType[ielem[offsetData[sectionElemIndex]]];

        elemInfo.push_back(pair<ElementType_t, vector<int>>(elemType, {}));

        for (uint vertIndex = offsetData[sectionElemIndex] + 1;
             vertIndex < offsetData[sectionElemIndex + 1]; vertIndex++)
        {
            elemInfo[elemInfo.size() - 1].second.push_back(
                nodeOffset + ielem[vertIndex] - 1);
        }
    }

    delete[] ielem;
    delete[] offsetData;
}

/**
 * @brief Read the FamilyName_t attached to one Elements_t section, if any.
 *
 * Returns an empty string when the section names no family, which is the normal
 * case: exporters put the family on the boundary condition instead, and some
 * CGNS builds refuse to write one here at all.
 */
std::string InputCGNS::ReadSectionFamilyName(int sectionInd)
{
    if (cg_goto(m_fileIndex, m_baseIndex, "Zone_t", m_zoneIndex, "Elements_t",
                sectionInd, "end"))
    {
        return "";
    }

    char famName[33];
    if (cg_famname_read(famName))
    {
        return "";
    }

    return SanitiseCompositeName(famName);
}

/**
 * @brief Read the FamilyName_t attached to one BC_t node, if any.
 *
 * As ReadSectionFamilyName(), but for a boundary condition. This is the usual
 * place a part ID ends up, since several boundary conditions naming the same
 * family are how a mesher says "these faces are all one patch".
 */
std::string InputCGNS::ReadBCFamilyName(int bocoInd)
{
    if (cg_goto(m_fileIndex, m_baseIndex, "Zone_t", m_zoneIndex, "ZoneBC_t", 1,
                "BC_t", bocoInd, "end"))
    {
        return "";
    }

    char famName[33];
    if (cg_famname_read(famName))
    {
        return "";
    }

    return SanitiseCompositeName(famName);
}

/**
 * @brief True if an element of this type is a face of the boundary.
 */
bool InputCGNS::IsBoundaryFace(ElementType_t elemType)
{
    LibUtilities::ShapeType st = m_elemType2ShapeType[elemType];
    return st == LibUtilities::eTriangle || st == LibUtilities::eQuadrilateral;
}

/**
 * @brief Build the tables that turn one zone's CGNS point set into faces.
 */
CGNSFaceLookup InputCGNS::BuildFaceLookup(
    const std::vector<CGNSSection> &sections,
    const std::vector<std::pair<ElementType_t, std::vector<int>>> &elemInfo,
    int zone)
{
    CGNSFaceLookup lookup;

    // Only this zone's sections: element numbering restarts in each, so a
    // number means nothing without knowing which zone named it.
    for (const auto &sec : sections)
    {
        if (sec.zone == zone)
        {
            lookup.byRange.push_back(&sec);
        }
    }

    std::sort(lookup.byRange.begin(), lookup.byRange.end(),
              [](const CGNSSection *a, const CGNSSection *b) {
                  return a->cgStart < b->cgStart;
              });

    for (size_t i = 0; i < elemInfo.size(); ++i)
    {
        if (IsBoundaryFace(elemInfo[i].first))
        {
            lookup.bndIdx.push_back((int)i);
        }
    }

    return lookup;
}

/**
 * @brief Turn one CGNS point set into the boundary faces it names.
 *
 * A point set addresses its faces in one of two ways, and both a boundary
 * condition and a grid connectivity may use either, so this is shared.
 *
 * Where it names elements -- an ElementRange or ElementList, or a point set
 * sitting at FaceCenter rather than Vertex -- the numbers are the file's own
 * element numbering, which translates straight into positions in @p elemInfo.
 *
 * Where it names vertices, which is the CGNS default, there is nothing to
 * translate: a face belongs to the set when every one of its corners is
 * listed. That test can in principle also catch a face of a neighbouring patch
 * whose corners all happen to lie on the shared edge, so a vertex-addressed
 * set is the less trustworthy of the two.
 *
 * @param what  How to describe the owner of this set in any warning.
 */
std::vector<int> InputCGNS::ResolveFaceSet(
    const CGNSFaceLookup &lookup, PointSetType_t ptsetType,
    GridLocation_t location, const std::vector<cgsize_t> &pnts,
    const std::vector<std::pair<ElementType_t, std::vector<int>>> &elemInfo,
    const std::string &what)
{
    std::vector<int> idx;

    if (pnts.empty())
    {
        return idx;
    }

    bool isRange = ptsetType == CGNS_ENUMV(ElementRange) ||
                   ptsetType == CGNS_ENUMV(PointRange);

    if (isRange && pnts.size() != 2)
    {
        m_log(WARNING) << what << " declares a range but gives " << pnts.size()
                       << " values; ignoring it." << endl;
        return idx;
    }

    bool elementBased = ptsetType == CGNS_ENUMV(ElementRange) ||
                        ptsetType == CGNS_ENUMV(ElementList) ||
                        location != CGNS_ENUMV(Vertex);

    if (elementBased)
    {
        auto toInfoIdx = [&](cgsize_t cgId) -> int {
            auto it =
                std::upper_bound(lookup.byRange.begin(), lookup.byRange.end(),
                                 cgId, [](cgsize_t v, const CGNSSection *sec) {
                                     return v < sec->cgStart;
                                 });

            if (it == lookup.byRange.begin())
            {
                return -1;
            }

            const CGNSSection *sec = *(--it);

            return cgId > sec->cgEnd
                       ? -1
                       : sec->infoBegin + (int)(cgId - sec->cgStart);
        };

        auto add = [&](cgsize_t cgId) {
            int i = toInfoIdx(cgId);
            if (i >= 0 && IsBoundaryFace(elemInfo[i].first))
            {
                idx.push_back(i);
            }
        };

        if (isRange)
        {
            for (cgsize_t e = pnts[0]; e <= pnts[1]; ++e)
            {
                add(e);
            }
        }
        else
        {
            for (cgsize_t e : pnts)
            {
                add(e);
            }
        }

        return idx;
    }

    // elemInfo holds node indices already shifted to start at zero, so the
    // set's own one-based numbering is shifted here to match rather than at
    // every comparison.
    std::unordered_set<cgsize_t> verts;

    if (isRange)
    {
        for (cgsize_t v = pnts[0]; v <= pnts[1]; ++v)
        {
            verts.insert(v - 1);
        }
    }
    else
    {
        for (cgsize_t v : pnts)
        {
            verts.insert(v - 1);
        }
    }

    for (int i : lookup.bndIdx)
    {
        bool isTri =
            m_elemType2ShapeType[elemInfo[i].first] == LibUtilities::eTriangle;
        int nCorners = isTri ? 3 : 4;

        // CGNS lists an element's corners before its higher-order nodes,
        // which a vertex-addressed set need not mention.
        bool allOn = true;
        for (int c = 0; c < nCorners && allOn; ++c)
        {
            allOn = verts.count(elemInfo[i].second[c]) > 0;
        }

        if (allOn)
        {
            idx.push_back(i);
        }
    }

    return idx;
}

/**
 * @brief Group boundary faces by the ZoneBC_t boundary conditions in the file.
 *
 * Fills @p groups with the faces each boundary condition names, keyed by the
 * name the composite will carry, and records every face so assigned in
 * @p claimed so that BuildBoundaryGroups() does not also hand it to its
 * section.
 */
void InputCGNS::ReadBoundaryConditionGroups(
    const CGNSFaceLookup &lookup,
    const std::vector<std::pair<ElementType_t, std::vector<int>>> &elemInfo,
    std::map<std::string, std::vector<int>> &groups,
    std::unordered_set<int> &claimed)
{
    int nBocos = 0;
    if (cg_nbocos(m_fileIndex, m_baseIndex, m_zoneIndex, &nBocos) ||
        nBocos == 0)
    {
        return;
    }

    m_log(VERBOSE) << "Number of boundary conditions: " << nBocos << endl;

    for (int bcInd = 1; bcInd <= nBocos; ++bcInd)
    {
        char bcName[33];
        BCType_t bcType;
        PointSetType_t ptsetType;
        cgsize_t npnts = 0, normalListSize;
        int normalIndex[3], ndataset;
        DataType_t normalDataType;

        if (cg_boco_info(m_fileIndex, m_baseIndex, m_zoneIndex, bcInd, bcName,
                         &bcType, &ptsetType, &npnts, normalIndex,
                         &normalListSize, &normalDataType, &ndataset))
        {
            m_log(WARNING) << "Could not read boundary condition " << bcInd
                           << "; its faces will be grouped by section instead."
                           << endl;
            continue;
        }

        if (npnts <= 0)
        {
            continue;
        }

        // Vertex is the CGNS default, and is what we get back if the
        // condition does not say otherwise.
        GridLocation_t location = CGNS_ENUMV(Vertex);
        cg_boco_gridlocation_read(m_fileIndex, m_baseIndex, m_zoneIndex, bcInd,
                                  &location);

        std::vector<cgsize_t> pnts(npnts);
        if (cg_boco_read(m_fileIndex, m_baseIndex, m_zoneIndex, bcInd,
                         pnts.data(), nullptr))
        {
            m_log(WARNING) << "Could not read the point set of boundary "
                           << "condition '" << SanitiseCompositeName(bcName)
                           << "'; its faces will be grouped by section instead."
                           << endl;
            continue;
        }

        std::string name = ReadBCFamilyName(bcInd);
        if (name.empty())
        {
            name = SanitiseCompositeName(bcName);
        }

        std::vector<int> all =
            ResolveFaceSet(lookup, ptsetType, location, pnts, elemInfo,
                           "Boundary condition '" + name + "'");

        // Faces a connectivity has already spoken for are left where they
        // are, since a periodic side is a more specific statement than a
        // condition covering the patch it sits in.
        std::vector<int> idx;
        for (int i : all)
        {
            if (!claimed.count(i))
            {
                idx.push_back(i);
            }
        }

        if (all.empty())
        {
            m_log(WARNING) << "Boundary condition '" << name
                           << "' matched no boundary faces; its faces, if any, "
                           << "will be grouped by section instead." << endl;
            continue;
        }

        if (idx.empty())
        {
            m_log(VERBOSE) << "  - boundary condition '" << name << "': all "
                           << all.size() << " faces already named by a grid "
                           << "connectivity" << endl;
            continue;
        }

        m_log(VERBOSE) << "  - boundary condition '" << name
                       << "': " << idx.size() << " faces" << endl;

        claimed.insert(idx.begin(), idx.end());
        groups[name].insert(groups[name].end(), idx.begin(), idx.end());
    }
}

/**
 * @brief Read the grid connectivities of every zone.
 *
 * A connectivity names a set of faces and says what lies on the other side of
 * them.
 *
 * Where the other side is the same zone, it closes a periodic pair. That is
 * the one thing the sections cannot express: a mesher writes both sides of a
 * periodic boundary into a single section, and a composite holding both is of
 * no use for setting the condition up, since a boundary cannot be made
 * periodic with itself. So each side is collected into @p periodicGroups under
 * the connectivity's name, and the pair can then be written in a session.
 *
 * Where the other side is a different zone, the faces are an interior surface
 * of the assembled mesh rather than a boundary of it. They are collected into
 * @p interfaceFaces so that no composite is made of them, and the face pairs
 * are collected into @p interfacePairs so that the vertex merge which stitches
 * the zones can be checked against what the file said.
 */
void InputCGNS::ReadConnectivities(
    int nZones, const std::vector<CGNSSection> &sections,
    const std::vector<std::pair<ElementType_t, std::vector<int>>> &elemInfo,
    std::map<std::string, std::vector<int>> &periodicGroups,
    std::unordered_set<int> &interfaceFaces,
    std::vector<std::pair<int, int>> &interfacePairs)
{
    std::map<int, CGNSFaceLookup> lookups;
    for (int zone = 1; zone <= nZones; ++zone)
    {
        lookups[zone] = BuildFaceLookup(sections, elemInfo, zone);
    }

    for (int zone = 1; zone <= nZones; ++zone)
    {
        int nConns = 0;
        if (cg_nconns(m_fileIndex, m_baseIndex, zone, &nConns) || nConns == 0)
        {
            continue;
        }

        m_log(VERBOSE) << "Zone '" << m_zoneNames[zone] << "': " << nConns
                       << " connectivities" << endl;

        for (int connInd = 1; connInd <= nConns; ++connInd)
        {
            char connName[33], donorName[33];
            GridLocation_t location;
            GridConnectivityType_t connType;
            PointSetType_t ptsetType, donorPtsetType;
            cgsize_t npnts = 0, ndataDonor;
            ZoneType_t donorZoneType;
            DataType_t donorDataType;

            if (cg_conn_info(m_fileIndex, m_baseIndex, zone, connInd, connName,
                             &location, &connType, &ptsetType, &npnts,
                             donorName, &donorZoneType, &donorPtsetType,
                             &donorDataType, &ndataDonor))
            {
                m_log(WARNING)
                    << "Could not read connectivity " << connInd << " of zone '"
                    << m_zoneNames[zone] << "'; ignoring it." << endl;
                continue;
            }

            std::string name  = SanitiseCompositeName(connName);
            std::string donor = SanitiseCompositeName(donorName);

            if (npnts <= 0)
            {
                continue;
            }

            std::vector<cgsize_t> pnts(npnts);
            std::vector<cgsize_t> donorPnts(ndataDonor > 0 ? ndataDonor : 1);

            if (cg_conn_read(m_fileIndex, m_baseIndex, zone, connInd,
                             pnts.data(), CGNS_ENUMV(LongInteger),
                             donorPnts.data()))
            {
                m_log(WARNING)
                    << "Could not read the point set of "
                    << "connectivity '" << name << "'; ignoring it." << endl;
                continue;
            }

            std::vector<int> idx =
                ResolveFaceSet(lookups[zone], ptsetType, location, pnts,
                               elemInfo, "Connectivity '" + name + "'");

            if (donor == m_zoneNames[zone])
            {
                // A periodic pair says how it is transformed; a connectivity
                // onto its own zone that does not is nothing this can make a
                // composite of.
                float rotCentre[3], rotAngle[3], translation[3];
                if (cg_conn_periodic_read(m_fileIndex, m_baseIndex, zone,
                                          connInd, rotCentre, rotAngle,
                                          translation))
                {
                    m_log(VERBOSE) << "  - connectivity '" << name
                                   << "' is not periodic; leaving its faces "
                                   << "to their section" << endl;
                    continue;
                }

                if (idx.empty())
                {
                    m_log(WARNING) << "Connectivity '" << name
                                   << "' matched no boundary faces; its faces, "
                                   << "if any, will be grouped by section "
                                   << "instead." << endl;
                    continue;
                }

                m_log(VERBOSE) << "  - periodic connectivity '" << name
                               << "': " << idx.size() << " faces, translation ("
                               << translation[0] << ", " << translation[1]
                               << ", " << translation[2] << ")" << endl;

                auto &grp = periodicGroups[name];
                grp.insert(grp.end(), idx.begin(), idx.end());
                continue;
            }

            // From here on the connectivity abuts another zone.
            auto donorZone =
                std::find_if(m_zoneNames.begin(), m_zoneNames.end(),
                             [&](const std::pair<const int, std::string> &z) {
                                 return z.second == donor;
                             });

            if (donorZone == m_zoneNames.end())
            {
                m_log(WARNING) << "Connectivity '" << name << "' names the "
                               << "donor zone '" << donor << "', which is not "
                               << "in this file; its faces are left to their "
                               << "section." << endl;
                continue;
            }

            if (connType != CGNS_ENUMV(Abutting1to1))
            {
                m_log(FATAL)
                    << "Connectivity '" << name << "' joins zones '"
                    << m_zoneNames[zone] << "' and '" << donor << "' as "
                    << GridConnectivityTypeName[connType]
                    << ". Only Abutting1to1 can be read, because the zones are "
                    << "merged into one mesh and that needs their shared faces "
                    << "to match one for one." << endl;
            }

            interfaceFaces.insert(idx.begin(), idx.end());

            // The donor list pairs the faces one for one, which CGNS enforces
            // for Abutting1to1. Recording the pairs lets the merge below be
            // checked against the file rather than merely believed.
            if (ndataDonor == npnts &&
                donorPtsetType == CGNS_ENUMV(PointListDonor))
            {
                std::vector<int> donorIdx = ResolveFaceSet(
                    lookups[donorZone->first], CGNS_ENUMV(PointList), location,
                    donorPnts, elemInfo, "Donor of '" + name + "'");

                if (donorIdx.size() == idx.size())
                {
                    for (size_t k = 0; k < idx.size(); ++k)
                    {
                        interfacePairs.emplace_back(idx[k], donorIdx[k]);
                    }
                }
            }

            m_log(VERBOSE) << "  - interface '" << name << "' to zone '"
                           << donor << "': " << idx.size()
                           << " faces, stitched rather than kept" << endl;
        }
    }
}

/**
 * @brief Give the zones one set of vertices where they share a surface.
 *
 * The zones of a file are parts of one model, so where two of them abut they
 * describe the same surface twice, with a vertex of each that stands for the
 * same point. Those have to become one vertex or the mesh remains nonconformal:
 * once they are shared, ReadFaces() finds the abutting faces identical and
 * stitches them.
 *
 * Vertices are merged on their coordinates being **equal**, not on being close.
 * There is no tolerance here and deliberately none offered. A mesher writing a
 * shared surface into two zones writes the same numbers into both, so equality
 * is enough; and a tolerance would be free to weld a vertex to its neighbour
 * across the first cell of a boundary layer, which is thinner than any length
 * scale one could pick from the mesh as a whole. Where equality is not enough,
 * we use @p interfacePairs to report a failed merge between zones.
 */
void InputCGNS::MergeZoneVertices(
    std::vector<std::pair<ElementType_t, std::vector<int>>> &elemInfo,
    const std::vector<std::pair<int, int>> &interfacePairs, int &nCoords)
{
    // Keyed on the bit pattern of the coordinates, so that equality here is
    // the same equality the file was written with. Adding zero first maps a
    // negative zero onto a positive one, which compare equal but would
    // otherwise hash apart.
    struct Point
    {
        double x, y, z;
        bool operator==(const Point &o) const
        {
            return x == o.x && y == o.y && z == o.z;
        }
    };
    struct PointHash
    {
        size_t operator()(const Point &p) const
        {
            std::hash<double> h;
            return h(p.x) ^ (h(p.y) << 1) ^ (h(p.z) << 2);
        }
    };

    std::unordered_map<Point, int, PointHash> seen;
    std::vector<int> canonical(nCoords);

    for (int i = 0; i < nCoords; ++i)
    {
        canonical[i] = i;

        Point p{(*m_node[i])[0] + 0.0, (*m_node[i])[1] + 0.0,
                (*m_node[i])[2] + 0.0};

        auto ins = seen.insert({p, i});
        if (!ins.second)
        {
            canonical[i] = ins.first->second;
        }
    }

    // Renumber what survives into a contiguous range, which the node
    // reordering that follows this requires.
    std::vector<int> newId(nCoords, -1);
    int nKept = 0;
    for (int i = 0; i < nCoords; ++i)
    {
        if (canonical[i] == i)
        {
            newId[i] = nKept++;
        }
    }
    for (int i = 0; i < nCoords; ++i)
    {
        if (canonical[i] != i)
        {
            newId[i] = newId[canonical[i]];
        }
    }

    int nMerged = nCoords - nKept;

    if (nMerged == 0)
    {
        m_log(WARNING) << "The zones of this file share no vertices, so they "
                       << "are read as separate pieces lying in one mesh. If "
                       << "they are meant to join, their shared surface is not "
                       << "written with matching coordinates." << endl;
        return;
    }

    for (auto &ei : elemInfo)
    {
        for (int &n : ei.second)
        {
            n = newId[n];
        }
    }

    SpatialDomains::GeomMap<SpatialDomains::PointGeom> keptNodes;
    std::vector<SpatialDomains::PointGeom *> keptNode(nKept, nullptr);
    for (int i = 0; i < nCoords; ++i)
    {
        if (canonical[i] == i)
        {
            m_loadedNodes[i]->SetGlobalID(newId[i]);
            keptNode[newId[i]]  = m_loadedNodes[i].get();
            keptNodes[newId[i]] = std::move(m_loadedNodes[i]);
        }
    }
    m_loadedNodes = std::move(keptNodes);
    m_node        = std::move(keptNode);
    nCoords       = nKept;

    m_log(VERBOSE) << "Merged " << nMerged << " duplicated vertices where the "
                   << "zones meet, leaving " << nKept << endl;

    // Check the result against what the file declared to abut. A pair whose
    // two faces have not become the same face means the surface was written
    // into the two zones with coordinates that are not equal, and the zones
    // are still apart along it.
    int unstitched = 0;
    for (const auto &pair : interfacePairs)
    {
        auto corners = [&](int i) {
            std::set<int> c(elemInfo[i].second.begin(),
                            elemInfo[i].second.end());
            return c;
        };

        if (corners(pair.first) != corners(pair.second))
        {
            ++unstitched;
        }
    }

    if (unstitched)
    {
        m_log(WARNING) << unstitched << " of " << interfacePairs.size()
                       << " faces the file declares to abut did not come "
                       << "together, so the zones are still separate along "
                       << "part of their shared surface. Their coordinates "
                       << "there are not equal." << endl;
    }
    else if (interfacePairs.size())
    {
        m_log(VERBOSE) << "All " << interfacePairs.size()
                       << " declared abutting face pairs came together" << endl;
    }
}

/**
 * @brief Decide which boundary faces form each composite, and what to call it.
 *
 * Two things in a CGNS file can say how the boundary is divided up, and
 * exporters disagree about which to use. The sections are always there, and
 * give a usable division when each patch was written as its own section. The
 * ZoneBC_t boundary conditions are the authoritative statement when they exist,
 * and are the only division available when the whole boundary was written as a
 * single section -- which is otherwise read as one undivided composite, losing
 * the distinction between inlet and wall entirely.
 *
 * So the grid connectivities and boundary conditions win where they apply and
 * the sections cover whatever is left over. A name comes from the Family_t
 * where there is one, since that is where a part ID from ANSA lives, and from
 * the boundary condition, connectivity or section node itself where there is
 * not. Groups that resolve to the same name are merged: sharing a family is
 * precisely how a file says two sets of faces are one patch.
 *
 * The result is ordered by name rather than by position in the file, so that
 * composite numbering does not shift when a mesh is re-exported with its
 * sections in a different order.
 */
std::vector<CGNSBoundaryGroup> InputCGNS::BuildBoundaryGroups(
    const std::vector<CGNSSection> &sections,
    const std::vector<std::pair<ElementType_t, std::vector<int>>> &elemInfo,
    int nZones, const std::map<std::string, std::vector<int>> &connGroups,
    const std::unordered_set<int> &interfaceFaces)
{
    std::map<std::string, std::vector<int>> groups;
    std::unordered_set<int> claimed;

    // The faces two zones share are interior to the assembled mesh, so they
    // belong to no composite. Claiming them keeps them out of everything that
    // follows, including their own section.
    claimed.insert(interfaceFaces.begin(), interfaceFaces.end());

    if (!m_config["nobcgroup"].beenSet)
    {
        // Each side of a periodic pair, which the connectivities have already
        // worked out. They come first because a connectivity splitting a patch
        // into its two sides says more than a condition naming the whole patch,
        // and the split is what a session needs.
        for (const auto &g : connGroups)
        {
            for (int i : g.second)
            {
                if (!claimed.count(i))
                {
                    groups[g.first].push_back(i);
                    claimed.insert(i);
                }
            }
        }

        for (m_zoneIndex = 1; m_zoneIndex <= nZones; ++m_zoneIndex)
        {
            CGNSFaceLookup lookup =
                BuildFaceLookup(sections, elemInfo, m_zoneIndex);
            ReadBoundaryConditionGroups(lookup, elemInfo, groups, claimed);
        }
    }

    // Whatever nothing else accounted for is grouped by the section it came
    // from, which is what this reader did before the other nodes of the file
    // were consulted at all.
    for (const auto &sec : sections)
    {
        if (!sec.isBoundary)
        {
            continue;
        }

        std::string name = sec.family.empty() ? sec.name : sec.family;

        for (int i = sec.infoBegin; i < sec.infoEnd; ++i)
        {
            if (IsBoundaryFace(elemInfo[i].first) && !claimed.count(i))
            {
                groups[name].push_back(i);
            }
        }
    }

    std::vector<CGNSBoundaryGroup> result;
    result.reserve(groups.size());

    for (auto &g : groups)
    {
        if (g.second.empty())
        {
            continue;
        }

        // Faces reach a group by two routes -- a boundary condition and a
        // section of the same name -- so sort and unique rather than assume
        // each arrives once and in order.
        std::sort(g.second.begin(), g.second.end());
        g.second.erase(std::unique(g.second.begin(), g.second.end()),
                       g.second.end());

        result.push_back(CGNSBoundaryGroup{g.first, std::move(g.second)});
    }

    return result;
}

/**
 * In the CGNS format, each element is defined by a list of vertices in a
 * certain order
 * To use ResetNodes, we need to create...
 *
 * FaceNodes: the nodes that form each face (tri or quad)
 * ElementFaces: the faces that form each element (tet, pyra, prism, hex)
 * BoundaryElems: the faces that form the boundary
 */
void InputCGNS::ReadFaces(vector<pair<ElementType_t, vector<int>>> &elemInfo,
                          std::unordered_map<int, vector<int>> &FaceNodes,
                          Array<OneD, vector<int>> &ElementFaces,
                          vector<int> &BoundaryElems, vector<int> &VolumeElems)
{
    set<int> internalFaces = {};

    map<set<int>, int> faceSets;
    int volElemIdx = 0;

    for (int elemIdx = 0; elemIdx < elemInfo.size(); elemIdx++)
    {
        ElementType_t elemType            = elemInfo[elemIdx].first;
        vector<int> elemNodes             = elemInfo[elemIdx].second;
        LibUtilities::ShapeType shapeType = m_elemType2ShapeType[elemType];
        if (shapeType == LibUtilities::eTriangle ||
            shapeType == LibUtilities::eQuadrilateral)
        {
            BoundaryElems.push_back(elemIdx);
            continue;
        }

        VolumeElems.push_back(elemIdx);

        ASSERTL0((shapeType == LibUtilities::eTetrahedron ||
                  shapeType == LibUtilities::eHexahedron ||
                  shapeType == LibUtilities::ePyramid ||
                  shapeType == LibUtilities::ePrism),
                 "shapeType not recognised");

        // loop over each face
        for (vector<int> localFaceNodes : m_shapeType2LocElemNodes[shapeType])
        {
            // the set is used to check for uniqueness
            set<int> globalFaceNodeSet;
            // the vector is appended to FaceNodes
            vector<int> globalFaceNodeVector;

            for (int i : localFaceNodes)
            {
                globalFaceNodeSet.insert(elemNodes[i]);
                globalFaceNodeVector.push_back(elemNodes[i]);
            }

            auto testIns =
                faceSets.insert({globalFaceNodeSet, faceSets.size()});

            int faceIdx = testIns.first->second;
            ElementFaces[volElemIdx].push_back(faceIdx);

            if (testIns.second) // if the face doesn't already exist in set
            {
                FaceNodes[faceIdx] = globalFaceNodeVector;
            }
        }
        volElemIdx++;
    }
}

/**
 * Reorder the node IDs to set the orientation of the elements
 */
void InputCGNS::ResetNodes(vector<SpatialDomains::PointGeom *> &Vnodes,
                           Array<OneD, vector<int>> &ElementFaces,
                           std::unordered_map<int, vector<int>> &FaceNodes,
                           vector<pair<ElementType_t, vector<int>>> &elemInfo,
                           vector<int> &VolumeElems)
{
    int i, j;
    vector<int> NodeReordering(Vnodes.size(), -1);
    int face1_map[3] = {0, 1, 4};
    int face3_map[3] = {3, 2, 5};
    int nodeid       = 0;
    int revNodeid    = Vnodes.size() - 1;

    // Determine Prism triangular face connectivity.
    vector<vector<int>> FaceToPrisms(FaceNodes.size());
    vector<vector<int>> PrismToFaces(ElementFaces.size());
    map<int, int> Prisms;

    map<int, int> Pyras;

    // the global tri/quad face IDs for each prism/pyra
    vector<vector<int>> GlobTriFaces(ElementFaces.size());
    vector<vector<int>> GlobQuadFaces(ElementFaces.size());

    // generate map of prism-faces to prisms and prism to
    // triangular-faces as well as ids of each prism.
    for (i = 0; i < ElementFaces.size(); ++i)
    {
        // Find Prism (and pyramids!).
        if (ElementFaces[i].size() == 5)
        {
            vector<int> LocTriFaces;
            // Find tri and quad faces faces
            for (j = 0; j < ElementFaces[i].size(); ++j)
            {
                if (FaceNodes[ElementFaces[i][j]].size() == 3)
                {
                    LocTriFaces.push_back(j);
                    GlobTriFaces[i].push_back(ElementFaces[i][j]);
                }
                else
                {
                    GlobQuadFaces[i].push_back(ElementFaces[i][j]);
                }
            }

            if (LocTriFaces.size() == 2) // prism otherwise a pyramid
            {
                Prisms[i] = i;

                PrismToFaces[i].push_back(ElementFaces[i][LocTriFaces[0]]);
                PrismToFaces[i].push_back(ElementFaces[i][LocTriFaces[1]]);

                FaceToPrisms[ElementFaces[i][LocTriFaces[0]]].push_back(i);
                FaceToPrisms[ElementFaces[i][LocTriFaces[1]]].push_back(i);
            }
            else if (LocTriFaces.size() == 4)
            {
                Pyras[i] = i;
            }
            else
            {
                ASSERTL0(
                    false,
                    "Not set up for elements which are not Prism or Pyramid");
            }
        }
    }

    // Set the apex node as node in each pyra and ensure any neighbouring prisms
    // align. The set of prisms already walked is kept out here and cleared per
    // line rather than built afresh, since there is one line per triangular
    // face of every pyramid in the mesh.
    std::unordered_set<int> prismLineVisited;

    for (auto &PyraIt : Pyras)
    {
        // find the ID of the apex node (the node in any of the tri faces which
        // is not also in the quad (base) face)
        int apexID            = -1;
        int pyraElemIdx       = PyraIt.second;
        vector<int> baseNodes = FaceNodes[GlobQuadFaces[pyraElemIdx][0]];

        bool pyramidShielded = false;

        // assert that none of the base nodes have already been ID'd
        for (int id : baseNodes)
        {
            if (NodeReordering[id] == -1)
            {
                continue;
            }

            // attempt to resolve issues with impossible meshes BUT this is
            // ideally resolved in the mesh generator
            PyramidShielding(Vnodes, ElementFaces, FaceNodes, elemInfo,
                             VolumeElems, NodeReordering, pyraElemIdx);

            pyramidShielded = true;
            break;
        }

        // The shielding has replaced this pyramid with a smaller one and four
        // tetrahedra, so there is no apex of the original left to number. Go
        // on to the next pyramid.
        if (pyramidShielded)
        {
            continue;
        }

        // The choice of the 0th tri face is arbitrary.
        for (int id : FaceNodes[GlobTriFaces[pyraElemIdx][0]])
        {
            // if a node `id` is not in the base, it must be the apex
            if (find(baseNodes.begin(), baseNodes.end(), id) == baseNodes.end())
            {
                apexID = id;
                break;
            }
        }

        ASSERTL0(apexID != -1, "Apex node not found in pyramid");

        // if apex node hasn't already been given an ID (because of sharing it
        // with another pyramid)
        if (NodeReordering[apexID] == -1)
        {
            NodeReordering[apexID] = revNodeid--;
        }

        // traverse along each prism line that exists out of the pyra
        for (int faceId : GlobTriFaces[pyraElemIdx]) // for each tri face
        {
            // Most of a pyramid's triangular faces have no prism on the other
            // side at all, and there is then no line to walk.
            if (FaceToPrisms[faceId].empty())
            {
                continue;
            }

            prismLineVisited.clear();
            TraversePyraPrismLine(pyraElemIdx, faceId, apexID, FaceToPrisms,
                                  GlobTriFaces, Vnodes, ElementFaces, FaceNodes,
                                  NodeReordering, revNodeid, prismLineVisited);
        }
    }

    // Number the prisms so that every prism collapses onto the same corner of
    // its triangular cross-section as its neighbours do.
    //
    // A prism is a triangle extruded along the three corner edges {0,3}, {1,2}
    // and {4,5}, and its two triangular faces are the ends of that extrusion:
    // entry k of face1_map and of face3_map are the two ends of corner
    // k. OrientPrism later rotates the prism so that the corner carrying the
    // highest global id becomes the singular edge {4,5}, which is where the
    // Duffy transformation of the modified C^0 basis collapses.  Two prisms
    // that share a face therefore only agree on where the collapsed coordinate
    // goes if they agree on which corner holds the highest id.
    //
    // Nominating a corner per prism and then numbering to suit does not work:
    // prisms side by side share the two corners along their common
    // quadrilateral face, so independent nominations contradict each other.
    // Instead the corner is read off a numbering built to make it consistent.
    // Joining the two ends of each corner groups the nodes into the lines
    // running through the prism blocks, and numbering those lines one after
    // another leaves the corner in the last-numbered line highest on every
    // cross-section at once -- every cross-section of a block has its three
    // corners on the same three lines, so that is the same corner the whole
    // way along, and neighbouring blocks agree because they read the same
    // line ordering.
    std::vector<int> lineOf(NodeReordering.size());
    std::iota(lineOf.begin(), lineOf.end(), 0);

    auto lineRoot = [&](int n) {
        while (lineOf[n] != n)
        {
            n = lineOf[n] = lineOf[lineOf[n]];
        }
        return n;
    };

    auto joinLines = [&](int a, int b) {
        a = lineRoot(a);
        b = lineRoot(b);
        if (a != b)
        {
            lineOf[std::max(a, b)] = std::min(a, b);
        }
    };

    std::vector<bool> onPrism(NodeReordering.size(), false);

    // Rebuilding an element is not cheap, so keep what it gives: the check
    // below needs the same answer once every corner has been joined.
    std::vector<Array<OneD, int>> prismNodes;
    prismNodes.reserve(Prisms.size());

    for (auto &PrismIt : Prisms)
    {
        prismNodes.push_back(
            SortFaceNodes(Vnodes, ElementFaces[PrismIt.first], FaceNodes));
        Array<OneD, int> &Nodes = prismNodes.back();

        for (int k = 0; k < 3; ++k)
        {
            onPrism[Nodes[face1_map[k]]] = true;
            onPrism[Nodes[face3_map[k]]] = true;
            joinLines(Nodes[face1_map[k]], Nodes[face3_map[k]]);
        }
    }

    // Collect the lines, each holding its nodes in ascending order. Lines
    // holding a node the pyramid pass has already numbered go last: those ids
    // come from the top of the range, so such a node is the highest on any
    // cross-section it sits on, and the line has to be the highest to match.
    std::map<int, std::vector<int>> lines;
    std::set<int> pinnedLines;

    for (int n = 0; n < (int)NodeReordering.size(); ++n)
    {
        if (!onPrism[n])
        {
            continue;
        }

        int root = lineRoot(n);
        lines[root].push_back(n);

        if (NodeReordering[n] != -1)
        {
            pinnedLines.insert(root);
        }
    }

    // What a prism actually asks for is small: one of its three corners has to
    // come out highest at both of its triangular ends, since that is the one
    // OrientPrism collapses onto. It does not ask for whole lines of nodes to
    // be numbered together, and numbering them that way is more than is
    // required; two prisms that share no face at all are then made to compete
    // over which of their lines is numbered first, and can want opposite
    // things. So the lines are used only to pick the corner, and the numbering
    // is taken from the orderings that choice actually implies.
    //
    // Rank the lines first. Where a prism has two corners on one line the
    // order of the lines cannot tell those two apart, but its third corner can
    // be made to beat them both if its line is ranked after theirs, so collect
    // that as a constraint between lines and rank them accordingly.
    std::map<int, std::set<int>> mustFollow;
    std::map<int, int> nBefore;

    for (auto &line : lines)
    {
        nBefore[line.first] = 0;
    }

    for (Array<OneD, int> &Nodes : prismNodes)
    {
        int root[3];
        for (int k = 0; k < 3; ++k)
        {
            root[k] = lineRoot(Nodes[face1_map[k]]);
        }

        for (int k = 0; k < 3; ++k)
        {
            const int l = (k + 1) % 3, m = (k + 2) % 3;
            if (root[l] == root[m] && root[k] != root[l] &&
                mustFollow[root[l]].insert(root[k]).second)
            {
                ++nBefore[root[k]];
            }
        }
    }

    // Lowest id first among the lines free to go next, except that a line
    // holding a node the pyramid pass has already numbered is held back as
    // long as the constraints allow: those ids come from the top of the range,
    // so such a node is the highest on any cross-section it sits on.
    std::set<std::pair<int, int>> readyLines;
    auto rank = [&](int root) {
        return std::make_pair(pinnedLines.count(root) ? 1 : 0, root);
    };

    for (auto &line : lines)
    {
        if (nBefore[line.first] == 0)
        {
            readyLines.insert(rank(line.first));
        }
    }

    std::map<int, int> lineRank;
    std::set<int> remaining;
    for (auto &line : lines)
    {
        remaining.insert(line.first);
    }

    while (!remaining.empty())
    {
        if (readyLines.empty())
        {
            // The constraints run in a circle, so not all can be met. Release
            // whichever line the fewest are waiting on.
            int best = -1;
            for (int root : remaining)
            {
                if (best < 0 || nBefore[root] < nBefore[best] ||
                    (nBefore[root] == nBefore[best] && rank(root) < rank(best)))
                {
                    best = root;
                }
            }
            nBefore[best] = 0;
            readyLines.insert(rank(best));
        }

        const int root = readyLines.begin()->second;
        readyLines.erase(readyLines.begin());
        remaining.erase(root);
        lineRank[root] = (int)lineRank.size();

        for (int next : mustFollow[root])
        {
            if (--nBefore[next] == 0 && remaining.count(next))
            {
                readyLines.insert(rank(next));
            }
        }
    }

    // Now the choice, made once per block of prisms rather than once per prism:
    // the prisms of a block are joined end to end, so they share the three
    // lines running through them and have to collapse onto the same one.
    // Walking the block gives each prism a map from its own corners to the
    // block's three.
    std::map<int, std::array<int, 3>> cornerSlot;
    std::map<int, int> blockOf;
    std::map<int, std::array<std::set<int>, 3>> blockSlotNodes;
    std::vector<bool> inBlock(ElementFaces.size(), false);
    int nBlocks = 0;

    std::map<int, size_t> prismAt;
    {
        size_t i = 0;
        for (auto &PrismIt : Prisms)
        {
            prismAt[PrismIt.first] = i++;
        }
    }

    for (auto &PrismIt : Prisms)
    {
        if (inBlock[PrismIt.first])
        {
            continue;
        }

        const int block = nBlocks++;
        std::deque<int> toVisit{PrismIt.first};
        inBlock[PrismIt.first]    = true;
        cornerSlot[PrismIt.first] = {0, 1, 2};

        while (!toVisit.empty())
        {
            const int e = toVisit.front();
            toVisit.pop_front();
            blockOf[e] = block;

            Array<OneD, int> &N = prismNodes[prismAt[e]];
            for (int k = 0; k < 3; ++k)
            {
                blockSlotNodes[block][cornerSlot[e][k]].insert(N[face1_map[k]]);
                blockSlotNodes[block][cornerSlot[e][k]].insert(N[face3_map[k]]);
            }

            for (int gf : PrismToFaces[e])
            {
                for (int nb : FaceToPrisms[gf])
                {
                    if (inBlock[nb])
                    {
                        continue;
                    }
                    inBlock[nb] = true;

                    // Match the neighbour's corners to this one's on the
                    // nodes they have in common across the face they share.
                    Array<OneD, int> &M = prismNodes[prismAt[nb]];
                    std::array<int, 3> slot{-1, -1, -1};
                    for (int kk = 0; kk < 3; ++kk)
                    {
                        for (int mm = 0; mm < 3; ++mm)
                        {
                            if (M[face1_map[mm]] == N[face1_map[kk]] ||
                                M[face1_map[mm]] == N[face3_map[kk]] ||
                                M[face3_map[mm]] == N[face1_map[kk]] ||
                                M[face3_map[mm]] == N[face3_map[kk]])
                            {
                                slot[mm] = cornerSlot[e][kk];
                            }
                        }
                    }

                    if (slot[0] < 0 || slot[1] < 0 || slot[2] < 0 ||
                        std::set<int>(slot.begin(), slot.end()).size() != 3)
                    {
                        slot = {0, 1, 2};
                    }

                    cornerSlot[nb] = slot;
                    toVisit.push_back(nb);
                }
            }
        }
    }

    // One choice per block: the highest ranked of its three lines, and where
    // two of them are the same line the lowest node anywhere along the corner,
    // which every prism of the block agrees on. A corner holding a node the
    // pyramid pass numbered is taken first, since that node already outranks
    // anything numbered here.
    std::map<int, int> blockChoice;
    for (auto &bs : blockSlotNodes)
    {
        int best = 0;
        for (int sl = 1; sl < 3; ++sl)
        {
            auto key = [&](int s) {
                bool pinned = false;
                for (int n : bs.second[s])
                {
                    pinned = pinned || NodeReordering[n] != -1;
                }
                return std::make_tuple(
                    pinned ? 1 : 0, lineRank[lineRoot(*bs.second[s].begin())],
                    -*bs.second[s].begin());
            };
            if (key(sl) > key(best))
            {
                best = sl;
            }
        }
        blockChoice[bs.first] = best;
    }

    // The orderings that choice implies, node by node, and nothing more.
    std::map<int, std::set<int>> nodeAbove;
    std::map<int, int> nodeBelow;

    for (auto &PrismIt : Prisms)
    {
        Array<OneD, int> &N = prismNodes[prismAt[PrismIt.first]];
        const int want      = blockChoice[blockOf[PrismIt.first]];

        int c = -1;
        for (int k = 0; k < 3; ++k)
        {
            if (cornerSlot[PrismIt.first][k] == want)
            {
                c = k;
            }
        }

        if (c < 0)
        {
            continue;
        }

        const int *ends[2] = {face1_map, face3_map};
        for (int e = 0; e < 2; ++e)
        {
            for (int k = 0; k < 3; ++k)
            {
                if (k == c)
                {
                    continue;
                }

                const int lo = N[ends[e][k]], hi = N[ends[e][c]];
                if (lo != hi && nodeAbove[lo].insert(hi).second)
                {
                    ++nodeBelow[hi];
                    nodeBelow.emplace(lo, 0);
                }
            }
        }
    }

    // Number the nodes in an order those hold in, lowest id first among those
    // free to go next so that the result does not depend on how the maps
    // happen to be walked.
    std::set<int> readyNodes;
    for (auto &d : nodeBelow)
    {
        if (d.second == 0)
        {
            readyNodes.insert(d.first);
        }
    }

    size_t ordered = 0;
    while (!readyNodes.empty())
    {
        const int n = *readyNodes.begin();
        readyNodes.erase(readyNodes.begin());
        ++ordered;

        if (NodeReordering[n] == -1)
        {
            NodeReordering[n] = nodeid++;
        }

        for (int m : nodeAbove[n])
        {
            if (--nodeBelow[m] == 0)
            {
                readyNodes.insert(m);
            }
        }
    }

    if (ordered != nodeBelow.size())
    {
        // What is left wants to come before itself, which happens when a block
        // of prisms closes on itself with a turn in it. Those prisms have no
        // numbering at all and are split further down.
        m_log(VERBOSE) << "  - " << (nodeBelow.size() - ordered)
                       << " prism nodes lie in a cycle of the orderings their "
                          "elements ask for"
                       << endl;
    }

    // Anything the prisms touch but did not order, then everything else.
    for (int n = 0; n < (int)NodeReordering.size(); ++n)
    {
        if (onPrism[n] && NodeReordering[n] == -1)
        {
            NodeReordering[n] = nodeid++;
        }
    }

    // fill in any unset nodes from other shapes
    for (i = 0; i < NodeReordering.size(); ++i)
    {
        if (NodeReordering[i] == -1)
        {
            NodeReordering[i] = nodeid++;
        }
    }

    ASSERTL1(nodeid == revNodeid + 1, "Have not renumbered all nodes");

    // What the ordering above was for: on each prism the corner that comes out
    // highest has to be the same at both ends, since that is the one
    // OrientPrism collapses onto. A block of prisms that closes on itself with
    // a turn in it cannot be numbered to suit, however the numbering is done,
    // so what is left over is split into elements that carry no such
    // constraint.
    std::vector<int> unorientable;
    {
        size_t i = 0;
        for (auto &PrismIt : Prisms)
        {
            Array<OneD, int> &Nodes = prismNodes[i++];
            int c1 = 0, c3 = 0;
            for (int k = 1; k < 3; ++k)
            {
                if (NodeReordering[Nodes[face1_map[k]]] >
                    NodeReordering[Nodes[face1_map[c1]]])
                {
                    c1 = k;
                }
                if (NodeReordering[Nodes[face3_map[k]]] >
                    NodeReordering[Nodes[face3_map[c3]]])
                {
                    c3 = k;
                }
            }
            if (c1 != c3)
            {
                unorientable.push_back(PrismIt.first);
            }
        }
    }

    if (!unorientable.empty())
    {
        m_log(WARNING) << unorientable.size() << " of " << prismNodes.size()
                       << " prisms collapse onto a different corner at each "
                       << "end, which is not supported by Nektar++. " << endl;
        m_log(WARNING) << "Each is replaced by two tetrahedra and three "
                       << "pyramids coned to a new node at its centre, which "
                       << "leaves its faces as they were." << endl;
    }

    // Helpful debugging code: if 'prismlines' option is set, we write out a
    // visualisation of the prism lines.
    if (m_config["prismlines"].beenSet)
    {
        std::vector<int> prismIds;
        for (auto &PrismIt : Prisms)
        {
            prismIds.push_back(PrismIt.first);
        }

        WritePrismLines(
            m_config["prismlines"].as<std::string>(), Vnodes, prismNodes,
            prismIds, lineRoot,
            std::set<int>(unorientable.begin(), unorientable.end()));
    }

    for (int prismElemIdx : unorientable)
    {
        SplitPrism(Vnodes, ElementFaces, FaceNodes, elemInfo, VolumeElems,
                   NodeReordering, prismElemIdx);
    }

    // Renumbering successful, so reset nodes and faceNodes.
    for (auto &it : FaceNodes)
    {
        for (j = 0; j < it.second.size(); ++j)
        {
            it.second[j] = NodeReordering[it.second[j]];
        }
    }

    SpatialDomains::GeomMap<SpatialDomains::PointGeom> newLoadedNodes;
    for (i = 0; i < Vnodes.size(); ++i)
    {
        m_loadedNodes[i]->SetGlobalID(NodeReordering[i]);
        newLoadedNodes[NodeReordering[i]] = std::move(m_loadedNodes[i]);
    }
    m_loadedNodes = std::move(newLoadedNodes);

    vector<SpatialDomains::PointGeom *> save(Vnodes);
    for (i = 0; i < Vnodes.size(); ++i)
    {
        Vnodes[NodeReordering[i]] = save[i];
    }

    for (auto &ei : elemInfo)
    {
        for (int vertIdx = 0; vertIdx < ei.second.size(); vertIdx++)
        {
            ei.second[vertIdx] = NodeReordering[ei.second[vertIdx]];
        }
    }
}

/**
 * @brief Write the prisms and the lines running through them to a VTK file.
 *
 * Joining the two ends of each of a prism's corner edges groups the nodes into
 * the lines that run through a block of prisms, and a prism can only be
 * oriented if its three corners lie on three different lines. Where boundary
 * layers meet at an angle a line runs out of one layer and into the other,
 * two of a prism's corners end up on it, and the order of the lines no longer
 * settles which of them collapses. This writes out what that looks like.
 *
 * The prisms go out as wedges and the corner edges as line segments, with
 * per-cell values to pick them apart in a viewer:
 *
 *   kind          0 for a prism, 1 for a corner edge
 *   sharedLines   on a prism, how many of its corners share a line: 0 when all
 *                 three differ and nothing is wrong, 2 when two of them share,
 *                 3 when all three do; -1 on an edge
 *   unorientable  1 on a prism that could not be oriented and was split, 0 on
 *                 any other, -1 on an edge
 *   line          on an edge, which line it belongs to; -1 on a prism
 *   lineEdges     how many edges the line has: on an edge its own line, on a
 *                 prism the longest of the lines its three corners sit on
 *   lineTurn      how far the line bends, as the largest angle in degrees
 *                 between one of its edges and their mean direction; again the
 *                 worst of the three on a prism
 *
 * Threshold on sharedLines to find the awkward prisms and colour the edges by
 * line to see the lines they sit on. A line standing in one boundary layer is
 * a few edges long and barely bends; one that has carried on into a second
 * layer meeting the first at an angle is far longer and turns through most of
 * a right angle, and that is where prisms stop being orientable. It is written
 * before the prisms that cannot be oriented are replaced, so they are still
 * there to be seen.
 */
void InputCGNS::WritePrismLines(
    const std::string &fname, std::vector<SpatialDomains::PointGeom *> &Vnodes,
    std::vector<Array<OneD, int>> &prismNodes, std::vector<int> &prismIds,
    const std::function<int(int)> &lineRoot, const std::set<int> &unorientable)
{
    static constexpr int face1_map[3] = {0, 1, 4};
    static constexpr int face3_map[3] = {3, 2, 5};

    // How long each line is and how far it bends, so that a column standing in
    // one boundary layer can be told from one that has turned a corner into
    // another.
    std::map<int, std::vector<std::array<double, 3>>> lineDirs;

    for (Array<OneD, int> &Nodes : prismNodes)
    {
        for (int k = 0; k < 3; ++k)
        {
            const int a = Nodes[face1_map[k]], b = Nodes[face3_map[k]];
            std::array<double, 3> d{(*Vnodes[b])[0] - (*Vnodes[a])[0],
                                    (*Vnodes[b])[1] - (*Vnodes[a])[1],
                                    (*Vnodes[b])[2] - (*Vnodes[a])[2]};
            const double n = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            if (n > 0.0)
            {
                for (double &x : d)
                {
                    x /= n;
                }
                lineDirs[lineRoot(a)].push_back(d);
            }
        }
    }

    std::map<int, int> lineLen, lineBend;

    for (auto &ld : lineDirs)
    {
        // Each edge taken the same way round as the first, since a line can
        // run back on itself through a block numbered the other way.
        std::array<double, 3> mean{0.0, 0.0, 0.0};
        for (auto &d : ld.second)
        {
            const double s = d[0] * ld.second[0][0] + d[1] * ld.second[0][1] +
                             d[2] * ld.second[0][2];
            for (int k = 0; k < 3; ++k)
            {
                mean[k] += (s < 0.0 ? -d[k] : d[k]);
            }
        }

        const double mn = std::sqrt(mean[0] * mean[0] + mean[1] * mean[1] +
                                    mean[2] * mean[2]);
        for (double &x : mean)
        {
            x /= (mn > 0.0 ? mn : 1.0);
        }

        double worst = 0.0;
        for (auto &d : ld.second)
        {
            const double c =
                std::fabs(d[0] * mean[0] + d[1] * mean[1] + d[2] * mean[2]);
            worst = std::max(worst, std::acos(std::min(1.0, c)));
        }

        lineLen[ld.first]  = (int)ld.second.size();
        lineBend[ld.first] = (int)std::lround(worst * 180.0 / M_PI);
    }

    // Only the nodes the prisms use, renumbered from nought.
    std::map<int, int> pointId;
    auto point = [&](int n) {
        auto it = pointId.find(n);
        if (it != pointId.end())
        {
            return it->second;
        }
        const int id = pointId.size();
        pointId[n]   = id;
        return id;
    };

    struct Cell
    {
        std::vector<int> nodes;
        int kind, shared, unorientable, line, lineEdges, lineTurn;
    };
    std::vector<Cell> cells;

    // One wedge per prism. VTK takes the two triangles in the order the
    // corners pair up in, which is what face1_map and face3_map give.
    for (size_t i = 0; i < prismNodes.size(); ++i)
    {
        Array<OneD, int> &Nodes = prismNodes[i];

        std::set<int> roots;
        for (int k = 0; k < 3; ++k)
        {
            roots.insert(lineRoot(Nodes[face1_map[k]]));
        }

        std::vector<int> nodes;
        for (int k = 0; k < 3; ++k)
        {
            nodes.push_back(point(Nodes[face1_map[k]]));
        }
        for (int k = 0; k < 3; ++k)
        {
            nodes.push_back(point(Nodes[face3_map[k]]));
        }

        int longest = 0, bendiest = 0;
        for (int root : roots)
        {
            longest  = std::max(longest, lineLen[root]);
            bendiest = std::max(bendiest, lineBend[root]);
        }

        cells.push_back(
            {nodes, 0, roots.size() == 3 ? 0 : 4 - (int)roots.size(),
             unorientable.count(prismIds[i]) ? 1 : 0, -1, longest, bendiest});
    }

    // One segment per corner edge, kept once however many prisms share it.
    std::set<std::pair<int, int>> done;
    std::map<int, int> lineNumber;

    for (Array<OneD, int> &Nodes : prismNodes)
    {
        for (int k = 0; k < 3; ++k)
        {
            const int a = Nodes[face1_map[k]], b = Nodes[face3_map[k]];
            if (!done.insert({std::min(a, b), std::max(a, b)}).second)
            {
                continue;
            }

            const int root = lineRoot(a);
            if (!lineNumber.count(root))
            {
                const int next   = lineNumber.size();
                lineNumber[root] = next;
            }

            cells.push_back({{point(a), point(b)},
                             1,
                             -1,
                             -1,
                             lineNumber[root],
                             lineLen[root],
                             lineBend[root]});
        }
    }

    std::ofstream out(fname.c_str());
    if (!out)
    {
        m_log(WARNING) << "Could not open '" << fname
                       << "' to write the prism lines to." << endl;
        return;
    }

    out << "# vtk DataFile Version 3.0\n"
        << "prisms and the lines running through them\n"
        << "ASCII\nDATASET UNSTRUCTURED_GRID\n";

    std::vector<int> back(pointId.size());
    for (auto &p : pointId)
    {
        back[p.second] = p.first;
    }

    out << "POINTS " << pointId.size() << " double\n";
    for (int n : back)
    {
        out << (*Vnodes[n])[0] << " " << (*Vnodes[n])[1] << " "
            << (*Vnodes[n])[2] << "\n";
    }

    size_t listed = 0;
    for (auto &c : cells)
    {
        listed += c.nodes.size() + 1;
    }

    out << "CELLS " << cells.size() << " " << listed << "\n";
    for (auto &c : cells)
    {
        out << c.nodes.size();
        for (int n : c.nodes)
        {
            out << " " << n;
        }
        out << "\n";
    }

    out << "CELL_TYPES " << cells.size() << "\n";
    for (auto &c : cells)
    {
        out << (c.kind == 0 ? 13 : 3) << "\n"; // VTK_WEDGE, VTK_LINE
    }

    out << "CELL_DATA " << cells.size() << "\n";
    const char *names[6] = {"kind", "sharedLines", "unorientable",
                            "line", "lineEdges",   "lineTurn"};
    for (int f = 0; f < 6; ++f)
    {
        out << "SCALARS " << names[f] << " int 1\nLOOKUP_TABLE default\n";
        for (auto &c : cells)
        {
            const int v[6] = {c.kind, c.shared,    c.unorientable,
                              c.line, c.lineEdges, c.lineTurn};
            out << v[f] << "\n";
        }
    }

    m_log(VERBOSE) << "  - wrote " << prismNodes.size() << " prisms and "
                   << (cells.size() - prismNodes.size()) << " corner edges on "
                   << lineNumber.size() << " lines to '" << fname << "'"
                   << endl;
}

/**
 * @brief Replace a prism that cannot be oriented with a cone of elements.
 *
 * A prism collapses onto one corner of its triangular cross-section, and the
 * two ends have to agree on which. Numbering settles that for all but a few
 * meshes: where a block of prisms closes on itself with a turn in it, as a
 * ring of prisms joined back to front does, no numbering can, since the choice
 * has to come back round to itself and does not.
 *
 * Such a prism is replaced here by the elements obtained by coning every one
 * of its faces to a new node at its centre: a tetrahedron on each of the two
 * triangles and a pyramid on each of the three quadrilaterals. None of them
 * is a prism, so none of them carries the constraint, and the apex they share
 * is a new node that takes an id above every other, which is what a pyramid
 * wants of its apex.
 *
 * Splitting the prism into three tetrahedra instead would be smaller but is
 * not open to us: its quadrilateral faces are shared with its neighbours, and
 * a tetrahedron cannot present a quadrilateral. Coning keeps all five of the
 * original faces exactly as they were, so the neighbours do not notice.
 */
void InputCGNS::SplitPrism(
    std::vector<SpatialDomains::PointGeom *> &Vnodes,
    Array<OneD, std::vector<int>> &ElementFaces,
    std::unordered_map<int, std::vector<int>> &FaceNodes,
    std::vector<std::pair<ElementType_t, std::vector<int>>> &elemInfo,
    std::vector<int> &VolumeElems, std::vector<int> &NodeReordering,
    int prismElemIdx)
{
    const ElementType_t prismType = elemInfo[VolumeElems[prismElemIdx]].first;
    const int order               = m_elemType2Order[prismType];

    if (order > 2)
    {
        m_log(WARNING) << "A prism that cannot be oriented is of order "
                       << order
                       << ", above what splitting it is set up for. It is "
                          "left as it is."
                       << endl;
        return;
    }

    std::vector<int> &prism = elemInfo[VolumeElems[prismElemIdx]].second;

    // The two ends of each of the prism's nine edges, in the order a CGNS
    // prism lists them, so that an edge's mid node can be found by its index.
    static constexpr int edgeEnds[9][2] = {
        {0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 4}, {2, 5}, {3, 4}, {4, 5}, {5, 3}};

    // A new node at the centre of the prism, which every new element cones to.
    // Every node of the prism is taken, not only its vertices: on a curved
    // element the vertices alone can put this outside the faces, and then the
    // cone is inside out.
    NekDouble cx = 0.0, cy = 0.0, cz = 0.0;
    for (int n : prism)
    {
        cx += (*Vnodes[n])[0];
        cy += (*Vnodes[n])[1];
        cz += (*Vnodes[n])[2];
    }
    cx /= prism.size();
    cy /= prism.size();
    cz /= prism.size();

    const int apex = Vnodes.size();
    SaveNode(apex, cx, cy, cz);

    // Half way along each of the six new edges, for a second order mesh.
    std::array<int, 6> apexEdge{};
    if (order == 2)
    {
        for (int v = 0; v < 6; ++v)
        {
            const int mid = Vnodes.size();
            SaveNode(mid, 0.5 * ((*Vnodes[prism[v]])[0] + cx),
                     0.5 * ((*Vnodes[prism[v]])[1] + cy),
                     0.5 * ((*Vnodes[prism[v]])[2] + cz));
            apexEdge[v] = mid;
        }
    }

    // The mid node of prism edge e, or -1 on a linear mesh.
    auto edgeNode = [&](int e) { return order == 2 ? prism[6 + e] : -1; };

    // One new triangular face per edge of the prism, each shared by the two
    // new elements that meet there.
    const int face0 = FaceNodes.size();
    for (int e = 0; e < 9; ++e)
    {
        FaceNodes[face0 + e] = {prism[edgeEnds[e][0]], prism[edgeEnds[e][1]],
                                apex};
    }

    // Which of the prism's faces the reader rebuilt it from, matched on their
    // nodes rather than assumed: the order they are listed in need not be the
    // order a CGNS prism gives them.
    const int faceVerts[5][4] = {
        {0, 2, 1, -1}, {3, 4, 5, -1}, {0, 1, 4, 3}, {1, 2, 5, 4}, {2, 0, 3, 5}};
    // The prism edges bounding each of those faces, in the same cycle.
    const int faceEdges[5][4] = {
        {2, 1, 0, -1}, {6, 7, 8, -1}, {0, 4, 6, 3}, {1, 5, 7, 4}, {2, 3, 8, 5}};

    int globFace[5] = {-1, -1, -1, -1, -1};
    for (int f = 0; f < 5; ++f)
    {
        std::set<int> want;
        for (int k = 0; k < 4 && faceVerts[f][k] >= 0; ++k)
        {
            want.insert(prism[faceVerts[f][k]]);
        }

        for (int gf : ElementFaces[prismElemIdx])
        {
            std::set<int> have(FaceNodes[gf].begin(), FaceNodes[gf].end());
            if (have == want)
            {
                globFace[f] = gf;
            }
        }
    }

    for (int f = 0; f < 5; ++f)
    {
        if (globFace[f] < 0)
        {
            m_log(FATAL) << "Could not match the faces of a prism that is to "
                            "be split to the ones it was rebuilt from."
                         << endl;
        }
    }

    // The five new elements: a tetrahedron on each triangle, a pyramid on each
    // quadrilateral, all coned to the apex. Their nodes are listed the way
    // CGNS lists them, vertices and then one node per edge.
    std::vector<std::pair<ElementType_t, std::vector<int>>> made;
    std::vector<std::vector<int>> madeFaces;

    for (int f = 0; f < 5; ++f)
    {
        const bool tri = faceVerts[f][3] < 0;
        const int nv   = tri ? 3 : 4;

        // The prism's faces are wound with their normals pointing out of it,
        // and the apex is inside, so the cycle has to be reversed: CGNS wants
        // the base of a tetrahedron or a pyramid wound the other way, its
        // normal towards the apex. Nothing puts this right afterwards -- a
        // pyramid is never reoriented, and above first order the reader takes
        // an element's nodes as it finds them rather than rebuilding it from
        // its faces.
        auto rv = [&](int k) { return k == 0 ? 0 : nv - k; };
        auto re = [&](int k) { return nv - 1 - k; };

        std::vector<int> nodes;
        for (int k = 0; k < nv; ++k)
        {
            nodes.push_back(prism[faceVerts[f][rv(k)]]);
        }
        nodes.push_back(apex);

        if (order == 2)
        {
            // Around the base first, which are edges the prism already has,
            // then out to the apex along the new ones.
            for (int k = 0; k < nv; ++k)
            {
                nodes.push_back(edgeNode(faceEdges[f][re(k)]));
            }
            for (int k = 0; k < nv; ++k)
            {
                nodes.push_back(apexEdge[faceVerts[f][rv(k)]]);
            }
        }

        made.push_back({tri ? (order == 2 ? TETRA_10 : TETRA_4)
                            : (order == 2 ? PYRA_13 : PYRA_5),
                        nodes});

        std::vector<int> faces{globFace[f]};
        for (int k = 0; k < nv; ++k)
        {
            faces.push_back(face0 + faceEdges[f][k]);
        }
        madeFaces.push_back(faces);
    }

    // The first of them takes the prism's place and the rest go on the end.
    elemInfo[VolumeElems[prismElemIdx]] = made[0];

    const int elem0 = ElementFaces.size();
    Array<OneD, std::vector<int>> grown(elem0 + 4);
    for (int e = 0; e < elem0; ++e)
    {
        grown[e] = std::move(ElementFaces[e]);
    }
    grown[prismElemIdx] = madeFaces[0];

    for (int i = 1; i < 5; ++i)
    {
        VolumeElems.push_back(elemInfo.size());
        elemInfo.push_back(made[i]);
        grown[elem0 + i - 1] = madeFaces[i];
    }

    ElementFaces = grown;

    // The new nodes are numbered above everything else, which is what makes
    // the apex the highest of each pyramid that cones to it.
    while ((int)NodeReordering.size() < (int)Vnodes.size())
    {
        NodeReordering.push_back(NodeReordering.size());
    }
}

/**
 * Replace a 'problematic' pyramid with a smaller pyramid and four tets.
 * The tets shield the apex and pyramid tri-faces from neighbouring
 * elements.
 */
void InputCGNS::PyramidShielding(
    vector<SpatialDomains::PointGeom *> &Vnodes,
    Array<OneD, vector<int>> &ElementFaces,
    std::unordered_map<int, vector<int>> &FaceNodes,
    vector<pair<ElementType_t, vector<int>>> &elemInfo,
    vector<int> &VolumeElems, vector<int> &NodeReordering, int pyraElemIdx)
{
    m_log(WARNING) << "Impossible mesh: cannot enforce apex as "
                      "collapsed point. Will attempt pyramid shielding"
                   << endl;

    ElementType_t pyraType = elemInfo[VolumeElems[pyraElemIdx]].first;
    int order              = m_elemType2Order[pyraType];
    int n                  = order - 1;
    bool hasFaceNodes      = m_elemType2FaceNodes[pyraType];
    bool hasVolNodes       = m_elemType2VolNodes[pyraType];

    vector<int> oldPyra = elemInfo[VolumeElems[pyraElemIdx]].second;

    // 1. Create the new nodes
    // define the pyramid vertices
    vector<NekVector<NekDouble>> oldVerts;

    for (int vertIdx = 0; vertIdx < 5; vertIdx++)
    {
        NekDouble vArray[] = {(*Vnodes[oldPyra[vertIdx]])[0],
                              (*Vnodes[oldPyra[vertIdx]])[1],
                              (*Vnodes[oldPyra[vertIdx]])[2]};
        NekVector<NekDouble> v(3, vArray);
        oldVerts.push_back(v);
    }

    // the new apex is the geometric mean of the old apex and the base centre
    // this is somewhat arbitrary
    NekVector<NekDouble> newApex = (4 * oldVerts[4] + oldVerts[0] +
                                    oldVerts[1] + oldVerts[2] + oldVerts[3]) /
                                   8;

    // Create nodes
    // a) new apex node
    int newApexIdx = Vnodes.size();
    SaveNode(newApexIdx, newApex[0], newApex[1], newApex[2]);

    vector<vector<int>> newEdgeNodesIdx(5);
    vector<vector<int>> baseEdgeNodesIdx(4);
    vector<vector<int>> sideEdgeNodesIdx(4);

    vector<vector<vector<int>>> newFaceNodesIdx(8);
    vector<vector<vector<int>>> sideFaceNodesIdx(4);

    vector<int> newPyraVolNodesIdx; // the volume nodes of the new pyra
    vector<int> newTetVolNodesIdx;  // the volume node for each tet
                                    // (if TETRA_35, empty otherwise)

    // b) mid-edge nodes
    if (order > 1)
    {
        NekVector<NekDouble> tmp;
        double frac;
        int newNodeIdx;

        // i) pyramid mid-edge nodes
        for (int edgeIdx = 0; edgeIdx < 4; edgeIdx++)
        {
            for (int nodeIdx = 0; nodeIdx < n; nodeIdx++)
            {
                // linearly interpolate between v[i] (base vert) and newApex
                frac = (nodeIdx + 1.0) / (n + 1.0);
                tmp  = oldVerts[edgeIdx] + frac * (newApex - oldVerts[edgeIdx]);
                newNodeIdx = Vnodes.size();
                SaveNode(newNodeIdx, tmp[0], tmp[1], tmp[2]);
                newEdgeNodesIdx[edgeIdx].push_back(newNodeIdx);
            }
        }

        // ii) tet mid-edge nodes
        for (int nodeIdx = 0; nodeIdx < n; nodeIdx++)
        {
            // linearly interpolate between the new and old apices
            frac       = (nodeIdx + 1.0) / (n + 1.0);
            tmp        = newApex + frac * (oldVerts[4] - newApex);
            newNodeIdx = Vnodes.size();
            SaveNode(newNodeIdx, tmp[0], tmp[1], tmp[2]);
            newEdgeNodesIdx[4].push_back(newNodeIdx);
        }

        // iii) old pyra base mid-edge nodes
        for (int edgeIdx = 0; edgeIdx < 4; edgeIdx++)
        {
            baseEdgeNodesIdx[edgeIdx].assign(oldPyra.begin() + 5 + edgeIdx * n,
                                             oldPyra.begin() + 5 +
                                                 (edgeIdx + 1) * n);
        }

        // iv) old pyra side mid-edge nodes
        for (int edgeIdx = 0; edgeIdx < 4; edgeIdx++)
        {
            sideEdgeNodesIdx[edgeIdx].assign(
                oldPyra.begin() + 5 + (edgeIdx + 4) * n,
                oldPyra.begin() + 5 + (edgeIdx + 5) * n);
        }
    }

    /** eg. if the 0th face represents...
     *            ^
     *           / \
     *          /   \
     *         /  f  \
     *        / d  e  \
     *       / a  b  c \
     *      /___________\
     *
     * newFaceNodesIdx[0] = {{a, b, c},
     *                       {d, e   },
     *                       {f      }}
     * We split the face by rows to make it easier to reverse the
     * orientation
     */

    // c) mid-face nodes
    if (hasFaceNodes)
    {
        NekVector<NekDouble> tmp;
        double frac_v, frac_u;
        int newNodeIdx;

        // i) pyramid mid-face nodes
        for (int faceIdx = 0; faceIdx < 4; faceIdx++)
        {
            for (int j = 0; j < n - 1; j++)
            {
                frac_v = (j + 1.0) / (n + 1); // 1.0 to convert to double
                newFaceNodesIdx[faceIdx].push_back({});

                for (double i = 0; i < n - 1 - j; i++)
                {
                    /**  tmp = a + f_u * (b-a) + f_v * (c-a)
                     *
                     *       i ^
                     *           |
                     *           c
                     *           | \
                     *           a - b -> i
                     *
                     *   where a = oldVert[i]        (base node)
                     *         b = oldVert[(i+1)%4]  (next base node)
                     *         c = new apex
                     */

                    frac_u = (i + 1) / (n + 1);

                    tmp = oldVerts[faceIdx] +
                          frac_u * (oldVerts[(faceIdx + 1) % 4] -
                                    oldVerts[faceIdx]) +
                          frac_v * (newApex - oldVerts[faceIdx]);

                    newNodeIdx = Vnodes.size();
                    SaveNode(newNodeIdx, tmp[0], tmp[1], tmp[2]);
                    newFaceNodesIdx[faceIdx][j].push_back(newNodeIdx);
                }
            }
        }

        // ii) tet mid-face nodes
        for (int faceIdx = 0; faceIdx < 4; faceIdx++)
        {
            for (int j = 0; j < n - 1; j++)
            {
                frac_v = (j + 1.0) / (n + 1); // 1.0 to convert to double
                newFaceNodesIdx[4 + faceIdx].push_back({});

                for (double i = 0; i < n - 1 - j; i++)
                {
                    /** tmp = a + f_u * (b-a) + f_v * (c-a)
                     *
                     *       i ^
                     *           |
                     *           c
                     *           | \
                     *           a - b -> i
                     *
                     *   tmp = a + f_u * (b-a) + f_v * (c-a)
                     *   where a = oldVert[i]  (base node)
                     *         b = new apex
                     *         c = old apex
                     */

                    frac_u = (i + 1) / (n + 1);

                    tmp = oldVerts[faceIdx] +
                          frac_u * (newApex - oldVerts[faceIdx]) +
                          frac_v * (oldVerts[4] - oldVerts[faceIdx]);

                    newNodeIdx = Vnodes.size();
                    SaveNode(newNodeIdx, tmp[0], tmp[1], tmp[2]);
                    newFaceNodesIdx[4 + faceIdx][j].push_back(newNodeIdx);
                }
            }
        }

        // iii) leave the base mid-face nodes

        // iv) old pyra triangular mid-face nodes
        int i = 5 + 8 * n + n * n; // index of the first tri mid-face node

        for (int faceIdx = 0; faceIdx < 4; faceIdx++)
        {
            for (int j = 0; j < n - 1; j++)
            {

                sideFaceNodesIdx[faceIdx].push_back({});
                sideFaceNodesIdx[faceIdx][j].assign(
                    oldPyra.begin() + i, oldPyra.begin() + i + n - 1 - j);

                i += n - 1 - j;
            }
        }
    }

    // d) mid-vol nodes
    if (hasVolNodes)
    {
        NekVector<NekDouble> tmp;
        int newNodeIdx;

        // i) pyramid mid-vol nodes
        for (int k = 0; k < n - 1; k++)
        {
            double frac_w = (k + 1.0) / (n + 1.0);

            if ((n - 1 - k) > 2)
            {
                m_log(WARNING)
                    << "Too many mid-volume nodes in this cross-section";
            }
            else if ((n - 1 - k) == 1)
            {
                tmp = oldVerts[0] + 0.5 * (oldVerts[1] - oldVerts[0]) +
                      0.5 * (oldVerts[3] - oldVerts[0]) +
                      frac_w * (newApex - oldVerts[0]);
                newNodeIdx = Vnodes.size();
                SaveNode(newNodeIdx, tmp[0], tmp[1], tmp[2]);
                newPyraVolNodesIdx.push_back(newNodeIdx);
            }
            else if ((n - 1 - k) == 2)
            {
                vector<int> spiral_i = {0, 1, 1, 0};
                vector<int> spiral_j = {0, 0, 1, 1};
                int i, j;

                for (int nodeIdx = 0; nodeIdx < 4; nodeIdx++)
                {
                    i = spiral_i[nodeIdx];
                    j = spiral_j[nodeIdx];

                    double frac_v = (j + 1.0) / (n - k);
                    double frac_u = (i + 1.0) / (n - k);

                    tmp = oldVerts[0] + frac_u * (oldVerts[1] - oldVerts[0]) +
                          frac_v * (oldVerts[3] - oldVerts[0]) +
                          frac_w * (newApex - oldVerts[0]);
                    newNodeIdx = Vnodes.size();
                    SaveNode(newNodeIdx, tmp[0], tmp[1], tmp[2]);
                    newPyraVolNodesIdx.push_back(newNodeIdx);
                }
            }
        }

        // ii) tet mid-vol nodes
        if (order == 4)
        {
            for (int tetIdx = 0; tetIdx < 4; tetIdx++)
            {
                tmp = (oldVerts[tetIdx] + oldVerts[(tetIdx + 1) % 4] +
                       oldVerts[4] + newApex) /
                      4;
                newNodeIdx = Vnodes.size();
                SaveNode(newNodeIdx, tmp[0], tmp[1], tmp[2]);
                newTetVolNodesIdx.push_back(newNodeIdx);
            }
        }
    }

    // 2 a). UPDATE ElemInfo for pyra
    vector<int> &pyraNodes = elemInfo[VolumeElems[pyraElemIdx]].second;

    // i) leave the base edge nodes as is

    // ii) change the apex
    pyraNodes[4] = newApexIdx;

    // iii) change the side edge nodes
    for (int edgeIdx = 0; edgeIdx < 4; edgeIdx++)
    {
        // copy from the new edge nodes to pyraNodes
        copy(newEdgeNodesIdx[edgeIdx].begin(), newEdgeNodesIdx[edgeIdx].end(),
             pyraNodes.begin() + 5 + (edgeIdx + 4) * n);
    }

    // iv) leave the base face nodes

    // v) change the tri face nodes
    int i =
        5 + 8 * n + hasFaceNodes * n * n; // index of first tri mid-face node

    for (int faceIdx = 0; faceIdx < 4; faceIdx++)
    {
        for (vector<int> &row : newFaceNodesIdx[faceIdx])
        {
            copy(row.begin(), row.end(), pyraNodes.begin() + i);
            i += row.size();
        }
    }

    // vi) change the volume nodes
    copy(newPyraVolNodesIdx.begin(), newPyraVolNodesIdx.end(),
         pyraNodes.begin() + i);
    i += newPyraVolNodesIdx.size();
    ASSERTL0(i == pyraNodes.size(),
             "The wrong number of nodes were changed in the process of "
             "PyramidShielding");

    // 2 b). UPDATE ElemInfo for tets
    for (int tetIdx = elemInfo.size(); tetIdx < elemInfo.size() + 4; tetIdx++)
    {
        VolumeElems.push_back(tetIdx);
    }

    ElementType_t tetType = m_pyrShieldingTetType[pyraType];

    for (int tetIdx = 0; tetIdx < 4; tetIdx++)
    {
        // i) allocate memory
        // vector<int> tetNodes(how many?); // TODO: pre-allocate for efficiency
        vector<int> tetNodes;

        // ii) add vertices
        tetNodes.push_back(oldPyra[tetIdx]);
        tetNodes.push_back(oldPyra[(tetIdx + 1) % 4]);
        tetNodes.push_back(newApexIdx);
        tetNodes.push_back(oldPyra[4]);

        // iii) define and insert edges
        if (order > 1)
        {
            vector<int> &temp0 = baseEdgeNodesIdx[tetIdx];
            vector<int> &temp1 = newEdgeNodesIdx[(tetIdx + 1) % 4];
            vector<int> &temp2 = newEdgeNodesIdx[tetIdx];
            vector<int> &temp3 = sideEdgeNodesIdx[tetIdx];
            vector<int> &temp4 = sideEdgeNodesIdx[(tetIdx + 1) % 4];
            vector<int> &temp5 = newEdgeNodesIdx[4];

            tetNodes.insert(tetNodes.end(), temp0.begin(), temp0.end());
            tetNodes.insert(tetNodes.end(), temp1.begin(), temp1.end());
            tetNodes.insert(tetNodes.end(), temp2.begin(), temp2.end());
            tetNodes.insert(tetNodes.end(), temp3.begin(), temp3.end());
            tetNodes.insert(tetNodes.end(), temp4.begin(), temp4.end());
            tetNodes.insert(tetNodes.end(), temp5.begin(), temp5.end());
        }

        // iv) define and insert faces
        if (hasFaceNodes)
        {
            vector<vector<vector<int>>> tetFaces0to2 = {
                newFaceNodesIdx[tetIdx], sideFaceNodesIdx[tetIdx],
                newFaceNodesIdx[4 + (tetIdx + 1) % 4]};

            vector<vector<int>> &tetFace3 = newFaceNodesIdx[4 + tetIdx];

            // append faces 0-2
            for (vector<vector<int>> &face : tetFaces0to2)
            {
                for (vector<int> &row : face)
                {
                    tetNodes.insert(tetNodes.end(), row.begin(), row.end());
                }
            }

            // append face 3 (reversed)
            for (vector<int> &row : tetFace3)
            {
                tetNodes.insert(tetNodes.end(), row.rbegin(), row.rend());
            }
        }

        if (hasVolNodes)
        {
            tetNodes.push_back(newTetVolNodesIdx[tetIdx]);
        }
        elemInfo.push_back({tetType, tetNodes});
    }

    // 3. ADD faces to FaceNodes and UPDATE ElementFaces
    // create new pyra faces
    int newPyraFace0 = FaceNodes.size();
    for (int faceIdx = 0; faceIdx < 4; faceIdx++)
    {
        FaceNodes[newPyraFace0 + faceIdx] = {
            oldPyra[faceIdx], oldPyra[(faceIdx + 1) % 4], newApexIdx};
    }

    // update ElementFaces for the shielded pyra
    ElementFaces[pyraElemIdx] = {newPyraFace0, newPyraFace0 + 1,
                                 newPyraFace0 + 2, newPyraFace0 + 3,
                                 ElementFaces[pyraElemIdx][4]};

    // add tet faces
    int tetFace0 = newPyraFace0 + 4;
    for (int faceIdx = 0; faceIdx < 4; faceIdx++)
    {
        FaceNodes[tetFace0 + faceIdx] = {oldPyra[faceIdx], oldPyra[4],
                                         newApexIdx};
    }

    // update ElementFaces with new tets
    int tet0 = ElementFaces.size();

    // What the new tetrahedra take from the shielded pyramid, read before the
    // array below takes the old vectors over.
    std::array<int, 4> pyraFace;
    for (int elemIdx = 0; elemIdx < 4; elemIdx++)
    {
        pyraFace[elemIdx] = ElementFaces[pyraElemIdx][elemIdx];
    }

    // Every shielded pyramid grows this array by four, and it holds a vector
    // per element of the whole mesh, so hand the old vectors over rather than
    // copying each one: the work is proportional to the size of the mesh and
    // is repeated once per pyramid shielded, which used to be at most once.
    Array<OneD, vector<int>> NewElementFaces(tet0 + 4);
    for (int elemIdx = 0; elemIdx < tet0; elemIdx++)
    {
        NewElementFaces[elemIdx] = std::move(ElementFaces[elemIdx]);
    }

    for (int elemIdx = 0; elemIdx < 4; elemIdx++)
    {
        NewElementFaces[tet0 + elemIdx] = {
            newPyraFace0 + elemIdx, tetFace0 + (elemIdx + 1) % 4,
            pyraFace[elemIdx], tetFace0 + elemIdx};
    }

    ElementFaces = NewElementFaces;

    // 4. Add to NodeReordering
    int numNodesCreated =
        1                                                     // 1 node created
        + 5 * n                                               // 5 edges
        + (8 * n * (n - 1) / 2) * hasFaceNodes                // 8 tri faces
        + ((n - 1) * n * (2 * (n + 1) - 3) / 6) * hasVolNodes // 1 pyra
        + (order == 4 ? 4 : 0) * hasVolNodes;                 // 4 tets
    for (int nodeIdx = 0; nodeIdx < numNodesCreated; nodeIdx++)
    {
        NodeReordering.push_back(NodeReordering.size());
    }
}

/**
 * Starting from a pyramid apex, traverse along a prism line (a line of
 * prisms with their tri faces touching) recursively Ensure that the
 * collapsed points for each of the tri faces match up with each other and
 * with the pyramid apex - this is done by IDing these points with
 * revNodeId--, ensuring that they have a higher index than the other tri
 * nodes
 */
void InputCGNS::TraversePyraPrismLine(
    int currElemId, int currFaceId, int currApexNode,
    const std::vector<std::vector<int>> &FaceToPrisms,
    const std::vector<std::vector<int>> &GlobTriFaces,
    std::vector<SpatialDomains::PointGeom *> &Vnodes,
    Array<OneD, std::vector<int>> &ElementFaces,
    std::unordered_map<int, std::vector<int>> &FaceNodes,
    std::vector<int> &NodeReordering, int &revNodeid,
    std::unordered_set<int> &visited)
{
    // these convert between the node ordering generated by OrderFaceNodes()
    // and the node ordering for the two triangular faces
    int face1_map[3] = {0, 1, 4};
    int face3_map[3] = {3, 2, 5};

    // iterate over the one/two prisms that connect the face
    for (int nextElemId : FaceToPrisms[currFaceId])
    {
        // only carry out the rest of the function if travelling in the right
        // direction
        if (nextElemId == currElemId)
        {
            continue;
        }

        // The two checks here only stop the walk turning straight back on
        // itself, so a line of prisms that closes into a ring would be walked
        // round for ever. Each prism is entered once.
        if (!visited.insert(nextElemId).second)
        {
            continue;
        }

        // iterate over the two tri faces (one at each end)
        for (int nextFaceId : GlobTriFaces[nextElemId])
        {
            // ensure that we are travelling in the right direction
            if (nextFaceId == currFaceId)
            {
                continue;
            }

            // reorder the nodes
            Array<OneD, int> Nodes =
                SortFaceNodes(Vnodes, ElementFaces[nextElemId], FaceNodes);

            int nextApexNode = -1; // the ID of the opposite collapsed point
            int currFace     = -1; // either 1 or 3

            // highest ID for each face
            int highestId1 = -1;
            int highestId3 = -1;

            // find the orientation of the prism (face 1 vs 3)
            for (int i = 0; i < 3; ++i)
            {
                highestId1 = (NodeReordering[Nodes[face1_map[i]]] > highestId1)
                                 ? NodeReordering[Nodes[face1_map[i]]]
                                 : highestId1;

                highestId3 = (NodeReordering[Nodes[face3_map[i]]] > highestId3)
                                 ? NodeReordering[Nodes[face3_map[i]]]
                                 : highestId3;

                if (Nodes[face1_map[i]] == currApexNode)
                {
                    nextApexNode = Nodes[face3_map[i]];
                    currFace     = 1;
                }

                if (Nodes[face3_map[i]] == currApexNode)
                {
                    nextApexNode = Nodes[face1_map[i]];
                    currFace     = 3;
                }
            }

            if (currFace == 1)
            {
                ASSERTL0(highestId3 == NodeReordering[nextApexNode],
                         "Error meshing: incompatible collapsed point");
            }
            else if (currFace == 3)
            {
                ASSERTL0(highestId1 == NodeReordering[nextApexNode],
                         "Error meshing: incompatible collapsed point");
            }
            else
            {
                m_log(FATAL) << "Error meshing: collapsed point not found";
            }

            if (NodeReordering[nextApexNode] == -1)
            {
                NodeReordering[nextApexNode] = revNodeid--;
            }

            // continue traversing along the line
            TraversePyraPrismLine(nextElemId, nextFaceId, nextApexNode,
                                  FaceToPrisms, GlobTriFaces, Vnodes,
                                  ElementFaces, FaceNodes, NodeReordering,
                                  revNodeid, visited);
        }
    }
}

void InputCGNS::GenElement2D(vector<SpatialDomains::PointGeom *> &VertNodes,
                             vector<int> elemNodes, ElementType_t elemType,
                             int nComposite, bool DoOrient)
{
    int order                         = m_elemType2Order[elemType];
    LibUtilities::ShapeType shapeType = m_elemType2ShapeType[elemType];
    LibUtilities::PointsType edgeCurveType;
    LibUtilities::PointsType faceCurveType;

    // Now the boundary face only needs the composite tag.
    // KK bugfix - It should not be duplicated if created by3D element and
    // exist!
    int fourthID            = (shapeType == LibUtilities::eTriangle)
                                  ? -1
                                  : VertNodes[elemNodes[3]]->GetGlobalID();
    std::array<int, 4> vids = {VertNodes[elemNodes[0]]->GetGlobalID(),
                               VertNodes[elemNodes[1]]->GetGlobalID(),
                               VertNodes[elemNodes[2]]->GetGlobalID(),
                               fourthID};
    auto fIt                = m_mesh->m_faceSet.find(vids);
    if (fIt != m_mesh->m_faceSet.end())
    {
        m_mesh->m_elementTags[2][fIt->second] = nComposite;
        return;
    }

    // make unique node list
    vector<SpatialDomains::PointGeom *> nodeList;

    // Look up reordering.
    auto oIt = m_orderingMap.find(elemType);

    // If it's not created, then create it.
    if (oIt == m_orderingMap.end())
    {
        oIt = m_orderingMap
                  .insert(make_pair(elemType, CGNSReordering(shapeType, order)))
                  .first;
    }

    for (int i = 0; i < elemNodes.size(); i++)
    {
        nodeList.push_back((oIt->second.size())
                               ? VertNodes[elemNodes[oIt->second[i]]]
                               : VertNodes[elemNodes[i]]);
    }

    bool faceNodes, volumeNodes;
    faceNodes     = m_elemType2FaceNodes[elemType];
    volumeNodes   = false;
    edgeCurveType = LibUtilities::ePolyEvenlySpaced;
    faceCurveType = (shapeType == LibUtilities::eTriangle)
                        ? LibUtilities::eNodalTriEvenlySpaced
                        : LibUtilities::ePolyEvenlySpaced;

    ElmtConfig conf(shapeType, order, faceNodes, volumeNodes, DoOrient,
                    edgeCurveType, faceCurveType);
    SpatialDomains::Geometry *E = GetElementFactory().CreateInstance(
        shapeType, nodeList, m_mesh->m_meshGraph, m_mesh->m_edgeSet,
        m_mesh->m_faceSet, conf, &m_vertIDs, &m_curveNodeIDs, nullptr, nullptr);

    m_mesh->m_elementTags[2][E] = nComposite;
}

void InputCGNS::GenElement3D(vector<SpatialDomains::PointGeom *> &VertNodes,
                             vector<int> elemNodes, vector<int> &ElementFace,
                             std::unordered_map<int, vector<int>> &FaceNodes,
                             ElementType_t elemType, int nComposite,
                             bool DoOrient)
{
    int order                         = m_elemType2Order[elemType];
    LibUtilities::ShapeType shapeType = m_elemType2ShapeType[elemType];
    LibUtilities::PointsType edgeCurveType;
    LibUtilities::PointsType faceCurveType;

    // make unique node list
    vector<SpatialDomains::PointGeom *> nodeList;

    if (order == 1)
    {
        Array<OneD, int> Nodes =
            SortFaceNodes(VertNodes, ElementFace, FaceNodes);

        for (int i = 0; i < Nodes.size(); i++)
        {
            nodeList.push_back(VertNodes[Nodes[i]]);
        }
    }
    else
    {
        // Look up reordering.
        auto oIt = m_orderingMap.find(elemType);

        // If it's not created, then create it.
        if (oIt == m_orderingMap.end())
        {
            oIt = m_orderingMap
                      .insert(
                          make_pair(elemType, CGNSReordering(shapeType, order)))
                      .first;
        }

        for (int i = 0; i < elemNodes.size(); i++)
        {
            nodeList.push_back((oIt->second.size())
                                   ? VertNodes[elemNodes[oIt->second[i]]]
                                   : VertNodes[elemNodes[i]]);
        }
    }

    bool faceNodes, volumeNodes;
    faceNodes     = m_elemType2FaceNodes[elemType];
    volumeNodes   = m_elemType2VolNodes[elemType];
    edgeCurveType = LibUtilities::ePolyEvenlySpaced;
    faceCurveType = (shapeType == LibUtilities::eTetrahedron)
                        ? LibUtilities::eNodalTriEvenlySpaced
                        : LibUtilities::ePolyEvenlySpaced;

    ElmtConfig conf(shapeType, order, faceNodes, volumeNodes, DoOrient,
                    edgeCurveType, faceCurveType);
    SpatialDomains::Geometry *E = GetElementFactory().CreateInstance(
        shapeType, nodeList, m_mesh->m_meshGraph, m_mesh->m_edgeSet,
        m_mesh->m_faceSet, conf, &m_vertIDs, &m_curveNodeIDs, nullptr, nullptr);

    m_mesh->m_elementTags[3][E] = nComposite;
}

vector<int> InputCGNS::CGNSReordering(LibUtilities::ShapeType shapeType,
                                      int order)
{
    vector<int> cgns2gmsh, gmsh2nek, mapping;

    switch (order)
    {
        case 1:    // 1st order
            break; // no reordering needed

        case 2: // 2nd order
            switch (shapeType)
            {
                case LibUtilities::eTriangle:
                    break;

                case LibUtilities::eQuadrilateral:
                    break;

                case LibUtilities::eTetrahedron:
                    break;

                case LibUtilities::ePyramid:
                    break;

                case LibUtilities::ePrism:
                    mapping = {3, 4,  1,  0, 5, 2,  12, 10, 6,
                               9, 14, 13, 7, 8, 11, 15, 16, 17};
                    break;

                case LibUtilities::eHexahedron:
                    break;

                default:
                    m_log(WARNING)
                        << "Shape type " << shapeType << " not recognised";
                    break;
            }
            break;

        case 3: // 3rd order
            switch (shapeType)
            {
                case LibUtilities::eTriangle:
                    break;

                case LibUtilities::eQuadrilateral:
                    mapping = {0, 1, 2,  3,  4,  5,  6,  7,
                               8, 9, 10, 11, 12, 13, 15, 14};
                    break;

                case LibUtilities::eTetrahedron:
                    mapping = {0,  1,  2,  3,  4,  5,  6,  7,  9,  8,
                               10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
                    break;

                case LibUtilities::ePyramid:
                    mapping = {0,  1,  2,  3,  4,  5,  6,  7,  8,  10,
                               9,  12, 11, 13, 14, 15, 16, 17, 18, 19,
                               20, 21, 22, 24, 23, 25, 26, 27, 28, 29};
                    break;

                case LibUtilities::ePrism:
                    mapping = {3,  4,  1,  0,  5,  2,  18, 19, 15, 14,
                               7,  6,  12, 13, 23, 22, 20, 21, 8,  9,
                               11, 10, 17, 16, 28, 27, 25, 26, 37, 32,
                               29, 31, 30, 24, 35, 34, 36, 33, 38, 39};
                    break;

                case LibUtilities::eHexahedron:
                    mapping = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                               11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                               22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                               33, 35, 34, 36, 37, 39, 38, 40, 41, 43, 42,
                               45, 44, 46, 47, 49, 48, 50, 51, 52, 53, 55,
                               54, 56, 57, 59, 58, 60, 61, 63, 62};
                    break;

                default:
                    m_log(WARNING)
                        << "Shape type " << shapeType << " not recognised";
                    break;
            }
            break;

        case 4: // 4th order
            switch (shapeType)
            {
                case LibUtilities::eTriangle:
                    break;

                case LibUtilities::eQuadrilateral:
                    mapping = {0,  1,  2,  3,  4,  5,  6,  7,  8,
                               9,  10, 11, 12, 13, 14, 15, 16, 17,
                               18, 23, 24, 19, 22, 21, 20};
                    break;

                case LibUtilities::eTetrahedron:
                    mapping = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  12, 11,
                               10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                               24, 25, 26, 27, 28, 29, 30, 32, 31, 33, 34};
                    break;

                case LibUtilities::ePyramid:
                    mapping = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                               13, 12, 11, 16, 15, 14, 17, 18, 19, 20, 21,
                               22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 36,
                               37, 32, 35, 34, 33, 38, 39, 40, 41, 42, 43,
                               45, 44, 46, 48, 47, 49, 50, 51, 53, 52, 54};
                    break;

                case LibUtilities::ePrism:
                    mapping = {3,  4,  1,  0,  5,  2,  24, 25, 26, 20, 19,
                               18, 8,  7,  6,  15, 16, 17, 32, 31, 30, 27,
                               28, 29, 9,  10, 11, 14, 13, 12, 23, 22, 21,
                               42, 41, 40, 43, 44, 39, 36, 37, 38, 63, 64,
                               65, 51, 52, 45, 50, 53, 46, 49, 48, 47, 33,
                               34, 35, 58, 57, 56, 59, 62, 55, 60, 61, 54,
                               72, 69, 66, 73, 70, 67, 74, 71, 68};
                    break;

                case LibUtilities::eHexahedron:
                    mapping = {0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
                               10,  11,  12,  13,  14,  15,  16,  17,  18,  19,
                               20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
                               30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
                               40,  41,  42,  43,  44,  45,  46,  51,  52,  47,
                               50,  49,  48,  53,  54,  55,  60,  61,  56,  59,
                               58,  57,  62,  63,  64,  69,  70,  65,  68,  67,
                               66,  73,  72,  71,  74,  79,  78,  75,  76,  77,
                               82,  81,  80,  83,  88,  87,  84,  85,  86,  89,
                               90,  91,  96,  97,  92,  95,  94,  93,  98,  99,
                               100, 105, 106, 101, 104, 103, 102, 107, 108, 109,
                               114, 115, 110, 113, 112, 111, 116, 117, 118, 123,
                               124, 119, 122, 121, 120};
                    break;

                default:
                    m_log(WARNING)
                        << "Shape type " << shapeType << " not recognised";
                    break;
            }
            break;

        default:
            m_log(FATAL) << "Order " << order << " not recognised" << endl;
            break;
    }

    return mapping;
}

Array<OneD, int> InputCGNS::SortEdgeNodes(
    vector<SpatialDomains::PointGeom *> &Vnodes, vector<int> &FaceNodes)
{
    Array<OneD, int> returnval;

    if (FaceNodes.size() == 3) // Triangle
    {
        returnval = Array<OneD, int>(3);

        returnval[0] = FaceNodes[0];
        returnval[1] = FaceNodes[1];
        returnval[2] = FaceNodes[2];
    }
    else if (FaceNodes.size() == 4) // quadrilateral
    {
        returnval = Array<OneD, int>(4);

        int indx0 = FaceNodes[0];
        int indx1 = FaceNodes[1];
        int indx2 = FaceNodes[2];
        int indx3 = FaceNodes[3];

        SpatialDomains::PointGeom a, b, c, d, acurlb, acurld;
        // calculate 0-1,
        a.Sub(*(Vnodes[indx1]), *(Vnodes[indx0]));
        // calculate 0-2,
        b.Sub(*(Vnodes[indx2]), *(Vnodes[indx0]));
        acurlb.Mult(a, b);

        // calculate 2-1,
        c.Sub(*(Vnodes[indx1]), *(Vnodes[indx2]));
        // calculate 3-2,
        d.Sub(*(Vnodes[indx3]), *(Vnodes[indx2]));
        acurld.Mult(a, d);

        NekDouble acurlb_dot_acurld = acurlb.dot(acurld);
        if (acurlb_dot_acurld > 0.0)
        {
            returnval[0] = indx0;
            returnval[1] = indx1;
            returnval[2] = indx2;
            returnval[3] = indx3;
        }
        else
        {
            returnval[0] = indx0;
            returnval[1] = indx1;
            returnval[2] = indx3;
            returnval[3] = indx2;
        }
    }

    return returnval;
}

Array<OneD, int> InputCGNS::SortFaceNodes(
    vector<SpatialDomains::PointGeom *> &Vnodes, vector<int> &ElementFaces,
    std::unordered_map<int, vector<int>> &FaceNodes)
{
    int i, j;
    Array<OneD, int> returnval;

    if (ElementFaces.size() == 4) // Tetrahedron
    {
        ASSERTL1(FaceNodes[ElementFaces[0]].size() == 3,
                 "Face is not triangular");

        returnval = Array<OneD, int>(4);

        auto it   = FaceNodes.find(ElementFaces[0]);
        int indx0 = it->second[0];
        int indx1 = it->second[1];
        int indx2 = it->second[2];
        int indx3 = -1;

        SpatialDomains::PointGeom a, b, c, acurlb;
        // calculate 0-1,
        a.Sub(*(Vnodes[indx1]), *(Vnodes[indx0]));
        // calculate 0-2,
        b.Sub(*(Vnodes[indx2]), *(Vnodes[indx0]));

        // Find fourth node index;
        ASSERTL1(FaceNodes[ElementFaces[1]].size() == 3,
                 "Face is not triangular");

        auto it2 = FaceNodes.find(ElementFaces[1]);
        for (i = 0; i < 3; ++i)
        {
            if ((it2->second[i] != indx0) && (it2->second[i] != indx1) &&
                (it2->second[i] != indx2))
            {
                indx3 = it2->second[i];
                break;
            }
        }

        // calculate 0-3,
        c.Sub(*(Vnodes[indx3]), *(Vnodes[indx0]));
        acurlb.Mult(a, b);

        NekDouble acurlb_dotc = acurlb.dot(c);
        if (acurlb_dotc < 0.0)
        {
            returnval[0] = indx0;
            returnval[1] = indx1;
            returnval[2] = indx2;
            returnval[3] = indx3;
        }
        else
        {
            returnval[0] = indx1;
            returnval[1] = indx0;
            returnval[2] = indx2;
            returnval[3] = indx3;
        }
    }
    else if (ElementFaces.size() == 5) // prism or pyramid
    {
        int triface0, triface1, triface2, triface3;
        int quadface0, quadface1, quadface2;
        bool isPrism = true;

        // find ids of tri faces and first quad face
        triface0 = triface1 = triface2 = triface3 = -1;
        quadface0 = quadface1 = quadface2 = -1;
        for (i = 0; i < 5; ++i)
        {
            auto it = FaceNodes.find(ElementFaces[i]);
            if (it->second.size() == 3)
            {
                if (triface0 == -1)
                {
                    triface0 = i;
                }
                else if (triface1 == -1)
                {
                    triface1 = i;
                }
                else if (triface2 == -1)
                {
                    triface2 = i;
                    isPrism  = false;
                }
                else if (triface3 == -1)
                {
                    triface3 = i;
                }
            }

            if (it->second.size() == 4)
            {
                if (quadface0 == -1)
                {
                    quadface0 = i;
                }
                else if (quadface1 == -1)
                {
                    quadface1 = i;
                }
                else if (quadface2 == -1)
                {
                    quadface2 = i;
                }
            }
        }

        if (isPrism) // Prism
        {
            returnval = Array<OneD, int>(6);
            ASSERTL1(quadface0 != -1, "Quad face 0 not found");
            ASSERTL1(quadface1 != -1, "Quad face 1 not found");
            ASSERTL1(quadface2 != -1, "Quad face 2 not found");
            ASSERTL1(triface0 != -1, "Tri face 0 not found");
            ASSERTL1(triface1 != -1, "Tri face 1 not found");
        }
        else // Pyramid
        {
            returnval = Array<OneD, int>(5);
            ASSERTL1(quadface0 != -1, "Quad face 0 not found");
            ASSERTL1(triface0 != -1, "Tri face 0 not found");
            ASSERTL1(triface1 != -1, "Tri face 1 not found");
            ASSERTL1(triface2 != -1, "Tri face 2 not found");
            ASSERTL1(triface3 != -1, "Tri face 3 not found");
        }

        // find matching nodes between triface0 and triquad0
        int indx0, indx1, indx2, indx3, indx4;

        indx0 = indx1 = indx2 = indx3 = indx4 = -1;
        // Loop over all quad nodes and if they match any
        // triangular nodes If they do set these to indx0 and
        // indx1 and if not set it to indx2, indx3

        auto &triface0_vec  = FaceNodes.find(ElementFaces[triface0])->second;
        auto &quadface0_vec = FaceNodes.find(ElementFaces[quadface0])->second;
        for (i = 0; i < 4; ++i)
        {
            for (j = 0; j < 3; ++j)
            {
                if (triface0_vec[j] == quadface0_vec[i])
                {
                    break; // same node break
                }
            }

            if (j == 3) // Vertex not in quad face
            {
                if (indx2 == -1)
                {
                    indx2 = quadface0_vec[i];
                }
                else if (indx3 == -1)
                {
                    indx3 = quadface0_vec[i];
                }
                else
                {
                    m_log(FATAL) << "More than two vertices do not match "
                                 << "triangular face" << endl;
                }
            }
            else // if found match then set indx0,indx1;
            {
                if (indx0 == -1)
                {
                    indx0 = quadface0_vec[i];
                }
                else
                {
                    indx1 = quadface0_vec[i];
                }
            }
        }

        // Finally check for top vertex
        for (int i = 0; i < 3; ++i)
        {
            if (triface0_vec[i] != indx0 && triface0_vec[i] != indx1 &&
                triface0_vec[i] != indx2)
            {
                indx4 = triface0_vec[i];
                break;
            }
        }

        SpatialDomains::PointGeom a, b, c, acurlb;
        // calculate 0-1,
        a.Sub(*(Vnodes[indx1]), *(Vnodes[indx0]));
        // calculate 0-4,
        b.Sub(*(Vnodes[indx4]), *(Vnodes[indx0]));
        // calculate 0-2,
        c.Sub(*(Vnodes[indx2]), *(Vnodes[indx0]));
        acurlb.Mult(a, b);

        NekDouble acurlb_dotc = acurlb.dot(c);
        if (acurlb_dotc < 0.0)
        {
            returnval[0] = indx0;
            returnval[1] = indx1;
            returnval[4] = indx4;
        }
        else
        {
            returnval[0] = indx1;
            returnval[1] = indx0;
            returnval[4] = indx4;
        }

        // check to see if two vertices are shared between one of the other
        // faces to define which is indx2 and indx3
        if (isPrism == true)
        {
            auto &quadface1_vec =
                FaceNodes.find(ElementFaces[quadface1])->second;
            auto &quadface2_vec =
                FaceNodes.find(ElementFaces[quadface2])->second;
            int cnt = 0;
            for (int i = 0; i < 4; ++i)
            {
                if (quadface1_vec[i] == returnval[1] ||
                    quadface1_vec[i] == indx2)
                {
                    cnt++;
                }
            }

            if (cnt == 2) // have two matching vertices
            {
                returnval[2] = indx2;
                returnval[3] = indx3;
            }
            else
            {
                cnt = 0;
                for (int i = 0; i < 4; ++i)
                {
                    if (quadface2_vec[i] == returnval[1] ||
                        quadface2_vec[i] == indx2)
                    {
                        cnt++;
                    }
                }

                if (cnt != 2) // neither of the other faces has two matching
                              // nodes so reverse
                {
                    returnval[2] = indx3;
                    returnval[3] = indx2;
                }
                else // have two matching vertices
                {
                    returnval[2] = indx2;
                    returnval[3] = indx3;
                }
            }
            // finally need to find last vertex from second triangular face.
            auto &triface1_vec = FaceNodes.find(ElementFaces[triface1])->second;
            for (int i = 0; i < 3; ++i)
            {
                if (triface1_vec[i] != indx2 && triface1_vec[i] != indx3)
                {
                    returnval[5] = triface1_vec[i];
                    break;
                }
            }
        }
        else
        {
            vector<int> trifaceid;
            trifaceid.push_back(triface1);
            trifaceid.push_back(triface2);
            trifaceid.push_back(triface3);
            int cnt = 0;
            for (auto id : trifaceid)
            {
                auto &triface_vec = FaceNodes.find(ElementFaces[id])->second;
                cnt               = 0;
                for (int i = 0; i < 3; ++i)
                {
                    if (triface_vec[i] == returnval[1] ||
                        triface_vec[i] == indx2)
                    {
                        cnt++;
                    }
                }

                if (cnt == 2) // have two matching vertices
                {
                    returnval[2] = indx2;
                    returnval[3] = indx3;
                    break;
                }
            }
            if (cnt != 2)
            {
                returnval[2] = indx3;
                returnval[3] = indx2;
            }
        }
    }
    else if (ElementFaces.size() == 6) // hexahedron
    {
        ASSERTL1(FaceNodes[ElementFaces[0]].size() == 4 &&
                     FaceNodes[ElementFaces[1]].size() == 4 &&
                     FaceNodes[ElementFaces[2]].size() == 4 &&
                     FaceNodes[ElementFaces[3]].size() == 4 &&
                     FaceNodes[ElementFaces[4]].size() == 4 &&
                     FaceNodes[ElementFaces[5]].size() == 4,
                 "Shape not recognised");

        returnval = Array<OneD, int>(8);

        // neighbours map contains a set of each neighbouring node for every
        // node
        map<int, std::unordered_set<int>> neighbours;

        // populate neighbours
        for (int faceId : ElementFaces)
        {
            vector<int> nodes = FaceNodes[faceId];
            neighbours[nodes[0]].insert({nodes[1], nodes[3]});
            neighbours[nodes[1]].insert({nodes[2], nodes[0]});
            neighbours[nodes[2]].insert({nodes[3], nodes[1]});
            neighbours[nodes[3]].insert({nodes[0], nodes[2]});
        }

        // the bottom face is chosen arbitrarily as the 0th indexed one
        vector<int> bottomFace = FaceNodes[ElementFaces[0]];
        std::unordered_set<int> bottomFaceSet(bottomFace.begin(),
                                              bottomFace.end());

        // the top face is the face opposite, sharing no edges or vertices
        vector<int> topFace;

        // find the top face
        for (int i = 1; i < 6; i++)
        {
            bool sharedEdge = false;
            for (int id : FaceNodes[ElementFaces[i]])
            {
                if (bottomFaceSet.count(id)) // if node @ id is in the
                                             // bottom face
                {
                    sharedEdge = true;
                    break;
                }
            }
            if (sharedEdge)
            {
                continue;
            }
            topFace = FaceNodes[ElementFaces[i]];
        }

        ASSERTL0(topFace.size() == 4, "Top face of hex not found");

        // insert the bottom face nodes into returnval[0:4]
        // insert the top face nodes into returnval[4:]
        // returnval[i+4] is the node opposite returnval[i]
        for (int i = 0; i < 4; i++)
        {
            returnval[i]     = bottomFace[i];
            bool oppositeSet = false;

            for (int id : neighbours[bottomFace[i]])
            {
                if (!bottomFaceSet.count(id))
                {
                    returnval[i + 4] = id;
                    oppositeSet      = true;
                    break;
                }
            }
            ASSERTL0(oppositeSet, "Node in the top face of hex not numbered");
        }

        // The face taken as the bottom carries whichever winding was stored
        // for it, which is the winding given by the first element to meet it;
        // for the element on its other side that is the wrong way round, and
        // the hexahedron comes out inside out. Nothing above can tell, since
        // the two elements differ only in which side of the face they are on,
        // so the handedness is settled here from the geometry, as the
        // tetrahedron, prism and pyramid cases above already do.
        SpatialDomains::PointGeom a, b, c, acurlb;
        // calculate 0-1,
        a.Sub(*(Vnodes[returnval[1]]), *(Vnodes[returnval[0]]));
        // calculate 0-3,
        b.Sub(*(Vnodes[returnval[3]]), *(Vnodes[returnval[0]]));
        // calculate 0-4, which leaves the bottom face,
        c.Sub(*(Vnodes[returnval[4]]), *(Vnodes[returnval[0]]));
        acurlb.Mult(a, b);

        if (acurlb.dot(c) < 0.0)
        {
            // Reverse both faces, which turns the element the right way out
            // while keeping returnval[i + 4] opposite returnval[i].
            std::swap(returnval[1], returnval[3]);
            std::swap(returnval[5], returnval[7]);
        }
    }
    else
    {
        m_log(FATAL) << "SortFaceNodes not set up for this number of faces"
                     << endl;
    }

    return returnval;
}
} // namespace Nektar::NekMesh

/*  TODO:
    - improve ResetNodes
        - test

    - figure out if part labels can be read
        (eg. 'inlet', 'outlet', ...)

    - add support for the structured mesh type?
*/
