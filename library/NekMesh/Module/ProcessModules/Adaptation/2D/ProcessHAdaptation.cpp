////////////////////////////////////////////////////////////////////////////////
//
//  File: ProcessHAdaptation.cpp
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
//  Description: Split Tets and Triags to ratio 1:2 ; 1:4 ; 1:8
//
////////////////////////////////////////////////////////////////////////////////

#include "NekMesh/Module/ProcessModules/ProcessVarOpti/ElUtil.h"
#include <NekMesh/MeshElements/Element.h>
#include <NekMesh/Module/ProcessModules/Adaptation/2D/ProcessHAdaptation.h>
#include <NekMesh/Module/ProcessModules/Adaptation/2D/ProcessHAdaptationMaps.h>
#include <NekMesh/Module/ProcessModules/Adaptation/HangingNodes.h>
#include <NekMesh/Module/ProcessModules/ProcessProjectCAD.h>
#include <NekMesh/Module/ProcessModules/ProcessVarOpti/ProcessVarOpti.h>

#include <boost/algorithm/string.hpp>

#include <LibUtilities/BasicUtils/Timer.h>
#include <LibUtilities/Foundations/NodalUtil.h>

#include <LibUtilities/BasicUtils/Progressbar.hpp>
#include <LibUtilities/Foundations/ManagerAccess.h>

#include <StdRegions/StdHexExp.h>
#include <StdRegions/StdPrismExp.h>
#include <StdRegions/StdPyrExp.h>
#include <StdRegions/StdQuadExp.h>
#include <StdRegions/StdTetExp.h>
#include <StdRegions/StdTriExp.h>

#include <SpatialDomains/GeometryLocator.h>

using namespace std;
using namespace Nektar::NekMesh;

namespace Nektar::NekMesh
{

ModuleKey ProcessHAdaptation::className =
    GetModuleFactory().RegisterCreatorFunction(
        ModuleKey(eProcessModule, "h_adaptation"), ProcessHAdaptation::create,
        "Performs refinement by splitting Triag elements into 1:2 1:3 1:4");

ProcessHAdaptation::ProcessHAdaptation(MeshSharedPtr m) : ProcessModule(m)
{
    m_config["file"]  = ConfigOption(false, "", "CAD file");
    m_config["order"] = ConfigOption(false, "4", "Order");
    m_config["adaptfile"] =
        ConfigOption(false, "", "Read Elements ID to adapt ");
    m_config["adaptfileEdges"] =
        ConfigOption(false, "", "Read Edges ID to adapt ");
    m_config["uniform"] = ConfigOption(false, "", "UnformRefinement");
    m_config["finest"] =
        ConfigOption(false, "", "Choose Finest Map for Refinement");
    m_config["multilevel"] = ConfigOption(false, "", "Multilevel call T/F");
    m_config["hr"] =
        ConfigOption(false, "", "Need to call r-adaptation after h? ");
    m_config["h_min"] =
        ConfigOption(false, "", "Minimum size to split the element ");
    m_config["areafile"] =
        ConfigOption(false, "",
                     "Read file with [x y z Area-desired] compare with given "
                     "mesh size (useful for hr-cycle)");
    m_config["swap"] =
        ConfigOption(false, "", "Performs isoparametric swapping ");
    m_config["grading"] =
        ConfigOption(false, "", "Grading for jumping el size (TO DO)");
    m_config["hangingnodes"] =
        ConfigOption(false, "",
                     "Split the edges at the hanging nodes passed by a "
                     "previous module");
}

ProcessHAdaptation::~ProcessHAdaptation()
{
    m_mesh->PrintStats(m_log);
}

// NB! - At the moment the splitting uses ePolyevenlySpaced - THIS IS NOT
// OPTIMAL and could produce oscillations !!! Change to GLL points with
// ChildEdgess [-1 ; 0]  and [0 :1] !!!

void ProcessHAdaptation::Process()
{
    QuadEdgeToOpposite[0] = 2;
    QuadEdgeToOpposite[1] = 3;
    QuadEdgeToOpposite[2] = 0;
    QuadEdgeToOpposite[3] = 1;
    DefineMaps();

    // ----------------------------------------------------------------------------------------
    // START MAIN --------------------------------------------------------
    SpatialDomains::MeshGraphSharedPtr graph = m_mesh->m_meshGraph;
    m_dim                                    = graph->GetMeshDimension();
    order                                    = stoi(m_config["order"].value);

    RebuildLinks();

    // Create a map to access elements in a fast way
    for (auto &[element, tag] : m_mesh->m_elementTags[m_dim])
    {
        ID_ElementPositionMap[element->GetGlobalID()] = element;
    }

    // 1. Identify ElementsIDs to adapt (Read adaptfile)
    if (m_config["adaptfile"].beenSet == 1)
    {
        m_log(VERBOSE) << "Reading adaptfile = " << m_config["adaptfile"].value
                       << '\n';

        // Open the file
        ifstream input_file(m_config["adaptfile"].value);

        // Fill vector to adapt
        string ID;
        while (getline(input_file, ID))
        {
            if (ID_ElementPositionMap.find(stoi(ID)) ==
                ID_ElementPositionMap.end())
            {
                m_log(FATAL) << "Element " << ID << " not in mesh " << endl;
            }
            m_ElementsIDAdapt.push_back(stoi(ID));
        }

        // Close the file
        input_file.close();
    }
    else if (m_config["adaptfileEdges"].beenSet == 1)
    {
        m_log(VERBOSE)
            << "Identification of elements to split comes from the Edges"
            << endl;
    }
    else if (m_config["uniform"].beenSet == 1)
    {
        for (auto &[el, tag] : m_mesh->m_elementTags[m_dim])
        {
            if (el->GetShapeType() == LibUtilities::eTriangle)
            { // we avoid splitting Quads directly for the moment
                m_ElementsIDAdapt.push_back(el->GetGlobalID());
            }
        }
    }
    else if (m_config["areafile"].beenSet == 1)
    {
        m_log(VERBOSE) << "Area file [x y z A_desired] with interpolation "
                       << endl;

        // Read the areafile with desired area
        GetDesiredAreaField();
        // Interpolate the desired area field to the elements & Select the
        // elements that need splitting
        InterpolateElementToAdapt();
    }
    else if (m_config["hangingnodes"].beenSet == 1)
    {
        auto *hangingNodes = m_mesh->GetContext().TryGet<EdgeToHangingNodes>();
        if (hangingNodes == nullptr)
        {
            m_log(FATAL) << "No hanging nodes passed by a previous module"
                         << endl;
        }

        for (auto &[edge, nodes] : hangingNodes->nodes)
        {
            m_EdgesToSplit.push_back(edge);
            for (auto &[element, localId] : graph->GetElementsFromEdge(edge))
            {
                if (element->GetShapeType() == LibUtilities::eQuadrilateral)
                {
                    m_ElementsIDAdapt.push_back(element->GetGlobalID());
                }
            }
        }
    }
    else
    {
        m_log(FATAL) << "No adaptfile or uniform settings given " << endl;
    }
    i_max = m_ElementsIDAdapt.size();

    if (m_config["grading"].beenSet == 1)
    {
        int tmp = m_ElementsIDAdapt.size();
        for (int i = 0; i < tmp; i++)
        {
            int ID                            = m_ElementsIDAdapt[i];
            SpatialDomains::Geometry *element = ID_ElementPositionMap[ID];
            // calculate area of the element
            SpatialDomains::PointGeom *v0, *v1, *v2;
            v0 = element->GetVertex(0);
            v1 = element->GetVertex(1);
            v2 = element->GetVertex(2);

            [[maybe_unused]] NekDouble areaNow =
                0.5 *
                (v0->x() * (v1->y() - v2->y()) + v1->x() * (v2->y() - v0->y()) +
                 v2->x() * (v0->y() - v1->y()));

            for (int e = 0; e < element->GetNumEdges(); e++)
            {
                auto edge =
                    static_cast<SpatialDomains::SegGeom *>(element->GetEdge(e));
                if (graph->GetElementsFromEdge(edge).size() != 2)
                {
                    continue;
                }

                if (element == graph->GetElementsFromEdge(edge)[0].first)
                {
                }
                else
                {
                }

                // ElementSharedPtr elementAdj =
                // EdgeID_Edge[stoi(ID)]->m_elLink[i].first.lock();
                //  calculate area adj
            }
        }
    }

    m_log(VERBOSE) << "1. Identify ElementsIDs Done " << endl;
    // 2. Identify Edges to Split

    if (m_config["adaptfileEdges"].beenSet == 1)
    {
        // create a map of <EdgeIDs , EdgeSharedPtr >  that belong to this
        // partition
        m_log(VERBOSE) << "adaptfileEdges " << endl;
        unordered_map<int, SpatialDomains::SegGeom *> EdgeID_Edge;
        for (auto &[key, edge] : m_mesh->m_edgeSet)
        {
            EdgeID_Edge[edge->GetGlobalID()] = edge;
        }
        m_log(VERBOSE) << "Reading adaptfileEdges = "
                       << m_config["adaptfileEdges"].value << '\n';

        // Open the file
        ifstream input_file(m_config["adaptfileEdges"].value);

        // Fill vector to adapt
        string ID;
        while (getline(input_file, ID))
        {
            if (EdgeID_Edge.find(stoi(ID)) == EdgeID_Edge.end())
            {
                m_log(FATAL) << "Edge " << ID
                             << " in adaptfileEdges not in mesh" << endl;
            }
            m_EdgesToSplit.push_back(EdgeID_Edge[stoi(ID)]);
            for (int i = 0;
                 i < graph->GetElementsFromEdge(EdgeID_Edge[stoi(ID)]).size();
                 i++)
            {
                SpatialDomains::Geometry *element =
                    graph->GetElementsFromEdge(EdgeID_Edge[stoi(ID)])[i].first;
                m_ElementsIDAdapt.push_back(element->GetGlobalID());

                if (m_config["finest"].beenSet == 1)
                {
                    for (int e = 0; e < element->GetNumEdges(); e++)
                    {
                        auto edge = static_cast<SpatialDomains::SegGeom *>(
                            element->GetEdge(e));
                        m_EdgesToSplit.push_back(
                            EdgeID_Edge[edge->GetGlobalID()]);
                        for (int i = 0;
                             i < graph->GetElementsFromEdge(edge).size(); i++)
                        {
                            SpatialDomains::Geometry *elementAdajent =
                                graph->GetElementsFromEdge(edge)[i].first;
                            m_ElementsIDAdapt.push_back(
                                elementAdajent->GetGlobalID());
                        }
                    }
                }
            }
        }

        // Close the file
        input_file.close();

        m_log(VERBOSE) << "Elements to split(repeating) = "
                       << m_ElementsIDAdapt.size() << '\n';
        m_log(VERBOSE) << "Edges to split  = " << m_EdgesToSplit.size() << '\n';
    }
    else if (m_config["hangingnodes"].beenSet == 0)
    {
        IdentifyEdgeToSplit();
    }

    m_log(VERBOSE) << "2. Identify Edges To Split Done " << endl;

    // 3. Split The Edges Isoparametrically

    if (m_config["hangingnodes"].beenSet != 1)
    {
        SplitEdges();
    }
    else
    {
        SplitEdgesHangingNodes();
    }

    m_log(VERBOSE) << "3. Start Splitting Edges" << endl;

    // 4. Split the elements & create new ones
    if (m_config["hangingnodes"].beenSet != 1)
    {
        SplitElements();
    }
    else
    {
        SplitElementsHangingNodes();
    }

    // 5. Swap through the elements that are marked for adaptation
    m_log(VERBOSE) << "Start swapping edges" << endl;
    if (m_config["swap"].beenSet == 1)
    {
        // ProcessEdges(); // Plenty of work ! - Requires the 1D Elements to be
        // ProcessElements();
        // ProcessComposites();

        RebuildLinks();
        SwapEdge();
    }

    // 6. Some extras
    // if r-adaptation needs to performed after h - we export scale field

    if (m_config["hr"].beenSet == true)
    {
        NekDouble r_scaling = 0.5;

        ofstream output_file1;
        output_file1.open("r-scales.txt", ios::app);
        output_file1 << setprecision(12);

        // All centers of the elements need x/y/z/scale
        for (auto &[element, tag] : m_mesh->m_elementTags[m_dim])
        {
            vector<NekDouble> loc = {0.0, 0.0, 0.0};
            for (int j = 0; j < element->GetNumVerts(); j++)
            {
                loc[0] += element->GetVertex(j)->x();
                loc[1] += element->GetVertex(j)->y();
                // loc[2] +=
            }

            loc[0] = loc[0] / element->GetNumVerts();
            loc[1] = loc[1] / element->GetNumVerts();

            // m_adapted
            output_file1 << loc[0] << " " << loc[1] << " " << 0 << " ";
            if (m_adapted.count(element) == 0)
            {
                output_file1 << 1.0 << '\n';
            }
            else
            {
                output_file1 << r_scaling << '\n';
            }
        }

        // 4. close the file
        output_file1.close();
    }

    // Multilevel adaptation - TO DO smarter inside the Process
    if (m_config["multilevel"].beenSet == true)
    {
        ofstream output_file;
        output_file.open("AdaptID_Serial.txt", ios::app);

        for (auto &[element, tag] : m_mesh->m_elementTags[m_dim])
        {
            if (m_adapted.count(element))
            {
                output_file << std::setw(14) << element->GetGlobalID() << '\n';
            }
        }
        // 4. close the file
        output_file.close();
    }

    RemoveOrphanedEntities();
    RebuildLinks();

    ProcessElements();
    ProcessComposites();
}

// 1. Size interpolation based on a file desired area value
void ProcessHAdaptation::GetDesiredAreaField()
{
    // 1. Import Desired Area Open the file
    ifstream input_file1(m_config["areafile"].value);
    NekDouble x, y, z, A;
    vector<vector<NekDouble>> xyzA_file(4);

    string dataline1;
    while (getline(input_file1, dataline1))
    {
        // Skip lines with wrong format non [x y z A] or blank
        istringstream line(dataline1);
        if (!(line >> x >> y >> z >> A))
        {
            continue;
        }
        xyzA_file[0].push_back(x);
        xyzA_file[1].push_back(y);
        xyzA_file[2].push_back(z);
        xyzA_file[3].push_back(A);
    }

    inPts = Array<OneD, Array<OneD, NekDouble>>(4);
    for (int i = 0; i < 4; i++)
    {
        inPts[i] = Array<OneD, NekDouble>(xyzA_file[0].size());
        for (int j = 0; j < xyzA_file[0].size(); ++j)
        {
            inPts[i][j] = xyzA_file[i][j];
        }
    }

    // Close the file
    input_file1.close();
    m_log(VERBOSE) << "Input file with Areas read succesfully " << endl;
}

void ProcessHAdaptation::InterpolateElementToAdapt()
{
    m_log(WARNING) << "There seems to be a bug in the AreaSize file function"
                   << endl;
    // 2.0 Interpolate to every element center
    LibUtilities::Interpolator interp        = GetField(inPts);
    SpatialDomains::MeshGraphSharedPtr graph = m_mesh->m_meshGraph;

    for (auto &[element, tag] : m_mesh->m_elementTags[m_dim])
    {
        if (element->GetShapeType() == LibUtilities::eQuadrilateral)
        {
            m_log(FATAL)
                << " The Area based h-adaptation with quads hasn't been tested "
                   "- comment out and proceed with caution "
                << endl;
        }
        // 2.1 Calculate the Desired Area
        NekDouble areaDesired = 1.0;
        if (interp.GetInField())
        {
            if (!m_interpField)
            {
                Array<OneD, Array<OneD, NekDouble>> centre(4);
                for (int i = 0; i < 4; ++i)
                {
                    centre[i] = Array<OneD, NekDouble>(1, 0.0);
                }

                vector<string> fieldNames;
                fieldNames.push_back("");

                map<LibUtilities::PtsInfo, int> ptsInfo =
                    LibUtilities::NullPtsInfoMap;

                m_interpField =
                    MemoryManager<LibUtilities::PtsField>::AllocateSharedPtr(
                        3, fieldNames, centre, ptsInfo);
            }
            vector<SpatialDomains::PointGeom *> nodes;
            for (int i = 0; i < element->GetNumVerts(); ++i)
            {
                nodes.push_back(element->GetVertex(i));
            }

            vector<NekDouble> centre(3, 0.0);
            for (int i = 0; i < nodes.size(); ++i)
            {
                centre[0] += nodes[i]->x();
                centre[1] += nodes[i]->y();
                centre[2] += nodes[i]->z();
            }

            m_interpField->SetPointVal(0, 0, centre[0] / nodes.size());
            m_interpField->SetPointVal(1, 0, centre[1] / nodes.size());
            m_interpField->SetPointVal(2, 0, centre[2] / nodes.size());

            interp.CalcWeights(interp.GetInField(), m_interpField, true);
            interp.Interpolate(interp.GetInField(), m_interpField);

            areaDesired = m_interpField->GetPointVal(3, 0);
        }

        // 2.2 Calculate the area of the current element
        // TO DO in STD space with getGeom ( NOW Straight sided )
        NekDouble areaNow;
        if (element->GetShapeType() == LibUtilities::eTriangle)
        {
            // Assume straight sided mesh
            SpatialDomains::PointGeom *v0, *v1, *v2;
            v0 = element->GetVertex(0);
            v1 = element->GetVertex(1);
            v2 = element->GetVertex(2);

            areaNow = 0.5 * (v0->x() * (v1->y() - v2->y()) +
                             v1->x() * (v2->y() - v0->y()) +
                             v2->x() * (v0->y() - v1->y()));
            // 2.3 Decide to adapt or not the element
            if (areaNow > areaDesired * 1.75)
            {
                m_ElementsIDAdapt.push_back(element->GetGlobalID());

                // Check the elements around if they are too large - if so -
                // split them as well
                for (int e = 0; e < element->GetNumEdges(); e++)
                {
                    auto edge = static_cast<SpatialDomains::SegGeom *>(
                        element->GetEdge(e));
                    for (auto link_el : graph->GetElementsFromEdge(edge))
                    {
                        if (link_el.first->GetShapeType() !=
                            LibUtilities::eTriangle)
                        {
                            continue;
                        }
                        SpatialDomains::PointGeom *v0 =
                            link_el.first->GetVertex(0);
                        SpatialDomains::PointGeom *v1 =
                            link_el.first->GetVertex(1);
                        SpatialDomains::PointGeom *v2 =
                            link_el.first->GetVertex(2);
                        NekDouble area = 0.5 * (v0->x() * (v1->y() - v2->y()) +
                                                v1->x() * (v2->y() - v0->y()) +
                                                v2->x() * (v0->y() - v1->y()));
                        if (area > 1.4 * areaNow)
                        {
                            m_ElementsIDAdapt.push_back(
                                link_el.first->GetGlobalID());
                        }
                    }
                }
            }
        }
        else
        {
            m_log(FATAL) << "interpolation not implemented to this shape "
                         << endl;
        }
    }
}

// Create an interpolation function
LibUtilities::Interpolator ProcessHAdaptation::GetField(
    Array<OneD, Array<OneD, NekDouble>> inPts)
{
    int dim = 3;

    vector<string> fieldNames;
    fieldNames.push_back("");

    map<LibUtilities::PtsInfo, int> ptsInfo = LibUtilities::NullPtsInfoMap;

    PtsFieldSharedPtr inField =
        MemoryManager<LibUtilities::PtsField>::AllocateSharedPtr(
            dim, fieldNames, inPts, ptsInfo);

    Array<OneD, Array<OneD, NekDouble>> dummyPts(dim + 1);
    for (int i = 0; i < dim + 1; ++i)
    {
        dummyPts[i] = Array<OneD, NekDouble>(0);
    }

    PtsFieldSharedPtr dummyField =
        MemoryManager<LibUtilities::PtsField>::AllocateSharedPtr(
            dim, fieldNames, dummyPts, ptsInfo);
    LibUtilities::Interpolator ret;
    ret.Interpolate(inField, dummyField);

    return ret;
}

// 2. Identify Edges ToSplit
void ProcessHAdaptation::IdentifyEdgeToSplit()
{
    SpatialDomains::MeshGraphSharedPtr graph = m_mesh->m_meshGraph;
    NekDouble h_min = std::numeric_limits<double>::max();
    if (m_config["h_min"].beenSet == 1)
    {
        h_min = stof(m_config["h_min"].value);
        m_log(VERBOSE) << " h_min  = " << h_min << endl;
    }

    // If Quad - split in a stack prism line so propagate the opposite one until
    // a non-quad element or bnd
    auto PropagateQuads = [&](SpatialDomains::Geometry *el,
                              SpatialDomains::SegGeom *edge) {
        while (el->GetShapeType() == LibUtilities::eQuadrilateral)
        {
            int EdgeLocalID = -1;
            for (int k = 0; k < el->GetNumEdges(); k++)
            {
                if (el->GetEdge(k) == edge)
                {
                    EdgeLocalID = k;
                    break;
                }
            }

            edge = static_cast<SpatialDomains::SegGeom *>(
                el->GetEdge(QuadEdgeToOpposite[EdgeLocalID]));
            m_EdgesToSplit.push_back(edge);

            if (graph->GetElementsFromEdge(edge).size() != 2)
            {
                break;
            }
            el = graph->GetElementsFromEdge(edge)[0].first == el
                     ? graph->GetElementsFromEdge(edge)[1].first
                     : graph->GetElementsFromEdge(edge)[0].first;
            m_ElementsIDAdapt.push_back(el->GetGlobalID());
        }
    };

    int i_max = m_ElementsIDAdapt.size();
    for (int i = 0; i < i_max; i++)
    {
        int ID = m_ElementsIDAdapt[i];
        // if(ID >= m_mesh->m_element[m_dim].size())
        // {
        //     m_log(FATAL) << "ID to split > m_element[m_dim].size() - need to
        //     find a way to call the element with this ID efficiently ID =" <<
        //     ID << endl  ;
        // }

        if (ID != ID_ElementPositionMap[ID]->GetGlobalID())
        {
            // Requirement
            m_log(WARNING)
                << "At the moment this module requires XML with sorted IDs. "
                << ID << " " << ID_ElementPositionMap[ID]->GetGlobalID()
                << " - Run an empty NekMesh module with ProcessElements() - "
                   "BEFORE you SIMULATE your first run."
                << endl;
        }

        SpatialDomains::Geometry *element = ID_ElementPositionMap[ID];

        // Use different rules to define the Edges to be split per shapetype
        if (element->GetShapeType() ==
            LibUtilities::eTriangle) // Triags - Largest edge
        {
            // Add min area as a criteria
            // Assume straight sided mesh
            SpatialDomains::PointGeom *v0, *v1, *v2;
            v0 = element->GetVertex(0);
            v1 = element->GetVertex(1);
            v2 = element->GetVertex(2);

            NekDouble area = 0.5 * (v0->x() * (v1->y() - v2->y()) +
                                    v1->x() * (v2->y() - v0->y()) +
                                    v2->x() * (v0->y() - v1->y()));

            // Apply the minimum size condition
            if (m_config["h_min"].beenSet == 1)
            {
                NekDouble h_el_min = std::numeric_limits<double>::max();
                for (int e = 0; e < element->GetNumEdges(); e++)
                {
                    auto edge = static_cast<SpatialDomains::SegGeom *>(
                        element->GetEdge(e));
                    if (h_el_min >
                        edge->GetVertex(0)->dist(*edge->GetVertex(1)))
                    {
                        h_el_min =
                            edge->GetVertex(0)->dist(*edge->GetVertex(1));
                    }
                }

                if (h_el_min < h_min || area < h_min * h_min / 2.0)
                {
                    continue;
                }
            }

            // Finest Level, if spefied
            if (m_config["finest"].beenSet == 1)
            {
                for (int e = 0; e < element->GetNumEdges(); e++)
                {
                    auto edge = static_cast<SpatialDomains::SegGeom *>(
                        element->GetEdge(e));
                    m_EdgesToSplit.push_back(edge);

                    // Add the conformity mirrored element IF
                    if (graph->GetElementsFromEdge(edge).size() == 2)
                    {

                        /// Additional work for ensuring
                        // Check the length of the new patch
                        NekDouble edgeLength =
                            edge->GetVertex(0)->dist(*edge->GetVertex(1));
                        vector<NekDouble> locS(3);
                        locS[0] = (edge->GetVertex(0)->x() +
                                   edge->GetVertex(1)->x()) /
                                  2.0;
                        locS[1] = (edge->GetVertex(0)->y() +
                                   edge->GetVertex(1)->y()) /
                                  2.0;
                        locS[2] = (edge->GetVertex(0)->z() +
                                   edge->GetVertex(1)->z()) /
                                  2.0;
                        SpatialDomains::PointGeom S(graph->GetSpaceDimension(),
                                                    0, locS[0], locS[1],
                                                    locS[2]);
                        NekDouble coef = 2.5 / 2.0;

                        // main job is to propagate the splitting
                        SpatialDomains::Geometry *adjacentEl;
                        if (element->GetGlobalID() ==
                            graph->GetElementsFromEdge(edge)[0]
                                .first->GetGlobalID())
                        {
                            adjacentEl =
                                graph->GetElementsFromEdge(edge)[1].first;
                            m_ElementsIDAdapt.push_back(
                                adjacentEl->GetGlobalID());
                        }
                        else
                        {
                            adjacentEl =
                                graph->GetElementsFromEdge(edge)[0].first;
                            m_ElementsIDAdapt.push_back(
                                adjacentEl->GetGlobalID());
                        }

                        // find the further away vertex to calculate max h for
                        // smoothing out the size split (EXTRA NOT NECESSARY)
                        SpatialDomains::PointGeom *v1 = nullptr;
                        for (int k = 0; k < adjacentEl->GetNumVerts(); k++)
                        {
                            SpatialDomains::PointGeom *vertex =
                                adjacentEl->GetVertex(k);
                            if (vertex != edge->GetVertex(0) &&
                                vertex != edge->GetVertex(1))
                            {
                                v1 = vertex;
                            }
                        }

                        if (v1->dist(S) > coef * edgeLength &&
                            adjacentEl->GetShapeType() ==
                                LibUtilities::eTriangle)
                        {
                            // add the adjacent of the adjacent for refinement
                            for (int k = 0; k < adjacentEl->GetNumEdges(); k++)
                            {
                                auto adjacentEdge =
                                    static_cast<SpatialDomains::SegGeom *>(
                                        adjacentEl->GetEdge(k));
                                m_EdgesToSplit.push_back(adjacentEdge);
                                if (graph->GetElementsFromEdge(adjacentEdge)
                                        .size() == 2)
                                {
                                    SpatialDomains::Geometry *otherAdjEl;
                                    if (adjacentEl->GetGlobalID() ==
                                        graph
                                            ->GetElementsFromEdge(
                                                adjacentEdge)[0]
                                            .first->GetGlobalID())
                                    {
                                        otherAdjEl = graph
                                                         ->GetElementsFromEdge(
                                                             adjacentEdge)[1]
                                                         .first;
                                    }
                                    else
                                    {
                                        otherAdjEl = graph
                                                         ->GetElementsFromEdge(
                                                             adjacentEdge)[0]
                                                         .first;
                                    }
                                    m_ElementsIDAdapt.push_back(
                                        otherAdjEl->GetGlobalID());
                                    PropagateQuads(otherAdjEl, adjacentEdge);
                                }
                            }
                        }

                        // If Quad - propagate down further
                        bool adaptfurther = true;
                        while (adjacentEl->GetShapeType() ==
                                   LibUtilities::eQuadrilateral &&
                               adaptfurther == true)
                        {

                            // find the element local edge ID
                            int EdgeLocalID = -1;
                            for (int k = 0; k < adjacentEl->GetNumEdges(); k++)
                            {
                                if (adjacentEl->GetEdge(k) == edge)
                                {
                                    EdgeLocalID = k;
                                    break;
                                }
                            }

                            // Use the local Quad map to find the opposite edge
                            // of the Quad to be split
                            edge = static_cast<SpatialDomains::SegGeom *>(
                                adjacentEl->GetEdge(
                                    QuadEdgeToOpposite[EdgeLocalID]));
                            m_EdgesToSplit.push_back(edge);

                            // decide if Boundary Surface is reached
                            if (graph->GetElementsFromEdge(edge).size() == 2)
                            {
                                if (adjacentEl->GetGlobalID() ==
                                    graph->GetElementsFromEdge(edge)[0]
                                        .first->GetGlobalID())
                                {
                                    adjacentEl =
                                        graph->GetElementsFromEdge(edge)[1]
                                            .first;
                                }
                                else
                                {
                                    adjacentEl =
                                        graph->GetElementsFromEdge(edge)[0]
                                            .first;
                                }
                                m_ElementsIDAdapt.push_back(
                                    adjacentEl->GetGlobalID());
                            }
                            else
                            {
                                adaptfurther = false;
                            }
                        }
                    }
                }
            }
            else
            {
                // Longest Edge
                NekDouble len                             = 0;
                SpatialDomains::SegGeom *EdgeToSplitTriag = nullptr;
                for (int e = 0; e < element->GetNumEdges(); e++)
                {
                    auto edge = static_cast<SpatialDomains::SegGeom *>(
                        element->GetEdge(e));
                    NekDouble leni = edge->GetVertex(0)->dist(*edge->GetVertex(
                        1)); // THIS assumes linear length of the edge !
                    if (len < leni)
                    {
                        EdgeToSplitTriag = edge;
                        len              = leni;
                    }
                }
                // Add to the vector of edges to be split
                m_EdgesToSplit.push_back(EdgeToSplitTriag);
                // Add the conformity mirrored element IF QUAD
                if (graph->GetElementsFromEdge(EdgeToSplitTriag).size() == 2)
                {
                    SpatialDomains::Geometry *el_adjacent;
                    if (element->GetGlobalID() ==
                        graph->GetElementsFromEdge(EdgeToSplitTriag)[0]
                            .first->GetGlobalID())
                    {
                        el_adjacent =
                            graph->GetElementsFromEdge(EdgeToSplitTriag)[1]
                                .first;
                    }
                    else
                    {
                        el_adjacent =
                            graph->GetElementsFromEdge(EdgeToSplitTriag)[0]
                                .first;
                    }

                    m_ElementsIDAdapt.push_back(el_adjacent->GetGlobalID());
                    bool adaptfurther = true;
                    while (el_adjacent->GetShapeType() ==
                               LibUtilities::eQuadrilateral &&
                           adaptfurther == true)
                    {

                        // find the element local edge ID
                        int EdgeLocalID = -1;
                        for (int k = 0; k < el_adjacent->GetNumEdges(); k++)
                        {
                            if (el_adjacent->GetEdge(k) == EdgeToSplitTriag)
                            {
                                EdgeLocalID = k;
                                break;
                            }
                        }

                        // Use the local Quad map to find the opposite edge of
                        // the Quad to be split
                        EdgeToSplitTriag =
                            static_cast<SpatialDomains::SegGeom *>(
                                el_adjacent->GetEdge(
                                    QuadEdgeToOpposite[EdgeLocalID]));
                        m_EdgesToSplit.push_back(EdgeToSplitTriag);

                        // decide if Boundary Surface is reached
                        if (graph->GetElementsFromEdge(EdgeToSplitTriag)
                                .size() == 2)
                        {
                            if (el_adjacent->GetGlobalID() ==
                                graph->GetElementsFromEdge(EdgeToSplitTriag)[0]
                                    .first->GetGlobalID())
                            {
                                el_adjacent = graph
                                                  ->GetElementsFromEdge(
                                                      EdgeToSplitTriag)[1]
                                                  .first;
                            }
                            else
                            {
                                el_adjacent = graph
                                                  ->GetElementsFromEdge(
                                                      EdgeToSplitTriag)[0]
                                                  .first;
                            }
                            m_ElementsIDAdapt.push_back(
                                el_adjacent->GetGlobalID());
                        }
                        else
                        {
                            adaptfurther = false;
                        }
                    }
                }
            }
        }
        else if (element->GetShapeType() == LibUtilities::eQuadrilateral)
        {
            // Just propagate the conformity at the moment - not full Quad based
            // mesh refinement
            m_log(WARNING) << "Only h-adaptation with triangles is implemented "
                              "at the moment"
                           << endl;
        }
        else
        {
            m_log(FATAL) << "3D not implemented";
        }
    }

    m_log(VERBOSE) << "Elements to split(repeating) = "
                   << m_ElementsIDAdapt.size() << '\n';
    m_log(VERBOSE) << "Edges to split  = " << m_EdgesToSplit.size() << '\n';
}

// 3. Split Edges Isoparametrically
void ProcessHAdaptation::SplitEdges()
{
    SpatialDomains::MeshGraphSharedPtr graph = m_mesh->m_meshGraph;
    for (auto edge : m_EdgesToSplit)
    {
        if (m_hanginNode.count(edge))
        {
            continue;
        }

        SpatialDomains::PointGeom *S;
        vector<SpatialDomains::PointGeom *> NewEdge1Nodes, NewEdge2Nodes;
        if (edge->GetCurve() == nullptr)
        {
            // Split straight edge
            Array<OneD, NekDouble> loc(3);

            loc[0] = (edge->GetVertex(0)->x() + edge->GetVertex(1)->x()) / 2.0;
            loc[1] = (edge->GetVertex(0)->y() + edge->GetVertex(1)->y()) / 2.0;
            loc[2] = (edge->GetVertex(0)->z() + edge->GetVertex(1)->z()) / 2.0;

            S = graph->CreatePointGeom(graph->GetSpaceDimension(),
                                       NextPointId(graph), loc[0], loc[1],
                                       loc[2]);
        }
        else
        {
            // Split Isoparametrically
            SpatialDomains::Geometry *geom = edge;
            geom->FillGeom();

            StdRegions::StdExpansion *xmap       = geom->GetXmap();
            Array<OneD, const NekDouble> coeffs0 = geom->GetCoeffs(0);
            Array<OneD, const NekDouble> coeffs1 = geom->GetCoeffs(1);

            Array<OneD, NekDouble> xc(xmap->GetTotPoints());
            Array<OneD, NekDouble> yc(xmap->GetTotPoints());

            xmap->BwdTrans(coeffs0, xc);
            xmap->BwdTrans(coeffs1, yc);

            // Create the new vertex S + Add it to the edge HangingNode
            Array<OneD, NekDouble> xp(1);
            xp[0] = 0.0;
            Array<OneD, NekDouble> loc(3);
            loc[0] = xmap->PhysEvaluate(xp, xc);
            loc[1] = xmap->PhysEvaluate(xp, yc);
            loc[2] = 0.0;

            S = graph->CreatePointGeom(graph->GetSpaceDimension(),
                                       NextPointId(graph), loc[0], loc[1],
                                       loc[2]);

            LibUtilities::PointsKey ekey(
                2 * order + 1,
                LibUtilities::ePolyEvenlySpaced); // eGaussLobattoLegendre //
                                                  // WRONG OSCILLATIONS !!!!!!
            Array<OneD, NekDouble> gll;
            LibUtilities::PointsManager()[ekey]->GetPoints(gll);

            for (int i = 1; i < order; i++)
            {
                xp[0] = gll[i];

                loc[0] = xmap->PhysEvaluate(xp, xc);
                loc[1] = xmap->PhysEvaluate(xp, yc);
                loc[2] = 0.0;

                SpatialDomains::PointGeom *edgeNode = graph->CreateCurveNode(
                    graph->GetSpaceDimension(), 0, loc[0], loc[1], loc[2]);
                NewEdge1Nodes.push_back(edgeNode);

                xp[0] = gll[order + i];

                loc[0] = xmap->PhysEvaluate(xp, xc);
                loc[1] = xmap->PhysEvaluate(xp, yc);
                loc[2] = 0.0;

                edgeNode = graph->CreateCurveNode(graph->GetSpaceDimension(), 0,
                                                  loc[0], loc[1], loc[2]);
                NewEdge2Nodes.push_back(edgeNode);
            }
        }
        // Need to reverse the orientation of the EdgeNodes of NewEdge2 (my
        // convention edgeNew->m_n1 = edgeOld->m_n1/2)
        reverse(NewEdge2Nodes.begin(), NewEdge2Nodes.end());

        // New Vertex S global definition + Hanging Node
        m_hanginNode[edge] = S;
        m_childEdges[edge].push_back(
            CreateEdge(edge->GetVertex(0), S, NewEdge1Nodes,
                       LibUtilities::ePolyEvenlySpaced));
        m_childEdges[edge].push_back(
            CreateEdge(edge->GetVertex(1), S, NewEdge2Nodes,
                       LibUtilities::ePolyEvenlySpaced));
    }
    m_log(VERBOSE) << "Edge Split Completed " << endl;
    m_log(VERBOSE) << "Edges after EdgeSplit()= " << m_mesh->m_edgeSet.size()
                   << endl;
}

// 3b. Split Edges Isoparametrically at the hanging nodes of a previous module
void ProcessHAdaptation::SplitEdgesHangingNodes()
{
    m_log(FATAL) << " Hanging Nodes Come with the next MR." << endl;
}

// 4b. Split Quads at the hanging nodes of a previous module
void ProcessHAdaptation::SplitElementsHangingNodes()
{
    m_log(FATAL) << " Hanging Nodes Come with the next MR." << endl;
}

// 4. Split the elements according to the mapping

void ProcessHAdaptation::SplitElements()
{

    SpatialDomains::MeshGraphSharedPtr graph = m_mesh->m_meshGraph;

    LibUtilities::PointsKey ekey(order + 1,
                                 LibUtilities::eGaussLobattoLegendre);
    Array<OneD, NekDouble> gll;
    LibUtilities::PointsManager()[ekey]->GetPoints(gll);
    for (auto ID : m_ElementsIDAdapt)
    {
        SpatialDomains::Geometry *element = ID_ElementPositionMap[ID];
        if (m_adapted.count(element))
        {
            continue;
        }

        // Decide which local Edges and how many of them are split
        bool IsElementHO = false;

        // Create NodeListFull with all Vertices + Hanging Nodes
        vector<SpatialDomains::PointGeom *> NodeListFull;
        for (int i = 0; i < element->GetNumVerts(); i++)
        {
            NodeListFull.push_back(element->GetVertex(i));
        }

        vector<int> LocalEdgeIDsplit; // This would be the internal 0 -1 -2
                                      // index of the Element
        for (int i = 0; i < element->GetNumEdges(); i++)
        {
            auto edge =
                static_cast<SpatialDomains::SegGeom *>(element->GetEdge(i));
            if (edge->GetCurve() != nullptr)
            {
                IsElementHO = true;
            }
            if (m_hanginNode.count(edge))
            {
                LocalEdgeIDsplit.push_back(i);

                NodeListFull.push_back(m_hanginNode[edge]);
            }
        }

        int N_edgesSplit = LocalEdgeIDsplit.size();
        int elTag        = m_mesh->m_elementTags[m_dim].find(element)->second;

        if (element->GetShapeType() == LibUtilities::eTriangle)
        {
            if (N_edgesSplit == 0)
            {
                continue;
            }

            // Specify the mapping
            if (SplitTriangleNodeList.find(LocalEdgeIDsplit) ==
                SplitTriangleNodeList.end())
            {
                m_log(FATAL)
                    << "No triangle mapping for the split edges" << endl;
            }
            const vector<int> &NodeListMaping =
                SplitTriangleNodeList.at(LocalEdgeIDsplit);
            const vector<int> &EdgeNodeListMapping =
                NewInternalEdgeNodeList.at(LocalEdgeIDsplit);
            const vector<NekDouble> &StdLocMapping =
                StdEdgeToStdLocHangingNode.at(LocalEdgeIDsplit);
            vector<SpatialDomains::SegGeom *>
                InternalEdges; // Necessary for SetEdges()
            if (IsElementHO == true)
            {
                // Create the HO internal EdgeNodes ;
                // Create the Tensor Product and the mapping to Std
                SpatialDomains::Geometry *geom = element;
                geom->FillGeom();

                StdRegions::StdExpansion *xmap       = geom->GetXmap();
                Array<OneD, const NekDouble> coeffs0 = geom->GetCoeffs(0);
                Array<OneD, const NekDouble> coeffs1 = geom->GetCoeffs(1);

                Array<OneD, NekDouble> xc(xmap->GetTotPoints());
                Array<OneD, NekDouble> yc(xmap->GetTotPoints());

                xmap->BwdTrans(coeffs0, xc);
                xmap->BwdTrans(coeffs1, yc);

                // Create New Internal Edges
                for (int i = 0; i < LocalEdgeIDsplit.size(); i++)
                {
                    vector<SpatialDomains::PointGeom *> edgeNodes;
                    vector<NekDouble> uvb(2), uve(2);
                    Nektar::Array<Nektar::OneD, Nektar::NekDouble> xp(2);
                    // Use the mapppings to check the coordinates of the new
                    // vertices to popiulate xp[0] xp[1] of every EdgeNode on
                    // the edge
                    uvb[0] = StdLocMapping[4 * i];
                    uvb[1] = StdLocMapping[4 * i + 1];

                    uve[0] = StdLocMapping[4 * i + 2];
                    uve[1] = StdLocMapping[4 * i + 3];

                    for (int k = 1; k < order; k++)
                    {

                        xp[0] = uvb[0] * (1.0 - gll[k]) / 2.0 +
                                uve[0] * (1.0 + gll[k]) / 2.0;
                        xp[1] = uvb[1] * (1.0 - gll[k]) / 2.0 +
                                uve[1] * (1.0 + gll[k]) / 2.0;

                        Array<OneD, NekDouble> loc(3);
                        loc[0] = xmap->PhysEvaluate(xp, xc);
                        loc[1] = xmap->PhysEvaluate(xp, yc);
                        loc[2] = 0.0;

                        SpatialDomains::PointGeom *EdgeNode =
                            graph->CreateCurveNode(graph->GetSpaceDimension(),
                                                   0, loc[0], loc[1], loc[2]);
                        edgeNodes.push_back(EdgeNode);
                    }

                    SpatialDomains::SegGeom *NewEdge = CreateEdge(
                        NodeListFull[EdgeNodeListMapping[2 * i]],
                        NodeListFull[EdgeNodeListMapping[2 * i + 1]], edgeNodes,
                        LibUtilities::eGaussLobattoLegendre);

                    InternalEdges.push_back(NewEdge);
                }
            }
            else
            {
                // Create New Internal Edges
                for (int i = 0; i < LocalEdgeIDsplit.size(); i++)
                {
                    vector<SpatialDomains::PointGeom *> edgeNodes;
                    CreateEdge(NodeListFull[EdgeNodeListMapping[2 * i]],
                               NodeListFull[EdgeNodeListMapping[2 * i + 1]],
                               edgeNodes, LibUtilities::ePolyEvenlySpaced);
                }
            }

            // Create the New Elements
            ElmtConfig conf(LibUtilities::eTriangle, 1, false, false, true);
            for (int i = 0; i <= LocalEdgeIDsplit.size(); i++)
            {
                vector<SpatialDomains::PointGeom *> nodeList(3);
                for (int j = 0; j < 3; j++)
                {
                    nodeList[j] = NodeListFull[NodeListMaping[3 * i + j]];
                }

                SpatialDomains::Geometry *elmtNew =
                    GetElementFactory().CreateInstance(
                        LibUtilities::eTriangle, nodeList, graph,
                        m_mesh->m_edgeSet, m_mesh->m_faceSet, conf, nullptr,
                        nullptr, nullptr, nullptr);

                m_adapted.insert(elmtNew);

                // Populate the new split elements to the m_mesh
                m_mesh->m_elementTags[m_dim][elmtNew] = elTag;
            }

            // If h-r adaptation required add the scaling file
        }
        else if (element->GetShapeType() == LibUtilities::eQuadrilateral)
        {
            if (N_edgesSplit == 0)
            {
                continue;
            }
            if (N_edgesSplit != 2)
            {
                m_log(WARNING) << "Quadrilateral element with N_edgesSplit = "
                               << N_edgesSplit << endl;
                m_log(FATAL) << "Only Conformal Quad Splitting for Triag-Quad "
                                "propagation was implemented - something is "
                                "wrong with it N_edgesSplit!=2"
                             << endl;
            }

            // Find map on opposite edge on the Quad
            int map = -1;
            if (LocalEdgeIDsplit[0] == 0 && LocalEdgeIDsplit[1] == 2)
            {
                // case 0 - 02
                map = 0;
            }
            else if (LocalEdgeIDsplit[0] == 1 && LocalEdgeIDsplit[1] == 3)
            {
                // case 1 - 13
                map = 1;
            }
            else
            {
                m_log(FATAL) << "No quad map available: " << LocalEdgeIDsplit[0]
                             << " " << LocalEdgeIDsplit[1] << endl;
            }

            // NodeListFull = [ MacroVertices, Hanging Nodes ] - the hanging
            // nodes are already in it from the edge loop

            //  Create the HO Isoparametric Internal edge between the hanging
            //  nodes, at the middle of the split edges
            if (IsElementHO == true)
            {
                // Std location of the hanging nodes pm the Quad - xi1, xi2;
                // end node xi1, end xi2
                const NekDouble stdLoc[2][4] = {{0.0, -1.0, 0.0, 1.0},
                                                {1.0, 0.0, -1.0, 0.0}};

                vector<SpatialDomains::PointGeom *> edgeNodes;
                Array<OneD, NekDouble> xp(2);
                for (int k = 1; k < order; k++)
                {
                    xp[0] = stdLoc[map][0] * (1.0 - gll[k]) / 2.0 +
                            stdLoc[map][2] * (1.0 + gll[k]) / 2.0;
                    xp[1] = stdLoc[map][1] * (1.0 - gll[k]) / 2.0 +
                            stdLoc[map][3] * (1.0 + gll[k]) / 2.0;

                    edgeNodes.push_back(graph->CreateCurveNode(
                        graph->GetSpaceDimension(), 0, element->GetCoord(0, xp),
                        element->GetCoord(1, xp), 0.0));
                }

                CreateEdge(NodeListFull[4], NodeListFull[5], edgeNodes,
                           LibUtilities::eGaussLobattoLegendre);
            }

            // Create the new elements
            ElmtConfig conf(LibUtilities::eQuadrilateral, 1, false, false,
                            true);
            for (int j = 0; j < LocalEdgeIDsplit.size(); j++)
            {
                vector<SpatialDomains::PointGeom *> nodeList(4);
                for (int k = 0; k < 4; k++)
                {
                    int localId = SplitQuadNodeList[map][j][k];
                    nodeList[k] = NodeListFull[localId];
                }

                SpatialDomains::Geometry *elmtNew =
                    GetElementFactory().CreateInstance(
                        LibUtilities::eQuadrilateral, nodeList, graph,
                        m_mesh->m_edgeSet, m_mesh->m_faceSet, conf, nullptr,
                        nullptr, nullptr, nullptr);

                m_adapted.insert(elmtNew);

                // Populate the new split elements to the m_mesh
                m_mesh->m_elementTags[m_dim][elmtNew] = elTag;
            }
        }
        else
        {
            m_log(FATAL) << "Only 2D elements are implemented " << endl;
        }

        m_adapted.insert(element);
        m_mesh->m_elementTags[m_dim].erase(element);

        // Update Boundary Element 1D if there are boundary edges
        for (int j = 0; j < element->GetNumEdges(); ++j)
        {
            auto edgeOld =
                static_cast<SpatialDomains::SegGeom *>(element->GetEdge(j));
            auto element1D = m_mesh->m_elementTags[m_dim - 1].find(edgeOld);

            if (element1D == m_mesh->m_elementTags[m_dim - 1].end() ||
                !m_hanginNode.count(edgeOld))
            {
                continue;
            }

            int nComposite = element1D->second;

            m_mesh->m_elementTags[m_dim - 1].erase(edgeOld);
            m_mesh->m_elementTags[m_dim - 1][m_childEdges[edgeOld][0]] =
                nComposite;
            m_mesh->m_elementTags[m_dim - 1][m_childEdges[edgeOld][1]] =
                nComposite;
        }
    }
}

// 5. Define Isoparametric Swapping function for triangles
// At the moment it assumes all triangles are HO - this is WRONG
// NEED to add function for distinguishing StraightSided Couples
// as it is for the splitting for efficiency gains
void ProcessHAdaptation::SwapEdge()
{
    // Create a map for adaptation <El Id , Swaped >
    // This could be optimized !
    SpatialDomains::MeshGraphSharedPtr graph = m_mesh->m_meshGraph;
    unordered_map<int, bool> elementIDisSwaped;
    vector<SpatialDomains::Geometry *> elements;
    for (auto &[element, tag] : m_mesh->m_elementTags[m_dim])
    {
        elementIDisSwaped[element->GetGlobalID()] = false;
        elements.push_back(element);
    }

    // calculate valence per vertex
    // map < ID, valence>
    unordered_map<int, int> vertexIDValence;
    for (auto &[edge, links] : graph->GetAllFacetToElMap())
    {
        vertexIDValence[edge->GetVertex(0)->GetGlobalID()] += 1;
        vertexIDValence[edge->GetVertex(1)->GetGlobalID()] += 1;

        if (links.size() == 1)
        {
            vertexIDValence[edge->GetVertex(0)->GetGlobalID()] += 1;
            vertexIDValence[edge->GetVertex(1)->GetGlobalID()] += 1;
        }
    }

    int cntswapped = 0;
    for (auto element : elements)
    {
        if ((element->GetShapeType() == LibUtilities::eTriangle) &&
            (!elementIDisSwaped[element->GetGlobalID()]))
        {
            // find longest edge (cartesian for faster )
            bool toskip                          = false;
            bool ho                              = false;
            SpatialDomains::SegGeom *edgeLongest = nullptr;
            NekDouble edgeLongestLength          = 0;
            for (int e = 0; e < element->GetNumEdges(); e++)
            {
                auto edge =
                    static_cast<SpatialDomains::SegGeom *>(element->GetEdge(e));
                NekDouble length =
                    edge->GetVertex(0)->dist(*edge->GetVertex(1));
                if (length > edgeLongestLength)
                {
                    edgeLongestLength = length;
                    edgeLongest       = edge;
                }

                if (edge->GetCurve() != nullptr)
                {
                    ho = true;
                }

                if ((graph->GetElementsFromEdge(edge).size()) != 2 ||
                    (elementIDisSwaped[graph->GetElementsFromEdge(edge)[0]
                                           .first->GetGlobalID()] ||
                     elementIDisSwaped[graph->GetElementsFromEdge(edge)[1]
                                           .first->GetGlobalID()]))
                {
                    toskip = true;
                }
            }

            // if boundary edge - no swapping there
            if (graph->GetElementsFromEdge(edgeLongest).size() != 2)
            {
                continue;
            }

            if (toskip)
            {
                continue;
                // m_log(FATAL ) << "The algorithm does not update the edge
                // m-edgelinks " << endl ;
            }

            // Find Adjacent element if Quad skip swapping
            SpatialDomains::Geometry *elementAdjacent;

            if (element == graph->GetElementsFromEdge(edgeLongest)[0].first)
            {
                elementAdjacent =
                    graph->GetElementsFromEdge(edgeLongest)[1].first;
            }
            else
            {
                elementAdjacent =
                    graph->GetElementsFromEdge(edgeLongest)[0].first;
            }
            // WHAT IF THERE IS NO ELEMENT ADJACENT FOUND !

            if (elementAdjacent->GetShapeType() != LibUtilities::eTriangle)
            {
                continue;
            }

            // Create Reference Quadrilateral
            // Quad Vertices {edge1 , element third, edge2 , AdjElmt third}
            vector<SpatialDomains::PointGeom *> NodeListQuad(4);
            NodeListQuad[0] = edgeLongest->GetVertex(0);
            NodeListQuad[2] = edgeLongest->GetVertex(1);

            // Find(NodeListQuad[1]) the opposite vertex in Element
            for (int k = 0; k < element->GetNumVerts(); k++)
            {
                SpatialDomains::PointGeom *vertex = element->GetVertex(k);
                if (vertex != NodeListQuad[0] && vertex != NodeListQuad[2])
                {
                    NodeListQuad[1] = vertex;
                    break;
                }
            }

            // Find (NodeListQuad[3]) the adj element and last quadvert
            for (int k = 0; k < elementAdjacent->GetNumVerts(); k++)
            {
                SpatialDomains::PointGeom *vertex =
                    elementAdjacent->GetVertex(k);
                if (vertex != edgeLongest->GetVertex(0) &&
                    vertex != edgeLongest->GetVertex(1))
                {
                    NodeListQuad[3] = vertex;
                    break;
                }
            }

            auto Cos = [](SpatialDomains::PointGeom *o,
                          SpatialDomains::PointGeom *a,
                          SpatialDomains::PointGeom *b) {
                NekDouble ax = a->x() - o->x(), ay = a->y() - o->y(),
                          az = a->z() - o->z();
                NekDouble bx = b->x() - o->x(), by = b->y() - o->y(),
                          bz = b->z() - o->z();
                return (ax * bx + ay * by + az * bz) /
                       (sqrt(ax * ax + ay * ay + az * az) *
                        sqrt(bx * bx + by * by + bz * bz));
            };

            // decide to swap (Angles on top of the new elements + Length on
            // DiagonaL) or not and swap
            NekDouble edgeLenght13 = NodeListQuad[3]->dist(*NodeListQuad[1]);
            NekDouble cosELnew0 =
                Cos(NodeListQuad[0], NodeListQuad[1], NodeListQuad[3]);
            NekDouble cosELnew1 =
                Cos(NodeListQuad[2], NodeListQuad[1], NodeListQuad[3]);

            NekDouble cosELnew2 =
                Cos(NodeListQuad[1], NodeListQuad[0], NodeListQuad[2]);
            NekDouble cosELnew3 =
                Cos(NodeListQuad[3], NodeListQuad[0], NodeListQuad[2]);

            // conditions 1. At least slightly shorter 2. At least keep the same
            // overal valence 3. Keep some quality ( all angles >30deg )
            int diffValence = vertexIDValence[NodeListQuad[0]->GetGlobalID()] +
                              vertexIDValence[NodeListQuad[2]->GetGlobalID()] -
                              vertexIDValence[NodeListQuad[1]->GetGlobalID()] -
                              vertexIDValence[NodeListQuad[3]->GetGlobalID()];
            if ((edgeLongestLength > edgeLenght13 * 0.8025 &&
                 diffValence > 2) &&
                ((abs(cosELnew0) < 0.9 && abs(cosELnew1) < 0.9) &&
                 (abs(cosELnew2) < 0.9 && abs(cosELnew3) < 0.9)))
            {
                if (elementIDisSwaped[elementAdjacent->GetGlobalID()] ||
                    elementIDisSwaped[element->GetGlobalID()])
                {
                    continue;
                }
                else
                {
                    elementIDisSwaped[element->GetGlobalID()]         = true;
                    elementIDisSwaped[elementAdjacent->GetGlobalID()] = true;
                }

                cntswapped++;

                // EdgeLists of the Quad sorter with respect to NodeListQuad
                vector<SpatialDomains::SegGeom *> QuadEdgeList(4);
                vector<SpatialDomains::PointGeom *> NodeListQuadExtended(5);
                for (int i = 0; i < 4; i++)
                {
                    NodeListQuadExtended[i] = NodeListQuad[i];
                }
                NodeListQuadExtended[4] = NodeListQuad[0];

                for (int i = 0; i < 4; i++)
                {
                    SpatialDomains::SegGeom *edge = nullptr;
                    for (int j = 0; j < 3; j++)
                    {
                        if (i < 2 &&
                            (element->GetEdge(j)->GetVertex(0) ==
                                 NodeListQuadExtended[i] ||
                             element->GetEdge(j)->GetVertex(1) ==
                                 NodeListQuadExtended[i]) &&
                            ((element->GetEdge(j)->GetVertex(0) ==
                                  NodeListQuadExtended[i + 1] ||
                              element->GetEdge(j)->GetVertex(1) ==
                                  NodeListQuadExtended[i + 1])))
                        {
                            edge = static_cast<SpatialDomains::SegGeom *>(
                                element->GetEdge(j));
                            break;
                        }
                        else if (i >= 2 &&
                                 (elementAdjacent->GetEdge(j)->GetVertex(0) ==
                                      NodeListQuadExtended[i] ||
                                  elementAdjacent->GetEdge(j)->GetVertex(1) ==
                                      NodeListQuadExtended[i]) &&
                                 ((elementAdjacent->GetEdge(j)->GetVertex(0) ==
                                       NodeListQuadExtended[i + 1] ||
                                   elementAdjacent->GetEdge(j)->GetVertex(1) ==
                                       NodeListQuadExtended[i + 1])))
                        {
                            edge = static_cast<SpatialDomains::SegGeom *>(
                                elementAdjacent->GetEdge(j));
                            break;
                        }
                    }

                    QuadEdgeList[i] = edge;
                }

                // Get quad elemental isoparametric geometry object.
                // ho=false ;
                if (ho == true)
                {
                    m_log(WARNING) << "Curved swap" << endl;
                    // Create the new QUAD element, with the edges but do not
                    // push it
                    auto elmtQuad = ObjPoolManager<SpatialDomains::QuadGeom>::
                        AllocateUniquePtr(
                            -1, std::array<SpatialDomains::SegGeom *, 4>{
                                    QuadEdgeList[0], QuadEdgeList[1],
                                    QuadEdgeList[2], QuadEdgeList[3]});

                    SpatialDomains::QuadGeom *geom = elmtQuad.get();

                    // Update EdgeNodes of the EdgeLongest wih the new ones
                    Array<OneD, NekDouble> xp(2);
                    // LibUtilities::PointsKey
                    // ekey(edgeLongest->m_edgeNodes.size()+2,
                    // LibUtilities::eGaussLobattoLegendre);
                    SpatialDomains::Curve *curve = edgeLongest->GetCurve();
                    LibUtilities::PointsType curveType =
                        curve ? curve->m_ptype
                              : LibUtilities::ePolyEvenlySpaced;
                    LibUtilities::PointsKey ekey(
                        curve ? curve->m_points.size() : 2, curveType);
                    Array<OneD, NekDouble> gll;
                    LibUtilities::PointsManager()[ekey]->GetPoints(gll);

                    vector<SpatialDomains::PointGeom *> edgeNodes;

                    vector<NekDouble> v1 = {1.0, -1.0};
                    vector<NekDouble> v3 = {-1.0, 1.0};
                    for (int k = 1; k < gll.size() - 1; k++)
                    {
                        xp[0] = v1[0] * (1.0 - gll[k]) / 2.0 +
                                v3[0] * (1.0 + gll[k]) / 2.0;
                        xp[1] = v1[1] * (1.0 - gll[k]) / 2.0 +
                                v3[1] * (1.0 + gll[k]) / 2.0;

                        SpatialDomains::PointGeom *node =
                            graph->CreateCurveNode(graph->GetSpaceDimension(),
                                                   0, geom->GetCoord(0, xp),
                                                   geom->GetCoord(1, xp), 0.0);
                        edgeNodes.push_back(node);
                    }

                    CreateEdge(NodeListQuad[1], NodeListQuad[3], edgeNodes,
                               curveType);
                }

                int elTag = m_mesh->m_elementTags[m_dim].find(element)->second;
                int elAdjTag =
                    m_mesh->m_elementTags[m_dim].find(elementAdjacent)->second;

                // Create new elements - 1
                ElmtConfig conf(LibUtilities::eTriangle, 1, false, false, true);
                vector<SpatialDomains::PointGeom *> nodeList1(3);
                nodeList1[0] = NodeListQuad[0];
                nodeList1[1] = NodeListQuad[1];
                nodeList1[2] = NodeListQuad[3];

                SpatialDomains::Geometry *elmtNew =
                    GetElementFactory().CreateInstance(
                        LibUtilities::eTriangle, nodeList1, graph,
                        m_mesh->m_edgeSet, m_mesh->m_faceSet, conf, nullptr,
                        nullptr, nullptr, nullptr);

                m_mesh->m_elementTags[m_dim].erase(element);
                m_mesh->m_elementTags[m_dim][elmtNew] = elTag;

                // Create new elements - 2
                ElmtConfig conf2(LibUtilities::eTriangle, 1, false, false,
                                 true);
                nodeList1[0] = NodeListQuad[2];
                nodeList1[1] = NodeListQuad[1];
                nodeList1[2] = NodeListQuad[3];

                elmtNew = GetElementFactory().CreateInstance(
                    LibUtilities::eTriangle, nodeList1, graph,
                    m_mesh->m_edgeSet, m_mesh->m_faceSet, conf2, nullptr,
                    nullptr, nullptr, nullptr);

                m_mesh->m_elementTags[m_dim].erase(elementAdjacent);
                m_mesh->m_elementTags[m_dim][elmtNew] = elAdjTag;
            }
        }
    }
    m_log(VERBOSE) << "N_swapped pairs = " << cntswapped << endl;
}

// 6. Collapse Edges when the ratio between
void ProcessHAdaptation::CollapseEdge()
{
    // TBD
}

SpatialDomains::SegGeom *ProcessHAdaptation::CreateEdge(
    SpatialDomains::PointGeom *n1, SpatialDomains::PointGeom *n2,
    vector<SpatialDomains::PointGeom *> &edgeNodes,
    LibUtilities::PointsType curveType)
{
    SpatialDomains::MeshGraphSharedPtr graph = m_mesh->m_meshGraph;

    auto key = make_pair(n1->GetGlobalID(), n2->GetGlobalID());
    auto it  = m_mesh->m_edgeSet.find(key);
    if (it != m_mesh->m_edgeSet.end())
    {
        return it->second;
    }

    int id                          = NextEdgeId(graph);
    SpatialDomains::Curve *curvePtr = nullptr;
    if (edgeNodes.size() != 0)
    {
        auto curve = ObjPoolManager<SpatialDomains::Curve>::AllocateUniquePtr(
            id, curveType);
        curve->m_points.push_back(n1);
        for (auto node : edgeNodes)
        {
            curve->m_points.push_back(node);
        }
        curve->m_points.push_back(n2);
        curvePtr                    = curve.get();
        graph->GetCurvedEdges()[id] = std::move(curve);
    }

    SpatialDomains::SegGeom *seg = graph->CreateSegGeom(
        id, graph->GetSpaceDimension(), {n1, n2}, curvePtr);
    m_mesh->m_edgeSet[key] = seg;
    return seg;
}

void ProcessHAdaptation::RebuildLinks()
{
    SpatialDomains::MeshGraphSharedPtr graph = m_mesh->m_meshGraph;

    m_mesh->m_edgeSet.clear();
    for (auto [id, seg] : graph->GetGeomMap<SpatialDomains::SegGeom>())
    {
        m_mesh->m_edgeSet[make_pair(seg->GetVertex(0)->GetGlobalID(),
                                    seg->GetVertex(1)->GetGlobalID())] = seg;
    }

    m_mesh->m_faceSet.clear();
    for (auto [id, tri] : graph->GetGeomMap<SpatialDomains::TriGeom>())
    {
        m_mesh->m_faceSet[{tri->GetVertex(0)->GetGlobalID(),
                           tri->GetVertex(1)->GetGlobalID(),
                           tri->GetVertex(2)->GetGlobalID(), -1}] = tri;
    }
    for (auto [id, quad] : graph->GetGeomMap<SpatialDomains::QuadGeom>())
    {
        m_mesh->m_faceSet[{quad->GetVertex(0)->GetGlobalID(),
                           quad->GetVertex(1)->GetGlobalID(),
                           quad->GetVertex(2)->GetGlobalID(),
                           quad->GetVertex(3)->GetGlobalID()}] = quad;
    }

    graph->GetAllFacetToElMap().clear();
    for (auto &[element, tag] : m_mesh->m_elementTags[m_dim])
    {
        graph->PopulateFacetToElMap(element);
    }
}
} // namespace Nektar::NekMesh
