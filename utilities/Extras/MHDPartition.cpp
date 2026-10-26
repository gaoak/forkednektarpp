///////////////////////////////////////////////////////////////////////////////
//
// The MIT License
// Copyright (c) 2026 Nektar++ contributors.
// See the LICENSE file in the root directory for details.
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/BasicUtils/StringUtils.hpp>
#include <SpatialDomains/MeshGraphIOXml.h>
#include <SpatialDomains/MeshPartition.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>

using namespace Nektar;
namespace SD = Nektar::SpatialDomains;

namespace
{
// Reuse the normal partition writer, including per-session boundary filtering
// and the global composite/boundary ordering required by parallel CG assembly.
class PartitionWriter : public SD::MeshGraphIOXml
{
public:
    PartitionWriter(SD::MeshGraphSharedPtr graph,
                    LibUtilities::SessionReaderSharedPtr session)
    {
        m_meshGraph = graph;
        m_session   = session;
        m_compOrder = CreateCompositeOrdering();
        m_meshGraph->SetCompositeOrdering(m_compOrder);
    }
};

std::set<unsigned int> DomainElements(SD::MeshGraphSharedPtr graph)
{
    std::set<unsigned int> elements;
    for (auto &domain : graph->GetDomain())
        for (auto &composite : domain.second)
            for (auto &geometry : composite.second->m_geomVec)
                elements.insert(geometry->GetGlobalID());
    return elements;
}

void SelectCoefficientMode(LibUtilities::SessionReaderSharedPtr session,
                           const std::string &role, int ranks)
{
    auto *coupling = session->GetElement("Nektar/Coupling");
    coupling->SetAttribute("TYPE", "Coefficients");
    for (auto *item = coupling->FirstChildElement(); item;)
    {
        auto *next           = item->NextSiblingElement();
        const char *property = item->Attribute("PROPERTY");
        std::string key = StringUtils::ToUpperCopy(property ? property : "");
        if (key != "REMOTENAME" && key != "SENDSTEPS" &&
            key != "RECEIVESTEPS" && key != "SENDVARIABLES" &&
            key != "RECEIVEVARIABLES")
            coupling->RemoveChild(item);
        item = next;
    }
    for (auto option : {std::make_pair("Role", role),
                        std::make_pair("FluidRanks", std::to_string(ranks))})
    {
        auto *item = new TiXmlElement("I");
        item->SetAttribute("PROPERTY", option.first);
        item->SetAttribute("VALUE", option.second);
        coupling->LinkEndChild(item);
    }
}

std::vector<std::set<unsigned int>> Partition(
    LibUtilities::SessionReaderSharedPtr session, SD::MeshGraphSharedPtr graph,
    const std::set<unsigned int> &ids, int ranks)
{
    ASSERTL0(ranks > 0 && ids.size() >= static_cast<size_t>(ranks),
             "Each region must have at least one volume element per requested "
             "rank.");
    auto entities = graph->CreateMeshEntities();
    for (auto it = entities.begin(); it != entities.end();)
        if (!ids.count(it->first))
            it = entities.erase(it);
        else
            ++it;
    ASSERTL0(entities.size() == ids.size(),
             "Material domain contains non-volume or missing elements.");
    auto partitioner = SD::GetMeshPartitionFactory().CreateInstance(
        "Scotch", session, session->GetComm(), graph->GetMeshDimension(),
        entities, graph->CreateCompositeDescriptor(),
        graph->GetCompositesLabels());
    partitioner->PartitionMesh(ranks, true);
    std::vector<std::set<unsigned int>> result(ranks);
    std::set<unsigned int> visited;
    for (int r = 0; r < ranks; ++r)
    {
        std::vector<unsigned int> elements;
        partitioner->GetElementIDs(r, elements);
        ASSERTL0(!elements.empty(), "Scotch produced an empty partition.");
        for (auto e : elements)
        {
            ASSERTL0(
                ids.count(e) && visited.insert(e).second,
                "Scotch partition contains duplicate or foreign elements.");
            result[r].insert(e);
        }
    }
    ASSERTL0(visited == ids,
             "Scotch partitions do not cover the selected region.");
    return result;
}

void Write(const std::filesystem::path &prefix,
           LibUtilities::SessionReaderSharedPtr session,
           SD::MeshGraphSharedPtr graph,
           const std::vector<std::set<unsigned int>> &elements)
{
    auto directory = std::filesystem::path(prefix.string() + "_xml");
    if (std::filesystem::exists(directory))
    {
        // Re-running the same partition count is allowed; changing the count
        // requires a new output directory so stale rank files cannot survive.
        int files = 0;
        for (auto &entry : std::filesystem::directory_iterator(directory))
            if (entry.path().extension() == ".xml")
                ++files;
        ASSERTL0(files == static_cast<int>(elements.size()),
                 "Existing partition directory has a different rank count; use "
                 "a new output directory.");
    }
    std::vector<unsigned int> ranks(elements.size());
    std::iota(ranks.begin(), ranks.end(), 0);
    PartitionWriter writer(graph, session);
    writer.WriteXMLGeometry(prefix.string(), elements, ranks);
}
} // namespace

int main(int argc, char *argv[])
{
    if (argc != 8)
    {
        std::cerr
            << "Usage: MHDPartition M N output-dir fluid-mesh.xml fluid.xml "
               "electric-mesh.xml electric.xml\n";
        return 1;
    }
    try
    {
        int m = std::stoi(argv[1]), n = std::stoi(argv[2]);
        ASSERTL0(m > 0 && n > 0, "M and N must be positive.");
        std::vector<std::string> fluidFiles{argv[4], argv[5]};
        auto fluid =
            LibUtilities::SessionReader::CreateInstance(1, argv, fluidFiles);
        ASSERTL0(fluid->GetComm()->GetSize() == 1,
                 "Run the unified partition utility with a single process.");
        std::vector<std::string> electricFiles{argv[6], argv[7]};
        auto electric = LibUtilities::SessionReader::CreateInstance(
            1, argv, electricFiles, fluid->GetComm());
        auto fg = SD::MeshGraphIO::Read(fluid);
        auto eg = SD::MeshGraphIO::Read(electric);
        ASSERTL0(fg->GetMeshDimension() == eg->GetMeshDimension(),
                 "Fluid/electric mesh dimensions differ.");
        auto fluidIds = DomainElements(fg), allIds = DomainElements(eg);
        std::set<unsigned int> solidIds;
        std::set_difference(allIds.begin(), allIds.end(), fluidIds.begin(),
                            fluidIds.end(),
                            std::inserter(solidIds, solidIds.end()));
        ASSERTL0(std::includes(allIds.begin(), allIds.end(), fluidIds.begin(),
                               fluidIds.end()),
                 "Fluid elements must retain their original IDs in the "
                 "electrical mesh.");
        auto fe = fg->CreateMeshEntities(), ee = eg->CreateMeshEntities();
        for (auto id : fluidIds)
            ASSERTL0(fe.at(id).list == ee.at(id).list,
                     "Paired fluid element connectivity differs between input "
                     "meshes.");
        auto &electricVertices = eg->GetGeomMap<SD::PointGeom>();
        for (auto &v : fg->GetGeomMap<SD::PointGeom>())
        {
            ASSERTL0(electricVertices.find(v.first) != electricVertices.end(),
                     "Fluid vertex is missing from electrical mesh.");
            NekDouble fx, fy, fz, ex, ey, ez;
            v.second->GetCoords(fx, fy, fz);
            electricVertices.at(v.first)->GetCoords(ex, ey, ez);
            ASSERTL0(std::abs(fx - ex) < 1e-12 && std::abs(fy - ey) < 1e-12 &&
                         std::abs(fz - ez) < 1e-12,
                     "Fluid/electric vertex coordinates differ.");
        }
        std::set<unsigned int> declaredFluid;
        auto *materials = electric->GetElement("Nektar/ElectricPotential");
        for (auto *region = materials->FirstChildElement("FLUID"); region;
             region       = region->NextSiblingElement("FLUID"))
        {
            const char *attribute = region->Attribute("COMPOSITE");
            ASSERTL0(attribute, "FLUID requires COMPOSITE.");
            std::string ids(attribute);
            ASSERTL0(ids.size() > 3 && ids.substr(0, 2) == "C[" &&
                         ids.back() == ']',
                     "FLUID COMPOSITE must be C[...].");
            SD::CompositeMap composites;
            eg->GetCompositeList(ids.substr(2, ids.size() - 3), composites);
            for (auto &c : composites)
                for (auto &g : c.second->m_geomVec)
                    declaredFluid.insert(g->GetGlobalID());
        }
        ASSERTL0(
            declaredFluid == fluidIds,
            "Electrical FLUID composites do not match the fluid mesh domain.");

        // The fluid Scotch result is generated once, then reused verbatim.
        auto fluidParts    = Partition(fluid, fg, fluidIds, m);
        auto solidParts    = Partition(electric, eg, solidIds, n);
        auto electricParts = fluidParts;
        electricParts.insert(electricParts.end(), solidParts.begin(),
                             solidParts.end());
        SelectCoefficientMode(fluid, "Fluid", m);
        SelectCoefficientMode(electric, "Electric", m);
        std::filesystem::path output(argv[3]);
        std::filesystem::create_directories(output);
        Write(output / "fluid", fluid, fg, fluidParts);
        Write(output / "electric", electric, eg, electricParts);
        std::ofstream manifest(output / "partition-map.csv");
        ASSERTL0(manifest.good(), "Cannot write partition manifest.");
        manifest << "region,element_id,incns_rank,adr_rank\n";
        for (int r = 0; r < m + n; ++r)
            for (auto id : electricParts[r])
                manifest << (r < m ? "fluid" : "solid") << ',' << id << ','
                         << (r < m ? r : -1) << ',' << r << '\n';
        std::cout << "Unified Scotch partition: " << m << " fluid + " << n
                  << " solid partitions; " << 2 * m + n
                  << " total MPI ranks.\n";
        fluid->Finalise();
    }
    catch (const std::exception &error)
    {
        std::cerr << "MHDPartition: " << error.what() << std::endl;
        return 1;
    }
    return 0;
}
