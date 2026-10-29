// Minimal adaptation of Mesh2D square: draw the full fluid/solid rectangle once.
#include "CAD2D/MeshRegions.h"
#include "CAD2D/RectRegion.h"
#include "edgefunctions.h"

int main()
{
    std::vector<void *> edges = {reinterpret_cast<void *>(edge03),
                                reinterpret_cast<void *>(edge39),
                                reinterpret_cast<void *>(edge69),
                                reinterpret_cast<void *>(edge06)};
    RectRegion region(edges, "full_fluid_solid");
    region.MeshGen(Nx, Ny);
    MeshRegions combined("domain", 1e-10);
    combined.AddRegion(region);
    // Preserve square's source boundary numbering: bottom, top, left, right.
    combined.defineBoundary(reinterpret_cast<void *>(edge03), Nx, 1);
    combined.defineBoundary(reinterpret_cast<void *>(edge69), Nx, 2);
    combined.defineBoundary(reinterpret_cast<void *>(edge06), Ny, 3);
    combined.defineBoundary(reinterpret_cast<void *>(edge39), Ny, 4);
    combined.CheckMesh();
    combined.outXml("full.xml");
    combined.outCOMPO("full.xml", {0});
    return 0;
}
