////////////////////////////////////////////////////////////////////////////////
//
//  File: TetGenInterface.cpp
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
//  Description: tetgen interface methods
//
////////////////////////////////////////////////////////////////////////////////

#include <NekMesh/ExtLibInterface/TetGenInterface.h>

#include <cstdlib>
#include <exception>

#define TETLIBRARY
#include <tetgen.h>

using namespace std;

namespace Nektar::NekMesh
{

namespace
{

/// The logger the terminate handler below reports through.
Logger *g_tetGenLog = nullptr;

/// TetGen's exit codes, from terminatetetgen() in tetgen.h.
std::string DescribeTetGenCode(int code)
{
    switch (code)
    {
        case 1:
            return "TetGen ran out of memory";
        case 2:
            return "TetGen hit an internal error";
        case 3:
            return "the surface mesh intersects itself: two triangles that "
                   "are not neighbours pass through one another. This usually "
                   "means the spacing is too coarse to separate two parts of "
                   "the geometry, so that a face is meshed flat enough to cut "
                   "through its neighbour -- reduce MinDelta";
        case 4:
            return "the geometry has a feature smaller than the tolerance "
                   "TetGen works to";
        case 5:
            return "two facets of the surface mesh lie very close together, "
                   "which again points at spacing too coarse for the "
                   "geometry -- reduce MinDelta";
        case 10:
            return "TetGen rejected the input mesh";
        default:
            return "TetGen stopped with code " + std::to_string(code);
    }
}

/**
 * @brief Report TetGen's exit code and stop.
 *
 * Built as a library, TetGen reports every failure by throwing a bare int and
 * printing nothing, so the only thing the user sees is `terminate called after
 * throwing an instance of 'int'` and a core dump.
 *
 * This is a terminate handler rather than a catch because TetGen cannot
 * survive having its stack unwound: terminatetetgen() frees the mesh before
 * throwing, and the cleanup the unwind then runs over that same memory
 * segfaults. With no handler anywhere the runtime calls terminate at the throw
 * point without unwinding, which is the one path through TetGen's error
 * handling that works. So the message is printed here and the process stops,
 * rather than being caught somewhere a recovery could be attempted.
 */
[[noreturn]] void TetGenTerminate()
{
    std::string why = "TetGen stopped without saying why";

    // Rethrowing inside this handler reads the exception without unwinding
    // the stack it was thrown from.
    if (auto e = std::current_exception())
    {
        try
        {
            std::rethrow_exception(e);
        }
        catch (int code)
        {
            why = DescribeTetGenCode(code);
        }
        catch (...)
        {
            // Not TetGen's; leave the default wording.
        }
    }

    if (g_tetGenLog)
    {
        // Not FATAL: that throws, and throwing from a terminate handler calls
        // terminate again.
        (*g_tetGenLog)(WARNING)
            << "tetrahedral meshing failed, and cannot be recovered from: "
            << why << "." << std::endl;
    }

    std::cout.flush();
    std::cerr.flush();
    std::_Exit(EXIT_FAILURE);
}

/**
 * @brief Run TetGen with @p cmd, reporting its exit code rather than aborting.
 *
 * The handler and the logger it reports through are process-wide, as
 * std::set_terminate is; TetGen is called from one thread at a time.
 */
void RunTetGen(const std::string &cmd, tetgenio *in, tetgenio *out, Logger &log)
{
    std::vector<char> switches(cmd.begin(), cmd.end());
    switches.push_back('\0');

    g_tetGenLog                 = &log;
    std::terminate_handler prev = std::set_terminate(TetGenTerminate);

    tetrahedralize(switches.data(), in, out);

    std::set_terminate(prev);
    g_tetGenLog = nullptr;
}

} // namespace

void TetGenInterface::InitialMesh(map<int, SpatialDomains::PointGeom *> tgidton,
                                  vector<std::array<int, 3>> tri)
{
    surface.initialize();
    output.initialize();

    // build surface input
    tetgenio::facet *f;
    tetgenio::polygon *p;

    surface.firstnumber    = 0;
    surface.numberofpoints = tgidton.size();
    surface.pointlist      = new REAL[surface.numberofpoints * 3];

    map<int, SpatialDomains::PointGeom *>::iterator it;
    for (it = tgidton.begin(); it != tgidton.end(); it++)
    {
        std::array<NekDouble, 3> loc;
        it->second->GetCoords(loc[0], loc[1], loc[2]);

        surface.pointlist[it->first * 3 + 0] = loc[0];
        surface.pointlist[it->first * 3 + 1] = loc[1];
        surface.pointlist[it->first * 3 + 2] = loc[2];
    }

    surface.numberoffacets  = tri.size();
    surface.facetlist       = new tetgenio::facet[surface.numberoffacets];
    surface.facetmarkerlist = new int[surface.numberoffacets];

    for (int i = 0; i < tri.size(); i++)
    {
        f                          = &surface.facetlist[i];
        f->numberofpolygons        = 1;
        f->polygonlist             = new tetgenio::polygon[f->numberofpolygons];
        f->numberofholes           = 0;
        f->holelist                = nullptr;
        p                          = &f->polygonlist[0];
        p->numberofvertices        = 3;
        p->vertexlist              = new int[p->numberofvertices];
        p->vertexlist[0]           = tri[i][0];
        p->vertexlist[1]           = tri[i][1];
        p->vertexlist[2]           = tri[i][2];
        surface.facetmarkerlist[i] = 1;
    }

    surface.numberofholes = m_holes.size();

    if (m_holes.size() > 0)
    {
        surface.holelist = new REAL[m_holes.size() * 3];
        for (int j = 0; j < m_holes.size(); ++j)
        {
            surface.holelist[3 * j + 0] = m_holes[j][0];
            surface.holelist[3 * j + 1] = m_holes[j][1];
            surface.holelist[3 * j + 2] = m_holes[j][2];
        }
    }
    else
    {
        surface.holelist = nullptr;
    }

    RunTetGen("pYzqQ", &surface, &output, m_log);
}

void TetGenInterface::GetNewPoints(int num,
                                   vector<std::array<NekDouble, 3>> &newp)
{
    for (int i = num; i < output.numberofpoints; i++)
    {
        newp.push_back({output.pointlist[i * 3 + 0],
                        output.pointlist[i * 3 + 1],
                        output.pointlist[i * 3 + 2]});
    }
}

void TetGenInterface::RefineMesh(std::map<int, NekDouble> delta)
{
    input = output;

    input.numberofpointmtrs = 1;

    input.pointmtrlist = new REAL[input.numberofpoints];

    for (int i = 0; i < input.numberofpoints; i++)
    {
        input.pointmtrlist[i] = delta[i];
    }

    RunTetGen("pYrmzq1.1/0QO2/7", &input, &output, m_log);
}

vector<std::array<int, 4>> TetGenInterface::Extract()
{
    vector<std::array<int, 4>> tets(output.numberoftetrahedra);

    for (int i = 0; i < output.numberoftetrahedra; i++)
    {
        tets[i] = {output.tetrahedronlist[i * 4 + 0],
                   output.tetrahedronlist[i * 4 + 1],
                   output.tetrahedronlist[i * 4 + 2],
                   output.tetrahedronlist[i * 4 + 3]};
    }

    return tets;
}

void TetGenInterface::freetet()
{
#ifdef TETGEN_HAS_DEINITIALIZE
    surface.deinitialize();
    input.deinitialize();
    output.deinitialize();
#else
    surface.clean_memory();
    input.clean_memory();
    output.clean_memory();
#endif
}
} // namespace Nektar::NekMesh
