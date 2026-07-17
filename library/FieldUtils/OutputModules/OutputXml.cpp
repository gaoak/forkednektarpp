////////////////////////////////////////////////////////////////////////////////
//
//  File: OutputXml.cpp
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
//  Description: VTK file format output.
//
////////////////////////////////////////////////////////////////////////////////

#include <SpatialDomains/MeshGraphIO.h>
#include <set>
#include <string>
using namespace std;

#include "OutputXml.h"

namespace Nektar::FieldUtils
{

ModuleKey OutputXml::m_className = GetModuleFactory().RegisterCreatorFunction(
    ModuleKey(eOutputModule, "xml"), OutputXml::createXml,
    "Writes an XML file.");
ModuleKey OutputXml::m_className2 = GetModuleFactory().RegisterCreatorFunction(
    ModuleKey(eOutputModule, "nekg"), OutputXml::createHDF5,
    "Writes a HDF5 mesh file.");

OutputXml::OutputXml(FieldSharedPtr f, bool hdf5)
    : OutputModule(f), m_hdf5(hdf5)
{
    m_config["writedefaultexp"] =
        ConfigOption(true, "0", "Write a default expansion.");
}

OutputXml::~OutputXml()
{
}

void OutputXml::v_Process(po::variables_map &vm)
{
    m_f->SetUpExp(vm);

    if (!m_f->m_exp.size()) // do nothing if no expansion defined
    {
        return;
    }

    // Extract the output filename and extension
    string filename = m_config["outfile"].as<string>();

    std::string outputFmt = "Xml";
    if (m_hdf5)
    {
        ASSERTL0(SpatialDomains::GetMeshGraphIOFactory().ModuleExists("HDF5"),
                 "Nektar++ must be compiled with HDF5 support to output in "
                 ".nekg format");

        outputFmt = "HDF5";
    }

    auto graphIO =
        SpatialDomains::GetMeshGraphIOFactory().CreateInstance(outputFmt);
    graphIO->SetMeshGraph(m_f->m_graph);
    graphIO->WriteGeometry(filename, m_config["writedefaultexp"].m_beenSet);

    if ((!graphIO->HasMultifileOutput() && m_f->m_comm->TreatAsRankZero()) ||
        graphIO->HasMultifileOutput())
    {
        cout << "Written file: " << filename << endl;
    }
}
} // namespace Nektar::FieldUtils
