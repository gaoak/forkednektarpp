///////////////////////////////////////////////////////////////////////////////
//
// File: init_parallel_reduce.hpp
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
// Description:
//
///////////////////////////////////////////////////////////////////////////////

#include "init_fields.hpp"

using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

class ReducerField
{
public:
    ReducerField()
    {
        meshName = "run/Helmholtz3D_Hex_AllBCs_P6.xml";
    }

    ~ReducerField()
    {
        BOOST_TEST_MESSAGE("teardown fixture");

        if (fixt_in)
        {
            delete fixt_in;
        }
    }

    void Configure()
    {
        BOOST_TEST_MESSAGE("Creating input and output fields");

        // Construct a fake command-line argument array to be fed to
        // Session::Reader::CreateInstance. The first element stands for
        // the name of the executable which, in our case, doesn't matter.
        int argc    = 2;
        char **argv = new char *[argc];
        argv[0]     = strdup("exe_name");
        argv[1]     = meshName.data();

        session    = LibUtilities::SessionReader::CreateInstance(argc, argv);
        auto graph = SpatialDomains::MeshGraphIO::Read(session);

        if (session->GetComm())
        {
            auto rank        = session->GetComm()->GetRank();
            auto num_device  = nekGetDeviceCount();
            auto device_rank = rank % num_device;
            nekSetDevice(device_rank);
        }

        fixt_explist = MemoryManager<MultiRegions::ExpList>::AllocateSharedPtr(
            session, graph, true, "u", Collections::eNoCollection);

        auto blocksAttr =
            GetBlockAttributes<double, FieldState::Phys>(fixt_explist);
        auto f_in = Field<double, FieldState::Phys>("f_in", blocksAttr, 1, 1);
        fixt_in   = new Field<double, FieldState::Phys>(std::move(f_in));
    }

    void SetTestCase()
    {
        Array<OneD, double> x(fixt_explist->GetTotPoints());
        Array<OneD, double> y(fixt_explist->GetTotPoints());
        Array<OneD, double> z(fixt_explist->GetTotPoints());
        Array<OneD, double> fce(fixt_explist->GetTotPoints());
        fixt_explist->GetCoords(x, y, z);

        auto func = session->GetFunction("Forcing", 0);
        func->Evaluate(x, y, z, fce);
        auto ptr = fce.data();
        for (unsigned int blk = 0; blk < fixt_in->GetBlocks().size(); ++blk)
        {
            auto inptr = fixt_in->GetBlocks()[blk]
                             .GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            auto size = fixt_in->GetBlocks()[blk].GetNumElements() *
                        fixt_in->GetBlocks()[blk].GetNumData();
            std::copy(ptr, ptr + size, inptr);
            ptr += size;
        }
    }

    double sum()
    {
        double out = 0;
        for (unsigned int blk = 0; blk < fixt_in->GetBlocks().size(); ++blk)
        {
            auto x = fixt_in->GetBlocks()[blk]
                         .GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto size = fixt_in->GetBlocks()[blk].GetNumElements() *
                        fixt_in->GetBlocks()[blk].GetNumData();
            out = std::accumulate(x, x + size, out);
        }
        return out;
    }

    double max()
    {
        double out = std::numeric_limits<double>::min();
        for (unsigned int blk = 0; blk < fixt_in->GetBlocks().size(); ++blk)
        {
            auto x = fixt_in->GetBlocks()[blk]
                         .GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto size = fixt_in->GetBlocks()[blk].GetNumElements() *
                        fixt_in->GetBlocks()[blk].GetNumData();
            out = std::max(out, *(std::max_element(x, x + size)));
        }
        return out;
    }

    double min()
    {
        double out = std::numeric_limits<double>::max();
        for (unsigned int blk = 0; blk < fixt_in->GetBlocks().size(); ++blk)
        {
            auto x = fixt_in->GetBlocks()[blk]
                         .GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto size = fixt_in->GetBlocks()[blk].GetNumElements() *
                        fixt_in->GetBlocks()[blk].GetNumData();
            out = std::min(out, *(std::min_element(x, x + size)));
        }
        return out;
    }

protected:
    std::string meshName                     = "";
    Field<double, FieldState::Phys> *fixt_in = nullptr;
    std::shared_ptr<MultiRegions::ExpList> fixt_explist;
    LibUtilities::SessionReaderSharedPtr session;
};
