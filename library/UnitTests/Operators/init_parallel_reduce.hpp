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

template <typename TData> class ReducerField
{
public:
    ReducerField()
    {
        meshName = "run/Helmholtz3D_Hex_AllBCs_P6.xml";
    }

    ~ReducerField()
    {
        BOOST_TEST_MESSAGE("teardown this->fixture");

        if (this->fixt_in)
        {
            delete this->fixt_in;
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

        this->session = LibUtilities::SessionReader::CreateInstance(argc, argv);
        auto graph    = SpatialDomains::MeshGraphIO::Read(this->session);

        if (this->session->GetComm())
        {
            auto rank        = this->session->GetComm()->GetRank();
            auto num_device  = nekGetDeviceCount();
            auto device_rank = rank % num_device;
            nekSetDevice(device_rank);
        }

        this->fixt_explist =
            MemoryManager<MultiRegions::ExpList>::AllocateSharedPtr(
                this->session, graph, true, "u", Collections::eNoCollection);

        auto blocksAttr =
            GetBlockAttributes<double, FieldState::Phys>(fixt_explist);
        auto f_in =
            Field<double, FieldState::Phys>("f_in", blocksAttr, {"u"}, 1);
        fixt_in = new Field<double, FieldState::Phys>(std::move(f_in));
    }

    void SetTestCase()
    {
        Array<OneD, TData> x(this->fixt_explist->GetTotPoints());
        Array<OneD, TData> y(this->fixt_explist->GetTotPoints());
        Array<OneD, TData> z(this->fixt_explist->GetTotPoints());
        Array<OneD, TData> fce(this->fixt_explist->GetTotPoints());
        this->fixt_explist->GetCoords(x, y, z);

        auto func = this->session->GetFunction("Forcing", 0);
        func->Evaluate(x, y, z, fce);
        auto ptr = fce.data();
        for (unsigned int blk = 0; blk < this->fixt_in->GetBlocks().size();
             ++blk)
        {
            auto inptr =
                this->fixt_in->GetBlocks()[blk]
                    .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            auto size = this->fixt_in->GetBlocks()[blk].GetNumElements() *
                        this->fixt_in->GetBlocks()[blk].GetNumData();
            std::copy(ptr, ptr + size, inptr);
            ptr += size;
        }
    }

    TData sum()
    {
        TData out = 0;
        for (unsigned int blk = 0; blk < this->fixt_in->GetBlocks().size();
             ++blk)
        {
            auto x = this->fixt_in->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto size = this->fixt_in->GetBlocks()[blk].GetNumElements() *
                        this->fixt_in->GetBlocks()[blk].GetNumData();
            out = std::accumulate(x, x + size, out);
        }
        return out;
    }

    TData max()
    {
        TData out = std::numeric_limits<TData>::min();
        for (unsigned int blk = 0; blk < this->fixt_in->GetBlocks().size();
             ++blk)
        {
            auto x = this->fixt_in->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto size = this->fixt_in->GetBlocks()[blk].GetNumElements() *
                        this->fixt_in->GetBlocks()[blk].GetNumData();
            out = std::max(out, *(std::max_element(x, x + size)));
        }
        return out;
    }

    TData min()
    {
        TData out = std::numeric_limits<TData>::max();
        for (unsigned int blk = 0; blk < this->fixt_in->GetBlocks().size();
             ++blk)
        {
            auto x = this->fixt_in->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto size = this->fixt_in->GetBlocks()[blk].GetNumElements() *
                        this->fixt_in->GetBlocks()[blk].GetNumData();
            out = std::min(out, *(std::min_element(x, x + size)));
        }
        return out;
    }

protected:
    std::string meshName                    = "";
    Field<TData, FieldState::Phys> *fixt_in = nullptr;
    std::shared_ptr<MultiRegions::ExpList> fixt_explist;
    LibUtilities::SessionReaderSharedPtr session;
};
