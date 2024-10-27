///////////////////////////////////////////////////////////////////////////////
//
// File: init_mathkernels.hpp
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

using namespace Nektar::LibUtilities;
using namespace Nektar;

class MathKernelsField
    : public InitFields<double, FieldState::Phys, FieldState::Phys,
                        MultiRegions::ContField>
{
public:
    MathKernelsField()
        : InitFields<double, FieldState::Phys, FieldState::Phys,
                     MultiRegions::ContField>()
    {
    }

    ~MathKernelsField()
    {
        if (fixt_in2)
        {
            delete fixt_in2;
        }
    }

    void SetTestCase()
    {
        auto blocks_in =
            GetBlockAttributes<double>(FieldState::Phys, fixt_explist);
        auto f_in = Field<double, FieldState::Phys>::template create<
            NektarSpaces::DeviceSpace>("f_in2", blocks_in, 1, alignment);
        fixt_in2 = new Field<double, FieldState::Phys>(std::move(f_in));

        Array<OneD, double> x(fixt_explist->GetTotPoints());
        Array<OneD, double> y(fixt_explist->GetTotPoints());
        Array<OneD, double> z(fixt_explist->GetTotPoints());
        Array<OneD, double> fce(fixt_explist->GetTotPoints());
        fixt_explist->GetCoords(x, y, z);
        auto func1 = fixt_explist->GetSession()->GetFunction("Forcing", 0);
        func1->Evaluate(x, y, z, fce);
        auto ptr   = fce.get();
        auto inptr = fixt_in->GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        for (auto const &block : fixt_in->GetBlocks())
        {
            auto n = block.num_elements * block.num_pts;
            std::copy(ptr, ptr + n, inptr);
            std::fill(inptr + n, inptr + block.block_size, 0);
            ptr += n;
            inptr += block.block_size;
        }

        auto func2 =
            fixt_explist->GetSession()->GetFunction("ExactSolution", 0);
        func2->Evaluate(x, y, z, fce);
        ptr   = fce.get();
        inptr = fixt_in2->GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        for (auto const &block : fixt_out->GetBlocks())
        {
            auto n = block.num_elements * block.num_pts;
            std::copy(ptr, ptr + n, inptr);
            std::fill(inptr + n, inptr + block.block_size, 0);
            ptr += n;
            inptr += block.block_size;
        }
    }

protected:
    Field<double, FieldState::Phys> *fixt_in2 = nullptr;
};

class MathKernels : public MathKernelsField
{
public:
    MathKernels()
    {
        meshName = "run/Helmholtz3D_Hex_AllBCs_P6.xml";
    }
};
