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

#if defined(_MSC_VER)
#undef max
#undef min
#endif

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
        for (const auto &block : fixt_in->GetBlocks())
        {
            auto size = block.num_elements * block.num_pts;
            std::copy(ptr, ptr + size, inptr);
            std::fill(inptr + size, inptr + block.block_size, 0);
            ptr += size;
            inptr += block.block_size;
        }

        auto func2 =
            fixt_explist->GetSession()->GetFunction("ExactSolution", 0);
        func2->Evaluate(x, y, z, fce);
        ptr   = fce.get();
        inptr = fixt_in2->GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        for (unsigned int block_idx = 0;
             block_idx < fixt_in2->GetBlocks().size(); ++block_idx)
        {
            auto &block = fixt_in2->GetBlocks()[block_idx];
            auto size   = block.num_elements * block.num_pts;
            std::copy(ptr, ptr + size, inptr);
            ptr += size;
            inptr += block.block_size;
        }
    }

    void neg()
    {
        auto x = fixt_in->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        auto y = fixt_expected->GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        for (unsigned int block_idx = 0;
             block_idx < fixt_in->GetBlocks().size(); ++block_idx)
        {
            auto &block = fixt_in->GetBlocks()[block_idx];
            auto size   = block.num_elements * block.num_pts;
            std::transform(x, x + size, y,
                           [](const double &xi) { return -xi; });
            x += block.block_size;
            y += block.block_size;
        }
    }

    void add()
    {
        auto x = fixt_in->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        auto y = fixt_in2->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        auto z = fixt_expected->GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        for (unsigned int block_idx = 0;
             block_idx < fixt_in->GetBlocks().size(); ++block_idx)
        {
            auto &block = fixt_in->GetBlocks()[block_idx];
            auto size   = block.num_elements * block.num_pts;
            std::transform(
                x, x + size, y, z,
                [](const double &xi, const double &yi) { return xi + yi; });
            x += block.block_size;
            y += block.block_size;
            z += block.block_size;
        }
    }

    void sub()
    {
        auto x = fixt_in->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        auto y = fixt_in2->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        auto z = fixt_expected->GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        for (unsigned int block_idx = 0;
             block_idx < fixt_in->GetBlocks().size(); ++block_idx)
        {
            auto &block = fixt_in->GetBlocks()[block_idx];
            auto size   = block.num_elements * block.num_pts;
            std::transform(
                x, x + size, y, z,
                [](const double &xi, const double &yi) { return xi - yi; });
            x += block.block_size;
            y += block.block_size;
            z += block.block_size;
        }
    }

    void daxpy(double alpha)
    {
        auto x = fixt_in->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        auto y = fixt_in2->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        auto z = fixt_expected->GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        for (unsigned int block_idx = 0;
             block_idx < fixt_in->GetBlocks().size(); ++block_idx)
        {
            auto &block = fixt_in->GetBlocks()[block_idx];
            auto size   = block.num_elements * block.num_pts;
            std::transform(x, x + size, y, z,
                           [=](const double &xi, const double &yi) {
                               return alpha * xi + yi;
                           });
            x += block.block_size;
            y += block.block_size;
            z += block.block_size;
        }
    }

    void div()
    {
        auto x = fixt_in->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        auto y = fixt_in2->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        auto z = fixt_expected->GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        for (unsigned int block_idx = 0;
             block_idx < fixt_in->GetBlocks().size(); ++block_idx)
        {
            auto &block = fixt_in->GetBlocks()[block_idx];
            auto size   = block.num_elements * block.num_pts;
            std::transform(
                x, x + size, y, z,
                [](const double &xi, const double &yi) { return xi / yi; });
            x += block.block_size;
            y += block.block_size;
            z += block.block_size;
        }
    }

    double sum()
    {
        double out = 0;
        auto x     = fixt_in->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        for (unsigned int block_idx = 0;
             block_idx < fixt_in->GetBlocks().size(); ++block_idx)
        {
            auto &block = fixt_in->GetBlocks()[block_idx];
            auto size   = block.num_elements * block.num_pts;
            out         = std::accumulate(x, x + size, out);
            x += block.block_size;
        }
        return out;
    }

    double max()
    {
        double out = std::numeric_limits<double>::min();
        auto x     = fixt_in->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        for (unsigned int block_idx = 0;
             block_idx < fixt_in->GetBlocks().size(); ++block_idx)
        {
            auto &block = fixt_in->GetBlocks()[block_idx];
            auto size   = block.num_elements * block.num_pts;
            out         = std::max(out, *(std::max_element(x, x + size)));
            x += block.block_size;
        }
        return out;
    }

    double min()
    {
        double out = std::numeric_limits<double>::max();
        auto x     = fixt_in->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        for (unsigned int block_idx = 0;
             block_idx < fixt_in->GetBlocks().size(); ++block_idx)
        {
            auto &block = fixt_in->GetBlocks()[block_idx];
            auto size   = block.num_elements * block.num_pts;
            out         = std::min(out, *(std::min_element(x, x + size)));
            x += block.block_size;
        }
        return out;
    }

    double inner_product()
    {
        double out = 0;
        auto x     = fixt_in->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        auto y     = fixt_in2->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        for (unsigned int block_idx = 0;
             block_idx < fixt_in->GetBlocks().size(); ++block_idx)
        {
            auto &block = fixt_in->GetBlocks()[block_idx];
            auto size   = block.num_elements * block.num_pts;
            out         = std::inner_product(x, x + size, y, out);
            x += block.block_size;
        }
        return out;
    }

    double l1norm()
    {
        double out = 0;
        auto x     = fixt_in->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        for (unsigned int block_idx = 0;
             block_idx < fixt_in->GetBlocks().size(); ++block_idx)
        {
            auto &block = fixt_in->GetBlocks()[block_idx];
            auto size   = block.num_elements * block.num_pts;
            out         = std::accumulate(x, x + size, out,
                                          [](const double &acc, const double &val) {
                                      return acc + std::abs(val);
                                  });
            x += block.block_size;
        }
        return out;
    }

    double l2norm()
    {
        double out = 0;
        auto x     = fixt_in->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        for (unsigned int block_idx = 0;
             block_idx < fixt_in->GetBlocks().size(); ++block_idx)
        {
            auto &block = fixt_in->GetBlocks()[block_idx];
            auto size   = block.num_elements * block.num_pts;
            out         = std::accumulate(x, x + size, out,
                                          [](const double &acc, const double &val) {
                                      return acc + val * val;
                                  });
            x += block.block_size;
        }
        return out;
    }

    double lpnorm(const unsigned int p)
    {
        double out = 0;
        auto x     = fixt_in->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        for (unsigned int block_idx = 0;
             block_idx < fixt_in->GetBlocks().size(); ++block_idx)
        {
            auto &block = fixt_in->GetBlocks()[block_idx];
            auto size   = block.num_elements * block.num_pts;
            out         = std::accumulate(x, x + size, out,
                                          [&](const double &acc, const double &val) {
                                      return acc + std::pow(std::abs(val), p);
                                  });
            x += block.block_size;
        }
        return out;
    }

    double linfnorm()
    {
        double out = std::numeric_limits<double>::min();
        auto x     = fixt_in->GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        for (unsigned int block_idx = 0;
             block_idx < fixt_in->GetBlocks().size(); ++block_idx)
        {
            auto &block = fixt_in->GetBlocks()[block_idx];
            auto size   = block.num_elements * block.num_pts;
            out         = std::accumulate(
                x, x + size, out, [](const double &acc, const double &val) {
                    return std::max(std::abs(acc), std::abs(val));
                });
            x += block.block_size;
        }
        return out;
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
