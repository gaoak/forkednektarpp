///////////////////////////////////////////////////////////////////////////////
//
// File: init_math.hpp
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

#include "Operators/MathKernels/Math.hpp"

using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

class MathField : public InitFields<double, FieldState::Phys, FieldState::Phys,
                                    MultiRegions::ExpList>
{
public:
    MathField()
        : InitFields<double, FieldState::Phys, FieldState::Phys,
                     MultiRegions::ExpList>()
    {
        meshName = "run/Helmholtz3D_Hex_AllBCs_P6.xml";
    }

    ~MathField()
    {
        if (fixt_in2)
        {
            delete fixt_in2;
        }
    }

    void SetTestCase()
    {
        std::string execName =
            session->GetCmdLineArgument<std::string>("opExecSpace");

        math = Math(execName);

        auto blocks_in =
            GetBlockAttributes<double>(FieldState::Phys, fixt_explist);
        auto f_in = Field<double, FieldState::Phys>::Create("f_in2", blocks_in,
                                                            1, 1, alignment);
        fixt_in2  = new Field<double, FieldState::Phys>(std::move(f_in));

        Array<OneD, double> x(fixt_explist->GetTotPoints());
        Array<OneD, double> y(fixt_explist->GetTotPoints());
        Array<OneD, double> z(fixt_explist->GetTotPoints());
        Array<OneD, double> fce(fixt_explist->GetTotPoints());
        fixt_explist->GetCoords(x, y, z);

        auto func1 = session->GetFunction("Forcing", 0);
        func1->Evaluate(x, y, z, fce);
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

        auto func2 = session->GetFunction("ExactSolution", 0);
        func2->Evaluate(x, y, z, fce);
        ptr = fce.data();
        for (unsigned int blk = 0; blk < fixt_in2->GetBlocks().size(); ++blk)
        {
            auto inptr = fixt_in2->GetBlocks()[blk]
                             .GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            auto size = fixt_in2->GetBlocks()[blk].GetNumElements() *
                        fixt_in2->GetBlocks()[blk].GetNumData();
            std::copy(ptr, ptr + size, inptr);
            ptr += size;
        }
    }

    void abs()
    {
        for (unsigned int blk = 0; blk < fixt_in->GetBlocks().size(); ++blk)
        {
            auto x = fixt_in->GetBlocks()[blk]
                         .GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto y = fixt_expected->GetBlocks()[blk]
                         .GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            auto size = fixt_in->GetBlocks()[blk].GetNumElements() *
                        fixt_in->GetBlocks()[blk].GetNumData();

            std::transform(x, x + size, y,
                           [](const double &xi) { return std::abs(xi); });
        }
    }

    void neg()
    {
        for (unsigned int blk = 0; blk < fixt_in->GetBlocks().size(); ++blk)
        {
            auto x = fixt_in->GetBlocks()[blk]
                         .GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto y = fixt_expected->GetBlocks()[blk]
                         .GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            auto size = fixt_in->GetBlocks()[blk].GetNumElements() *
                        fixt_in->GetBlocks()[blk].GetNumData();
            std::transform(x, x + size, y,
                           [](const double &xi) { return -xi; });
        }
    }

    void sqrt()
    {
        for (unsigned int blk = 0; blk < fixt_in->GetBlocks().size(); ++blk)
        {
            auto x = fixt_in->GetBlocks()[blk]
                         .GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto y = fixt_expected->GetBlocks()[blk]
                         .GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            auto size = fixt_in->GetBlocks()[blk].GetNumElements() *
                        fixt_in->GetBlocks()[blk].GetNumData();

            // If inputs might be negative and want to avoid NaNs,
            // replace std::sqrt(xi) with std::sqrt(std::abs(xi)).
            std::transform(x, x + size, y, [](const double &xi) {
                return std::sqrt(std::abs(xi));
            });
        }
    }

    void add()
    {
        for (unsigned int blk = 0; blk < fixt_in->GetBlocks().size(); ++blk)
        {
            auto x = fixt_in->GetBlocks()[blk]
                         .GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto y = fixt_in2->GetBlocks()[blk]
                         .GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto z = fixt_expected->GetBlocks()[blk]
                         .GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            auto size = fixt_in->GetBlocks()[blk].GetNumElements() *
                        fixt_in->GetBlocks()[blk].GetNumData();
            std::transform(
                x, x + size, y, z,
                [](const double &xi, const double &yi) { return xi + yi; });
        }
    }

    void sub()
    {
        for (unsigned int blk = 0; blk < fixt_in->GetBlocks().size(); ++blk)
        {
            auto x = fixt_in->GetBlocks()[blk]
                         .GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto y = fixt_in2->GetBlocks()[blk]
                         .GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto z = fixt_expected->GetBlocks()[blk]
                         .GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            auto size = fixt_in->GetBlocks()[blk].GetNumElements() *
                        fixt_in->GetBlocks()[blk].GetNumData();
            std::transform(
                x, x + size, y, z,
                [](const double &xi, const double &yi) { return xi - yi; });
        }
    }

    void mul()
    {
        for (unsigned int blk = 0; blk < fixt_in->GetBlocks().size(); ++blk)
        {
            auto x = fixt_in->GetBlocks()[blk]
                         .GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto y = fixt_in2->GetBlocks()[blk]
                         .GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto z = fixt_expected->GetBlocks()[blk]
                         .GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            auto size = fixt_in->GetBlocks()[blk].GetNumElements() *
                        fixt_in->GetBlocks()[blk].GetNumData();
            std::transform(
                x, x + size, y, z,
                [](const double &xi, const double &yi) { return xi * yi; });
        }
    }

    void div()
    {
        for (unsigned int blk = 0; blk < fixt_in->GetBlocks().size(); ++blk)
        {
            auto x = fixt_in->GetBlocks()[blk]
                         .GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto y = fixt_in2->GetBlocks()[blk]
                         .GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto z = fixt_expected->GetBlocks()[blk]
                         .GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            auto size = fixt_in->GetBlocks()[blk].GetNumElements() *
                        fixt_in->GetBlocks()[blk].GetNumData();
            std::transform(
                x, x + size, y, z,
                [](const double &xi, const double &yi) { return xi / yi; });
        }
    }

    void daxpy(double alpha)
    {
        for (unsigned int blk = 0; blk < fixt_in->GetBlocks().size(); ++blk)
        {
            auto x = fixt_in->GetBlocks()[blk]
                         .GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto y = fixt_in2->GetBlocks()[blk]
                         .GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto z = fixt_expected->GetBlocks()[blk]
                         .GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            auto size = fixt_in->GetBlocks()[blk].GetNumElements() *
                        fixt_in->GetBlocks()[blk].GetNumData();
            std::transform(x, x + size, y, z,
                           [=](const double &xi, const double &yi) {
                               return alpha * xi + yi;
                           });
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

    double inner_product()
    {
        double out = 0;
        for (unsigned int blk = 0; blk < fixt_in->GetBlocks().size(); ++blk)
        {
            auto x = fixt_in->GetBlocks()[blk]
                         .GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto y = fixt_in2->GetBlocks()[blk]
                         .GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto size = fixt_in->GetBlocks()[blk].GetNumElements() *
                        fixt_in->GetBlocks()[blk].GetNumData();
            out = std::inner_product(x, x + size, y, out);
        }
        return out;
    }

    double l1norm()
    {
        double out = 0;
        for (unsigned int blk = 0; blk < fixt_in->GetBlocks().size(); ++blk)
        {
            auto x = fixt_in->GetBlocks()[blk]
                         .GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto size = fixt_in->GetBlocks()[blk].GetNumElements() *
                        fixt_in->GetBlocks()[blk].GetNumData();
            out = std::accumulate(x, x + size, out,
                                  [](const double &acc, const double &val) {
                                      return acc + std::abs(val);
                                  });
        }
        return out;
    }

    double l2norm()
    {
        double out = 0;
        for (unsigned int blk = 0; blk < fixt_in->GetBlocks().size(); ++blk)
        {
            auto x = fixt_in->GetBlocks()[blk]
                         .GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto size = fixt_in->GetBlocks()[blk].GetNumElements() *
                        fixt_in->GetBlocks()[blk].GetNumData();
            out = std::accumulate(x, x + size, out,
                                  [](const double &acc, const double &val) {
                                      return acc + val * val;
                                  });
        }
        return out;
    }

    double lpnorm(const unsigned int p)
    {
        double out = 0;
        for (unsigned int blk = 0; blk < fixt_in->GetBlocks().size(); ++blk)
        {
            auto x = fixt_in->GetBlocks()[blk]
                         .GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto size = fixt_in->GetBlocks()[blk].GetNumElements() *
                        fixt_in->GetBlocks()[blk].GetNumData();
            out = std::accumulate(x, x + size, out,
                                  [&](const double &acc, const double &val) {
                                      return acc + std::pow(std::abs(val), p);
                                  });
        }
        return out;
    }

    double linfnorm()
    {
        double out = std::numeric_limits<double>::min();
        for (unsigned int blk = 0; blk < fixt_in->GetBlocks().size(); ++blk)
        {
            auto x = fixt_in->GetBlocks()[blk]
                         .GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto size = fixt_in->GetBlocks()[blk].GetNumElements() *
                        fixt_in->GetBlocks()[blk].GetNumData();
            out = std::accumulate(
                x, x + size, out, [](const double &acc, const double &val) {
                    return std::max(std::abs(acc), std::abs(val));
                });
        }
        return out;
    }

protected:
    Field<double, FieldState::Phys> *fixt_in2 = nullptr;
    Math math;
};
