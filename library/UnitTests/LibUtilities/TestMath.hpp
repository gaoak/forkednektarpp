///////////////////////////////////////////////////////////////////////////////
//
// File: TestMath.hpp
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

#pragma once

#include <MultiRegions/ExpList.h>
#include <SpatialDomains/MeshGraphIO.h>

#include "LibUtilities/BasicUtils/Math/MathHelper.hpp"
#include <LibUtilities/BasicUtils/Field/Field.hpp>

#include <UnitTests/TestBoostSetup.hpp>
#include <UnitTests/TestGlobalConfiguration.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#undef max
#undef min
#endif

using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::MultiRegions;

NEKTAR_TEST_GLOBAL_CONFIGURATION(Nektar::UnitTests::TestArgs::None);

template <typename TData> class TestMath
{
public:
    TestMath()
    {
        m_meshName = "run/segment.xml";
    }

    ~TestMath()
    {
        BOOST_TEST_MESSAGE("teardown this->fixture");

        if (this->m_in1)
        {
            delete this->m_in1;
        }
        if (this->m_in2)
        {
            delete this->m_in2;
        }
        if (this->m_out)
        {
            delete this->m_out;
        }
        if (this->m_expected)
        {
            delete this->m_expected;
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
        argv[1]     = strdup(m_meshName.data());

        this->m_session = SessionReader::CreateInstance(argc, argv);
        auto graph      = SpatialDomains::MeshGraphIO::Read(this->m_session);

        for (int i = 0; i < argc; ++i)
        {
            free(argv[i]);
        }
        delete[] argv;

        this->m_expList = MemoryManager<ExpList>::AllocateSharedPtr(
            this->m_session, graph, true, "u", Collections::eNoCollection);

        // One component per session variable, so the kernels are exercised
        // over a field wider than a single component.
        const auto &components = this->m_session->GetVariables();
        auto blockAttr = GetBlockAttributes<TData, FieldState::Phys>(m_expList);
        auto f_in1 =
            Field<TData, FieldState::Phys>("f_in1", blockAttr, components, 1);
        auto f_in2 =
            Field<TData, FieldState::Phys>("f_in2", blockAttr, components, 1);
        auto f_out =
            Field<TData, FieldState::Phys>("f_out", blockAttr, components, 1);
        auto f_expected = Field<TData, FieldState::Phys>(
            "f_expected", blockAttr, components, 1);
        this->m_in1 = new Field<TData, FieldState::Phys>(std::move(f_in1));
        this->m_in2 = new Field<TData, FieldState::Phys>(std::move(f_in2));
        this->m_out = new Field<TData, FieldState::Phys>(std::move(f_out));
        this->m_expected =
            new Field<TData, FieldState::Phys>(std::move(f_expected));

        std::string execName(
            boost::unit_test::framework::master_test_suite().argv[1]);
        m_math = Math::MathHelper(execName);
    }

    void SetTestCase()
    {
        // GetCoords() and Equation::Evaluate() take double arrays, whatever
        // TData is; the values are narrowed when copied into the field.
        Array<OneD, double> x(this->m_expList->GetTotPoints(), 0.0);
        Array<OneD, double> y(this->m_expList->GetTotPoints(), 0.0);
        Array<OneD, double> z(this->m_expList->GetTotPoints(), 0.0);
        Array<OneD, double> fce(this->m_expList->GetTotPoints());
        this->m_expList->GetCoords(x, y, z);

        Evaluate("Forcing", *this->m_in1, x, y, z, fce);
        Evaluate("ExactSolution", *this->m_in2, x, y, z, fce);
    }

    void abs()
    {
        for (unsigned int blk = 0; blk < this->m_in1->GetBlocks().size(); ++blk)
        {
            auto x = this->m_in1->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto y = this->m_expected->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            auto size = this->m_in1->GetBlocks()[blk].GetNumElements() *
                        this->m_in1->GetBlocks()[blk].GetNumData();
            auto stride = this->m_in1->GetBlocks()[blk].CompSize();

            for (unsigned int nc = 0; nc < this->m_in1->GetNumComponents();
                 ++nc, x += stride, y += stride)
            {
                std::transform(x, x + size, y,
                               [](const TData &xi) { return std::abs(xi); });
            }
        }
    }

    void neg()
    {
        for (unsigned int blk = 0; blk < this->m_in1->GetBlocks().size(); ++blk)
        {
            auto x = this->m_in1->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto y = this->m_expected->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            auto size = this->m_in1->GetBlocks()[blk].GetNumElements() *
                        this->m_in1->GetBlocks()[blk].GetNumData();
            auto stride = this->m_in1->GetBlocks()[blk].CompSize();

            for (unsigned int nc = 0; nc < this->m_in1->GetNumComponents();
                 ++nc, x += stride, y += stride)
            {
                std::transform(x, x + size, y,
                               [](const TData &xi) { return -xi; });
            }
        }
    }

    void sqrt()
    {
        for (unsigned int blk = 0; blk < this->m_in1->GetBlocks().size(); ++blk)
        {
            auto x = this->m_in1->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto y = this->m_expected->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            auto size = this->m_in1->GetBlocks()[blk].GetNumElements() *
                        this->m_in1->GetBlocks()[blk].GetNumData();
            auto stride = this->m_in1->GetBlocks()[blk].CompSize();

            for (unsigned int nc = 0; nc < this->m_in1->GetNumComponents();
                 ++nc, x += stride, y += stride)
            {
                // If inputs might be negative and want to avoid NaNs,
                // replace std::sqrt(xi) with std::sqrt(std::abs(xi)).
                std::transform(x, x + size, y, [](const TData &xi) {
                    return std::sqrt(std::abs(xi));
                });
            }
        }
    }

    void add()
    {
        for (unsigned int blk = 0; blk < this->m_in1->GetBlocks().size(); ++blk)
        {
            auto x = this->m_in1->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto y = this->m_in2->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto z = this->m_expected->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            auto size = this->m_in1->GetBlocks()[blk].GetNumElements() *
                        this->m_in1->GetBlocks()[blk].GetNumData();
            auto stride = this->m_in1->GetBlocks()[blk].CompSize();

            for (unsigned int nc = 0; nc < this->m_in1->GetNumComponents();
                 ++nc, x += stride, y += stride, z += stride)
            {
                std::transform(
                    x, x + size, y, z,
                    [](const TData &xi, const TData &yi) { return xi + yi; });
            }
        }
    }

    void sub()
    {
        for (unsigned int blk = 0; blk < this->m_in1->GetBlocks().size(); ++blk)
        {
            auto x = this->m_in1->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto y = this->m_in2->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto z = this->m_expected->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            auto size = this->m_in1->GetBlocks()[blk].GetNumElements() *
                        this->m_in1->GetBlocks()[blk].GetNumData();
            auto stride = this->m_in1->GetBlocks()[blk].CompSize();

            for (unsigned int nc = 0; nc < this->m_in1->GetNumComponents();
                 ++nc, x += stride, y += stride, z += stride)
            {
                std::transform(
                    x, x + size, y, z,
                    [](const TData &xi, const TData &yi) { return xi - yi; });
            }
        }
    }

    void mul()
    {
        for (unsigned int blk = 0; blk < this->m_in1->GetBlocks().size(); ++blk)
        {
            auto x = this->m_in1->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto y = this->m_in2->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto z = this->m_expected->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            auto size = this->m_in1->GetBlocks()[blk].GetNumElements() *
                        this->m_in1->GetBlocks()[blk].GetNumData();
            auto stride = this->m_in1->GetBlocks()[blk].CompSize();

            for (unsigned int nc = 0; nc < this->m_in1->GetNumComponents();
                 ++nc, x += stride, y += stride, z += stride)
            {
                std::transform(
                    x, x + size, y, z,
                    [](const TData &xi, const TData &yi) { return xi * yi; });
            }
        }
    }

    void div()
    {
        for (unsigned int blk = 0; blk < this->m_in1->GetBlocks().size(); ++blk)
        {
            auto x = this->m_in1->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto y = this->m_in2->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto z = this->m_expected->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            auto size = this->m_in1->GetBlocks()[blk].GetNumElements() *
                        this->m_in1->GetBlocks()[blk].GetNumData();
            auto stride = this->m_in1->GetBlocks()[blk].CompSize();

            for (unsigned int nc = 0; nc < this->m_in1->GetNumComponents();
                 ++nc, x += stride, y += stride, z += stride)
            {
                std::transform(
                    x, x + size, y, z,
                    [](const TData &xi, const TData &yi) { return xi / yi; });
            }
        }
    }

    void daxpy(TData alpha)
    {
        for (unsigned int blk = 0; blk < this->m_in1->GetBlocks().size(); ++blk)
        {
            auto x = this->m_in1->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto y = this->m_in2->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto z = this->m_expected->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            auto size = this->m_in1->GetBlocks()[blk].GetNumElements() *
                        this->m_in1->GetBlocks()[blk].GetNumData();
            auto stride = this->m_in1->GetBlocks()[blk].CompSize();

            for (unsigned int nc = 0; nc < this->m_in1->GetNumComponents();
                 ++nc, x += stride, y += stride, z += stride)
            {
                std::transform(x, x + size, y, z,
                               [=](const TData &xi, const TData &yi) {
                                   return alpha * xi + yi;
                               });
            }
        }
    }

    void daxpby(TData alpha, TData beta)
    {
        for (unsigned int blk = 0; blk < this->m_in1->GetBlocks().size(); ++blk)
        {
            auto x = this->m_in1->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto y = this->m_in2->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto z = this->m_expected->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            auto size = this->m_in1->GetBlocks()[blk].GetNumElements() *
                        this->m_in1->GetBlocks()[blk].GetNumData();
            auto stride = this->m_in1->GetBlocks()[blk].CompSize();

            for (unsigned int nc = 0; nc < this->m_in1->GetNumComponents();
                 ++nc, x += stride, y += stride, z += stride)
            {
                std::transform(x, x + size, y, z,
                               [=](const TData &xi, const TData &yi) {
                                   return alpha * xi + beta * yi;
                               });
            }
        }
    }

    /// w = alpha x + beta y + gamma z, with z = x.
    void daxpbypcz(TData alpha, TData beta, TData gamma)
    {
        for (unsigned int blk = 0; blk < this->m_in1->GetBlocks().size(); ++blk)
        {
            auto x = this->m_in1->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto y = this->m_in2->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto z = this->m_expected->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            auto size = this->m_in1->GetBlocks()[blk].GetNumElements() *
                        this->m_in1->GetBlocks()[blk].GetNumData();
            auto stride = this->m_in1->GetBlocks()[blk].CompSize();

            for (unsigned int nc = 0; nc < this->m_in1->GetNumComponents();
                 ++nc, x += stride, y += stride, z += stride)
            {
                std::transform(x, x + size, y, z,
                               [=](const TData &xi, const TData &yi) {
                                   return alpha * xi + beta * yi + gamma * xi;
                               });
            }
        }
    }

    TData sum()
    {
        TData out = 0;
        for (unsigned int blk = 0; blk < this->m_in1->GetBlocks().size(); ++blk)
        {
            auto x = this->m_in1->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto size = this->m_in1->GetBlocks()[blk].GetNumElements() *
                        this->m_in1->GetBlocks()[blk].GetNumData();
            auto stride = this->m_in1->GetBlocks()[blk].CompSize();

            for (unsigned int nc = 0; nc < this->m_in1->GetNumComponents();
                 ++nc, x += stride)
            {
                out = std::accumulate(x, x + size, out);
            }
        }
        return out;
    }

    TData max()
    {
        TData out = std::numeric_limits<TData>::lowest();
        for (unsigned int blk = 0; blk < this->m_in1->GetBlocks().size(); ++blk)
        {
            auto x = this->m_in1->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto size = this->m_in1->GetBlocks()[blk].GetNumElements() *
                        this->m_in1->GetBlocks()[blk].GetNumData();
            auto stride = this->m_in1->GetBlocks()[blk].CompSize();

            for (unsigned int nc = 0; nc < this->m_in1->GetNumComponents();
                 ++nc, x += stride)
            {
                out = std::max(out, *(std::max_element(x, x + size)));
            }
        }
        return out;
    }

    TData min()
    {
        TData out = std::numeric_limits<TData>::max();
        for (unsigned int blk = 0; blk < this->m_in1->GetBlocks().size(); ++blk)
        {
            auto x = this->m_in1->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto size = this->m_in1->GetBlocks()[blk].GetNumElements() *
                        this->m_in1->GetBlocks()[blk].GetNumData();
            auto stride = this->m_in1->GetBlocks()[blk].CompSize();

            for (unsigned int nc = 0; nc < this->m_in1->GetNumComponents();
                 ++nc, x += stride)
            {
                out = std::min(out, *(std::min_element(x, x + size)));
            }
        }
        return out;
    }

    TData inner_product()
    {
        TData out = 0;
        for (unsigned int blk = 0; blk < this->m_in1->GetBlocks().size(); ++blk)
        {
            auto x = this->m_in1->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto y = this->m_in2->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto size = this->m_in1->GetBlocks()[blk].GetNumElements() *
                        this->m_in1->GetBlocks()[blk].GetNumData();
            auto stride = this->m_in1->GetBlocks()[blk].CompSize();

            for (unsigned int nc = 0; nc < this->m_in1->GetNumComponents();
                 ++nc, x += stride, y += stride)
            {
                out = std::inner_product(x, x + size, y, out);
            }
        }
        return out;
    }

    TData l1norm()
    {
        TData out = 0;
        for (unsigned int blk = 0; blk < this->m_in1->GetBlocks().size(); ++blk)
        {
            auto x = this->m_in1->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto size = this->m_in1->GetBlocks()[blk].GetNumElements() *
                        this->m_in1->GetBlocks()[blk].GetNumData();
            auto stride = this->m_in1->GetBlocks()[blk].CompSize();

            for (unsigned int nc = 0; nc < this->m_in1->GetNumComponents();
                 ++nc, x += stride)
            {
                out = std::accumulate(x, x + size, out,
                                      [](const TData &acc, const TData &val) {
                                          return acc + std::abs(val);
                                      });
            }
        }
        return out;
    }

    TData l2norm()
    {
        TData out = 0;
        for (unsigned int blk = 0; blk < this->m_in1->GetBlocks().size(); ++blk)
        {
            auto x = this->m_in1->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto size = this->m_in1->GetBlocks()[blk].GetNumElements() *
                        this->m_in1->GetBlocks()[blk].GetNumData();
            auto stride = this->m_in1->GetBlocks()[blk].CompSize();

            for (unsigned int nc = 0; nc < this->m_in1->GetNumComponents();
                 ++nc, x += stride)
            {
                out = std::accumulate(x, x + size, out,
                                      [](const TData &acc, const TData &val) {
                                          return acc + val * val;
                                      });
            }
        }
        return out;
    }

    TData lpnorm(const unsigned int p)
    {
        TData out = 0;
        for (unsigned int blk = 0; blk < this->m_in1->GetBlocks().size(); ++blk)
        {
            auto x = this->m_in1->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto size = this->m_in1->GetBlocks()[blk].GetNumElements() *
                        this->m_in1->GetBlocks()[blk].GetNumData();
            auto stride = this->m_in1->GetBlocks()[blk].CompSize();

            for (unsigned int nc = 0; nc < this->m_in1->GetNumComponents();
                 ++nc, x += stride)
            {
                out = std::accumulate(
                    x, x + size, out, [&](const TData &acc, const TData &val) {
                        return acc + std::pow(std::abs(val), p);
                    });
            }
        }
        return out;
    }

    TData linfnorm()
    {
        TData out = 0.0;
        for (unsigned int blk = 0; blk < this->m_in1->GetBlocks().size(); ++blk)
        {
            auto x = this->m_in1->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto size = this->m_in1->GetBlocks()[blk].GetNumElements() *
                        this->m_in1->GetBlocks()[blk].GetNumData();
            auto stride = this->m_in1->GetBlocks()[blk].CompSize();

            for (unsigned int nc = 0; nc < this->m_in1->GetNumComponents();
                 ++nc, x += stride)
            {
                out = std::accumulate(
                    x, x + size, out, [](const TData &acc, const TData &val) {
                        return std::max(std::abs(acc), std::abs(val));
                    });
            }
        }
        return out;
    }

    /**
     * @brief Compare this field to another field, with absolute
     * tolerance tol. Two fields must have same storage shape
     * and same components.
     *
     * @return bool
     */
    bool Compare(TData tol)
    {
        auto rank = this->m_session->GetComm()->GetRank();

        if (this->m_expected->GetNumComponents() !=
            this->m_out->GetNumComponents())
        {
            std::cout << "Mismatch of number of components." << std::endl;
            return false;
        }

        if (this->m_expected->GetNumHomoModes() !=
            this->m_out->GetNumHomoModes())
        {
            std::cout << "Mismatch of number of homogeneous modes."
                      << std::endl;
            return false;
        }

        if (this->m_expected->GetBlocks().size() !=
            this->m_out->GetBlocks().size())
        {
            std::cout << "Mismatch of block size." << std::endl;
            return false;
        }

        bool isMatch = true;

        if (rank == 0)
        {
            printf("#elm #pts output               expected            "
                   "difference\n");
        }
        for (unsigned int blk = 0; blk < this->m_out->GetBlocks().size(); ++blk)
        {
            const TData *outptr =
                this->m_out->GetBlocks()[blk]
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            const TData *expptr =
                this->m_expected->GetBlocks()[blk]
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

            if ((this->m_out->GetBlocks()[blk].GetNumElements() !=
                 this->m_expected->GetBlocks()[blk].GetNumElements()) ||
                (this->m_out->GetBlocks()[blk].GetNumData() !=
                 this->m_expected->GetBlocks()[blk].GetNumData()))
            {
                std::cout << "Mismatch of block structure." << std::endl;
                return false;
            }

            for (unsigned int n = 0; n < this->m_out->GetNumComponents() *
                                             this->m_out->GetNumHomoModes();
                 ++n)
            {
                size_t MisMatchcnt = 0, total = 0;

                for (size_t el = 0, cnt = 0;
                     el < this->m_out->GetBlocks()[blk].GetNumElements(); ++el)
                {
                    for (unsigned int pts = 0;
                         pts < this->m_out->GetBlocks()[blk].GetNumData();
                         ++pts, ++cnt)
                    {
                        if (std::isnan(outptr[cnt]) ||
                            std::isinf(outptr[cnt]) ||
                            std::abs(outptr[cnt] - expptr[cnt]) > tol)
                        {
                            printf("%04lu %04u %20.16f %20.16f %20.16f\n", el,
                                   pts, outptr[cnt], expptr[cnt],
                                   std::abs(outptr[cnt] - expptr[cnt]));
                            MisMatchcnt++;
                        }
                        total++;
                    }
                }

                outptr += this->m_out->GetBlocks()[blk].CompSize();
                expptr += this->m_expected->GetBlocks()[blk].CompSize();

                if (MisMatchcnt)
                {
                    std::cout << "Number of mismatches in component " << n
                              << " on block " << blk << " is " << MisMatchcnt
                              << " out of " << total << " on rank: " << rank
                              << std::endl;
                    isMatch = false;
                }
            }
        }

        if (isMatch)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

protected:
    std::string m_meshName                     = "";
    Field<TData, FieldState::Phys> *m_in1      = nullptr;
    Field<TData, FieldState::Phys> *m_in2      = nullptr;
    Field<TData, FieldState::Phys> *m_out      = nullptr;
    Field<TData, FieldState::Phys> *m_expected = nullptr;
    ExpListSharedPtr m_expList;
    SessionReaderSharedPtr m_session;
    Math::MathHelper m_math;

    /**
     * @brief Fills every component of @p field with the session function
     * @p name evaluated for the variable of that component.
     */
    void Evaluate(const std::string &name,
                  Field<TData, FieldState::Phys> &field, Array<OneD, double> &x,
                  Array<OneD, double> &y, Array<OneD, double> &z,
                  Array<OneD, double> &fce)
    {
        for (unsigned int nc = 0; nc < field.GetNumComponents(); ++nc)
        {
            auto func = this->m_session->GetFunction(name, nc);
            func->Evaluate(x, y, z, fce);

            auto ptr = fce.data();
            for (unsigned int blk = 0; blk < field.GetBlocks().size(); ++blk)
            {
                auto &block = field.GetBlocks()[blk];
                auto inptr =
                    block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
                auto size = block.GetNumElements() * block.GetNumData();
                std::copy(ptr, ptr + size, inptr + nc * block.CompSize());
                ptr += size;
            }
        }
    }
};

#include <UnitTests/TestBoostTeardown.hpp>
