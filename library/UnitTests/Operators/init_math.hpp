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

#include <MultiRegions/ExpList.h>
#include <SpatialDomains/MeshGraphIO.h>

#include "LibUtilities/BasicUtils/Math/MathHelper.hpp"
#include <MultiRegions/Field/Field.hpp>

// Currently the BOOST_TEST_DYN_LINK is local only to this unit
// test. It is undefined at the bottom of the file.
#if defined(OPERATORS_BOOST_TEST_DYN_LINK)
#if !defined(BOOST_TEST_DYN_LINK)
#define LOCALLY_DEFINED_BOOST_TEST_DYN_LINK
#define BOOST_TEST_DYN_LINK
#endif
#endif

// Currently the BOOST_TEST_NO_MAIN is local only to this unit
// test. It is undefined at the bottom of the file.
#if defined(OPERATORS_BOOST_TEST_NO_MAIN)
#if !defined(BOOST_TEST_NO_MAIN)
#define LOCALLY_DEFINED_BOOST_TEST_NO_MAIN
#define BOOST_TEST_NO_MAIN
#endif
#endif

#if defined(BOOST_TEST_DYN_LINK) || defined(BOOST_TEST_NO_MAIN)
#define BOOST_TEST_ALTERNATIVE_INIT_API
#endif

#if defined(BOOST_TEST_DYN_LINK)
#include <boost/test/unit_test.hpp>
#else
#include <boost/test/included/unit_test.hpp>
#endif

#include <boost/test/unit_test_log.hpp>

#include <string>
#include <type_traits>
#include <vector>

#if defined(_MSC_VER)
#undef max
#undef min
#endif

using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::MultiRegions;
using namespace Nektar::Operators;

struct GlobalConfiguration
{
    GlobalConfiguration()
    {
        [[maybe_unused]] int argc =
            boost::unit_test::framework::master_test_suite().argc;
        [[maybe_unused]] char **argv =
            boost::unit_test::framework::master_test_suite().argv;

#ifdef NEKTAR_USE_MPI
        MPI_Init(&argc, &argv);
#endif
    }

    ~GlobalConfiguration()
    {
#ifdef NEKTAR_USE_MPI
        MPI_Finalize();
#endif
    }
};

#if defined(BOOST_TEST_NO_MAIN)

bool init_function()
{
    return true;
}

int main(int argc, char *argv[])
{
    GlobalConfiguration gc;

    return boost::unit_test::unit_test_main(&init_function, argc, argv);
}

#else
BOOST_TEST_GLOBAL_CONFIGURATION(GlobalConfiguration);
#endif

template <typename TData> class MathField
{
public:
    MathField()
    {
        meshName = "run/Helmholtz3D_Hex_AllBCs_P6.xml";
    }

    ~MathField()
    {
        BOOST_TEST_MESSAGE("teardown this->fixture");

        if (this->fixt_in)
        {
            delete this->fixt_in;
        }
        if (this->fixt_in2)
        {
            delete this->fixt_in2;
        }
        if (this->fixt_out)
        {
            delete this->fixt_out;
        }
        if (this->fixt_expected)
        {
            delete this->fixt_expected;
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
        argv[1]     = strdup(meshName.data());

        this->session = LibUtilities::SessionReader::CreateInstance(argc, argv);
        auto graph    = SpatialDomains::MeshGraphIO::Read(this->session);

        for (int i = 0; i < argc; ++i)
        {
            free(argv[i]);
        }
        delete[] argv;

        this->fixt_explist =
            MemoryManager<MultiRegions::ExpList>::AllocateSharedPtr(
                this->session, graph, true, "u", Collections::eNoCollection);

        auto blockAttr =
            GetBlockAttributes<double, FieldState::Phys>(fixt_explist);
        auto f_in =
            Field<double, FieldState::Phys>("f_in", blockAttr, {"u"}, 1);
        auto f_in2 =
            Field<double, FieldState::Phys>("f_in2", blockAttr, {"u"}, 1);
        auto f_out =
            Field<double, FieldState::Phys>("f_out", blockAttr, {"u"}, 1);
        auto f_expected =
            Field<TData, FieldState::Phys>("f_expected", blockAttr, {"u"}, 1);
        this->fixt_in  = new Field<TData, FieldState::Phys>(std::move(f_in));
        this->fixt_in2 = new Field<TData, FieldState::Phys>(std::move(f_in2));
        this->fixt_out = new Field<TData, FieldState::Phys>(std::move(f_out));
        this->fixt_expected =
            new Field<TData, FieldState::Phys>(std::move(f_expected));

        std::string execName(
            boost::unit_test::framework::master_test_suite().argv[1]);
        math = Math::MathHelper(execName);
    }

    void SetTestCase()
    {
        Array<OneD, TData> x(this->fixt_explist->GetTotPoints());
        Array<OneD, TData> y(this->fixt_explist->GetTotPoints());
        Array<OneD, TData> z(this->fixt_explist->GetTotPoints());
        Array<OneD, TData> fce(this->fixt_explist->GetTotPoints());
        this->fixt_explist->GetCoords(x, y, z);

        auto func1 = this->session->GetFunction("Forcing", 0);
        func1->Evaluate(x, y, z, fce);
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

        auto func2 = this->session->GetFunction("ExactSolution", 0);
        func2->Evaluate(x, y, z, fce);
        ptr = fce.data();
        for (unsigned int blk = 0; blk < this->fixt_in2->GetBlocks().size();
             ++blk)
        {
            auto inptr =
                this->fixt_in2->GetBlocks()[blk]
                    .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            auto size = this->fixt_in2->GetBlocks()[blk].GetNumElements() *
                        this->fixt_in2->GetBlocks()[blk].GetNumData();
            std::copy(ptr, ptr + size, inptr);
            ptr += size;
        }
    }

    void abs()
    {
        for (unsigned int blk = 0; blk < this->fixt_in->GetBlocks().size();
             ++blk)
        {
            auto x = this->fixt_in->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto y = this->fixt_expected->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            auto size = this->fixt_in->GetBlocks()[blk].GetNumElements() *
                        this->fixt_in->GetBlocks()[blk].GetNumData();

            std::transform(x, x + size, y,
                           [](const TData &xi) { return std::abs(xi); });
        }
    }

    void neg()
    {
        for (unsigned int blk = 0; blk < this->fixt_in->GetBlocks().size();
             ++blk)
        {
            auto x = this->fixt_in->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto y = this->fixt_expected->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            auto size = this->fixt_in->GetBlocks()[blk].GetNumElements() *
                        this->fixt_in->GetBlocks()[blk].GetNumData();
            std::transform(x, x + size, y, [](const TData &xi) { return -xi; });
        }
    }

    void sqrt()
    {
        for (unsigned int blk = 0; blk < this->fixt_in->GetBlocks().size();
             ++blk)
        {
            auto x = this->fixt_in->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto y = this->fixt_expected->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            auto size = this->fixt_in->GetBlocks()[blk].GetNumElements() *
                        this->fixt_in->GetBlocks()[blk].GetNumData();

            // If inputs might be negative and want to avoid NaNs,
            // replace std::sqrt(xi) with std::sqrt(std::abs(xi)).
            std::transform(x, x + size, y, [](const TData &xi) {
                return std::sqrt(std::abs(xi));
            });
        }
    }

    void add()
    {
        for (unsigned int blk = 0; blk < this->fixt_in->GetBlocks().size();
             ++blk)
        {
            auto x = this->fixt_in->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto y = this->fixt_in2->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto z = this->fixt_expected->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            auto size = this->fixt_in->GetBlocks()[blk].GetNumElements() *
                        this->fixt_in->GetBlocks()[blk].GetNumData();
            std::transform(
                x, x + size, y, z,
                [](const TData &xi, const TData &yi) { return xi + yi; });
        }
    }

    void sub()
    {
        for (unsigned int blk = 0; blk < this->fixt_in->GetBlocks().size();
             ++blk)
        {
            auto x = this->fixt_in->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto y = this->fixt_in2->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto z = this->fixt_expected->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            auto size = this->fixt_in->GetBlocks()[blk].GetNumElements() *
                        this->fixt_in->GetBlocks()[blk].GetNumData();
            std::transform(
                x, x + size, y, z,
                [](const TData &xi, const TData &yi) { return xi - yi; });
        }
    }

    void mul()
    {
        for (unsigned int blk = 0; blk < this->fixt_in->GetBlocks().size();
             ++blk)
        {
            auto x = this->fixt_in->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto y = this->fixt_in2->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto z = this->fixt_expected->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            auto size = this->fixt_in->GetBlocks()[blk].GetNumElements() *
                        this->fixt_in->GetBlocks()[blk].GetNumData();
            std::transform(
                x, x + size, y, z,
                [](const TData &xi, const TData &yi) { return xi * yi; });
        }
    }

    void div()
    {
        for (unsigned int blk = 0; blk < this->fixt_in->GetBlocks().size();
             ++blk)
        {
            auto x = this->fixt_in->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto y = this->fixt_in2->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto z = this->fixt_expected->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            auto size = this->fixt_in->GetBlocks()[blk].GetNumElements() *
                        this->fixt_in->GetBlocks()[blk].GetNumData();
            std::transform(
                x, x + size, y, z,
                [](const TData &xi, const TData &yi) { return xi / yi; });
        }
    }

    void daxpy(TData alpha)
    {
        for (unsigned int blk = 0; blk < this->fixt_in->GetBlocks().size();
             ++blk)
        {
            auto x = this->fixt_in->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto y = this->fixt_in2->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto z = this->fixt_expected->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            auto size = this->fixt_in->GetBlocks()[blk].GetNumElements() *
                        this->fixt_in->GetBlocks()[blk].GetNumData();
            std::transform(x, x + size, y, z,
                           [=](const TData &xi, const TData &yi) {
                               return alpha * xi + yi;
                           });
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
        TData out = std::numeric_limits<TData>::lowest();
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

    TData inner_product()
    {
        TData out = 0;
        for (unsigned int blk = 0; blk < this->fixt_in->GetBlocks().size();
             ++blk)
        {
            auto x = this->fixt_in->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto y = this->fixt_in2->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto size = this->fixt_in->GetBlocks()[blk].GetNumElements() *
                        this->fixt_in->GetBlocks()[blk].GetNumData();
            out = std::inner_product(x, x + size, y, out);
        }
        return out;
    }

    TData l1norm()
    {
        TData out = 0;
        for (unsigned int blk = 0; blk < this->fixt_in->GetBlocks().size();
             ++blk)
        {
            auto x = this->fixt_in->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto size = this->fixt_in->GetBlocks()[blk].GetNumElements() *
                        this->fixt_in->GetBlocks()[blk].GetNumData();
            out = std::accumulate(x, x + size, out,
                                  [](const TData &acc, const TData &val) {
                                      return acc + std::abs(val);
                                  });
        }
        return out;
    }

    TData l2norm()
    {
        TData out = 0;
        for (unsigned int blk = 0; blk < this->fixt_in->GetBlocks().size();
             ++blk)
        {
            auto x = this->fixt_in->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto size = this->fixt_in->GetBlocks()[blk].GetNumElements() *
                        this->fixt_in->GetBlocks()[blk].GetNumData();
            out = std::accumulate(x, x + size, out,
                                  [](const TData &acc, const TData &val) {
                                      return acc + val * val;
                                  });
        }
        return out;
    }

    TData lpnorm(const unsigned int p)
    {
        TData out = 0;
        for (unsigned int blk = 0; blk < this->fixt_in->GetBlocks().size();
             ++blk)
        {
            auto x = this->fixt_in->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto size = this->fixt_in->GetBlocks()[blk].GetNumElements() *
                        this->fixt_in->GetBlocks()[blk].GetNumData();
            out = std::accumulate(x, x + size, out,
                                  [&](const TData &acc, const TData &val) {
                                      return acc + std::pow(std::abs(val), p);
                                  });
        }
        return out;
    }

    TData linfnorm()
    {
        TData out = 0.0;
        for (unsigned int blk = 0; blk < this->fixt_in->GetBlocks().size();
             ++blk)
        {
            auto x = this->fixt_in->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto size = this->fixt_in->GetBlocks()[blk].GetNumElements() *
                        this->fixt_in->GetBlocks()[blk].GetNumData();
            out = std::accumulate(
                x, x + size, out, [](const TData &acc, const TData &val) {
                    return std::max(std::abs(acc), std::abs(val));
                });
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
        auto rank = this->session->GetComm()->GetRank();

        if (this->fixt_expected->GetNumComponents() !=
            this->fixt_out->GetNumComponents())
        {
            std::cout << "Mismatch of number of components." << std::endl;
            return false;
        }

        if (this->fixt_expected->GetNumHomoModes() !=
            this->fixt_out->GetNumHomoModes())
        {
            std::cout << "Mismatch of number of homogeneous modes."
                      << std::endl;
            return false;
        }

        if (this->fixt_expected->GetBlocks().size() !=
            this->fixt_out->GetBlocks().size())
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
        for (unsigned int blk = 0; blk < this->fixt_out->GetBlocks().size();
             ++blk)
        {
            const TData *outptr =
                this->fixt_out->GetBlocks()[blk]
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            const TData *expptr =
                this->fixt_expected->GetBlocks()[blk]
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

            if ((this->fixt_out->GetBlocks()[blk].GetNumElements() !=
                 this->fixt_expected->GetBlocks()[blk].GetNumElements()) ||
                (this->fixt_out->GetBlocks()[blk].GetNumData() !=
                 this->fixt_expected->GetBlocks()[blk].GetNumData()))
            {
                std::cout << "Mismatch of block structure." << std::endl;
                return false;
            }

            for (unsigned int n = 0; n < this->fixt_out->GetNumComponents() *
                                             this->fixt_out->GetNumHomoModes();
                 ++n)
            {
                size_t MisMatchcnt = 0, total = 0;

                for (size_t el = 0, cnt = 0;
                     el < this->fixt_out->GetBlocks()[blk].GetNumElements();
                     ++el)
                {
                    for (unsigned int pts = 0;
                         pts < this->fixt_out->GetBlocks()[blk].GetNumData();
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

                outptr += this->fixt_out->GetBlocks()[blk].CompSize();
                expptr += this->fixt_expected->GetBlocks()[blk].CompSize();

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
    std::string meshName                          = "";
    Field<TData, FieldState::Phys> *fixt_in       = nullptr;
    Field<TData, FieldState::Phys> *fixt_in2      = nullptr;
    Field<TData, FieldState::Phys> *fixt_out      = nullptr;
    Field<TData, FieldState::Phys> *fixt_expected = nullptr;
    std::shared_ptr<MultiRegions::ExpList> fixt_explist;
    LibUtilities::SessionReaderSharedPtr session;
    Math::MathHelper math;
};
