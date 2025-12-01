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

#include "Operators/Math/Math.hpp"

using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

class MathField
{
public:
    MathField()
    {
        meshName = "run/Helmholtz3D_Hex_AllBCs_P6.xml";
    }

    ~MathField()
    {
        BOOST_TEST_MESSAGE("teardown fixture");

        if (fixt_in)
        {
            delete fixt_in;
        }
        if (fixt_in2)
        {
            delete fixt_in2;
        }
        if (fixt_out)
        {
            delete fixt_out;
        }
        if (fixt_expected)
        {
            delete fixt_expected;
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

        auto blocks =
            GetBlockAttributes<double>(FieldState::Phys, fixt_explist);
        auto f_in  = Field<double, FieldState::Phys>("f_in", blocks, 1, 1);
        auto f_in2 = Field<double, FieldState::Phys>("f_in2", blocks, 1, 1);
        auto f_out = Field<double, FieldState::Phys>("f_out", blocks, 1, 1);
        auto f_expected =
            Field<double, FieldState::Phys>("f_expected", blocks, 1, 1);
        fixt_in  = new Field<double, FieldState::Phys>(std::move(f_in));
        fixt_in2 = new Field<double, FieldState::Phys>(std::move(f_in2));
        fixt_out = new Field<double, FieldState::Phys>(std::move(f_out));
        fixt_expected =
            new Field<double, FieldState::Phys>(std::move(f_expected));

        std::string execName(
            boost::unit_test::framework::master_test_suite().argv[1]);
        math = Math(execName);
    }

    void SetTestCase()
    {
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

    /**
     * @brief Compare this field to another field, with absolute
     * tolerance tol. Two fields must have same storage shape
     * and same components.
     *
     * @return bool
     */
    bool Compare(double tol)
    {
        auto rank = session->GetComm()->GetRank();

        if (fixt_expected->GetNumComponents() != fixt_out->GetNumComponents())
        {
            std::cout << "Mismatch of number of components." << std::endl;
            return false;
        }

        if (fixt_expected->GetNumHomoModes() != fixt_out->GetNumHomoModes())
        {
            std::cout << "Mismatch of number of homogeneous modes."
                      << std::endl;
            return false;
        }

        if (fixt_expected->GetBlocks().size() != fixt_out->GetBlocks().size())
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
        for (unsigned int blk = 0; blk < fixt_out->GetBlocks().size(); ++blk)
        {
            const double *outptr =
                fixt_out->GetBlocks()[blk]
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            const double *expptr =
                fixt_expected->GetBlocks()[blk]
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

            if ((fixt_out->GetBlocks()[blk].GetNumElements() !=
                 fixt_expected->GetBlocks()[blk].GetNumElements()) ||
                (fixt_out->GetBlocks()[blk].GetNumData() !=
                 fixt_expected->GetBlocks()[blk].GetNumData()))
            {
                std::cout << "Mismatch of block structure." << std::endl;
                return false;
            }

            for (unsigned int n = 0;
                 n < fixt_out->GetNumComponents() * fixt_out->GetNumHomoModes();
                 ++n)
            {
                size_t MisMatchcnt = 0, total = 0;

                for (size_t el = 0, cnt = 0;
                     el < fixt_out->GetBlocks()[blk].GetNumElements(); ++el)
                {
                    for (unsigned int pts = 0;
                         pts < fixt_out->GetBlocks()[blk].GetNumData();
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

                outptr += fixt_out->GetBlocks()[blk].size();
                expptr += fixt_expected->GetBlocks()[blk].size();

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
    std::string meshName                           = "";
    Field<double, FieldState::Phys> *fixt_in       = nullptr;
    Field<double, FieldState::Phys> *fixt_in2      = nullptr;
    Field<double, FieldState::Phys> *fixt_out      = nullptr;
    Field<double, FieldState::Phys> *fixt_expected = nullptr;
    std::shared_ptr<MultiRegions::ExpList> fixt_explist;
    LibUtilities::SessionReaderSharedPtr session;
    std::string testModule{STRVX(BOOST_TEST_MODULE)};
    Math math;
};
