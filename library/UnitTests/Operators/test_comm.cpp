///////////////////////////////////////////////////////////////////////////////
//
// File: test_comm.cpp
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

#define BOOST_TEST_MODULE TestComm

#include "init_comm.hpp"

#include <boost/test/tools/output_test_stream.hpp>
#include <iostream>
#include <memory>

#if defined(_MSC_VER)
#undef max
#undef min
#endif

using namespace Nektar::Operators;

BOOST_AUTO_TEST_SUITE(TestComm)

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
BOOST_FIXTURE_TEST_CASE(bcast, InitComm)
{
    std::string execStr(
        boost::unit_test::framework::master_test_suite().argv[1]);

    Configure();
    double val = 3.0;
    auto bcast = MemoryRegion<double>("Bcast", 10);

    // Initialize data.
    if (m_comm->GetRank() == 0)
    {
        if (execStr == "HostSpace")
        {
            bcast.template Initialize<NektarSpaces::HostSpace>(val);
        }
        else if (execStr == "DeviceSpace")
        {
            bcast.template Initialize<NektarSpaces::DeviceSpace>(val);
        }
    }

    // Bcast data.
    if (execStr == "HostSpace")
    {
        m_comm->Bcast<NektarSpaces::HostSpace>(bcast, 0);
    }
    else if (execStr == "DeviceSpace")
    {
        m_comm->Bcast<NektarSpaces::DeviceSpace>(bcast, 0);
    }

    // Check results.
    auto ptr = bcast.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(std::count(ptr, ptr + bcast.size(), val) == bcast.size());
    }
}

BOOST_FIXTURE_TEST_CASE(send_and_recv, InitComm)
{
    std::string execStr(
        boost::unit_test::framework::master_test_suite().argv[1]);

    Configure();
    double val0 = 3.0;
    double val1 = 5.0;
    auto send   = MemoryRegion<double>("Send", 10);
    auto recv   = MemoryRegion<double>("Recv", 10);

    // Initialize data.
    if (m_comm->GetRank() == 0)
    {
        if (execStr == "HostSpace")
        {
            send.template Initialize<NektarSpaces::HostSpace>(val0);
        }
        else if (execStr == "DeviceSpace")
        {
            send.template Initialize<NektarSpaces::DeviceSpace>(val0);
        }
    }
    else if (m_comm->GetRank() == 1)
    {
        if (execStr == "HostSpace")
        {
            send.template Initialize<NektarSpaces::HostSpace>(val1);
        }
        else if (execStr == "DeviceSpace")
        {
            send.template Initialize<NektarSpaces::DeviceSpace>(val1);
        }
    }

    // Send/Receive data.
    if (m_comm->GetRank() == 0)
    {
        if (execStr == "HostSpace")
        {
            m_comm->Send<NektarSpaces::HostSpace>(1, send);
            m_comm->Recv<NektarSpaces::HostSpace>(1, recv);
        }
        else if (execStr == "DeviceSpace")
        {
            m_comm->Send<NektarSpaces::DeviceSpace>(1, send);
            m_comm->Recv<NektarSpaces::DeviceSpace>(1, recv);
        }
    }
    else if (m_comm->GetRank() == 1)
    {
        if (execStr == "HostSpace")
        {
            m_comm->Recv<NektarSpaces::HostSpace>(0, recv);
            m_comm->Send<NektarSpaces::HostSpace>(0, send);
        }
        else if (execStr == "DeviceSpace")
        {
            m_comm->Recv<NektarSpaces::DeviceSpace>(0, recv);
            m_comm->Send<NektarSpaces::DeviceSpace>(0, send);
        }
    }

    // Check results.
    auto ptr = recv.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
    if (m_comm->GetRank() == 0)
    {
        boost::test_tools::output_test_stream output;
        {
            BOOST_TEST(std::count(ptr, ptr + recv.size(), val1) == recv.size());
        }
    }
    else if (m_comm->GetRank() == 1)
    {
        boost::test_tools::output_test_stream output;
        {
            BOOST_TEST(std::count(ptr, ptr + recv.size(), val0) == recv.size());
        }
    }
}

BOOST_FIXTURE_TEST_CASE(sendrecv, InitComm)
{
    std::string execStr(
        boost::unit_test::framework::master_test_suite().argv[1]);

    Configure();
    double val0 = 3.0;
    double val1 = 5.0;
    auto send   = MemoryRegion<double>("Send", 10);
    auto recv   = MemoryRegion<double>("Recv", 10);

    // Initialize data.
    if (m_comm->GetRank() == 0)
    {
        if (execStr == "HostSpace")
        {
            send.template Initialize<NektarSpaces::HostSpace>(val0);
        }
        else if (execStr == "DeviceSpace")
        {
            send.template Initialize<NektarSpaces::DeviceSpace>(val0);
        }
    }
    else if (m_comm->GetRank() == 1)
    {
        if (execStr == "HostSpace")
        {
            send.template Initialize<NektarSpaces::HostSpace>(val1);
        }
        else if (execStr == "DeviceSpace")
        {
            send.template Initialize<NektarSpaces::DeviceSpace>(val1);
        }
    }

    // Send/Receive data.
    int toRank   = (m_comm->GetRank() == 0) ? 1 : 0;
    int fromRank = (m_comm->GetRank() == 0) ? 1 : 0;
    if (execStr == "HostSpace")
    {
        m_comm->SendRecv<NektarSpaces::HostSpace>(toRank, send, fromRank, recv);
    }
    else if (execStr == "DeviceSpace")
    {
        m_comm->SendRecv<NektarSpaces::DeviceSpace>(toRank, send, fromRank,
                                                    recv);
    }

    // Check results.
    auto ptr = recv.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
    if (m_comm->GetRank() == 0)
    {
        boost::test_tools::output_test_stream output;
        {
            BOOST_TEST(std::count(ptr, ptr + recv.size(), val1) == recv.size());
        }
    }
    else if (m_comm->GetRank() == 1)
    {
        boost::test_tools::output_test_stream output;
        {
            BOOST_TEST(std::count(ptr, ptr + recv.size(), val0) == recv.size());
        }
    }
}

BOOST_FIXTURE_TEST_CASE(allreduce, InitComm)
{
    std::string execStr(
        boost::unit_test::framework::master_test_suite().argv[1]);

    Configure();
    double val0  = 3.0;
    double val1  = 5.0;
    auto reducer = MemoryRegion<double>("Send", 10);

    // Initialize data.
    if (m_comm->GetRank() == 0)
    {
        if (execStr == "HostSpace")
        {
            reducer.template Initialize<NektarSpaces::HostSpace>(val0);
        }
        else if (execStr == "DeviceSpace")
        {
            reducer.template Initialize<NektarSpaces::DeviceSpace>(val0);
        }
    }
    else if (m_comm->GetRank() == 1)
    {
        if (execStr == "HostSpace")
        {
            reducer.template Initialize<NektarSpaces::HostSpace>(val1);
        }
        else if (execStr == "DeviceSpace")
        {
            reducer.template Initialize<NektarSpaces::DeviceSpace>(val1);
        }
    }

    // Send/Receive data.
    if (execStr == "HostSpace")
    {
        m_comm->AllReduce<NektarSpaces::HostSpace>(
            reducer, Nektar::LibUtilities::ReduceSum);
    }
    else if (execStr == "DeviceSpace")
    {
        m_comm->AllReduce<NektarSpaces::DeviceSpace>(
            reducer, Nektar::LibUtilities::ReduceSum);
    }

    // Check results.
    auto ptr = reducer.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(std::count(ptr, ptr + reducer.size(), val0 + val1) ==
                   reducer.size());
    }
}

BOOST_FIXTURE_TEST_CASE(scatter, InitComm)
{
    std::string execStr(
        boost::unit_test::framework::master_test_suite().argv[1]);

    Configure();
    double val0  = 3.0;
    double val1  = 5.0;
    auto scatter = MemoryRegion<double>("Scatter", 20);

    // Initialize data.
    if (m_comm->GetRank() == 0)
    {
        if (execStr == "HostSpace")
        {
            scatter.template Initialize<NektarSpaces::HostSpace>(val0, 10);
            scatter.template Initialize<NektarSpaces::HostSpace>(val1, 10, 10);
        }
        else if (execStr == "DeviceSpace")
        {
            scatter.template Initialize<NektarSpaces::DeviceSpace>(val0, 10);
            scatter.template Initialize<NektarSpaces::DeviceSpace>(val1, 10,
                                                                   10);
        }
    }

    // Send/Receive data.
    MemoryRegion<double> ans;
    if (execStr == "HostSpace")
    {
        ans = m_comm->Scatter<NektarSpaces::HostSpace>(0, scatter);
    }
    else if (execStr == "DeviceSpace")
    {
        ans = m_comm->Scatter<NektarSpaces::DeviceSpace>(0, scatter);
    }

    // Check results.
    auto ptr = ans.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
    if (m_comm->GetRank() == 0)
    {
        boost::test_tools::output_test_stream output;
        {
            BOOST_TEST(std::count(ptr, ptr + ans.size(), val0) == ans.size());
        }
    }
    else if (m_comm->GetRank() == 1)
    {
        boost::test_tools::output_test_stream output;
        {
            BOOST_TEST(std::count(ptr, ptr + ans.size(), val1) == ans.size());
        }
    }
}

BOOST_FIXTURE_TEST_CASE(alltoall, InitComm)
{
    std::string execStr(
        boost::unit_test::framework::master_test_suite().argv[1]);

    Configure();
    double val0 = 3.0;
    double val1 = 5.0;
    auto send   = MemoryRegion<double>("send", 20);
    auto recv   = MemoryRegion<double>("recv", 20);

    // Initialize data.
    if (m_comm->GetRank() == 0)
    {
        if (execStr == "HostSpace")
        {
            send.template Initialize<NektarSpaces::HostSpace>(val0, 10);
            send.template Initialize<NektarSpaces::HostSpace>(val1, 10, 10);
        }
        else if (execStr == "DeviceSpace")
        {
            send.template Initialize<NektarSpaces::DeviceSpace>(val0, 10);
            send.template Initialize<NektarSpaces::DeviceSpace>(val1, 10, 10);
        }
    }
    else if (m_comm->GetRank() == 1)
    {
        if (execStr == "HostSpace")
        {
            send.template Initialize<NektarSpaces::HostSpace>(2 * val0, 10);
            send.template Initialize<NektarSpaces::HostSpace>(2 * val1, 10, 10);
        }
        else if (execStr == "DeviceSpace")
        {
            send.template Initialize<NektarSpaces::DeviceSpace>(2 * val0, 10);
            send.template Initialize<NektarSpaces::DeviceSpace>(2 * val1, 10,
                                                                10);
        }
    }

    // Send/Receive data.
    if (execStr == "HostSpace")
    {
        m_comm->AlltoAll<NektarSpaces::HostSpace>(send, recv);
    }
    else if (execStr == "DeviceSpace")
    {
        m_comm->AlltoAll<NektarSpaces::DeviceSpace>(send, recv);
    }

    // Check results.
    auto ptr = recv.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
    if (m_comm->GetRank() == 0)
    {
        boost::test_tools::output_test_stream output;
        {
            bool pass =
                (std::count(ptr, ptr + recv.size(), val0) == recv.size() / 2) &&
                (std::count(ptr, ptr + recv.size(), 2 * val0) ==
                 recv.size() / 2);
            BOOST_TEST(pass);
        }
    }
    else if (m_comm->GetRank() == 1)
    {
        boost::test_tools::output_test_stream output;
        {
            bool pass =
                (std::count(ptr, ptr + recv.size(), val1) == recv.size() / 2) &&
                (std::count(ptr, ptr + recv.size(), 2 * val1) ==
                 recv.size() / 2);
            BOOST_TEST(pass);
        }
    }
}

BOOST_FIXTURE_TEST_CASE(gather, InitComm)
{
    std::string execStr(
        boost::unit_test::framework::master_test_suite().argv[1]);

    Configure();
    double val0 = 3.0;
    double val1 = 5.0;
    auto mr     = MemoryRegion<double>("mr", 10);

    // Initialize data.
    if (m_comm->GetRank() == 0)
    {
        if (execStr == "HostSpace")
        {
            mr.template Initialize<NektarSpaces::HostSpace>(val0);
        }
        else if (execStr == "DeviceSpace")
        {
            mr.template Initialize<NektarSpaces::DeviceSpace>(val0);
        }
    }
    else if (m_comm->GetRank() == 1)
    {
        if (execStr == "HostSpace")
        {
            mr.template Initialize<NektarSpaces::HostSpace>(val1);
        }
        else if (execStr == "DeviceSpace")
        {
            mr.template Initialize<NektarSpaces::DeviceSpace>(val1);
        }
    }

    // Send/Receive data.
    MemoryRegion<double> ans;
    if (execStr == "HostSpace")
    {
        ans = m_comm->Gather<NektarSpaces::HostSpace>(0, mr);
    }
    else if (execStr == "DeviceSpace")
    {
        ans = m_comm->Gather<NektarSpaces::DeviceSpace>(0, mr);
    }

    // Check results.
    if (m_comm->GetRank() == 0)
    {
        auto ptr = ans.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        boost::test_tools::output_test_stream output;
        {
            bool pass = (std::count(ptr, ptr + mr.size(), val0) == mr.size()) &&
                        (std::count(ptr + mr.size(), ptr + ans.size(), val1) ==
                         mr.size());
            BOOST_TEST(pass);
        }
    }
}

BOOST_FIXTURE_TEST_CASE(allgather, InitComm)
{
    std::string execStr(
        boost::unit_test::framework::master_test_suite().argv[1]);

    Configure();
    double val0 = 3.0;
    double val1 = 5.0;
    auto send   = MemoryRegion<double>("Send", 10);
    auto recv   = MemoryRegion<double>("Recv", 20);

    // Initialize data.
    if (m_comm->GetRank() == 0)
    {
        if (execStr == "HostSpace")
        {
            send.template Initialize<NektarSpaces::HostSpace>(val0);
        }
        else if (execStr == "DeviceSpace")
        {
            send.template Initialize<NektarSpaces::DeviceSpace>(val0);
        }
    }
    else if (m_comm->GetRank() == 1)
    {
        if (execStr == "HostSpace")
        {
            send.template Initialize<NektarSpaces::HostSpace>(val1);
        }
        else if (execStr == "DeviceSpace")
        {
            send.template Initialize<NektarSpaces::DeviceSpace>(val1);
        }
    }

    // Send/Receive data.
    if (execStr == "HostSpace")
    {
        m_comm->AllGather<NektarSpaces::HostSpace>(send, recv);
    }
    else if (execStr == "DeviceSpace")
    {
        m_comm->AllGather<NektarSpaces::DeviceSpace>(send, recv);
    }

    // Check results.
    auto ptr = recv.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
    boost::test_tools::output_test_stream output;
    {
        bool pass = (std::count(ptr, ptr + recv.size(), val0) == send.size()) &&
                    (std::count(ptr, ptr + recv.size(), val1) == send.size());
        BOOST_TEST(pass);
    }
}
#endif

BOOST_AUTO_TEST_SUITE_END()
