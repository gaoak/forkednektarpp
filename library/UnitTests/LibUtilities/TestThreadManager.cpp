///////////////////////////////////////////////////////////////////////////////
//
// File: TestThreadManager.cpp
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
// Description: unit tests for ThreadManagerStd
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/Thread.h>
#include <LibUtilities/BasicUtils/ThreadStd.h>

#include <atomic>
#include <vector>

#include <boost/test/unit_test.hpp>

namespace Nektar::ThreadManagerUnitTests
{

namespace
{
std::atomic<int> g_jobsRun{0};

/// A job which does nothing but record that it ran.
class CountingJob : public Thread::ThreadJob
{
public:
    void Run() override
    {
        ++g_jobsRun;
    }
};

/**
 * @brief Puts @p numJobs jobs through a manager with @p numWorkers workers.
 *
 * Deactivating the workers, filling the queue and activating them again is
 * how ProcessVarOpti drives the manager, so that is the shape used here.
 */
void RunOneCycle(const unsigned int numWorkers, const int numJobs)
{
    Thread::ThreadManagerSharedPtr tm =
        Thread::ThreadManagerStd::Create(numWorkers);

    tm->SetNumWorkers(0);

    std::vector<Thread::ThreadJob *> joblist;
    joblist.reserve(numJobs);
    for (int j = 0; j < numJobs; ++j)
    {
        joblist.push_back(new CountingJob());
    }
    tm->QueueJobs(joblist);

    tm->SetNumWorkers(numWorkers);
    tm->Wait();
    // tm leaves scope here, so the manager is torn down and its workers
    // joined.
}
} // namespace

BOOST_AUTO_TEST_CASE(TestThreadManagerRunsEveryJob)
{
    g_jobsRun = 0;
    RunOneCycle(4, 32);
    BOOST_CHECK_EQUAL(g_jobsRun.load(), 32);
}

/**
 * Tears a manager down repeatedly, which is where the workers' shutdown flag
 * races with their waits.
 *
 * A worker holds one of the two master mutexes from reading that flag until
 * its wait registers on the condition variable. Clearing the flag outside
 * those mutexes can land in between, and then the notification which follows
 * reaches nothing, the worker waits for a wake-up already sent, and the join
 * in the destructor waits on the worker.
 *
 * @note A regression here hangs rather than fails, so this case relies on the
 * test binary's timeout to report it. Unfixed, four workers hung inside the
 * first hundred cycles; two hung within a thousand. Five hundred cycles cost
 * about three tenths of a second.
 */
BOOST_AUTO_TEST_CASE(TestThreadManagerTeardownDoesNotHang)
{
    g_jobsRun = 0;

    constexpr int nCycles = 500;
    for (int c = 0; c < nCycles; ++c)
    {
        RunOneCycle(4, 8);
    }

    BOOST_CHECK_EQUAL(g_jobsRun.load(), nCycles * 8);
}

/// The same teardown, for worker counts either side of the four above.
BOOST_AUTO_TEST_CASE(TestThreadManagerTeardownWorkerCounts)
{
    for (unsigned int numWorkers : {1u, 2u, 8u})
    {
        g_jobsRun = 0;
        for (int c = 0; c < 100; ++c)
        {
            RunOneCycle(numWorkers, 8);
        }
        BOOST_CHECK_EQUAL(g_jobsRun.load(), 100 * 8);
    }
}

} // namespace Nektar::ThreadManagerUnitTests
