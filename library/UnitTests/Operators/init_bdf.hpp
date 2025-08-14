///////////////////////////////////////////////////////////////////////////////
//
// File: init_bdf.hpp
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
#include "Operators/TimeOps/BDF/BDFOp.hpp"

using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

class BDFField : public InitFields<double, FieldState::Phys, FieldState::Phys>
{
public:
    BDFField() : InitFields<double, FieldState::Phys, FieldState::Phys>()
    {
    }

    void SetTestCase()
    {
        // Initialise math kernel
        std::string execName =
            session->GetCmdLineArgument<std::string>("opExecSpace");
        math = Math(execName);

        // Set initial value
        for (unsigned int blk = 0; blk < fixt_in->GetBlocks().size(); ++blk)
        {
            auto &block = fixt_in->GetBlocks()[blk];
            auto inptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            for (unsigned int nc = 0; nc < fixt_in->GetNumComponents(); ++nc)
            {
                for (size_t el = 0, cnt = 0; el < block.GetNumElements(); ++el)
                {
                    for (unsigned int phys = 0; phys < block.GetNumData();
                         ++phys, ++cnt)
                    {
                        inptr[cnt] = 1.0;
                    }
                }
                inptr += block.size();
            }
        }
    }

    void RunTestCase()
    {
        // Copy fixt_in to fixt_out since operator uses apply with inout type
        fixt_out->Copy<NektarSpaces::HostSpace>(*fixt_in);

        // Initialise Time-stepping operator
        auto op = BDFOp<double>::Create(fixt_explist);
        op->DefineImplicit(&BDFField::DoLHS, this);

        // Initialise timestepping operator
        auto time   = 0.0;
        size_t step = 0;
        op->Initialise(*fixt_out, time, step);

        // Loop all steps
        while (step < m_numsteps)
        {
            // Evolve PDE for one timestep
            op->Apply(*fixt_out);

            // Increment steps
            time += m_timestep;
            ++step;
        }
    }

    void ExpectedSolution()
    {
        // We solve the analytic problem du/dt = u
        // with solution u = e^t and u(t=0) = 1.0

        // Assuming no spatial variation,
        // the analytic solution at final time is: exp(final_time)
        fixt_expected->Initialize<NektarSpaces::HostSpace>(exp(m_final_time));
    }

    bool CheckOrderOfAccuracy(int expectedOrder)
    {
        std::vector<double> timesteps = {0.1, 0.05, 0.01, 0.005, 0.001, 0.0001};
        std::vector<double> errors;

        for (double dt : timesteps)
        {
            m_timestep = dt;
            // Change timestep in session to propagate to TimeOp at runtime
            session->SetParameter("TimeStep", dt);

            m_numsteps   = 1.0 / m_timestep;
            m_final_time = m_timestep * m_numsteps;

            RunTestCase();      // Run simulation
            ExpectedSolution(); // Compute expected solution

            // Compute error at final time (L2 norm; adapt as needed)
            math.sub(*fixt_out, *fixt_expected, *fixt_out);
            double error =
                std::sqrt(math.l2norm(*fixt_out) / math.l2norm(*fixt_in));
            // std::cout << "dt = " << dt << "\tError = " << error << std::endl;
            errors.push_back(error);
        }

        // Compute observed order of accuracy from (dt, error) pairs
        std::vector<double> orders;
        for (size_t i = 0; i < timesteps.size() - 1; ++i)
        {
            double order = log(errors[i] / errors[i + 1]) /
                           log(timesteps[i] / timesteps[i + 1]);
            // std::cout << "dt = " << timesteps[i] << "\tOrder = " << order <<
            // std::endl;
            orders.push_back(order);
        }

        // Compute average observed order
        double observedOrder =
            std::accumulate(orders.begin(), orders.end(), 0.0) / orders.size();

        // Check if observed order is close to expected order (within tolerance)
        if (fabs(observedOrder - expectedOrder) >
            0.1 * expectedOrder) // Choose tolerance
        {
            std::cerr << "Order of accuracy test failed! Observed: "
                      << observedOrder << ", Expected: " << expectedOrder
                      << std::endl;
            return false;
        }
        else
        {
            std::cout << "Order of accuracy verified: " << observedOrder
                      << std::endl;
            return true;
        }
    }

protected:
    int m_numsteps;
    double m_timestep;
    double m_final_time;
    Math math;

    // Do evaluation for test problem:
    // du/dt = u, u(t=0) = 1
    // with solution u = e^t
    void DoLHS(Field<double, FieldState::Phys> &inout, double &gamma)
    {
        // Denominator for analytic test problem (mimics LHS/RHS eval)
        auto denominator = 1.0 / (gamma / m_timestep - 1.0);

        // Multiply extrapolated solution (inout) by denominator
        math.mul(denominator, inout, inout);
    }
};

#define TEST(type, filename)                                                   \
    class type : public BDFField                                               \
    {                                                                          \
    public:                                                                    \
        type()                                                                 \
        {                                                                      \
            meshName = filename;                                               \
        }                                                                      \
    };

TEST(segment_order_1, "run/segment_time_order_1.xml")
TEST(segment_order_2, "run/segment_time_order_2.xml")
TEST(segment_order_3, "run/segment_time_order_3.xml")
TEST(segment_order_4, "run/segment_time_order_4.xml")
