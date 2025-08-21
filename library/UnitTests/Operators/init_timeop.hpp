///////////////////////////////////////////////////////////////////////////////
//
// File: init_timeop.hpp
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
#include "Operators/TimeOps/TimeOp.hpp"

using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

class TimeOpField
    : public InitFields<double, FieldState::Phys, FieldState::Phys>
{
public:
    TimeOpField() : InitFields<double, FieldState::Phys, FieldState::Phys>()
    {
    }

    void SetTestCase(const double alpha, const double beta)
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

        // Parameters for analytic solution
        m_alpha = alpha; // non-stiff factor
        m_beta  = beta;  // stiff factor
    }

    void RunTestCase(const std::string scheme, unsigned int numsteps)
    {
        // Copy fixt_in to fixt_out since operator uses apply with inout type
        fixt_out->Copy<NektarSpaces::HostSpace>(*fixt_in);

        // Initialise Time-stepping operator
        auto op = TimeOp<double>::Create(fixt_explist, scheme);
        op->DefineExplicit(&TimeOpField::DoRHS, this);
        op->DefineImplicit(&TimeOpField::DoLHS, this);

        // Initialise timestepping operator
        // Loop all steps
        while (op->GetNumStep() < numsteps)
        {
            // Evolve PDE for one timestep
            op->Apply(*fixt_out);
        }
    }

    void ExpectedSolution(double final_time)
    {
        // We solve the analytic problem du/dt = \alpha u + \beta u,
        // where \alpha is mild parameter leading to the explicit part
        // and \beta is a stiff parameter leading to the implicit part.
        // The solution is u = e^((\alpha + \beta)*t) and u(t=0) = 1.0
        fixt_expected->Initialize<NektarSpaces::HostSpace>(
            exp((m_alpha + m_beta) * final_time));
    }

    bool CheckOrderOfAccuracy(std::string scheme, int expectedOrder)
    {
        std::vector<double> timesteps = {0.1, 0.05, 0.01, 0.005, 0.001, 0.0001};
        std::vector<double> errors;

        double final_time = 0.5;

        // Compute expected solution
        ExpectedSolution(final_time);

        for (double dt : timesteps)
        {
            // Change timestep in session to propagate to TimeOp at runtime
            session->SetParameter("TimeStep", dt);

            // Run simulation
            RunTestCase(scheme, final_time / dt);

            // Compute error at final time (L2 norm; adapt as needed)
            math.sub(*fixt_out, *fixt_expected, *fixt_out);
            double error =
                std::sqrt(math.l2norm(*fixt_out) / math.l2norm(*fixt_in));
            errors.push_back(error);
        }

        // Compute observed order of accuracy from (dt, error) pairs
        std::vector<double> orders;
        for (size_t i = 0; i < timesteps.size() - 1; ++i)
        {
            double order = log(errors[i] / errors[i + 1]) /
                           log(timesteps[i] / timesteps[i + 1]);
            orders.push_back(order);
        }

        // Use last observed order
        double observedOrder = orders.back();

        // Check if observed order is close to expected order (within tolerance)
        if (std::isnan(observedOrder) || std::isinf(observedOrder) ||
            fabs(observedOrder - expectedOrder) >
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
    double m_alpha;
    double m_beta;

    Math math;

    void DoLHS(Field<double, FieldState::Phys> &in,
               Field<double, FieldState::Phys> &out,
               [[maybe_unused]] const double &time, const double &lambda)
    {
        // Factor for implicit/stiff part of analytic test problem
        auto factor = 1.0 / (1.0 - lambda * m_beta);

        // Multiply extrapolated rhs
        math.mul(factor, in, out);
    }

    void DoRHS(Field<double, FieldState::Phys> &in,
               Field<double, FieldState::Phys> &out,
               [[maybe_unused]] const double &time, const double &factor)
    {
        // Multiply solution by factor
        math.mul(m_alpha * factor, in, out);
    }
};

#define TEST(type, filename)                                                   \
    class type : public TimeOpField                                            \
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
