///////////////////////////////////////////////////////////////////////////////
//
// File: ExplicitSDCOpImpl.hpp
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

#include "SolverCore/TimeOps/SDC/ExplicitSDC/ExplicitSDCOp.hpp"

#include "SolverCore/TimeOps/SDC/ExplicitSDC/ExplicitSDCKernelLaunchers.hpp"

using namespace Nektar;

namespace Nektar::SolverCore::detail
{

template <typename ExecSpace, typename TData>
class ExplicitSDCOpImpl : public ExplicitSDCOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    ExplicitSDCOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                      const std::vector<std::string> &components,
                      const unsigned int &order, const std::string &variant,
                      const std::vector<TData> &freeParams)
        : ExplicitSDCOp<TData>(expansionList, components, order, variant,
                               freeParams)
    {
        this->template Initialize<ExecSpace>();

        auto blockAttr =
            MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                this->m_expansionList);
        for (unsigned int m = 0; m < this->m_nQuadPts; ++m)
        {
            this->m_SFint.push_back(
                LibUtilities::Field<TData, FieldState::Phys>(
                    blockAttr, this->m_components, 1));

            this->m_solutions.push_back(
                LibUtilities::Field<TData, FieldState::Phys>(
                    blockAttr, this->m_components, 1));

            this->m_residuals.push_back(
                LibUtilities::Field<TData, FieldState::Phys>(
                    blockAttr, this->m_components, 1));
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<TimeOp<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components, const unsigned int &order,
        const std::string &variant, const std::vector<TData> &freeParams)
    {
        return std::make_unique<ExplicitSDCOpImpl<ExecSpace, TData>>(
            expansionList, components, order, variant, freeParams);
    }

protected:
    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &inout) override
    {
        // Check that required functions are defined.
        ASSERTL0(
            this->m_explicitRhsFunctor,
            "ExplicitSDC schemes require a DoExplicitRhs method. Define with "
            "ExplicitSDCOp->DefineExplicit().");
        ASSERTL0(
            this->m_projectionFunctor,
            "ExplicitSDC schemes require a DoProjection method. Define with "
            "ExplicitSDCOp->DefineProjection().");

        // Store the initial values.
        this->m_solutions[0].template Copy<MemSpace>(inout);

        for (unsigned int k = 0; k < this->m_order; ++k)
        {
            // Compute initial guess.
            if (k == 0)
            {
                ComputeInitialGuess();
            }
            // Apply SDC correction loop.
            else
            {
                SDCIterationLoop();
            }
        }

        // Update solution.
        this->template UpdateSolution<ExecSpace>(inout);

        // Increment step and time.
        this->m_time += this->m_timestep;
        this->m_step++;
    }

    void ComputeInitialGuess()
    {
        auto tau =
            this->m_tau.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

        // Loop over quadrature.
        for (unsigned int n = 0; n < this->m_nQuadPts; ++n)
        {
            // Compute residual.
            this->DoProjection(this->m_solutions[n], this->m_solutions[n],
                               this->m_time + this->m_timestep * tau[n]);

            this->DoExplicitRhs(this->m_solutions[n], this->m_residuals[n],
                                this->m_time + this->m_timestep * tau[n],
                                this->m_timestep);

            // Use explicit Euler as a first guess.
            if (n < this->m_nQuadPts - 1)
            {
                TData dtn = (tau[n + 1] - tau[n]);
                Math::daxpy<ExecSpace>(dtn, this->m_residuals[n],
                                       this->m_solutions[n],
                                       this->m_solutions[n + 1]);
            }
        }
    }

    void SDCIterationLoop()
    {
        // Update integrated residual.
        this->template InitializeIntegratedResidual<ExecSpace>();

        // Loop over quadrature points.
        auto tau =
            this->m_tau.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        for (unsigned int n = 1; n < this->m_nQuadPts; ++n)
        {
            TData dtn  = this->m_theta * (tau[n] - tau[n - 1]);
            TData dtnp = (n < this->m_nQuadPts - 1)
                             ? this->m_theta * (tau[n + 1] - tau[n])
                             : 0.0;
            // Loop over the blocks.
            for (unsigned int blk = 0;
                 blk < this->m_solutions[n].GetBlocks().size(); ++blk)
            {
                const unsigned int streamID = blk + 1;

                // Determine shape and type of the element.
                auto &block = this->m_solutions[n].GetBlocks()[blk];
                auto nsize  = block.GetNumElementsWithPadding() *
                             block.GetNumData() * block.GetNumComponents() *
                             block.GetNumHomoModes();

                // Compute solution.
                if (n == 1)
                {
                    IterateExplicitSDCSolutionKernel<ExecSpace>(
                        streamID, nsize,
                        this->m_solutions[n - 1]
                            .GetBlocks()[blk]
                            .template GetPtr<MemSpace, ReadOnly>(streamID),
                        this->m_SFint[n]
                            .GetBlocks()[blk]
                            .template GetPtr<MemSpace, ReadOnly>(streamID),
                        this->m_solutions[n]
                            .GetBlocks()[blk]
                            .template GetPtr<MemSpace, WriteOnly>(streamID));
                }
                else
                {
                    IterateExplicitSDCSolutionKernel<ExecSpace>(
                        streamID, nsize, dtn,
                        this->m_solutions[n - 1]
                            .GetBlocks()[blk]
                            .template GetPtr<MemSpace, ReadOnly>(streamID),
                        this->m_residuals[n - 1]
                            .GetBlocks()[blk]
                            .template GetPtr<MemSpace, ReadOnly>(streamID),
                        this->m_SFint[n]
                            .GetBlocks()[blk]
                            .template GetPtr<MemSpace, ReadOnly>(streamID),
                        this->m_solutions[n]
                            .GetBlocks()[blk]
                            .template GetPtr<MemSpace, WriteOnly>(streamID));
                }

                if (n < this->m_nQuadPts - 1)
                {
                    UpdateExplicitIntegratedResidualKernel<ExecSpace>(
                        streamID, nsize, dtnp,
                        this->m_residuals[n]
                            .GetBlocks()[blk]
                            .template GetPtr<MemSpace, ReadOnly>(streamID),
                        this->m_SFint[n + 1]
                            .GetBlocks()[blk]
                            .template GetPtr<MemSpace, ReadWrite>(streamID));
                }
            }

            // Compute residual.
            this->DoProjection(this->m_solutions[n], this->m_solutions[n],
                               this->m_time + this->m_timestep * tau[n]);
            this->DoExplicitRhs(this->m_solutions[n], this->m_residuals[n],
                                this->m_time + this->m_timestep * tau[n],
                                this->m_timestep);
        }
    }
};

} // namespace Nektar::SolverCore::detail
