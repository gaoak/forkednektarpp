///////////////////////////////////////////////////////////////////////////////
//
// File: SDCOp.hpp
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

#include <LibUtilities/Foundations/ManagerAccess.h>
#include <LibUtilities/Foundations/Points.h>
#include <LibUtilities/Polylib/Polylib.h>
#include <Operators/TimeOps/TimeOp.hpp>

#include <Operators/TimeOps/SDC/SDCKernelLaunchers.hpp>

namespace Nektar::Operators
{

// SDC base class
// Defines the apply operator to enforce apply parameter types
template <typename TData> class SDCOp : public TimeOp<TData>
{
public:
    static std::shared_ptr<SDCOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &method = "", const unsigned int &order = 0,
        const std::string &variant           = "",
        const std::vector<TData> &freeParams = std::vector<TData>{},
        const std::string &execStr           = "")
    {
        return std::dynamic_pointer_cast<SDCOp<TData>>(
            TimeOp<TData>::Create(expansionList, components, method, order,
                                  variant, freeParams, execStr));
    }

protected:
    unsigned int m_order;
    std::string m_variant;
    std::vector<TData> m_freeParams;
    bool m_first_quadrature{true};
    bool m_last_quadrature{true};
    unsigned int m_ordermin{0}; /// Minimum order of the integration scheme
    unsigned int m_ordermax{0}; /// Maximum order of the integration scheme
    unsigned int m_nQuadPts{0}; /// Number of quadrature points
    TData m_theta{1.0};         /// SDC parameter
    LibUtilities::PointsKey m_pointsKey; /// Object containing quadrature data
    MemoryRegion<TData const *> m_mr0;
    MemoryRegion<TData *> m_mr1;
    std::deque<Field<TData, FieldState::Phys>> m_residuals;
    std::deque<Field<TData, FieldState::Phys>>
        m_SFint;               /// Array containing the integrated residual term
    MemoryRegion<TData> m_tau; /// Array containing the quadrature points
    MemoryRegion<TData> m_QMat; /// Array containing the integration matrix
    MemoryRegion<TData>
        m_interp; /// Array containing the interpolation coefficients

    SDCOp(const MultiRegions::ExpListSharedPtr &expansionList,
          const std::vector<std::string> components, const unsigned int &order,
          const std::string &variant, const std::vector<TData> &freeParams)
        : TimeOp<TData>(expansionList, components), m_order(order),
          m_variant(variant)
    {
        if (freeParams.size() == 0)
        {
            for (unsigned int i = 0; i < expansionList->GetSession()
                                             ->GetTimeIntScheme()
                                             .freeParams.size();
                 i++)
            {
                this->m_freeParams.push_back(expansionList->GetSession()
                                                 ->GetTimeIntScheme()
                                                 .freeParams[i]);
            }
        }
        else
        {
            this->m_freeParams = freeParams;
        }
    }

    ~SDCOp() override = default;

    template <typename ExecSpace> void Initialize(void)
    {
        using MemSpace = typename ExecSpace::memory_space;

        // Set operator meta data
        ASSERTL0(this->m_freeParams.size() == 2,
                 "SDC Time integration scheme invalid number "
                 "of free parameters, expected two "
                 "<theta, number of quadrature>, received " +
                     std::to_string(this->m_freeParams.size()));

        ASSERTL0(0.0 <= this->m_freeParams[0] && this->m_freeParams[0] <= 1.0,
                 "Spectral Deferred Correction Time integration "
                 "scheme bad parameter numbers (0.0 - 1.0): " +
                     std::to_string(this->m_freeParams[0]));

        this->m_theta    = this->m_freeParams[0];
        this->m_nQuadPts = this->m_freeParams[1];

        if (this->m_variant == "Equidistant")
        {
            ASSERTL0(1 <= this->m_nQuadPts,
                     this->m_variant +
                         " quadrature require quadrature "
                         "numbers (>=1" +
                         "): " + std::to_string(this->m_nQuadPts));

            this->m_first_quadrature = (this->m_nQuadPts == 1) ? false : true;
            this->m_last_quadrature  = (this->m_nQuadPts == 1) ? false : true;
            this->m_ordermin         = 1;
            this->m_ordermax         = this->m_nQuadPts;
            this->m_pointsKey        = LibUtilities::PointsKey(
                this->m_nQuadPts, LibUtilities::ePolyEvenlySpaced);
        }
        else if (this->m_variant == "GaussLobattoLegendre")
        {
            ASSERTL0(2 <= this->m_nQuadPts,
                     this->m_variant +
                         " quadrature require quadrature "
                         "numbers (>=2" +
                         "): " + std::to_string(this->m_nQuadPts));

            this->m_first_quadrature = true;
            this->m_last_quadrature  = true;
            this->m_ordermin         = 1;
            this->m_ordermax         = 2 * this->m_nQuadPts - 2;
            this->m_pointsKey        = LibUtilities::PointsKey(
                this->m_nQuadPts, LibUtilities::eGaussLobattoLegendre);
        }
        else if (this->m_variant == "GaussRadauLegendre")
        {
            ASSERTL0(2 <= this->m_nQuadPts,
                     this->m_variant +
                         " quadrature require quadrature "
                         "numbers (>=2" +
                         "): " + std::to_string(this->m_nQuadPts));

            this->m_first_quadrature = false;
            this->m_last_quadrature  = true;
            this->m_ordermin         = 1;
            this->m_ordermax         = 2 * this->m_nQuadPts - 1;
            this->m_pointsKey        = LibUtilities::PointsKey(
                this->m_nQuadPts, LibUtilities::eGaussRadauPLegendre);
        }
        else if (this->m_variant == "GaussGaussLegendre")
        {
            ASSERTL0(1 <= this->m_nQuadPts,
                     this->m_variant +
                         " quadrature require quadrature "
                         "numbers (>=1" +
                         "): " + std::to_string(this->m_nQuadPts));

            this->m_first_quadrature = false;
            this->m_last_quadrature  = false;
            this->m_ordermin         = 1;
            this->m_ordermax         = 2 * this->m_nQuadPts;
            this->m_pointsKey        = LibUtilities::PointsKey(
                this->m_nQuadPts, LibUtilities::eGaussGaussLegendre);
        }
        else
        {
            ASSERTL0(false, "unknow variant (quadrature) type");
        }

        ASSERTL0(this->m_ordermin <= this->m_order,
                 "Spectral Deferred Correction Time integration "
                 "scheme bad order numbers (>=" +
                     std::to_string(this->m_ordermin) +
                     "): " + std::to_string(this->m_order));

        ASSERTL0(this->m_ordermax >= this->m_order,
                 "Spectral Deferred Correction Time integration "
                 "scheme bad order numbers (<=" +
                     std::to_string(this->m_ordermax) +
                     "): " + std::to_string(this->m_order));

        // Add one extra quadrature points for i.c., if necessary
        if (!this->m_first_quadrature)
        {
            this->m_nQuadPts += 1;
        }
        // Get quadrature points and rescale to [0, 1]
        unsigned int offset = this->m_first_quadrature ? 0 : 1;
        std::vector<double> tau(this->m_nQuadPts, 0.0);
        for (unsigned int i = offset; i < this->m_nQuadPts; i++)
        {
            TData tmp = LibUtilities::PointsManager()[this->m_pointsKey]
                            ->GetZ()[i - offset];
            tau[i] = (tmp + 1.0) / 2.0;
        }
        this->m_tau = MemoryRegion<TData>::template FromVector<MemSpace>(tau);

        // Compute integration matrix.
        unsigned int colOffset = this->m_first_quadrature ? 0 : 1;
        unsigned int rowOffset =
            this->m_first_quadrature ? 0 : this->m_nQuadPts - 1;
        unsigned int nCols       = this->m_nQuadPts - colOffset;
        unsigned int nRows       = this->m_nQuadPts;
        std::vector<double> QMat = std::vector<double>(nRows * nCols, 0.0);
        Polylib::Qg(&QMat[rowOffset], &tau[colOffset], nCols);
        this->m_QMat = MemoryRegion<TData>::template FromVector<MemSpace>(QMat);

        // Compute intepolation coefficient.
        std::vector<TData> interp(this->m_nQuadPts);
        for (unsigned int i = 0; i < this->m_nQuadPts; i++)
        {
            interp[i] =
                Polylib::hgj(i, 1.0, &tau[0], this->m_nQuadPts, 0.0, 0.0);
        }
        this->m_interp =
            MemoryRegion<TData>::template FromVector<MemSpace>(interp);

        // Buffer for memory transfer
        this->m_mr0 = MemoryRegion<const TData *>(this->m_nQuadPts);
        this->m_mr1 = MemoryRegion<TData *>(this->m_nQuadPts);
    }

    template <typename ExecSpace>
    void UpdateSolution(Field<TData, FieldState::Phys> &inout)
    {
        using MemSpace = typename ExecSpace::memory_space;

        // Copy solution for last quadrature points.
        if (this->m_last_quadrature)
        {
            inout.template Copy<MemSpace>(this->m_solutions.back());
        }
        // Interpolate solutions.
        else
        {
            auto interp = this->m_interp.template GetPtr<MemSpace, ReadOnly>();

            // Loop over the blocks.
            for (unsigned int blk = 0; blk < inout.GetBlocks().size(); ++blk)
            {
                // Determine shape and type of the element.
                auto &inoutBlock = inout.GetBlocks()[blk];
                auto nsize       = inoutBlock.GetNumElementsWithPadding() *
                             inoutBlock.GetNumData() *
                             inoutBlock.GetNumComponents() *
                             inoutBlock.GetNumHomoModes();

                // Initialize pointers.
                auto hostPtr =
                    this->m_mr0
                        .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
                for (unsigned int n = 0; n < this->m_nQuadPts; ++n)
                {
                    hostPtr[n] = this->m_solutions[n]
                                     .GetBlocks()[blk]
                                     .template GetPtr<MemSpace, ReadOnly>();
                }

                // Update solution.
                detail::UpdateSolutionKernel<ExecSpace>(
                    nsize, this->m_nQuadPts, interp,
                    this->m_mr0.template GetPtr<MemSpace, ReadOnly>(),
                    inout.GetBlocks()[blk]
                        .template GetPtr<MemSpace, WriteOnly>());
            }
        }
    }

    template <typename ExecSpace> void InitializeIntegratedResidual()
    {
        using MemSpace = typename ExecSpace::memory_space;

        auto QMat = this->m_QMat.template GetPtr<MemSpace, ReadOnly>();

        // Loop over the blocks.
        for (unsigned int blk = 0; blk < this->m_SFint[0].GetBlocks().size();
             ++blk)
        {
            // Determine shape and type of the element.
            auto &block = this->m_SFint[0].GetBlocks()[blk];
            auto nsize  = block.GetNumElementsWithPadding() *
                         block.GetNumData() * block.GetNumComponents() *
                         block.GetNumHomoModes();

            // Initialize pointers.
            auto hostPtr0 =
                this->m_mr0
                    .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            for (unsigned int n = 0; n < this->m_nQuadPts; ++n)
            {
                hostPtr0[n] = this->m_residuals[n]
                                  .GetBlocks()[blk]
                                  .template GetPtr<MemSpace, ReadOnly>();
            }

            auto hostPtr1 =
                this->m_mr1
                    .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            for (unsigned int n = 0; n < this->m_nQuadPts; ++n)
            {
                hostPtr1[n] = this->m_SFint[n]
                                  .GetBlocks()[blk]
                                  .template GetPtr<MemSpace, WriteOnly>();
            }

            // Update solution.
            if (this->m_first_quadrature)
            {
                detail::InitializeIntegratedResidualKernel<ExecSpace, true>(
                    nsize, this->m_nQuadPts, QMat,
                    this->m_mr0.template GetPtr<MemSpace, ReadOnly>(),
                    this->m_mr1.template GetPtr<MemSpace, ReadOnly>());
            }
            else
            {
                detail::InitializeIntegratedResidualKernel<ExecSpace, false>(
                    nsize, this->m_nQuadPts, QMat,
                    this->m_mr0.template GetPtr<MemSpace, ReadOnly>(),
                    this->m_mr1.template GetPtr<MemSpace, ReadOnly>());
            }
        }
    }
};

} // namespace Nektar::Operators
