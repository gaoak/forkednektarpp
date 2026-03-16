///////////////////////////////////////////////////////////////////////////////
//
// File: BICGSTABOpImpl.hpp
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

#include "Operators/GlobalLinSysOps/LinearSolvers/BICGSTAB/BICGSTABOp.hpp"
#include "Operators/GlobalLinSysOps/LinearSolvers/BICGSTABR/BICGSTABROpImpl.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

using namespace Nektar;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class BICGSTABOpImpl : public BICGSTABOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    BICGSTABOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                   const std::vector<std::string> &components)
        : BICGSTABOp<TData>(expansionList, components),
          m_p(Field<TData, FieldState::Coeff>(
              "BICGSTABOp p",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_v(Field<TData, FieldState::Coeff>(
              "BICGSTABOp v",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_w(Field<TData, FieldState::Coeff>(
              "BICGSTABOp w",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_z(Field<TData, FieldState::Coeff>(
              "BICGSTABOp z",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_r(Field<TData, FieldState::Coeff>(
              "BICGSTABOp r",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1)),
          m_rtilde(Field<TData, FieldState::Coeff>(
              "BICGSTABOp rtilde",
              GetBlockAttributes<TData, FieldState::Coeff>(expansionList),
              components, 1))
    {
        this->template SetLinearSolver<ExecSpace>();

        auto session = this->m_expansionList->GetSession();

        // Set parameters.
        this->m_leftPreconditioner =
            session->DefinesParameter("LinSysLeftPrecon")
                ? session->GetParameter("LinSysLeftPrecon")
                : false;
        this->m_rightPreconditioner =
            session->DefinesParameter("LinSysRightPrecon")
                ? session->GetParameter("LinSysRightPrecon")
                : true;
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<BICGSTABOpImpl<ExecSpace, TData>>(expansionList,
                                                                  components);
    }

protected:
    struct FieldStats
    {
        size_t validCount         = 0;
        size_t paddingCount       = 0;
        size_t validNonFinite     = 0;
        size_t paddingNonFinite   = 0;
        TData validMaxAbs         = 0.0;
        TData paddingMaxAbs       = 0.0;
        std::string firstValidBad = "";
        std::string firstPadBad   = "";
    };

    struct DotComparison
    {
        std::string label = "";
        TData kernel      = std::numeric_limits<TData>::quiet_NaN();
        TData reference   = std::numeric_limits<TData>::quiet_NaN();
        TData absError    = std::numeric_limits<TData>::quiet_NaN();
        TData relError    = std::numeric_limits<TData>::quiet_NaN();
        unsigned int iter = 0;
        std::string stage = "";
        bool valid        = false;
        bool mismatch     = false;
    };

    Field<TData, FieldState::Coeff> m_p;
    Field<TData, FieldState::Coeff> m_v;
    Field<TData, FieldState::Coeff> m_w;
    Field<TData, FieldState::Coeff> m_z;
    Field<TData, FieldState::Coeff> m_r;
    Field<TData, FieldState::Coeff> m_rtilde;

    bool HasPadding(Field<TData, FieldState::Coeff> &field)
    {
        for (auto &block : field.GetBlocks())
        {
            if (block.CompSize() != block.GetNumElements() * block.GetNumData())
            {
                return true;
            }
        }

        return false;
    }

    TData GetReferenceDdot(Field<TData, FieldState::Coeff> &x,
                           Field<TData, FieldState::Coeff> &y)
    {
        long double sum = 0.0;

        ASSERTL1(x.GetBlocks().size() == y.GetBlocks().size(),
                 "Reference ddot requires matching block layouts.");

        for (unsigned int blk = 0; blk < x.GetBlocks().size(); ++blk)
        {
            auto &xBlock = x.GetBlocks()[blk];
            auto &yBlock = y.GetBlocks()[blk];

            ASSERTL1(xBlock.CompSize() == yBlock.CompSize() &&
                         xBlock.GetNumElements() == yBlock.GetNumElements() &&
                         xBlock.GetNumData() == yBlock.GetNumData(),
                     "Reference ddot requires matching block sizes.");

            auto xPtr =
                xBlock.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto yPtr =
                yBlock.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            const size_t validSize =
                xBlock.GetNumElements() * xBlock.GetNumData();
            const size_t stride = xBlock.CompSize();
            const unsigned int ncomp =
                xBlock.GetNumComponents() * xBlock.GetNumHomoModes();

            for (unsigned int n = 0; n < ncomp; ++n)
            {
                auto xComp = xPtr + n * stride;
                auto yComp = yPtr + n * stride;

                for (size_t i = 0; i < validSize; ++i)
                {
                    sum += static_cast<long double>(xComp[i]) *
                           static_cast<long double>(yComp[i]);
                }
            }
        }

        TData ref = static_cast<TData>(sum);
        this->m_rowComm->AllReduce(ref, Nektar::LibUtilities::ReduceSum);
        return ref;
    }

    DotComparison CompareDot(const std::string &label, const unsigned int iter,
                             const std::string &stage, const TData kernelValue,
                             Field<TData, FieldState::Coeff> &x,
                             Field<TData, FieldState::Coeff> &y)
    {
        DotComparison cmp;
        cmp.label     = label;
        cmp.kernel    = kernelValue;
        cmp.reference = GetReferenceDdot(x, y);
        cmp.iter      = iter;
        cmp.stage     = stage;
        cmp.valid = std::isfinite(cmp.kernel) && std::isfinite(cmp.reference);

        if (!cmp.valid)
        {
            cmp.absError = std::numeric_limits<TData>::quiet_NaN();
            cmp.relError = std::numeric_limits<TData>::quiet_NaN();
            cmp.mismatch = true;
            return cmp;
        }

        cmp.absError      = std::abs(cmp.kernel - cmp.reference);
        const TData scale = std::max<TData>(
            {1.0, std::abs(cmp.kernel), std::abs(cmp.reference)});
        cmp.relError = cmp.absError / scale;

        const TData absTol = 1.0e3 * std::numeric_limits<TData>::epsilon();
        const TData relTol = 1.0e6 * std::numeric_limits<TData>::epsilon();
        cmp.mismatch       = cmp.absError > std::max(absTol, relTol * scale);

        return cmp;
    }

    void PrintDotComparison(const DotComparison &cmp)
    {
        if (!cmp.valid)
        {
            std::cout << "  " << cmp.label << ": kernel=" << cmp.kernel
                      << ", reference=" << cmp.reference
                      << ", iter=" << cmp.iter << ", stage=" << cmp.stage
                      << std::endl;
            return;
        }

        std::cout << "  " << cmp.label << ": kernel=" << cmp.kernel
                  << ", reference=" << cmp.reference
                  << ", abs_error=" << cmp.absError
                  << ", rel_error=" << cmp.relError << ", iter=" << cmp.iter
                  << ", stage=" << cmp.stage << std::endl;
    }

    void ReportDotMismatch(const DotComparison &cmp)
    {
        if (!this->m_root)
        {
            return;
        }

        std::cout << "BICGSTAB ddot mismatch detected" << std::endl;
        PrintDotComparison(cmp);
    }

    FieldStats GetFieldStats(Field<TData, FieldState::Coeff> &field)
    {
        FieldStats stats;

        for (unsigned int blk = 0; blk < field.GetBlocks().size(); ++blk)
        {
            auto &block = field.GetBlocks()[blk];
            auto ptr =
                block.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            const size_t validSize =
                block.GetNumElements() * block.GetNumData();
            const size_t stride = block.CompSize();
            const unsigned int ncomp =
                block.GetNumComponents() * block.GetNumHomoModes();

            stats.validCount += validSize * ncomp;
            stats.paddingCount += (stride - validSize) * ncomp;

            for (unsigned int n = 0; n < ncomp; ++n)
            {
                auto compPtr = ptr + n * stride;

                for (size_t i = 0; i < validSize; ++i)
                {
                    TData value = compPtr[i];
                    if (!std::isfinite(value))
                    {
                        ++stats.validNonFinite;
                        if (stats.firstValidBad.empty())
                        {
                            std::stringstream ss;
                            ss << "blk=" << blk << " comp=" << n << " idx=" << i
                               << " value=" << value;
                            stats.firstValidBad = ss.str();
                        }
                        continue;
                    }

                    stats.validMaxAbs =
                        std::max(stats.validMaxAbs, std::abs(value));
                }

                for (size_t i = validSize; i < stride; ++i)
                {
                    TData value = compPtr[i];
                    if (!std::isfinite(value))
                    {
                        ++stats.paddingNonFinite;
                        if (stats.firstPadBad.empty())
                        {
                            std::stringstream ss;
                            ss << "blk=" << blk << " comp=" << n
                               << " pad_idx=" << (i - validSize)
                               << " value=" << value;
                            stats.firstPadBad = ss.str();
                        }
                        continue;
                    }

                    stats.paddingMaxAbs =
                        std::max(stats.paddingMaxAbs, std::abs(value));
                }
            }
        }

        return stats;
    }

    void PrintFieldStats(const std::string &label,
                         Field<TData, FieldState::Coeff> &field)
    {
        auto stats = GetFieldStats(field);

        std::cout << "  " << label
                  << ": valid_nonfinite=" << stats.validNonFinite << "/"
                  << stats.validCount
                  << ", padding_nonfinite=" << stats.paddingNonFinite << "/"
                  << stats.paddingCount
                  << ", valid_max_abs=" << stats.validMaxAbs
                  << ", padding_max_abs=" << stats.paddingMaxAbs;

        if (!stats.firstValidBad.empty())
        {
            std::cout << ", first_valid_bad={" << stats.firstValidBad << "}";
        }

        if (!stats.firstPadBad.empty())
        {
            std::cout << ", first_padding_bad={" << stats.firstPadBad << "}";
        }

        std::cout << std::endl;
    }

    void PrintBreakdownDiagnostics(
        const std::string &reason, const unsigned int iter,
        const std::string &stage, const TData rhsMagnitude, const TData eps,
        const TData rho, const TData rhoNew, const TData alphaDen,
        const TData omegaNum, const TData omegaDen, const TData alpha,
        const TData omega, const TData beta, const DotComparison &rhsCmp,
        const DotComparison &epsCmp, const DotComparison &rhoCmp,
        const DotComparison &alphaDenCmp, const DotComparison &omegaNumCmp,
        const DotComparison &omegaDenCmp, Field<TData, FieldState::Coeff> &in,
        Field<TData, FieldState::Coeff> &out)
    {
        if (!this->m_root)
        {
            return;
        }

        std::cout << "BICGSTAB diagnostic: reason=" << reason
                  << ", iter=" << iter << ", stage=" << stage
                  << ", left_precon=" << this->m_leftPreconditioner
                  << ", right_precon=" << this->m_rightPreconditioner
                  << ", tol=" << this->m_tol << ", rhs_mag=" << rhsMagnitude
                  << ", eps=" << eps << ", rho=" << rho
                  << ", rho_new=" << rhoNew << ", alpha_den=" << alphaDen
                  << ", omega_num=" << omegaNum << ", omega_den=" << omegaDen
                  << ", alpha=" << alpha << ", omega=" << omega
                  << ", beta=" << beta << std::endl;

        if (rhsCmp.valid || rhsCmp.mismatch)
        {
            PrintDotComparison(rhsCmp);
        }
        if (epsCmp.valid || epsCmp.mismatch)
        {
            PrintDotComparison(epsCmp);
        }
        if (rhoCmp.valid || rhoCmp.mismatch)
        {
            PrintDotComparison(rhoCmp);
        }
        if (alphaDenCmp.valid || alphaDenCmp.mismatch)
        {
            PrintDotComparison(alphaDenCmp);
        }
        if (omegaNumCmp.valid || omegaNumCmp.mismatch)
        {
            PrintDotComparison(omegaNumCmp);
        }
        if (omegaDenCmp.valid || omegaDenCmp.mismatch)
        {
            PrintDotComparison(omegaDenCmp);
        }

        PrintFieldStats("in", in);
        PrintFieldStats("out", out);
        PrintFieldStats("p", m_p);
        PrintFieldStats("v", m_v);
        PrintFieldStats("w", m_w);
        PrintFieldStats("z", m_z);
        PrintFieldStats("r", m_r);
        PrintFieldStats("rtilde", m_rtilde);
    }

    void FallbackToBICGSTABR(
        const std::string &reason, const std::string &stage,
        const TData rhsMagnitude, const TData eps, const TData rho,
        const TData rhoNew, const TData alphaDen, const TData omegaNum,
        const TData omegaDen, const TData alpha, const TData omega,
        const TData beta, const DotComparison &rhsCmp,
        const DotComparison &epsCmp, const DotComparison &rhoCmp,
        const DotComparison &alphaDenCmp, const DotComparison &omegaNumCmp,
        const DotComparison &omegaDenCmp, Field<TData, FieldState::Coeff> &in,
        Field<TData, FieldState::Coeff> &out)
    {
        auto session = this->m_expansionList->GetSession();
        PrintBreakdownDiagnostics(
            reason, this->m_niter, stage, rhsMagnitude, eps, rho, rhoNew,
            alphaDen, omegaNum, omegaDen, alpha, omega, beta, rhsCmp, epsCmp,
            rhoCmp, alphaDenCmp, omegaNumCmp, omegaDenCmp, in, out);

        if (this->m_root && session->DefinesCmdLineArgument("verbose"))
        {
            std::cout << "BICGSTAB breakdown detected (" << reason
                      << "). Falling back to BICGSTABR." << std::endl;
        }

        BICGSTABROpImpl<ExecSpace, TData> fallback(this->m_expansionList,
                                                   this->m_components);
        fallback.SetLHS(this->m_lhs);
        fallback.SetPrecon(this->m_precon);
        fallback.UpdatePrecon();
        fallback.Apply(in, out);
        this->m_niter += fallback.GetNiterations();
    }

    void v_Apply(Field<TData, FieldState::Coeff> &in,
                 Field<TData, FieldState::Coeff> &out) override
    {
        // Convergence parameters.
        this->m_niter = 0;
        TData rhsMagnitude, eps;
        TData alpha = std::numeric_limits<TData>::quiet_NaN();
        TData beta  = 0.0;
        TData rho   = std::numeric_limits<TData>::quiet_NaN();
        TData rho_new;
        TData omega    = 0.0;
        TData omega0   = std::numeric_limits<TData>::quiet_NaN();
        TData omega1   = std::numeric_limits<TData>::quiet_NaN();
        TData alphaDen = std::numeric_limits<TData>::quiet_NaN();
        std::string stage("initial");
        const bool instrumentReference =
            in.GetNumComponents() > 1 || HasPadding(in);
        bool dotMismatchReported = false;
        DotComparison rhsCmp;
        DotComparison epsCmp;
        DotComparison rhoCmp;
        DotComparison alphaDenCmp;
        DotComparison omegaNumCmp;
        DotComparison omegaDenCmp;

        // Reset the fields to zero.
        out.template Initialize<MemSpace>(0);

        // Calculate inital rhs magnitude.
        m_r.template Copy<MemSpace>(in);
        this->m_assmbScatrOp->Apply(m_r);
        rhsMagnitude = this->m_math.ddot(in, m_r);
        this->m_rowComm->AllReduce(rhsMagnitude,
                                   Nektar::LibUtilities::ReduceSum);
        if (instrumentReference)
        {
            rhsCmp = CompareDot("rhs_dot", this->m_niter, "initial rhs",
                                rhsMagnitude, in, m_r);
            if (rhsCmp.mismatch && !dotMismatchReported)
            {
                ReportDotMismatch(rhsCmp);
                dotMismatchReported = true;
            }
        }
        rhsMagnitude = (rhsMagnitude > 1.0e-6) ? rhsMagnitude : 1.0;

        // Iteration 0
        // Copy RHS into initial residual and assemble with Zero Dirichlet BCs.
        m_r.template Copy<MemSpace>(in);
        this->m_assmbScatrZeroDirOp->Apply(m_r);

        if (this->m_leftPreconditioner)
        {
            this->m_precon->Apply(m_r, m_r);
        }

        eps = this->m_math.ddot(in, m_r);
        this->m_rowComm->AllReduce(eps, Nektar::LibUtilities::ReduceSum);
        if (instrumentReference)
        {
            epsCmp = CompareDot("initial_eps_dot", this->m_niter,
                                "initial residual", eps, in, m_r);
            if (epsCmp.mismatch && !dotMismatchReported)
            {
                ReportDotMismatch(epsCmp);
                dotMismatchReported = true;
            }
        }

        // If the input residual is less than tolerance then skip solve.
        if (eps < this->m_tol * this->m_tol * rhsMagnitude)
        {
            return;
        }

        // Iteration >= 1
        m_p.template Copy<MemSpace>(m_r);
        m_rtilde.template Copy<MemSpace>(m_r);
        rho_new = this->m_math.ddot(m_rtilde, m_r);
        this->m_rowComm->AllReduce(rho_new, Nektar::LibUtilities::ReduceSum);
        if (instrumentReference)
        {
            rhoCmp = CompareDot("rho_dot", this->m_niter, "initial rho",
                                rho_new, m_rtilde, m_r);
            if (rhoCmp.mismatch && !dotMismatchReported)
            {
                ReportDotMismatch(rhoCmp);
                dotMismatchReported = true;
            }
        }
        while (true)
        {
            if (this->m_niter > this->m_maxIter)
            {
                FallbackToBICGSTABR("Exceeded max iterations", stage,
                                    rhsMagnitude, eps, rho, rho_new, alphaDen,
                                    omega0, omega1, alpha, omega, beta, rhsCmp,
                                    epsCmp, rhoCmp, alphaDenCmp, omegaNumCmp,
                                    omegaDenCmp, in, out);
                return;
            }

            // Update search vectors.
            if (this->m_niter > 0)
            {
                daxpy<ExecSpace>(-omega, m_v, m_p, m_p);
                daxpy<ExecSpace>(beta, m_p, m_r, m_p);
            }

            // Perform the method-specific matrix-vector multiply operation.
            auto &tmp = (this->m_rightPreconditioner) ? m_w : m_p;
            if (this->m_rightPreconditioner)
            {
                stage = "apply right preconditioner to p";
                this->m_precon->Apply(m_p, tmp);
            }
            stage = "apply lhs to search direction";
            this->m_lhs->Apply(tmp, m_v);
            this->m_robBndCondOp->Apply(tmp, m_v);
            this->m_assmbScatrZeroDirOp->Apply(m_v);
            if (this->m_leftPreconditioner)
            {
                stage = "apply left preconditioner to v";
                this->m_precon->Apply(m_v, m_v);
            }

            // Update coefficients.
            stage    = "compute alpha denominator";
            alphaDen = this->m_math.ddot(m_v, m_rtilde);
            this->m_rowComm->AllReduce(alphaDen, LibUtilities::ReduceSum);
            if (instrumentReference)
            {
                alphaDenCmp = CompareDot("alpha_den_dot", this->m_niter, stage,
                                         alphaDen, m_v, m_rtilde);
                if (alphaDenCmp.mismatch && !dotMismatchReported)
                {
                    ReportDotMismatch(alphaDenCmp);
                    dotMismatchReported = true;
                }
            }
            if (!std::isfinite(alphaDen) || alphaDen == 0.0)
            {
                FallbackToBICGSTABR("<v, rtilde> breakdown", stage,
                                    rhsMagnitude, eps, rho, rho_new, alphaDen,
                                    omega0, omega1, alpha, omega, beta, rhsCmp,
                                    epsCmp, rhoCmp, alphaDenCmp, omegaNumCmp,
                                    omegaDenCmp, in, out);
                return;
            }
            alpha = rho_new / alphaDen;
            if (!std::isfinite(alpha))
            {
                FallbackToBICGSTABR("alpha is not finite", stage, rhsMagnitude,
                                    eps, rho, rho_new, alphaDen, omega0, omega1,
                                    alpha, omega, beta, rhsCmp, epsCmp, rhoCmp,
                                    alphaDenCmp, omegaNumCmp, omegaDenCmp, in,
                                    out);
                return;
            }

            // Update solution.
            daxpy<ExecSpace>(alpha, tmp, out, out);
            daxpy<ExecSpace>(-alpha, m_v, m_r, m_r);

            // Test if norm is within tolerance.
            eps = this->m_math.ddot(m_r, m_r);
            this->m_rowComm->AllReduce(eps, LibUtilities::ReduceSum);
            if (eps < this->m_tol * this->m_tol * rhsMagnitude)
            {
                if (this->m_root)
                {
                    std::cout << this->name
                              << " iterations made = " << this->m_niter
                              << " using tolerance of " << this->m_tol
                              << " error = " << std::sqrt(eps / rhsMagnitude)
                              << " rhs_mag = " << std::sqrt(rhsMagnitude)
                              << std::endl;
                }
                break;
            }

            // Perform the method-specific matrix-vector multiply operation.
            auto &tmp2 = (this->m_rightPreconditioner) ? m_w : m_r;
            if (this->m_rightPreconditioner)
            {
                stage = "apply right preconditioner to r";
                this->m_precon->Apply(m_r, tmp2);
            }
            stage = "apply lhs to residual direction";
            this->m_lhs->Apply(tmp2, m_z);
            this->m_robBndCondOp->Apply(tmp2, m_z);
            this->m_assmbScatrZeroDirOp->Apply(m_z);
            if (this->m_leftPreconditioner)
            {
                stage = "apply left preconditioner to z";
                this->m_precon->Apply(m_z, m_z);
            }

            // Update coefficients.
            stage  = "compute omega denominator";
            omega0 = this->m_math.ddot(m_r, m_z);
            omega1 = this->m_math.ddot(m_z, m_z);
            this->m_rowComm->AllReduce(omega0, LibUtilities::ReduceSum);
            this->m_rowComm->AllReduce(omega1, LibUtilities::ReduceSum);
            if (instrumentReference)
            {
                omegaNumCmp = CompareDot("omega_num_dot", this->m_niter, stage,
                                         omega0, m_r, m_z);
                if (omegaNumCmp.mismatch && !dotMismatchReported)
                {
                    ReportDotMismatch(omegaNumCmp);
                    dotMismatchReported = true;
                }

                omegaDenCmp = CompareDot("omega_den_dot", this->m_niter, stage,
                                         omega1, m_z, m_z);
                if (omegaDenCmp.mismatch && !dotMismatchReported)
                {
                    ReportDotMismatch(omegaDenCmp);
                    dotMismatchReported = true;
                }
            }
            if (!std::isfinite(omega0) || !std::isfinite(omega1) ||
                omega1 == 0.0)
            {
                FallbackToBICGSTABR("<z, z> breakdown", stage, rhsMagnitude,
                                    eps, rho, rho_new, alphaDen, omega0, omega1,
                                    alpha, omega, beta, rhsCmp, epsCmp, rhoCmp,
                                    alphaDenCmp, omegaNumCmp, omegaDenCmp, in,
                                    out);
                return;
            }
            omega = omega0 / omega1;
            if (!std::isfinite(omega))
            {
                FallbackToBICGSTABR("omega is not finite", stage, rhsMagnitude,
                                    eps, rho, rho_new, alphaDen, omega0, omega1,
                                    alpha, omega, beta, rhsCmp, epsCmp, rhoCmp,
                                    alphaDenCmp, omegaNumCmp, omegaDenCmp, in,
                                    out);
                return;
            }

            // Update solution.
            daxpy<ExecSpace>(omega, tmp2, out, out);
            daxpy<ExecSpace>(-omega, m_z, m_r, m_r);

            ++this->m_niter;

            // Test if norm is within tolerance.
            eps = this->m_math.ddot(m_r, m_r);
            this->m_rowComm->AllReduce(eps, Nektar::LibUtilities::ReduceSum);
            if (eps < this->m_tol * this->m_tol * rhsMagnitude)
            {
                if (this->m_root)
                {
                    std::cout << this->name
                              << " iterations made = " << this->m_niter
                              << " using tolerance of " << this->m_tol
                              << " error = " << std::sqrt(eps / rhsMagnitude)
                              << " rhs_mag = " << std::sqrt(rhsMagnitude)
                              << std::endl;
                }
                break;
            }

            // Update coefficients.
            rho     = rho_new;
            stage   = "compute rho update";
            rho_new = this->m_math.ddot(m_rtilde, m_r);
            this->m_rowComm->AllReduce(rho_new,
                                       Nektar::LibUtilities::ReduceSum);
            if (instrumentReference)
            {
                rhoCmp = CompareDot("rho_dot", this->m_niter, stage, rho_new,
                                    m_rtilde, m_r);
                if (rhoCmp.mismatch && !dotMismatchReported)
                {
                    ReportDotMismatch(rhoCmp);
                    dotMismatchReported = true;
                }
            }
            if (!std::isfinite(rho_new) || !std::isfinite(rho))
            {
                FallbackToBICGSTABR("rho breakdown", stage, rhsMagnitude, eps,
                                    rho, rho_new, alphaDen, omega0, omega1,
                                    alpha, omega, beta, rhsCmp, epsCmp, rhoCmp,
                                    alphaDenCmp, omegaNumCmp, omegaDenCmp, in,
                                    out);
                return;
            }
            beta = rho_new / rho * (alpha / omega);
            if (!std::isfinite(beta))
            {
                FallbackToBICGSTABR("beta is not finite", stage, rhsMagnitude,
                                    eps, rho, rho_new, alphaDen, omega0, omega1,
                                    alpha, omega, beta, rhsCmp, epsCmp, rhoCmp,
                                    alphaDenCmp, omegaNumCmp, omegaDenCmp, in,
                                    out);
                return;
            }
        }
    }
};

} // namespace Nektar::Operators::detail
