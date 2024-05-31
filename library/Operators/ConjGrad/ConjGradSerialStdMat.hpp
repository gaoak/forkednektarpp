///////////////////////////////////////////////////////////////////////////////
//
// File: ConjGradSerialStdMat.hpp
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

#include "ConjGradImplBase.hpp"

namespace Nektar::Operators::detail
{

// Standard matrix implementation
template <typename ExecSpace, typename Implementation, typename TData,
          typename = typename std::enable_if<
              std::is_same<ExecSpace, NektarSpaces::Serial>::value &&
              std::is_same<Implementation, Operators::StdMat>::value>::type>
class OperatorConjGradImpl
    : public OperatorConjGradImplBase<ExecSpace, Implementation, TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorConjGradImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorConjGradImplBase<ExecSpace, Implementation, TData>(
              expansionList)
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        m_rowComm = contfield->GetSession()->GetComm()->GetRowComm();
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        if (m_rowComm->GetSize() > 1 && m_rowComm->GetRank() == 0)
        {
            std::cout << "Solve ConjGrad with " << m_rowComm->GetSize()
                      << " processors" << std::endl;
        }

        // Set the fields to zero
        out.initialize(0);

        this->m_w_A.initialize(0);
        this->m_s_A.initialize(0);
        this->m_wk.initialize(0);

        this->m_p_A.initialize(0);
        this->m_q_A.initialize(0);

        // Copy the data from the input field.
        Array<OneD, TData> inArray = in.toArray();
        Array<OneD, TData> outArray(this->m_nloc, 0.0);

        // Pointers to the temporary fields
        auto *inPtr  = inArray.get();
        auto *outPtr = outArray.get();

        auto *w_APtr = this->m_w_A.template GetPtr<MemSpace>();
        auto *s_APtr = this->m_s_A.template GetPtr<MemSpace>();
        auto *r_APtr = this->m_r_A.template GetPtr<MemSpace>();
        auto *wkPtr  = this->m_wk.template GetPtr<MemSpace>();

        auto *p_APtr = this->m_p_A.template GetPtr<MemSpace>();
        auto *q_APtr = this->m_q_A.template GetPtr<MemSpace>();

        // Convergence parameters
        size_t totalIterations = 0;
        TData rhsMagnitude;
        TData alpha;
        TData beta;
        TData rho;
        TData rho_new;
        TData mu;
        TData eps;
        Array<OneD, TData> vExchange(3, 0.0);

        // Copy RHS into the initial residual
        std::copy(inPtr, inPtr + this->m_nloc, r_APtr);

        // Assembly (communication)
        this->m_assmbScatrOp->apply(this->m_r_A, this->m_wk, true);
        vExchange[2] =
            std::inner_product(wkPtr, wkPtr + this->m_nloc, r_APtr, 0.0);

        // Perform inner-product exchanges
        m_rowComm->AllReduce(vExchange, Nektar::LibUtilities::ReduceSum);

        eps = vExchange[2];

        // Calculate the rhs magnitude
        this->m_assmbScatrOp->apply(this->m_r_A, this->m_wk);
        rhsMagnitude =
            std::inner_product(inPtr, inPtr + this->m_nloc, wkPtr, 0.0);
        m_rowComm->AllReduce(rhsMagnitude, Nektar::LibUtilities::ReduceSum);
        rhsMagnitude = (rhsMagnitude > 1.0e-6) ? rhsMagnitude : 1.0;

        // If the input residual is less than the tolerance skip solve.
        if (eps < this->m_tol * this->m_tol * rhsMagnitude)
        {
            return;
        }

        // Anytime there is a mix of internal kernel calls and
        // external operator calls. The memory region being used must
        // be marked as being valid which more importantly invalidates
        // the sibling memory region.
        this->m_r_A.template setValid<MemSpace>();

        // Apply the preconditioner
        this->m_precon->apply(this->m_r_A, this->m_w_A);

        // Perform the method-specific matrix-vector multiply operation.
        this->m_LHS->apply(this->m_w_A, this->m_s_A);

        // Apply the Robin BCs
        this->m_robBndCondOp->apply(this->m_w_A, this->m_s_A);

        // Anytime there is a mix of internal kernel calls and
        // external operator calls the memory region being used must
        // be copied back.
        w_APtr = this->m_w_A.template GetPtr<MemSpace>();
        s_APtr = this->m_s_A.template GetPtr<MemSpace>();

        vExchange[0] =
            std::inner_product(r_APtr, r_APtr + this->m_nloc, w_APtr, 0.0);
        vExchange[1] =
            std::inner_product(s_APtr, s_APtr + this->m_nloc, w_APtr, 0.0);

        m_rowComm->AllReduce(vExchange, Nektar::LibUtilities::ReduceSum);

        rho             = vExchange[0];
        mu              = vExchange[1];
        beta            = 0.0;
        alpha           = rho / mu;
        totalIterations = 1;

        while (true)
        {
            if (totalIterations > this->m_maxIter)
            {
                std::stringstream msg;
                msg << "Exceeded max iterations: " << totalIterations;
                WARNINGL0(false, msg.str());

                return;
            }

            // Compute new search direction p_k
            std::transform(p_APtr, p_APtr + this->m_nloc, w_APtr, p_APtr,
                           [&beta](const TData &pElem, const TData &wElem) {
                               return beta * pElem + wElem;
                           });

            // Compute new search direction q_k
            std::transform(q_APtr, q_APtr + this->m_nloc, s_APtr, q_APtr,
                           [&beta](const TData &qElem, const TData &sElem) {
                               return beta * qElem + sElem;
                           });

            // Update solution x_{k+1}
            std::transform(p_APtr, p_APtr + this->m_nloc, outPtr, outPtr,
                           [&alpha](const TData &pElem, const TData &xElem) {
                               return alpha * pElem + xElem;
                           });

            // Update residual vector r_{k+1}
            std::transform(q_APtr, q_APtr + this->m_nloc, r_APtr, r_APtr,
                           [&alpha](const TData &qElem, const TData &rElem) {
                               return -alpha * qElem + rElem;
                           });

            // Anytime there is a mix of internal kernel calls and
            // external operator calls. The memory region being
            // used must be marked as being valid which more
            // importantly invalidates the sibling memory region.
            this->m_r_A.template setValid<MemSpace>();

            // Apply preconditioner
            this->m_precon->apply(this->m_r_A, this->m_w_A);

            // Perform the method-specific matrix-vector multiply operation.
            this->m_LHS->apply(this->m_w_A, this->m_s_A);

            // Apply Robin BCs
            this->m_robBndCondOp->apply(this->m_w_A, this->m_s_A);

            // Anytime there is a mix of internal kernel calls and
            // external operator calls the memory region being used
            // must be copied back.
            w_APtr = this->m_w_A.template GetPtr<MemSpace>();
            s_APtr = this->m_s_A.template GetPtr<MemSpace>();

            // <r_{k+1}, w_{k+1}>
            vExchange[0] =
                std::inner_product(r_APtr, r_APtr + this->m_nloc, w_APtr, 0.0);

            // <s_{k+1}, w_{k+1}>
            vExchange[1] =
                std::inner_product(s_APtr, s_APtr + this->m_nloc, w_APtr, 0.0);

            // <r_{k+1}, r_{k+1}>
            this->m_assmbScatrOp->apply(this->m_r_A, this->m_wk, true);
            vExchange[2] =
                std::inner_product(wkPtr, wkPtr + this->m_nloc, r_APtr, 0.0);

            // Perform inner-product exchanges
            m_rowComm->AllReduce(vExchange, Nektar::LibUtilities::ReduceSum);

            rho_new = vExchange[0];
            mu      = vExchange[1];
            eps     = vExchange[2];

            // std::cout << "Iteration " << totalIterations << " -- eps = " <<
            // eps
            //           << "\n";

            totalIterations++;

            // Test if norm is within tolerance
            if (eps < this->m_tol * this->m_tol * rhsMagnitude)
            {
                break;
            }

            // Compute search direction and solution coefficients
            beta  = rho_new / rho;
            alpha = rho_new / (mu - rho_new * beta / alpha);
            rho   = rho_new;
        }

        // Copy the data to the output field.
        out.template copyArray<MemSpace>(outArray);
    }

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorConjGradImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

private:
    LibUtilities::CommSharedPtr m_rowComm;
};

} // namespace Nektar::Operators::detail
