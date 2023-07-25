#include <array>
#include <memory>
#include <cmath>
#include <algorithm>

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/BasicUtils/Vmath.hpp>
#include <MultiRegions/ContField.h>

#include "CG.hpp"

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{

template <typename TData, FieldState TFieldState>
void OperatorConjGradImpl<TData, TFieldState>::apply(Field<TData, TFieldState> &in, Field<TData, TFieldState> &out)
{
    // these values should be referenced from Nektar
    TData tol = 1.e-6;
    size_t maxIter = 100000;

    // get size of vector from in (this needs to be the same for out - maybe assert this?)
    size_t N = in.GetStorage().size();

    // get block attributes (must be same as out.GetBlocks())
    auto blockAttributes = in.GetBlocks();

    // create temporary fields
    auto w_A = Field<TData, TFieldState>::create(blockAttributes);
    auto s_A = Field<TData, TFieldState>::create(blockAttributes);
    auto p_A = Field<TData, TFieldState>::create(blockAttributes);
    auto r_A = Field<TData, TFieldState>::create(blockAttributes);
    auto q_A = Field<TData, TFieldState>::create(blockAttributes);
    auto wk  = Field<TData, TFieldState>::create(blockAttributes);

    // store pointers to temporary fields
    auto *p_in  = in.GetStorage().GetCPUPtr();
    auto *p_out = out.GetStorage().GetCPUPtr();
    auto *p_w_A = w_A.GetStorage().GetCPUPtr();
    auto *p_s_A = s_A.GetStorage().GetCPUPtr();
    auto *p_p_A = p_A.GetStorage().GetCPUPtr();
    auto *p_r_A = r_A.GetStorage().GetCPUPtr();
    auto *p_q_A = q_A.GetStorage().GetCPUPtr();
    auto *p_wk  = wk.GetStorage().GetCPUPtr();

    // set the fields to zero
    std::fill(p_out, p_out + N, 0.);
    std::fill(p_w_A, p_w_A + N, 0.);
    std::fill(p_s_A, p_s_A + N, 0.);
    std::fill(p_p_A, p_p_A + N, 0.);
    std::fill(p_r_A, p_r_A + N, 0.);
    std::fill(p_q_A, p_q_A + N, 0.);
    std::fill(p_wk,  p_wk + N, 0.);

    size_t k;
    size_t totalIterations;
    TData rhsMagnitude;
    TData alpha;
    TData beta;
    TData rho;
    TData rho_new;
    TData mu;
    TData eps; 
    TData min_resid;
    std::array<TData, 3> vExchange;

    // initial residual    
    assembleScatter(N, r_A, wk, true);
    vExchange[2] = std::inner_product(p_wk, p_wk + N, p_r_A, 0.);
    //m_Comm->AllReduce(vExchange, Nektar::LibUtilities::ReduceSum);

    // calculate rhs magnitude
    rhsMagnitude = 0.;
    std::for_each(p_in, p_in + N, [&rhsMagnitude](const TData &x)
    {
        rhsMagnitude += x*x;
    });
    rhsMagnitude = std::sqrt(rhsMagnitude);

    eps = vExchange[2];
    if (eps < tol*tol*rhsMagnitude)
        return;

    totalIterations = 0;

    this->m_precon->apply(r_A, w_A);
    this->m_LHS->apply(w_A, s_A);

    k = 0;

    vExchange[0] = std::inner_product(p_r_A, p_r_A + N, p_w_A, 0.);
    vExchange[1] = std::inner_product(p_s_A, p_s_A + N, p_w_A, 0.);
    //m_Comm->AllReduce(vExchange, Nektar::LibUtilities::ReduceSum);

    rho = vExchange[0];
    mu = vExchange[1];
    min_resid = rhsMagnitude;
    beta = 0.;
    alpha = rho / mu;
    totalIterations = 1;

    while (true)
    {
        if (k >= maxIter)
        {
            std::cout << "Exceeded max iterations\n";
            return;
        }

        // Compute new search direction p_k, q_k
        //Vmath::Svtvp(nLocal, beta, p_A, 1, w_A, 1, p_A, 1);
        //Vmath::Svtvp(nLocal, beta, q_A, 1, s_A, 1, q_A, 1);
        std::transform(p_p_A, p_p_A + N, p_w_A, p_p_A, [&](const TData &pElem, const TData &wElem){
            return beta * pElem + wElem;
        });
        std::transform(p_q_A, p_q_A + N, p_s_A, p_q_A, [&](const TData &qElem, const TData &sElem){
            return beta * qElem + sElem;
        });

        // Update solution x_{k+1}
        //Vmath::Svtvp(nLocal, alpha, p_A, 1, pOutput, 1, pOutput, 1);
        std::transform(p_p_A, p_p_A + N, p_out, p_out, [&](const TData &pElem, const TData &xElem){
            return alpha * pElem + xElem;
        });
        
        // Update residual vector r_{k+1}
        //Vmath::Svtvp(nLocal, -alpha, q_A, 1, r_A, 1, r_A, 1);
        std::transform(p_q_A, p_q_A + N, p_r_A, p_r_A, [&](const TData &qElem, const TData &rElem){
            return -alpha * qElem + rElem;
        });

        // Apply preconditioner
        //m_operator.DoNekSysPrecon(r_A, w_A, true);
        this->m_precon->apply(r_A, w_A);

        // Perform the method-specific matrix-vector multiply operation.
        //m_operator.DoNekSysLhsEval(w_A, s_A);
        this->m_LHS->apply(w_A, s_A);

        // <r_{k+1}, w_{k+1}>
        //vExchange[0] = Vmath::Dot(nLocal, r_A, w_A);
        vExchange[0] = std::inner_product(p_r_A, p_r_A + N, p_w_A, 0.);
        
        // <s_{k+1}, w_{k+1}>
        //vExchange[1] = Vmath::Dot(nLocal, s_A, w_A);
        vExchange[1] = std::inner_product(p_s_A, p_s_A + N, p_w_A, 0.);
        
        // <r_{k+1}, r_{k+1}>
        //m_operator.assembleScatter(r_A, wk, true);
        //vExchange[2] = Vmath::Dot(nLocal, wk, r_A);
        assembleScatter(N, r_A, wk, true);
        vExchange[2] = std::inner_product(p_wk, p_wk + N, p_r_A, 0.);

        // Perform inner-product exchanges
        //m_Comm->AllReduce(vExchange, Nektar::LibUtilities::ReduceSum);

        rho_new = vExchange[0];
        mu      = vExchange[1];
        eps     = vExchange[2];

        std::cout << "Iteration " << k << " -- eps = " << eps << "\n";

        totalIterations++;

        // Test if norm is within tolerance
        if (eps < tol * tol * rhsMagnitude)
            break;
        min_resid = std::min(min_resid, eps);

        // Compute search direction and solution coefficients
        beta  = rho_new / rho;
        alpha = rho_new / (mu - rho_new * beta / alpha);
        rho   = rho_new;
        k++;
    }
}

// Assembly to global space followed by scatter to local space
template <typename TData, FieldState TFieldState>
void OperatorConjGradImpl<TData, TFieldState>::assembleScatter(
    const size_t &N,
    Field<TData, TFieldState> &in,
    Field<TData, TFieldState> &out,
    const bool &ZeroDir
)
{
    // FIELD -> ARRAY
    // create array objects for input, output as work around to use existing functions that take array objects
    Array<OneD, TData> inArr(N, in.GetStorage().GetCPUPtr());
    Array<OneD, TData> outArr(N, out.GetStorage().GetCPUPtr());

    // ***********************************************************************

    // cast the expansion list to continuous field to access assembly map
    std::shared_ptr<ContField> expListCF = std::dynamic_pointer_cast<ContField>(this->m_expansionList);

    // get assembly map
    std::shared_ptr<AssemblyMapCG> assMap = expListCF->GetLocalToGlobalMap();
    // auto assMap                  = m_locToGloMap.lock();

    // do assembly followed by scatter
    // Lifted from NekLinSysCGLoc
    GlobalSysSolnType solvertype = assMap->GetGlobalSysSolnType();
    if (solvertype == eIterativeFull)
    {
        assMap->Assemble(inArr, outArr);
        if (ZeroDir)
        {
            int nDir = assMap->GetNumGlobalDirBndCoeffs();
            Vmath::Zero(nDir, outArr, 1);
        }
        assMap->GlobalToLocal(outArr, outArr);
    }
    else // bnd version.
    {
        assMap->AssembleBnd(inArr, outArr);
        if (ZeroDir)
        {
            int nDir = assMap->GetNumGlobalDirBndCoeffs();
            Vmath::Zero(nDir, outArr, 1);
        }
        assMap->GlobalToLocalBnd(outArr, outArr);
    }

    // ***********************************************************************
    
    // ARRAY -> FIELD
    // transfer data from output array to output field
    std::copy(outArr.data(), outArr.data() + N, out.GetStorage().GetCPUPtr());
}

// ****************************************************************************************************************

// Register implementation with Operator Factory
// Coeff <-> Coeff
template <>
std::string OperatorConjGradImpl<double, FieldState::Coeff>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "ConjGrad",
        OperatorConjGradImpl<double, FieldState::Coeff>::instantiate, 
        ""
    );

}