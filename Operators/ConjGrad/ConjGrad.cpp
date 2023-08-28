#include <array>
#include <memory>
#include <cmath>
#include <algorithm>
#include <assert.h>

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/BasicUtils/Vmath.hpp>
#include <MultiRegions/ContField.h>

#include "ConjGrad.hpp"
#include "Field.hpp"

using namespace Nektar;
using namespace Nektar::MultiRegions;

#define SOLVE_VERSION 2

namespace Nektar::Operators::detail
{

#if (SOLVE_VERSION == 1)

namespace cg_debug
{

// ****************************************************************************
// Helper functions

template <typename TData, FieldState TFieldState>
void PrintField(Field<TData, TFieldState> &field)
{
    size_t N = field.GetStorage().size();
    auto iter = field.GetStorage().GetCPUPtr();
    for (int i = 0; i < N; ++i)
        std::cout << (*iter) << " ";
        iter++;
    std::cout << "\n\n";
}

template <typename TData>
void PrintArray(const int N, const Array<OneD, TData> &arr)
{
    for (int i = 0; i < N; ++i)
        std::cout << arr[i] << " ";
    std::cout << "\n\n";
}

// Copy Field -> Array
template <typename TData, FieldState TFieldState>
void CopyField2Array(
    const int N,
    Field<TData, TFieldState> &field,
    Array<OneD, TData> &arr
)
{
    auto iter = field.GetStorage().GetCPUPtr();
    for (int i = 0; i < N; ++i)
    {
        arr[i] = *iter;
        iter++;
    }
}

// Copy Array -> Field
template <typename TData, FieldState TFieldState>
void CopyArray2Field(
    const int N,
    const Array<OneD, TData> &arr,
    Field<TData, TFieldState> &field
)
{
    auto iter = field.GetStorage().GetCPUPtr();
    for (int i = 0; i < N; ++i)
    {
        *iter = arr[i];
        iter++;
    }
}

// Execute operator on input array and output array
// Converts arrays to fields and back
template <typename TData, class TOp, FieldState TFieldState>
void DoOperator(
    const int N,
    std::vector<BlockAttributes> blockAttributes,
    const Array<OneD, const TData> &pInput,
    Array<OneD, TData> &pOutput,
    std::shared_ptr<TOp> pOperator
    )
{
    // create temporary input/output fields
    auto inField = Field<TData, TFieldState>::create(blockAttributes);
    auto outField = Field<TData, TFieldState>::create(blockAttributes);

    // copy input data from array to field
    cg_debug::CopyArray2Field<TData, TFieldState>(N, pInput, inField);

    // apply operator
    pOperator->apply(inField, outField);

    // copy output data from field to array
    cg_debug::CopyField2Array<TData, TFieldState>(N, outField, pOutput);
}

// ****************************************************************************
// Functions from Nektar

// Assemble + Scatter
template <typename TData>
void DoAssembleLoc(
    std::shared_ptr<ContField> explistCF,
    const Array<OneD, TData> &pInput,
    Array<OneD, TData> &pOutput,
    const bool &ZeroDir
)
{
    //std::shared_ptr<ContField> expListCF = std::dynamic_pointer_cast<ContField>(this->m_expansionList);
    std::shared_ptr<AssemblyMapCG> assMap = explistCF->GetLocalToGlobalMap();

    //auto assMap                  = m_locToGloMap.lock();
    GlobalSysSolnType solvertype = assMap->GetGlobalSysSolnType();

    if (solvertype == eIterativeFull)
    {
        assMap->Assemble(pInput, pOutput);
        if (ZeroDir)
        {
            int nDir = assMap->GetNumGlobalDirBndCoeffs();
            Vmath::Zero(nDir, pOutput, 1);
        }
        assMap->GlobalToLocal(pOutput, pOutput);
    }
    else // bnd version.
    {
        assMap->AssembleBnd(pInput, pOutput);
        if (ZeroDir)
        {
            int nDir = assMap->GetNumGlobalDirBndCoeffs();
            Vmath::Zero(nDir, pOutput, 1);
        }
        assMap->GlobalToLocalBnd(pOutput, pOutput);
    }
}

// Conjugate gradient algorithm
template <typename TData>
void DoConjugateGradient(
    std::shared_ptr<ContField> explistCF,
    std::shared_ptr<OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>> pLHS,
    std::shared_ptr<OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>> pPrecon,
    std::vector<BlockAttributes> blockAttributes,
    const int nLocal, 
    const Array<OneD, const TData> &pInput,
    Array<OneD, TData> &pOutput
)
{
    // Allocate array storage
    Array<OneD, TData> w_A(nLocal, 0.0);
    Array<OneD, TData> s_A(nLocal, 0.0);
    Array<OneD, TData> p_A(nLocal, 0.0);
    Array<OneD, TData> r_A(nLocal, 0.0);
    Array<OneD, TData> q_A(nLocal, 0.0);
    Array<OneD, TData> wk(nLocal, 0.0);

    int k;
    TData alpha;
    TData beta;
    TData rho;
    TData rho_new;
    TData mu;
    TData eps;
    TData min_resid;
    Array<OneD, TData> vExchange(3, 0.0);

    // **** ADDITIONAL DECLARATIONS *********************
    int totalIterations;
    TData tolerance = 1e-6;
    TData rhs_magnitude;
    int maxIter = 100;
    // **************************************************

    // Copy initial residual from input
    Vmath::Vcopy(nLocal, pInput, 1, r_A, 1);

    // zero homogeneous out array ready for solution updates
    // Should not be earlier in case input vector is same as
    // output and above copy has been peformed
    Vmath::Zero(nLocal, pOutput, 1);

    // evaluate initial residual error for exit check
    //m_operator.DoAssembleLoc(r_A, wk, true);
    cg_debug::DoAssembleLoc(explistCF, r_A, wk, true);
    vExchange[2] = Vmath::Dot(nLocal, wk, r_A);
    //m_Comm->AllReduce(vExchange, Nektar::LibUtilities::ReduceSum);

    eps = vExchange[2];

    // CALCULATE RHS MAGNITUDE
    rhs_magnitude = std::sqrt(Vmath::Dot(nLocal, pInput, pInput));
    std::cout << "Rhs magnitude = " << rhs_magnitude << "\n";
    /*
    if (m_rhs_magnitude == NekConstants::kNekUnsetTData)
    {
        Set_Rhs_Magnitude(pInput);
    }
    */

    totalIterations = 0;

    // If input residual is less than tolerance skip solve.
    if (eps < tolerance * tolerance * rhs_magnitude)
    {
        /*
        if (m_verbose && m_root)
        {
            cout << "CG iterations made = " << m_totalIterations
                 << " using tolerance of " << m_tolerance
                 << " (error = " << sqrt(eps / m_rhs_magnitude)
                 << ", rhs_mag = " << sqrt(m_rhs_magnitude) << ")" << endl;
        }
        */
        std::cout << "Terminated CG early\n";
        return;
    }

    //m_operator.DoNekSysPrecon(r_A, w_A, true);
    cg_debug::DoOperator<TData, OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>, FieldState::Coeff>
        (nLocal, blockAttributes, r_A, w_A, pPrecon);
    //m_operator.DoNekSysLhsEval(w_A, s_A);
    cg_debug::DoOperator<TData, OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>, FieldState::Coeff>
        (nLocal, blockAttributes, w_A, s_A, pLHS);
    std::cout << "Initial operators done\n";
    k = 0;

    vExchange[0] = Vmath::Dot(nLocal, r_A, w_A);
    vExchange[1] = Vmath::Dot(nLocal, s_A, w_A);
    //m_Comm->AllReduce(vExchange, Nektar::LibUtilities::ReduceSum);

    rho               = vExchange[0];
    mu                = vExchange[1];
    min_resid         = rhs_magnitude;
    beta              = 0.0;
    alpha             = rho / mu;
    totalIterations = 1;

    // Continue until convergence
    while (true)
    {
        if (k >= maxIter)
        {
            /*
            if (m_root)
            {
                cout << "CG iterations made = " << m_totalIterations
                     << " using tolerance of " << m_tolerance
                     << " (error = " << sqrt(eps / m_rhs_magnitude)
                     << ", rhs_mag = " << sqrt(m_rhs_magnitude) << ")" << endl;
            }
            ROOTONLY_NEKERROR(ErrorUtil::efatal,
                              "Exceeded maximum number of iterations");
            */
            std::cout << "Exceeded max number of iterations\n";
            return;
        }

        // Compute new search direction p_k, q_k
        Vmath::Svtvp(nLocal, beta, p_A, 1, w_A, 1, p_A, 1);
        Vmath::Svtvp(nLocal, beta, q_A, 1, s_A, 1, q_A, 1);

        // Update solution x_{k+1}
        Vmath::Svtvp(nLocal, alpha, p_A, 1, pOutput, 1, pOutput, 1);

        // Update residual vector r_{k+1}
        Vmath::Svtvp(nLocal, -alpha, q_A, 1, r_A, 1, r_A, 1);

        // Apply preconditioner
        //m_operator.DoNekSysPrecon(r_A, w_A, true);
        cg_debug::DoOperator<TData, OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>, FieldState::Coeff>
            (nLocal, blockAttributes, r_A, w_A, pPrecon);
        // Perform the method-specific matrix-vector multiply operation.
        //m_operator.DoNekSysLhsEval(w_A, s_A);
        cg_debug::DoOperator<TData, OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>, FieldState::Coeff>
            (nLocal, blockAttributes, w_A, s_A, pLHS);
        
        // <r_{k+1}, w_{k+1}>
        vExchange[0] = Vmath::Dot(nLocal, r_A, w_A);
        // <s_{k+1}, w_{k+1}>
        vExchange[1] = Vmath::Dot(nLocal, s_A, w_A);
        // <r_{k+1}, r_{k+1}>
        // m_operator.DoAssembleLoc(r_A, wk, true);
        cg_debug::DoAssembleLoc(explistCF, r_A, wk, true);
        vExchange[2] = Vmath::Dot(nLocal, wk, r_A);

        // Perform inner-product exchanges
        //m_Comm->AllReduce(vExchange, Nektar::LibUtilities::ReduceSum);

        rho_new = vExchange[0];
        mu      = vExchange[1];
        eps     = vExchange[2];

        totalIterations++;

        // Test if norm is within tolerance
        if (eps < tolerance * tolerance * rhs_magnitude)
        {
            /*if (m_verbose && m_root)
            {
                cout << "CG iterations made = " << m_totalIterations
                     << " using tolerance of " << m_tolerance
                     << " (error = " << std::sqrt(eps / m_rhs_magnitude)
                     << ", rhs_mag = " << std::sqrt(m_rhs_magnitude) << ")" << endl;
            }*/
            break;
        }
        min_resid = std::min(min_resid, eps);

        // Compute search direction and solution coefficients
        beta  = rho_new / rho;
        alpha = rho_new / (mu - rho_new * beta / alpha);
        rho   = rho_new;
        k++;
    }
}

}

// Operator apply
template <typename TData>
void OperatorConjGradImpl<TData>::apply(Field<TData, FieldState::Coeff> &in, Field<TData, FieldState::Coeff> &out)
{
    // recast explist ptr as a continuous field ptr
    std::shared_ptr<ContField> explistCF = std::dynamic_pointer_cast<ContField>(this->m_expansionList);
    
    // get number of local coeffs and block attributes
    size_t nLocal = in.GetStorage().size();
    auto blockAttributes = in.GetBlocks();

    // create temporary input/output arrays
    Array<OneD, TData> pInput(nLocal);
    Array<OneD, TData> pOutput(nLocal);

    // Field -> Array
    cg_debug::CopyField2Array<TData, FieldState::Coeff>(nLocal, in, pInput);

    // Conjugate gradient algorithm
    cg_debug::DoConjugateGradient(explistCF, this->m_LHS, this->m_precon, blockAttributes, 
        nLocal, pInput, pOutput);

    // Array -> Field
    cg_debug::CopyArray2Field<TData, FieldState::Coeff>(nLocal, pOutput, out);
}

#endif

// **********************************************************************

#if (SOLVE_VERSION == 2)

/*
template <typename TData, FieldState TFieldState>
void print_field(Field<TData, TFieldState> &field)
{
    auto *iter = field.GetStorage().GetCPUPtr();
    for (int i = 0; i < field.GetStorage().size(); ++i)
    {
        std::cout << *iter << " ";
        iter++;
    }
    std::cout << "\n";
}
*/

template <typename TData>
void OperatorConjGradImpl<TData>::apply(Field<TData, FieldState::Coeff> &in, Field<TData, FieldState::Coeff> &out)
{
    // these values should be referenced from Nektar
    TData tol = 1.e-6;   // ** CHANGE THIS **
    size_t maxIter = 10; // ** CHANGE THIS **

    // get size of vector from in (this needs to be the same for out - maybe assert this?)
    size_t N = in.GetStorage().size();

    // get block attributes (must be same as out.GetBlocks())
    auto blockAttributes = in.GetBlocks();

    // create temporary fields
    auto w_A = Field<TData, FieldState::Coeff>::create(blockAttributes);
    auto s_A = Field<TData, FieldState::Coeff>::create(blockAttributes);
    auto p_A = Field<TData, FieldState::Coeff>::create(blockAttributes);
    auto r_A = Field<TData, FieldState::Coeff>::create(blockAttributes);
    auto q_A = Field<TData, FieldState::Coeff>::create(blockAttributes);
    auto wk  = Field<TData, FieldState::Coeff>::create(blockAttributes);

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

    // copy RHS into initial residual
    std::copy(p_in, p_in + N, p_r_A);

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
template <typename TData>
void OperatorConjGradImpl<TData>::assembleScatter(
    const size_t &N,
    Field<TData, FieldState::Coeff> &in,
    Field<TData, FieldState::Coeff> &out,
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
        std::cout << "Iterative full\n";

        // inArr and outArr has size = number of local coeffs
        // Assemble takes the inArr, accumulates on the shared local dofs to produce a vector of length
        // equal to the number of global coeffs in outArr (note outArr is actually larger -- size equal to num local coeffs)
        // 
        assMap->Assemble(inArr, outArr);
        
        if (ZeroDir)
        {
            // zero out the first nDir values in outArr that are the dirichlet boundary coefficients
            int nDir = assMap->GetNumGlobalDirBndCoeffs();
            Vmath::Zero(nDir, outArr, 1);
        }
        // transfer the global space outArr to local space outArr
        // i.e. distribute accumulated values on global dofs to local dofs in same array
        assMap->GlobalToLocal(outArr, outArr);
    }
    else // bnd version.
    {
        std::cout << "Bnd\n";
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

#endif

// ****************************************************************************************************************

// Register implementation with Operator Factory
// Coeff <-> Coeff
template <>
std::string OperatorConjGradImpl<double>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "ConjGrad",
        OperatorConjGradImpl<double>::instantiate, 
        ""
    );

}