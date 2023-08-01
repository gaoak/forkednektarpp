#include "Mass.hpp"
#include "Operators/OperatorBwdTrans.hpp"
#include "Operators/OperatorIProductWRTBase.hpp"
#include <StdRegions/StdExpansion.h>

namespace Nektar::Operators::detail
{

template <typename TData>
void OperatorMassImpl<TData>::apply(Field<TData, FieldState::Coeff> &in, Field<TData, FieldState::Coeff> &out)
{
    std::cout << "Applying mass operator\n";

    // create temporary field for physical points
    auto blocks = in.GetBlocks();
    auto tmp = Field<TData, FieldState::Phys>::create(blocks);

    // transform coefficients into physical points
    this->m_BwdTransOp->apply(in, tmp);
/*    
    // Initialize pointers.
    auto const *inptr = in.GetStorage().GetCPUPtr();
    auto *outptr      = out.GetStorage().GetCPUPtr();

    size_t expIdx = 0;
    for (size_t block_idx = 0; block_idx < in.GetBlocks().size();
            ++block_idx)
    {
        // Determine shape and type of the element.
        auto const expPtr = this->m_expansionList->GetExp(expIdx);
        auto nElmts       = in.GetBlocks()[block_idx].num_elements;
        auto nmTot        = expPtr->GetNcoeffs();
        auto nqTot        = expPtr->GetTotPoints();

        // Get BwdTrans matrix.
        Nektar::StdRegions::StdMatrixKey key(
            StdRegions::eBwdTrans, expPtr->DetShapeType(), *expPtr);
        auto const matPtr = expPtr->GetStdMatrix(key);

        // Perform matrix-matrix multiply.
        Blas::Dgemm('N', 'N', nqTot, nElmts, nmTot, 1.0,
                    matPtr->GetRawPtr(), nqTot, inptr, nmTot, 0.0, outptr,
                    nqTot);

        // Increment pointer and index for next element type.
        inptr += nmTot * nElmts;
        outptr += nqTot * nElmts;
        expIdx += nElmts;
    }
*/
    // take inner product of physical points
    this->m_IProductWRTBaseOp->apply(tmp, out);

    std::cout << "Finished mass operator\n";
}

// ****************************************************************************************************************

// Register implementation with Operator Factory
template <>
std::string OperatorMassImpl<double>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "Mass",
        OperatorMassImpl<double>::instantiate, 
        ""
    );
}