#pragma once

#include "Operators/OperatorRobBndCond.hpp"

#include <MultiRegions/AssemblyMap/AssemblyMapCG.h>
#include <MultiRegions/ContField.h>

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{
template <typename TData>
class OperatorRobBndCondImpl<TData, ImplStdMat>
    : public OperatorRobBndCond<TData>
{
public:
    OperatorRobBndCondImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorRobBndCond<TData>(expansionList)
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        m_assmbMap = contfield->GetLocalToGlobalMap();
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out,
               const bool &negflag) override
    {
        auto ncoeffs = m_assmbMap->GetNumLocalCoeffs();

        // Copy data from input field
        Array<OneD, NekDouble> inarray(ncoeffs, 0.0);
        auto *inarrptr = inarray.data();
        auto *inptr    = in.GetStorage().GetCPUPtr();

        for (size_t block_idx = 0; block_idx < in.GetBlocks().size();
             ++block_idx)
        {
            auto nSize  = in.GetBlocks()[block_idx].block_size;
            auto nElmts = in.GetBlocks()[block_idx].num_elements;
            auto nmTot  = in.GetBlocks()[block_idx].num_pts;

            std::copy(inptr, inptr + nElmts * nmTot, inarrptr);

            inarrptr += nElmts * nmTot;
            inptr += nSize;
        }

        Array<OneD, NekDouble> robin(ncoeffs, 0.0);
        auto *robinptr   = robin.get();
        auto robinBCInfo = this->m_expansionList->GetRobinBCInfo();
        for (auto &r : robinBCInfo) // add robin mass matrix
        {
            auto n      = r.first;
            auto offset = this->m_expansionList->GetCoeff_Offset(n);
            auto expPtr = this->m_expansionList->GetExp(n);
            Array<OneD, NekDouble> tmp;
            for (auto rBC = r.second; rBC; rBC = rBC->next)
            {
                expPtr->AddRobinTraceContribution(
                    rBC->m_robinID, rBC->m_robinPrimitiveCoeffs,
                    inarray + offset, tmp = robin + offset);
            }
        }

        // Copy data to output field
        auto *outptr = out.GetStorage().GetCPUPtr();
        for (size_t block_idx = 0; block_idx < out.GetBlocks().size();
             ++block_idx)
        {
            auto nSize  = out.GetBlocks()[block_idx].block_size;
            auto nElmts = out.GetBlocks()[block_idx].num_elements;
            auto nmTot  = out.GetBlocks()[block_idx].num_pts;

            if (negflag)
            {
                std::transform(robinptr, robinptr + nElmts * nmTot, outptr,
                               outptr, [](const TData &rob, const TData &out) {
                                   return out - rob;
                               });
            }
            else
            {
                std::transform(robinptr, robinptr + nElmts * nmTot, outptr,
                               outptr, [](const TData &rob, const TData &out) {
                                   return out + rob;
                               });
            }

            robinptr += nElmts * nmTot;
            outptr += nSize;
        }
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorRobBndCondImpl<TData, ImplStdMat>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;

protected:
    AssemblyMapCGSharedPtr m_assmbMap;
};

} // namespace Nektar::Operators::detail
