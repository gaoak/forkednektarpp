#pragma once

#include "Operators/OperatorNeuBndCond.hpp"

#include <MultiRegions/AssemblyMap/AssemblyMapCG.h>
#include <MultiRegions/ContField.h>

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{
template <typename TData>
class OperatorNeuBndCondImpl<TData, ImplStdMat>
    : public OperatorNeuBndCond<TData>
{
public:
    OperatorNeuBndCondImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorNeuBndCond<TData>(expansionList)
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        m_assmbMap = contfield->GetLocalToGlobalMap();
    }

    void apply(Field<TData, FieldState::Coeff> &inout) override
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        auto &bndCondExpansions = contfield->GetBndCondExpansions();
        auto &bndConditions     = contfield->GetBndConditions();
        auto &sign   = m_assmbMap->GetBndCondCoeffsToLocalCoeffsSign();
        auto &map    = m_assmbMap->GetBndCondCoeffsToLocalCoeffsMap();
        auto ncoeffs = m_assmbMap->GetNumLocalCoeffs();

        // Copy data from input field
        Array<OneD, NekDouble> inoutarr(ncoeffs);
        auto *inarrptr = inoutarr.data();
        auto *inptr    = inout.GetStorage().GetCPUPtr();

        for (size_t block_idx = 0; block_idx < inout.GetBlocks().size();
             ++block_idx)
        {
            auto nSize  = inout.GetBlocks()[block_idx].block_size;
            auto nElmts = inout.GetBlocks()[block_idx].num_elements;
            auto nmTot  = inout.GetBlocks()[block_idx].num_pts;

            std::copy(inptr, inptr + nElmts * nmTot, inarrptr);

            inarrptr += nElmts * nmTot;
            inptr += nSize;
        }

        size_t bndcnt = 0;
        // Add weak boundary conditions to forcing
        for (size_t i = 0; i < bndCondExpansions.size(); ++i)
        {
            if (bndConditions[i]->GetBoundaryConditionType() ==
                    SpatialDomains::eNeumann ||
                bndConditions[i]->GetBoundaryConditionType() ==
                    SpatialDomains::eRobin)
            {
                auto &bndcoeff = bndCondExpansions[i]->GetCoeffs();
                if (m_assmbMap->GetSignChange())
                {
                    for (size_t j = 0; j < bndCondExpansions[i]->GetNcoeffs();
                         j++)
                    {
                        inoutarr[map[bndcnt + j]] +=
                            sign[bndcnt + j] * bndcoeff[j];
                    }
                }
                else
                {
                    for (size_t j = 0; j < bndCondExpansions[i]->GetNcoeffs();
                         j++)
                    {
                        inoutarr[map[bndcnt + j]] += bndcoeff[j];
                    }
                }
            }
            bndcnt += bndCondExpansions[i]->GetNcoeffs();
        }

        // Copy data to output field
        auto *outarrptr = inoutarr.data();
        auto *outptr    = inout.GetStorage().GetCPUPtr();

        for (size_t block_idx = 0; block_idx < inout.GetBlocks().size();
             ++block_idx)
        {
            auto nSize  = inout.GetBlocks()[block_idx].block_size;
            auto nElmts = inout.GetBlocks()[block_idx].num_elements;
            auto nmTot  = inout.GetBlocks()[block_idx].num_pts;

            std::copy(outarrptr, outarrptr + nElmts * nmTot, outptr);

            outarrptr += nElmts * nmTot;
            outptr += nSize;
        }
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorNeuBndCondImpl<TData, ImplStdMat>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;

protected:
    AssemblyMapCGSharedPtr m_assmbMap;
};

} // namespace Nektar::Operators::detail
