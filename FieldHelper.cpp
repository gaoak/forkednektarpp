#include "Field.hpp"
#include <MultiRegions/ExpList.h>
#include <StdRegions/StdExpansion.h>

std::vector<BlockAttributes> GetBlockAttributes(
        FieldState state,
        const Nektar::MultiRegions::ExpListSharedPtr explist)
{
    const int n = explist->GetNumElmts();
    std::map<std::tuple<Nektar::LibUtilities::ShapeType, unsigned int, unsigned int>,
         std::pair<size_t, size_t>> blockList;
    for (int i = 0; i < explist->GetNumElmts(); ++i)
    {
        auto e = explist->GetExp(i);
        // Accumulate the number of elements of each type
        blockList[{e->DetShapeType(),e->GetNcoeffs(),e->GetTotPoints()}].first++;
        // Store the last element id of each type
        blockList[{e->DetShapeType(),e->GetNcoeffs(),e->GetTotPoints()}].second = i;
    }
    std::vector<BlockAttributes> blockAttr;
    for (auto &x : blockList)
    {
        auto val = state == FieldState::Phys ? std::get<2>(x.first) : std::get<1>(x.first);
        blockAttr.push_back( { x.second.first, val, x.second.second } );
    }
    return blockAttr;
}