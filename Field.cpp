#include "Field.hpp"
#include <MultiRegions/ExpList.h>
#include <StdRegions/StdExpansion.h>

using namespace Nektar;
using namespace LibUtilities;

std::vector<BlockAttributes> GetBlockAttributes(
    FieldState state, const MultiRegions::ExpListSharedPtr explist)
{
    std::vector<BlockAttributes> blockAttr;

    // initialize the basisKeys
    std::vector<BasisKey> prevbasisKeys(3, NullBasisKey);
    std::vector<BasisKey> thisbasisKeys(3, NullBasisKey);

    // loop over elements
    for (int i = 0; i < explist->GetNumElmts(); i++)
    {
        auto exp = explist->GetExp(i);

        // fetch basiskeys of current element
        for (int d = 0; d < exp->GetNumBases(); d++)
        {
            thisbasisKeys[d] = exp->GetBasis(d)->GetBasisKey();
        }

        // if the basis is the same as the previous one,
        // increment the number of elements
        if (thisbasisKeys == prevbasisKeys)
        {
            blockAttr.back().num_elements++;
            blockAttr.back().block_size += blockAttr.back().num_pts;
        }
        else // if not, create a new block with the number of elements = 1
        {
            size_t num_pts = state == FieldState::Phys ? exp->GetTotPoints()
                                                   : exp->GetNcoeffs();
            blockAttr.push_back({1, num_pts});
            prevbasisKeys = thisbasisKeys;
        }
    }

    return blockAttr;
}