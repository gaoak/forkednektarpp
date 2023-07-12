#include "Field.hpp"
#include <MultiRegions/ExpList.h>

using namespace Nektar;
using namespace LibUtilities;

std::vector<BlockAttributes> GetBlockAttributes(
    FieldState state, const MultiRegions::ExpListSharedPtr explist)
{
    std::vector<BlockAttributes> blockAttr;

    // initialize the basisKeys
    std::vector<BasisKey> prevbasisKeys(3, NullBasisKey);
    std::vector<BasisKey> thisbasisKeys(3, NullBasisKey);
    int prevIsDeformed = -1, thisIsDeformed = -1;

    // loop over elements
    for (int i = 0; i < explist->GetNumElmts(); i++)
    {
        auto expPtr = explist->GetExp(i);

        // fetch basiskeys of current element
        for (int d = 0; d < expPtr->GetNumBases(); d++)
        {
            thisbasisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
        }

        thisIsDeformed = expPtr->GetMetricInfo()->GetGtype();

        // if the basis is the same as the previous one,
        // increment the number of elements
        if (thisbasisKeys == prevbasisKeys && thisIsDeformed == prevIsDeformed)
        {
            blockAttr.back().num_elements++;
            blockAttr.back().block_size += blockAttr.back().num_pts;
        }
        else // if not, create a new block with the number of elements = 1
        {
            size_t num_pts = state == FieldState::Phys ? expPtr->GetTotPoints()
                                                       : expPtr->GetNcoeffs();
            blockAttr.push_back({1, num_pts});
            prevbasisKeys  = thisbasisKeys;
            prevIsDeformed = thisIsDeformed;
        }
    }

    return blockAttr;
}
