///////////////////////////////////////////////////////////////////////////////
//
// File: LocalToGlobalDataWarehouseDef.hpp
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

#include "Operators/Common/LocalToGlobalDataWarehouse.hpp"
#include "Operators/Field/Field.hpp"

#include <MultiRegions/ContField.h>

namespace Nektar::Operators
{

template <typename MemSpace, typename TPadding>
MemoryRegion<typename DeviceLocalToGlobalKey<TPadding>::value_type>
LocalToGlobalDataCreator::Create(
    const DeviceLocalToGlobalKey<TPadding> &LocToGloKey)
{
    using value_type = typename DeviceLocalToGlobalKey<TPadding>::value_type;

    // Get Local To Global Map.
    auto zeroDir  = LocToGloKey.m_zeroDir;
    auto &loc2glo = LocToGloKey.m_assemblyMap;
    auto width    = LocToGloKey.m_width;
    auto numComp  = loc2glo.size();

    Array<OneD, unsigned> ndir(numComp);
    for (unsigned i = 0; i < numComp; ++i)
    {
        ndir[i] = loc2glo[i]->GetNumGlobalDirBndCoeffs();
    }

    auto l2gmap0 = loc2glo[0]->GetLocalToGlobalMap();

    // set for fast lookup of parallel gids
    std::unordered_set<size_t> parallel_gid(loc2glo[0]->GetSREntries().begin(),
                                            loc2glo[0]->GetSREntries().end());

    // here we are using double as the type since this is what the datatype
    // of the assembled data is assumed to be. Not sure what we shoudl do it
    // is float however but legacy code is not set up for this either
    auto blockAttr =
        GetBlockAttributes<TPadding, FieldState::Coeff>(m_expansionList);

    auto nblks = blockAttr.size();
    std::map<unsigned, std::vector<std::pair<unsigned, unsigned>>> GloToLoc;

    // Set up a map of the local ids and blk ids that are
    // associated with the global id of this expansion excluding point on
    // parallel boundaries
    unsigned coeff_offset = 0;
    for (unsigned blk = 0; blk < nblks; ++blk)
    {
        auto blksize =
            blockAttr[blk].GetNumData() * blockAttr[blk].GetNumElements();

        // Gather local id in block that share the same global id.
        for (unsigned lid = 0; lid < blksize; ++lid)
        {
            auto gid = l2gmap0[lid + coeff_offset];
            if (parallel_gid.count(gid) == 0) // not a parallel gid
            {
                GloToLoc[gid].push_back(
                    std::pair<unsigned, unsigned>(blk, lid));
            }
        }
        coeff_offset += blksize;
    }

    // At this point we have a list of global ids not on parallel boundaries
    // and the related blocks and ids within the blocks that are connected
    // to this global id. Next assemble a list of gids and components that
    // should be ordered in the final output.
    std::map<unsigned, std::pair<unsigned, unsigned>> SortVals;
    std::vector<unsigned> BlkOffset(nblks);
    std::set<unsigned> gidDone;

    unsigned nvalstot = 0;
    unsigned nidx     = 0;
    coeff_offset      = 0;

    for (unsigned blk = 0; blk < nblks; ++blk)
    {
        unsigned blksize =
            blockAttr[blk].GetNumElements() * blockAttr[blk].GetNumData();
        for (int nc = 0; nc < numComp; ++nc)
        {
            for (int lid = 0; lid < blksize; ++lid)
            {
                unsigned gid = l2gmap0[lid + coeff_offset];

                if ((gidDone.count(gid * numComp + nc) == 0) &&
                    (GloToLoc.count(gid)))
                {
                    unsigned val = GloToLoc[gid].size();
                    if (zeroDir)
                    {
                        // Add point if valance is more than one
                        // or if global dof < ndir on this component
                        if ((val > 1) || (loc2glo[nc]->GetLocalToGlobalMap(
                                              lid + coeff_offset) < ndir[nc]))
                        {
                            nidx += val;
                            SortVals[nvalstot++] =
                                std::pair<unsigned, unsigned>(gid, nc);
                        }
                    }
                    else
                    {
                        // Only add  points with valence more than 1
                        if (val > 1)
                        {
                            nidx += val;
                            SortVals[nvalstot++] =
                                std::pair<unsigned, unsigned>(gid, nc);
                        }
                    }
                    gidDone.insert(gid * numComp + nc);
                }
            }
        }
        // set up block offset to allow for local id determineation
        if (blk)
        {
            BlkOffset[blk] =
                BlkOffset[blk - 1] + blockAttr[blk - 1].size() * numComp;
        }
        else
        {
            BlkOffset[blk] = 0;
        }
        coeff_offset += blksize;
    }

    // If width > 1 ensure that the valence associated with
    // SortVals[i*width] is as large the valence associated with SortVals[j]
    // for i*width < j < (i+1)*width. This is because we will use the
    // valence associated with SortVals[i*width] to store the index and sign
    // of the following values in this width block -- Neede for efficient
    // access on Devices
    if (width > 1)
    {
        unsigned ind = 0;
        while (ind < nvalstot)
        {
            auto val0  = GloToLoc[SortVals[ind].first].size();
            unsigned w = 1;
            for (; w < width; ++w)
            {
                if (ind + w == nvalstot) // ensure we do not overun map
                {
                    break;
                }
                auto valw = GloToLoc[SortVals[ind + w].first].size();

                if (valw > val0) // swap values
                {
                    auto save = SortVals[ind + w];

                    SortVals[ind + w] = SortVals[ind];
                    SortVals[ind]     = save;
                    val0              = valw;
                }
            }
            ind += width;
        }
    }

    // Decalare memory for all local to global informaiton.
    auto LocToGlo = MemoryRegion<value_type>(nvalstot + 2 + nidx);
    auto ptr = LocToGlo.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    // Fill the pointer
    //  ptr[0] = nvals = number global points i
    //  ptr[1] = starting offset of local index to be assembled
    //  ptr[2] = starting offset of local index to be assembled
    //  ptr[3] = ....
    //  ptr[nvals+1] = total number of local indices to be assembled
    //  ptr[nvals+2] = local index of point to be assembled
    //  ptr[nvals+3] = local index of point to be assembled
    //  ptr[nvals+...] =  ...
    //  ptr[nvals+1+ ptr[nvals+1]] = last  index of point to be assembled
    auto ptr1     = ptr + nvalstot + 2;
    ptr[0]        = nvalstot;
    ptr[1]        = 0;
    unsigned cnt  = 2;
    unsigned cnt1 = 0;

    for (auto &sort : SortVals)
    {
        auto gid = sort.second.first;
        auto nc  = sort.second.second;
        ptr[cnt] = ptr[cnt - 1]; // start new offset

        for (auto &iter : GloToLoc[gid])
        {
            unsigned blk = iter.first;
            unsigned lid = iter.second;

            // offset local id by Block offset and num component * each
            // block size;
            lid += BlkOffset[blk] + blockAttr[blk].size() * nc;

            ptr1[cnt1++] = lid; // set index
            ptr[cnt] += 1;      // add one to offset
        }
        cnt++;
    }

    return LocToGlo;
}

template <typename MemSpace, typename TPadding>
MemoryRegion<typename DeviceLocalToGlobalNumAssembleKey<TPadding>::value_type>
LocalToGlobalDataCreator::Create(
    const DeviceLocalToGlobalNumAssembleKey<TPadding> &LocToGloKey)
{
    using value_type =
        typename DeviceLocalToGlobalNumAssembleKey<TPadding>::value_type;

    // Get Local To Global Map.
    auto &loc2glo      = LocToGloKey.m_assemblyMap;
    auto zeroDir       = LocToGloKey.m_zeroDir;
    auto width         = LocToGloKey.m_width;
    auto dataWarehouse = this->m_expansionList->GetDataWarehouseSharedPtr();

    const auto *gsinfo =
        dataWarehouse->template GetData<NektarSpaces::HostSpace>(
            DeviceLocalToGlobalKey<TPadding>(loc2glo, zeroDir, width));

    auto nvals = gsinfo[0];
    unsigned nvalswidth =
        (nvals + (width - 1)) / width * width; // width aligned length

    // Decalare memory
    auto LocToGlo = MemoryRegion<value_type>(nvalswidth);
    auto ptr = LocToGlo.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    for (unsigned i = 0; i < nvals; ++i)
    {
        ptr[i] = gsinfo[2 + i] - gsinfo[1 + i];
    }
    // pack out array to be multiple of widths
    for (unsigned i = nvals; i < nvalswidth; ++i)
    {
        ptr[i] = 0;
    }

    return LocToGlo;
}

template <typename MemSpace, typename TPadding>
MemoryRegion<typename DeviceLocalToGlobalIndexKey<TPadding>::value_type>
LocalToGlobalDataCreator::Create(
    const DeviceLocalToGlobalIndexKey<TPadding> &LocToGloKey)
{
    using value_type =
        typename DeviceLocalToGlobalIndexKey<TPadding>::value_type;

    // Get Local To Global Map.
    auto &loc2glo      = LocToGloKey.m_assemblyMap;
    auto zeroDir       = LocToGloKey.m_zeroDir;
    auto width         = LocToGloKey.m_width;
    auto dataWarehouse = this->m_expansionList->GetDataWarehouseSharedPtr();

    const auto *gsinfo =
        dataWarehouse->template GetData<NektarSpaces::HostSpace>(
            DeviceLocalToGlobalKey<TPadding>(loc2glo, zeroDir, width));

    auto nvals = gsinfo[0];
    auto nidx  = 0;

    // calculate number of indices as sum of i*width point times width
    for (unsigned i = 0; i < nvals; i = i + width)
    {
        nidx += (gsinfo[i + 2] - gsinfo[i + 1]) * width;
    }

    // Decalare memory
    auto LocToGlo = MemoryRegion<value_type>(nidx);
    auto ptr = LocToGlo.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    unsigned offset = nvals + 2;
    unsigned cnt    = 0;
    // pack points in width consecutive order
    for (unsigned i = 0; i < nvals; i += width)
    {
        auto wres    = (i + width < nvals) ? width : nvals - i;
        auto nassmb0 = gsinfo[i + 2] - gsinfo[i + 1];

        unsigned j, k, index = 0;
        for (j = 0; j < wres; ++j)
        {
            // location of starting index for this point
            auto loc = gsinfo[i + j + 1];
            // number of points to be assembled
            auto nassmb = gsinfo[i + j + 2] - loc;
            // redistribute indices in width major format
            for (k = 0; k < nassmb; ++k)
            {
                index                    = gsinfo[offset + loc + k];
                ptr[cnt + k * width + j] = index;
            }
            // fill out any unused value with the last index.
            for (; k < nassmb0; ++k)
            {
                ptr[cnt + k * width + j] = index;
            }
        }
        // add in additional padded values
        for (; j < width; ++j)
        {
            for (k = 0; k < nassmb0; ++k)
            {
                ptr[cnt + k * width + j] = index;
            }
        }
        cnt += width * nassmb0;
    }
    return LocToGlo;
}

template <typename MemSpace, typename TPadding>
MemoryRegion<typename DeviceLocalToGlobalIndexOffsetKey<TPadding>::value_type>
LocalToGlobalDataCreator::Create(
    const DeviceLocalToGlobalIndexOffsetKey<TPadding> &LocToGloKey)
{
    using value_type =
        typename DeviceLocalToGlobalIndexOffsetKey<TPadding>::value_type;

    // Get Local To Global Map.
    auto &loc2glo      = LocToGloKey.m_assemblyMap;
    auto zeroDir       = LocToGloKey.m_zeroDir;
    auto width         = LocToGloKey.m_width;
    auto dataWarehouse = this->m_expansionList->GetDataWarehouseSharedPtr();

    const auto *gsinfo =
        dataWarehouse->template GetData<NektarSpaces::HostSpace>(
            DeviceLocalToGlobalKey<TPadding>(loc2glo, zeroDir, width));

    auto nvals = gsinfo[0];
    unsigned nvalswidth =
        (nvals + (width - 1)) / width * width; // width aligned length

    // Decalare memory
    auto LocToGlo = MemoryRegion<value_type>(nvalswidth);
    auto ptr = LocToGlo.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    // Determine the offset for index and sign data
    // possibly could be in LocalToGlobalDatawarehouse if used elsewhere
    unsigned cnt = 0;
    for (unsigned i = 0; i < nvals; i += width)
    {
        for (unsigned j = 0; j < width; ++j)
        {
            // put in lcoal offset of jth point in width
            ptr[i + j] = cnt + j;
        }
        // skip forward to next block of indices
        cnt += (gsinfo[i + 2] - gsinfo[i + 1]) * width;
    }

    return LocToGlo;
}

template <typename TPadding>
void FillSignArray(
    std::vector<unsigned> &index, const MultiRegions::ExpListSharedPtr &expList,
    const std::vector<MultiRegions::AssemblyMapCGSharedPtr> &loc2glo,
    bool zeroDir, bool signChange, int *out)
{
    auto numComp = loc2glo.size();

    Array<OneD, unsigned> ndir(numComp);
    for (unsigned i = 0; i < numComp; ++i)
    {
        ndir[i] = loc2glo[i]->GetNumGlobalDirBndCoeffs();
    }

    auto sign = loc2glo[0]->GetSignChange()
                    ? loc2glo[0]->GetLocalToGlobalSign()
                    : Array<OneD, double>(expList->GetNcoeffs(), 1.0);

    auto blockAttr = GetBlockAttributes<TPadding, FieldState::Coeff>(expList);

    auto nblks = blockAttr.size();
    std::vector<unsigned> blkoffset(nblks + 1);
    std::vector<unsigned> coeffoffset(nblks);
    blkoffset[0] = 0;

    unsigned offset = 0;
    for (unsigned blk = 0; blk < nblks; ++blk)
    {
        blkoffset[blk + 1] = blkoffset[blk] + blockAttr[blk].size() * numComp;
        coeffoffset[blk]   = offset;
        offset += blockAttr[blk].GetNumElements() * blockAttr[blk].GetNumData();
    }

    // Fill the pointer with sign of local index.
    unsigned cnt = 0;
    if (zeroDir)
    {
        for (auto idx : index)
        {
            unsigned nc = 0;
            for (int blk = 0; blk < nblks; ++blk)
            {
                if ((idx >= blkoffset[blk]) && (idx < blkoffset[blk + 1]))
                {
                    // offset index to this block
                    idx -= blkoffset[blk];
                    // offset index for number of components;
                    nc  = idx / blockAttr[blk].size();
                    idx = idx % blockAttr[blk].size();
                    // add back in coeff offset to be able to access legacy
                    // index
                    idx += coeffoffset[blk];
                    break;
                }
            }

            if (signChange)
            {
                out[cnt++] = (loc2glo[nc]->GetLocalToGlobalMap(idx) < ndir[nc])
                                 ? 0
                                 : sign[idx];
            }
            else // not sure this case will be used.
            {
                out[cnt++] = (loc2glo[nc]->GetLocalToGlobalMap(idx) < ndir[nc])
                                 ? 0
                                 : std::abs(sign[idx]);
            }
        }
    }
    else
    {
        for (auto idx : index)
        {
            // Evaluate legacy index of point
            for (int blk = 0; blk < nblks; ++blk)
            {
                if ((idx >= blkoffset[blk]) && (idx < blkoffset[blk + 1]))
                {
                    // offset index to this block
                    idx -= blkoffset[blk];
                    // offset index for number of components;
                    idx = idx % blockAttr[blk].size();
                    // add back in coeff offset to be able to access legacy
                    // index
                    idx += coeffoffset[blk];
                    break;
                }
            }

            if (signChange)
            {
                out[cnt++] = sign[idx];
            }
            else
            {
                out[cnt++] = std::abs(sign[idx]);
            }
        }
    }
}

template <typename MemSpace, typename TPadding>
MemoryRegion<typename DeviceLocalToGlobalSignKey<TPadding>::value_type>
LocalToGlobalDataCreator::Create(
    const DeviceLocalToGlobalSignKey<TPadding> &LocToGloKey)
{
    using value_type =
        typename DeviceLocalToGlobalSignKey<TPadding>::value_type;

    // Get Local To Global Map.
    auto &loc2glo      = LocToGloKey.m_assemblyMap;
    auto zeroDir       = LocToGloKey.m_zeroDir;
    auto signChange    = LocToGloKey.m_signChange;
    auto width         = LocToGloKey.m_width;
    auto dataWarehouse = this->m_expansionList->GetDataWarehouseSharedPtr();

    const auto *gsinfo =
        dataWarehouse->template GetData<NektarSpaces::HostSpace>(
            DeviceLocalToGlobalKey<TPadding>(loc2glo, zeroDir, width));

    auto nvals = gsinfo[0];
    auto nidx  = gsinfo[nvals + 1];

    // get a vector of the index values
    std::vector<unsigned> index;
    for (unsigned i = 0; i < nidx; ++i)
    {
        index.push_back(gsinfo[nvals + 2 + i]);
    }

    // evaluate the sign of each point
    std::vector<int> sign(nidx);
    FillSignArray<TPadding>(index, this->m_expansionList, loc2glo, zeroDir,
                            signChange, sign.data());

    // calculate number of sign values as sum of i*width point times width
    nidx = 0;
    for (unsigned i = 0; i < nvals; i += width)
    {
        nidx += (gsinfo[i + 2] - gsinfo[i + 1]) * width;
    }

    // Decalare memory for all local to global information.
    auto LocToGloSign = MemoryRegion<value_type>(nidx);
    auto ptr =
        LocToGloSign.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    unsigned cnt = 0;
    // pack points in width consecutive order
    for (unsigned i = 0; i < nvals; i = i + width)
    {
        auto wres    = (i + width < nvals) ? width : nvals - i;
        auto nassmb0 = (gsinfo[i + 2] - gsinfo[i + 1]);

        unsigned j, k;
        for (j = 0; j < wres; ++j)
        {
            // location of starting index for this point
            auto loc = gsinfo[i + j + 1];
            // number of sign values to be assembled
            auto nassmb = gsinfo[i + j + 2] - loc;

            for (k = 0; k < nassmb; ++k)
            {
                ptr[cnt + k * width + j] = sign[loc + k];
            }
            // zero out any unused values
            for (; k < nassmb0; ++k)
            {
                ptr[cnt + k * width + j] = 0;
            }
        }
        // add in additional padded values
        for (; j < width; ++j)
        {
            for (k = 0; k < nassmb0; ++k)
            {
                ptr[cnt + k * width + j] = 0;
            }
        }
        cnt += width * nassmb0;
    }

    return LocToGloSign;
}

template <typename MemSpace, typename TPadding>
MemoryRegion<typename DeviceBndLocalToGlobalKey<TPadding>::value_type>
LocalToGlobalDataCreator::Create(
    const DeviceBndLocalToGlobalKey<TPadding> &LocToGloKey)
{
    using value_type = typename DeviceBndLocalToGlobalKey<TPadding>::value_type;

    // Get Local To Global Map.
    auto numComp = LocToGloKey.m_numComp;

    auto contfield =
        std::dynamic_pointer_cast<MultiRegions::ContField>(m_expansionList);
    auto loc2glo = std::dynamic_pointer_cast<MultiRegions::AssemblyMapCG>(
        contfield->GetLocalToGlobalMap());

    unsigned myrank = contfield->GetSession()->GetComm()->GetRank();

    // local to global mapping from legacy code
    auto loc2glomap = loc2glo->GetLocalToGlobalMap();

    // set for fast lookup of parallel gids
    std::unordered_set<size_t> parallel_gid(loc2glo->GetSREntries().begin(),
                                            loc2glo->GetSREntries().end());

    auto blockAttr =
        GetBlockAttributes<TPadding, FieldState::Coeff>(m_expansionList);

    auto nblks = blockAttr.size();
    std::map<unsigned, std::vector<std::pair<unsigned, unsigned>>> GloToLoc;

    unsigned coeff_offset = 0;
    for (unsigned blk = 0; blk < nblks; ++blk)
    {
        auto blksize =
            blockAttr[blk].GetNumData() * blockAttr[blk].GetNumElements();

        // Gather local id in block that share the same global id.
        for (unsigned lid = 0; lid < blksize; ++lid)
        {
            auto gid = loc2glomap[lid + coeff_offset];
            if (parallel_gid.count(gid)) // Is a parallel gid
            {
                GloToLoc[gid].push_back(
                    std::pair<unsigned, unsigned>(blk, lid));
            }
        }
        coeff_offset += blksize;
    }

    // At this point we have a list of global ids on parallel boundary
    // and the related blocks and ids within the blocks that are
    // connected to this global id.

    // Get vector of send receive entries of gids in order that are required
    // for communication in AssemblyCommCG
    auto SREntries = loc2glo->GetSREntries();
    // Get from rank index of each sent/receive entry to ensure rank
    // ordering of assemble points for inter node consistency
    auto fromRank = loc2glo->GetFromRank();

    // set up block offset to allow for local id determination
    std::vector<unsigned> BlkOffset(nblks);
    std::vector<unsigned> BlkSize(nblks);
    for (unsigned blk = 0; blk < nblks; ++blk)
    {
        BlkSize[blk] = blockAttr[blk].size();
        if (blk)
        {
            BlkOffset[blk] = BlkOffset[blk - 1] + BlkSize[blk - 1] * numComp;
        }
        else
        {
            BlkOffset[blk] = 0;
        }
    }

    // Set up sorted list that can be used to fill in return data.
    std::map<unsigned, std::pair<std::vector<unsigned>, std::vector<unsigned>>>
        SortVals;
    std::map<unsigned, unsigned> GidExists;
    std::map<unsigned, std::vector<unsigned>> rankOrder;
    unsigned nvalstot = 0;
    unsigned nidx     = 0;
    // esure each compoent is done in sequence with nc in outer loop
    for (unsigned nc = 0; nc < numComp; ++nc)
    {
        for (unsigned i = 0; i < SREntries.size(); ++i)
        {
            unsigned gid = SREntries[i];

            if (GidExists.count(gid * numComp + nc))
            {
                SortVals[GidExists[gid * numComp + nc]].second.push_back(
                    i * numComp + nc);
                rankOrder[GidExists[gid * numComp + nc]].push_back(fromRank[i]);
                nidx++;
            }
            else // setup new sort value with local ids
            {
                std::pair<std::vector<unsigned>, std::vector<unsigned>> data;
                // offset for number of points and assemble order point
                nidx += 3;

                // gather the local ids
                for (auto [blk, lids] : GloToLoc[gid])
                {
                    data.first.push_back(lids + BlkSize[blk] * nc +
                                         BlkOffset[blk]);
                    nidx++;
                }
                data.second.push_back(i * numComp + nc);
                nidx++;

                SortVals[nvalstot] = data;

                GidExists[gid * numComp + nc] = nvalstot;
                rankOrder[nvalstot].push_back(fromRank[i]);
                nvalstot++;
            }
        }
    }

    // now have a list of sorted values containing a vector of local points
    // and a vector of global boundary entries. Next sort global boundary
    // entries into rank order so that boundary assembly is acheived in rank
    // order for consistency. Also identify order of this rank
    std::map<unsigned, unsigned> LocRankOrder;
    for (auto &vals : SortVals)
    {
        auto ind  = vals.first;
        auto npts = vals.second.second.size();
        ASSERTL1(npts == rankOrder[ind].size(),
                 "mismatch between bnd index and rank order");

        // determine ordering of ranks
        std::vector<unsigned> order(npts);
        for (int i = 0; i < npts; ++i)
        {
            unsigned rk  = rankOrder[ind][i];
            unsigned ord = 0;
            for (int j = 0; j < npts; ++j)
            {
                if (rk > rankOrder[ind][j])
                {
                    ord++;
                }
            }
            order[i] = ord;
        }
        std::vector<unsigned> bvals = vals.second.second;
        // reorder bvals
        for (int i = 0; i < npts; ++i)
        {
            vals.second.second[order[i]] = bvals[i];
        }

        // finally determien order of this rank
        unsigned ord = 0;
        for (int j = 0; j < npts; ++j)
        {
            if (myrank > rankOrder[ind][j])
            {
                ord++;
            }
        }
        LocRankOrder[vals.first] = ord;
    }

    // Decalare memory for all local to global informaiton.
    auto LocToGlo = MemoryRegion<value_type>(nvalstot + 2 + nidx);
    auto ptr = LocToGlo.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    // Fill the pointer
    //  ptr[0] = nvals = number global points i
    //  ptr[1] = starting offset of local index to be assembled
    //  ptr[2] = starting offset of local index to be assembled
    //  ptr[3] = ....
    //  ptr[nvals+1] = total number of indices that follow
    //
    //  ptr[nvals+1 +0] = number of local indices to be assembled (nloc)
    //  ptr[nvals+1 +1] = number of bnd index to write (nbnd)
    //  ptr[nvals+1 +2] = local index of point to be assembled
    //  ptr[nvals+1 +x] = ..
    //  ptr[nvals+1 +nloc  ] = index of boundary point to write into
    //  ptr[nvals+1 +nloc+1] = index of boundary point to write into
    //  ptr[nvals+1 +nloc+x] = ..
    //  ptr[nvals+1 +nloc+nbnd] = rank order in which to assmeble
    //
    //  repeat above pattern for nvals entries
    auto ptr1     = ptr + nvalstot + 2;
    ptr[0]        = nvalstot;
    ptr[1]        = 0;
    unsigned cnt  = 2;
    unsigned cnt1 = 0;

    for (auto &vals : SortVals)
    {
        ptr[cnt] = ptr[cnt - 1]; // start new offset

        ptr1[cnt1++] = vals.second.first.size();
        ptr1[cnt1++] = vals.second.second.size();
        for (auto &iter : vals.second.first)
        {
            ptr1[cnt1++] = iter;
        }
        for (auto &iter : vals.second.second)
        {
            ptr1[cnt1++] = iter;
        }

        // rank offset of lcoal point
        ptr1[cnt1++] = LocRankOrder[vals.first];

        ptr[cnt] += 3 + vals.second.first.size() + vals.second.second.size();
        cnt++;
    }

    return LocToGlo;
}

template <typename MemSpace, typename TPadding>
MemoryRegion<typename DeviceBndLocalToGlobalSignKey<TPadding>::value_type>
LocalToGlobalDataCreator::Create(
    const DeviceBndLocalToGlobalSignKey<TPadding> &LocToGloKey)
{
    using value_type =
        typename DeviceBndLocalToGlobalSignKey<TPadding>::value_type;

    auto &loc2glo      = LocToGloKey.m_assemblyMap;
    auto zeroDir       = LocToGloKey.m_zeroDir;
    auto signChange    = LocToGloKey.m_signChange;
    auto dataWarehouse = this->m_expansionList->GetDataWarehouseSharedPtr();

    const auto *gsinfo =
        dataWarehouse->template GetData<NektarSpaces::HostSpace>(
            DeviceBndLocalToGlobalKey<TPadding>(loc2glo.size()));

    // extract lids from gsinfo
    std::vector<unsigned> lids;
    unsigned nvals = gsinfo[0];
    auto idx_ptr   = gsinfo + nvals + 2;
    for (unsigned i = 1; i <= nvals; ++i)
    {
        auto ptr  = idx_ptr + gsinfo[i];
        auto nidx = ptr[0];
        for (unsigned j = 0; j < nidx; ++j)
        {
            lids.push_back(ptr[j + 2]);
        }
    }

    // Decalare memory for all local to global information.
    auto LocToGloSign = MemoryRegion<value_type>(lids.size());
    auto ptr =
        LocToGloSign.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    FillSignArray<TPadding>(lids, this->m_expansionList, loc2glo, zeroDir,
                            signChange, ptr);

    return LocToGloSign;
}

template <typename MemSpace, typename TPadding>
MemoryRegion<typename LocalToGlobalMaskKey<TPadding>::value_type>
LocalToGlobalDataCreator::Create(
    [[maybe_unused]] const LocalToGlobalMaskKey<TPadding> &LocToGloKey)
{
    using value_type = typename LocalToGlobalMaskKey<TPadding>::value_type;

    // Get Local To Global Map.
    auto contfield =
        std::dynamic_pointer_cast<MultiRegions::ContField>(m_expansionList);
    auto loc2glo = std::dynamic_pointer_cast<MultiRegions::AssemblyMapCG>(
        contfield->GetLocalToGlobalMap());

    auto l2gmap0 = loc2glo->GetLocalToGlobalMap();

    auto blockAttr =
        GetBlockAttributes<TPadding, FieldState::Coeff>(m_expansionList);
    unsigned ntot = 0;
    unsigned blk  = 0;
    for (blk = 0; blk < blockAttr.size(); ++blk)
    {
        ntot += blockAttr[blk].size();
    }

    // Decalare memory for all local to global informaiton.
    auto LocToGlo = MemoryRegion<value_type>(ntot);
    auto ptr = LocToGlo.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    std::set<unsigned> done;
    unsigned offset = 0;
    unsigned cnt    = 0;
    for (blk = 0; blk < blockAttr.size(); ++blk)
    {
        auto &block           = blockAttr[blk];
        auto num_elements     = block.GetNumElements();
        auto num_elements_pad = block.GetNumElementsWithPadding();
        auto num_data         = block.GetNumData();

        // Loop over chunks.
        for (unsigned el = 0; el < num_elements; ++el)
        {
            for (unsigned j = 0; j < num_data; ++j)
            {
                auto lid    = offset + el * num_data + j;
                auto gid    = l2gmap0[lid];
                auto unique = loc2glo->GetGlobalToUniversalMapUnique(gid);

                if (unique && done.count(gid) == 0)
                {
                    ptr[cnt++] = 1;
                    done.insert(gid);
                }
                else
                {
                    ptr[cnt++] = 0;
                }
            }
        }

        for (size_t el = num_elements; el < num_elements_pad; ++el)
        {
            for (unsigned j = 0; j < num_data; ++j)
            {
                ptr[cnt++] = 0;
            }
        }

        offset += num_data * num_elements;
    }
    return LocToGlo;
}

} // namespace Nektar::Operators
