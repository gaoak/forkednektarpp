///////////////////////////////////////////////////////////////////////////////
//
// File: GeometricDataWarehouse.hpp
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

#include "Operators/Common/NekDataWarehouse.hpp"

namespace Nektar::Operators
{

class GeometricDataCreator;

template <typename TData> class JacobianKey : public BaseKey
{
public:
    using creator = GeometricDataCreator;

    ~JacobianKey() override = default;
    JacobianKey(const unsigned int exp_idx, const unsigned int interleave_width,
                const unsigned int num_elements)
        : m_exp_idx(exp_idx), m_interleave_width(interleave_width),
          m_num_elements(num_elements)
    {
        hash_combine(m_hash, m_exp_idx, m_interleave_width, m_num_elements,
                     m_name);
    }

    unsigned int m_exp_idx;
    unsigned int m_interleave_width;
    unsigned int m_num_elements;
    typedef TData m_data_type;

private:
    inline static const std::string m_name = "JacobianKey";
};

template <typename TData> class DerivFactorKey : public BaseKey
{
public:
    using creator = GeometricDataCreator;

    ~DerivFactorKey() override = default;
    DerivFactorKey(const unsigned int exp_idx,
                   const unsigned int interleave_width,
                   const unsigned int num_elements, const bool transpose)
        : m_exp_idx(exp_idx), m_interleave_width(interleave_width),
          m_num_elements(num_elements), m_transpose(transpose)
    {
        hash_combine(m_hash, m_exp_idx, m_interleave_width, m_num_elements,
                     m_transpose, m_name);
    }

    unsigned int m_exp_idx;
    unsigned int m_interleave_width;
    unsigned int m_num_elements;
    bool m_transpose;
    typedef TData m_data_type;

private:
    inline static const std::string m_name = "DerivFactorKey";
};

class GeometricDataCreator : public DataCreatorClass
{
public:
    ~GeometricDataCreator() override = default;
    GeometricDataCreator(const MultiRegions::ExpListSharedPtr &expansionList)
        : m_expansionList(expansionList)
    {
    }

    template <typename MemSpace, typename TData>
    MemoryRegion<TData> Create(const JacobianKey<TData> &jacobianKey,
                               const unsigned int alignment)
    {
        auto vector_width = NektarSpaces::vector_width<TData>::value;

        auto exp_idx          = jacobianKey.m_exp_idx;
        auto interleave_width = jacobianKey.m_interleave_width;
        auto num_elements     = jacobianKey.m_num_elements;
        auto num_elmt_groups =
            ((num_elements + vector_width - 1) / vector_width) * vector_width /
            interleave_width;

        auto expPtr = m_expansionList->GetExp(exp_idx);

        // Deformed geometry.
        if (expPtr->GetMetricInfo()->GetGtype() == SpatialDomains::eDeformed)
        {
            Array<OneD, Array<OneD, NekDouble>> jacArray(interleave_width);
            auto jac = MemoryRegion<TData>::Create(
                num_elmt_groups * interleave_width * expPtr->GetTotPoints(),
                alignment);
            auto jacptr =
                jac.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

            // Loop over chunks.
            for (unsigned int chunk = 0, el = 0, jac_id = 0;
                 chunk < num_elmt_groups; ++chunk)
            {
                for (unsigned int i = 0; i < interleave_width; ++i, ++el)
                {
                    if (el < num_elements)
                    {
                        jacArray[i] = m_expansionList->GetExp(exp_idx + el)
                                          ->GetMetricInfo()
                                          ->GetJac(expPtr->GetPointsKeys());
                    }
                    else
                    {
                        jacArray[i] =
                            Array<OneD, NekDouble>(expPtr->GetTotPoints(), 0.0);
                    }
                }

                for (unsigned int pt = 0; pt < expPtr->GetTotPoints(); ++pt)
                {
                    for (unsigned int i = 0; i < interleave_width;
                         ++i, ++jac_id)
                    {
                        jacptr[jac_id] = jacArray[i][pt];
                    }
                }
            }

            return jac;
        }
        // Regular geometry.
        else
        {
            auto jac = MemoryRegion<TData>::Create(
                num_elmt_groups * interleave_width, alignment);
            auto jacptr =
                jac.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

            // Loop over chunks.
            for (unsigned int chunk = 0, el = 0; chunk < num_elmt_groups;
                 ++chunk)
            {
                for (unsigned int i = 0; i < interleave_width; ++i, ++el)
                {
                    if (el < num_elements)
                    {
                        auto &auxJac = m_expansionList->GetExp(exp_idx + el)
                                           ->GetMetricInfo()
                                           ->GetJac(expPtr->GetPointsKeys());
                        jacptr[el] = auxJac[0];
                    }
                    else
                    {
                        jacptr[el] = 0.0;
                    }
                }
            }

            return jac;
        }
    }

    template <typename MemSpace, typename TData>
    MemoryRegion<TData> Create(const DerivFactorKey<TData> &derivFactorKey,
                               const unsigned int alignment)
    {
        auto vector_width = NektarSpaces::vector_width<TData>::value;

        auto exp_idx          = derivFactorKey.m_exp_idx;
        auto interleave_width = derivFactorKey.m_interleave_width;
        auto num_elements     = derivFactorKey.m_num_elements;
        auto transpose        = derivFactorKey.m_transpose;
        auto num_elmt_groups =
            ((num_elements + vector_width - 1) / vector_width) * vector_width /
            interleave_width;

        auto expPtr         = m_expansionList->GetExp(exp_idx);
        unsigned int nDim   = expPtr->GetShapeDimension();
        unsigned int nCoord = expPtr->GetCoordim();

        unsigned int range1 =
            transpose ? nDim * nCoord : expPtr->GetTotPoints();
        unsigned int range2 =
            transpose ? expPtr->GetTotPoints() : nDim * nCoord;

        // Deformed geometry.
        if (expPtr->GetMetricInfo()->GetGtype() == SpatialDomains::eDeformed)
        {
            // Allocate memory and get pointer.
            auto df = MemoryRegion<TData>::Create(
                num_elmt_groups * interleave_width * expPtr->GetTotPoints() *
                    nDim * nCoord,
                alignment);
            auto dfptr =
                df.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

            // Loop over chunks.
            for (unsigned int chunk = 0, el = 0, df_id = 0;
                 chunk < num_elmt_groups; ++chunk)
            {
                for (unsigned int index1 = 0; index1 < range1; ++index1)
                {
                    for (unsigned int index2 = 0; index2 < range2; ++index2)
                    {
                        for (unsigned int i = 0; i < interleave_width;
                             ++i, ++df_id)
                        {
                            if (el + i < num_elements)
                            {
                                unsigned int d  = transpose ? index1 : index2;
                                unsigned int pt = transpose ? index2 : index1;

                                auto &tmp =
                                    m_expansionList->GetExp(exp_idx + el + i)
                                        ->GetMetricInfo()
                                        ->GetDerivFactors(
                                            expPtr->GetPointsKeys());
                                dfptr[df_id] = tmp[d][pt];
                            }
                            else
                            {
                                dfptr[df_id] = 0.0;
                            }
                        }
                    }
                }

                el += interleave_width;
            }

            return df;
        }
        // Regular geometry.
        else
        {
            auto df = MemoryRegion<TData>::Create(
                num_elmt_groups * interleave_width * nDim * nCoord, alignment);
            auto dfptr =
                df.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

            // Loop over chunks.
            for (unsigned int chunk = 0, el = 0, df_id = 0;
                 chunk < num_elmt_groups; ++chunk)
            {
                for (unsigned int d = 0; d < nDim * nCoord; ++d)
                {
                    for (unsigned int i = 0; i < interleave_width; ++i, ++df_id)
                    {
                        if (el + i < num_elements)
                        {
                            auto &tmp =
                                m_expansionList->GetExp(exp_idx + el + i)
                                    ->GetMetricInfo()
                                    ->GetDerivFactors(expPtr->GetPointsKeys());
                            dfptr[df_id] = tmp[d][0];
                        }
                        else
                        {
                            dfptr[df_id] = 0.0;
                        }
                    }
                }

                el += interleave_width;
            }

            return df;
        }
    }

    inline static const std::string m_name = "GeometricDataCreator";

private:
    MultiRegions::ExpListSharedPtr m_expansionList;
};

} // namespace Nektar::Operators
