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

#if defined(_MSC_VER)
#undef max
#undef min
#endif

namespace Nektar::Operators
{

class GeometricDataCreator;

template <typename TData> class JacobianKey : public BaseKey
{
    friend class GeometricDataCreator;

public:
    using creator = GeometricDataCreator;
    typedef TData value_type;

    ~JacobianKey() override = default;

    JacobianKey(const unsigned int exp_idx, const unsigned int interleave_width,
                const unsigned int num_elements)
        : m_exp_idx(exp_idx), m_interleave_width(interleave_width),
          m_num_elements(num_elements)
    {
        hash_combine(m_hash, m_exp_idx, m_interleave_width, m_num_elements,
                     typeid(value_type).name(), "JacobianKey");
    }

private:
    unsigned int m_exp_idx;
    unsigned int m_interleave_width;
    unsigned int m_num_elements;
};

template <typename TData> class DerivFactorKey : public BaseKey
{
    friend class GeometricDataCreator;

public:
    using creator = GeometricDataCreator;
    typedef TData value_type;

    ~DerivFactorKey() override = default;

    DerivFactorKey(const unsigned int exp_idx,
                   const unsigned int interleave_width,
                   const unsigned int num_elements, const bool transpose)
        : m_exp_idx(exp_idx), m_interleave_width(interleave_width),
          m_num_elements(num_elements), m_transpose(transpose)
    {
        hash_combine(m_hash, m_exp_idx, m_interleave_width, m_num_elements,
                     m_transpose, typeid(value_type).name(), "DerivFactorKey");
    }

private:
    unsigned int m_exp_idx;
    unsigned int m_interleave_width;
    unsigned int m_num_elements;
    bool m_transpose;
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
        const auto vector_width = NektarSpaces::vector_width<TData>::value;

        const auto exp_idx          = jacobianKey.m_exp_idx;
        const auto interleave_width = jacobianKey.m_interleave_width;
        const auto num_elements     = jacobianKey.m_num_elements;
        const auto num_elmt_groups =
            ((num_elements + vector_width - 1) / vector_width) * vector_width /
            interleave_width;

        auto expPtr = m_expansionList->GetExp(exp_idx);

        // Deformed geometry.
        if (expPtr->GetMetricInfo()->GetGtype() == SpatialDomains::eDeformed)
        {
            const auto memsize =
                num_elmt_groups * interleave_width * expPtr->GetTotPoints();
            Array<OneD, Array<OneD, NekDouble>> jacArray(interleave_width);
            auto jac = MemoryRegion<TData>::Create(memsize, alignment);
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
                    for (unsigned int i = 0; i < interleave_width; ++i)
                    {
                        *(jacptr++) = jacArray[i][pt];
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
        const auto vector_width = NektarSpaces::vector_width<TData>::value;

        const auto exp_idx          = derivFactorKey.m_exp_idx;
        const auto interleave_width = derivFactorKey.m_interleave_width;
        const auto num_elements     = derivFactorKey.m_num_elements;
        const auto transpose        = derivFactorKey.m_transpose;
        const auto num_elmt_groups =
            ((num_elements + vector_width - 1) / vector_width) * vector_width /
            interleave_width;

        auto expPtr               = m_expansionList->GetExp(exp_idx);
        const unsigned int nDim   = expPtr->GetShapeDimension();
        const unsigned int nCoord = expPtr->GetCoordim();

        const unsigned int range1 =
            transpose ? nDim * nCoord : expPtr->GetTotPoints();
        const unsigned int range2 =
            transpose ? expPtr->GetTotPoints() : nDim * nCoord;

        // Deformed geometry.
        if (expPtr->GetMetricInfo()->GetGtype() == SpatialDomains::eDeformed)
        {
            // Allocate memory and get pointer.
            const auto memsize = num_elmt_groups * interleave_width *
                                 expPtr->GetTotPoints() * nDim * nCoord;
            auto df = MemoryRegion<TData>::Create(memsize, alignment);
            auto dfptr =
                df.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

            // Loop over chunks.
            for (unsigned int chunk = 0, el = 0; chunk < num_elmt_groups;
                 ++chunk)
            {
                for (unsigned int index1 = 0; index1 < range1; ++index1)
                {
                    for (unsigned int index2 = 0; index2 < range2; ++index2)
                    {
                        for (unsigned int i = 0; i < interleave_width; ++i)
                        {
                            if (el + i < num_elements)
                            {
                                const unsigned int d =
                                    transpose ? index1 : index2;
                                const unsigned int pt =
                                    transpose ? index2 : index1;

                                auto &tmp =
                                    m_expansionList->GetExp(exp_idx + el + i)
                                        ->GetMetricInfo()
                                        ->GetDerivFactors(
                                            expPtr->GetPointsKeys());
                                *(dfptr++) = tmp[d][pt];
                            }
                            else
                            {
                                *(dfptr++) = 0.0;
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
            for (unsigned int chunk = 0, el = 0; chunk < num_elmt_groups;
                 ++chunk)
            {
                for (unsigned int d = 0; d < nDim * nCoord; ++d)
                {
                    for (unsigned int i = 0; i < interleave_width; ++i)
                    {
                        if (el + i < num_elements)
                        {
                            auto &tmp =
                                m_expansionList->GetExp(exp_idx + el + i)
                                    ->GetMetricInfo()
                                    ->GetDerivFactors(expPtr->GetPointsKeys());
                            *(dfptr++) = tmp[d][0];
                        }
                        else
                        {
                            *(dfptr++) = 0.0;
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
