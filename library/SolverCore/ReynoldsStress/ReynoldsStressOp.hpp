///////////////////////////////////////////////////////////////////////////////
//
// File: ReynoldsStressOp.hpp
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
// Description: Device-portable velocity-product operator.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <string>
#include <vector>

#include <MultiRegions/Common/Operator.hpp>
#include <SolverCore/SolverCore.hpp>

namespace Nektar::SolverCore
{

/**
 * @brief Accumulate the scaled pairwise products of the velocity components.
 *
 * For the \f$ d \f$ velocity components \f$ u_a \f$ selected with
 * Configure(), Apply() adds \f$ s \, u_a u_b \f$, \f$ a \le b \f$,
 * pointwise to the \f$ d(d+1)/2 \f$ components of a product field, ordered
 * \f$ (a,b) = (0,0), (0,1), \ldots, (0,d-1), (1,1), \ldots, (d-1,d-1) \f$,
 * i.e. \f$ uu, uv, uw, vv, vw, ww \f$ in three dimensions. The scale
 * \f$ s \f$ is one unless changed with SetScale().
 *
 * The velocity field may carry further components, such as passive scalars;
 * only the selected ones are read. The product field has the operator's block
 * attributes. A product block at another interleave width than the velocity's
 * is reshaped to it, keeping its content, so that the products leave Apply()
 * laid out like the velocity. Block \f$ b \f$ executes on stream
 * \f$ b+1 \f$.
 */
template <typename TData>
class ReynoldsStressOp : public MultiRegions::Operator<TData>
{
public:
    static std::shared_ptr<ReynoldsStressOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "")
    {
        EnsureLinked();
        return MultiRegions::Operator<TData>::template Create<ReynoldsStressOp>(
            expansionList, components, execStr);
    }

    static inline const std::string name = "ReynoldsStress";

    /// Names of the product components, e.g. uu, uv, uw, vv, vw, ww, formed
    /// from the names of the selected velocity components.
    static std::vector<std::string> ProductNames(
        const std::vector<std::string> &velocityNames)
    {
        std::vector<std::string> names;
        for (unsigned int a = 0; a < velocityNames.size(); ++a)
        {
            for (unsigned int b = a; b < velocityNames.size(); ++b)
            {
                names.push_back(velocityNames[a] + velocityNames[b]);
            }
        }
        return names;
    }

    /// @p velocityComponents are the indices of the velocity components in
    /// the velocity field, which may hold further components.
    void Configure(const std::vector<unsigned int> &velocityComponents)
    {
        v_Configure(velocityComponents);
    }

    /// Set the factor applied to the products before they are added.
    void SetScale(const TData scale)
    {
        m_scale = scale;
    }

    /// Add the scaled products of @p velocity to @p products, which leave
    /// at the interleave width of @p velocity.
    void Apply(LibUtilities::Field<TData, FieldState::Phys> &velocity,
               LibUtilities::Field<TData, FieldState::Phys> &products)
    {
        v_Apply(velocity, products);
    }

    /// As Apply().
    void operator()(LibUtilities::Field<TData, FieldState::Phys> &velocity,
                    LibUtilities::Field<TData, FieldState::Phys> &products)
    {
        v_Apply(velocity, products);
    }

protected:
    /// Factor applied to the products before they are added.
    TData m_scale = TData{1};

    ReynoldsStressOp(const MultiRegions::ExpListSharedPtr &expansionList,
                     const std::vector<std::string> &components)
        : MultiRegions::Operator<TData>(expansionList, components)
    {
    }

    ~ReynoldsStressOp() override = default;

    virtual void v_Configure(
        const std::vector<unsigned int> &velocityComponents) = 0;
    virtual void v_Apply(
        LibUtilities::Field<TData, FieldState::Phys> &velocity,
        LibUtilities::Field<TData, FieldState::Phys> &products) = 0;
};

} // namespace Nektar::SolverCore
