///////////////////////////////////////////////////////////////////////////////
//
// File: FilterContext.h
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
// Description: Per-phase services and field views for SolverCore filters.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <LibUtilities/BasicUtils/Field/Field.hpp>
#include <LibUtilities/BasicUtils/FieldIO.h>
#include <MultiRegions/ExpList.h>
#include <SolverCore/SolverCoreDeclspec.h>

namespace Nektar::SolverCore
{

namespace FieldRoles
{
inline constexpr std::string_view Solution = "solution";
inline constexpr std::string_view Velocity = "velocity";
inline constexpr std::string_view Pressure = "pressure";
} // namespace FieldRoles

struct SOLVER_CORE_EXPORT FieldComponent
{
    std::string name;
    unsigned int index = 0;
};

template <FieldState State> struct StateFieldView
{
    std::reference_wrapper<LibUtilities::Field<double, State>> field;
    MultiRegions::ExpListSharedPtr expansion;
    unsigned int spaceDimension = 0;
    std::vector<FieldComponent> components;
};

using PhysicalFieldView     = StateFieldView<FieldState::Phys>;
using CoefficientFieldView  = StateFieldView<FieldState::Coeff>;
using PhysicalFieldViews    = std::vector<PhysicalFieldView>;
using CoefficientFieldViews = std::vector<CoefficientFieldView>;

SOLVER_CORE_EXPORT std::vector<FieldComponent> MakeFieldComponents(
    const std::vector<std::string> &names, unsigned int first = 0,
    unsigned int count = 0);
SOLVER_CORE_EXPORT std::vector<std::string> ComponentNames(
    const std::vector<FieldComponent> &components);
SOLVER_CORE_EXPORT std::vector<unsigned int> ComponentIndices(
    const std::vector<FieldComponent> &components);

/**
 * @brief Field roles and canonical, non-overlapping field groups.
 *
 * Physical groups are used by point/history output. Coefficient groups are
 * produced lazily because obtaining output coefficients may require a forward
 * transform.
 */
class SOLVER_CORE_EXPORT FieldRegistry
{
public:
    using CoefficientProvider = std::function<CoefficientFieldViews()>;

    void SetRole(std::string role, PhysicalFieldView view);
    std::optional<PhysicalFieldView> GetRole(std::string_view role) const;

    void SetPhysicalGroups(PhysicalFieldViews groups);
    const PhysicalFieldViews &GetPhysicalGroups() const;

    void SetCoefficientProvider(CoefficientProvider provider);
    CoefficientFieldViews GetCoefficientGroups() const;

private:
    std::map<std::string, PhysicalFieldView, std::less<>> m_roles;
    PhysicalFieldViews m_physicalGroups;
    CoefficientProvider m_coefficientProvider;
};

struct SOLVER_CORE_EXPORT FilterOutputServices
{
    std::function<void(const std::string &)> writeFields;
    std::function<void(const std::string &, const CoefficientFieldViews &,
                       const LibUtilities::FieldMetaDataMap &, double)>
        writeOutputFields;
    std::function<const LibUtilities::FieldMetaDataMap &()> getMetadata;
    std::function<void(const std::string &, const std::string &)> setMetadata;
};

/**
 * @brief Cache of data derived from physical fields, shared by the filters
 * of one phase.
 *
 * The operators and storage of a result are built on its first request and
 * kept; a result is recomputed on its first request after Invalidate().
 */
class SOLVER_CORE_EXPORT DerivedDataCache
{
public:
    DerivedDataCache();
    ~DerivedDataCache();
    DerivedDataCache(DerivedDataCache &&) noexcept;
    DerivedDataCache &operator=(DerivedDataCache &&) noexcept;
    DerivedDataCache(const DerivedDataCache &)            = delete;
    DerivedDataCache &operator=(const DerivedDataCache &) = delete;

    LibUtilities::Field<double, FieldState::Phys> &GetGradient(
        const PhysicalFieldView &view, const std::string &executionSpace);

    void Invalidate();

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

struct SOLVER_CORE_EXPORT FilterContext
{
    double time       = 0.0;
    unsigned int step = 0;
    bool initial      = false;
    bool final        = false;
    FilterOutputServices &output;
    const FieldRegistry &fields;
    DerivedDataCache &derived;
};

} // namespace Nektar::SolverCore
