///////////////////////////////////////////////////////////////////////////////
//
// File: FilterContext.cpp
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

#include <SolverCore/Filters/FilterContext.h>

#include <algorithm>
#include <tuple>

#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivOp.hpp>

namespace Nektar::SolverCore
{

std::vector<FieldComponent> MakeFieldComponents(
    const std::vector<std::string> &names, const unsigned int first,
    unsigned int count)
{
    ASSERTL0(first <= names.size(), "Field component offset is out of range.");
    if (count == 0)
    {
        count = static_cast<unsigned int>(names.size() - first);
    }
    ASSERTL0(static_cast<size_t>(first) + count <= names.size(),
             "Field component selection is out of range.");

    std::vector<FieldComponent> components;
    components.reserve(count);
    for (unsigned int i = 0; i < count; ++i)
    {
        components.push_back({names[first + i], first + i});
    }
    return components;
}

std::vector<std::string> ComponentNames(
    const std::vector<FieldComponent> &components)
{
    std::vector<std::string> names;
    names.reserve(components.size());
    std::transform(
        components.begin(), components.end(), std::back_inserter(names),
        [](const FieldComponent &component) { return component.name; });
    return names;
}

std::vector<unsigned int> ComponentIndices(
    const std::vector<FieldComponent> &components)
{
    std::vector<unsigned int> indices;
    indices.reserve(components.size());
    std::transform(
        components.begin(), components.end(), std::back_inserter(indices),
        [](const FieldComponent &component) { return component.index; });
    return indices;
}

void FieldRegistry::SetRole(std::string role, PhysicalFieldView view)
{
    ASSERTL0(!role.empty(), "A field role must have a name.");
    m_roles.insert_or_assign(std::move(role), std::move(view));
}

std::optional<PhysicalFieldView> FieldRegistry::GetRole(
    const std::string_view role) const
{
    const auto it = m_roles.find(role);
    return it == m_roles.end() ? std::nullopt
                               : std::optional<PhysicalFieldView>(it->second);
}

void FieldRegistry::SetPhysicalGroups(PhysicalFieldViews groups)
{
    m_physicalGroups = std::move(groups);
}

const PhysicalFieldViews &FieldRegistry::GetPhysicalGroups() const
{
    return m_physicalGroups;
}

void FieldRegistry::SetCoefficientProvider(CoefficientProvider provider)
{
    m_coefficientProvider = std::move(provider);
}

CoefficientFieldViews FieldRegistry::GetCoefficientGroups() const
{
    ASSERTL0(m_coefficientProvider,
             "No coefficient-field provider is registered.");
    return m_coefficientProvider();
}

struct DerivedDataCache::Impl
{
    struct Gradient
    {
        std::shared_ptr<MultiRegions::PhysDerivOp<double>> op;
        LibUtilities::Field<double, FieldState::Phys> field;
        bool current = false;
    };

    using Key = std::tuple<const void *, const void *,
                           std::vector<unsigned int>, std::string>;
    std::map<Key, Gradient> gradients;
};

DerivedDataCache::DerivedDataCache() : m_impl(std::make_unique<Impl>())
{
}

DerivedDataCache::~DerivedDataCache()                            = default;
DerivedDataCache::DerivedDataCache(DerivedDataCache &&) noexcept = default;
DerivedDataCache &DerivedDataCache::operator=(DerivedDataCache &&) noexcept =
    default;

LibUtilities::Field<double, FieldState::Phys> &DerivedDataCache::GetGradient(
    const PhysicalFieldView &view, const std::string &executionSpace)
{
    auto &source = view.field.get();
    const Impl::Key key{std::addressof(source), view.expansion.get(),
                        ComponentIndices(view.components), executionSpace};
    auto it = m_impl->gradients.find(key);
    if (it == m_impl->gradients.end())
    {
        std::vector<std::string> names(source.GetNumComponents());
        for (unsigned int i = 0; i < names.size(); ++i)
        {
            names[i] = "component" + std::to_string(i);
        }
        auto op = MultiRegions::PhysDerivOp<double>::Create(
            view.expansion, names, executionSpace);
        const unsigned int dimension = view.expansion->GetCoordim(0);
        LibUtilities::Field<double, FieldState::Phys> gradient(
            "cached physical gradient",
            MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
                view.expansion),
            source.GetNumComponents() * dimension, source.GetNumHomoModes());
        it = m_impl->gradients
                 .emplace(key,
                          Impl::Gradient{std::move(op), std::move(gradient)})
                 .first;
    }
    if (!it->second.current)
    {
        it->second.op->Apply(source, it->second.field);
        it->second.current = true;
    }
    return it->second.field;
}

/**
 * @brief Mark every cached result out of date, keeping its operators and
 * storage for the next request.
 */
void DerivedDataCache::Invalidate()
{
    for (auto &[key, gradient] : m_impl->gradients)
    {
        static_cast<void>(key);
        gradient.current = false;
    }
}

} // namespace Nektar::SolverCore
