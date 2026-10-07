///////////////////////////////////////////////////////////////////////////////
//
// File: TestFieldStorage.cpp
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
// Description: Tests of the MemoryRegion and Field move assignments. Whether
// the storage an object held before the assignment is freed cannot be observed
// from here; these pin down what the assignment must still do once it frees
// it.
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/Field/Field.hpp>
#include <LibUtilities/BasicUtils/Field/MemoryRegion.hpp>

#include <boost/test/unit_test.hpp>

namespace Nektar::FieldStorageUnitTests
{
using LibUtilities::BlockAttributes;
using LibUtilities::MemoryRegion;
using NektarSpaces::HostSpace;
using PhysField = LibUtilities::Field<double, FieldState::Phys>;

/// Move assignment onto a region that already holds storage takes over the
/// right-hand side's storage and leaves the right-hand side empty.
BOOST_AUTO_TEST_CASE(TestMoveAssignOntoAllocatedRegion)
{
    MemoryRegion<double> lhs(8);
    lhs.Initialize<HostSpace>(1.0);
    MemoryRegion<double> rhs(16);
    rhs.Initialize<HostSpace>(2.0);
    const double *rhsPtr = rhs.GetPtr<HostSpace, ReadOnly>();

    lhs = std::move(rhs);

    BOOST_CHECK_EQUAL(lhs.size(), 16);
    BOOST_CHECK_EQUAL(rhs.size(), 0);
    const double *lhsPtr = lhs.GetPtr<HostSpace, ReadOnly>();
    BOOST_CHECK_EQUAL(lhsPtr, rhsPtr);
    for (size_t i = 0; i < lhs.size(); ++i)
    {
        BOOST_CHECK_EQUAL(lhsPtr[i], 2.0);
    }
}

/// Move assignment of a region to itself keeps its storage and values; freeing
/// the old storage first must not free the storage being assigned.
BOOST_AUTO_TEST_CASE(TestSelfMoveAssign)
{
    MemoryRegion<double> region(8);
    region.Initialize<HostSpace>(3.0);
    const double *before = region.GetPtr<HostSpace, ReadOnly>();

    MemoryRegion<double> &alias = region;
    region                      = std::move(alias);

    BOOST_CHECK_EQUAL(region.size(), 8);
    const double *after = region.GetPtr<HostSpace, ReadOnly>();
    BOOST_CHECK_EQUAL(after, before);
    for (size_t i = 0; i < region.size(); ++i)
    {
        BOOST_CHECK_EQUAL(after[i], 3.0);
    }
}

/// A region emptied by being moved from can be assigned again.
BOOST_AUTO_TEST_CASE(TestMoveAssignOntoMovedFromRegion)
{
    MemoryRegion<double> a(4);
    a.Initialize<HostSpace>(5.0);
    MemoryRegion<double> b(std::move(a));

    MemoryRegion<double> c(6);
    c.Initialize<HostSpace>(7.0);
    a = std::move(c);

    BOOST_CHECK_EQUAL(a.size(), 6);
    BOOST_CHECK_EQUAL(b.size(), 4);
    const double *aPtr = a.GetPtr<HostSpace, ReadOnly>();
    const double *bPtr = b.GetPtr<HostSpace, ReadOnly>();
    BOOST_CHECK_EQUAL(aPtr[5], 7.0);
    BOOST_CHECK_EQUAL(bPtr[3], 5.0);
}

/// A field of @p nelmt elements with @p ndata points each, every value set
/// to @p val; Initialize() allocates the field's contiguous storage.
static PhysField MakeField(const size_t nelmt, const unsigned int ndata,
                           const unsigned int ncomp, const double val)
{
    std::vector<BlockAttributes<FieldState::Phys>> attr{
        BlockAttributes<FieldState::Phys>(nelmt, nelmt, ndata, 1)};
    PhysField field(attr, ncomp, 1);
    field.Initialize<HostSpace>(val);
    return field;
}

/// Move assignment onto a field that already holds contiguous storage takes
/// over the right-hand side's storage and blocks.
BOOST_AUTO_TEST_CASE(TestFieldMoveAssignOntoAllocatedField)
{
    PhysField lhs        = MakeField(4, 8, 2, 1.0);
    PhysField rhs        = MakeField(6, 8, 3, 2.0);
    const double *rhsPtr = rhs.GetBlocks()[0].GetPtr<HostSpace, ReadOnly>();

    lhs = std::move(rhs);

    BOOST_CHECK_EQUAL(lhs.GetNumComponents(), 3);
    BOOST_CHECK_EQUAL(lhs.size(), 6 * 8 * 3);
    const double *lhsPtr = lhs.GetBlocks()[0].GetPtr<HostSpace, ReadOnly>();
    BOOST_CHECK_EQUAL(lhsPtr, rhsPtr);
    for (size_t i = 0; i < lhs.size(); ++i)
    {
        BOOST_CHECK_EQUAL(lhsPtr[i], 2.0);
    }
}

/// Move assignment of a field to itself keeps its storage and values.
BOOST_AUTO_TEST_CASE(TestFieldSelfMoveAssign)
{
    PhysField field      = MakeField(4, 8, 2, 3.0);
    const double *before = field.GetBlocks()[0].GetPtr<HostSpace, ReadOnly>();

    PhysField &alias = field;
    field            = std::move(alias);

    BOOST_CHECK_EQUAL(field.size(), 4 * 8 * 2);
    const double *after = field.GetBlocks()[0].GetPtr<HostSpace, ReadOnly>();
    BOOST_CHECK_EQUAL(after, before);
    for (size_t i = 0; i < field.size(); ++i)
    {
        BOOST_CHECK_EQUAL(after[i], 3.0);
    }
}

} // namespace Nektar::FieldStorageUnitTests
