///////////////////////////////////////////////////////////////////////////////
//
// File: TestStatisticsFilters.cpp
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
// Description: Unit tests of the values written by the AverageFields and
// ReynoldsStresses filters.
//
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_MODULE TestStatisticsFilters

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <LibUtilities/BasicUtils/FieldIO.h>
#include <LibUtilities/BasicUtils/Filesystem.hpp>
#include <MultiRegions/ElmtOps/BwdTrans/BwdTransOp.hpp>
#include <MultiRegions/ElmtOps/IProductWRTBase/IProductWRTBaseOp.hpp>
#include <MultiRegions/ElmtOps/MultiplyByElmtInvMass/MultiplyByElmtInvMassOp.hpp>
#include <MultiRegions/ExpList.h>
#include <SpatialDomains/MeshGraphIO.h>

#include <LibUtilities/BasicUtils/Field/Field.hpp>

#include <SolverCore/Filters/Filter.h>
#include <SolverCore/Filters/FilterContext.h>

#include <UnitTests/TestBoostSetup.hpp>
#include <UnitTests/TestGlobalConfiguration.hpp>

using namespace Nektar;
using namespace Nektar::SolverCore;

NEKTAR_TEST_GLOBAL_CONFIGURATION(Nektar::UnitTests::TestArgs::None);

#include <UnitTests/TestBoostTeardown.hpp>

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
namespace
{

// Time step of the steps the filters are applied at.
constexpr double timeStep = 0.25;

// The meshes: straight-sided triangles and quadrilaterals in 2D, and in 3D
// four boundary-layer prisms on a curved pipe wall with six straight-sided
// tetrahedra next to them.
const std::string triQuadMesh        = "run/square_all_elements.xml";
const std::string curvedPrismTetMesh = "run/pipe_prismtet_curved.xml";

// Tolerance, relative to the largest reference value, of a check that holds
// to rounding: the analytic field lies in the expansion space, or the check
// is the defining property of the element-local projection. The rounding is
// that of the element mass matrices, of condition number below 1e4 at these
// orders, times the cancellation of the two-moment covariance, below ten for
// these amplitudes.
constexpr double exactTolerance = 1.0e-10;

// Bound, relative to the largest analytic value, on the projection error of
// the covariances on the curved prisms; see CheckStresses(). The error is
// about 1e-5 at most on these prisms, against values of order one for a
// wrong projection.
constexpr double curvedProjectionTolerance = 1.0e-4;

// Amplitude of velocity component @p a at sample n.
double Amplitude(const unsigned int a, const unsigned int n)
{
    switch (a)
    {
        case 0:
            return 1.0 + n;
        case 1:
            return 2.0 - 0.25 * n * n;
        default:
            return -0.5 + 0.75 * n - 0.1 * n * n;
    }
}

// Mean of the amplitudes of component @p a, and the population covariance of
// those of components @p a and @p b, over @p samples.
double Mean(const unsigned int a, const std::vector<unsigned int> &samples)
{
    double mean = 0.0;
    for (const unsigned int n : samples)
    {
        mean += Amplitude(a, n);
    }
    return mean / samples.size();
}

double Covariance(const unsigned int a, const unsigned int b,
                  const std::vector<unsigned int> &samples)
{
    const double meanA = Mean(a, samples), meanB = Mean(b, samples);
    double covariance = 0.0;
    for (const unsigned int n : samples)
    {
        covariance += (Amplitude(a, n) - meanA) * (Amplitude(b, n) - meanB);
    }
    return covariance / samples.size();
}

// The velocity functions f_a and the steady scalar, linear in the
// coordinates: they lie in the expansion space of every element of both
// meshes, the curved prisms included, whose geometry is of the order of
// their expansion. Their products, quadratic, lie in the space of the
// straight-sided elements only.
using Function = std::function<double(double, double, double)>;
const std::vector<Function> functions2D = {
    [](double x, double y, double) { return 1.0 + x - 0.5 * y; },
    [](double x, double y, double) { return 0.5 - x + 2.0 * y; },
    [](double x, double y, double) { return 0.3 + x - y; }};
const std::vector<Function> functions3D = {
    [](double x, double, double z) { return 1.0 + x - 0.5 * z; },
    [](double x, double y, double z) { return 0.5 - 2.0 * y + z + 0.2 * x; },
    [](double x, double y, double z) { return 0.2 + 0.5 * x + y - z; },
    [](double x, double y, double z) { return 0.3 + x - 2.0 * y + z; }};

LibUtilities::SessionReaderSharedPtr SetSession(const std::string &mesh,
                                                const std::string &execStr)
{
    const std::vector<std::string> args = {
        "TestStatisticsFilters", mesh, "--opExecSpace=" + execStr,
        "--force-output",        "-P", "TimeStep=" + std::to_string(timeStep)};
    int argc    = static_cast<int>(args.size());
    char **argv = new char *[argc];
    for (int i = 0; i < argc; ++i)
    {
        argv[i] = strdup(args[i].c_str());
    }

    auto session = LibUtilities::SessionReader::CreateInstance(argc, argv);

    for (int i = 0; i < argc; ++i)
    {
        free(argv[i]);
    }
    delete[] argv;

    return session;
}

// The scaled coefficients of every written variable, in the expansion's
// coefficient order, and the metadata of one written file.
struct WrittenFile
{
    std::map<std::string, std::vector<double>> variables;
    LibUtilities::FieldMetaDataMap metadata;
};

// Largest absolute value of @p values.
double MaxAbs(const Array<OneD, double> &values)
{
    double maxAbs = 0.0;
    for (size_t i = 0; i < values.size(); ++i)
    {
        maxAbs = (std::max)(maxAbs, std::abs(values[i]));
    }
    return maxAbs;
}

// Scale of a check of values of order one: their largest absolute value,
// and one at least.
double Scale(const Array<OneD, double> &values)
{
    return (std::max)(1.0, MaxAbs(values));
}

/**
 * Drives a filter through the phases of a run, with the velocity published
 * as the velocity role and, with a steady scalar, as two coefficient groups.
 * Sample n holds the velocity components u_a = Amplitude(a, n) f_a, with
 * analytic f_a and scalar. Their values at the quadrature points are the
 * analytic values, and their coefficients the element-local L2 projections,
 * formed with the IProductWRTBase and MultiplyByElmtInvMass operators; the
 * two agree, since the functions lie in the expansion space. The output
 * services record every file a filter writes and write it to disk, so that a
 * later filter can restart from it; the files are removed with the run.
 */
class FilterRun
{
public:
    FilterRun(const std::string &execStr, const std::string &mesh)
        : m_execStr(execStr), m_session(SetSession(mesh, execStr))
    {
        auto graph  = SpatialDomains::MeshGraphIO::Read(m_session);
        m_expansion = MemoryManager<MultiRegions::ExpList>::AllocateSharedPtr(
            m_session, graph, true, "u", Collections::eNoCollection);
        m_expansion->SetDataWarehouse();
        m_dimension = m_expansion->GetCoordim(0);

        const std::vector<std::string> allNames = {"u", "v", "w"};
        m_velocityNames.assign(allNames.begin(),
                               allNames.begin() + m_dimension);
        m_scalarName = m_dimension == 3 ? "p" : "s";

        m_coeffBlocks =
            MultiRegions::GetBlockAttributes<double, FieldState::Coeff>(
                m_expansion);
        m_physBlocks =
            MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
                m_expansion);
        m_bwdTransOp =
            MultiRegions::BwdTransOp<double>::Create(m_expansion, {"q"});
        m_iProductWRTBaseOp =
            MultiRegions::IProductWRTBaseOp<double>::Create(m_expansion, {"q"});
        m_multiplyByElmtInvMassOp =
            MultiRegions::MultiplyByElmtInvMassOp<double>::Create(m_expansion,
                                                                  {"q"});

        const size_t nPoints = m_expansion->GetTotPoints();
        Array<OneD, double> x(nPoints, 0.0), y(nPoints, 0.0), z(nPoints, 0.0);
        m_expansion->GetCoords(x, y, z);
        for (const auto &function :
             m_dimension == 3 ? functions3D : functions2D)
        {
            Array<OneD, double> phys(nPoints);
            for (size_t i = 0; i < nPoints; ++i)
            {
                phys[i] = function(x[i], y[i], z[i]);
            }
            m_phys.push_back(phys);
            m_coeffs.push_back(Project(phys));
        }

        m_velocityCoeffs = LibUtilities::Field<double, FieldState::Coeff>(
            "velocity coefficients", m_coeffBlocks, m_dimension, 1);
        m_scalarCoeffs = LibUtilities::Field<double, FieldState::Coeff>(
            "scalar coefficients", m_coeffBlocks, 1, 1);
        m_velocity = LibUtilities::Field<double, FieldState::Phys>(
            "velocity", m_physBlocks, m_dimension, 1);

        const auto velocityComponents = MakeFieldComponents(m_velocityNames);
        m_registry.SetRole(std::string(FieldRoles::Velocity),
                           {std::ref(m_velocity), m_expansion, m_dimension,
                            velocityComponents});
        m_registry.SetCoefficientProvider([this, velocityComponents]() {
            return CoefficientFieldViews{
                {std::ref(m_velocityCoeffs), m_expansion, m_dimension,
                 velocityComponents},
                {std::ref(m_scalarCoeffs), m_expansion, m_dimension,
                 MakeFieldComponents({m_scalarName})}};
        });

        m_output.writeFields = [](const std::string &) {};
        m_output.writeOutputFields =
            [this](const std::string &path, const CoefficientFieldViews &views,
                   const LibUtilities::FieldMetaDataMap &metadata,
                   const double scale) { Write(path, views, metadata, scale); };
        m_output.getMetadata =
            [this]() -> const LibUtilities::FieldMetaDataMap & {
            return m_metadata;
        };
        m_output.setMetadata = [this](const std::string &key,
                                      const std::string &value) {
            m_metadata[key] = value;
        };
    }

    // Remove the files the filters wrote.
    ~FilterRun()
    {
        for (const auto &file : m_written)
        {
            fs::remove(file.first);
        }
    }

    // A file name of this test and execution space, so that the tests of the
    // execution spaces can run concurrently in one directory.
    std::string FileName(const std::string &name) const
    {
        return "statistics_filters_" + m_execStr + "_" + name;
    }

    FilterSharedPtr CreateFilter(const std::string &type,
                                 const Filter::ParamMap &params)
    {
        return GetFilterFactory().CreateInstance(type, m_session, params);
    }

    // Initialise @p filter at step @p first - 1, apply it at steps first to
    // last and finalise it. Odd steps hold the fields at interleave width
    // one, even steps at the width of the execution space, so that the
    // accumulators and the products meet sources at both widths.
    void Run(Filter &filter, const unsigned int first, const unsigned int last)
    {
        SetSample(first - 1);
        Context(first - 1, true, false,
                [&](FilterContext &context) { filter.Initialise(context); });
        for (unsigned int step = first; step <= last; ++step)
        {
            SetSample(step);
            Context(step, false, false,
                    [&](FilterContext &context) { filter.Apply(context); });
        }
        Context(last, false, true,
                [&](FilterContext &context) { filter.Finalise(context); });
    }

    const std::map<std::string, WrittenFile> &Written() const
    {
        return m_written;
    }

    // Check that the sample functions lie in the expansion space: the
    // backward transform of their projections recovers their values at the
    // quadrature points.
    void CheckRepresentable() const
    {
        for (size_t k = 0; k < m_phys.size(); ++k)
        {
            BOOST_TEST_CONTEXT("function=" << k)
            {
                CheckPoints(BwdTrans(m_coeffs[k]), m_phys[k], false);
            }
        }
    }

    // Check that the mesh holds curved, deformed prisms and straight-sided
    // tetrahedra.
    void CheckCurvedPrismTetMesh() const
    {
        unsigned int curvedPrisms = 0, straightTets = 0;
        for (size_t e = 0; e < m_expansion->GetExpSize(); ++e)
        {
            const auto exp = m_expansion->GetExp(e);
            if (exp->DetShapeType() == LibUtilities::ePrism)
            {
                curvedPrisms +=
                    !IsStraightSided(e) && exp->GetGeomFactors()->GetGtype() ==
                                               SpatialDomains::eDeformed;
            }
            straightTets += exp->DetShapeType() == LibUtilities::eTetrahedron &&
                            IsStraightSided(e);
        }
        BOOST_REQUIRE_GT(curvedPrisms, 0u);
        BOOST_REQUIRE_GT(straightTets, 0u);
    }

    // Check that @p file holds the averages over @p samples: mean(A_a) f_a
    // for the velocity and the scalar itself. Both lie in the expansion
    // space, so the written fields equal them at every quadrature point.
    void CheckAverages(const WrittenFile &file,
                       const std::vector<unsigned int> &samples) const
    {
        for (unsigned int a = 0; a <= m_dimension; ++a)
        {
            const std::string name =
                a < m_dimension ? m_velocityNames[a] : m_scalarName;
            const double mean = a < m_dimension ? Mean(a, samples) : 1.0;
            BOOST_TEST_CONTEXT("variable=" << name)
            {
                Array<OneD, double> expected(m_phys[a].size());
                for (size_t i = 0; i < expected.size(); ++i)
                {
                    expected[i] = mean * m_phys[a][i];
                }
                CheckPoints(BwdTrans(Variable(file, name)), expected, false);
            }
        }
        BOOST_CHECK_EQUAL(file.metadata.at("NumberOfFieldDumps"),
                          std::to_string(samples.size()));
    }

    // Check that @p file holds the covariances of the velocity over
    // @p samples against g = cov(A_a, A_b) f_a f_b at the quadrature points:
    // - the written field r is the element-local L2 projection of g, so
    //   IProductWRTBase(r - g) vanishes on every element;
    // - on the straight-sided elements g lies in the expansion space, so r
    //   equals g at every quadrature point;
    // - on the curved prisms g does not, and r is within the projection
    //   error of g, which is small: f_a f_b is quadratic in the
    //   coordinates, and the mapping departs from an affine one only by the
    //   wall curvature.
    void CheckStresses(const WrittenFile &file,
                       const std::vector<unsigned int> &samples) const
    {
        for (unsigned int a = 0; a < m_dimension; ++a)
        {
            for (unsigned int b = a; b < m_dimension; ++b)
            {
                const std::string name =
                    m_velocityNames[a] + m_velocityNames[b];
                BOOST_TEST_CONTEXT("variable=" << name)
                {
                    const double covariance = Covariance(a, b, samples);
                    Array<OneD, double> expected(m_phys[a].size());
                    for (size_t i = 0; i < expected.size(); ++i)
                    {
                        expected[i] = covariance * m_phys[a][i] * m_phys[b][i];
                    }
                    const auto written = BwdTrans(Variable(file, name));

                    Array<OneD, double> residual(written.size());
                    for (size_t i = 0; i < residual.size(); ++i)
                    {
                        residual[i] = written[i] - expected[i];
                    }
                    const auto innerProduct = IProductWRTBase(residual);
                    const double scale      = MaxAbs(IProductWRTBase(expected));
                    for (size_t i = 0; i < innerProduct.size(); ++i)
                    {
                        BOOST_CHECK_SMALL(innerProduct[i],
                                          exactTolerance * scale);
                    }

                    CheckPoints(written, expected, true);
                }
            }
        }
    }

private:
    std::string m_execStr;
    LibUtilities::SessionReaderSharedPtr m_session;
    MultiRegions::ExpListSharedPtr m_expansion;
    unsigned int m_dimension = 0;
    std::vector<std::string> m_velocityNames;
    std::string m_scalarName;
    std::vector<LibUtilities::BlockAttributes<FieldState::Coeff>> m_coeffBlocks;
    std::vector<LibUtilities::BlockAttributes<FieldState::Phys>> m_physBlocks;
    std::shared_ptr<MultiRegions::BwdTransOp<double>> m_bwdTransOp;
    std::shared_ptr<MultiRegions::IProductWRTBaseOp<double>>
        m_iProductWRTBaseOp;
    std::shared_ptr<MultiRegions::MultiplyByElmtInvMassOp<double>>
        m_multiplyByElmtInvMassOp;
    /// Values at the quadrature points of each f_a and of the scalar, and
    /// their coefficients.
    std::vector<Array<OneD, double>> m_phys;
    std::vector<Array<OneD, double>> m_coeffs;
    LibUtilities::Field<double, FieldState::Coeff> m_velocityCoeffs;
    LibUtilities::Field<double, FieldState::Coeff> m_scalarCoeffs;
    LibUtilities::Field<double, FieldState::Phys> m_velocity;
    FieldRegistry m_registry;
    FilterOutputServices m_output;
    DerivedDataCache m_derived;
    LibUtilities::FieldMetaDataMap m_metadata;
    std::map<std::string, WrittenFile> m_written;

    void Context(const unsigned int step, const bool initial, const bool final,
                 const std::function<void(FilterContext &)> &phase)
    {
        FilterContext context{step * timeStep, step,       initial,  final,
                              m_output,        m_registry, m_derived};
        phase(context);
    }

    // Fill the fields with sample @p n.
    void SetSample(const unsigned int n)
    {
        const unsigned int interleaveWidth =
            n % 2 == 0 ? NektarSpaces::GetVectorWidth<double>(m_execStr) : 1u;
        const size_t nCoeffs = m_expansion->GetNcoeffs();
        const size_t nPoints = m_expansion->GetTotPoints();

        Array<OneD, double> coeffs(m_dimension * nCoeffs),
            phys(m_dimension * nPoints);
        for (unsigned int a = 0; a < m_dimension; ++a)
        {
            const double amplitude = Amplitude(a, n);
            for (size_t i = 0; i < nCoeffs; ++i)
            {
                coeffs[a * nCoeffs + i] = amplitude * m_coeffs[a][i];
            }
            for (size_t i = 0; i < nPoints; ++i)
            {
                phys[a * nPoints + i] = amplitude * m_phys[a][i];
            }
        }
        m_velocityCoeffs.SetInterleaveWidth(interleaveWidth);
        m_velocityCoeffs.CopyArray<NektarSpaces::HostSpace>(coeffs);
        m_scalarCoeffs.SetInterleaveWidth(interleaveWidth);
        m_scalarCoeffs.CopyArray<NektarSpaces::HostSpace>(
            m_coeffs[m_dimension]);
        m_velocity.SetInterleaveWidth(interleaveWidth);
        m_velocity.CopyArray<NektarSpaces::HostSpace>(phys);
    }

    // Record the scaled coefficients of every view and write them to @p path.
    void Write(const std::string &path, const CoefficientFieldViews &views,
               const LibUtilities::FieldMetaDataMap &metadata,
               const double scale)
    {
        std::vector<LibUtilities::FieldDefinitionsSharedPtr> fieldDefs;
        std::vector<std::vector<double>> fieldData;
        WrittenFile &file = m_written[path];
        file.metadata     = metadata;
        for (const auto &view : views)
        {
            const auto coeffs    = view.field.get().ToVector<double>();
            const auto names     = ComponentNames(view.components);
            const size_t nCoeffs = coeffs.size() / names.size();
            auto viewDefs        = view.expansion->GetFieldDefinitions();
            std::vector<std::vector<double>> viewData(viewDefs.size());
            for (size_t c = 0; c < names.size(); ++c)
            {
                std::vector<double> component(nCoeffs);
                for (size_t i = 0; i < nCoeffs; ++i)
                {
                    component[i] = scale * coeffs[c * nCoeffs + i];
                }
                for (size_t d = 0; d < viewDefs.size(); ++d)
                {
                    viewDefs[d]->m_fields.push_back(names[c]);
                    view.expansion->AppendFieldData(viewDefs[d], viewData[d],
                                                    component);
                }
                file.variables[names[c]] = component;
            }
            fieldDefs.insert(fieldDefs.end(), viewDefs.begin(), viewDefs.end());
            fieldData.insert(fieldData.end(), viewData.begin(), viewData.end());
        }
        LibUtilities::FieldIO::CreateDefault(m_session)->Write(
            path, fieldDefs, fieldData, metadata);
    }

    // The coefficients of variable @p name of @p file.
    static Array<OneD, double> Variable(const WrittenFile &file,
                                        const std::string &name)
    {
        const auto found = file.variables.find(name);
        BOOST_REQUIRE(found != file.variables.end());
        return Array<OneD, double>(found->second.size(), found->second.data());
    }

    // Values at the quadrature points of the coefficients @p coeffs.
    Array<OneD, double> BwdTrans(const Array<OneD, double> &coeffs) const
    {
        LibUtilities::Field<double, FieldState::Coeff> in("coefficients",
                                                          m_coeffBlocks, 1, 1);
        in.CopyArray<NektarSpaces::HostSpace>(coeffs);
        LibUtilities::Field<double, FieldState::Phys> out("values",
                                                          m_physBlocks, 1, 1);
        (*m_bwdTransOp)(in, out);
        return out.ToArray<double>();
    }

    // Inner products with the basis of the values @p phys.
    Array<OneD, double> IProductWRTBase(const Array<OneD, double> &phys) const
    {
        LibUtilities::Field<double, FieldState::Phys> in("values", m_physBlocks,
                                                         1, 1);
        in.CopyArray<NektarSpaces::HostSpace>(phys);
        LibUtilities::Field<double, FieldState::Coeff> out("inner products",
                                                           m_coeffBlocks, 1, 1);
        (*m_iProductWRTBaseOp)(in, out);
        return out.ToArray<double>();
    }

    // Element-local L2 projection of the values @p phys.
    Array<OneD, double> Project(const Array<OneD, double> &phys) const
    {
        LibUtilities::Field<double, FieldState::Phys> in("values", m_physBlocks,
                                                         1, 1);
        in.CopyArray<NektarSpaces::HostSpace>(phys);
        LibUtilities::Field<double, FieldState::Coeff> innerProduct(
            "inner products", m_coeffBlocks, 1, 1);
        (*m_iProductWRTBaseOp)(in, innerProduct);
        LibUtilities::Field<double, FieldState::Coeff> out("coefficients",
                                                           m_coeffBlocks, 1, 1);
        (*m_multiplyByElmtInvMassOp)(innerProduct, out);
        return out.ToArray<double>();
    }

    // Whether the geometry of element @p e is linear in every direction.
    bool IsStraightSided(const size_t e) const
    {
        const auto *xmap = m_expansion->GetExp(e)->GetGeom()->GetXmap();
        for (int dir = 0; dir < xmap->GetShapeDimension(); ++dir)
        {
            if (xmap->GetBasisNumModes(dir) != 2)
            {
                return false;
            }
        }
        return true;
    }

    // Check that @p values equal @p expected at the quadrature points, to
    // rounding, of every element or, if @p straightSidedOnly, of the
    // straight-sided elements, and on the others to the projection error
    // bound; both relative to the largest expected value.
    void CheckPoints(const Array<OneD, double> &values,
                     const Array<OneD, double> &expected,
                     const bool straightSidedOnly) const
    {
        BOOST_REQUIRE_EQUAL(values.size(), expected.size());
        const double scale    = Scale(expected);
        double curvedMaxError = 0.0;
        for (size_t e = 0; e < m_expansion->GetExpSize(); ++e)
        {
            const bool exact     = !straightSidedOnly || IsStraightSided(e);
            const size_t offset  = m_expansion->GetPhys_Offset(e);
            const size_t nPoints = m_expansion->GetExp(e)->GetTotPoints();
            for (size_t i = offset; i < offset + nPoints; ++i)
            {
                const double error = values[i] - expected[i];
                if (exact)
                {
                    BOOST_CHECK_SMALL(error, exactTolerance * scale);
                }
                else
                {
                    curvedMaxError =
                        (std::max)(curvedMaxError, std::abs(error) / scale);
                }
            }
        }
        if (straightSidedOnly)
        {
            BOOST_TEST_MESSAGE("relative projection error on curved "
                               "elements: "
                               << curvedMaxError);
            BOOST_CHECK_LE(curvedMaxError, curvedProjectionTolerance);
        }
    }
};

std::string ExecSpace()
{
    return boost::unit_test::framework::master_test_suite().argv[1];
}

// The file names a run wrote, without directories.
std::set<std::string> WrittenNames(const FilterRun &run)
{
    std::set<std::string> names;
    for (const auto &file : run.Written())
    {
        names.insert(fs::path(file.first).filename().string());
    }
    return names;
}

const WrittenFile &GetFile(const FilterRun &run, const std::string &name)
{
    for (const auto &file : run.Written())
    {
        if (fs::path(file.first).filename().string() == name)
        {
            return file.second;
        }
    }
    BOOST_FAIL("No file " + name + " was written.");
    return run.Written().begin()->second;
}

// Samples at steps 1 to 4 and output every two steps: the numbered files
// hold the averages of samples 1-2 and 1-4, and the final file that of 1-4.
void CheckAverageValues(const std::string &mesh)
{
    FilterRun run(ExecSpace(), mesh);
    run.CheckRepresentable();
    const std::string base = run.FileName("average");
    auto filter            = run.CreateFilter(
        "AverageFields", {{"OutputFile", base}, {"OutputFrequency", "2"}});
    run.Run(*filter, 1, 4);

    BOOST_CHECK(WrittenNames(run) ==
                (std::set<std::string>{base + "_0_avg.fld", base + "_1_avg.fld",
                                       base + "_avg.fld"}));
    run.CheckAverages(GetFile(run, base + "_0_avg.fld"), {1, 2});
    run.CheckAverages(GetFile(run, base + "_1_avg.fld"), {1, 2, 3, 4});
    run.CheckAverages(GetFile(run, base + "_avg.fld"), {1, 2, 3, 4});
}

// A run of samples 1 to 4 writes its average, and a second filter restarting
// from that file adds samples 5 and 6: its average is that of samples 1 to 6
// and keeps the initial time of the first run.
void CheckAverageRestart(const std::string &mesh)
{
    FilterRun run(ExecSpace(), mesh);
    const std::string first  = run.FileName("average_first");
    const std::string second = run.FileName("average_second");
    auto firstFilter         = run.CreateFilter(
        "AverageFields", {{"OutputFile", first}, {"OutputFrequency", "100"}});
    run.Run(*firstFilter, 1, 4);
    auto secondFilter = run.CreateFilter("AverageFields",
                                         {{"OutputFile", second},
                                          {"OutputFrequency", "100"},
                                          {"RestartFile", first + "_avg.fld"}});
    run.Run(*secondFilter, 5, 6);

    const std::string restartedName = second + "_avg.fld";
    const auto &restarted           = GetFile(run, restartedName);
    run.CheckAverages(restarted, {1, 2, 3, 4, 5, 6});
    BOOST_CHECK_EQUAL(restarted.metadata.at("InitialTime"),
                      std::to_string(0.0));
}

// Samples at steps 1 to 4 and output every two steps: each file holds the
// averages of the velocity and the scalar followed by the covariances of the
// velocity.
void CheckStressValues(const std::string &mesh)
{
    FilterRun run(ExecSpace(), mesh);
    run.CheckRepresentable();
    const std::string base = run.FileName("stresses");
    auto filter            = run.CreateFilter(
        "ReynoldsStresses", {{"OutputFile", base}, {"OutputFrequency", "2"}});
    run.Run(*filter, 1, 4);

    BOOST_CHECK(
        WrittenNames(run) ==
        (std::set<std::string>{base + "_0_stress.fld", base + "_1_stress.fld",
                               base + "_stress.fld"}));
    for (const auto &[name, samples] :
         std::vector<std::pair<std::string, std::vector<unsigned int>>>{
             {base + "_0_stress.fld", {1, 2}},
             {base + "_stress.fld", {1, 2, 3, 4}}})
    {
        BOOST_TEST_CONTEXT("file=" << name)
        {
            run.CheckAverages(GetFile(run, name), samples);
            run.CheckStresses(GetFile(run, name), samples);
        }
    }
}

} // namespace
#endif

BOOST_AUTO_TEST_SUITE(TestSuiteStatisticsFilters)

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
BOOST_AUTO_TEST_CASE(AverageFields_Values)
{
    CheckAverageValues(triQuadMesh);
}

// Samples at steps 2, 4 and 6, and output steps 3 and 6: only step 6 is due
// for both, so one numbered file is written.
BOOST_AUTO_TEST_CASE(AverageFields_OutputOnSampledSteps)
{
    FilterRun run(ExecSpace(), triQuadMesh);
    const std::string base = run.FileName("average_sampled");
    auto filter = run.CreateFilter("AverageFields", {{"OutputFile", base},
                                                     {"OutputFrequency", "3"},
                                                     {"SampleFrequency", "2"}});
    run.Run(*filter, 1, 6);

    BOOST_CHECK(
        WrittenNames(run) ==
        (std::set<std::string>{base + "_0_avg.fld", base + "_avg.fld"}));
    run.CheckAverages(GetFile(run, base + "_0_avg.fld"), {2, 4, 6});
    run.CheckAverages(GetFile(run, base + "_avg.fld"), {2, 4, 6});
}

// Phase 0.5 of period 1 at time step 0.25 samples the times 0.5 and 1.5, at
// steps 2 and 6. Step 8 is due for sampling and output and writes the
// numbered file, although it adds no sample.
BOOST_AUTO_TEST_CASE(AverageFields_PhaseAverage)
{
    FilterRun run(ExecSpace(), triQuadMesh);
    const std::string base = run.FileName("average_phase");
    auto filter =
        run.CreateFilter("AverageFields", {{"OutputFile", base},
                                           {"OutputFrequency", "8"},
                                           {"PhaseAverage", "true"},
                                           {"PhaseAveragePeriod", "1.0"},
                                           {"PhaseAveragePhase", "0.5"}});
    run.Run(*filter, 1, 8);

    BOOST_CHECK(
        WrittenNames(run) ==
        (std::set<std::string>{base + "_0_avg.fld", base + "_avg.fld"}));
    run.CheckAverages(GetFile(run, base + "_0_avg.fld"), {2, 6});
    run.CheckAverages(GetFile(run, base + "_avg.fld"), {2, 6});
}

BOOST_AUTO_TEST_CASE(AverageFields_Restart)
{
    CheckAverageRestart(triQuadMesh);
}

BOOST_AUTO_TEST_CASE(ReynoldsStresses_Values)
{
    CheckStressValues(triQuadMesh);
}

BOOST_AUTO_TEST_CASE(CurvedPrismTet_Mesh)
{
    FilterRun run(ExecSpace(), curvedPrismTetMesh);
    run.CheckCurvedPrismTetMesh();
}

BOOST_AUTO_TEST_CASE(AverageFields_CurvedPrismTet_Values)
{
    CheckAverageValues(curvedPrismTetMesh);
}

BOOST_AUTO_TEST_CASE(AverageFields_CurvedPrismTet_Restart)
{
    CheckAverageRestart(curvedPrismTetMesh);
}

BOOST_AUTO_TEST_CASE(ReynoldsStresses_CurvedPrismTet_Values)
{
    CheckStressValues(curvedPrismTetMesh);
}
#endif

BOOST_AUTO_TEST_SUITE_END()
