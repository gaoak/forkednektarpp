////////////////////////////////////////////////////////////////////////////////
//
// File: Misc.cpp
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
///
#include <SolverCore/Core/Misc.h>

#include <iomanip>
#include <sstream>

namespace Nektar::SolverCore
{

std::string GetProjectionString(MultiRegions::ProjectionType projectionType)
{
    switch (projectionType)
    {
        case MultiRegions::eGalerkin:
            return "Continuous Galerkin";
        case MultiRegions::eDiscontinuous:
            return "Discontinuous Galerkin";
        case MultiRegions::eMixed_CG_Discontinuous:
            return "Mixed Continuous Galerkin and Discontinuous";
        default:
            return "Unknown";
    }
}

MultiRegions::ProjectionType GetProjectionType(std::string projectionString)
{
    // Transform string to lower case
    std::transform(projectionString.begin(), projectionString.end(),
                   projectionString.begin(),
                   [](unsigned char c) { return std::toupper(c); });

    if (projectionString == "CONTINUOUS" || projectionString == "GALERKIN" ||
        projectionString == "CG")
    {
        return MultiRegions::eGalerkin;
    }
    else if (projectionString == "DG" || projectionString == "DISCONTINUOUS")
    {
        return MultiRegions::eDiscontinuous;
    }
    else if (projectionString == "MIXEDCGDG" ||
             projectionString == "MIXED_CG_DISCONTINUOUS")
    {
        return MultiRegions::eMixed_CG_Discontinuous;
    }
    else
    {
        ASSERTL0(false, "PROJECTION value not recognised");
        return MultiRegions::eGalerkin;
    }
}

} // namespace Nektar::SolverCore
