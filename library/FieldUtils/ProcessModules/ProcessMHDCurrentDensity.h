///////////////////////////////////////////////////////////////////////////////
//
//  File: ProcessMHDCurrentDensity.h
//
//  For more information, please see: http://www.nektar.info/
//
//  The MIT License
//
//  Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
//  Department of Aeronautics, Imperial College London (UK), and Scientific
//  Computing and Imaging Institute, University of Utah (USA).
//
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//
//  Description: Computes quasi-static MHD electric current density.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef FIELDUTILS_PROCESSMHDCURRENTDENSITY
#define FIELDUTILS_PROCESSMHDCURRENTDENSITY

#include "../Module.h"

namespace Nektar::FieldUtils
{

/**
 * @brief Calculate the quasi-static MHD electric current density
 * \f$\boldsymbol{J} = \sigma(\boldsymbol{E}_0 + \boldsymbol{u}\times
 * \boldsymbol{B}_0 - \nabla\phi)\f$ and append Jx, Jy and Jz to the field.
 */
class ProcessMHDCurrentDensity : public ProcessModule
{
public:
    static std::shared_ptr<Module> create(FieldSharedPtr f)
    {
        return MemoryManager<ProcessMHDCurrentDensity>::AllocateSharedPtr(f);
    }

    static ModuleKey className;

    ProcessMHDCurrentDensity(FieldSharedPtr f);
    ~ProcessMHDCurrentDensity() override;

protected:
    void v_Process(po::variables_map &vm) override;

    std::string v_GetModuleName() override
    {
        return "ProcessMHDCurrentDensity";
    }

    std::string v_GetModuleDescription() override
    {
        return "Calculating quasi-static MHD electric current density";
    }

    ModulePriority v_GetModulePriority() override
    {
        return eModifyExp;
    }

private:
    int FindField(const std::string &name) const;
    std::string GetElectromagneticFieldFunction() const;
};

} // namespace Nektar::FieldUtils

#endif
