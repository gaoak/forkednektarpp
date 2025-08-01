<?
#
# model.h
# A pype template for Nektar simulation
#
# Required variables
# ---------------------------
# model    A model
# protocol A pacing protocol
# ---------------------------
#
# This file is derived from Myokit.
# See http://myokit.org for copyright, sharing, and licensing details.
#
# Get model name
model_name = name if name is not None else model.name();
?>
///////////////////////////////////////////////////////////////////////////////
//
// File: <?= model_name ?>.h
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
// Description: <?= model_name ?> cell model.
//              Generated from CellML on <?= myokit.date() ?>
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_SOLVERS_CARDIACEPSOLVER_CELLMODELS_<?= model_name.upper() ?>_H_
#define NEKTAR_SOLVERS_CARDIACEPSOLVER_CELLMODELS_<?= model_name.upper() ?>_H_

#include <CardiacEPSolver/CellModels/CellModel.h>

namespace Nektar
{

class <?= model_name ?> : public CellModel
{

public:
    /// Creates an instance of this class
    static CellModelSharedPtr create(
            const LibUtilities::SessionReaderSharedPtr& pSession,
            const MultiRegions::ExpListSharedPtr& pField)
    {
        return MemoryManager<<?= model_name ?>>::AllocateSharedPtr(pSession, pField);
    }

    /// Name of class
    static std::string className;

    /// Constructor
    <?= model_name ?>(const LibUtilities::SessionReaderSharedPtr& pSession,
        const MultiRegions::ExpListSharedPtr& pField);

    /// Destructor
    ~<?= model_name ?>() override;

protected:
    /// Computes the reaction terms $f(u,v)$ and $g(u,v)$.
    void v_Update(
            const Array<OneD, const Array<OneD, NekDouble> > &inarray,
                  Array<OneD,       Array<OneD, NekDouble> > &outarray,
            const NekDouble                                   time) override;

    /// Prints a summary of the model parameters.
    void v_GenerateSummary(SummaryList& s) override;

    /// Set initial conditions for the cell model
    void v_SetInitialConditions() override;

    /// Returns the name of a variable for a given index
    std::string v_GetCellVarName(size_t idx) override;

private:
    /// Non-state variables
<?
for var in model.variables(state=False, deep=True):
    if var not in tau_vars and var not in inf_vars:
        print(tab + 'NekDouble ' + v(var) + '; // ' + str(var.unit()))
?>
<?
if variants:
    print(tab + 'enum Variants {')
    for name in variants.keys():
        print(2*tab + 'e' + name + ',')
    print(tab + '};')
    print(tab + 'enum Variants model_variant;')
    print('')
    print(tab + 'static std::string lookupIds[];')
    print(tab + 'static std::string def;')
?>};

} // namespace Nektar

#endif
