<?
#
# model.cpp
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
import myokit
import myokit.lib.guess as guess
import myokit.lib.hh as hh
import nektar

# Get model
model.reserve_unique_names(*nektar.keywords)
model.create_unique_names()

# Find transmembrane potential
vm = guess.membrane_potential(model)

# Set Vm to first index by swapping if necessary
indices = {var.qname(): var.indice() for var in model.states()}
if indices[vm.qname()] != 0:
    for var, i in indices.items():
        if i == 0:
            indices[var] = vm.indice()
            indices[vm.qname()] = 0
            break

# Find any gating variables for Rush-Larsen integration
gate_vars = []
concentration_vars = []
inf_vars = {}
tau_vars = {} 
for var in model.states():
    if var == vm:
        continue
    ret = hh.get_inf_and_tau(var, vm)
    if ret:
        gate_vars.append(var)
        inf_vars[ret[0]] = indices[var.qname()]
        tau_vars[ret[1]] = len(gate_vars)-1
    else:
        concentration_vars.append(var)

# Define lhs function
def v(var):
    # if a gate var the outarray is updated with the steady-state
    if isinstance(var, myokit.Derivative) and var not in gate_vars:
        return 'outarray[' + str(indices[var.var().qname()]) + '][i]'
    elif isinstance(var, myokit.Name):
        var = var.var()
    if var.is_state():
        return 'inarray[' + str(indices[var.qname()]) + '][i]'
    elif var.is_constant():
        return 'AC_' + var.uname()
    elif var in tau_vars:
        return 'm_gates_tau[' + str(tau_vars[var]) + '][i]'
    elif var in inf_vars:
        return 'outarray[' + str(inf_vars[var]) + '][i]'
    else:
        return 'AV_' + var.uname()

# Create expression writer
w = nektar.NektarExpressionWriter()
w.set_lhs_function(v)

# Tab is four spaces
tab = '    '

# Get equations
equations = model.solvable_order()

# Get model name
model_name = name if name is not None else model.name();
?>
///////////////////////////////////////////////////////////////////////////////
//
// File: <?= model_name ?>.cpp
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

#include <iostream>
#include <string>

#include <CardiacEPSolver/CellModels/<?= model_name ?>.h>

namespace Nektar
{

std::string <?= model_name ?>::className
    = GetCellModelFactory().RegisterCreatorFunction(
            "<?= model_name ?>",
            <?= model_name ?>::create,
            "");

<?
if variants:
    print('// Register cell model variants')
    print('std::string ' + model_name + '::lookupIds[' + str(len(variants)) + '] = {')
    for variant in variants.keys():
        print(tab + 'LibUtilities::SessionReader::RegisterEnumValue("CellModelVariant",')
        print(2*tab + '"' + variant + '", ' + model_name + '::e' + variant + '),')
    print('};')
    print('\n// Register default variant')
    print('std::string ' + model_name + '::def =')
    print(tab + 'LibUtilities::SessionReader::RegisterDefaultSolverInfo(')
    print(2*tab + '"CellModelVariant", "' + list(variants.keys())[0] + '");')
?>
/**
 * 
 */
<?= model_name ?>::<?= model_name ?>(
        const LibUtilities::SessionReaderSharedPtr& pSession,
        const MultiRegions::ExpListSharedPtr& pField)
    : CellModel(pSession, pField)
{
<?
if variants:
    print(tab + 'model_variant = pSession->GetSolverInfoAsEnum<')
    print(3*tab + model_name + '::Variants>("CellModelVariant");')
?>
    m_nvar = <?= model.count_states() ?>;

    // Gating states (Rush-Larsen)
<?
for var in gate_vars:
    print(tab + 'm_gates.push_back(' + str(indices[var.qname()]) + '); // ' + str(var.qname()))
?>
    // Remaining states (Forward Euler)
<?
for var in concentration_vars:
    print(tab + 'm_concentrations.push_back(' + str(indices[var.qname()]) + '); // ' + str(var.qname()))
?>

    // Set values of constants
<?
variant_dict = {}
for label, eqs in equations.items():
    if eqs.has_equations(const=True):
        print(tab + '/* ' + label + ' */')
        for eq in eqs.equations(const=True):
            # check if included as a variant and will be handled after
            if variants and eq.lhs.var().qname() in variants[list(variants.keys())[0]]:
                variant_dict[eq.lhs.var().qname()] = eq.lhs.var()
                continue
            print(tab + w.eq(eq) + '; // ' + str(eq.lhs.var().unit()))
        print('')

if variants:
    print(tab + 'switch (model_variant) {')
    for variant_name, constant_dict in variants.items():
        print(2*tab + 'case e' + variant_name + ':')
        for name, value in constant_dict.items():
            print(3*tab + v(variant_dict[name]) + ' = ' + str(value) + ';')
        print(3*tab + 'break;')
    print(tab + '}')
?>}

<?= model_name ?>::~<?= model_name ?>()
{
}


void <?= model_name ?>::v_Update(
        const Array<OneD, const Array<OneD, NekDouble> > &inarray,
              Array<OneD,       Array<OneD, NekDouble> > &outarray,
        [[maybe_unused]] const NekDouble                  time)
{
    ASSERTL0(inarray.data() != outarray.data(),
             "Must have different arrays for input and output.");

    /* State variable ordering
<?
for var in model.states():
    print(tab + v(var) + ' ' + var.qname())
?>    */

    for (unsigned int i = 0; i < m_nq; ++i)
    {
<?
for label, eqs in equations.items():
    if eqs.has_equations(const=False):
        print(2*tab + '/* ' + label + ' */')
        for eq in eqs.equations(const=False):
            var = eq.lhs.var()
            if var in model.bindings():
                print(2*tab + v(var) + ' = ' + model.binding(var) + ';')
            elif var in gate_vars:
                # don't want to fill derivatives of Rush-Larsen gating variables
                continue
            else:
                print(2*tab + w.eq(eq) + ';')
        print('')
?>    }
}

void <?= model_name ?>::v_GenerateSummary(SummaryList& s)
{
    SolverUtils::AddSummaryItem(s, "Cell model", "<?= model_name ?>");
}

void <?= model_name ?>::v_SetInitialConditions()
{
<?
if initial_states:
    print(tab + 'switch (model_variant) {')
    for variant_name, variant_inits in initial_states.items():
        print(2*tab + 'case e' + variant_name + ':')
        for eq in variant_inits:
            print(3*tab + 'Vmath::Fill(m_nq, (NekDouble)' + str(eq.rhs) + ', m_cellSol[' + str(indices[eq.lhs.var().qname()]) + '], 1);')
        print(3*tab + 'break;')
    print(tab + '}')
else:
    for eq in model.inits():
        print(tab + 'Vmath::Fill(m_nq, (NekDouble)' + str(eq.rhs) + ', m_cellSol[' + str(indices[eq.lhs.var().qname()]) + '], 1);')
?>
}

std::string <?= model_name ?>::v_GetCellVarName(size_t idx)
{
<?
print(tab + 'switch (idx) {')
for var in model.states():
    print(2*tab + 'case ' + str(var.index()) + ':')
    print(3*tab + 'return "' + var.name() + '";')
print(2*tab + 'default:')
print(3*tab + 'return "unknown";')
print(tab + '}')
?>
}

}
