# Python environment with the myokit installed is required.
# To compile the resulting C++ code, need libsundials-dev package.

# A conda environment with the correct dependencies can be create to
# run the converter using the `environment.yml` file:
#    conda env create -f environment.yml

import myokit
import myokit.formats.cellml as cellml
import myokit.lib.guess as guess
import myokit.lib.hh as hh
import nektar

# Create importer for CellML file
importer = cellml.CellMLImporter()

# Create exporter to Nektar
exporter = nektar.NektarExporter()

# The model is first imported into Myokit
model = importer.model('courtemanche_ramirez_nattel_1998.cellml')

# At this point, it can be manipulated as a Myokit model before
# exporting (see Myokit docs)
# Note that it may be necessary to remove any embedded protocol
# at either this step or before loading the CellML model
_ = guess.remove_embedded_protocol(model)

# The exporter takes 4 arguments:
#   Name of the directory in which to save the source and header
#   The Myokit model to export
#   A Myokit protocol which for the Nektar exporter will be ignored
#   A dictionary of variants (not required, see example)
#   A dictionary of initial states (not required, see example)

# The below code will export the model without any variant information
exporter.runnable('example', model, None)

# If instead we want to include two variants of the courtemanche model
# we can include that information for it to be included in the model
variants = {'Original': {'transient_outward_K_current.g_to': 0.1652,
                         'L_type_Ca_channel.g_Ca_L': 0.12375},
            'AF':       {'transient_outward_K_current.g_to':0.0826,
                         'L_type_Ca_channel.g_Ca_L': 0.037125}}
exporter.runnable('example_variants', model, None, variants)

# Including different initial conditions by running to steady state in Myokit
initial_states = {}
for variant_name, parameter_dict in variants.items():
    # Set parameters in variant
    for p_name, p_val in parameter_dict.items():
        model.set_value(p_name, p_val)
    # Create simulation and run to steady-state
    sim = myokit.Simulation(model)
    sim.pre(100e3)
    # Set state to end-state and store Equations
    model.set_initial_values(sim.state())
    initial_states[variant_name] = list(model.inits())
exporter.runnable('example_initial_states', model, None, variants, initial_states)
