# Python environment with the myokit installed is required.
# To compile the resulting C++ code, need libsundials-dev package.

# A conda environment with the correct dependencies can be create to
# run the converter using the `environment.yml` file:
#    conda env create -f environment.yml
import os
import argparse
import myokit
import myokit.formats.cellml as cellml
import myokit.lib.guess as guess
import myokit.lib.hh as hh
import nektar

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('-n','--name', 
                help='Override the CellML model name used as class name.')
parser.add_argument('-o','--outputpath', default='.',
                help='Specify the output directory for the generated model.')
parser.add_argument('filename',
                help='CellML model to convert')
args = parser.parse_args()

# Create importer for CellML file
importer = cellml.CellMLImporter()

# Create exporter to Nektar
exporter = nektar.NektarExporter()

# The model is first imported into Myokit
#model = importer.model('courtemanche_ramirez_nattel_1998.cellml')
model = importer.model(args.filename)

# Get model name
model_name = args.name if args.name is not None else model.name();

# At this point, it can be manipulated as a Myokit model before
# exporting (see Myokit docs)
# Note that it may be necessary to remove any embedded protocol
# at either this step or before loading the CellML model
_ = guess.remove_embedded_protocol(model)

# The exporter takes 4 arguments:
#   Name of the directory in which to save the source and header
#   The Myokit model to export
#   A Myokit protocol which for the Nektar exporter will be ignored
#   A name for the generated model class (not required)
#   A dictionary of variants (not required, see example)
#   A dictionary of initial states (not required, see example)

# The below code will export the model without any variant information
exporter.runnable(args.outputpath, model, None, args.name)
os.rename(args.outputpath + "/model.cpp",
          args.outputpath + "/" + model_name + ".cpp")
os.rename(args.outputpath + "/model.h",
          args.outputpath + "/" + model_name + ".h")
