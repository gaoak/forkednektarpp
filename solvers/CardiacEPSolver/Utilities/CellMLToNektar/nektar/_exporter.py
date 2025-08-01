#
# Exports to Nektar
#
# This file is derived from AnsiCExporter in myokit.
# See http://myokit.org for copyright, sharing, and licensing details.
#

import os

import myokit.formats

class NektarExporter(myokit.formats.TemplatedRunnableExporter):
    """
    This class exports a myokit cell model to Nektar++ code for
    use in the CardiacEPSolver.
    """
    def info(self):
        import inspect
        return inspect.getdoc(self)

    def _dir(self, root):
        return os.path.join(os.getcwd(), 'nektar', 'template')

    def _dict(self):
        return {'model.cpp': 'model.cpp',
                'model.h': 'model.h'}

    def _vars(self, model, protocol, name=None, variants=None, initial_states=None):
        if initial_states and not variants:
            raise AssertionError('If `initial_states` is passed, `variants` must also be passed')
        return {'model': model, 
                'protocol': protocol,
                'name': name,
                'variants': variants,
                'initial_states': initial_states}
