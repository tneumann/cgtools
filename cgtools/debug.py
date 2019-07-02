from IPython.core import debugger


__all__ = ['debug']

debug = debugger.Pdb().set_trace
