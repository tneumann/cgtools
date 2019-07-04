try:
    import ipdb
    debug = ipdb.set_trace
except ImportError:
    from IPython.core import debugger
    debug = debugger.Pdb().set_trace

__all__ = ['debug']
