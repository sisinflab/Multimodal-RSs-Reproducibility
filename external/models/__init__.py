def import_model_by_backend(tensorflow_cmd, pytorch_cmd):
    import sys
    for _backend in sys.modules["external"].backend:
        if _backend == "tensorflow":
            exec(tensorflow_cmd)
        elif _backend == "pytorch":
            exec(pytorch_cmd)
            break


import sys
for _backend in sys.modules["external"].backend:
    if _backend == "tensorflow":
        pass
    elif _backend == "pytorch":
        from .grcn.GRCN import GRCN
        from .lattice.LATTICE import LATTICE
        from .lightgcn_m.LightGCNM import LightGCNM
        from .freedom.FREEDOM import FREEDOM
        from .mmgcn.MMGCN import MMGCN
        from .mbpr.MBPR import MBPR
        from .bm3.BM3 import BM3
