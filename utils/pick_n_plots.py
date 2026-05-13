import numpy as np

def pick_indices(n,lista):
    """
    n (int) amount of plots to pick.
    lista (list) list of numbers.
    """
    length = len(lista)
    if n==-1:
        return [length-1]
    if length <= n:
        return list(range(n))
    else:
        _indices=np.linspace(0,length-1,n).tolist()
        indices = []
        for f in _indices:
            indices.append(int(f))
        return indices
        


