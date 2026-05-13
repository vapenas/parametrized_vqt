import numpy as np

def mostrar_archivo (nombre):
    data = np.load(nombre)
    # Acceder a cada array
    for nombre in data.files:
        print(nombre)
        print(data[nombre])





import numpy as np

def base_canonica_diagonal(n):
    """
    Genera las matrices de la base canónica diagonal de dimensión n×n.
    Cada matriz tiene un único 1 en la diagonal, el resto son ceros.
    """
    bases = []
    for i in range(n):
        M = np.zeros((n, n))
        M[i, i] = 1
        bases.append(M)
    return bases



# Cantidad de qubits
n = 5
# Generar las matrices
bases = base_canonica_diagonal(2**n)
nombre = 'Base_Diagonal_' + str (n) + '.npz'

# Guardarlas en un archivo .npz
np.savez(nombre, **{f"arr_{i}": B for i, B in enumerate(bases)})
mostrar_archivo (nombre)