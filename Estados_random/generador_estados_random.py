"""
Correr este .py desde Estados_random/
"""
###############################################################################
#Cargo paquetes y funciones
import qutip as qt
import pickle
import os
import numpy as np

#from numpy.linalg import eig
#import sys 
#from scipy.special import comb
#from scipy.stats import unitary_group
###############################################################################


def generar_estados_cuanticos(n_est, n_q, rango):
    """
    Genera n_est matrices densidad (qutip.Qobj) aleatorias de n_q qubits de rango rango usando QuTiP, con semillas que van desde 0 hasta n_est.
    
    Parámetros:
    - n_q: número de qubits
    - n_est: número de estados a generar
    - rango: rango de la matriz densidad
    Retorna:
    - lista de n_est estados cuánticos random de n_q qubits  (qutip.Qobj)
    """
       
    dim = 2** n_q
    if 0 < rango < dim + 1:
        estados = []
        for i in range(n_est):
            #estado = qt.rand_dm_ginibre(N = dim, rank=rango, seed=i).full()
            estado = qt.rand_dm(dim,distribution='ginibre',rank=rango,seed=i).full()
            estados.append(estado) 
    if rango == 0: print('El rango no puede ser nulo') 
    if rango > dim:  print('El rango no puede ser mayor a la dimensión del espacio de Hilbert') 

    return estados



def crear_carpetas(n_q):
    """
    Crea estructura de carpetas para n_q qubits. Una carpeta llamada 'Estados_de_' + str (n_q) + '_qubits' y adentro carpetas para cada rango posible.
    
    Parámetros:
    - n_q: número de qubits
    """
    carpeta = 'Estados_de_' + str (n_q) + '_qubits'

    if not os.path.exists(carpeta):
        os.makedirs(carpeta)
        print(f"Carpeta '{carpeta}' creada.")
    else:
        print(f"La carpeta '{carpeta}' ya existe.")

    # Crear las subcarpetas numeradas
    n = 2 ** n_q
    for i in range(1, n+1):
        subcarpeta = os.path.join(carpeta, 'Rango'  + str(i))
        if not os.path.exists(subcarpeta):
            os.makedirs(subcarpeta)
            print(f"Subcarpeta '{subcarpeta}' creada.")
        else:
            print(f"La subcarpeta '{subcarpeta}' ya existe.")




def guardar_lista_estados(estados, n_est, n_q, rango):
    """
    Guarda una lista de objetos Qobj de QuTiP como un único archivo pickle.
    
    Parámetros:
    - estados: lista de objetos Qobj
    - n_q: número de qubits
    - n_est: número de estados a generar
    - rango: rango de la matriz densidad
    """
    # Crear la carpeta si no existe
    carpeta = 'Estados_de_' + str (n_q) + '_qubits'
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)

    subcarpeta = os.path.join(carpeta, 'Rango'  + str(rango))
    if not os.path.exists(subcarpeta):
        os.makedirs(subcarpeta)

    # Ruta completa del archivo
    nombre_archivo = 'cant_estados_' + str(n_est)  + '-nqubits_' + str (n_q)  + '-rango_'+ str(rango) + '.pkl'
    ruta_archivo = os.path.join(subcarpeta, nombre_archivo)
    
    # Guardar la lista completa
    with open(ruta_archivo, 'wb') as f:
        pickle.dump(estados, f)
    
    print(f"Lista de estados guardada en: {ruta_archivo}")



def cargar_lista_estados(n_est, n_q, rango):
    """
    Carga una lista de objetos Qobj de QuTiP desde un archivo pickle.
    
    Parámetros:
    - n_q: número de qubits
    - n_est: número de estados a generar
    - rango: rango de la matriz densidad
    
    Retorna:
    - lista de objetos Qobj
    """
    carpeta = 'Estados_de_' + str (n_q) + '_qubits'
    subcarpeta = 'Rango' + str(rango)
    ruta_carpeta = os.path.join(carpeta, subcarpeta)
    nombre_archivo = 'cant_estados_' + str(n_est)  + '-nqubits_' + str (n_q)  + '-rango_'+ str(rango) + '.pkl'
    ruta_archivo = os.path.join(ruta_carpeta, nombre_archivo)
    
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_archivo}")
    
    with open(ruta_archivo, 'rb') as f:
        estados = pickle.load(f)
    
    print(f"Se cargaron {len(estados)} estados desde {ruta_archivo}")
    return estados


def chequear_estados (estados, n_est, n_q, rango):
    condicion_cant_estados = len(estados) == n_est
    condicion_rango = [rango == np.linalg.matrix_rank(estado) for estado in estados  ]    
    condicion_nq = [2**n_q == estado.shape[0] for estado in estados  ]    
    valor = (condicion_cant_estados -1 ) + sum(np.array(condicion_rango) - 1) + sum(np.array( condicion_nq) - 1)   
    if valor == 0 : print ('Chequeo OK')
    if not valor == 0 : print ('Chequeo ERROR')
    return valor

def generar_guardar_estados( n_est, n_q):
    """
    Genera n_est matrices densidad (qutip.Qobj) aleatorias de n_q qubits para cada rango posible usando QuTiP, con semillas que van desde 0 hasta n_est.
    Guarda los estados para cada rango en una lista de objetos Qobj de QuTiP como un único archivo pickle.
    
    Parámetros:
    - n_q: número de qubits
    - n_est: número de estados a generar
    """
    dim = 2** n_q
    for rango in range (1, dim + 1 ):
        estados = generar_estados_cuanticos(n_est, n_q, rango)
        valor = chequear_estados (estados, n_est, n_q, rango)
        if valor == 0: guardar_lista_estados(estados, n_est, n_q , rango)


###############################################################################

# Generar estados
n_est = 100
n_q = 5
generar_guardar_estados( n_est, n_q)


