"""
Correr desde root.
"""
import os
import pickle

import numpy as np


def calcular_mediciones_rho_vs_base(matriz_densidad,ruta_base:"str o lista de narray"):
    """
    Dado una matriz densidad calcula los valores medios con respecto
    a base (e.g. sic_povm, diagonal).
    matriz_densidad:numpy array
    ruta_base: ruta file path.
    """
    if type(ruta_base)==str:
        file_base = open(ruta_base,'rb')
        base_from_pickle = pickle.load(file_base)
        file_base.close()
    else:
        #NOTE: Assume lista de narrays
        base_from_pickle = ruta_base

    measurements = []
    for element in base_from_pickle:
        measurements.append(np.trace(element @ matriz_densidad).real)
    return measurements



def calcular_mediciones_de_lista_de_estados(ruta_lista_de_estados,ruta_base:"path or list of narrays"):
    """
    Calcula mediciones de una lista de estados contra base (e.g. sic_povm,etc). 
    ruta_lista_de_estados: pickle file de lista de matrices densidad.
    ruta_base: ruta file path.
    return: lista de lista de mediciones por estado.
    """
    file_lista_de_estados = open(ruta_lista_de_estados,'rb')
    lista_de_estados = pickle.load(file_lista_de_estados)
    file_lista_de_estados.close()

    if isinstance(ruta_base,str):
        file_base = open(ruta_base,'rb')
        base_from_pickle = pickle.load(file_base)
        file_base.close()
    else:
        base_from_pickle = ruta_base #asume lista de narrays.

    lista_de_mediciones_por_estado = []
    for c, estado in enumerate(lista_de_estados):
        measurements = []
        for element in base_from_pickle:
            measurements.append(np.trace(element @ estado).real)
        lista_de_mediciones_por_estado.append(measurements)
    return lista_de_mediciones_por_estado


#if __name__=='__main__':
#    route_base_sic_povm_file = os.path.join('tomografia','bases_sic-povm','base_{}_{}a.pickle'.format(NAME_BASE,2**NQUBITS)) 
#    lista_de_mediciones_por_estado = calcular_mediciones_de_lista_de_estados(RUTA_PICKLE_LISTA_ESTADOS,route_base_sic_povm_file)
#    file_lista_de_mediciones_por_estado = open(RUTA_PICKLE_LISTA_MEDIC_POR_ESTADOS,'wb')
#    pickle.dump(lista_de_mediciones_por_estado,file_lista_de_mediciones_por_estado)
#    file_lista_de_mediciones_por_estado.close()



