import os
CWD = os.getcwd()
print('CWD',CWD)
#import pickle
#import itertools as it
import numpy as np
import cvxpy as cp
from numpy import linalg as LA
import qutip as qu
import json
import itertools as it

##############################################################################
#Funcion save y load 
def save_pet(pet,filename):
    with open(filename, 'w') as f:
        f.write(json.dumps(pet))
        
def load_pet(filename):
    with open(filename) as f:
        pet = json.loads(f.read())
    return pet
##############################################################################


def armar_keys(cantidad_qubits, tipo_tomografia):
    if tipo_tomografia == 'Permutacional':
        lista_keys = []
        for i in range(cantidad_qubits + 1):
            Xs = ['X0'] *i
            for j in range(cantidad_qubits + 1  - i):
                Ys = ['Y0'] *j
                for k in range(cantidad_qubits + 1 - i - j):
                    Z0s = ['Z0'] * k 
                    Z1s = ['Z1'] * (cantidad_qubits-i-j- k)
                    lista_keys.append('_'.join(Xs + Ys + Z0s + Z1s)) 
                    #tuple (Xs + Ys + Z0s + Z1s )
    if tipo_tomografia == 'Completa':
        lista_keys = list(it.product(['X0','Y0','Z0','Z1'], repeat=cantidad_qubits))
        lista_keys = ['_'.join(key) for key in lista_keys]
    return lista_keys

def dic_tomografia_to_dic_proyectores (dic_tomografia, tipo_tomografia):
    cantidad_qubits = len(list(dic_tomografia.keys())[0])
    lista_keys = armar_keys(cantidad_qubits, tipo_tomografia)
    dic_proyectores = {}
    for key in lista_keys:
        sigma_key = ''.join ([Sigma[0] for Sigma in key.split('_') ])
        binary_key = ''.join ([Sigma[1] for Sigma in key.split('_') ])
        ##
        try: 
            dic_proyectores[key] = dic_tomografia[sigma_key][binary_key]
        except:
            dic_proyectores[key] = 0.0
        ##
    return dic_proyectores


##############################################################################
def cargar_base_espacio_simetrico (tipo_base,cantidad_qubits,
                                   nombre_carpeta_bases = os.path.join(CWD,'tomografia','bases_espacio_simetrico')
                                   ):
    nombre_archivo_base ="/Base_" + tipo_base + "_" + str(cantidad_qubits)
    base_espacio_simetrico_load = np.load(nombre_carpeta_bases + nombre_archivo_base + ".npz", allow_pickle=True)
    base_espacio_simetrico = [base_espacio_simetrico_load[key] for key in base_espacio_simetrico_load] 
    return base_espacio_simetrico

def armar_lista_proyectores_frecuencias (dic_proyectores):
    #3 dicc_tomografia_fede, numero_shots,  
    Z0 = np.array([[1,0],[0,0]])
    Z1 = np.array([[0,0],[0,1]])
    X0 = np.array([[1],[1]])@ np.array([[1],[1]]).T.conjugate()  /2
    Y0 = np.array([[1],[1j]])@ np.array([[1],[1j]]).T.conjugate()  /2
    proyec_elementales = {'Z0':Z0, 'Z1': Z1, 'X0':X0, 'Y0': Y0}
    proyectores = []
    frecuencias = []
    for key in dic_proyectores:
        frecuencia = dic_proyectores[key]
        lista_nombre_proyectores = key.split('_')
        
        proyector = 1
        for i in lista_nombre_proyectores:
            proyector = np.kron(proyector, proyec_elementales[i])

        proyectores.append(proyector)
        frecuencias.append(frecuencia)
        
    return proyectores, frecuencias




def comparar_con_estado_real(rho_tomo, rho_real ):
    fidelidad = qu.fidelity(qu.Qobj (rho_real), qu.Qobj (rho_tomo))
    trace_distance = qu.tracedist(qu.Qobj (rho_real), qu.Qobj (rho_tomo))
    return fidelidad, trace_distance
##############################################################################



