import os
import json
import numpy as np
import matplotlib.pyplot as plt
import qutip as qu
import tomografia.modulo_auxiliar_general as mag
import tomografia.modulo_auxiliar_optimizacion as mao


class Tomografia():
    def __init__(self,nqubits,tipo_tomografia='Completa'):
        self.nqubits = nqubits
        self.tipo_tomografia = tipo_tomografia
        self.base_espacio_simetrico = mag.cargar_base_espacio_simetrico (self.tipo_tomografia,self.nqubits)

    def build_projectors(self,cant_obs,file_path:'path to json file measurements'):
        """
        Carga los atributos observables_medidos, valores_medios_medidos, observables_no_medidos utilizando
        los proyectores de Pauli provenientes de un json_file.

        cant_obs (int): cantidad de observables elegidos.
        file_path (json file): json file de las mediciones de frecuencias (3**nqubits pauli strings con 
                               2**nqubits freqs cada una). 
                               Ejemplo nqubits=3:
                               json_file = {'XZY':{'000':0.2,
                                                   '001':0.1,
                                                   ...
                                                  },
                                            'XZZ':{'000':0.4,
                                                   '001':0.0,
                                                   ...
                                                  },
                                            ...
                                           }
        """
        dir_DicTomografia = file_path
        dic_tomografia =json.load(open(dir_DicTomografia) )


        dic_proyectores = mag.dic_tomografia_to_dic_proyectores (dic_tomografia, self.tipo_tomografia)
        
        observables, valores_medios = mag.armar_lista_proyectores_frecuencias(dic_proyectores)

        try:
            cant_obs<=len(observables)
        except:
            raise Exception('La cantidad de observables no puede superar el maximo: {}'.format(len(observables)))
        
        observables_medidos = observables[:cant_obs] 
        valores_medios_medidos = valores_medios[:cant_obs]
        observables_no_medidos = observables[cant_obs:] 

        self.observables_medidos = observables_medidos
        self.valores_medios_medidos = valores_medios_medidos
        self.observables_no_medidos = observables_no_medidos


    def optimization(self,optimizador:'callable'=None):
        """
        optimizador (function callable): funcion optimzacion elegida.
        """
        try:
            self.observables_medidos
            self.valores_medios_medidos
            self.observables_no_medidos
        except:
            msg = ("Antes de optimizar, los atributos self.observables_medidos, self.valores_medios_medidos, "
                   +" self.observables_no_medidos tienen que estar cargados. Ver self.build_projectors().")
            raise Exception(msg)
        parametros_estimados, rho_estimado = optimizador(self.base_espacio_simetrico, 
                                                         self.observables_medidos, 
                                                         self.valores_medios_medidos, 
                                                         self.observables_no_medidos)
        if type(rho_estimado)==type(np.array([])):
            pass
        elif type(rho_estimado)==type(qu.Qobj()):
            rho_estimado = rho_estimado.full()

        rho_estimado_corregido = mao.corregir_autoval_neg(rho_estimado)

        return parametros_estimados, rho_estimado_corregido
        
