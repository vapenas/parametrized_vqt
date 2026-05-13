import os
import json

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
#from qiskit.circuit import QuantumRegister
from qiskit.quantum_info import  DensityMatrix
#from qiskit_aer import AerSimulator
#from qiskit_ibm_runtime.fake_provider import FakePerth
#from qiskit_experiments.framework import ParallelExperiment
#from qiskit_experiments.library import StateTomography
from qiskit.visualization import plot_state_city
import qutip as qu

from tests.mapas_y_mediciones_para_tomografia.generador_mediciones_proyectores_estandar import QiskitSimulacionTomografia
from tomografia.tomografia import Tomografia
import tomografia.modulo_auxiliar_general as mag
import tomografia.modulo_auxiliar_optimizacion as mao
from utils.pick_n_plots import pick_indices


class StateTomography():
    def __init__(self,estado,optimizador_tomografia='vqt',**kwargs):
        """
        estado: density matrix (narray) or qiskit quantum circuit.
        optimizador_tomografia (str): tipo de optimizador de tomografia: 
                                      'vqt','vqt_inf','vqt_hib', 'maxent', 'maxent_exp'.
        """
        if isinstance(estado,type(QuantumCircuit())):
            self.qcircuit = qcircuit
            self.nqubits = qcircuit.num_qubits
            self.name = qcircuit.name

            density_matrix_target = DensityMatrix(qcircuit)
            density_matrix_target_reverse = density_matrix_target.reverse_qargs()
            #self.density_matrix_exact_reverse = density_matrix_exact_reverse
            density_matrix_target_para_marce_arr =np.array(density_matrix_target_reverse) 
            self.rho_target = density_matrix_target_para_marce_arr 
        else:
            try:
                assert isinstance(estado,type(np.array([])))
            except:
                raise Exception('Se esperaba un numpy array como density matrix.')

            if estado.shape == (len(estado),):#Si estado es un vector.
                self.rho_target = np.kron(np.conjugate(estado),estado.reshape((len(estado),1)))
            elif estado.shape == (len(estado),len(estado)):
                self.rho_target = estado
            else:
                raise Exception('El estado tiene que ser un numpy array vector o matriz cuadrada.')
            try:
                self.name = kwargs['nombre_rho_target']
            except:
                raise Exception(('Se requiere algun nombre para el estado'+ 
                                    ': nombre_rho_target en StateTomograpy()'))
            self.nqubits = int(np.log2(len(estado))) 

        #self.max_cant_obs = max_cant_obs
        self.optimizador_tomografia = optimizador_tomografia 

        try:
            self.tipo_tomografia = kwargs['tipo_tomografia']
        except:
            self.tipo_tomografia = 'Completa'

        print('TIPO TOMOGRAFIA: ',self.tipo_tomografia)

        self.metadata = {}
        try:
            self.metadata['dir_rho_target'] = kwargs['dir_rho_target']
        except:
            self.metadata['dir_rho_target'] = None
        try:
            self.metadata['dir_rho_estimado'] = kwargs['dir_rho_estimado']
        except:
            self.metadata['dir_rho_estimado'] = None
        try:
            self.metadata['dir_tom_fidelidad'] = kwargs['dir_tom_fidelidad']
        except:
            self.metadata['dir_tom_fidelidad'] = None
        try:
            self.metadata['dir_graficos_densitymatrices'] = kwargs['dir_graficos_densitymatrices']
        except:
            self.metadata['dir_graficos_densitymatrices'] = None
        try:
            self.metadata['dir_graficos_tom'] = kwargs['dir_graficos_tom']
        except:
            self.metadata['dir_graficos_tom'] = None
        try:
            self.metadata['dir_measurements'] = kwargs['dir_measurements']
        except:
            self.metadata['dir_measurements'] = None

    def _metadata_file_path(self,default_dir_path, default_name_file, file_path):
        if file_path != None:
            return file_path
        else:
            try:
                self.metadata[default_dir_path] != None
                file_path = os.path.join(self.metadata[default_dir_path],
                                         default_name_file)
                return file_path
            except:
                raise Exception('A file_path must be given to this method.')

    def _default_rho_target_npy_file_or_user_file_path(self,file_path):
        default_name='rho_target_{}_{}q.npy'.format(self.name,self.nqubits)
        file_path = self._metadata_file_path('dir_rho_target',default_name,file_path) 
        return file_path

    def _default_rho_target_png_file_or_user_file_path(self,file_path):
        default_name = 'rho_target_{}_{}q_qiskitreversed.png'.format(self.name,self.nqubits)
        file_path = self._metadata_file_path('dir_graficos_densitymatrices',default_name,file_path)
        return file_path

    def _default_measurements_json_file_name(self,shots,file_path):
        default_name = 'Measurements_{}_{}q_shots-{}.json'.format(self.name,self.nqubits,shots)
        file_path = self._metadata_file_path('dir_measurements',default_name,file_path)
        return file_path

    def _default_rho_estimado_npy_file_path(self,files_dir_path,maxcantobs):
        default_name='rho_estimada_{}_{}_{}q_maxcantobs-{}.npy'.format(self.optimizador_tomografia,self.name,self.nqubits,maxcantobs)
        if files_dir_path != None:
            try:
                os.path.isdir(files_dir_path)
                user_file_path = os.path.join(files_dir_path,default_name)
                file_path = user_file_path
            except:
                raise Exception('A directory path must be given to save estimated rhos, intermediate rhos and fidelity.')
        else:
            file_path = self._metadata_file_path('dir_rho_estimado',default_name,None) 

        return file_path


    def _default_rho_estimado_png_file_path(self,files_dir_path,maxcantobs):
        default_name='rho_estimada_{}_{}_{}q_maxcantobs-{}.png'.format(self.optimizador_tomografia,self.name,self.nqubits,maxcantobs)
        if files_dir_path != None:
            try:
                os.path.isdir(files_dir_path)
                user_file_path = os.path.join(files_dir_path,default_name)
                file_path = user_file_path
            except:
                raise Exception('A directory path must be given to save estimated rhos, intermediate rhos and fidelity.')
        else:
            file_path = self._metadata_file_path('dir_graficos_tom',default_name,None) 

        return file_path


    def _default_rho_estimado_intermedio_npy_file_path(self,files_dir_path,cantobs):
        default_name='_rho_estimada_{}_{}_{}q_cantobs-{}.npy'.format(self.optimizador_tomografia,self.name,self.nqubits,cantobs)
        if files_dir_path != None:
            try:
                os.path.isdir(files_dir_path)
                user_file_path = os.path.join(files_dir_path,default_name)
                file_path = user_file_path
            except:
                raise Exception('A directory path must be given to save estimated rhos, intermediate rhos and fidelity.')
        else:
            file_path = self._metadata_file_path('dir_rho_estimado',default_name,None) 

        return file_path


    def _default_rho_estimado_intermedio_png_file_path(self,files_dir_path,cantobs):
        default_name='_rho_estimada_{}_{}_{}q_cantobs-{}.png'.format(self.optimizador_tomografia,self.name,self.nqubits,cantobs)
        if files_dir_path != None:
            try:
                os.path.isdir(files_dir_path)
                user_file_path = os.path.join(files_dir_path,default_name)
                file_path = user_file_path
            except:
                raise Exception('A directory path must be given to plot estimated rhos, intermediate rhos and fidelity.')
        else:
            file_path = self._metadata_file_path('dir_graficos_tom',default_name,None) 

        return file_path


    def _default_saved_indices_rho_intermedios_npy_file_path(self,files_dir_path):
        default_name = '_saved_indices_rho_intermedios_{}_{}_{}q.npy'.format(self.optimizador_tomografia,self.name,self.nqubits)
        if files_dir_path != None:
            try:
                os.path.isdir(files_dir_path)
                user_file_path = os.path.join(files_dir_path,default_name)
                file_path = user_file_path
            except:
                raise Exception('A directory path must be given to save estimated rhos, intermediate rhos and fidelity.')
        else:
            file_path = self._metadata_file_path('dir_rho_estimado',default_name,None) 

        return file_path


    def _default_fidelidad_npy_file_path(self,files_dir_path,min_cant_obs,max_cant_obs):
        default_name='fidelidad_{}_{}_mincantobs-{}_maxcantobs-{}_{}q.npy'.format(self.optimizador_tomografia,self.name,min_cant_obs,max_cant_obs,self.nqubits)
        if files_dir_path != None:
            try:
                os.path.isdir(files_dir_path)
                user_file_path = os.path.join(files_dir_path,default_name)
                file_path = user_file_path
            except:
                raise Exception('A directory path must be given to save estimated rhos, intermediate rhos and fidelity.')
        else:
            file_path = self._metadata_file_path('dir_tom_fidelidad',default_name,None) 

        return file_path


    def _default_fidelidad_png_file_path(self,files_dir_path,min_cant_obs,max_cant_obs):
        default_name='fidelidad_{}_{}_mincantobs-{}_maxcantobs-{}_{}q.png'.format(self.optimizador_tomografia,self.name,min_cant_obs,max_cant_obs,self.nqubits)
        if files_dir_path != None:
            try:
                os.path.isdir(files_dir_path)
                user_file_path = os.path.join(files_dir_path,default_name)
                file_path = user_file_path
            except:
                raise Exception('A directory path must be given to save estimated rhos, intermediate rhos and fidelity.')
        else:
            file_path = self._metadata_file_path('dir_graficos_tom',default_name,None) 

        return file_path


    def _default_trace_dist_npy_file_path(self,files_dir_path,min_cant_obs,max_cant_obs):
        default_name='trace_dist_{}_{}_mincantobs-{}_maxcantobs-{}_{}q.npy'.format(self.optimizador_tomografia,self.name,min_cant_obs,max_cant_obs,self.nqubits)
        if files_dir_path != None:
            try:
                os.path.isdir(files_dir_path)
                user_file_path = os.path.join(files_dir_path,default_name)
                file_path = user_file_path
            except:
                raise Exception('A directory path must be given to save estimated rhos, intermediate rhos and fidelities.')
        else:
            file_path = self._metadata_file_path('dir_tom_fidelidad',default_name,None) 

        return file_path


    def _default_trace_dist_png_file_path(self,files_dir_path,min_cant_obs,max_cant_obs):
        default_name='trace_dist_{}_{}_mincantobs-{}_maxcantobs-{}_{}q.png'.format(self.optimizador_tomografia,self.name,min_cant_obs,max_cant_obsself.nqubits)
        if files_dir_path != None:
            try:
                os.path.isdir(files_dir_path)
                user_file_path = os.path.join(files_dir_path,default_name)
                file_path = user_file_path
            except:
                raise Exception('A directory path must be given to save estimated rhos, intermediate rhos and fidelities.')
        else:
            file_path = self._metadata_file_path('dir_graficos_tom',default_name,None) 

        return file_path


    def save_rho_target(self,file_path=None):
        """
        file_path: Path of npy file to save rho_target. If None redirects to default parameters.
        """
        file_path = self._default_rho_target_npy_file_or_user_file_path(file_path) 
        np.save(file_path,self.rho_target)

    def graficar_rho_target(self,file_path=None):
        """
        file_path: Path of png file to save rho_target. If None redirects to default parameters.
        """
        file_path = self._default_rho_target_png_file_or_user_file_path(file_path)
        try:
            os.path.isfile(file_path)
        except:
            raise Exception('No se encuentra file_path del rho_target para poder graficar.')
        #plot_state_city(self.density_matrix_exact_reverse, 
        plot_state_city(self.rho_target, 
                        title='Density Matrix Exact',
                        filename=file_path
                        )

    def realizar_mediciones(self,seed=None,shots=None,file_path=None):
        """
        Mediciones de circuito cuantico.

        seed (int) valor de seed para realizar las mediciones.
        shots (int) cantidad de shots del circuito cuantico.
        file_path: Path of json file to save measurements. If None redirects to default parameters.
        """
        if type(self.seed) == int:
            pass
        else:
            raise Exception('An int value for seed must be given.')
        if type(shots)==int:
            self.shots = shots
        else:
            Exception('An int value for shots must be given.')
        qst = QiskitSimulacionTomografia(self.qcircuit)
        mediciones,_,__ = qst.mediciones(shots=shots,seed=seed)
        file_path = self._default_measurements_json_file_name(shots,file_path)
        with open(file_path,'w') as f:
            f.write(json.dumps(mediciones,indent=10))

    #DEPRECATED:
    def realizar_tomografia_estandar(self,
                                     max_cant_obs=15,
                                     measurement_file_path=None,
                                     save_intermediate_rho_states=True,
                                     files_dir_path=None):
        """
        NOTE: Se encuentra deprecada. Hay que modificarla o eliminarla.

        Realiza tomografia estandar usando mediciones de proyectores de Pauli y un mapeo interno entre las 
        mediciones que devuelve qiskit y la implementacion de marce de la tomografia. Este metodo consume un 
        json de mediciones para efectuarse.

        max_cant_obs (int): Maxima cantidad de observables a barrer.
        measurement_file_path: User path al json de realizar_medicion(). Por ahora tomografia solo arranca 
                               si existe el json guardado proveniente de realizar_medicion(). Si None, 
                               lo busca dentro de directorios defaults usando los parametros instanciados.
        save_intermediate_rho_states (bool): Guarda rhos intermedios (npy) con una cierta frecuencia hardcodeada.
        files_dir_path: User directory path donde van a ser guardados los rho estimados, rho intermedios y la 
                        fidelidad. Si None, usa default parameters para guardarlos.
        """
        if measurement_file_path != None:
            pass
        else:
            try:
                measurement_file_path = self._default_measurements_json_file_name(self.shots, measurement_file_path)
                os.path.isfile(measurement_file_path)
            except:
                raise Exception('A measurement_file_path must be given.')

        self.max_cant_obs = max_cant_obs
        self.cant_obs_list = list(range(1,max_cant_obs+1))
        cant_obs_list = self.cant_obs_list
        optimizador_tomografia = self.optimizador_tomografia
        fidelity_list = []
        trace_dist_list = []
        indices_a_guardar_rho_intermedios = pick_indices(4,cant_obs_list)
        if save_intermediate_rho_states:
            saved_indices_rho_intermedios_file_path = self._default_saved_indices_rho_intermedios_npy_file_path(
                                                                                                   files_dir_path)
            np.save(saved_indices_rho_intermedios_file_path, np.array(indices_a_guardar_rho_intermedios))

        for c, i in enumerate(cant_obs_list):
            #Ojo que tipo_tomografia='Completa' aca refiere a si es simetrica o no. 
            #No tiene nada que ver con la cant de obs elegidos.
            tom = Tomografia(self.nqubits,tipo_tomografia=self.tipo_tomografia) 
            tom.build_projectors(i,file_path=measurement_file_path)
            parametros_estimados, rho_estimado = tom.optimization(
                                    optimizador=eval('mao.optimizacion_cvxpy_{}'.format(optimizador_tomografia)))
            if c in indices_a_guardar_rho_intermedios:
                if save_intermediate_rho_states:
                    rho_estimado_intermedio_file_path = self._default_rho_estimado_intermedio_npy_file_path(
                                                                                              files_dir_path,i)
                    np.save(rho_estimado_intermedio_file_path,rho_estimado)

            
            #FIDELITY
            rho_target = self.rho_target
            fidelidad, trace_distance = mag.comparar_con_estado_real(rho_estimado, rho_target )
            fidelity_list.append(fidelidad)
            trace_dist_list.append(trace_distance)
            print(fidelidad, trace_distance)



        maxcantobs = i
        rho_estimado_file_path = self._default_rho_estimado_npy_file_path(files_dir_path,maxcantobs)
        fidelity_file_path = self._default_fidelidad_npy_file_path(files_dir_path)
        trace_dist_file_path = self._default_trace_dist_npy_file_path(files_dir_path)
        np.save(rho_estimado_file_path,rho_estimado)
        np.save(fidelity_file_path,np.array(fidelity_list))
        np.save(trace_dist_file_path,np.array(trace_dist_list))
        return fidelity_list, trace_dist_list, rho_estimado


    def realizar_tomografia_custom(self,
                                   min_cant_obs,
                                   max_cant_obs,
                                   observables_medidos=None,
                                   valores_medios_medidos=None,
                                   #observables_no_medidos = None,
                                   save_intermediate_rho_states=True,
                                   files_dir_path=None):
        """
        Realiza tomografia usando mediciones de observables customs. 

        min_cant_obs (int): Minima cantidad de observables para empezar a barrer.
        max_cant_obs (int): Maxima cantidad de observables a barrer.
        save_intermediate_rho_states (bool): Guarda rhos intermedios (npy) con una cierta frecuencia hardcodeada.
        files_dir_path: User directory path donde van a ser guardados los rho estimados, rho intermedios y la 
                        fidelidad. Si None, usa default parameters para guardarlos.
        """

        if min_cant_obs<=0:
            raise Exception("Minima cantidad de observables a barrer es 1.")
        self.min_cant_obs = min_cant_obs #El minimo es 1
        self.max_cant_obs = max_cant_obs
        self.cant_obs_list = list(range(min_cant_obs,max_cant_obs+1))
        cant_obs_list = self.cant_obs_list
        optimizador_tomografia = self.optimizador_tomografia
        fidelity_list = []
        trace_dist_list = []
        #NOTE:De ahora en mas se guardan todos los estados e indices para los estados intermedios:
        #indices_a_guardar_rho_intermedios = pick_indices(max_cant_obs,cant_obs_list)
        indices_a_guardar_rho_intermedios = cant_obs_list
        if save_intermediate_rho_states:
            saved_indices_rho_intermedios_file_path = self._default_saved_indices_rho_intermedios_npy_file_path(
                                                                                                   files_dir_path)
            np.save(saved_indices_rho_intermedios_file_path, np.array(indices_a_guardar_rho_intermedios))

        #for c, i in enumerate(cant_obs_list):
        for i in cant_obs_list:
            #Ojo que tipo_tomografia='Completa' aca refiere a si es simetrica o no. 
            #No tiene nada que ver con la cant de obs elegidos.
            tom = Tomografia(self.nqubits,tipo_tomografia=self.tipo_tomografia) 
            tom.observables_medidos = observables_medidos[:i]
            tom.valores_medios_medidos = valores_medios_medidos[:i]
            tom.observables_no_medidos = observables_medidos[i:]
            if self.optimizador_tomografia != 'maxent_exp':
                parametros_estimados, rho_estimado = tom.optimization(
                                optimizador=eval('mao.optimizacion_cvxpy_{}'.format(optimizador_tomografia)))
            else:
                parametros_estimados, rho_estimado = tom.optimization(
                                optimizador=eval('mao.optimizacion_maxent_de_d'))

            #if c in indices_a_guardar_rho_intermedios:
            if save_intermediate_rho_states:
                rho_estimado_intermedio_file_path = self._default_rho_estimado_intermedio_npy_file_path(
                                                                                         files_dir_path,i)
                np.save(rho_estimado_intermedio_file_path,rho_estimado)

            
            #FIDELITY
            rho_target = self.rho_target
            fidelidad, trace_distance = mag.comparar_con_estado_real(rho_estimado, rho_target )
            fidelity_list.append(fidelidad)
            trace_dist_list.append(trace_distance)
            print(fidelidad, trace_distance)



        maxcantobs = i
        rho_estimado_file_path = self._default_rho_estimado_npy_file_path(files_dir_path,maxcantobs)
        fidelity_file_path = self._default_fidelidad_npy_file_path(files_dir_path,
                                                                   self.min_cant_obs,
                                                                   self.max_cant_obs)
        trace_dist_file_path = self._default_trace_dist_npy_file_path(files_dir_path,
                                                                      self.min_cant_obs,
                                                                      self.max_cant_obs)
        np.save(rho_estimado_file_path,rho_estimado)
        np.save(fidelity_file_path,np.array(fidelity_list))
        np.save(trace_dist_file_path,np.array(trace_dist_list))

        return fidelity_list, trace_dist_list, rho_estimado


    def graficar_rho_estimado(self,files_dir_path=None):
        rho_estimado_npy_file_path = self._default_rho_estimado_npy_file_path(files_dir_path,self.max_cant_obs)
        rho_estimado_png_file_path = self._default_rho_estimado_png_file_path(files_dir_path,self.max_cant_obs)
        try:
            os.path.isfile(rho_estimado_npy_file_path)
        except:
            raise Exception("No se encontro file_path para graficar rho estimado.")
        rho_estimado = np.load(rho_estimado_npy_file_path)
        plot_state_city(rho_estimado,
                        title='Estimated Density Matrix, {} tomography'.format(self.optimizador_tomografia),
                        filename = rho_estimado_png_file_path
                        )

    def graficar_rho_estimados_intermedios(self,files_dir_path=None):
        """
        files_dir_path: User directory path donde fueron guardados los rho estimados, rho intermedios y la fidelidad.
                        Si None, usa default parameters para guardarlos.
        """
        saved_indices_rho_intermedios_file_path = self._default_saved_indices_rho_intermedios_npy_file_path(files_dir_path)
        try:
            os.path.isfile(saved_indices_rho_intermedios_file_path)
        except:
            raise Exception('Faltan indices para graficar rho intermedios.')
        indices_guardados_rho_intermedios = np.load(saved_indices_rho_intermedios_file_path).tolist()


        for i in indices_guardados_rho_intermedios:
        #for c, i in enumerate(self.cant_obs_list):
        #if c in indices_guardados_rho_intermedios:
            rho_estimado_intermedio_file_path =self._default_rho_estimado_intermedio_npy_file_path(files_dir_path,i) 
            try:
                os.path.isfile(rho_estimado_intermedio_file_path)
            except:
                raise Exception('Oops. No se encontro el path para rho_intermedio cant_obs-{}'.format(i))
            rho_estimado_intermedio = np.load(rho_estimado_intermedio_file_path)
            rho_estimado_intermedio_png_file_path = self._default_rho_estimado_intermedio_png_file_path(files_dir_path,i)
            plot_state_city(rho_estimado_intermedio, 
                            title='Estimated Density Matrix, {} tomography'.format(self.optimizador_tomografia),
                            filename=rho_estimado_intermedio_png_file_path
                           )

    def graficar_fidelidad(self,files_dir_path=None):
        """
        files_dir_path: User directory path donde fueron guardados los rho estimados, rho intermedios y la fidelidad.
                        Si None, usa default parameters para guardarlos.
        """
        fidelity_npy_file_path = self._default_fidelidad_npy_file_path(files_dir_path,self.min_cant_obs,self.max_cant_obs)
        fidelity_png_file_path = self._default_fidelidad_png_file_path(files_dir_path,self.min_cant_obs,self.max_cant_obs)
        try:
            os.path.isfile(fidelity_npy_file_path)
        except:
            raise Exception('No se encuentra file_path de fidelidad.')
        fidelity_list = np.load(fidelity_npy_file_path)
        #GRAFICOS
        # Define the size of the figure in pixels
        width_pixels = 1400
        height_pixels = 1000
        # Set the dpi (dots per inch)
        dpi = 100
        # Convert pixels to inches
        width_inches = width_pixels / dpi
        height_inches = height_pixels / dpi
        fig, ax = plt.subplots()
        ax.plot(self.cant_obs_list,fidelity_list,label=self.optimizador_tomografia)
        ax.legend()
        plt.xlabel('Number of observables')
        plt.ylabel('Fidelity')
        plt.title('Fidelity vs Number of Observables ({} {}qubits)'.format(self.name,self.nqubits))
        fig.set_size_inches(width_inches, height_inches)
        fig.savefig(fidelity_png_file_path)
        plt.close(fig)
