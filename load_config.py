import configparser
import os
import pickle

import numpy as np

from generate_mediciones import calcular_mediciones_de_lista_de_estados

config = configparser.ConfigParser()
config.read('config.ini')


DIR_ROOT_RUNS = os.path.join(*config['DEFAULT']['DIR_ROOT_RUNS'].split('/'))
dir_user_run = os.path.join(*config['user_run']['dir_user_run'].split('/'))
dir_user_estados = os.path.join(*config['user_run']['dir_user_estados'].split('/'))
NQUBITS = config.getint('user_run','nqubits')
NOMBRE_TIPO_TOMOGRAFIA = str(config['user_run']['nombre_tipo_tomografia'])
NAME_BASE = str(config['user_run']['nombre_base_observables'])
DESC_ESTADOS = str(config['user_run']['descripcion_estados']).replace(' ','_')
#RANGO = config.getint('user_run','rango')
RANGO = config['user_run']['rango']
VQT = config.getboolean('user_run','vqt')
VQT_INF = config.getboolean('user_run','vqt_inf')
VQT_HIB = config.getboolean('user_run','vqt_hib')
MAXENT = config.getboolean('user_run','maxent')
MAXENT_EXP = config.getboolean('user_run','maxent_exp')
#################
ALPHA = config.getfloat('parametros_opt','alpha')
BETA = config.getfloat('parametros_opt','beta')
MAX_ITER_VQT = config.getint('parametros_opt','max_iter_vqt')
MAX_ITER_VQT_INF = config.getint('parametros_opt','max_iter_vqt_inf')
MAX_ITER_VQT_HIB = config.getint('parametros_opt','max_iter_vqt_hib')
MAX_ITER_MAXENT = config.getint('parametros_opt','max_iter_maxent')
MAX_ITER_MAXENT_EXP = config.getint('parametros_opt','max_iter_maxent_exp')
STOP_LOSS_MAXENT_EXP = config.getfloat('parametros_opt','stop_loss_maxent_exp')
#####################3
REALIZAR_TOM_BARRIDO = config.getboolean('tomografia_barrido','realizar')
MIN_CANT_OBS = config.getint('tomografia_barrido','min_cant_obs')
MAX_CANT_OBS = config.getint('tomografia_barrido','max_cant_obs')
MAX_CANT_ESTADOS = config.getint('tomografia_barrido','max_cant_estados')
SAVE_INTER_STATES = config.getboolean('tomografia_barrido','save_intermediate_states')
DESDE_ESTADO_INDEX = config.getint('tomografia_barrido','desde_estado_index')
#####################3
REALIZAR_TOM_TEST = config.getboolean('tomografia_test','realizar')
TOM_TEST_CANT_OBS = config.getint('tomografia_test','cant_obs')
TOM_TEST_CANT_ESTADOS = config.getint('tomografia_test','cant_estados')
TOM_TEST_DESDE_ESTADO_INDEX = config.getint('tomografia_test','desde_estado_index')
#####################
GRAFICAR_BARRIDO = config.getboolean('graficar','graficar_barrido')
####################
RUIDO = config.getboolean('mediciones','ruido')
TIPO_RUIDO = config.get('mediciones','tipo_ruido', fallback='')
####################
_ruta_pickle_lista_estados_dir = os.path.join(dir_user_estados,
                     'Estados_de_{}_qubits'.format(NQUBITS),
                     'Rango{}'.format(RANGO))
try:
    lista_files = os.listdir(_ruta_pickle_lista_estados_dir)
except:
    raise Exception('No existe lista de estados (.pickle) en {}'.format(_ruta_pickle_lista_estados_dir))
try:
    assert len(lista_files)==1
except:
    raise Exception('Solo puede haber un solo .pickle en {}'.format(_ruta_pickle_lista_estados_dir))
RUTA_PICKLE_LISTA_ESTADOS = os.path.join(_ruta_pickle_lista_estados_dir,
                                     os.listdir(_ruta_pickle_lista_estados_dir)[0])



dir_prefix = os.path.join(DIR_ROOT_RUNS,dir_user_run)
DIR_RHO_TARGET = os.path.join(dir_prefix,*config['DEFAULT']['DIR_RHO_TARGET'].split('/'))
DIR_RHO_ESTIMADO = os.path.join(dir_prefix,*config['DEFAULT']['DIR_RHO_ESTIMADO'].split('/'))
DIR_RHO_ESTIMADO_TEST = os.path.join(dir_prefix,*config['DEFAULT']['DIR_RHO_ESTIMADO_TEST'].split('/'))
DIR_TOM_FIDELIDAD = os.path.join(dir_prefix,*config['DEFAULT']['DIR_TOM_FIDELIDAD'].split('/'))
DIR_GRAFICOS_DENSITYMATRICES = os.path.join(dir_prefix,*config['DEFAULT']['DIR_GRAFICOS_DENSITYMATRICES'].split('/'))
DIR_GRAFICOS_TOM = os.path.join(dir_prefix,*config['DEFAULT']['DIR_GRAFICOS_TOM'].split('/'))
DIR_GRAFICOS_TOM_TEST = os.path.join(dir_prefix,*config['DEFAULT']['DIR_GRAFICOS_TOM_TEST'].split('/'))
DIR_MEASUREMENTS = os.path.join(dir_prefix,*config['DEFAULT']['DIR_MEASUREMENTS'].split('/'))

os.makedirs(DIR_RHO_TARGET, exist_ok=True) 
os.makedirs(DIR_RHO_ESTIMADO, exist_ok=True) 
os.makedirs(DIR_RHO_ESTIMADO_TEST, exist_ok=True) 
os.makedirs(DIR_TOM_FIDELIDAD, exist_ok=True) 
os.makedirs(DIR_GRAFICOS_DENSITYMATRICES, exist_ok=True)
os.makedirs(DIR_GRAFICOS_TOM, exist_ok=True) 
os.makedirs(DIR_GRAFICOS_TOM_TEST, exist_ok=True) 
os.makedirs(DIR_MEASUREMENTS, exist_ok=True) 

def lista_estados_from_pickle():
    try:
        file_pickle_lista_estados = open(RUTA_PICKLE_LISTA_ESTADOS,'rb')
        lista_estados = pickle.load(file_pickle_lista_estados) 
        file_pickle_lista_estados.close()
    except:
        raise Exception('Algo salio mal con la carga del pickle de lista estados.')
    return lista_estados 


def base_observables_from_pickle():
    vec_dim = 2**int(NQUBITS)
    if NAME_BASE=='sic_povm' or NAME_BASE=='sic-povm':
        route_base_file = os.path.join('tomografia','bases_sic-povm','base_{}_{}a.pickle'.format(NAME_BASE,vec_dim)) 
    elif NAME_BASE=='diagonal':
        route_base_file = os.path.join('tomografia','bases_diagonal','base_{}_{}.pickle'.format(NAME_BASE,vec_dim)) 
    try:
        file = open(route_base_file,'rb')
        lista_observables = pickle.load(file)
        file.close()
    except:
        raise Exception('La base {} no esta cargada para nqubits={}.'.format(NAME_BASE,NQUBITS))
    return lista_observables 


def _generar_mediciones():
    generar = config.getboolean('mediciones','generar')
    __ruta_pickle_lista_medic_por_estados_dir = os.path.join('Mediciones',
             'Mediciones_de_estados_contra_base_{}'.format(NAME_BASE),
             'Estados_de_{}_qubits'.format(NQUBITS),'Rango{}'.format(RANGO))
    try:
        ruta_pickle_lista_mediciones_por_estados = config.get('mediciones',
                                               'ruta_pickle_lista_mediciones_por_estados_custom')
        if ruta_pickle_lista_mediciones_por_estados == '':
            _ruta_pickle_lista_medic_por_estados_dir = __ruta_pickle_lista_medic_por_estados_dir 
            ruta_pickle_custom_exist = False
        else:
            _ruta_pickle_lista_medic_por_estados_dir = os.path.dirname(
                                        ruta_pickle_lista_mediciones_por_estados)
            ruta_pickle_custom_exist = True
    except:
            _ruta_pickle_lista_medic_por_estados_dir = __ruta_pickle_lista_medic_por_estados_dir 
            ruta_pickle_custom_exist = False
    os.makedirs(_ruta_pickle_lista_medic_por_estados_dir,exist_ok=True)
    if ruta_pickle_custom_exist:
        ruta_pickle_lista_medic_por_estados = ruta_pickle_lista_mediciones_por_estados
    else:
        if RUIDO:
            print('RUIDO-->',RUIDO)
            ruta_pickle_lista_medic_por_estados = os.path.join(
                    _ruta_pickle_lista_medic_por_estados_dir,
                    'mediciones_con_ruido_{}_de_'.format(TIPO_RUIDO)+os.listdir(_ruta_pickle_lista_estados_dir)[0]
                    )
        else:
            print('RUIDO-->',RUIDO)
            ruta_pickle_lista_medic_por_estados = os.path.join(
                    _ruta_pickle_lista_medic_por_estados_dir,
                    'mediciones_de_'+os.listdir(_ruta_pickle_lista_estados_dir)[0]
                    )
    if generar and ruta_pickle_custom_exist:
        raise Exception(("WARNING: LLave 'generar' en section [mediciones] en config.ini esta puesta ",
                 "en True y a su vez ruta_pickle_lista_mediciones_por_estados fue definido ",
                 "por el usuario. No se generan mediciones nuevas."))
    elif generar and not ruta_pickle_custom_exist:
        print('Mediciones-->(re)generando mediciones.')
        lista_de_mediciones_por_estado = calcular_mediciones_de_lista_de_estados(
                RUTA_PICKLE_LISTA_ESTADOS,
                base_observables_from_pickle()
                )
        if RUIDO:
            if TIPO_RUIDO == 'uniforme':
                lista_mediciones_por_estados_ruidosa = []
                for c, mediciones in enumerate(lista_de_mediciones_por_estado):
                    mediciones_arr = np.array(mediciones, dtype=float)
                    rng = np.random.default_rng(42+c)
                    ruido = mediciones_arr * rng.uniform(-0.05, 0.05, size=mediciones_arr.shape)
                    lista_mediciones_por_estados_ruidosa.append((mediciones_arr + ruido).tolist())
                lista_de_mediciones_por_estado = lista_mediciones_por_estados_ruidosa
                print('TIPO_RUIDO', TIPO_RUIDO)

            elif TIPO_RUIDO == '':
                raise Exception("Se necesita especificar tipo de ruido.")
            else:
                raise Exception("Tipo de ruido '{}' no soportado.".format(TIPO_RUIDO))
        file_lista_de_mediciones_por_estado = open(ruta_pickle_lista_medic_por_estados,'wb')
        pickle.dump(lista_de_mediciones_por_estado,file_lista_de_mediciones_por_estado)
        file_lista_de_mediciones_por_estado.close()
    else:
        print("Mediciones--> NO se (re)generan nuevas.")
    return  ruta_pickle_lista_medic_por_estados

RUTA_PICKLE_LISTA_MEDIC_POR_ESTADOS = _generar_mediciones()


def lista_mediciones_por_estados_from_pickle():
    try:
        file_pickle_lista_medic_por_estados = open(RUTA_PICKLE_LISTA_MEDIC_POR_ESTADOS,'rb')
        lista_mediciones_por_estados = pickle.load(file_pickle_lista_medic_por_estados)
        file_pickle_lista_medic_por_estados.close()
    except:
        raise Exception('Algo salio mal con la carga del pickle de lista de mediciones por estados.')

    return lista_mediciones_por_estados
