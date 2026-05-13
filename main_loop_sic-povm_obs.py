"""
Correr este file desde root.
"""
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np


from load_config import (DIR_RHO_TARGET,
                         DIR_RHO_ESTIMADO,
                         DIR_RHO_ESTIMADO_TEST,
                         DIR_TOM_FIDELIDAD,
                         DIR_GRAFICOS_DENSITYMATRICES,
                         DIR_GRAFICOS_TOM,
                         DIR_GRAFICOS_TOM_TEST,
                         DIR_MEASUREMENTS,
                         NQUBITS,
                         NOMBRE_TIPO_TOMOGRAFIA,
                         NAME_BASE, 
                         MIN_CANT_OBS,
                         MAX_CANT_OBS,
                         MAX_CANT_ESTADOS,
                         SAVE_INTER_STATES,
                         DESDE_ESTADO_INDEX,
                         DESC_ESTADOS,
                         RANGO,
                         VQT,
                         VQT_INF,
                         VQT_HIB,
                         MAXENT,
                         MAXENT_EXP,
                         REALIZAR_TOM_BARRIDO,
                         REALIZAR_TOM_TEST,
                         TOM_TEST_CANT_OBS,
                         TOM_TEST_CANT_ESTADOS,
                         TOM_TEST_DESDE_ESTADO_INDEX,
                         GRAFICAR_BARRIDO,
                         lista_estados_from_pickle,
                         lista_mediciones_por_estados_from_pickle,
                         base_observables_from_pickle
                         )

from state_tomography import StateTomography
from utils.join_filenames_fidelities import build_index_dictionary,get_fidelity_list

if MAX_CANT_OBS > 4**NQUBITS:
    raise Exception("Maxima cantidad de observables tiene que ser menor a 4**NQUBITS.")
if DESDE_ESTADO_INDEX <0:
    raise Exception('Indices are positive integers.')
elif DESDE_ESTADO_INDEX > MAX_CANT_ESTADOS-1:
    raise Exception('Indice maximo tiene que ser igual a la maxima cantidad de estados.')

if REALIZAR_TOM_BARRIDO:
    index_list = list(range(DESDE_ESTADO_INDEX,MAX_CANT_ESTADOS))
    for index in index_list:
        comienzo_for = """


                        COMIENZO ESTADO: {}


        """.format(index)
        print(comienzo_for)
        NOMBRE_RHO_TARGET = '{}_{}'.format(DESC_ESTADOS,index)
        ############VQT STATE TOMOGRAPHY
        if VQT:
            st = StateTomography(lista_estados_from_pickle()[index],
                                 optimizador_tomografia = 'vqt',
                                 tipo_tomografia = NOMBRE_TIPO_TOMOGRAFIA,
                                 dir_rho_target               = DIR_RHO_TARGET,
                                 dir_rho_estimado             = DIR_RHO_ESTIMADO,
                                 dir_tom_fidelidad            = DIR_TOM_FIDELIDAD,
                                 dir_graficos_densitymatrices = DIR_GRAFICOS_DENSITYMATRICES, 
                                 dir_graficos_tom             = DIR_GRAFICOS_TOM,
                                 dir_measurements             = DIR_MEASUREMENTS,
                                 nombre_rho_target            = NOMBRE_RHO_TARGET
                                 )
            st.save_rho_target()
            #st.graficar_rho_target()
            ti = time.time()
            fidelidad_vqt_list,trace_dist_vqt_list,rho_estimado_vqt = st.realizar_tomografia_custom(
                    min_cant_obs=MIN_CANT_OBS,
                    max_cant_obs=MAX_CANT_OBS,
                    observables_medidos = base_observables_from_pickle(),
                    valores_medios_medidos = lista_mediciones_por_estados_from_pickle()[index],
                    save_intermediate_rho_states=SAVE_INTER_STATES)
            #st.graficar_rho_estimado()
            #st.graficar_rho_estimados_intermedios()
            #st.graficar_fidelidad()
            #total_fidelity_vqt.append(fidelidad_vqt_list)
            #total_trace_dist_vqt.append(trace_dist_vqt_list)
            #total_time_vqt_list.append(time.time()-ti)
            #rho_estimado_del_maxcantobs_dic["VQT"].append(rho_estimado_vqt)
            time_vqt_file_name = 'time_vqt_{}_{}-index_{}q.npy'.format(DESC_ESTADOS,index,NQUBITS)
            np.save(os.path.join(DIR_TOM_FIDELIDAD,time_vqt_file_name),np.array(time.time()-ti)) 
        ############MAXENT STATE TOMOGRAPHY
        if MAXENT:
            st = StateTomography(lista_estados_from_pickle()[index],
                                 optimizador_tomografia = 'maxent',
                                 tipo_tomografia = NOMBRE_TIPO_TOMOGRAFIA,
                                 dir_rho_target               = DIR_RHO_TARGET,
                                 dir_rho_estimado             = DIR_RHO_ESTIMADO,
                                 dir_tom_fidelidad            = DIR_TOM_FIDELIDAD,
                                 dir_graficos_densitymatrices = DIR_GRAFICOS_DENSITYMATRICES, 
                                 dir_graficos_tom             = DIR_GRAFICOS_TOM,
                                 dir_measurements             = DIR_MEASUREMENTS,
                                 nombre_rho_target            = NOMBRE_RHO_TARGET
                                 )
            #st.save_rho_target()
            #st.graficar_rho_target()
            ti = time.time()
            fidelidad_maxent_list,trace_dist_maxent_list,rho_estimado_maxent = st.realizar_tomografia_custom(
                    min_cant_obs=MIN_CANT_OBS,
                    max_cant_obs=MAX_CANT_OBS,
                    observables_medidos = base_observables_from_pickle(),
                    valores_medios_medidos = lista_mediciones_por_estados_from_pickle()[index],
                    save_intermediate_rho_states=SAVE_INTER_STATES)
            #st.graficar_rho_estimado()
            #st.graficar_rho_estimados_intermedios()
            #st.graficar_fidelidad()
            #total_fidelity_maxent.append(fidelidad_maxent_list)
            #total_trace_dist_maxent.append(trace_dist_maxent_list)
            #total_time_maxent_list.append(time.time()-ti)
            #rho_estimado_del_maxcantobs_dic["MAXENT"].append(rho_estimado_maxent)
            time_maxent_file_name = 'time_maxent_{}_{}-index_{}q.npy'.format(DESC_ESTADOS,index,NQUBITS)
            np.save(os.path.join(DIR_TOM_FIDELIDAD,time_maxent_file_name),np.array(time.time()-ti)) 
        ############VQT_INF STATE TOMOGRAPHY
        if VQT_INF:
            st = StateTomography(lista_estados_from_pickle()[index],
                                 optimizador_tomografia = 'vqt_inf',
                                 tipo_tomografia = NOMBRE_TIPO_TOMOGRAFIA,
                                 dir_rho_target               = DIR_RHO_TARGET,
                                 dir_rho_estimado             = DIR_RHO_ESTIMADO,
                                 dir_tom_fidelidad            = DIR_TOM_FIDELIDAD,
                                 dir_graficos_densitymatrices = DIR_GRAFICOS_DENSITYMATRICES, 
                                 dir_graficos_tom             = DIR_GRAFICOS_TOM,
                                 dir_measurements             = DIR_MEASUREMENTS,
                                 nombre_rho_target            = NOMBRE_RHO_TARGET
                                 )
            #st.save_rho_target()
            #st.graficar_rho_target()
            #st.realizar_mediciones()
            ti = time.time()
            fidelidad_vqt_inf_list,trace_dist_vqt_inf_list,rho_estimado_vqt_inf = st.realizar_tomografia_custom(
                    min_cant_obs=MIN_CANT_OBS,
                    max_cant_obs=MAX_CANT_OBS,
                    observables_medidos = base_observables_from_pickle(),
                    valores_medios_medidos = lista_mediciones_por_estados_from_pickle()[index],
                    save_intermediate_rho_states=SAVE_INTER_STATES)
            #st.graficar_rho_estimado()
            #st.graficar_rho_estimados_intermedios()
            #st.graficar_fidelidad()
            #total_fidelity_vqt_inf.append(fidelidad_vqt_inf_list)
            #total_trace_dist_vqt_inf.append(trace_dist_vqt_inf_list)
            #total_time_vqt_inf_list.append(time.time()-ti)
            #rho_estimado_del_maxcantobs_dic["VQT_INF"].append(rho_estimado_vqt_inf)
            time_vqt_inf_file_name = 'time_vqt_inf_{}_{}-index_{}q.npy'.format(DESC_ESTADOS,index,NQUBITS)
            np.save(os.path.join(DIR_TOM_FIDELIDAD,time_vqt_inf_file_name),np.array(time.time()-ti)) 

        ############VQT_HIB STATE TOMOGRAPHY
        if VQT_HIB:
            st = StateTomography(lista_estados_from_pickle()[index],
                                 optimizador_tomografia = 'vqt_hib',
                                 tipo_tomografia = NOMBRE_TIPO_TOMOGRAFIA,
                                 dir_rho_target               = DIR_RHO_TARGET,
                                 dir_rho_estimado             = DIR_RHO_ESTIMADO,
                                 dir_tom_fidelidad            = DIR_TOM_FIDELIDAD,
                                 dir_graficos_densitymatrices = DIR_GRAFICOS_DENSITYMATRICES, 
                                 dir_graficos_tom             = DIR_GRAFICOS_TOM,
                                 dir_measurements             = DIR_MEASUREMENTS,
                                 nombre_rho_target            = NOMBRE_RHO_TARGET
                                 )
            #st.save_rho_target()
            #st.graficar_rho_target()
            #st.realizar_mediciones()
            ti = time.time()
            fidelidad_vqt_hib_list,trace_dist_vqt_hib_list,rho_estimado_vqt_hib = st.realizar_tomografia_custom(
                    min_cant_obs=MIN_CANT_OBS,
                    max_cant_obs=MAX_CANT_OBS,
                    observables_medidos = base_observables_from_pickle(),
                    valores_medios_medidos = lista_mediciones_por_estados_from_pickle()[index],
                    save_intermediate_rho_states=SAVE_INTER_STATES)
            time_vqt_hib_file_name = 'time_vqt_hib_{}_{}-index_{}q.npy'.format(DESC_ESTADOS,index,NQUBITS)
            np.save(os.path.join(DIR_TOM_FIDELIDAD,time_vqt_hib_file_name),np.array(time.time()-ti)) 

        ############MAXENT_EXP STATE TOMOGRAPHY
        if MAXENT_EXP:
            st = StateTomography(lista_estados_from_pickle()[index],
                                 optimizador_tomografia = 'maxent_exp',
                                 tipo_tomografia = NOMBRE_TIPO_TOMOGRAFIA,
                                 dir_rho_target               = DIR_RHO_TARGET,
                                 dir_rho_estimado             = DIR_RHO_ESTIMADO,
                                 dir_tom_fidelidad            = DIR_TOM_FIDELIDAD,
                                 dir_graficos_densitymatrices = DIR_GRAFICOS_DENSITYMATRICES, 
                                 dir_graficos_tom             = DIR_GRAFICOS_TOM,
                                 dir_measurements             = DIR_MEASUREMENTS,
                                 nombre_rho_target            = NOMBRE_RHO_TARGET
                                 )
            #st.save_rho_target()
            #st.graficar_rho_target()
            #st.realizar_mediciones()
            ti = time.time()
            fidelidad_maxent_exp_list,trace_dist_maxent_exp_list,rho_estimado_maxent_exp = st.realizar_tomografia_custom(
                    min_cant_obs=MIN_CANT_OBS,
                    max_cant_obs=MAX_CANT_OBS,
                    observables_medidos = base_observables_from_pickle(),
                    valores_medios_medidos = lista_mediciones_por_estados_from_pickle()[index],
                    save_intermediate_rho_states=SAVE_INTER_STATES)
            #st.graficar_rho_estimado()
            #st.graficar_rho_estimados_intermedios()
            #st.graficar_fidelidad()
            #total_fidelity_maxent_exp.append(fidelidad_maxent_exp_list)
            #total_trace_dist_maxent_exp.append(trace_dist_maxent_exp_list)
            #total_time_maxent_exp_list.append(time.time()-ti)
            #rho_estimado_del_maxcantobs_dic["MAXENT_EXP"].append(rho_estimado_maxent_exp)
            time_maxent_exp_file_name = 'time_maxent_exp_{}_{}-index_{}q.npy'.format(DESC_ESTADOS,index,NQUBITS)
            np.save(os.path.join(DIR_TOM_FIDELIDAD,time_maxent_exp_file_name),np.array(time.time()-ti)) 

        cant_obs_list = st.cant_obs_list
    

if GRAFICAR_BARRIDO:

    cant_obs_list = list(range(1,MAX_CANT_OBS+1)) #Notar que esto asume un barrido comenzando en 1. Se puede usar MIN_CANT_OBS.

    #GRAFICOS FIDELIDAD:
    print('Comienzo Graficos...')
    print('LEN CANTLIST',len(cant_obs_list))

    total_fidelity_vqt = []
    total_fidelity_maxent = []
    total_fidelity_vqt_inf = []
    total_fidelity_vqt_hib = []
    total_fidelity_maxent_exp = []
    
    total_trace_dist_vqt = []
    total_trace_dist_maxent = []
    total_trace_dist_vqt_inf = []
    total_trace_dist_vqt_hib = []
    total_trace_dist_maxent_exp = []

    total_time_vqt_list = []
    total_time_maxent_list = []
    total_time_vqt_inf_list = []
    total_time_vqt_hib_list = []
    total_time_maxent_exp_list = []


    index_list = list(range(0,MAX_CANT_ESTADOS))#notar que dice 0, en vez de DESDE_ESTADO_INDEX
    indices_encontrados = 0
    lista_from_listdir = os.listdir(os.path.join(DIR_TOM_FIDELIDAD))
    for index in index_list:
        try:
            if VQT:

                index_and_obs_dicc_full_vqt = build_index_dictionary(lista_from_listdir,tipo_opt='vqt')

                fidelidad_vqt_list = get_fidelity_list(str(index),index_and_obs_dicc_full_vqt,MAX_CANT_OBS,'vqt',DESC_ESTADOS,NQUBITS,metrica='fidelidad',folder_location=DIR_TOM_FIDELIDAD)
                total_fidelity_vqt.append(fidelidad_vqt_list)

                trace_dist_vqt_list = get_fidelity_list(str(index),index_and_obs_dicc_full_vqt,MAX_CANT_OBS,'vqt',DESC_ESTADOS,NQUBITS,metrica='trace_dist',folder_location=DIR_TOM_FIDELIDAD)
                total_trace_dist_vqt.append(trace_dist_vqt_list)

                try:
                    time_vqt_file_name = 'time_vqt_{}_{}-index_{}q.npy'.format(DESC_ESTADOS,index,NQUBITS)
                    time_vqt=np.load(os.path.join(DIR_TOM_FIDELIDAD,time_vqt_file_name)) 
                    total_time_vqt_list.append(time_vqt) 
                except:
                    total_time_vqt_list.append(0)


            if MAXENT:
                #ruta_fidelidad_maxent = os.path.join(DIR_TOM_FIDELIDAD,'fidelidad_{}_{}_{}_{}q.npy'.format('maxent',DESC_ESTADOS,index,NQUBITS))
                #fidelidad_maxent_list=np.load(ruta_fidelidad_maxent)
                #total_fidelity_maxent.append(fidelidad_maxent_list)

                #ruta_trace_dist_maxent = os.path.join(DIR_TOM_FIDELIDAD,'trace_dist_{}_{}_{}_{}q.npy'.format('maxent',DESC_ESTADOS,index,NQUBITS))
                #trace_dist_maxent_list=np.load(ruta_trace_dist_maxent)
                #total_trace_dist_maxent.append(trace_dist_maxent_list)

                index_and_obs_dicc_full_maxent = build_index_dictionary(lista_from_listdir,tipo_opt='maxent')

                fidelidad_maxent_list = get_fidelity_list(str(index),index_and_obs_dicc_full_maxent,MAX_CANT_OBS,'maxent',DESC_ESTADOS,NQUBITS,metrica='fidelidad',folder_location=DIR_TOM_FIDELIDAD)
                total_fidelity_maxent.append(fidelidad_maxent_list)

                trace_dist_maxent_list = get_fidelity_list(str(index),index_and_obs_dicc_full_maxent,MAX_CANT_OBS,'maxent',DESC_ESTADOS,NQUBITS,metrica='trace_dist',folder_location=DIR_TOM_FIDELIDAD)
                total_trace_dist_maxent.append(trace_dist_maxent_list)

                try:
                    time_maxent_file_name = 'time_maxent_{}_{}-index_{}q.npy'.format(DESC_ESTADOS,index,NQUBITS)
                    time_maxent=np.load(os.path.join(DIR_TOM_FIDELIDAD,time_maxent_file_name)) 
                    total_time_maxent_list.append(time_maxent) 
                except:
                    total_time_maxent_list.append(0)

            if VQT_INF:

                index_and_obs_dicc_full_vqt_inf = build_index_dictionary(lista_from_listdir,tipo_opt='vqt_inf')

                fidelidad_vqt_inf_list = get_fidelity_list(str(index),index_and_obs_dicc_full_vqt_inf,MAX_CANT_OBS,'vqt_inf',DESC_ESTADOS,NQUBITS,metrica='fidelidad',folder_location=DIR_TOM_FIDELIDAD)
                total_fidelity_vqt_inf.append(fidelidad_vqt_inf_list)

                trace_dist_vqt_inf_list = get_fidelity_list(str(index),index_and_obs_dicc_full_vqt_inf,MAX_CANT_OBS,'vqt_inf',DESC_ESTADOS,NQUBITS,metrica='trace_dist',folder_location=DIR_TOM_FIDELIDAD)
                total_trace_dist_vqt_inf.append(trace_dist_vqt_inf_list)

                try:
                    time_vqt_inf_file_name = 'time_vqt_inf_{}_{}-index_{}q.npy'.format(DESC_ESTADOS,index,NQUBITS)
                    time_vqt_inf=np.load(os.path.join(DIR_TOM_FIDELIDAD,time_vqt_inf_file_name)) 
                    total_time_vqt_inf_list.append(time_vqt_inf) 
                except:
                    total_time_vqt_inf_list.append(0)


            if VQT_HIB:

                index_and_obs_dicc_full_vqt_hib = build_index_dictionary(lista_from_listdir,tipo_opt='vqt_hib')

                fidelidad_vqt_hib_list = get_fidelity_list(str(index),index_and_obs_dicc_full_vqt_hib,MAX_CANT_OBS,'vqt_hib',DESC_ESTADOS,NQUBITS,metrica='fidelidad',folder_location=DIR_TOM_FIDELIDAD)
                total_fidelity_vqt_hib.append(fidelidad_vqt_hib_list)

                trace_dist_vqt_hib_list = get_fidelity_list(str(index),index_and_obs_dicc_full_vqt_hib,MAX_CANT_OBS,'vqt_hib',DESC_ESTADOS,NQUBITS,metrica='trace_dist',folder_location=DIR_TOM_FIDELIDAD)
                total_trace_dist_vqt_hib.append(trace_dist_vqt_hib_list)

                try:
                    time_vqt_hib_file_name = 'time_vqt_hib_{}_{}-index_{}q.npy'.format(DESC_ESTADOS,index,NQUBITS)
                    time_vqt_hib=np.load(os.path.join(DIR_TOM_FIDELIDAD,time_vqt_hib_file_name)) 
                    total_time_vqt_hib_list.append(time_vqt_hib) 
                except:
                    total_time_vqt_hib_list.append(0)


            if MAXENT_EXP:
                #ruta_fidelidad_maxent_exp = os.path.join(DIR_TOM_FIDELIDAD,'fidelidad_{}_{}_{}_{}q.npy'.format('maxent_exp',DESC_ESTADOS,index,NQUBITS))
                #fidelidad_maxent_exp_list=np.load(ruta_fidelidad_maxent_exp)
                #total_fidelity_maxent_exp.append(fidelidad_maxent_exp_list)

                #ruta_trace_dist_maxent_exp = os.path.join(DIR_TOM_FIDELIDAD,'trace_dist_{}_{}_{}_{}q.npy'.format('maxent_exp',DESC_ESTADOS,index,NQUBITS))
                #trace_dist_maxent_exp_list=np.load(ruta_trace_dist_maxent_exp)
                #total_trace_dist_maxent_exp.append(trace_dist_maxent_exp_list)

                index_and_obs_dicc_full_maxent_exp = build_index_dictionary(lista_from_listdir,tipo_opt='maxent_exp')
                fidelidad_maxent_exp_list = get_fidelity_list(str(index),index_and_obs_dicc_full_maxent_exp,MAX_CANT_OBS,'maxent_exp',DESC_ESTADOS,NQUBITS,metrica='fidelidad',folder_location=DIR_TOM_FIDELIDAD)
                total_fidelity_maxent_exp.append(fidelidad_maxent_exp_list)

                trace_dist_maxent_exp_list = get_fidelity_list(str(index),index_and_obs_dicc_full_maxent_exp,MAX_CANT_OBS,'maxent_exp',DESC_ESTADOS,NQUBITS,metrica='trace_dist',folder_location=DIR_TOM_FIDELIDAD)
                total_trace_dist_maxent_exp.append(trace_dist_maxent_exp_list)

                try:
                    time_maxent_exp_file_name = 'time_maxent_exp_{}_{}-index_{}q.npy'.format(DESC_ESTADOS,index,NQUBITS)
                    time_maxent_exp=np.load(os.path.join(DIR_TOM_FIDELIDAD,time_maxent_exp_file_name)) 
                    total_time_maxent_exp_list.append(time_maxent_exp) 
                except:
                    total_time_maxent_exp_list.append(0)

            indices_encontrados = indices_encontrados + 1
        except Exception as e:
            print("No se pudo cargar fidelity, trace_dist para estado: index = {}".format(index))
            continue

        
    if VQT:
        total_fidelity_vqt_average    =np.nanmean(np.array(total_fidelity_vqt),axis=0)
        total_trace_dist_vqt_average  =np.nanmean(np.array(total_trace_dist_vqt),axis=0)
        total_time_vqt_average        =np.average(np.array(total_time_vqt_list))
        print('TOTAL TIME VQT: ',len(total_time_vqt_list)*total_time_vqt_average)
        np.save(os.path.join(DIR_TOM_FIDELIDAD, 'Average_fidelity_vqt_{}.npy'.format(NQUBITS)),
                                        total_fidelity_vqt_average)
        np.save(os.path.join(DIR_TOM_FIDELIDAD,'Average_trace_dist_vqt_{}.npy'.format(NQUBITS)),
                                        total_trace_dist_vqt_average)
    if MAXENT:
        total_fidelity_maxent_average  =np.nanmean(np.array(total_fidelity_maxent),axis=0)
        total_trace_dist_maxent_average=np.nanmean(np.array(total_trace_dist_maxent),axis=0)
        total_time_maxent_average = np.average(np.array(total_time_maxent_list))
        print('TOTAL TIME MAXENT: ',len(total_time_maxent_list)*total_time_maxent_average)
        np.save(os.path.join(DIR_TOM_FIDELIDAD,'Average_fidelity_maxent_{}.npy'.format(NQUBITS)),
                                        total_fidelity_maxent_average)
        np.save(os.path.join(DIR_TOM_FIDELIDAD,'Average_trace_dist_maxent_{}.npy'.format(NQUBITS)),
                                        total_trace_dist_maxent_average)
    if VQT_INF:
        total_fidelity_vqt_inf_average   =np.nanmean(np.array(total_fidelity_vqt_inf),axis=0)
        total_trace_dist_vqt_inf_average =np.nanmean(np.array(total_trace_dist_vqt_inf),axis=0)
        total_time_vqt_inf_average = np.average(np.array(total_time_vqt_inf_list))
        print('TOTAL TIME VQT_INF: ',len(total_time_vqt_inf_list)*total_time_vqt_inf_average)
        np.save(os.path.join(DIR_TOM_FIDELIDAD,'Average_fidelity_vqt_inf_{}.npy'.format(NQUBITS)),
                                        total_fidelity_vqt_inf_average)
        np.save(os.path.join(DIR_TOM_FIDELIDAD,'Average_trace_dist_vqt_inf_{}.npy'.format(NQUBITS)),
                                        total_trace_dist_vqt_inf_average)


    if VQT_HIB:
        total_fidelity_vqt_hib_average   =np.nanmean(np.array(total_fidelity_vqt_hib),axis=0)
        total_trace_dist_vqt_hib_average =np.nanmean(np.array(total_trace_dist_vqt_hib),axis=0)
        total_time_vqt_hib_average = np.average(np.array(total_time_vqt_hib_list))
        print('TOTAL TIME VQT_HIB: ',len(total_time_vqt_hib_list)*total_time_vqt_hib_average)
        np.save(os.path.join(DIR_TOM_FIDELIDAD,'Average_fidelity_vqt_hib_{}.npy'.format(NQUBITS)),
                                        total_fidelity_vqt_hib_average)
        np.save(os.path.join(DIR_TOM_FIDELIDAD,'Average_trace_dist_vqt_hib_{}.npy'.format(NQUBITS)),
                                        total_trace_dist_vqt_hib_average)

    if MAXENT_EXP:
        total_fidelity_maxent_exp_average   =np.nanmean(np.array(total_fidelity_maxent_exp),axis=0)
        total_trace_dist_maxent_exp_average =np.nanmean(np.array(total_trace_dist_maxent_exp),axis=0)
        total_time_maxent_exp_average = np.average(np.array(total_time_maxent_exp_list))
        print('TOTAL TIME MAXENT_EXP: ',len(total_time_maxent_exp_list)*total_time_maxent_exp_average)
        np.save(os.path.join(DIR_TOM_FIDELIDAD,'Average_fidelity_maxent_exp_{}.npy'.format(NQUBITS)),
                                        total_fidelity_maxent_exp_average)
        np.save(os.path.join(DIR_TOM_FIDELIDAD,'Average_trace_dist_maxent_exp_{}.npy'.format(NQUBITS)),
                                        total_trace_dist_maxent_exp_average)



    print("{} samples".format(indices_encontrados))
    import matplotlib.pyplot as plt
    
    width_pixels = 1400
    height_pixels = 1000
    # Set the dpi (dots per inch)
    dpi = 100
    # Convert pixels to inches
    width_inches = width_pixels / dpi
    height_inches = height_pixels / dpi
    
    #Fidelidad
    fig, ax = plt.subplots()
    #ax.plot([i in range(iterations)],fidelity_list_vqt,marker='o')
    if VQT:
        ax.scatter(cant_obs_list, total_fidelity_vqt_average,   label='VQT',marker='d')
    if MAXENT:
        ax.scatter(cant_obs_list, total_fidelity_maxent_average,label='MaxEnt_standard',marker='o')
    if VQT_INF:
        print("LEN TOTAL_FIDELITY_VQT_INF",len(total_fidelity_vqt_inf_average))
        #print("TOTAL_FIDELITY_VQT_INF",total_fidelity_vqt_inf_average)
        ax.scatter(cant_obs_list, total_fidelity_vqt_inf_average,   label=r'VQT$_\infty$',marker='s')
    if VQT_HIB:
        print("LEN TOTAL_FIDELITY_VQT_HIB",len(total_fidelity_vqt_hib_average))
        #print("TOTAL_FIDELITY_VQT_INF",total_fidelity_vqt_hib_average)
        ax.scatter(cant_obs_list, total_fidelity_vqt_hib_average,   label=r'VQT$_{hib}$',marker='s')
    if MAXENT_EXP:
        ax.scatter(cant_obs_list, total_fidelity_maxent_exp_average,label='MaxEnt',marker='*')
    #legend=['VQT','MaxEnt']
    #ax.legend(legend)
    ax.legend(fontsize=15)
    xlabel='Number of Observables ({})'.format(NAME_BASE)
    ylabel='Average Fidelity'
    #plt.xlabel(xlabel)
    #plt.ylabel(ylabel)
    ax.set_xlabel(xlabel,fontsize=15)
    ax.set_ylabel(ylabel,fontsize=15)
    ax.tick_params(axis='both', labelsize=12)
    #title = 'Average Fidelity vs Number of Observables ({} samples, {} qubits)'.format(indices_encontrados,NQUBITS)
    #plt.title(title)
    fig.set_size_inches(width_inches, height_inches)
    default_name_png = 'fidelidades_vqt-vqt_inf-maxent-maxent_exp_Average_{}_Rango{}_{}q.png'.format(
                                 DESC_ESTADOS,RANGO,NQUBITS)
    fig.savefig(os.path.join(DIR_GRAFICOS_TOM,default_name_png))
    plt.close(fig)
    
    #Trace dist:
    fig, ax = plt.subplots()
    #ax.plot([i in range(iterations)],fidelity_list_vqt,marker='o')
    if VQT:
        ax.scatter(cant_obs_list, total_trace_dist_vqt_average,   label='VQT',marker='d')
    if MAXENT:
        ax.scatter(cant_obs_list, total_trace_dist_maxent_average,label='MaxEnt_standard',marker='o')
    if VQT_INF:
        print("TOTAL_TRACE_DIST_VQT_INF",total_trace_dist_vqt_inf_average)
        ax.scatter(cant_obs_list, total_trace_dist_vqt_inf_average,   label=r'VQT$_\infty$',marker='s')
    if VQT_HIB:
        print("TOTAL_TRACE_DIST_VQT_HIB",total_trace_dist_vqt_hib_average)
        ax.scatter(cant_obs_list, total_trace_dist_vqt_hib_average,   label=r'VQT$_{hib}$',marker='s')
    if MAXENT_EXP:
        print("TOTAL_TRACE_DIST_MAXENT_EXP",total_trace_dist_maxent_exp_average)
        ax.scatter(cant_obs_list, total_trace_dist_maxent_exp_average,label='MaxEnt',marker='*')
    #legend=['VQT','MaxEnt']
    #ax.legend(legend)
    ax.legend(fontsize=15)
    xlabel='Number of Observables ({})'.format(NAME_BASE)
    ylabel='Average Trace Distance'
    #plt.xlabel(xlabel)
    #plt.ylabel(ylabel)
    ax.set_xlabel(xlabel,fontsize=15)
    ax.set_ylabel(ylabel,fontsize=15)
    ax.tick_params(axis='both', labelsize=12)
    #title = 'Average Trace Distance vs Number of Observables ({} samples, {} qubits)'.format(indices_encontrados,NQUBITS)
    #plt.title(title)
    fig.set_size_inches(width_inches, height_inches)
    default_name_png = 'trace_distances_vqt-vqt_inf-maxent-maxent_exp_Average_{}_Rango{}_{}q.png'.format(
                                 DESC_ESTADOS,RANGO,NQUBITS)
    fig.savefig(os.path.join(DIR_GRAFICOS_TOM,default_name_png))
    plt.close(fig)



#REALIZAR TOMOGRAFIA TEST 
if REALIZAR_TOM_TEST:
    MAX_CANT_OBS=TOM_TEST_CANT_OBS
    MAX_CANT_ESTADOS = TOM_TEST_CANT_ESTADOS
    print("\n")
    print("\n")
    print("#################################################################################################")
    print("COSTOS")
    from qutip import Qobj, entropy_vn
    from generate_mediciones import calcular_mediciones_rho_vs_base
    from tomografia.tomografia import Tomografia
    import tomografia.modulo_auxiliar_optimizacion as mao
    import tomografia.modulo_auxiliar_general as mag
    width_pixels = 1400
    height_pixels = 1000
    # Set the dpi (dots per inch)
    dpi = 100
    # Convert pixels to inches
    width_inches = width_pixels / dpi
    height_inches = height_pixels / dpi
    fig, ax = plt.subplots()
    observables_medidos = base_observables_from_pickle()
    index_list = list(range(TOM_TEST_DESDE_ESTADO_INDEX,MAX_CANT_ESTADOS))
    costo_por_estado_dic = {}
    entropy_vn_por_estado_dic = {}
    fidelidad_por_estado_dic = {}
    trace_dist_por_estado_dic = {}
    tiempo_por_estado_dic = {}
    tom = Tomografia(NQUBITS,tipo_tomografia=NOMBRE_TIPO_TOMOGRAFIA) 
    tom.observables_medidos = observables_medidos[:MAX_CANT_OBS]#NOTAR el MAX_CANT_OBS
    tom.observables_no_medidos = observables_medidos[MAX_CANT_OBS:]
    if VQT:
        tipo = 'VQT'
        costo_por_estado_dic[tipo]=[]
        entropy_vn_por_estado_dic[tipo]=[]
        fidelidad_por_estado_dic[tipo] = []
        trace_dist_por_estado_dic[tipo] = []
        tiempo_por_estado_dic[tipo] = []
    if MAXENT:
        tipo = 'MAXENT'
        costo_por_estado_dic[tipo]=[]
        entropy_vn_por_estado_dic[tipo]=[]
        fidelidad_por_estado_dic[tipo] = []
        trace_dist_por_estado_dic[tipo] = []
        tiempo_por_estado_dic[tipo] = []
    if VQT_INF:
        tipo = 'VQT_INF'
        costo_por_estado_dic[tipo]=[]
        entropy_vn_por_estado_dic[tipo]=[]
        fidelidad_por_estado_dic[tipo] = []
        trace_dist_por_estado_dic[tipo] = []
        tiempo_por_estado_dic[tipo] = []
    if VQT_HIB:
        tipo = 'VQT_HIB'
        costo_por_estado_dic[tipo]=[]
        entropy_vn_por_estado_dic[tipo]=[]
        fidelidad_por_estado_dic[tipo] = []
        trace_dist_por_estado_dic[tipo] = []
        tiempo_por_estado_dic[tipo] = []
    if MAXENT_EXP:
        tipo = "MAXENT_EXP"
        costo_por_estado_dic[tipo]=[]
        entropy_vn_por_estado_dic[tipo]=[]
        fidelidad_por_estado_dic[tipo] = []
        trace_dist_por_estado_dic[tipo] = []
        tiempo_por_estado_dic[tipo] = []
    for index in index_list:
        valores_medios_medidos = lista_mediciones_por_estados_from_pickle()[index]
        tom.valores_medios_medidos = valores_medios_medidos[:MAX_CANT_OBS]
        if VQT:
            ti = time.time()
            tipo = "VQT"
            _, rho_estimado = tom.optimization(
                    optimizador=eval('mao.optimizacion_cvxpy_{}'.format(tipo.lower())))
            lista_mediciones_estimadas = calcular_mediciones_rho_vs_base(rho_estimado,tom.observables_medidos)
            costo = np.linalg.norm(np.array(tom.valores_medios_medidos)-np.array(lista_mediciones_estimadas))**2
            entropy_vn_value =entropy_vn(Qobj(rho_estimado)) 
            costo_por_estado_dic[tipo].append(costo)
            entropy_vn_por_estado_dic[tipo].append(entropy_vn_value)
            fidelidad,trace_distance = mag.comparar_con_estado_real(rho_estimado,lista_estados_from_pickle()[index])#OJO, usa qutip (libreria fue modificada)
            fidelidad_por_estado_dic[tipo].append(fidelidad)
            trace_dist_por_estado_dic[tipo].append(trace_distance)
            tiempo_por_estado_dic[tipo].append(time.time()-ti)

            NOMBRE_RHO_TARGET = '{}_{}'.format(DESC_ESTADOS,index)
            rho_estimado_intermedio_file_path = '_rho_estimada_{}_{}_{}q_cantobs-{}.npy'.format(
                            tipo.lower(),NOMBRE_RHO_TARGET,NQUBITS,MAX_CANT_OBS)
            np.save(os.path.join(DIR_RHO_ESTIMADO_TEST,rho_estimado_intermedio_file_path),rho_estimado)


        if MAXENT:
            ti = time.time()
            tipo = "MAXENT"
            _, rho_estimado = tom.optimization(
                    optimizador=eval('mao.optimizacion_cvxpy_{}'.format(tipo.lower())))
            lista_mediciones_estimadas = calcular_mediciones_rho_vs_base(rho_estimado,tom.observables_medidos)
            costo = np.linalg.norm(np.array(tom.valores_medios_medidos)-np.array(lista_mediciones_estimadas))**2
            entropy_vn_value =entropy_vn(Qobj(rho_estimado)) 
            costo_por_estado_dic[tipo].append(costo)
            entropy_vn_por_estado_dic[tipo].append(entropy_vn_value)
            fidelidad,trace_distance = mag.comparar_con_estado_real(rho_estimado,lista_estados_from_pickle()[index])#OJO, usa qutip (libreria fue modificada)
            fidelidad_por_estado_dic[tipo].append(fidelidad)
            trace_dist_por_estado_dic[tipo].append(trace_distance)
            tiempo_por_estado_dic[tipo].append(time.time()-ti)

            NOMBRE_RHO_TARGET = '{}_{}'.format(DESC_ESTADOS,index)
            rho_estimado_intermedio_file_path = '_rho_estimada_{}_{}_{}q_cantobs-{}.npy'.format(
                            tipo.lower(),NOMBRE_RHO_TARGET,NQUBITS,MAX_CANT_OBS)
            np.save(os.path.join(DIR_RHO_ESTIMADO_TEST,rho_estimado_intermedio_file_path),rho_estimado)

        if VQT_INF:
            ti = time.time()
            tipo = "VQT_INF"
            _, rho_estimado = tom.optimization(
                    optimizador=eval('mao.optimizacion_cvxpy_{}'.format(tipo.lower())))
            lista_mediciones_estimadas = calcular_mediciones_rho_vs_base(rho_estimado,tom.observables_medidos)
            costo = np.linalg.norm(np.array(tom.valores_medios_medidos)-np.array(lista_mediciones_estimadas))**2
            entropy_vn_value =entropy_vn(Qobj(rho_estimado)) 
            costo_por_estado_dic[tipo].append(costo)
            entropy_vn_por_estado_dic[tipo].append(entropy_vn_value)
            fidelidad,trace_distance = mag.comparar_con_estado_real(rho_estimado,lista_estados_from_pickle()[index])#OJO, usa qutip (libreria fue modificada)
            fidelidad_por_estado_dic[tipo].append(fidelidad)
            trace_dist_por_estado_dic[tipo].append(trace_distance)
            tiempo_por_estado_dic[tipo].append(time.time()-ti)
            print(f'{tipo} trace_dist por estado: ',trace_dist_por_estado_dic[tipo])

            NOMBRE_RHO_TARGET = '{}_{}'.format(DESC_ESTADOS,index)
            rho_estimado_intermedio_file_path = '_rho_estimada_{}_{}_{}q_cantobs-{}.npy'.format(
                            tipo.lower(),NOMBRE_RHO_TARGET,NQUBITS,MAX_CANT_OBS)
            np.save(os.path.join(DIR_RHO_ESTIMADO_TEST,rho_estimado_intermedio_file_path),rho_estimado)


        if VQT_HIB:
            ti = time.time()
            tipo = "VQT_HIB"
            _, rho_estimado = tom.optimization(
                    optimizador=eval('mao.optimizacion_cvxpy_{}'.format(tipo.lower())))
            lista_mediciones_estimadas = calcular_mediciones_rho_vs_base(rho_estimado,tom.observables_medidos)
            costo = np.linalg.norm(np.array(tom.valores_medios_medidos)-np.array(lista_mediciones_estimadas))**2
            entropy_vn_value =entropy_vn(Qobj(rho_estimado)) 
            costo_por_estado_dic[tipo].append(costo)
            entropy_vn_por_estado_dic[tipo].append(entropy_vn_value)
            fidelidad,trace_distance = mag.comparar_con_estado_real(rho_estimado,lista_estados_from_pickle()[index])#OJO, usa qutip (libreria fue modificada)
            fidelidad_por_estado_dic[tipo].append(fidelidad)
            trace_dist_por_estado_dic[tipo].append(trace_distance)
            tiempo_por_estado_dic[tipo].append(time.time()-ti)
            print(f'{tipo} trace_dist por estado: ',trace_dist_por_estado_dic[tipo])

            NOMBRE_RHO_TARGET = '{}_{}'.format(DESC_ESTADOS,index)
            rho_estimado_intermedio_file_path = '_rho_estimada_{}_{}_{}q_cantobs-{}.npy'.format(
                            tipo.lower(),NOMBRE_RHO_TARGET,NQUBITS,MAX_CANT_OBS)
            np.save(os.path.join(DIR_RHO_ESTIMADO_TEST,rho_estimado_intermedio_file_path),rho_estimado)


        if MAXENT_EXP:
            ti = time.time()
            tipo = "MAXENT_EXP"
            _, rho_estimado = tom.optimization(
                                optimizador=eval('mao.optimizacion_maxent_de_d'))
            lista_mediciones_estimadas = calcular_mediciones_rho_vs_base(rho_estimado,tom.observables_medidos)
            costo = np.linalg.norm(np.array(tom.valores_medios_medidos)-np.array(lista_mediciones_estimadas))**2
            entropy_vn_value =entropy_vn(Qobj(rho_estimado)) 
            costo_por_estado_dic[tipo].append(costo)
            entropy_vn_por_estado_dic[tipo].append(entropy_vn_value)
            fidelidad,trace_distance = mag.comparar_con_estado_real(rho_estimado,lista_estados_from_pickle()[index])#OJO, usa qutip (libreria fue modificada)
            fidelidad_por_estado_dic[tipo].append(fidelidad)
            trace_dist_por_estado_dic[tipo].append(trace_distance)
            tiempo_por_estado_dic[tipo].append(time.time()-ti)
            print(f'{tipo} trace_dist por estado: ',trace_dist_por_estado_dic[tipo])

            NOMBRE_RHO_TARGET = '{}_{}'.format(DESC_ESTADOS,index)
            rho_estimado_intermedio_file_path = '_rho_estimada_{}_{}_{}q_cantobs-{}.npy'.format(
                            tipo.lower(),NOMBRE_RHO_TARGET,NQUBITS,MAX_CANT_OBS)
            np.save(os.path.join(DIR_RHO_ESTIMADO_TEST,rho_estimado_intermedio_file_path),rho_estimado)


    x_labels =list(costo_por_estado_dic.keys())
    y_values_costo = [np.average(costo_por_estado_dic[key]) for key in costo_por_estado_dic.keys()]
    ax.scatter(x_labels,y_values_costo,marker='o',label="Costo euclideo")
    ax.legend()
    #xlabel='Number of Observables ({})'.format(NAME_BASE)
    ylabel='Costo euclideo '
    #plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    title = 'Promedio mediciones exactas vs estimadas (tomografia de {} observables, {} samples, {} qubits)'.format(MAX_CANT_OBS,MAX_CANT_ESTADOS,NQUBITS)
    plt.title(title)
    fig.set_size_inches(width_inches, height_inches)
    default_name_png = 'costo_{}-observables_vqt-vqt_inf-maxent-maxent_exp_Average_{}_Rango{}_{}q.png'.format(
                                 MAX_CANT_OBS,DESC_ESTADOS,RANGO,NQUBITS)
    fig.savefig(os.path.join(DIR_GRAFICOS_TOM_TEST,default_name_png))
    plt.close(fig)
    #################################################
    fig, ax = plt.subplots()
    x_labels =list(entropy_vn_por_estado_dic.keys())
    y_values_entropy = [np.average(entropy_vn_por_estado_dic[key]) for key in entropy_vn_por_estado_dic.keys()]
    ax.scatter(x_labels,y_values_entropy,marker='d',label="Entropy_vn")
    ax.legend()
    ylabel='Entropy VN'
    plt.ylabel(ylabel)
    title = 'Promedio entropy VN (tomografia de {} observables, {} samples, {} qubits)'.format(MAX_CANT_OBS,MAX_CANT_ESTADOS,NQUBITS)
    plt.title(title)
    fig.set_size_inches(width_inches, height_inches)
    default_name_png = 'entropy_vn_{}-observables_vqt-vqt_inf-maxent-maxent_exp_Average_{}_Rango{}_{}q.png'.format(
                                 MAX_CANT_OBS,DESC_ESTADOS,RANGO,NQUBITS)
    fig.savefig(os.path.join(DIR_GRAFICOS_TOM_TEST,default_name_png))
    plt.close(fig)
    #################################################
    fig, ax = plt.subplots()
    x_labels =list(fidelidad_por_estado_dic.keys())
    y_values_fid = [np.average(fidelidad_por_estado_dic[key]) for key in fidelidad_por_estado_dic.keys()]
    ax.scatter(x_labels,y_values_fid,marker='o',label="Fidelidad")
    ax.legend()
    ylabel='Fidelidad'
    plt.ylabel(ylabel)
    title = 'Promedio fidelidad (tomografia de {} observables, {} samples, {} qubits)'.format(MAX_CANT_OBS,MAX_CANT_ESTADOS,NQUBITS)
    plt.title(title)
    fig.set_size_inches(width_inches, height_inches)
    default_name_png = 'fidelidad_{}-observables_vqt-vqt_inf-maxent-maxent_exp_Average_{}_Rango{}_{}q.png'.format(
                                 MAX_CANT_OBS,DESC_ESTADOS,RANGO,NQUBITS)
    fig.savefig(os.path.join(DIR_GRAFICOS_TOM_TEST,default_name_png))
    plt.close(fig)
    #################################################
    fig, ax = plt.subplots()
    x_labels =list(trace_dist_por_estado_dic.keys())
    y_values_tr_dist = [np.average(trace_dist_por_estado_dic[key]) for key in trace_dist_por_estado_dic.keys()]
    ax.scatter(x_labels,y_values_tr_dist,marker='o',label="Trace distance")
    ax.legend()
    ylabel='Trace distance'
    plt.ylabel(ylabel)
    title = 'Promedio trace distance (tomografia de {} observables, {} samples, {} qubits)'.format(MAX_CANT_OBS,MAX_CANT_ESTADOS,NQUBITS)
    plt.title(title)
    fig.set_size_inches(width_inches, height_inches)
    default_name_png = 'trace_dist_{}-observables_vqt-vqt_inf-maxent-maxent_exp_Average_{}_Rango{}_{}q.png'.format(
                                 MAX_CANT_OBS,DESC_ESTADOS,RANGO,NQUBITS)
    fig.savefig(os.path.join(DIR_GRAFICOS_TOM_TEST,default_name_png))
    plt.close(fig)
    ########################################################
    fig, ax = plt.subplots()
    x_labels =list(tiempo_por_estado_dic.keys())
    y_values_tiempo_total = [np.sum(tiempo_por_estado_dic[key]) for key in tiempo_por_estado_dic.keys()]
    ax.scatter(x_labels,y_values_tiempo_total,marker='o',label="Tiempo (seg)")
    ax.legend()
    ylabel='Tiempo Total (seg)'
    plt.ylabel(ylabel)
    title = 'Tiempo total (tomografia de {} observables, {} samples, {} qubits)'.format(MAX_CANT_OBS,MAX_CANT_ESTADOS,NQUBITS)
    plt.title(title)
    fig.set_size_inches(width_inches, height_inches)
    default_name_png = 'tiempo_total_{}-observables_vqt-vqt_inf-maxent-maxent_exp_Average_{}_Rango{}_{}q.png'.format(
                                 MAX_CANT_OBS,DESC_ESTADOS,RANGO,NQUBITS)
    fig.savefig(os.path.join(DIR_GRAFICOS_TOM_TEST,default_name_png))
    plt.close(fig)



