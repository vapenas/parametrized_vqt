import os
import re

import numpy as np


def build_index_dictionary(file_list,tipo_opt):
    """
    Ejemplo:

    file_list = [
        'fidelidad_vqt_descrp_0_mincantobs-1_maxcantobs-8_4q.npy',
        'fidelidad_vqt_descrp_0_mincantobs-9_maxcantobs-10_4q.npy',
        'trace_dist_vqt_descrp_5_mincantobs-1_maxcantobs-5_4q.npy',
        'trace_dist_vqt_descrp_5_mincantobs-7_maxcantobs-9_4q.npy',
    
    ]

    tipo_opt='vqt'

    return:  
           {'fidelidad': {'0': [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10]]},
            'trace_dist': {'5': [[1, 2, 3, 4, 5], [7, 8, 9]],
             }}

    """
    lista_full = {'fidelidad':[],'trace_dist':[]}
    index_and_obs_dicc_full = {'fidelidad':[],'trace_dist':[]}
    for filename in file_list:
        if ('fidelidad' in filename) and ('_{}_'.format(tipo_opt) in filename):
            if tipo_opt=='vqt' and (('vqt_inf' in filename) or ('vqt_hib' in filename)):
                pass
            elif tipo_opt=='maxent' and ('maxent_exp' in filename):
                pass
            else:
                lista_full['fidelidad'].append(filename)
        elif ('trace_dist' in filename) and ('_{}_'.format(tipo_opt) in filename):
            if tipo_opt=='vqt' and (('vqt_inf' in filename) or ('vqt_hib' in filename)):
                pass
            elif tipo_opt=='maxent' and ('maxent_exp' in filename):
                pass
            else:
                lista_full['trace_dist'].append(filename)

    # Regex to extract index, min, and max safely
    # Pattern looks for: ..._descrp_{INDEX}_mincantobs-{MIN}_maxcantobs-{MAX}_...
    pattern = r"_(\d+)_mincantobs-(\d+)_maxcantobs-(\d+)"

    if lista_full['fidelidad']!=[]:
        file_list = lista_full['fidelidad']
        index_and_obs_dicc = {}
        for filename in file_list:
            match = re.search(pattern, filename)
            if match:
                idx_str = match.group(1) # The index (e.g., '0')
                min_obs = int(match.group(2))
                max_obs = int(match.group(3))
                
                # Create the list range [min, ..., max]
                range_list = list(range(min_obs, max_obs + 1))
                
                if idx_str not in index_and_obs_dicc:
                    index_and_obs_dicc[idx_str] = []
                
                index_and_obs_dicc[idx_str].append(range_list)
        
        # Sort the lists within the dictionary values based on their first element
        # This ensures we process [1..5] before [7..9]
        for key in index_and_obs_dicc:
            index_and_obs_dicc[key].sort(key=lambda x: x[0])

        index_and_obs_dicc_full['fidelidad']=index_and_obs_dicc

    if lista_full['trace_dist']!=[]:
        file_list = lista_full['trace_dist']
        index_and_obs_dicc = {}
        for filename in file_list:
            match = re.search(pattern, filename)
            if match:
                idx_str = match.group(1) # The index (e.g., '0')
                min_obs = int(match.group(2))
                max_obs = int(match.group(3))
                
                # Create the list range [min, ..., max]
                range_list = list(range(min_obs, max_obs + 1))
                
                if idx_str not in index_and_obs_dicc:
                    index_and_obs_dicc[idx_str] = []
                
                index_and_obs_dicc[idx_str].append(range_list)
        
        # Sort the lists within the dictionary values based on their first element
        # This ensures we process [1..5] before [7..9]
        for key in index_and_obs_dicc:
            index_and_obs_dicc[key].sort(key=lambda x: x[0])

        index_and_obs_dicc_full['trace_dist']=index_and_obs_dicc
        
    return index_and_obs_dicc_full



def get_fidelity_list(index, index_and_obs_dicc_full, MAX,tipo_opt,descrp,nqubits,metrica='fidelidad',folder_location=None):
    """
    Constructs the concatenated fidelity list for a specific index.
    
    Args:
        index (str): The index key (e.g., '0').
        index_and_obs_dicc_full (dict): The dictionary built in Part 1.
        MAX (int): The maximum length of the final array.
        abs_file_location: Locacion de la carpeta donde se encuentra el archivo .npy
                           desde el root folder del programa.
        
    Returns:
        list: The concatenated list containing data and np.nan.
    """
    
    # 1. Retrieve the list of ranges for this index
    index_and_obs_dicc=index_and_obs_dicc_full[metrica]
    if index not in index_and_obs_dicc:
        return [np.nan] * MAX
    
    # Ensure ranges are sorted by start value (just in case)
    ranges = sorted(index_and_obs_dicc[index], key=lambda x: x[0])
    
    final_list = []
    
    # expected_next_pos tracks the next number we expect to append (1-based index)
    expected_next_pos = 1
    
    for r_list in ranges:
        min_cant = r_list[0]
        max_cant = r_list[-1]
        
        # --- Reconstruct Filename ---
        # We assume the file exists if it is in the dictionary.
        filename = "{}_{}_{}_{}_mincantobs-{}_maxcantobs-{}_{}q.npy".format(
            metrica,tipo_opt, descrp, index, min_cant, max_cant, nqubits
        )
        
        # Load the file
        # Note: In a real scenario, ensure files exist or use try/except.
        try:
            # We use tolist() immediately as we are working with python lists for concatenation
            if folder_location==None:
                current_data = np.load(filename).tolist()
            else:
                current_data = np.load(os.path.join(folder_location,filename)).tolist()
        except FileNotFoundError:
            # Fallback for demonstration if files aren't physically present during testing
            print(f"Warning: File {filename} not found. Folder location: {folder_location}")
            raise Exception(f"Warning: File {filename} not found. Folder location: {folder_location}")
            #return []

        # --- Logic 1 & 3: Handle Gaps ---
        # If the file starts at 6 but we are at 1, we need 5 NaNs.
        if min_cant > expected_next_pos:
            gap_size = min_cant - expected_next_pos
            final_list += [np.nan] * gap_size
            expected_next_pos = min_cant

        # --- Logic 2: Handle Overlaps ---
        # We need to determine where to start slicing the current_data.
        # If expected_next_pos is 9, and file starts at 6 (min_cant),
        # we skip the first (9 - 6) = 3 elements.
        
        slice_start_index = 0
        if expected_next_pos > min_cant:
            slice_start_index = expected_next_pos - min_cant
            
        # Append the relevant part of the data
        final_list += current_data[slice_start_index:]
        
        # Update expected_next_pos to one past the end of this file
        expected_next_pos = max_cant + 1

    # --- Logic 5: Constraints on MAX ---
    
    # If the list is shorter than MAX, pad with NaNs
    if len(final_list) < MAX:
        padding = MAX - len(final_list)
        final_list += [np.nan] * padding
        
    # If the list is longer than MAX, truncate it
    if len(final_list) > MAX:
        final_list = final_list[:MAX]
        
    return final_list


