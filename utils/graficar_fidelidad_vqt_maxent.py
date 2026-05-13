import os
import matplotlib.pyplot as plt
import numpy as np



def graficar_fidelidades(nqubits,
                         cant_obs_list,
                         fidelidad_vqt,
                         fidelidad_vqt_inf,
                         fidelidad_maxent,
                         fidelidad_maxent_exp,
                         dir_path,
                         title=None,
                         circuit_name=None,
                         xlabel=None,
                         ylabel=None,
                         default_name_png=None):
    #fidelity_list = np.load(fidelity_npy_file_path)
    #GRAFICOS
    # Define the size of the figure in pixels
    qc_name = circuit_name
    if type(fidelidad_vqt) == str:
        try:
            os.path.isfile(fidelidad_vqt)
            fidelidad_vqt = np.load(fidelidad_vqt)
        except:
            raise Exception('El file_path de fidelidad_vqt no se encuentra.')
    if type(fidelidad_vqt_inf) == str:
        try:
            os.path.isfile(fidelidad_vqt_inf)
            fidelidad_vqt_inf = np.load(fidelidad_vqt_inf)
        except:
            raise Exception('El file_path de fidelidad_vqt_inf no se encuentra.')
    if type(fidelidad_maxent) == str:
        try:
            os.path.isfile(fidelidad_maxent)
            fidelidad_maxent = np.load(fidelidad_maxent)
        except:
            raise Exception('El file_path de fidelidad_maxent no se encuentra.')
    if type(fidelidad_maxent_exp) == str:
        try:
            os.path.isfile(fidelidad_maxent_exp)
            fidelidad_maxent_exp = np.load(fidelidad_maxent_exp)
        except:
            raise Exception('El file_path de fidelidad_maxent_exp no se encuentra.')
    try:
        os.path.isdir(dir_path)
    except:
        raise Exception('dir_path must be a directory.')
    width_pixels = 1400
    height_pixels = 1000
    # Set the dpi (dots per inch)
    dpi = 100
    # Convert pixels to inches
    width_inches = width_pixels / dpi
    height_inches = height_pixels / dpi
    fig, ax = plt.subplots()
    #ax.plot([i in range(iterations)],fidelity_list_vqt,marker='o')
    ax.scatter(cant_obs_list, fidelidad_vqt,   label='VQT',marker='d')
    ax.scatter(cant_obs_list, fidelidad_vqt_inf,   label='VQT_inf',marker='s')
    ax.scatter(cant_obs_list, fidelidad_maxent,label='MaxEnt',marker='o')
    ax.scatter(cant_obs_list, fidelidad_maxent_exp,label='MaxEntExp',marker='*')
    #legend=['VQT','MaxEnt']
    #ax.legend(legend)
    ax.legend()
    if xlabel==None:
        xlabel='Number of observables'
    if ylabel==None:
        ylabel='Fidelity'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title==None:
        title = 'Fidelity vs Number of Observables ({} qubits)'.format(nqubits)
    plt.title(title)
    fig.set_size_inches(width_inches, height_inches)
    if default_name_png == None:
        default_name_png = 'fidelidades_vqt-vqt_inf-maxent-maxent_exp_{}_{}q.png'.format(qc_name,nqubits)
    else:
        pass
    fig.savefig(os.path.join(dir_path,default_name_png))
    plt.close(fig)
