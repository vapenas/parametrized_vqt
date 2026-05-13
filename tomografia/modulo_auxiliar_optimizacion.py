#import pickle
#import itertools as it
import numpy as np
import cvxpy as cp
from numpy import linalg as LA
import qutip as qu
import json
import itertools as it

from load_config import ALPHA,BETA,MAX_ITER_VQT,MAX_ITER_VQT_INF,MAX_ITER_VQT_HIB,MAX_ITER_MAXENT,MAX_ITER_MAXENT_EXP, STOP_LOSS_MAXENT_EXP


def optimizacion_cvxpy_vqt_hib_OLD (base_espacio_simetrico=None, 
                                observables_medidos=None, 
                                valores_medios_medidos=None, 
                                observables_no_medidos=None,
                                ):

    alpha = ALPHA
    beta = BETA
    dim_espacio_sim = len(base_espacio_simetrico)
    cant_obs_medidos = len(observables_medidos)
    cant_obs_no_medidos = len(observables_no_medidos)

    parametros = cp.Variable(dim_espacio_sim)
    rho = sum(parametros[i] * base_espacio_simetrico[i] for i in range(dim_espacio_sim)) 
    
    Delta = cp.Variable(cant_obs_medidos)
    delta = cp.Variable(1)

    ValoresMedidos = list(
        Delta[i] * np.abs (valores_medios_medidos[i])
        >= cp.abs(cp.trace(rho @ observables_medidos[i]) - valores_medios_medidos[i])
        for i in range(cant_obs_medidos))
    ValoresNoMedidos = list(
        delta >= cp.abs(cp.trace(rho @ observables_no_medidos[i]))
        for i in range(cant_obs_no_medidos))
    
    Ineqs = list(  ( Delta[i] >= 0 ) for i in range(cant_obs_medidos)  ) + [delta  >= 0]
    constr1 = [rho >> 0, cp.trace(rho) == 1] + ValoresMedidos +  ValoresNoMedidos +  Ineqs
    Variational = sum(Delta[i] for i in range(cant_obs_medidos))  + beta*delta + alpha*sum(cp.real(cp.trace(rho @ observables_no_medidos[i] )) for i in range(cant_obs_no_medidos))
  
    obj = cp.Minimize(Variational)
    prob = cp.Problem(obj, constr1)
    prob.solve(
            solver=cp.SCS,
            max_iters = MAX_ITER_VQT_HIB,
            verbose=True,
            )

    parametros_estimados = parametros.value

    rho_estimado = sum(parametros_estimados[i] * base_espacio_simetrico[i] for i in range(dim_espacio_sim))

    return parametros_estimados, rho_estimado


def optimizacion_cvxpy_vqt_hib (base_espacio_simetrico=None, 
                                observables_medidos=None, 
                                valores_medios_medidos=None, 
                                observables_no_medidos=None,
                                ):

    def compute_interaction_matrix(observables_list, basis_list):
        # Convert lists to tensors: shapes (N_obs, d, d) and (N_basis, d, d)
        obs_tensor = np.array(observables_list)
        basis_tensor = np.array(basis_list)
        
        return np.einsum('nkl, mlk -> nm', obs_tensor, basis_tensor)

    alpha = ALPHA
    beta = BETA
    dim_espacio_sim = len(base_espacio_simetrico)
    cant_obs_medidos = len(observables_medidos)
    cant_obs_no_medidos = len(observables_no_medidos)

    parametros = cp.Variable(dim_espacio_sim)
    Delta = cp.Variable(cant_obs_medidos,nonneg=True)
    delta = cp.Variable(1,nonneg=True)

    # Shape: (cant_obs_medidos, dim_espacio_sim)
    matrix_medidos = compute_interaction_matrix(observables_medidos, 
                                                base_espacio_simetrico)
    # Shape: (cant_obs_no_medidos, dim_espacio_sim)
    matrix_no_medidos = compute_interaction_matrix(observables_no_medidos, 
                                                   base_espacio_simetrico)

    expectations_medidos = matrix_medidos @ parametros
    expectations_no_medidos = matrix_no_medidos @ parametros

    #ValoresMedidos: Delta * |val| >= |E - val|
    lhs_medidos = cp.multiply(Delta, np.abs(valores_medios_medidos))
    rhs_medidos = cp.abs(expectations_medidos - valores_medios_medidos)
    constraints_medidos = [lhs_medidos >= rhs_medidos]

    #ValoresNoMedidos: delta >= |E|
    constraints_no_medidos = [delta >= cp.abs(expectations_no_medidos)]

    #Calculo del rho:
    base_espacio_simetrico_ar = np.array(base_espacio_simetrico)
    base_espacio_simetrico_f = base_espacio_simetrico_ar.reshape(
                                     base_espacio_simetrico_ar.shape[0],-1).T
    rho = cp.reshape((base_espacio_simetrico_f @ parametros),
              (base_espacio_simetrico_ar.shape[1],base_espacio_simetrico_ar.shape[1])).T

    
    constr_totales = [rho >> 0, cp.trace(rho) == 1] + constraints_medidos + constraints_no_medidos
    Variational = cp.sum(Delta)  + beta*delta + alpha*cp.sum(cp.real(expectations_no_medidos))
  
    obj = cp.Minimize(Variational)
    prob = cp.Problem(obj, constr_totales)
    prob.solve(
            solver=cp.SCS,
            max_iters = MAX_ITER_VQT_HIB,
            verbose=True,
            )

    parametros_estimados = parametros.value

    rho_estimado = np.tensordot(parametros_estimados, base_espacio_simetrico, axes=(0, 0))
    return parametros_estimados, rho_estimado


def optimizacion_cvxpy_vqt_inf_old (base_espacio_simetrico=None, 
                                observables_medidos=None, 
                                valores_medios_medidos=None, 
                                observables_no_medidos=None):

    dim_espacio_sim = len(base_espacio_simetrico)
    cant_obs_medidos = len(observables_medidos)
    cant_obs_no_medidos = len(observables_no_medidos)

    parametros = cp.Variable(dim_espacio_sim)
    rho = sum(parametros[i] * base_espacio_simetrico[i] for i in range(dim_espacio_sim)) 
    
    Delta = cp.Variable(cant_obs_medidos)
    delta = cp.Variable(1)

    ValoresMedidos = list(
        Delta[i] * np.abs (valores_medios_medidos[i])
        >= cp.abs(cp.trace(rho @ observables_medidos[i]) - valores_medios_medidos[i])
        for i in range(cant_obs_medidos))
    ValoresNoMedidos = list(
        delta >= cp.abs(cp.trace(rho @ observables_no_medidos[i]))
        for i in range(cant_obs_no_medidos))
    
    Ineqs = list(  ( Delta[i] >= 0 ) for i in range(cant_obs_medidos)  ) + [delta  >= 0]
    constr1 = [rho >> 0, cp.trace(rho) == 1] + ValoresMedidos +  ValoresNoMedidos +  Ineqs
    Variational = sum(Delta[i] for i in range(cant_obs_medidos))  + delta
  
    obj = cp.Minimize(Variational)
    prob = cp.Problem(obj, constr1)
    prob.solve(
            solver=cp.SCS,
            max_iters = MAX_ITER_VQT_INF,
            verbose=True,
            )

    parametros_estimados = parametros.value

    rho_estimado = sum(parametros_estimados[i] * base_espacio_simetrico[i] for i in range(dim_espacio_sim))

    return parametros_estimados, rho_estimado


def optimizacion_cvxpy_vqt_inf (base_espacio_simetrico=None, 
                                observables_medidos=None, 
                                valores_medios_medidos=None, 
                                observables_no_medidos=None):

    def compute_interaction_matrix(observables_list, basis_list):
        # Convert lists to tensors: shapes (N_obs, d, d) and (N_basis, d, d)
        obs_tensor = np.array(observables_list)
        basis_tensor = np.array(basis_list)
        
        return np.einsum('nkl, mlk -> nm', obs_tensor, basis_tensor)

    dim_espacio_sim = len(base_espacio_simetrico)
    cant_obs_medidos = len(observables_medidos)
    cant_obs_no_medidos = len(observables_no_medidos)

    parametros = cp.Variable(dim_espacio_sim)
    Delta = cp.Variable(cant_obs_medidos,nonneg=True)
    delta = cp.Variable(1,nonneg=True)

    # Shape: (cant_obs_medidos, dim_espacio_sim)
    matrix_medidos = compute_interaction_matrix(observables_medidos, 
                                                base_espacio_simetrico)
    # Shape: (cant_obs_no_medidos, dim_espacio_sim)
    matrix_no_medidos = compute_interaction_matrix(observables_no_medidos, 
                                                   base_espacio_simetrico)

    expectations_medidos = matrix_medidos @ parametros
    expectations_no_medidos = matrix_no_medidos @ parametros

    #ValoresMedidos: Delta * |val| >= |E - val|
    lhs_medidos = cp.multiply(Delta, np.abs(valores_medios_medidos))
    rhs_medidos = cp.abs(expectations_medidos - valores_medios_medidos)
    constraints_medidos = [lhs_medidos >= rhs_medidos]

    #ValoresNoMedidos: delta >= |E|
    constraints_no_medidos = [delta >= cp.abs(expectations_no_medidos)]

    #Calculo del rho:
    base_espacio_simetrico_ar = np.array(base_espacio_simetrico)
    base_espacio_simetrico_f = base_espacio_simetrico_ar.reshape(
                                     base_espacio_simetrico_ar.shape[0],-1).T
    rho = cp.reshape((base_espacio_simetrico_f @ parametros),
              (base_espacio_simetrico_ar.shape[1],base_espacio_simetrico_ar.shape[1])).T

    
    constr_totales = [rho >> 0, cp.trace(rho) == 1] + constraints_medidos + constraints_no_medidos
    Variational = cp.sum(Delta)  + delta
  
    obj = cp.Minimize(Variational)
    prob = cp.Problem(obj, constr_totales)
    prob.solve(
            solver=cp.SCS,
            max_iters = MAX_ITER_VQT_INF,
            verbose=True,
            )

    parametros_estimados = parametros.value


    rho_estimado = np.tensordot(parametros_estimados, base_espacio_simetrico, axes=(0, 0))
    return parametros_estimados, rho_estimado



def optimizacion_cvxpy_vqt_OLD (base_espacio_simetrico=None, 
                            observables_medidos=None, 
                            valores_medios_medidos=None, 
                            observables_no_medidos=None):

    dim_espacio_sim = len(base_espacio_simetrico)
    cant_obs_medidos = len(observables_medidos)
    cant_obs_no_medidos = len(observables_no_medidos)

    parametros = cp.Variable(dim_espacio_sim)
    rho = sum(parametros[i] * base_espacio_simetrico[i] for i in range(dim_espacio_sim)) 
    
    Delta = cp.Variable(cant_obs_medidos)

    ValoresMedidos = list(
        Delta[i] * np.abs (valores_medios_medidos[i])
        >= cp.abs(cp.trace(rho @ observables_medidos[i]) - valores_medios_medidos[i])
        for i in range(cant_obs_medidos))
    Ineqs = list(Delta[i] >= 0 for i in range(cant_obs_medidos))
    constr1 = [rho >> 0, cp.trace(rho) == 1] + ValoresMedidos + Ineqs
    Variational =  sum(Delta[i] for i in range(cant_obs_medidos)) + sum(cp.real(cp.trace(rho @ observables_no_medidos[i] )) for i in range(cant_obs_no_medidos) )



    obj = cp.Minimize(Variational)
    prob = cp.Problem(obj, constr1)
    prob.solve(
            solver=cp.SCS,
            max_iters = MAX_ITER_VQT,
            verbose=True,
            )

    parametros_estimados = parametros.value

    rho_estimado = sum(parametros_estimados[i] * base_espacio_simetrico[i] for i in range(dim_espacio_sim))

    return parametros_estimados, rho_estimado


def optimizacion_cvxpy_vqt(base_espacio_simetrico=None, 
                           observables_medidos=None, 
                           valores_medios_medidos=None, 
                           observables_no_medidos=None):

    def compute_interaction_matrix(observables_list, basis_list):
        # Convert lists to tensors: shapes (N_obs, d, d) and (N_basis, d, d)
        obs_tensor = np.array(observables_list)
        basis_tensor = np.array(basis_list)
        
        return np.einsum('nkl, mlk -> nm', obs_tensor, basis_tensor)

    dim_espacio_sim = len(base_espacio_simetrico)
    cant_obs_medidos = len(observables_medidos)
    cant_obs_no_medidos = len(observables_no_medidos)

    parametros = cp.Variable(dim_espacio_sim)
    Delta = cp.Variable(cant_obs_medidos, nonneg=True)

    # Shape: (cant_obs_medidos, dim_espacio_sim)
    matrix_medidos = compute_interaction_matrix(observables_medidos, 
                                                base_espacio_simetrico)
    # Shape: (cant_obs_no_medidos, dim_espacio_sim)
    matrix_no_medidos = compute_interaction_matrix(observables_no_medidos, 
                                                   base_espacio_simetrico)

    expectations_medidos = matrix_medidos @ parametros
    expectations_no_medidos = matrix_no_medidos @ parametros

    # ValoresMedidos: Delta * |val| >= |E - val|
    lhs_medidos = cp.multiply(Delta, np.abs(valores_medios_medidos))
    rhs_medidos = cp.abs(expectations_medidos - valores_medios_medidos)
    constraints_medidos = [lhs_medidos >= rhs_medidos]

    # Calculo del rho:
    base_espacio_simetrico_ar = np.array(base_espacio_simetrico)
    base_espacio_simetrico_f = base_espacio_simetrico_ar.reshape(
                                     base_espacio_simetrico_ar.shape[0],-1).T
    rho = cp.reshape((base_espacio_simetrico_f @ parametros),
              (base_espacio_simetrico_ar.shape[1],base_espacio_simetrico_ar.shape[1])).T

    constr_totales = [rho >> 0, cp.trace(rho) == 1] + constraints_medidos
    Variational = cp.sum(Delta) + cp.sum(cp.real(expectations_no_medidos))

    obj = cp.Minimize(Variational)
    prob = cp.Problem(obj, constr_totales)
    prob.solve(
            solver=cp.SCS,
            max_iters = MAX_ITER_VQT,
            verbose=True,
            )

    parametros_estimados = parametros.value

    rho_estimado = np.tensordot(parametros_estimados, base_espacio_simetrico, axes=(0, 0))
    return parametros_estimados, rho_estimado




#NOTE: observables_no_medidos no se usa dentro de optimizacion_cvxpy_maxent al final.
def optimizacion_cvxpy_maxent (base_espacio_simetrico=None, 
                               observables_medidos=None, 
                               valores_medios_medidos=None, 
                               observables_no_medidos=None):

    dim_espacio_sim = len(base_espacio_simetrico)
    cant_obs_medidos = len(observables_medidos)
    cant_obs_no_medidos = len(observables_no_medidos)

    parametros = cp.Variable(dim_espacio_sim)
    rho = sum(parametros[i] * base_espacio_simetrico[i] for i in range(dim_espacio_sim)) 
    
    Delta = cp.Variable(cant_obs_medidos)

    ValoresMedidos = list(
        Delta[i] * np.abs (valores_medios_medidos[i])
        >= cp.abs(cp.trace(rho @ observables_medidos[i]) - valores_medios_medidos[i])
        for i in range(cant_obs_medidos))
    Ineqs = list(Delta[i] >= 0 for i in range(cant_obs_medidos))
    constr1 = [rho >> 0, cp.trace(rho) == 1] + ValoresMedidos + Ineqs
    Variational = 10*sum(Delta[i] for i in range(cant_obs_medidos)) -  cp.von_neumann_entr(rho)  



    obj = cp.Minimize(Variational)
    prob = cp.Problem(obj, constr1)
    prob.solve(
            solver=cp.SCS,
            max_iters = MAX_ITER_MAXENT,
            verbose=True,
            )

    parametros_estimados = parametros.value

    rho_estimado = sum(parametros_estimados[i] * base_espacio_simetrico[i] for i in range(dim_espacio_sim))

    return parametros_estimados, rho_estimado



def corregir_autoval_neg (rho_estimado):

    eigenval, eigenvec = LA.eig(rho_estimado)
    eigenval = np.real(eigenval)
    eigenval = (eigenval>= 0)*eigenval
    eigenval = eigenval/np.sum(eigenval)
    rho_corregido = eigenvec@ np.diag(eigenval) @eigenvec.T.conjugate() 
    
    return rho_corregido



#####OPTIMIZADOR exp:
from qutip import Qobj,ket2dm,w_state, fidelity
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Paquetes de JAX y otros
from jax import numpy as jnp
import qutip_jax
import optax
import jax

# Configuro el Optimizador
jax.config.update("jax_enable_x64", True)
opt = optax.adam(learning_rate=0.3)

# Defino las funciones a usar
@jax.jit
#from functools import partial
#@partial(jax.jit,static_argnames=['caso_hermitico'])
def rho(lams, Obs,caso_hermitico=True):
    lamsObs = 0
    for lam, Ob in zip(lams, Obs):
        lamOb = lam*Ob
        lamsObs+=lamOb
    E=lamsObs.expm()
    E = E.to('jax')
    Etr= E.tr().real
    return E/Etr

@jax.jit
#from functools import partial
#@partial(jax.jit,static_argnames=['caso_hermitico'])
def cost(lams, Obs, mvals,caso_hermitico=True):
    rhol=rho(lams, Obs,caso_hermitico=caso_hermitico)
    difs=[1000*((rhol*Obs[i]).tr()-mvals[i])**2 for i in range(len(Obs))]
    return jnp.sum(jnp.real(jnp.array(difs)))



@jax.jit
def optimization_jit(params, opt_state, Obs, mvals, max_iter, stop_threshold=1e-7, print_training=False, caso_hermitico=True):
    """
    Optimization using jax.lax.while_loop for early stopping.
    """

    # The state of our loop will be a tuple: (params, opt_state, current_cost, iteration_count)
    # We initialize the cost to infinity to ensure the loop runs at least once.
    init_val = (params, opt_state, jnp.inf, 0)

    def cond_fun(state):
        """Continue looping if cost > threshold AND iteration < max_iter."""
        _, _, cost_val, i = state
        # Using jnp.logical_and is the standard way to combine conditions in JAX.
        return jnp.logical_and(cost_val > stop_threshold, i < max_iter)

    def body_fun(state):
        """Performs one optimization step."""
        params, opt_state, _, i = state

        # Calculate cost and gradients
        cost_val, grads = jax.value_and_grad(cost, argnums=0,holomorphic=False)(params, Obs, mvals, caso_hermitico)

        # Update optimizer state and parameters
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        # Debug printing
        def print_fn():
            jax.debug.print("Step: {i} Loss: {cost_val}", i=i, cost_val=cost_val)
        jax.lax.cond((jnp.mod(i, 5) == 0) & print_training, print_fn, lambda: None)

        # Return the updated state for the next iteration
        return (params, opt_state, cost_val, i + 1)

    # Run the while_loop
    final_state = jax.lax.while_loop(cond_fun, body_fun, init_val)

    # Unpack the final parameters from the state
    final_params, _, final_cost, final_iter = final_state

    # Optional: print final status
    def print_final():
        jax.debug.print("Optimization finished at iteration {i} with final cost {cost}", i=final_iter, cost=final_cost)
    jax.lax.cond(print_training, print_final, lambda: None)

    return final_params



def optimizacion_maxent_de_d(base_espacio_simetrico=None,
                                 observables_medidos=None,
                                 valores_medios_medidos=None,
                                 observables_no_medidos=None):
    ObsMed = observables_medidos
    mvals = valores_medios_medidos
    try:
        assert len(ObsMed)==len(mvals)
    except:
        raise Exception("La cantidad de observables_medidos y valores_medios_medidos tiene que coincidir.")

    # Configuro el Optimizador
    #jax.config.update("jax_enable_x64", True)
    #opt = optax.adam(learning_rate=0.3)

    Nq = int(np.log2(observables_medidos[0].shape[0]))
    #Nobs = len(observables_medidos)

    Obs=[Qobj(pr,dims=[[2]*Nq, [2]*Nq]).to("jax") for pr in ObsMed]

    params = jnp.zeros(len(Obs))
    opt_state = opt.init(params)
    #
    max_iter=MAX_ITER_MAXENT_EXP+1
    params_finales=optimization_jit(params, opt_state, Obs, mvals,max_iter,stop_threshold=STOP_LOSS_MAXENT_EXP, print_training=True)
    rho_op=rho(params_finales, Obs)

    jax.clear_caches()

    return params_finales, rho_op




