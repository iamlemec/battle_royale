import pytoml as toml
import numpy as np
import pandas as pd
from collections import OrderedDict
import scipy.interpolate as interp
import scipy.special as special
import scipy.optimize as opt

from mectools.bundle import Bundle
from mectools.endy import random_vec

##
## tools
##

def load_toml(fname):
    return toml.load(open(fname), object_pairs_hook=OrderedDict)

def save_toml(d, fname):
    toml.dump(d, open(fname, 'w+'))

def file_or_dict(fod):
    return Bundle(load_toml(fod)) if type(fod) is str else fod

def dict_to_vec(d):
    return np.array([x for x in d.values()])

def vec_to_dict(vec, names):
    return {k: v for k, v in zip(names, vec)}

def pprint(x):
    print(' '.join([f'{z:+0.8f}' for z in x]))

##
## special
##

def tlog(x, theta):
    if theta == 0:
        return np.log(x)
    else:
        return (x**theta-1)/theta

##
## model
##

class Model:
    def __init__(m, alg='config/alg.toml', par='config/par.toml', var='config/var.toml', pol='config/pol.toml'):
        if alg is not None:
            m.load_algpar(alg)
        if par is not None:
            m.load_params(par)
        if var is not None:
            m.load_eqvars(var)
        if pol is not None:
            m.load_policy(pol)

    def load_algpar(m, alg):
        m.alg = file_or_dict(alg)
        m.__dict__.update(m.alg)

        # q grid
        m.q_bins = np.linspace(m.q_min, m.q_max, m.N+1)
        m.q_grid = 0.5*(m.q_bins[:-1]+m.q_bins[1:])
        m.q_width = np.diff(m.q_bins)

        # simulation reps
        m.R_burn = int(m.T_burn/m.s_delt)
        m.R_rec = int(m.T_rec/m.s_delt)

    def load_params(m, par):
        m.par = file_or_dict(par)
        m.__dict__.update(m.par)

        # derived
        m.ehat = (1-m.eps)/m.eps
        m.tlam = np.log(1+m.lam)
        m.qe_grid = m.q_grid**m.ehat

        # interp prev q
        m.x_grow = m.lam/(m.q_grid**(1-m.beta))
        m.q_x_up = m.q_grid*(1+m.x_grow)
        m.q_x_down = interp.interp1d(m.q_x_up, m.q_grid, bounds_error=False, fill_value=0)(m.q_grid)

    def load_eqvars(m, var):
        m.var = file_or_dict(var)
        m.__dict__.update(m.var)

        # creative destruction
        m.vbar = m.chi*m.w
        m.x = (m.vbar/(m.w*m.kappa*m.eta))**(1/(m.eta-1))
        m.tau = m.x + m.e

        m.x_cost = m.kappa*m.x**m.eta
        m.x_value = m.x*m.vbar - m.w*m.x_cost

        # find qbar
        def q_func(q):
            v_bar = (1-m.eps)*q**(m.ehat-1)/(m.rho+m.tau+(1-m.alpha)*m.g)
            z_opt = m.g*q**(1-m.alpha)
            v_lft = m.w*m.gamma*m.eta*z_opt**(m.eta-1)
            v_rgt = q**m.alpha*v_bar
            return v_lft - v_rgt
        m.qbar, = opt.fsolve(q_func, 1)

    def load_policy(m, pol):
        m.pol = file_or_dict(pol)
        m.__dict__.update(m.pol)

    def value_update(m):
        m.z_grow = m.z_vals/m.q_grid**(1-m.alpha)
        m.z_sig = special.expit(m.sig_fact*(m.z_grow-m.g))

        disc = m.rho + m.tau
        m.z_cost = m.gamma*m.z_vals**m.eta
        u_term = m.eps*(m.q_grid**m.ehat) - m.w*m.z_cost + m.x_value

        q_next = m.q_grid*(1+m.v_delt*(m.z_grow-m.g))
        v_next = interp.interp1d(m.q_grid, m.v_vals, fill_value='extrapolate')(q_next)
        v_prim = m.v_delt*u_term + (1/(1+m.v_delt*disc))*v_next

        dv_base = np.diff(m.v_vals)/np.diff(m.q_grid)
        dv_lo = np.r_[dv_base[0], dv_base]
        dv_hi = np.r_[dv_base, dv_base[-1]]
        dv_vals = m.z_sig*dv_hi + (1-m.z_sig)*dv_lo

        z_gain = dv_vals*(m.q_grid**m.alpha)
        z_prim = (np.maximum(0, z_gain)/(m.w*m.gamma*m.eta))**(1/(m.eta-1))

        return v_prim, z_prim

    def value_solve(m, output=False):
        for i in range(m.v_K):
            v_prim, z_prim = m.value_update()

            v_err = np.max(np.abs(v_prim-m.v_vals))
            z_err = np.max(np.abs(z_prim-m.z_vals))
            err = np.maximum(v_err, z_err)

            m.v_vals[:] = (1-m.v_upd)*m.v_vals + m.v_upd*v_prim
            m.z_vals[:] = (1-m.v_upd)*m.z_vals + m.v_upd*z_prim

            if output and i % m.v_per == 0:
                print(i, v_err, z_err)
            if err < m.v_tol:
                if output:
                    print(f'converged: {v_err=}, {z_err=}')
                break

    def dist_update(m):
        f_base = np.diff(m.F_vals)/np.diff(m.q_grid)
        f_lo = np.r_[f_base[0], f_base]
        f_hi = np.r_[f_base, f_base[-1]]
        m.f_vals = (1-m.z_sig)*f_hi + m.z_sig*f_lo

        F_down = interp.interp1d(m.q_grid, m.F_vals, bounds_error=False, fill_value=0)(m.q_x_down)
        F_diff = (m.g-m.z_grow)*m.q_grid*m.f_vals - m.tau*(m.F_vals - F_down)
        F_prim = np.clip(m.F_vals + m.d_delt*F_diff, 0, 1)

        return F_prim

    def dist_solve(m, output=False):
        for i in range(m.d_K):
            F_prim = m.dist_update()

            F_err = np.max(np.abs(F_prim-m.F_vals))

            m.F_vals[:] = (1-m.d_upd)*m.F_vals + m.d_upd*F_prim

            if output and i % m.d_per == 0:
                print(i, F_err)
            if F_err < m.d_tol:
                if output:
                    print(f'converged: {F_err=}')
                break

        # get probability mass
        m.q_dist = m.f_vals*m.q_width

    def q_expect(m, x):
        return np.sum(x*m.q_dist)

    def find_growth(m):
        xstep = m.q_expect(tlog(1+m.x_grow, m.ehat)*m.qe_grid)
        m.g_z = m.q_expect(m.z_grow*m.qe_grid)
        m.g_x = m.x*xstep
        m.g_e = m.e*xstep
        return m.g_z + m.g_x + m.g_e

    def find_labor(m):
        m.lab_prod = (1-m.eps)/m.w
        m.lab_int = m.q_expect(m.z_cost)
        m.lab_ext = m.x_cost
        m.lab_ent = m.chi*m.e
        return m.lab_prod + m.lab_int + m.lab_ext + m.lab_ent

    def find_vbar(m):
        vqbar = interp.interp1d(m.q_grid, m.v_vals, fill_value='extrapolate')(m.q_x_up)
        return m.q_expect(vqbar)

    def guess(m):
        m.v_vals = np.ones(m.N)
        m.z_vals = m.g*np.ones(m.N)
        m.F_vals = np.linspace(0, 1, m.N)

    def eqfunc(m, output=False):
        m.value_solve(output=output)
        m.dist_solve(output=output)

        vbar_new = m.find_vbar()
        g_new = m.find_growth()
        labor = m.find_labor()

        # number of firms
        m.F = (m.e/m.x)*np.log(m.tau/m.e)

        return {
            'vbar': vbar_new - m.vbar,
            'growth': g_new - m.g,
            'labor': 1 - labor
        }

    def solve(m, reset=True, **kwargs):
        # objective that takes a vector
        names = list(m.var.keys())
        def eqeval(x):
            m.load_eqvars(vec_to_dict(x, names))
            diff = m.eqfunc()
            ret = list(diff.values())
            pprint(ret)
            return ret

        # reset functional guesses
        if reset:
            m.guess()

        # solve it
        x0 = dict_to_vec(m.var)
        x1 = opt.fsolve(eqeval, x0, **kwargs)
        eq = vec_to_dict(x1, names)

        # final numbers
        m.load_eqvars(eq)
        m.eqfunc(output=True)

        return eq

    def simulate(m, seed=None, fuzz=False):
        if seed == None: seed = m.seed
        state = np.random.RandomState(seed)

        # interpolators
        gz_func = interp.interp1d(m.q_grid, m.z_grow, fill_value='extrapolate')
        gt_func = interp.interp1d(m.q_grid, m.x_grow, fill_value='extrapolate')
        z_cost = interp.interp1d(m.q_grid, m.z_cost, fill_value='extrapolate')

        # product distribution
        nprod = np.arange(1, m.P_max)
        tmu = (m.e/m.tau)*(m.x/m.tau)**(nprod-1)
        mu = tmu/nprod
        mu /= np.sum(mu)
        cmu = np.cumsum(mu)

        # make firms
        F = int(m.F*m.P)
        fsize = random_vec(cmu, F, state=state) + 1
        firm = np.concatenate([[f]*s for f, s in enumerate(fsize)])
        qual = m.q_grid[random_vec(m.F_vals, m.P, state=state)]

        # make sure size P
        P0 = len(firm)
        print(P0, m.P)
        if P0 > m.P:
            firm = firm[:m.P]
        elif P0 < m.P:
            firm = np.r_[firm, state.randint(F, size=(m.P-P0))]

        # track max firm id
        max_fid = np.max(firm)

        # iterate
        panel = []
        for t in range(m.R_burn+m.R_rec):
            # external
            x_jump, = np.nonzero(state.rand(m.P) < m.s_delt*m.x)
            x_idxs = state.randint(m.P, size=len(x_jump))
            firm[x_jump] = firm[x_idxs]
            qual[x_jump] *= 1 + gt_func(qual[x_jump])

            # entry
            e_jump, = np.nonzero(state.rand(m.P) < m.s_delt*m.e)
            e_firm = max_fid + 1 + np.arange(len(e_jump))
            firm[e_jump] = e_firm
            qual[e_jump] *= 1 + gt_func(qual[e_jump])

            # internal + normalize
            qual *= 1 + m.s_delt*(gz_func(qual)-m.g)

            # track max firm id
            max_fid = np.max(firm)

            # record
            if t == m.R_burn:
                date = 0.0
            if t >= m.R_burn:
                equal = qual**m.ehat

                prods = pd.DataFrame({'firm': firm, 'qual': qual})
                prods['revenue'] = equal
                prods['wages'] = (1-m.eps)*equal
                prods['rnd'] = m.w*(z_cost(qual) + m.x_cost)
                prods['employ'] = prods['wages']/m.w
                prods['profit'] = prods['revenue'] - prods['wages'] - prods['rnd']
                prods['products'] = 1

                firms = prods.groupby('firm').sum().reset_index()
                firms['date'] = date
                panel.append(firms)

                date += m.s_delt

        # stack it up
        panel = pd.concat(panel, axis=0).reset_index(drop=True)

        # remap firm ids
        uniq_firms = pd.Index(panel['firm'].unique(), name='firm')
        firm_map = pd.Series(np.arange(len(uniq_firms)), name='firm_id', index=uniq_firms)
        panel = panel.join(firm_map, on='firm').drop(columns='firm')

        # fuzz if desired
        if fuzz:
            panel = m.fuzz(panel)

        return panel

    def fuzz(m, panel):
        N = len(panel)
        panel = panel.copy()

        err_prop = {
            'revenue': 0.05,
            'employ': 0.05,
            'rnd': 0.1,
            'wages': 0.05,
            'profit': 0.1
        }

        err_add = {
            'revenue': 0.05,
            'employ': 0.05,
            'rnd': 0.02,
            'wages': 0.05,
            'profit': 0.03
        }

        for col, prop in err_prop.items():
            panel[col] *= np.exp(prop*np.random.randn(N))

        for col, add in err_add.items():
            panel[col] += np.maximum(0, add*np.random.randn(N))

        return panel
