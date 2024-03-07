import cvxopt
from qnregipm import IneqConstProblem
from numeric_kernel.gp import GeometricProgram
from numeric_kernel.plot_functions import *
from numeric_kernel.generate_all_plots import generate_all_plots

# This code was developed for the master thesis of Ole Gildemeister,
# written at the Institute of Mathematics and Image Computing, University of Luebeck.
# Code and thesis are available at https://github.com/gildol22/code-master-thesis

# This script generates, solves and plots a large-scale geometric program,
# corresponds to results presented in Section 5.4


if __name__ == "__main__":
    plots = True  # generate and save all plots?
    cvx_bool = False  # comparison with cvxopt results (only seems to work if solution is not 0)?

    # set problem parameters
    n = 5000
    m = 500
    k_obj = 50
    k_ineq = 5

    # set solver parameters
    eps0 = 1
    mu0 = 1
    beta = .95
    gamma = 1.1
    tol_outer = 1e-3
    C = 100

    # set solver settings
    structured = True
    matrix_free = True
    proximal = False

    # build problem
    gp = GeometricProgram(n=n, m=m, k_obj=k_obj, k_ineq=k_ineq, c_min=3)
    np.random.seed(2206)
    gp.generate_problem()
    x0 = gp.x0
    f = lambda x: gp.evaluate_objective(x)
    g = lambda x: gp.evaluate_constraints(x)

    # initialise and solve problem
    p = IneqConstProblem(f=f, g=g, x0=x0, eps0=eps0, mu0=mu0, beta=beta, gamma=gamma, tol_outer=tol_outer, C=C,
                         outer_max_iter=500, lmax_lbfgs=5, rho_ls=.5, c_ls=1e-5, inner_max_iter=100,
                         structured_lbfgs=structured, proximal_point=proximal, display_outer=True, display_inner=False,
                         tol_minres=1e-10, matrix_free=matrix_free)

    print(f"\n\n### Test interior point method for inequality constrained geometric program of size "
          f"n = {n}, m = {m}, objecive depth = {k_obj}, constraint depth = {k_ineq} and solver set to "
          f"matrix-{'free' if matrix_free else 'based'} {'structured' if structured else 'standard'} l-BFGS and"
          f"{'proximal point' if proximal else 'Tikhonov'} regularisation###")
    p.solve_regularised_ipm()
    print('Done.')

    # save problem and results
    np.save('../results/gp_large_log_outer', p.log_outer)
    np.save('../results/gp_large_log_inner', p.log_inner)
    np.save('../results/gp_large_gp', gp)

    # get results
    fstar = np.array(p.log_outer['val_list']).min()
    xstar = p.log_outer['x_list'][np.argmin(np.array(p.log_outer['val_list']))]

    # compare to result from cvxopt?
    if cvx_bool:
        cvx_K = [k_obj] * 2 + [k_ineq] * m
        cvx_F = np.vstack((gp.F, -gp.F, gp.G.reshape((gp.m*gp.k_ineq, gp.n))))
        cvx_g = np.hstack((gp.b, -gp.b + 1 - 2 * np.log(gp.k_obj), gp.c.flatten() + gp.c_min))
        cvx_gpsolver = cvxopt.solvers.gp(cvx_K, cvxopt.matrix(cvx_F), cvxopt.matrix(cvx_g))
        cvx_xstar = np.array(cvx_gpsolver['x']).flatten()
        cvx_fstar = cvx_gpsolver['primal objective']

        if fstar > cvx_fstar:
            print(f"-------------------\n"
                  f"Own solution is worse than the one from CVXOPT by difference {fstar - cvx_fstar} in function value and "
                  f"{np.linalg.norm(xstar - cvx_xstar)} in solution norm")
            fstar = cvx_fstar
            xstar = cvx_xstar
        else:
            print(f"+++++++++++++++++++\n"
                  f"Own solution is better than the one from CVXOPT by difference {cvx_fstar - fstar} in function value and "
                  f"{np.linalg.norm(xstar - cvx_xstar)} in solution norm")

    fstar = 1.0

    # plot function behaviour
    if plots:
        generate_all_plots(p, 'gp_large', fstar=fstar, xstar=xstar, cliplast=2, show=False)

        # for plot_outer_fvals_update a clip in the beginning is required to properly see the results
        fig, ax = plot_outer_fvals_update(p, clipfirst=3, cliplast=2, fstar=fstar)
        fig.savefig('../figures/' + 'gp_large' + '_' + 'plot_outer_fvals_update')


