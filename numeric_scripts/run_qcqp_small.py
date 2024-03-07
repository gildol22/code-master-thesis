import cvxpy as cp
from qnregipm import IneqConstProblem
from numeric_kernel.qcqp import QuadraticProgram
from numeric_kernel.plot_functions import *
from numeric_kernel.generate_all_plots import generate_all_plots

# This code was developed for the master thesis of Ole Gildemeister,
# written at the Institute of Mathematics and Image Computing, University of Luebeck.
# Code and thesis are available at https://github.com/gildol22/code-master-thesis

# This script generates, solves and plots a small-scale quadratically constrained quadratic program,
# corresponds to results presented in Section 5.3


if __name__ == "__main__":
    plots = True  # generate and save all plots?
    cvx_bool = True  # comparison with cvxpy results (only seems to work if solution is not 0)?

    # set problem parameters
    n = 10
    m = 5

    # set solver parameters
    eps0 = 1
    mu0 = 1
    beta = .9
    gamma = 1.1
    tol_outer = 1e-6
    C = 1

    # set solver settings
    structured = True
    proximal = False
    matrix_free = False

    # build problem
    qp = QuadraticProgram(n=n, m=m, density=n**(-3/4))
    np.random.seed(2206)
    qp.generate_problem()

    x0 = qp.x0
    f = lambda x: qp.evaluate_objective(x)
    g = lambda x: qp.evaluate_constraints(x)

    # initialise and solve problem
    p = IneqConstProblem(f=f, g=g, x0=x0, eps0=eps0, mu0=mu0, beta=beta, gamma=gamma, tol_outer=tol_outer, C=C,
                         outer_max_iter=500, lmax_lbfgs=5, rho_ls=.5, c_ls=1e-5, inner_max_iter=300,
                         structured_lbfgs=structured, proximal_point=proximal, display_outer=True, display_inner=False,
                         tol_minres=1e-10, matrix_free=matrix_free)

    print(f"\n\n### Test interior point method for quadratically inequality constrained quadratic program of size "
          f"n = {n}, m = {m} and solver set to "
          f"matrix-{'free' if matrix_free else 'based'} {'structured' if structured else 'standard'} l-BFGS and"
          f"{'proximal point' if proximal else 'Tikhonov'} regularisation###")
    p.solve_regularised_ipm()
    print('Done.')

    # save problem and results
    np.save('../results/qcqp_small_log_outer', p.log_outer)
    np.save('../results/qcqp_small_log_inner', p.log_inner)
    np.save('../results/qcqp_small_qp', qp)

    # get results
    fstar = np.array(p.log_outer['val_list']).min()
    xstar = p.log_outer['x_list'][np.argmin(np.array(p.log_outer['val_list']))]

    # compare to result from cvxpy?
    if cvx_bool:
        x = cp.Variable(n)
        cvx_qpprob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, qp.P0) + qp.q0.T @ x),
                          [(1 / 2) * cp.quad_form(x, qp.Pi[i]) + qp.qi[i].T @ x + qp.ri[i] <= 0 for i in range(m)])
        cvx_qpprob.solve()
        cvx_xstar = cvx_qpprob.solution.primal_vars[1]
        cvx_fstar = cvx_qpprob.solution.opt_val

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

    # plot function behaviour
    if plots:
        generate_all_plots(p, 'qcqp_small', fstar=fstar, xstar=xstar, show=False)

        # for plot_inner_fepsmuvals a clip in the beginning is required to properly see the results
        fig, ax = plot_inner_fepsmuvals(p, clipfirst=20)
        fig.savefig('../figures/' + 'qcqp_small' + '_' + 'plot_inner_fepsmuvals')

