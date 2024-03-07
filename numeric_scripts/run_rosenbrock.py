from qnregipm import IneqConstProblem
from numeric_kernel.plot_functions import *
from numeric_kernel.generate_all_plots import generate_all_plots

# This code was developed for the master thesis of Ole Gildemeister,
# written at the Institute of Mathematics and Image Computing, University of Luebeck.
# Code and thesis are available at https://github.com/gildol22/code-master-thesis

# This script solves and plots an inequality-constrained problem based on the Rosenbrock function,
# corresponds to results presented in Section 5.3


def rosenbrock(x):
    """
    Implementation of Rosenbrock function

    Input:
        x: vector (2)

    Output:
        val: function value
        grad: gradient
    """

    val = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    grad = np.array([- 400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)])

    return val, grad


def gs(x):
    val = np.array([-x[0], x[1] ** 2 - 3, x[1] - 1, (x[0] - 1)**2 + (x[1] + 1)**2 - 4])
    grad = np.array([[-1, 0, 0, 2 * (x[0] - 1)], [0, 2 * x[1], 1, 2 * (x[1] + 1)]])

    return val, grad



if __name__ == '__main__':
    plots = False  # generate and save all plots?

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
    f = lambda x: rosenbrock(x)
    g = lambda x: gs(x)
    x0 = np.array([1.5, 0.5])

    # initialise and solve problem
    p = IneqConstProblem(f=f, g=g, x0=x0, eps0=eps0, beta=beta, gamma=gamma, tol_outer=tol_outer, C=C,
                         outer_max_iter=500, lmax_lbfgs=5, rho_ls=.5, c_ls=1e-5, inner_max_iter=300,
                         structured_lbfgs=structured, proximal_point=proximal, display_outer=True, display_inner=False,
                         tol_minres=1e-10, matrix_free=matrix_free)
    print(f"\n\n### Test interior point method with Rosenbrock function and inequality constraints with solver set to"
          f"matrix-{'free' if matrix_free else 'based'} {'structured' if structured else 'standard'} l-BFGS and"
          f"{'proximal point' if proximal else 'Tikhonov'} regularisation###")
    p.solve_regularised_ipm()
    print('Done.')

    # save problem and results
    np.save('../results/rosenbrock_log_outer', p.log_outer)
    np.save('../results/rosenbrock_log_inner', p.log_inner)

    # get results
    fstar = np.array(p.log_outer['val_list']).min()
    xstar = p.log_outer['x_list'][np.argmin(np.array(p.log_outer['val_list']))]

    print(f"-------------------\n"
          f"Own solution missed optimum by difference {fstar} in function value and "
          f"{np.linalg.norm(xstar - np.array([1,1]))} in solution norm")

    fstar = 0.0
    xstar = np.array([1,1])

    # plot function behaviour
    if plots:
        generate_all_plots(p, 'rosenbrock', fstar=fstar, xstar=xstar, show=False)

        # for plot_inner_fepsmuvals a clip in the beginning is required to properly see the results
        fig, ax = plot_inner_fepsmuvals(p, clipfirst=9)
        fig.savefig('../figures/' + 'rosenbrock' + '_' + 'plot_inner_fepsmuvals')


