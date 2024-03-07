import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from qnregipm import *

# This code was developed for the master thesis of Ole Gildemeister,
# written at the Institute of Mathematics and Image Computing, University of Luebeck.
# Code and thesis are available at https://github.com/gildol22/code-master-thesis

# This script solves and plots the ''plate problem'' described in Example 3.7 for different parameter choices,
# corresponds to Figure 3.2


# set global variables
plate_eps = 1
plate_mu = 1
n_iter = 30
mu_eps_exp = 1.5


def plate_f(x):
    if x @ x > 1:
        val = (np.linalg.norm(x) - 1)**4
        grad = (4 * (np.linalg.norm(x) - 1)**3 / np.linalg.norm(x)) * x
    else:
        val = 0
        grad = np.zeros_like(x)
    return val, grad


def plate_feps(x):
    valf, gradf = plate_f(x)
    val = valf + plate_eps / 2 * (x + 2) @ (x + 2)
    grad = gradf + plate_eps * (x + 2)
    return val, grad


def plate_fmu(x):
    valf, gradf = plate_f(x)
    if not x[1] > 0:
        return np.inf, np.inf * np.ones_like(gradf)

    val = valf - plate_mu * np.log(x[1])
    grad = gradf - plate_mu * np.array([0, 1 / x[1]])
    return val, grad


def plate_fepsmu(x):
    valf, gradf = plate_f(x)
    if not x[1] > 0:
        return np.inf, np.inf * np.ones_like(gradf)

    val = valf + plate_eps / 2 * (x + 2) @ (x + 2) - plate_mu * np.log(x[1])
    grad = gradf + plate_eps * (x + 2) - plate_mu * np.array([0, 1 / x[1]])
    return val, grad


def g(x):
    val = np.array(- x[1])
    grad = np.array([[0],[-1]])

    return val, grad


def run_plate_min(fctn):
    x0 = np.array([-1, 3])
    eps0 = .1
    beta = .5
    tol_outer = -1 # won't stop!
    C = 1

    # initialise problem and solve lbfgs
    p = IneqConstProblem(f=fctn, g=g, x0=x0, eps0=eps0, beta=beta, tol_outer=tol_outer, C=C,
                         outer_max_iter=30, lmax_lbfgs=5, rho_ls=.5, c_ls=1e-3, inner_max_iter=100,
                         structured_lbfgs=True, proximal_point=False, display_outer=True)
    print(f"\n\n### Test interior point method with plate function and inequality constraint")
    p.solve_regularised_ipm()

    return p.log_outer['xipm']


def compute_xeps_trajectory(eps_list):
    global plate_eps
    xeps_traj = np.zeros((2, n_iter))

    for i in range(n_iter):
        plate_eps = eps_list[i]
        xeps = run_plate_min(fctn=plate_feps)
        xeps_traj[:, i] = xeps

    return xeps_traj


def compute_xmu_trajectory(mu_list):
    global plate_mu
    xmu_traj = np.zeros((2, n_iter))

    for i in range(n_iter):
        plate_mu = mu_list[i]
        xmu = run_plate_min(fctn=plate_fmu)
        xmu_traj[:, i] = xmu

    return xmu_traj


def compute_xepsmu_trajectory(eps_list, mu_list):
    global plate_eps, plate_mu
    xepsmu_traj = np.zeros((2, n_iter))

    for i in range(n_iter):
        plate_eps = eps_list[i]
        plate_mu = mu_list[i]
        xepsmu = run_plate_min(fctn=plate_fepsmu)
        xepsmu_traj[:, i] = xepsmu

    return xepsmu_traj


def plot(xeps_traj, xmu_traj, xepsmu_traj):
    # create plot with solution set circle and feasible boundary
    fig, ax = plt.subplots(1, 1, dpi=800)
    ax.add_patch(patches.Circle((2, 0), radius=1, edgecolor='None', facecolor='blue', alpha=.2, zorder=-4))
    ax.add_patch(patches.Circle((2, 0), radius=1, edgecolor='b', facecolor='None', zorder=-4))
    ax.text(1.95, .3, "$\mathcal{S}$", color='blue')

    ax.add_patch(patches.Rectangle((.9, -1.1), 2.2, 1.1, edgecolor='None', facecolor='white', alpha=1, zorder=-3))
    ax.add_patch(patches.Rectangle((-.5, -1.5), 4, 1.5, edgecolor='None', facecolor='red', alpha=.1, zorder=-2))
    ax.axhline(0, color='red', zorder=-1)
    ax.text(1.3, -.8, '$x_2<0$', color='red')

    # add trajectory points
    ax.scatter(xeps_traj[0] + 2, xeps_traj[1], zorder=1, label="$\\bar{x}_{\\epsilon}$")
    ax.scatter(xmu_traj[0] + 2, xmu_traj[1], zorder=1, label="$\\bar{x}_{\\mu}$")
    ax.scatter(xepsmu_traj[0] + 2, xepsmu_traj[1], zorder=1, label="$\\bar{x}_{\\epsilon, \\mu}$")

    # add origin and minimisers
    ax.scatter([0, 1, 2], [0, 0, 1], s=50, color='black', marker='x', linewidths=2, zorder=2)
    ax.text(0, -.2, '(0,0)', color='black')
    ax.text(1.05, -.2, "$x^\\ast_M$", color='black')
    ax.text(2, .8, "$x^\\ast_{an}$", color='black')

    # set axis limits, legends and title
    ax.set_xlim([-0.5,3.5])
    ax.set_ylim([-1.5,2.5])
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title("Trajectories of $\\bar{x}_{\\epsilon}$, $\\bar{x}_{\\mu}$ and $\\bar{x}_{\\epsilon, \\mu}$" +
              f" for $\\mu = O(\\epsilon^{{{mu_eps_exp}}})$")

    # draw and save figure
    plt.draw()


def run(paramcase=1):
    global n_iter, mu_eps_exp
    n_iter = 30

    match paramcase:
        case 1:
            plate_eps0 = 2 ** 12
            plate_mu0 = 2 ** 14
            plate_beta = 0.5
            mu_eps_exp = 1.5
        case 2:
            plate_eps0 = 2 ** 12
            plate_mu0 = 2 ** 12
            plate_beta = 0.5
            mu_eps_exp = 1.1
        case 3:
            plate_eps0 = 2 ** 12
            plate_mu0 = 2 ** 14
            plate_beta = 0.5
            mu_eps_exp = 1
        case 4:
            plate_eps0 = 2 ** 12
            plate_mu0 = 2 ** 11
            plate_beta = 0.5
            mu_eps_exp = .8

    eps_list = plate_eps0 * plate_beta ** np.arange(n_iter)
    mu_list = plate_mu0 * (plate_beta ** mu_eps_exp) ** np.arange(n_iter)

    xeps_traj = compute_xeps_trajectory(eps_list)
    xmu_traj = compute_xmu_trajectory(mu_list)
    xepsmu_traj = compute_xepsmu_trajectory(eps_list, mu_list)

    plot(xeps_traj, xmu_traj, xepsmu_traj)


if __name__ == '__main__':
    for i in range(1,5):
        run(i)
        plt.savefig(f'../../figures/ex_plate_minimisation_{i}', bbox_inches="tight")

    plt.show()





