import numpy as np
import scipy.sparse as ssp
import scipy.sparse.linalg as sspla
from collections import deque
from dataclasses import dataclass
from typing import Callable

# This code was developed for the master thesis of Ole Gildemeister,
# written at the Institute of Mathematics and Image Computing, University of Luebeck.
# Code and thesis are available at https://github.com/gildol22/code-master-thesis


@dataclass
class IneqConstProblem:
    """
    Limited memory Quasi Newton based primal interior point method for convex inequality constraint problems
    with  either Tikhonov or proximal point regularisation and structured l-BFGS as inner solver

    :param f: function, convex target function to be minimised, with f(x) = (val, grad (n,))
    :param g: function, m-dim. convex constraint function to be non-positive, with g(x) = (val (m,), grad (n,m))

    :param x0: vector (n,), initial point
    :param x_regcentre: vector (n,), centre point for Tikhonov regularisation

    :param proximal_point: boolean, if proximal point method instead of Tikhonov regularisation is employed
    :param structured_lbfgs: boolean, if structured l-BFGS is employed as inner solver

    :param eps0: scalar, initial regularisation parameter
    :param mu0: scalar, initial barrier parameter
    :param beta: scalar in (0,1), reduction factor for epsilon
    :param gamma: scalar>1, reduction exponent for mu

    :param tol_outer: scalar >0, tolerance for upper bound on function total residuum
    :param C: scalar >0, tolerance factor in inner iterations (C * eps**2)
    :param outer_max_iter: int, maximum number of iterations for outer steps

    :param lmax_lbfgs: int, memory length in l-BFGS
    :param rho_ls, c_ls: scalars >0, parameters for Armijo line search,
    :param inner_max_iter: int, maximum number of iterations for inner / lbfgs steps
    :param tol_minres: scalar >0, if non-inverse seed matrix (e.g. structured l-BFGS), tolerance for accuracy in initial system of equations; None for exact
    :param matrix_free: boolean, if structured hessian matrix is implemented as matrix free linear operator; only when tol_minres != None

    :param display_outer, display_inner: booleans, determining whether progress in inner resp. outer iteration is displayed
    :param xstar: If known, vector with correct minimiser, for convergence analysis
    """

    # Problem defining variables
    f: Callable
    g: Callable

    # Initial points
    x0: np.ndarray = None
    x_regcentre: np.ndarray = 0

    # Method specification
    proximal_point: bool = False
    structured_lbfgs: bool = False

    # Outer problem parameters
    eps0: float = 1
    mu0: float = 1
    beta: float = .5
    gamma: float = 2
    tol_outer: float = 1e-8
    C: float = 1
    outer_max_iter: int = 100

    # Inner problem parameters
    lmax_lbfgs: int = 5
    rho_ls: float = .5
    c_ls: float = 1e-3
    inner_max_iter: int = 100
    tol_minres: float = 1e-2
    matrix_free: bool = False

    # booleans determining whether progress is displayed
    display_outer: bool = True
    display_inner: bool = False

    # If available: Correct minimiser
    xstar: np.ndarray = None

    # Dictionary to save outer iteration history and results
    log_outer = {
        'xipm': None,  # final approximation to solution
        'jipm': None,  # number of used outer iterations
        'eps_list': [],  # list of used regularisation parameter in each iteration
        'mu_list': [],  # list of used barrier parameter in each iteration
        'klbfgs_list': [],  # list of number of inner iterations
        'x_list': [],  # list of outer iterates
        'val_list': [],  # list of function values evaluated in outer iterates
        'norm_grad_list': [],  # list of gradient norms evaluated in iterates
        'res_upper_list': [],  # list of upper bound of accuracy in function value; used as stopping criterion
    }

    # Dictionary to save inner iteration history
    log_inner = []

    def __post_init__(self):
        # Get problem size and initialise x0 / lmb0 to default values if not given otherwise
        if self.x0 is None:
            raise ValueError('number of parameters unknown, please use x0=np.zeros(n) instead')
        self.n = self.x0.size
        self.n_ineq_constraints = self.g(self.x0)[0].size
        self.inverse_lbfgs = not self.structured_lbfgs

        if self.matrix_free:
            self.current_valg, self.current_gradg = self.g(self.x0)


    def regularised_barrier_function(self, eps, mu, x, xi=0):
        """
        Computes function value and gradient of Tikhonov or proximal point regularised augmented barrier function f_{eps,rho,mu}

        :param lmb: vector (l,), approximation to Lagrange multiplier
        :param eps: scalar, current Tikhonov regularisation weight
        :param mu: scalar, current barrier parameter
        :param x: vector (n,), point at which function is evaluated
        :param xi: vector (n,), if proximal point used, current approximation used for proximal point regularisation; otherwise 0

        :return: val: function value
        :return: grad: gradient
        """
        # compute values and gradients
        valf, gradf = self.f(x)
        valg, gradg = self.g(x)

        # check if x is feasible, i.e. g(x) < 0
        if not (valg < 0).all():
            return np.inf, np.inf * np.ones_like(gradf)

        val = valf + eps / 2 * (x - xi) @ (x - xi) - mu * np.sum(np.log(-valg))
        grad = gradf + eps * (x - xi) - mu * np.asarray((gradg / valg).sum(axis=1)).flatten()
        # Note: np.asarray necessary because if sparse matrix used, a numpy matrix (deprecated) is returned first
        return val, grad

    def structured_hessian(self, eps, mu, xk):
        """
        Computes structured part of hessian of regularised augmented barrier function f_{eps,rho,mu}

        :param eps: scalar, current Tikhonov regularisation weight
        :param mu: scalar, current barrier parameter
        :param xk: vector (n,), point at which structured hessian is computed

        :return: Sxk: matrix(n,n), structured part of hessian at point xk
        """
        valg, gradg = self.g(xk)

        if ssp.issparse(gradg):
            # use ssp.eye
            Sxk = eps * ssp.eye(self.n, self.n) + mu * (gradg / valg ** 2) @ gradg.T
        else:
            # use np.eye; convert to ndarray if scipy operation yields numpy matrix
            Sxk = np.array(eps * np.eye(self.n, self.n) + mu * (gradg / valg ** 2) @ gradg.T)
        return Sxk

    def structured_hessian_matvec(self, eps, mu, xk, x):
        """
        Matrix vector product of structured part of hessian of regularised augmented barrier function f_{eps,rho,mu}
        with vector x

        :param eps: scalar, current Tikhonov regularisation weight
        :param mu: scalar, current barrier parameter
        :param xk: vector (n,), point at which structured hessian is computed
        :param x: vector (n,), point whose product is computed with

        :return: Sxk_mv: vector(n,), matrix vector product of structured part of hessian at point xk with vector x
        """
        Sxk_mv = eps * x + mu * (self.current_gradg / self.current_valg ** 2) @ (self.current_gradg.T @ x)

        return Sxk_mv

    def armijo(self, fctn, p, x):
        """
        Armijo backtracking line search

        :param fctn: function, merit function for armijo condition, with fctn(x) = (val, grad)
        :param p: vector, search direction
        :param x: vector, current iterate

        :return: alpha: scalar, step length with sufficient decrease
        """

        # Initialisation
        alpha = 1.
        lsiter = 0

        valk, gradk = fctn(x)
        descent = gradk.T @ p

        while alpha > 1e-12:
            valkp1, _ = fctn(x + alpha * p)

            # Test armijo condition (sufficient decrease)
            if valkp1 <= valk + self.c_ls * alpha * descent:
                return alpha

            # Reduce alpha
            alpha = self.rho_ls * alpha
            lsiter += 1


        # Message if line search was not successfully ended
        print(f"Line search was unsuccessfully ended after {lsiter} iterations, returning alpha = 0.")
        return 0

    def two_loop_lbfgs(self, p, s, y, seed_matrix):
        """
        two loop l-BFGS recursion (cf. NW06, Algorithm 7.4)

        :param p: vector, initial search direction (neg. gradient)
        :param s: deque(nd.array(n, 1)), last m values of n-dimensional vectors sk-1, ..., sk-m
        :param y: deque(nd.array(n, 1)), last m values of n-dimensional vectors yk-1, ..., yk-m
        :param seed_matrix: matrix, seed matrix for l-BFGS step - approximation to (inverse) hessian H0k / B0k

        :return: p: vector, new search direction
        """

        # initialise
        rho = []
        alpha = []

        # 1. loop
        for yi, si in zip(y, s):
            rho.append(1 / (yi.T @ si))
            alpha.append(rho[-1] * (si.T @ p))
            p = p - alpha[-1] * yi

        if self.inverse_lbfgs:
            # multiplication with seed approximation to inverse Hessian H0k
            if np.ndim(seed_matrix) == 0:
                p = seed_matrix * p
            else:
                p = seed_matrix @ p
        else:
            # system with seed approximation to Hessian B0k, solved exact or up to given tolerance
            if self.tol_minres is None:
                p = sspla.spsolve(seed_matrix, p)
            else:
                p = sspla.minres(seed_matrix, p, tol=self.tol_minres)[0]

        # 2. loop
        for yi, si, rhoi, alphai in zip(reversed(y), reversed(s), reversed(rho), reversed(alpha)):
            beta = rhoi * yi.T @ p
            p = p + (alphai - beta) * si

        return p

    def lbfgs(self, fctn, x0, tau0, tol, S=None):
        """
        Limited memory Quasi Newton method with
        BFGS update via two loop recursion, backtracking line search, gradient norm as stopping criterion
        Optionally use structured l-BFGS as given in

        :param fctn: function, target function to be minimised, with fctn(x) = (val, grad)
        :param x0: vector, initial point
        :param tau0: scalar, initial scale for approximation to (inverse) hessian
        :param tol: scalar, tolerance for stopping criteria
        :param S: function, If structured l-BFGS: Structured part of seed matrix, with S(xk) yielding a matrix (n,n)
                            If further matrix_free: Matrix-vector product of S and x, with S(xk,x) yielding a vector (n,)

        :return: log: dictionary, information on iteration history and results
        """

        # Dictionary to save results and iteration history
        log = {
            'x0': x0,  # initial value
            'xlbfgs': None,  # final approximation to solution
            'klbfgs': None,  # number of used iterations
            'x_list': [],  # list of iterates
            'val_list': [],  # list of function values evaluated in iterates
            'norm_grad_list': [],  # list of gradient norms evaluated in iterates
            'step_size_list': [],  # list of step sizes
        }

        # Initialisation
        xk = x0
        valk, gradk = fctn(xk)
        normgradk = np.linalg.norm(gradk)
        tauk = tau0

        s = deque([], maxlen=self.lmax_lbfgs)
        y = deque([], maxlen=self.lmax_lbfgs)
        self.inverse_lbfgs = not self.structured_lbfgs

        # Initialisation of structured matrix Sk
        if self.structured_lbfgs:
            if self.matrix_free:
                # matrix vector product with structured matrix
                Sk = sspla.LinearOperator((self.n, self.n), matvec=lambda x: S(xk, x))
            else:
                Sk = S(xk)

        # Save initial values
        log['x_list'].append(xk)
        log['val_list'].append(valk)
        log['norm_grad_list'].append(normgradk)


        # l-BFGS iteration
        for k in range(self.inner_max_iter):

            # Stopping criterion
            if normgradk < tol:
                k -= 1
                break

            # compute search direction
            if self.structured_lbfgs:
                # seed matrix: scaled identity plus structured part
                if self.matrix_free:
                    # matrix vector product with seed matrix
                    self.current_valg, self.current_gradg = self.g(xk)
                    B0k = sspla.LinearOperator((self.n, self.n), matvec=lambda x: tauk * x + S(xk, x))
                else:
                    if ssp.issparse(Sk):
                        # use ssp.eye
                        B0k = tauk * ssp.eye(self.n, self.n) + Sk
                    else:
                        # use np.eye
                        B0k = tauk * np.eye(self.n, self.n) + Sk
                # search direction
                pk = - self.two_loop_lbfgs(p=gradk, s=s, y=y, seed_matrix=B0k)
            else:
                # seed matrix: scaled identity
                H0k = tauk * ssp.eye(self.n, self.n)
                # search direction
                pk = - self.two_loop_lbfgs(p=gradk, s=s, y=y, seed_matrix=H0k)

            # compute step
            alphak = self.armijo(fctn=fctn, p=pk, x=xk)
            xkplus1 = xk + alphak * pk
            valkplus1, gradkplus1 = fctn(xkplus1)
            normgradkplus1 = np.linalg.norm(gradkplus1)

            # Save new values
            log['x_list'].append(xkplus1)
            log['val_list'].append(valkplus1)
            log['norm_grad_list'].append(normgradkplus1)
            log['step_size_list'].append(alphak)

            # display progress ???
            if self.display_inner:
                display = f'Iteration k = {k}: \talpha_k = {alphak: .4e}, \tphi(xkp1) = {np.round(valkplus1, 4)}, '
                if self.n < 5:
                    display += f'\txkp1 = {str(np.round(xkplus1, 4)):{1+7*self.n}}'
                else:
                    display += f'\t||xkp1|| = {np.linalg.norm(xkplus1):.4e}, \t||xkp1-xk|| = {np.linalg.norm(xkplus1-xk): .4e}'
                display += f', \t||gradphi(xkp1)|| = {normgradkplus1:.4e}'
                print(display)

            # save update information for l-BFGS
            if len(s) == self.lmax_lbfgs:
                s.pop()
                y.pop()
            s.appendleft(xkplus1 - xk)
            y.appendleft(gradkplus1 - gradk)

            # compute seed matrix for next iteration
            if self.structured_lbfgs:
                # compute next structured matrix Sk
                if self.matrix_free:
                    # matrix vector product with structured matrix
                    Sk = sspla.LinearOperator((self.n, self.n), matvec=lambda x: S(xkplus1, x))
                else:
                    Sk = S(xk)

                # compute parameter tau
                zk = y[0] - Sk @ s[0]
                if (s[0] @ s[0]) > 1e-13:  # otherwise use tau from last iteration
                    tauk = ((zk @ zk) / (s[0] @ s[0]))**0.5  # corresponds to tau^g
            else:
                # compute parameter tau
                if (y[0] @ y[0]) > 1e-13:  # otherwise use tau from last iteration
                    tauk = (s[0] @ y[0]) / (y[0] @ y[0])  # corresponds to tau^y

            # update
            xk = xkplus1
            gradk = gradkplus1
            normgradk = normgradkplus1

            # break if update in iterates is too close to machine accuracy
            if np.linalg.norm(alphak * pk) < 1e-12:
                print(
                    f"l-BFGS was ended after {k+1} steps due to insufficiently large update in iterate, reaching "
                    f"tolerance of {np.linalg.norm(gradk):.4e} instead of {tol:.4e}")
                break

        # Increase k by one
        k += 1

        # Message if iteration did not break
        if k == self.inner_max_iter and normgradk >= tol:
            print(f"l-BFGS did not converge to tolerance of {tol:.4e} after {k} steps, reached only {normgradk}.")

        log['xlbfgs'] = xk
        log['klbfgs'] = k
        return log

    def solve_regularised_ipm(self):
        """
        Solve convex inequality constraint problem by employing a primal interior point method with
        either Tikhonov or proximal point regularisation and chosen l-BFGS method as inner solver
        """

        # Initialisation
        epsj = self.eps0
        muj = self.mu0
        xj = self.x0  # current iterate

        if self.proximal_point:  # current proximal point (here: Latest iterate)
            xij = self.x0
        else:
            xij = self.x_regcentre

        if self.xstar is not None:
            valstar, _ = self.f(self.xstar)


        # Outer iteration
        for j in range(1, self.outer_max_iter+1):
            # build current functions
            def fepsmuj(x): return self.regularised_barrier_function(eps=epsj, mu=muj, x=x, xi=xij)

            # initialise and solve inner problem
            tol_inner = np.max([self.C * epsj ** 2, 1e-8])

            if self.structured_lbfgs:
                if self.matrix_free:
                    def S(xk, x): return self.structured_hessian_matvec(eps=epsj, mu=muj, xk=xk, x=x)
                else:
                    def S(xk): return self.structured_hessian(eps=epsj, mu=muj, xk=xk)
                self.log_inner.append(self.lbfgs(fctn=fepsmuj, x0=xj, tau0=1, tol=tol_inner, S=S))
            else:
                self.log_inner.append(self.lbfgs(fctn=fepsmuj, x0=xj, tau0=1, tol=tol_inner))

            # compute and save results
            xj = self.log_inner[-1]['xlbfgs']
            valj, gradj = self.f(xj)
            normgradj = np.linalg.norm(gradj)

            self.log_outer['eps_list'].append(epsj)
            self.log_outer['mu_list'].append(muj)
            self.log_outer['klbfgs_list'].append(self.log_inner[-1]["klbfgs"])
            self.log_outer['x_list'].append(xj)
            self.log_outer['val_list'].append(valj)
            self.log_outer['norm_grad_list'].append(normgradj)

            # upper bound of accuracy in function value
            # res_upper = epsj * (self.C * normgradj + 2 * (xj - self.x_regcentre) @ (xj - self.x_regcentre)) + self.n_ineq_constraints * muj
            tol_inner_reached = self.log_inner[-1]['norm_grad_list'][-1]
            res_upper = normgradj * tol_inner_reached / epsj + \
                        2 * epsj * (xj - self.x_regcentre) @ (xj - self.x_regcentre) + self.n_ineq_constraints * muj
            self.log_outer['res_upper_list'].append(res_upper)


            # display progress
            if self.display_outer:
                display = f'Outer it. j = {j}: \teps = {epsj:.2e}, \tmu = {muj:.2e}, '\
                          f'\tl-BFGS it. = {self.log_inner[-1]["klbfgs"]}, '\
                          f'\tf(xj) = {np.round(valj, 4):6}, \tfepsmu(xj) = {np.round(self.log_inner[-1]["val_list"][-1], 4):6}, '\
                          f'\t||gradf(xj)|| = {normgradj: .4e}, '
                if self.n < 5:
                    display += f'\txj = {str(np.round(xj, 4)):{1+7*self.n}}, '
                else:
                    display += f'\t||xj|| = {np.linalg.norm(xj):.4e}, '
                display += f'\tres_upper = {res_upper:.3e}'
                if self.xstar is not None:
                    display += f', \tres_correct = {valj - valstar:.3e}'  # Note: xj might not be feasible, so informativity is limited
                print(display)

            # stopping criterion;
            if res_upper < self.tol_outer:  # accuracy in function value
                break

            # update
            epsj = self.beta * epsj
            muj = self.beta**self.gamma * muj
            if self.proximal_point:
                xij = xj  # change current proximal point to latest iterate

        # yield message if tolerance was not achieved and return results;
        if res_upper >= self.tol_outer:
            print(f"Regularised IPM did not necessarily reach residuum tolerance of < {self.tol_outer:.4e} after {j} outer iterations.")
        else:
            print(f"Residuum tolerance of < {res_upper:.4e} was reached after {j} outer iterations.")
        self.log_outer['jipm'] = j
        self.log_outer['xipm'] = xj









# Test functionality


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
    val = np.array([-x[0], x[1] ** 2 - 3, x[1] - 2])
    grad = np.array([[-1, 0, 0], [0, 2 * x[1], 1]])

    return val, grad


def test_lbfgs():
    # define problem (only solved by lbfgs, ignore other part)
    x0 = np.array([.5, .5])
    tol = 1e-10
    rho = .5
    c_ls = 1e-3
    kmax = 100
    m = 5

    f = lambda x: rosenbrock(x)
    tau0 = 1

    # Ignored values needed for problem class
    g = lambda x: gs(x)

    # initialise problem and solve lbfgs
    p = IneqConstProblem(f=f, g=g, x0=x0, lmax_lbfgs=m, rho_ls=rho, c_ls=c_ls, inner_max_iter=kmax, structured_lbfgs=False, display_inner=True)
    print("\n\n### Test l-BFGS method with Rosenbrock ###")
    log = p.lbfgs(fctn=f, x0=x0, tau0=tau0, tol=tol)
    print('Done.')


def test_slbfgs():
    # give partial hessian
    def rosenbrock_parthess(x):
        return np.diag([1200 * x[0] ** 2 + 2, 200])

    # define problem (only solved by structured lbfgs, ignore other part)
    x0 = np.array([.5, .5])
    tol = 1e-10
    rho = .5
    c_ls = 1e-3
    kmax = 100
    m = 5
    tol_minres = 1e-2

    f = lambda x: rosenbrock(x)
    S = lambda x: rosenbrock_parthess(x)
    tau0 = 1

    # Ignored values needed for problem class
    g = lambda x: gs(x)

    # initialise problem and solve lbfgs
    p = IneqConstProblem(f=f, g=g, x0=x0, lmax_lbfgs=m, rho_ls=rho, c_ls=c_ls, inner_max_iter=kmax,
                         structured_lbfgs=True, display_inner=True)
    print("\n\n### Test structured l-BFGS method with Rosenbrock and partially given Hessian ###")
    log = p.lbfgs(fctn=f, x0=x0, tau0=tau0, tol=tol, S=S)
    print('Done.')


def test_ipm(structured=False, proximal=False):
    # define problem
    f = lambda x: rosenbrock(x)
    g = lambda x: gs(x)

    x0 = np.array([1.1, 0.9])
    xstar = np.array([1, 1])
    eps0 = 1
    beta = .5
    gamma = 1.2
    tol_outer = 1e-6
    C = 1

    # initialise and solve problem
    p = IneqConstProblem(f=f, g=g, x0=x0, eps0=eps0, beta=beta, gamma=gamma, tol_outer=tol_outer, C=C,
                         outer_max_iter=100, lmax_lbfgs=5, rho_ls=.5, c_ls=1e-3, inner_max_iter=100,
                         structured_lbfgs=structured, proximal_point=proximal, xstar=xstar)
    print(f"\n\n### Test interior point method with Rosenbrock function and inequality constraints with "
          f"{'structured' if structured else 'standard'} l-BFGS and {'proximal point' if proximal else 'Tikhonov'} "
          f"regularisation###")
    p.solve_regularised_ipm()
    print('Done.')


if __name__ == '__main__':
    test_lbfgs()
    test_slbfgs()
    test_ipm(structured=True, proximal=False)
