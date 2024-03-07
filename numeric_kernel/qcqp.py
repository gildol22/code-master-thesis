import numpy as np
import scipy.sparse as ssp
from dataclasses import dataclass
from qnregipm import IneqConstProblem

# This code was developed for the master thesis of Ole Gildemeister,
# written at the Institute of Mathematics and Image Computing, University of Luebeck.
# Code and thesis are available at https://github.com/gildol22/code-master-thesis


@dataclass()
class QuadraticProgram:
    """
    Defines and generates quadratically constrained quadratic optimisation program of the form
    inf_x 1/2 x.T P_0 x + q_0.T x
    s.t.
    1/2 x.T P_i x + q_i.T + r_i

    By r_i<0 it is guaranteed that x0=0 is strictly feasible
    """

    # problem size
    n: int  # number of variables
    m: int  # number of inequality constraints


    # problem parameters
    P0: ssp.csr_matrix = None  # quadratic matrix of objective, size (n,n)
    q0: np.ndarray = None  # linear vector in objective, size (n,)

    #Pi: ssp.csr_matrix = None  # quadratic matrix of objective, size (m,n,n)
    Pi: [] = None
    qi: np.ndarray = None  # linear vector in objective, size (m,n)

    density: float = 0.1  # density of the problem matrices

    x0: np.ndarray = None  # strictly feasible starting point


    # construct random problem with symmetric positive semi-definite matrices
    def generate_problem(self):
        A = 10 * ssp.random(self.n, self.n, density=self.density, format='csr', random_state=np.random.default_rng(2206))
        self.P0 = A.T @ A + ssp.eye(self.n)
        self.q0 = np.random.rand(self.n)

        self.Pi = []
        for i in range(self.m):
            A = 10 * ssp.random(self.n, self.n, density=self.density, format='csr', random_state=np.random.default_rng(2206))
            self.Pi.append(A.T @ A + ssp.eye(self.n))

        self.qi = np.random.rand(self.m, self.n)
        self.ri = -99 * np.random.rand(self.m) - 1

        # generate feasible starting point
        self.x0 = np.zeros(self.n)


    def evaluate_objective(self, x):
        valf = x.T @ self.P0 @ x / 2 + self.q0 @ x
        gradf = self.P0 @ x + self.q0
        return valf, gradf

    def evaluate_constraints(self, x):
        valg = np.array([x.T @ self.Pi[i] @ x / 2 + self.qi[i,:] @ x + self.ri[i] for i in range(self.m)])
        gradg = np.array([self.Pi[i] @ x + self.qi[i,:] for i in range(self.m)])
        return valg, gradg.T



def test_gp():

    # build problem
    n = 10
    m = 5

    qp = QuadraticProgram(n=n, m=m)
    np.random.seed(2206)
    qp.generate_problem()

    x0 = qp.x0
    f = lambda x: qp.evaluate_objective(x)
    g = lambda x: qp.evaluate_constraints(x)


    # set solver parameters
    eps0 = 1
    mu0 = 1
    beta = .8
    gamma = 1.01
    tol_outer = 1e-3
    C = 1

    # set solver settings
    structured = True
    proximal = False


    # initialise and solve problem
    p = IneqConstProblem(f=f, g=g, x0=x0, eps0=eps0, mu0=mu0, beta=beta, gamma=gamma, tol_outer=tol_outer, C=C,
                          outer_max_iter=200, lmax_lbfgs=5, rho_ls=.5, c_ls=1e-3, inner_max_iter=100,
                          structured_lbfgs=structured, proximal_point=proximal, display_outer=True, display_inner=False,
                          tol_minres=1e-10)

    print(f"\n\n### Test interior point method for quadratically inequality constrained quadratic program of size "
          f"n = {n}, m = {m} and solver set to "
          f"{'structured' if structured else 'standard'} l-BFGS and {'proximal point' if proximal else 'Tikhonov'} "
          f"regularisation###")
    p.solve_regularised_ipm()
    print('Done.')



if __name__ == "__main__":
    test_gp()
