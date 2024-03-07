import numpy as np
from dataclasses import dataclass
from qnregipm import IneqConstProblem

# This code was developed for the master thesis of Ole Gildemeister,
# written at the Institute of Mathematics and Image Computing, University of Luebeck.
# Code and thesis are available at https://github.com/gildol22/code-master-thesis


@dataclass()
class GeometricProgram:
    """
    Defines and generates geometric optimisation program of the form
    inf_x ln(sum(exp(F_j x + b_j)))
    s.t.
    ln(sum(exp(- F_j x - b_j + 1 - 2 log(k_obj)))) <= 0
    ln(sum(exp(G_ij x + cij))) <= 0 for i=1,...,m

    Optimal value is guaranteed to lie >= 1
    """

    # problem size
    n: int  # number of variables
    m: int  # number of inequality constraints
    k_obj: int  # number of summands in objective
    k_ineq: int  # number of summands in each inequality constraint

    # problem parameters
    F: np.ndarray = None  # linear factors in objective, size (k_obj,n)
    b: np.ndarray = None  # offset in objective, size (k_obj,)

    G: np.ndarray = None  # linear factors in inequality constraints, size (m,k_ineq,n)
    c: np.ndarray = None  # offsets in inequality constraints, size (m,k_ineq)

    c_min: float = 4
    x0: np.ndarray = None  # strictly feasible starting point

    def generate_problem(self):
        self.F = np.random.rand(self.k_obj, self.n) - 2  # in [-2,-1)
        self.b = np.random.rand(self.k_obj)  # in [0,1)

        self.G = np.random.rand(self.m, self.k_ineq, self.n) + 1 # in [1,2)
        self.c = np.random.rand(self.m, self.k_ineq)  # in [c_min, c_min+1); c_min is added manually for numeric reasons

        # generate strictly feasible starting point
        self.x0 = min( (-np.max(self.c+self.c_min)-np.log(self.k_ineq)) / (np.min(self.G) * self.n),
                          (np.max(self.b)+np.log(self.k_obj)-1) / (-np.max(self.F) * self.n) ) * np.ones(self.n)

    def evaluate_objective(self, x):
        s = np.sum( np.exp(self.F @ x + self.b) )
        valf = np.log(s)
        gradf = np.sum(np.exp(self.F @ x + self.b) * self.F.T, axis=1) / s

        return valf, gradf

    def evaluate_constraints(self, x):
        # first constraint corresponds to reciprocal objective plus shift such that objective >= 1
        s = np.sum( np.exp(- self.F @ x - self.b + 1 - 2 * np.log(self.k_obj)) )
        valf = np.log(s)
        gradf = np.sum(- np.exp(- self.F @ x - self.b) * self.F.T, axis=1) / s

        # further constraints
        s = np.sum( np.exp(self.G @ x + self.c), axis=1)
        valg = np.log(s) + self.c_min
        gradg = np.sum( np.exp(self.G @ x + self.c) * self.G.transpose((2,0,1)), axis=2) / s

        return np.hstack((valf, valg)), np.hstack((gradf[:,np.newaxis], gradg))



def test_gp():

    # build problem
    n = 5
    m = 10
    k_obj = 5
    k_ineq = 4

    gp = GeometricProgram(n=n, m=m, k_obj=k_obj, k_ineq=k_ineq)
    np.random.seed(2206)
    gp.generate_problem()

    x0 = gp.x0
    f = lambda x: gp.evaluate_objective(x)
    g = lambda x: gp.evaluate_constraints(x)


    # set solver parameters
    eps0 = 1
    mu0 = 1
    beta = .8
    gamma = 1.2
    tol_outer = 1e-3
    C = 10

    # set solver settings
    structured = True
    proximal = False


    # initialise and solve problem
    p = IneqConstProblem(f=f, g=g, x0=x0, eps0=eps0, mu0=mu0, beta=beta, gamma=gamma, tol_outer=tol_outer, C=C,
                          outer_max_iter=100, lmax_lbfgs=5, rho_ls=.5, c_ls=1e-3, inner_max_iter=100,
                          structured_lbfgs=structured, proximal_point=proximal, display_outer=True, display_inner=False,
                          tol_minres=1e-10)

    print(f"\n\n### Test interior point method for inequality constrained geometric program of size "
          f"n = {n}, m = {m}, objecive depth = {k_obj}, constraint depth = {k_ineq} and solver set to "
          f"{'structured' if structured else 'standard'} l-BFGS and {'proximal point' if proximal else 'Tikhonov'} "
          f"regularisation###")
    p.solve_regularised_ipm()
    print('Done.')



if __name__ == "__main__":
    test_gp()
