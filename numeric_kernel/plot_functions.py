import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.legend_handler import HandlerBase

# This code was developed for the master thesis of Ole Gildemeister,
# written at the Institute of Mathematics and Image Computing, University of Luebeck.
# Code and thesis are available at https://github.com/gildol22/code-master-thesis


"""
## File consists of functions that create plots to evaluate evolution of the method. All functions take arguments
    :param p: IneqConstProblem which has been solved
    :param clipfirst: int - number of first outer iterations to skip
    :param cliplast: int - number of last outer iterations to skip
    
   and return (fig, ax) with plot
   Optionally, some methods allow to pass a float 'fstar' or ndarray 'xstar'
"""


# Stuff to get a color gradient in the legend
class GradientHandler(HandlerBase):
    def __init__(self, num_stripes=50, **kw):
        HandlerBase.__init__(self, **kw)
        self.cdict = {
            "blue": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            "red": [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]],
            "green": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        }
        self.cmap = LinearSegmentedColormap('Red2Blue', self.cdict)
        self.num_stripes = num_stripes

    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        artists = []
        if '\u200B' in orig_handle.get_label():
            # RedBlue gradient
            for i in range(self.num_stripes):
                x = [x0 + i * width / self.num_stripes,
                     x0 + (i+1) * width / self.num_stripes]
                y = [y0 + height / 2] * 2
                line = plt.Line2D(x, y, color=self.cmap(i / self.num_stripes))
                artists.append(line)
        else:
            # Solid line
            x = [x0, x0 + width]
            y = [y0 + height / 2] * 2
            line = plt.Line2D(x, y, color=orig_handle.get_color(), solid_capstyle='butt')
            artists.append(line)
        return artists


def redbluecmap(mix):
    """
    Interpolates
    :param mix: in [0,1] -> interpolation value from red to blue
    :return:
    """
    c1 = np.array(mpl.colors.to_rgb('red'))
    c2 = np.array(mpl.colors.to_rgb('blue'))
    return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)


def plot_outer_fvals_raw(p, clipfirst=0, cliplast=0):
    """
    Plots outer evolution of objective function values, i.e. f(xj)
    """
    fval0, _ = p.f(p.x0)
    fvals = np.hstack((fval0, np.array(p.log_outer['val_list'])))

    fig, ax = plt.subplots(dpi=600)
    last = fvals.size - cliplast
    ax.plot(np.arange(clipfirst, last), fvals[clipfirst:last], label='$f(x^j)$')
    ax.set_title("Evolution of $f(x^j)$")
    ax.legend(handler_map={plt.Line2D: GradientHandler()})
    ax.annotate('$j$', xy=(1.01, 0), xycoords='axes fraction', ha='left', va='bottom')
    return fig, ax


def plot_outer_fvals_diff(p, clipfirst=0, cliplast=0, fstar=None):
    """
    Plots outer evolution of difference objective function, i.e. (f(xj)-fstar)
    """
    fval0, _ = p.f(p.x0)
    fvals = np.hstack((fval0, np.array(p.log_outer['val_list'])))

    if fstar is None:
        fstar = fvals.min()
    fval_diff = fvals - fstar

    fig, ax = plt.subplots(dpi=600)
    last = fval_diff.size - cliplast
    ax.plot(np.arange(clipfirst, last), fval_diff[clipfirst:last], label='$f(x^j) - f^\\ast$')
    ax.set_yscale('log')
    ax.set_title("Evolution of $f(x^j) - f^\\ast$")
    ax.legend(handler_map={plt.Line2D: GradientHandler()})
    ax.annotate('$j$', xy=(1.01, 0), xycoords='axes fraction', ha='left', va='bottom')
    return fig, ax


def plot_outer_fvals_resupper(p, clipfirst=0, cliplast=0, fstar=None):
    """
    Plots outer evolution of difference objective function, i.e. (f(xj)-fstar)
    with upper bound used as stopping criterion
    """
    fval0, _ = p.f(p.x0)
    fvals = np.hstack((fval0, np.array(p.log_outer['val_list'])))
    resuppers = np.array(p.log_outer['res_upper_list'])

    if fstar is None:
        fstar = fvals.min()
    fval_diff = fvals - fstar

    fig, ax = plt.subplots(dpi=600)
    last = fval_diff.size - cliplast
    ax.plot(np.arange(clipfirst, last), fval_diff[clipfirst:last], label='$f(x^j) - f^\\ast$')
    ax.plot(np.arange(clipfirst+1, last), resuppers[clipfirst:last-1], label='Upper bound')
    ax.set_yscale('log')
    ax.set_title("Evolution of $f(x^j) - f^\\ast$ and upper bound")
    ax.legend(handler_map={plt.Line2D: GradientHandler()})
    ax.annotate('$j$', xy=(1.01, 0), xycoords='axes fraction', ha='left', va='bottom')
    return fig, ax


def plot_outer_fvals_update(p, clipfirst=0, cliplast=0, fstar=None):
    """
    Plots update in outer evolution of objective function values, i.e. (f(xj)-fstar)/(f(xjm1)-fstar)
    """
    fval0, _ = p.f(p.x0)
    fvals = np.hstack((fval0, np.array(p.log_outer['val_list'])))

    if fstar is None:
        fstar = fvals.min()
    fval_update = (fvals[1:] - fstar) / (fvals[:-1] - fstar)

    fig, ax = plt.subplots(dpi=600)
    last = fval_update.size - cliplast
    ax.plot(np.arange(clipfirst, last), fval_update[clipfirst:last],
            label='$(f(x^{j+1}) - f^\\ast) / (f(x^j) - f^\\ast)$')
    ax.set_ylim([0, None])
    ax.set_title("Evolution of $(f(x^{j+1}) - f^\\ast) / (f(x^j) - f^\\ast)$")
    ax.legend(handler_map={plt.Line2D: GradientHandler()})
    ax.annotate('$j$', xy=(1.01, 0), xycoords='axes fraction', ha='left', va='bottom')
    return fig, ax


def plot_inner_fepsmuvals(p, clipfirst=0, cliplast=0):
    """
    Plots inner evolution of regularised barrier function values, i.e. fepsmu(xjk)
    with color gradient
    """
    ks = np.array(p.log_outer['klbfgs_list'])
    iter_marks = np.hstack((0, np.cumsum(ks + 1)))

    fig, ax = plt.subplots(dpi=600)
    last = ks.size - cliplast
    for innerlog, kmin, kmax in zip(p.log_inner[clipfirst:last], iter_marks[clipfirst:last-1], iter_marks[clipfirst+1:last]):
        x = np.arange(kmin, kmax)
        y = np.array(innerlog['val_list'])
        for i in range(len(x)):
            ax.plot(x[i:i + 2], y[i:i + 2], color=redbluecmap(i / (len(x) - 0.9)), linewidth=1, label='_nolegend_')
            
    ax.set_title("Evolution of $f_{\\epsilon, \\mu} (x^j_k)$")
    ax.plot([], label='\u200B$f_{\\epsilon, \\mu} (x^j_k)$')  # dummy plot for legend
    ax.legend(handler_map={plt.Line2D: GradientHandler()})
    ax.annotate('$\\bar{k}$', xy=(1.01, 0), xycoords='axes fraction', ha='left', va='bottom')
    return fig, ax


def plot_outer_xdiff_raw(p, clipfirst=0, cliplast=0, xstar = None):
    """
    Plots outer evolution of iterate difference towards optimal point, i.e. ||xj - xstar||
    """
    xlist = np.array([p.x0] + p.log_outer['x_list'])
    if xstar is None:
        xstar = xlist[np.argmin(np.array(p.log_outer['val_list'])) + 1]
    xdiffs = np.linalg.norm(xlist - xstar, axis=1)

    fig, ax = plt.subplots(dpi=600)
    last = xdiffs.size - cliplast
    ax.plot(np.arange(clipfirst, last), xdiffs[clipfirst:last], label='$||x^j - x^\\ast||$')
    ax.set_yscale('log')
    ax.set_title("Evolution of $||x^j - x^\\ast||$")
    ax.legend(handler_map={plt.Line2D: GradientHandler()})
    ax.annotate('$j$', xy=(1.01, 0), xycoords='axes fraction', ha='left', va='bottom')
    return fig, ax


def plot_outer_xdiff_update(p, clipfirst=0, cliplast=0, xstar = None):
    """
    Plots update in outer evolution of iterate difference towards optimal point, i.e.
    ||xj - xstar||/||xjm1 - xstar||
    """
    xlist = np.array([p.x0] + p.log_outer['x_list'])
    if xstar is None:
        xstar = xlist[np.argmin(np.array(p.log_outer['val_list'])) + 1]
    xdiffs = np.linalg.norm(xlist - xstar, axis=1)
    xdiff_update = xdiffs[1:] / xdiffs[:-1]

    fig, ax = plt.subplots(dpi=600)
    last = xdiff_update.size - cliplast
    ax.plot(np.arange(clipfirst, last), xdiff_update[clipfirst:last], label='$||x^j - x^\\ast||/||x^{j-1} - x^\\ast||$')
    ax.set_ylim([0, None])
    ax.set_title("Evolution of $||x^j - x^\\ast||/||x^{j-1} - x^\\ast||$")
    ax.legend(handler_map={plt.Line2D: GradientHandler()})
    ax.annotate('$j$', xy=(1.01, 0), xycoords='axes fraction', ha='left', va='bottom')
    return fig, ax


def plot_inner_xdiff(p, clipfirst=0, cliplast=0, xstar=None):
    """
    Plots inner evolution of iterate difference towards optimal point, i.e. ||xjk - xstar||
    with color gradient
    """
    ks = np.array(p.log_outer['klbfgs_list'])
    iter_marks = np.hstack((0, np.cumsum(ks + 1)))

    if xstar is None:
        xstar = p.log_outer['x_list'][np.argmin(np.array(p.log_outer['val_list']))]

    fig, ax = plt.subplots(dpi=600)
    last = ks.size - cliplast
    for innerlog, kmin, kmax in zip(p.log_inner[clipfirst:last], iter_marks[clipfirst:last-1], iter_marks[clipfirst+1:last]):
        x = np.arange(kmin, kmax)
        y = np.linalg.norm(np.array(innerlog['x_list']) - xstar, axis=1)
        for i in range(len(x)):
            ax.plot(x[i:i+2], y[i:i+2], color=redbluecmap(i/(len(x)-0.9)), linewidth=1, label='_nolegend_')

    ax.set_yscale('log')
    ax.set_title("Evolution of $||x^j_k - x^\\ast||$")
    ax.plot([], label='\u200B$||x^j_k - x^\\ast||$')  # dummy plot for legend
    ax.legend(handler_map={plt.Line2D: GradientHandler()})
    ax.annotate('$\\bar{k}$', xy=(1.01, 0), xycoords='axes fraction', ha='left', va='bottom')
    return fig, ax


def plot_both_xdiff(p, clipfirst=0, cliplast=0, xstar=None):
    """
    Plots inner evolution of iterate difference towards optimal point, i.e. ||xjk - xstar||
    with color gradient and outer plot
    """
    ks = np.array(p.log_outer['klbfgs_list'])
    iter_marks = np.hstack((0, np.cumsum(ks + 1)))

    xlist = np.array([p.x0] + p.log_outer['x_list'])
    if xstar is None:
        xstar = xlist[np.argmin(np.array(p.log_outer['val_list'])) + 1]
    xdiffs = np.linalg.norm(xlist - xstar, axis=1)

    fig, ax = plt.subplots(dpi=600)
    last = ks.size - cliplast
    ax.plot([], label='\u200B$||x^j_k - x^\\ast||$')  # dummy plot for legend
    ax.plot(iter_marks[clipfirst:last], xdiffs[clipfirst:last], color='teal', linewidth=1.2, label='$||x^j - x^\\ast||$')
    for innerlog, kmin, kmax in zip(p.log_inner[clipfirst:last], iter_marks[clipfirst:last-1], iter_marks[clipfirst+1:last]):
        x = np.arange(kmin, kmax)
        y = np.linalg.norm(np.array(innerlog['x_list']) - xstar, axis=1)
        for i in range(len(x)):
            ax.plot(x[i:i+2], y[i:i+2], color=redbluecmap(i/(len(x)-0.9)), linewidth=1, label='_nolegend_')

    ax.set_yscale('log')
    ax.set_title("Evolution of $||x^j_k - x^\\ast||$")
    ax.legend(handler_map={plt.Line2D: GradientHandler()})
    ax.annotate('$\\bar{k}$', xy=(1.01, 0), xycoords='axes fraction', ha='left', va='bottom')
    return fig, ax


def plot_outer_gradfepsmu(p, clipfirst=0, cliplast=0):
    """
    Plots outer evolution of gradient norm of regularised barrier function, i.e. ||grad fepsmu(xj)||
    """
    grad0 = np.linalg.norm(p.regularised_barrier_function(p.eps0, p.mu0, p.x0, p.x_regcentre)[1])
    gradnorms = np.hstack((grad0, np.array([i['norm_grad_list'][-1] for i in p.log_inner])))

    fig, ax = plt.subplots(dpi=600)
    last = gradnorms.size - cliplast
    ax.plot(np.arange(clipfirst, last), gradnorms[clipfirst:last], label='$|| \\nabla f_{\\epsilon, \\mu} (x^j) ||$')
    ax.set_yscale('log')
    ax.set_title("Evolution of $|| \\nabla f_{\\epsilon, \\mu} (x^j) ||$")
    ax.legend(handler_map={plt.Line2D: GradientHandler()})
    ax.annotate('$j$', xy=(1.01, 0), xycoords='axes fraction', ha='left', va='bottom')
    return fig, ax


def plot_inner_gradfepsmu(p, clipfirst=0, cliplast=0):
    """
    Plots inner evolution of gradient norm of regularised barrier function, i.e. ||grad fepsmu(xjk)||
    with color gradient
    """
    ks = np.array(p.log_outer['klbfgs_list'])
    iter_marks = np.hstack((0, np.cumsum(ks + 1)))

    fig, ax = plt.subplots(dpi=600)
    last = ks.size - cliplast
    for innerlog, kmin, kmax in zip(p.log_inner[clipfirst:last], iter_marks[clipfirst:last-1], iter_marks[clipfirst+1:last]):
        x = np.arange(kmin, kmax)
        y = np.array(innerlog['norm_grad_list'])
        for i in range(len(x)):
            ax.plot(x[i:i+2], y[i:i+2], color=redbluecmap(i/(len(x)-0.9)), linewidth=1, label='_nolegend_')

    ax.set_yscale('log')
    ax.set_title("Evolution of $|| \\nabla f_{\\epsilon, \\mu} (x^j_k) ||$")
    ax.plot([], label='\u200B$|| \\nabla f_{\\epsilon, \\mu} (x^j_k) ||$')  # dummy plot for legend
    ax.legend(handler_map={plt.Line2D: GradientHandler()})
    ax.annotate('$\\bar{k}$', xy=(1.01, 0), xycoords='axes fraction', ha='left', va='bottom')
    return fig, ax


def plot_both_gradfepsmu(p, clipfirst=0, cliplast=0):
    """
    Plots inner evolution of gradient norm of regularised barrier function, i.e. ||grad fepsmu(xjk)||
    with color gradient and outer plot
    """
    ks = np.array(p.log_outer['klbfgs_list'])
    iter_marks = np.hstack((0, np.cumsum(ks + 1)))

    grad0 = np.linalg.norm(p.regularised_barrier_function(p.eps0, p.mu0, p.x0, p.x_regcentre)[1])
    outernorms = np.hstack((grad0, np.array([i['norm_grad_list'][-1] for i in p.log_inner])))

    fig, ax = plt.subplots(dpi=600)
    last = ks.size - cliplast
    ax.plot([], label='\u200B$|| \\nabla f_{\\epsilon, \\mu} (x^j_k) ||$')  # dummy plot for legend
    ax.plot(iter_marks[clipfirst:last], outernorms[clipfirst:last], color='teal', linewidth=1.2,
            label='$|| \\nabla f_{\\epsilon, \\mu} (x^j) ||$')
    for innerlog, kmin, kmax in zip(p.log_inner[clipfirst:last], iter_marks[clipfirst:last-1], iter_marks[clipfirst+1:last]):
        x = np.arange(kmin, kmax)
        y = np.array(innerlog['norm_grad_list'])
        for i in range(len(x)):
            ax.plot(x[i:i + 2], y[i:i + 2], color=redbluecmap(i / (len(x) - 0.9)), linewidth=.8, label='_nolegend_')

    ax.set_yscale('log')
    ax.set_title("Evolution of $|| \\nabla f_{\\epsilon, \\mu} (x^j_k) ||$")
    ax.legend(handler_map={plt.Line2D: GradientHandler()})
    ax.annotate('$\\bar{k}$', xy=(1.01, 0), xycoords='axes fraction', ha='left', va='bottom')
    return fig, ax


def plot_outer_kktres(p, clipfirst=0, cliplast=0):
    """
    Plots outer evolution of KKT residuum, i.e. ||grad f(xj) - mu sum grad g_i(xj) / gi(xj)|| + mu * sqrt(m)
    """
    kktres = []
    for x, mu in zip(p.log_outer['x_list'], p.log_outer['mu_list']):
        # compute values, gradients and auxiliary values
        _, gradf = p.f(x)
        valg, gradg = p.g(x)
        grad = gradf - mu * np.asarray((gradg / valg).sum(axis=1)).flatten()
        kktres.append(np.linalg.norm(grad) + np.sqrt(p.n_ineq_constraints) * mu)
    kktres = np.array(kktres)

    fig, ax = plt.subplots(dpi=600)
    last = kktres.size - cliplast
    ax.plot(np.arange(clipfirst + 1, last + 1), kktres[clipfirst:last], label='KKT residual')
    ax.set_yscale('log')
    ax.set_title("Evolution of the KKT residual")
    ax.legend(handler_map={plt.Line2D: GradientHandler()})
    ax.annotate('$j$', xy=(1.01, 0), xycoords='axes fraction', ha='left', va='bottom')
    return fig, ax


def plot_inner_stepsize(p, clipfirst=0, cliplast=0):
    """
    Plots step sizes alpha_k during the inner iterations
    with color gradient
    :param p: IneqConstProblem
    :return: fig, ax with plot
    """
    ks = np.array(p.log_outer['klbfgs_list'])
    iter_marks = np.hstack((0, np.cumsum(ks + 1)))

    fig, ax = plt.subplots(dpi=600)
    last = ks.size - cliplast
    for innerlog, kmin, kmax in zip(p.log_inner[clipfirst:last], iter_marks[clipfirst:last-1], iter_marks[clipfirst+1:last]):
        x = np.arange(kmin, kmax)
        y = np.array(innerlog['step_size_list']) + 1e-14
        for i in range(len(x)-1):
            # Note: In difference to other inner-plots, there is no k+1 information. Thus, runs only until len(x)-1
            ax.scatter(x[i], y[i], color=redbluecmap(i/(len(x)-1.9)), s=.3, label='_nolegend_')

    ax.set_yscale('log')
    ax.set_title("Evolution of the step sizes $\\alpha_k$")
    ax.plot([], label='\u200B$\\alpha_k$')  # dummy plot for legend
    ax.legend(handler_map={plt.Line2D: GradientHandler()})
    ax.annotate('$\\bar{k}$', xy=(1.01, 0), xycoords='axes fraction', ha='left', va='bottom')
    return fig, ax


def plot_outer_lbfgsiter(p, clipfirst=0, cliplast=0):
    """
    Plots number of l-BFGS iterations employed in each outer iteration
    """
    ks = np.array(p.log_outer['klbfgs_list'])

    fig, ax = plt.subplots(dpi=600)
    last = ks.size - cliplast
    ax.plot(np.arange(clipfirst, last), ks[clipfirst:last], label='l-BFGS iterations')
    ax.set_ylim([0, None])
    ax.set_title("Evolution of l-BFGS iterations")
    ax.legend(handler_map={plt.Line2D: GradientHandler()})
    ax.annotate('$j$', xy=(1.01, 0), xycoords='axes fraction', ha='left', va='bottom')
    return fig, ax






