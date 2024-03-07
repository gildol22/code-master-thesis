import matplotlib.pyplot as plt
from plot_functions import *

# This code was developed for the master thesis of Ole Gildemeister,
# written at the Institute of Mathematics and Image Computing, University of Luebeck.
# Code and thesis are available at https://github.com/gildol22/code-master-thesis


def generate_all_plots(p, name, fstar=None, xstar=None, clipfirst=0, cliplast=0, show=True):
    """
    Generates and saves all plots defined in plot_functions without clipping.
    :param p: IneqConstProblem which has been solved
    :param name: String containing the name which is added as prefix to the saved plots
    :param fstar: If available: Float with optimal function value
    :param xstar: If available: ndarray with optimal function value
    :param clipfirst: int - number of first outer iterations to skip
    :param cliplast: int - number of last outer iterations to skip
    :param show: boolean whether plt.show() is called in the end
    :return:
    """
    print('#############\nGenerating plots', end='')
    # generate all plots which don't require fstar or xstar
    plot_fctn_list = ['plot_outer_fvals_raw', 'plot_inner_fepsmuvals', 'plot_outer_gradfepsmu', 'plot_inner_gradfepsmu',
                      'plot_both_gradfepsmu', 'plot_outer_kktres', 'plot_inner_stepsize', 'plot_outer_lbfgsiter']
    for plot_fctn in plot_fctn_list:
        fig, _ = eval(plot_fctn)(p, clipfirst=clipfirst, cliplast=cliplast)
        fig.savefig('../figures/' + name + '_' + plot_fctn)
        print('.', end='')

    print(' Halfway through', end='')

    # generate all plots which require fstar
    plot_fctn_list = ['plot_outer_fvals_diff', 'plot_outer_fvals_resupper', 'plot_outer_fvals_update']
    for plot_fctn in plot_fctn_list:
        fig, _ = eval(plot_fctn)(p, clipfirst=clipfirst, cliplast=cliplast, fstar=fstar)
        fig.savefig('../figures/' + name + '_' + plot_fctn)
        print('.', end='')

    # generate all plots which require xstar
    plot_fctn_list = ['plot_outer_xdiff_raw', 'plot_outer_xdiff_update', 'plot_inner_xdiff', 'plot_both_xdiff']
    for plot_fctn in plot_fctn_list:
        fig, _ = eval(plot_fctn)(p, clipfirst=clipfirst, cliplast=cliplast, xstar=xstar)
        fig.savefig('../figures/' + name + '_' + plot_fctn)
        print('.', end='')

    print(' Done.')

    # show figures
    if show:
        plt.show()

    plt.close('all')  # all open plots are correctly closed after each run

    return

