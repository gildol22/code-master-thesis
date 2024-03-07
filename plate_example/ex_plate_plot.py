import numpy as np
from matplotlib import pyplot as plt
from matplotlib import tri

# This code was developed for the master thesis of Ole Gildemeister,
# written at the Institute of Mathematics and Image Computing, University of Luebeck.
# Code and thesis are available at https://github.com/gildol22/code-master-thesis

# This script generates a plot of the objective function and feasible set for the problem
# described in Example 3.7 for different parameter choices,
# corresponds to Figure 3.1


def create_triangs(x, y):
    t = tri.Triangulation(x, y)

    # find lowest boundary point and remove it from triangulation -> makes issues in triangulation
    problematic_points = np.where((np.round(t.x,13) == 1) + (np.round(t.x,13) == 3))
    mask = []
    for tr in t.triangles:
        mask.append((np.isin(tr, problematic_points).any()))

    t.set_mask(mask)
    return t


if __name__ == '__main__':

    n_points = 100
    n_vals = 100

    # generate function values
    zvals = np.linspace(0,2.25,n_vals)**2

    # compute and save coordinates corresponding to sought values in first quadrant
    xx = np.zeros((n_points, n_vals+1))
    yy = np.zeros_like(xx)
    zz = np.zeros_like(xx)

    for i in range(n_vals):
        z = zvals[i]
        x = np.linspace(1e-8, 1 + z**.25 - 1e-8,n_points)
        if ((z**.25 + 1)**2 - x**2 < 0).any():
            print('ALARM')
        y = np.sqrt((z**.25 + 1)**2 - x**2)

        xx[:,i] = x
        yy[:,i] = y
        zz[:,i] = z


    # add unit circle in first quadrant
    t = np.linspace(0,np.pi/2,n_points)
    xx[:,-1:] = np.outer(np.sin(t), np.array([1]))
    yy[:,-1:] = np.outer(np.cos(t), np.array([1]))


    # fill other quadrants, separated by feasible and infeasible area, flatten array and append centre (2,0,0)
    xx_feas = np.hstack((np.block([xx,-xx]).flatten() + 2, 2))
    yy_feas = np.hstack((np.block([yy, yy]).flatten(), 0))
    zz_feas = np.hstack((np.block([zz, zz]).flatten(), 0))

    xx_infeas = np.hstack((np.block([xx,-xx]).flatten() + 2, 2))
    yy_infeas = np.hstack((np.block([-yy,-yy]).flatten(), 0))
    zz_infeas = np.hstack((np.block([zz, zz]).flatten(), 0))


    # create triangulations for plot
    tri_feas = create_triangs(xx_feas, yy_feas)
    tri_infeas = create_triangs(xx_infeas, yy_infeas)


    # create plot and add surfaces of function
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=800)
    ax.plot_trisurf(tri_feas, zz_feas.flatten(), color='blue', alpha=.9, antialiased=False, zorder=-1)
    ax.plot_trisurf(tri_infeas,  zz_infeas.flatten(), color='rebeccapurple', alpha=.15, zorder=-1)

    # add unit circle for contour
    ax.plot(np.cos(2*t[1:]) + 2, np.sin(2*t[1:]), 0, color='darkblue', alpha=.9, zorder=5)
    ax.plot(np.cos(np.pi + 2*t) + 2, np.sin(np.pi + 2*t), 0, color='rebeccapurple', alpha=.1)

    # define plot limits
    xmin, xmax = [-.5, 4.5]
    ymin, ymax = [-2.5, 2.5]


    # add feasible and infeasible area
    ax.plot([xmin, xmax], [0,0], [0,0], color='red', linewidth=1, zorder=6)  # x_2-axis -> feasible set
    ax.plot_surface(np.array([[xmin, xmin], [xmax, xmax]]), np.array([[ymin, 0], [ymin, 0]]), np.array([[0,0], [0,0]]), color='red', alpha=.2)
    ax.plot_surface(np.array([[xmin, xmin], [xmax, xmax]]), np.array([[ymax, 0], [ymax, 0]]), np.array([[0,0], [0,0]]), color='green', alpha=.3)
    ax.text(3, -1.5, 0, '$x_2<0$', color='red')
    ax.text(0, 1.2, 0, "$\\mathcal{F}$", color='green')

    # add origin and minmisers
    ax.plot(0,0,0, '.', color='black', zorder=7)
    ax.text(-.2, -.6, 0, '(0,0)', color='black')
    ax.plot(1,0,0, '.', color='orange', zorder=15)
    ax.text(.9, -.5, 0, "$x^\\ast_M$", color='orange', zorder=15)
    ax.plot(2,1,0, '.', color='orange', zorder=7)
    ax.text(1.9, .5, 0, "$x^\\ast_{an}$", color='orange', zorder=7)

    # add axis labels
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    # set view settings
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_zlim([0, 5])
    ax.view_init(azim=-115, elev=25)

    # show and / or save result
    plt.savefig(f'../../figures/ex_plate_function', bbox_inches="tight")
    plt.show()




