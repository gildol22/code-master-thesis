# Code to the master thesis from Ole Gildemeister
## Title: "An l-BFGS based Tikhonov-regularised interior-point method in large-scale nonlinear inequality-constrained convex optimisation"
### Advisors: Prof. Dr. Jan Lellmann, Dr. Florian Mannel
### Written at the Institute of Mathematics and Image Computing, University of Luebeck, Germany
### Study programme: Mathematik in Medizin und Lebenswissenschaften
### Date of submission: 07.03.2024

The code is written in Python and ordered in the following structure:
1. Kernel files (contains the Tikhonov-regularised interio-point method):
    * "qnregipm.py" (Quasi-Newton based regularised interior-point method) contains a class with a solver for purely inequality constrained problems
    * "qnregipmm.py" (Quasi-Newton based regularised interior-point method of multipliers) contains a class with a solver for inequality constrained problems with additional linear equality constraints
2. Numeric kernel files (contains the functions used for the numeric results presented in Chapter 5):
    * "generate_all_plots.py" is a function which generates all plots for a solved problem
    * "plot_function.py" contains all kernel functions called to generate a specific plot for a solved problem
    * "gp.py", contains a class which implements and generates an inequality-constrained geometric program as described in Section 5.2
    * "qcqp.py" contains a class which implements and generates a quadratically constrained quadratic program as described in Section 5.2
3. The script files used to generate the numerical results presented in Sections 5.3 and 5.4:
    * "run_rosenbrock.py", "run_gp_small.py", "run_qcqp_small.py" generates the results for the small-scale problems of Section 5.3
    * "run_gp_large.py", "run_qcqp_large.py" generates the results for the large-scale problems of Section 5.4
    * "run_gp.py", "run_qcqp.py" are backup files
4. Plate example files used to generate the plots for Example 3.7
    * "ex_plate_plot.py" plots the objective function and feasible set for the problem, corresponding to Figure 3.1
    * "ex_plate_minimisation" solves the ''plate problem'' described in Example 3.7 for different parameter choices and generates plots on the trajectories, corresponding to Figure 3.2
5. Figures
    * contains all figures on the evolution of the method for aforementioned numerical test cases
    * contains figures of Example 4.7
6. Results
    * contains backup-files from the numerical experiments


Finally, the master thesis itself is attached.

For questions and comments, please contact me by mail: ole.gildemeister@student.uni-luebeck.de.
