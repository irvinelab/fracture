# fracture
Python codes for simulating and computing fracture paths in sheets draped on curved surfaces, 
released with N. P. Mitchell, V. Koning, V. Vitelli, W. T. M. Irvine, “Fracture in sheets 
draped on curved surfaces.” Nature Materials 16, 89-93 (2017)


Usage
-----
Cotterell & Rice perturbation theory evolution example is run via
$ python CotterellRice_curved_crack_bump_finite.py

Example usage for phase field evolution of fracture in the presence of a curved surface:
First install FEniCS from https://fenicsproject.org/ then run
$ python phasefield_iterative_tensiledamage_simple.py
Note that this example evolves a crack in a square domain with Dirichlet Boundary conditions.
These boundary conditions strongly affect the trajectory of the crack. To mimick the results 
of the paper https://www.nature.com/articles/nmat4733, create an XML mesh with circular boundary
conditions and replace the unit square mesh built-in from dolfin with your loaded XML mesh.


c. Noah P Mitchell, IrvineLab UChicago
