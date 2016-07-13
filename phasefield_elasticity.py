import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
import socket
import copy
import glob
try: import dolfin as dolf
except: 
    print '\n\nWARNING: Could not import dolfin\n (Limits usage of module phasefield_elasticity).'
from matplotlib import cm
try:
    from mayavi import mlab
    print 'Imported mayavi successfully!'
except:
    pass
    #print 'WARNING: Could not import mayavi-->This does not matter.'

hostname = socket.gethostname()
if hostname[0:6] != 'midway':
    import scipy
    from scipy.interpolate import griddata    
    try: import sympy as sp
    except: print 'WARNING: Could not import sympy!\n (Should not matter though).'
    from scipy.spatial import Delaunay


"""Module for elasticity of phase field modeling.

Table of Contents
-----------------
1. FEniCS definitions
        (these functions generally use UFL)
        generating dolfin meshes from saved triangulations, defining boundary conditions, library of surfaces,
        initial phase conditions for cracks and interacting cracks
2. Curvature Functions
        stresses from curvature and displacements from curvature and boundary stress; for Bumps, Spherical caps
3. Physical Observables
        stress-> strain, strain-> stress, compute Gaussian curvature of unstructured meshes, compute bond lengths
4. Lattice Generation
        generate a lattice given lattice vectors, create a Vogel-triangulated mesh (useful for triangulating disks)
        arrow function for making mesh arrows
5. Data Handling
        converting vectors and tensors from polar to cartesian, converting triangulations to bond lists,
        cutting bonds based on length, determining if points lie on lines or linesegments,
        calculating nearest points on a line segment or line, minimum distance from any linesegment in list, etc,
        creating unique-rowed arrays, matching orientations of triangulations, computing initial phase profile of a crack,
        kd trees, looking up values for a point based on proximity via a table,
        check if a variable is a number, rounding based on arbitrary thresholds, find a variable definition in a txt file        
6. Loading/Interpolating Data
        load and interpolating values from a file (such as griffith lengths), interpolate values from a meshgrid,
        interpolating values onto a mesh, load parameters from a parameters.txt file
7. Saving Data
        save height, save parameters.txt file
8. Plotting
        matplotlib made even easier --> plot tensors, vectors, or scalars with appropriate titles, limits, etc, all in one line
4. Specific Geometric Setups
        A. Inclined Crack in Uniaxial loading
        B. Quenched Glass Plate (QGP)
    

Dictionary of acronyms used in this doc
---------------------------------------
=======  ============
=======  ============
UFL      Unified Form Language, used by dolfin/FEniCS codes
QGP      quenched glass plate
ICUL     inclined crack under uniaxial loading
BC       boundary condition
BCUP     whether a boundary condition is dirichlet (U for displacement) or natural (P for traction)
ECUML    Edge Crack Under Mixed Loading
nn       nearest neighbors
BL       bond list (uppercase means 2D)
bL       bond length (lowercase means 1D)
P        traction at boundary or Peclet number, depending on the context
PURL     boundary condition situation with essential BC on left side of a plate and natural BC on the right side
=======  ============

List of Predefined Boundary Conditions
--------------------------------------
==========================   ============
For use with BCUP ==         'natural-essential':
==========================   ============
Usingleptcorner_Puniaxial    fix singlept in bottom left corner as u=(0,0), P is uniaxial
UsingleptcornerP*            fix singlept in bottom left corner as u=(0,0), P can take other configuration (see previous for example)
==========================   ============

==========================   ============
For use with BCUP ==         'essential' (applied to u) or 'natural' (applied to P, with U->E*U):
==========================   ============
uniaxialfree                 'U*(x[0]-xc)' for Vv.sub(0) --> constrain one dimension (x dim) on sides, also do DirichletBC(Vv.sub(1), Constant(0.0), boundary_singleptcorner)
uniaxialfreeX                constrain both dimensions on sides (free refers to the top and bottom)
uniaxialfreeY                constrain both dimensions on top and bottom (free refers to the left and right)
biaxial                      ('U*sqrt( pow(x[0]-xc,2)+pow(x[1]-yc, 2) )*cos(atan2(x[1]-yc,x[0]-xc))', 'U*sqrt( pow(x[0]-xc,2)+pow(x[1]-yc, 2) )*sin(atan2(x[1]-yc,x[0]-xc))')
fixleftX                     ('U*(x[0]-xc)' , '0.0') for bcu = dolf.DirichletBC(Vv, u_0, boundary_leftside), 
uniaxial-PURL                ('U*(x[0]-xc)' , '0.0')
uniaxialDisc                 ('U*sqrt( pow(x[0]-xc,2)+pow(x[1]-yc, 2) )*cos(atan2(x[1]-yc,x[0]-xc))' ,'0.0')
==========================   ============

==========================   ============
For use with BCUP ==         'essential':
==========================   ============
uniaxialmixedfree_u1s1       ('U*(x[0]-xc)' , 'U*(x[0]-xc)')
uniaxialmixedfree_uvals1     ('val*U*(x[0]-xc)' , 'U*(x[0]-xc)')
uniaxialmixedfree_u1sval     ('U*(x[0]-xc)' , 'val*U*(x[0]-xc)')
free                         no constraint
fixbotY                      ('0.' ,'0.') along bottom edge
fixtopY                      ('0.' ,'0.') along top edge
fixbotcorner                 ('0.' ,'0.') just in the corner, one mesh triangle
==========================   ============    

List of Predefined Surfaces
---------------------------
===========  ============
Surface      Description
===========  ============
bump
flat
sphere       spherical cap, radius x0, alph=R/x0
bumps2x1
QGP          Quenched Glass Plate --> tube with polynomial transition to exponential decay in radius
===========  ============
"""




##########################################
## 1. FEniCS definitions
##########################################

##############################
##############################
def planestress2strain_dolf(stress,E,nu,alph,T):
    """Assuming plane stress conditions, convert dolfin stress tensor object into dolfin strain tensor object"""
    strain = (1.+nu)/E * (stress - nu/(1.+nu)*dolf.tr(stress)*dolf.Identity(2))+ alph*T*dolf.Identity(2)
    return strain

def Vf2xy(Vf,mesh):
    """Return the xy pts associated with a Vector Function Space Vf.
    Parameters
    ----------
    Vf : Vector Function Space object (dolfin)
        description of surface 
        
    Returns
    -------
    """ 
    n = Vf.dim()
    dof_coordinates = Vf.dofmap().tabulate_all_coordinates(mesh)
    dof_coordinates.resize((n, 2))
    xy = np.dstack((dof_coordinates[:, 0] ,dof_coordinates[:, 1]))[0]
    return xy
    
def create_CGspaces(mesh,order=1):
    """Create Lagrangian tensor, vector, and scalar spaces of order 'order' from mesh.
    """
    print 'Creating Lagrangian function space...'
    Vt = dolf.TensorFunctionSpace(mesh, "Lagrange", order)
    Vv = dolf.VectorFunctionSpace(mesh, "Lagrange", order)
    Vf = dolf.FunctionSpace(mesh, "Lagrange", order)
    return Vt, Vv, Vf


def dolf_laplacian(f):
    """Using UFL, calc the laplacian of a scalar field"""
    return dolf.div(dolf.grad(f))

def Es_f(u,h,E,nu):
    """Using UFL, calculate the strain energy density of curved sheet => 0.5*epsilon*sigma, in plane stress approximation."""
    epsilon = dolf.sym(dolf.nabla_grad(u))
    epsf = 0.5* dolf.outer(dolf.grad(h),dolf.grad(h)) #curvature contribution to strain
    # PLANE STRESS
    sigma = E/(1+nu)* (epsilon) + E*nu/((1-nu**2)) * dolf.tr(epsilon)*dolf.Identity(2)
    sigf = E/(1+nu)* (epsf) + E*nu/((1-nu**2)) * dolf.tr(epsf)*dolf.Identity(2)
    return E*nu/(2*(1-nu**2))*tr(epsilon+epsf)*tr(epsilon+epsf) + E/(2*(1+nu))*inner(epsilon.T+epsf.T,epsilon+epsf)

def Ephi_f(u,h,E,nu,A):
    """Using UFL, calculate the strain energy density of curved sheet, modulated for tensile => 0.5*epsilon*sigma, in plane stress approximation."""
    epsilon_k = dolf.sym(dolf.nabla_grad(u))
    epsf = 0.5* dolf.outer(dolf.grad(h),dolf.grad(h)) #curvature contribution to strain
    # PLANE STRESS
    #sigma = E/(1+nu)* (epsilon_k) + E*nu/((1-nu**2)) * dolf.tr(epsilon_k)*dolf.Identity(2)
    #sigf = E/(1+nu)* (epsf) + E*nu/((1-nu**2)) * dolf.tr(epsf)*dolf.Identity(2)
    return dolf.conditional(dolf.lt(dolf.tr(epsilon_k),0), A*E/(4*(1-nu))*dolf.tr(epsilon_k)**2,\
                                                                   E/(4*(1-nu))*dolf.tr(epsilon_k)**2 ) +\
                E/(2*(1+nu)) * dolf.inner( epsilon_k.T-0.5*dolf.Identity(2).T*dolf.tr(epsilon_k) , \
                                           epsilon_k -0.5*dolf.Identity(2)*dolf.tr(epsilon_k) )

def Es(u,E,nu):
    """Using dolfin UFL, Calculate the strain energy density without curvature here => 0.5*epsilon*sigma, in plane stress approximation."""
    epsilon = dolf.sym(dolf.nabla_grad(u))
    return E*nu/(2*(1-nu**2))*dolf.tr(epsilon)*dolf.tr(epsilon) + E/(2*(1+nu))*dolf.inner(epsilon.T,epsilon)



def quasistatic_KKL_PDEs_iterative(mesh,Vv,Vf,phi_k,BCUP,BCtype,E,nu,U,R,N,surf,x0,Ec,kappa,chi,dt,\
                   alph='default',shape='square',tensile='on',A=-1.5):
    """Define simple version of iterative phase field PDEs in FEniCS,
    using CG1 elements. Boundaries are marked for uniaxial forcing (LR).
    Returns a long list of objects: u, phi, h, Vf, Vv, Vt, arngmt, au, Lu, epsilon_k, Es_phi, aphi, Lphi, ds
    
    Parameters
    ----------
    surf : string
        description of surface ('flat' 'sphere' 'bump' 'bumps2x1' 'QGP' etc)
    bcphi_on : int  (0,1,2)
        0: no phi BC
        1: phi held fixed via phik0 function/class/values
        (2: phi held at 1.0 on phi_boundary; this is just a convention and
        is indistinguishable from 1 in the context of this function)
    phik0 : class/dolfin object for evaluation
        if bcphi_on, pass this to get phi values on phi_boundary
    phi_boundary : dolfin object for phi boundaries
    alph : float or 'default' (string)
        aspect ratio parameter of surface. If 'default', will use default asp.ratio of
        that surface in function surf_vec()
    shape : string
        shape of the mesh
    tensile : string ('on' or 'off')
        whether to decrease phi only along tensile regions
    A : float
        value controlling the protection of compressed regions, should be <= 0
        
    Returns
    -------
    u, phi, h, arngmt, au, Lu, epsilon_k, Es_phi, aphi, Lphi, ds
    """
    # Dirichlet or Neumann BC
    if BCUP == 'essential':
        bcu, u_0 = def_bcu(Vv,BCtype,U,R=R,xc=0.,yc=0., N=N, shape=shape)
        P = 'none'
    elif BCUP == 'natural':
        bcu, u_0 = 'none', 'none'
        P = def_bcP(BCtype,U,E,R=R,xc=0.,yc=0.)
    elif BCUP == 'natural-essential':
        P = def_bcP(BCtype,U,E,R=R,xc=0.,yc=0.)
        bcu, u_0 = def_bcu(Vv,BCtype, U,R=R,xc=0.,yc=0.)
    
    # Check the boundary
    class CheckBoundary(dolf.SubDomain):
        def inside(self, x, on_boundary):
            """define the Dirichlet boundary as pts on the left and right sides that are on the boundary"""
            if N>60000:
                if shape=='square':
                    return on_boundary and (x[0]<-R*(0.98) or x[0]>R*(0.98)) and x[1]<R and x[1]>-R
                elif shape=='rectangle2x1':
                    return on_boundary and (x[0]<-R*(0.98) or x[0]>R*(0.98)) and x[1]<2*R*0.99 and x[1]>-2*R*0.99
                elif shape=='rectangle1x2':
                    return on_boundary and (x[0]<-2*R*(0.98) or x[0]>2*R*(0.98)) and x[1]<R*0.99 and x[1]>-R*0.99
            elif N<20000:
                if shape=='square':
                    return on_boundary and (x[0]<-R*(0.96) or x[0]>R*(0.96)) and x[1]<R*0.99 and x[1]>-R*0.99
                elif shape=='rectangle2x1':
                    return on_boundary and (x[0]<-R*(0.96) or x[0]>R*(0.96)) and x[1]<2*R*0.99 and x[1]>-2*R*0.99
                elif shape=='rectangle1x2':
                    return on_boundary and (x[0]<-2*R*(0.96) or x[0]>2*R*(0.96)) and x[1]<R*0.99 and x[1]>-R*0.99
            else:
                if shape=='square':
                    return on_boundary and (x[0]<-R*(0.98) or x[0]>R*(0.98)) and x[1]<R*0.99 and x[1]>-R*0.99
                elif shape=='rectangle2x1':
                    return on_boundary and (x[0]<-R*(0.98) or x[0]>R*(0.98)) and x[1]<2*R*0.99 and x[1]>-2*R*0.99
                elif shape=='rectangle1x2':
                    return on_boundary and (x[0]<-2*R*(0.98) or x[0]>2*R*(0.98)) and x[1]<R*0.99 and x[1]>-R*0.99
    
    check_boundary = CheckBoundary()
    
    boundary_parts = dolf.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_parts.set_all(0)
    check_boundary.mark(boundary_parts, 1)
    ds = dolf.Measure("ds")[boundary_parts]
    #dolf.plot(boundary_parts, interactive=True)
    
    ##############################
    # Define variational problem #
    ##############################
    print 'Defining test and trial functions...'
    v = dolf.TestFunction(Vf)
    tau = dolf.TestFunction(Vv)
    phi = dolf.TrialFunction(Vf)
    u = dolf.TrialFunction(Vv)
    
    ###############################
    # SURFACE #####################
    ###############################
    xy = Vf2xy(Vf,mesh)
    alph, x0, fv, arngmt = surf_vec(xy,surf,R=R,alph=alph,x0=x0)
    h = dolf.Function(Vf)
    h.vector()[:] = fv
    print 'maximum height is ', max(h.vector().array())
    
    # functions for variational problem
    d = u.geometric_dimension()
    epsilon = dolf.sym(dolf.nabla_grad(u))
    epsf = 0.5* dolf.outer(dolf.grad(h),dolf.grad(h)) #curvature contribution to strain
    # PLANE STRESS
    sigma = E/(1+nu)* (epsilon) + E*nu/((1-nu**2)) * dolf.tr(epsilon)*dolf.Identity(2)
    sigf = E/(1+nu)* (epsf) + E*nu/((1-nu**2)) * dolf.tr(epsf)*dolf.Identity(2)
    #Es = E*nu/(2*(1-nu**2))*tr(epsilon+epsf)*tr(epsilon+epsf) + mu*inner(epsilon.T+epsf.T,epsilon+epsf)
    g = 4*phi_k**3 - 3*phi_k**4
    gprime = 12*(phi_k**2-phi_k**3)
    
    
    #################
    # ITERATIVE
    #################
    print('Defining force balance...')
    # Force balance (a(u,v) = L(v) = Integrate[f*v, dx] )
    au = -g*dolf.inner(sigma, dolf.sym(dolf.nabla_grad(tau)))*dolf.dx \
         + gprime*dolf.inner(sigma, dolf.outer(dolf.grad(phi_k),tau) )*dolf.dx #using symmetry properties of sigma
    if BCUP == 'essential':
        Lu = g*dolf.inner(sigf, dolf.sym(dolf.nabla_grad(tau)))*dolf.dx \
             - gprime*dolf.inner(sigf, dolf.outer(dolf.grad(phi_k),tau) )*dolf.dx
    elif BCUP == 'natural' or BCUP=='natural-essential':
        Lu = g*dolf.inner(sigf, dolf.sym(dolf.nabla_grad(tau)))*dolf.dx \
             - gprime*dolf.inner(sigf, dolf.outer(dolf.grad(phi_k),tau) )*dolf.dx \
             -dolf.dot(P,tau)*ds(1) - dolf.dot(P,tau)*ds(2)
    u = dolf.Function(Vv)
    if BCUP == 'essential' or BCUP=='natural-essential':
        dolf.solve(au == Lu, u, bcu, solver_parameters={'linear_solver': 'gmres',  'preconditioner': 'petsc_amg'}) 
    if BCUP == 'natural':
        dolf.solve(au == Lu, u, solver_parameters={'linear_solver': 'gmres',  'preconditioner': 'petsc_amg'})
        
    # dphi/dt
    print('Setting up Linear problem for Phi-- Implicit Euler')
    epsilon_k = dolf.sym(dolf.nabla_grad(u)) + epsf
    if tensile == 'on':
        A = -1.5
        Es_phi = dolf.conditional(dolf.lt(dolf.tr(epsilon_k),0), A*E/(4*(1-nu))*dolf.tr(epsilon_k)**2,\
                                                                   E/(4*(1-nu))*dolf.tr(epsilon_k)**2 ) +\
                E/(2*(1+nu)) * dolf.inner( epsilon_k.T-0.5*dolf.Identity(d).T*dolf.tr(epsilon_k) , \
                                           epsilon_k -0.5*dolf.Identity(d)*dolf.tr(epsilon_k) )
    else:
        Es_phi =  E/(4*(1-nu))*dolf.tr(epsilon_k)**2  +\
                E/(2*(1+nu)) * dolf.inner( epsilon_k.T-0.5*dolf.Identity(d).T*dolf.tr(epsilon_k) ,\
                                           epsilon_k -0.5*dolf.Identity(d)*dolf.tr(epsilon_k) )
    
    # conditional(condition, true_value, false_value)
    aphi = phi*v*dolf.dx + dt*chi*kappa* dolf.dot(dolf.nabla_grad(phi), dolf.nabla_grad(v)) *dolf.dx
    fphi = phi_k  - dt*chi*( gprime *(Es_phi -Ec) )
    Lphi = fphi*v*dolf.dx
    phi = dolf.Function(Vf)
    
    return u, phi, h, bcu, u_0, P, alph, arngmt, au, Lu, g, gprime, epsilon_k, Es_phi, aphi, Lphi, ds



def def_bcu(Vv,BCtype,U,R=0.06,xc=0.,yc=0., val=1.0, N=40000,shape='square'):
    """Define (fenics mesh) boundary condition bcu with appropriate u_0 from prescribed params.
    This function also assigns the boundary condition to the (hopefully) appropriate boundary.
    
    WARNING: Recently (12-2015?) changed this function to have TWO outputs (including u_0).
    FEniCS will fail to solve if it thinks the boundary condition is the output tuple!
    
    Parameters
    ----------
    Vv : Vector Function Space instance
    BCtype : string
    U : float
        The magnitude of the displacement or stress/E
    R : float
        Half the characteristic extent of the mesh
    xc : float
        The center x coordinate of the boundary
    """
    
    if BCtype=='shear':
        print 'Creating shear BCs...'
        u_0 = dolf.Expression(('0.0' ,
                          'U*(x[0]-xc)'), U=U, xc=xc)
    elif BCtype=='biaxial':
        print 'Creating biaxial BCs...'
        u_0 = dolf.Expression(('U*sqrt( pow(x[0]-xc,2)+pow(x[1]-yc, 2) )*cos(atan2(x[1]-yc,x[0]-xc))' ,
                          'U*sqrt( pow(x[0]-xc,2)+pow(x[1]-yc, 2) )*sin(atan2(x[1]-yc,x[0]-xc))'), U=U, xc=xc, yc=yc)
    elif BCtype == 'uniaxial' or BCtype =='fixleftX' or BCtype =='uniaxial-PURL':
        print 'Creating uniaxial BCs: (U*(x[0]-xc) , 0.0) ...'
        u_0 = dolf.Expression(('U*(x[0]-xc)' , '0.0'), U=U, xc=xc, yc=yc)
    elif BCtype == 'uniaxialfree':
        print 'Creating uniaxialfree BCs (constrained in 1D only)...'
        u_0 = dolf.Expression('U*(x[0]-xc)', U=U, xc=xc, yc=yc)
        # testing 20151103
        #u_0 = dolf.Expression(('U*(x[0]-xc)','0.0'), U=U, xc=xc, yc=yc)
    elif BCtype == 'uniaxialfreeX':
        print 'Creating uniaxialfreeX BCs (fix LR and pull in X )...'
        u_0 = dolf.Expression(('U*(x[0]-xc)','0.0'), U=U, xc=xc, yc=yc)
    elif BCtype == 'uniaxialfreeY':
        print 'Creating uniaxialfreeY BCs (fix TB and pull in Y)...'
        u_0 = dolf.Expression(('0.0','U*(x[1]-xc)'), U=U, xc=xc, yc=yc)
    elif BCtype == 'uniaxialDisc':
        print 'Creating uniaxial BCs on disc...'
        u_0 = dolf.Expression(('U*sqrt( pow(x[0]-xc,2)+pow(x[1]-yc, 2) )*cos(atan2(x[1]-yc,x[0]-xc))' ,
                          '0.0'), U=U, xc=xc, yc=yc)
    elif BCtype == 'uniaxialmixedfree_u1s1' :
        print 'Creating mixed BCs...'
        u_0 = dolf.Expression(('U*(x[0]-xc)' , 'U*(x[0]-xc)'), U=U, xc=xc, yc=yc)
    elif BCtype == 'uniaxialmixedfree_uvals1':
        print 'Creating mixed BCs...'
        u_0 = dolf.Expression(('val*U*(x[0]-xc)' , 'U*(x[0]-xc)'), U=U, xc=xc, yc=yc, val=val)
    elif BCtype == 'uniaxialmixedfree_u1sval':
        print 'Creating mixed BCs: tensile U*(x[0]-xc), but shear val*U*(x[0]-xc) applied to sides...'
        u_0 = dolf.Expression(('U*(x[0]-xc)' , 'val*U*(x[0]-xc)'), U=U, xc=xc, yc=yc, val=val)
    elif BCtype == 'mixed_u1s1'  :
        print 'Creating Dirichlet BCs (mixed or fixbotY)...'
        u_0 = dolf.Expression(('U*(x[1]-xc)' ,'U*(x[1]-yc)'), U=U, xc=xc, yc=yc)
    elif BCtype == 'free' or BCtype == 'fixbotY' or BCtype == 'fixtopY' or BCtype=='fixbotcorner' \
                 or BCtype[0:15]=='Usingleptcorner':
        print 'Defining u_0 = (0,0)...'
        u_0 = dolf.Expression(('0.' ,'0.'), U=U, xc=xc, yc=yc)
    
    
    ## APPLY BOUNDARY CONDITION
    bcu = apply_bcu(Vv,BCtype,u_0,R,xc,yc,N,shape)
    
    return bcu, u_0

def apply_bcu(Vv,BCtype,u_0,R,xc,yc,N,shape):
    """Apply FEniCS boundary condition to boundary determined by the library of definitions inside this def"""
    def boundary(x, on_boundary):
        """define the Dirichlet boundary as pts on boundary"""
        return on_boundary
    
    def boundary_sides(x, on_boundary):
        """define the Dirichlet boundary as pts on the left and right sides that are on the boundary"""
        if N>60000:
            if shape=='square':
                return on_boundary and (x[0]<-R*(0.98) or x[0]>R*(0.98)) and x[1]<R and x[1]>-R
            elif shape=='rectangle2x1':
                return on_boundary and (x[0]<-R*(0.98) or x[0]>R*(0.98)) and x[1]<2*R*0.99 and x[1]>-2*R*0.99
            elif shape=='rectangle1x2':
                return on_boundary and (x[0]<-2*R*(0.98) or x[0]>2*R*(0.98)) and x[1]<R*0.99 and x[1]>-R*0.99
        elif N<20000:
            if shape=='square':
                return on_boundary and (x[0]<-R*(0.96) or x[0]>R*(0.96)) and x[1]<R*0.99 and x[1]>-R*0.99
            elif shape=='rectangle2x1':
                return on_boundary and (x[0]<-R*(0.96) or x[0]>R*(0.96)) and x[1]<2*R*0.99 and x[1]>-2*R*0.99
            elif shape=='rectangle1x2':
                return on_boundary and (x[0]<-2*R*(0.96) or x[0]>2*R*(0.96)) and x[1]<R*0.99 and x[1]>-R*0.99
        else:
            if shape=='square':
                return on_boundary and (x[0]<-R*(0.98) or x[0]>R*(0.98)) and x[1]<R*0.99 and x[1]>-R*0.99
            elif shape=='rectangle2x1':
                return on_boundary and (x[0]<-R*(0.98) or x[0]>R*(0.98)) and x[1]<2*R*0.99 and x[1]>-2*R*0.99
            elif shape=='rectangle1x2':
                return on_boundary and (x[0]<-2*R*(0.98) or x[0]>2*R*(0.98)) and x[1]<R*0.99 and x[1]>-R*0.99
    
    def boundary_topbot(x, on_boundary):
        """define the Dirichlet boundary as pts on the left and right sides that are on the boundary"""
        return on_boundary and (x[1]<-R*(0.98) or x[1]>R*(0.98))
    
    def boundary_leftside(x, on_boundary):
        """define the Dirichlet boundary as pts on the left and right sides that are on the boundary"""
        return on_boundary and x[0]<-R*(0.98) 
    
    def boundary_topside(x, on_boundary):
        """define the Dirichlet boundary as pts on the top side that are on the boundary"""
        return on_boundary and x[1]>R and x[0]<R*0.1 and x[0]>-R*0.1
    
    def boundary_bottomside(x, on_boundary):
        """define the Dirichlet boundary as pts on the bottom side that are on the boundary"""
        return on_boundary and x[1]<-R and x[0]<R*0.1 and x[0]>-R*0.1
    
    def boundary_bottomcorner(x, on_boundary):
        """define the Dirichlet boundary as pts on the bottom left side that are on the boundary"""
        return on_boundary and x[1]<-R and x[0]<-R*0.97
    
    def boundary_singleptcorner(x, on_boundary):
        """define the Dirichlet boundary on a very small number of facets in the bottom left corner"""
        if shape == 'square':
            if N>60000:
                return on_boundary and x[1]<-R*(-0.99) and x[0]<-R*(-0.99)
            elif N<20000:
                return on_boundary and x[1]<R*(-1.96) and x[0]< R*(-0.96)
            else:
                return on_boundary and x[1]<R*(-0.98) and x[0]< R*(-0.98)
        elif shape == 'rectangle2x1':
            if N>60000:
                return on_boundary and x[1]<R*(-1.99) and x[0]< R*(-0.99)
            elif N<20000:
                return on_boundary and x[1]<R*(-1.96) and x[0]< R*(-0.96)
            else:
                return on_boundary and x[1]<R*(-1.97) and x[0]< R*(-0.98)
 
    ## APPLY BOUNDARY CONDITION
    if BCtype == 'uniaxialmixedfree_u1s1' or BCtype == 'uniaxialmixedfree_uvals1' or BCtype == 'uniaxialmixedfree_u1sval':
        bcu = dolf.DirichletBC(Vv, u_0, boundary_sides)
    elif BCtype == 'fixtopY':
        bcu = dolf.DirichletBC(Vv, u_0, boundary_topside)
    elif BCtype == 'fixbotY':
        bcu = dolf.DirichletBC(Vv, u_0, boundary_bottomside)
    elif BCtype == 'fixbotcorner':
        bcu = dolf.DirichletBC(Vv, u_0, boundary_bottomcorner)
    elif BCtype == 'uniaxialfree':
        """only constrain one dimension along sides"""
        bc0 = dolf.DirichletBC(Vv.sub(0), u_0, boundary_sides)
        u_1 = dolf.Constant(0.0)
        bc1 = dolf.DirichletBC(Vv.sub(1), u_1, boundary_singleptcorner)
        # Collect boundary conditions
        bcu = [bc0, bc1]
        # testing 20151103
        #bcu = dolf.DirichletBC(Vv, u_0, boundary_sides)
    elif BCtype[0:15] == 'Usingleptcorner':
        bcu = dolf.DirichletBC(Vv, u_0, boundary_singleptcorner)
    elif BCtype == 'uniaxialfreeX':
        print 'constrain both dimensions on sides'
        bcu = dolf.DirichletBC(Vv, u_0, boundary_sides)
    elif BCtype == 'uniaxialfreeY':
        print 'constrain both dimensions on top and bottom'
        bcu = dolf.DirichletBC(Vv, u_0, boundary_topbot)
    elif BCtype == 'fixleftX' or BCtype =='uniaxial-PURL':
        bcu = dolf.DirichletBC(Vv, u_0, boundary_leftside)
    elif BCtype != 'free':
        print 'Assigning BCs to all boundaries...'
        bcu = dolf.DirichletBC(Vv, u_0, boundary)
    
    return bcu



def def_bcP(BCtype,U,E,R=0.06,xc=0.,yc=0.):
    """Define the traction on the boundary of a sample."""
    if BCtype == 'uniaxial' or 'uniaxial-PURL' or BCtype == 'Usingleptcorner_Puniaxial':
        P = dolf.Expression(('E*U*(x[0]-xc)/R','0.0'),U=U,E=E,xc=xc,R=R)
    elif BCtype == 'uniaxialmixed_u1s1':
        P = dolf.Expression(('E*U*(x[0]-xc)/R','E*U*(x[0]-xc)/R'),U=U,E=E,xc=xc,R=R)
    elif BCtype == 'shear':
        P = dolf.Expression(('0','E*U'),U=U,E=E,xc=xc,R=R)
    elif BCtype == 'free' or BCtype == 'fixbotY' or BCtype == 'fixbotcorner' or BCtype == 'fixtopY':
        P = dolf.Constant(('0','0'))
    elif BCtype == 'biaxial':
        P = dolf.Expression(('E*U*sqrt( pow(x[0]-xc,2)+pow(x[1]-yc, 2) )*cos(atan2(x[1]-yc,x[0]-xc))/R' ,
                          'E*U*sqrt( pow(x[0]-xc,2)+pow(x[1]-yc, 2) )*sin(atan2(x[1]-yc,x[0]-xc))/R'), U=U, E=E, xc=xc, yc=yc, R=R)
    return P


def genmesh(shape,meshtype,N,xi,theta,R,eta):
    """load correct mesh from FEniCS/meshes/ directory
    """
    Rstr = '{0:.3f}'.format(R).replace('.','p') 
    etastr = '{0:.3f}'.format(eta).replace('.','p')
    if shape == 'square':
        nx = int(np.sqrt(N))
        meshd = nx/(2*R)*float(xi)
        if meshtype == 'UnitSquare':
            print 'Creating unit square mesh of ', meshtype, ' lattice topology...'
            mesh = dolf.UnitSquareMesh(nx, nx)
        else:
            print 'Creating square-shaped mesh of ', meshtype, ' lattice topology...'
            meshfile = '../meshes/'+shape+'Mesh_'+meshtype+'_eta'+etastr+'_R'+Rstr+'_N'+str(int(N))+'.xml'
            mesh = dolf.Mesh(meshfile)
    elif shape == 'circle' :
        print 'Creating circle-shaped mesh of ', meshtype, ' lattice topology...'
        meshd = 2*np.sqrt(N/np.pi)*float(xi)/(2*R)
        if meshtype == 'Trisel':
            add_exten = '_Nsp'+str(Nsp)+'_H'+'{0:.2f}'.format(H/R).replace('.','p')+\
                    '_Y'+'{0:.2f}'.format(Y/R).replace('.','p')+\
                    '_beta'+'{0:.2f}'.format(beta/np.pi).replace('.','p') +\
                    '_theta'+'{0:.2f}'.format(theta/np.pi).replace('.','p') 
        else:
            add_exten = ''
            
        meshfile = '../meshes/'+shape+'Mesh_'+meshtype+add_exten+'_eta'+etastr+'_R'+Rstr+'_N'+str(int(N))+'.xml'
        mesh = dolf.Mesh(meshfile)
    elif shape == 'rectangle2x1' or shape == 'rectangle1x2' :
        print 'Creating circle-shaped mesh of ', meshtype, ' lattice topology...'
        meshd = np.sqrt(N*0.5)*float(xi)/(2*R)
        if meshtype == 'Trisel':
            H, Y, beta = crack_loc()
            add_exten = '_Nsp'+str(Nsp)+'_H'+'{0:.2f}'.format(H/R).replace('.','p')+\
                    '_Y'+'{0:.2f}'.format(Y/R).replace('.','p')+\
                    '_beta'+'{0:.2f}'.format(beta/np.pi).replace('.','p') +\
                    '_theta'+'{0:.2f}'.format(theta/np.pi).replace('.','p') 
        else:
            add_exten = ''
            
        meshfile = '../meshes/'+shape+'Mesh_'+meshtype+add_exten+'_eta'+etastr+'_R'+Rstr+'_N'+str(int(N))+'.xml'
        print 'loading meshfile = ', meshfile
        mesh = dolf.Mesh(meshfile)
        
    print 'found meshfile = ', meshfile
    return mesh, meshd, meshfile



##########################################
## 2. Curvature Functions
##########################################

# stress from gaussian curvature, infinite system. Y=1. alpha=1. w=1.
def GB_sigma_theta(t):
    """Return the azimuthal Stress from a gaussian bump (div by Young's modulus) from analytic expression"""
    return 1/8. * ( -t**-2 * (1 - np.exp(-t*t)) + 2 * np.exp(-t*t) )

def GB_sigma_r(t):
    """Return the radial Stress from a gaussian bump (div by Young's modulus) from analytic expression"""
    return 1/8. * ( t**-2 * (1 - np.exp(-t*t)) )

def GB_P_uDirichlet(alph,x0,R,U,nu):
    """The boundary stress, given the bump profile and boundary displacement"""
    return 1/(1-nu)*(U - 0.25*alph**2 *((x0/R)**2 *( np.exp(-(R/x0)**2)-1.)+ np.exp(-(R/x0)**2)))

def GB_U_from_P(alph,x0,R,P,nu):
    """The boundary stress, given the bump profile. P is supplied in units of E."""
    U = (1-nu)*P + (alph/2.)**2. * ((x0/R)**2.*(np.exp(-R**2./x0**2.)-1.0) + np.exp(-R**2./x0**2.))
    return U

def U_displ_constP_changealpha(nu,U_old,alpha_old,x0,R,alpha_new):
    """Return the displacement for a new aspect ratio such that
    the boundary stress of a disc sample of radius R has the same sigma_rr at r=R,
    ie. Pstar, given in units of E"""
    #0.0, 0.2121320343559643, .4242640687119285, 0.7064460135092848
    #returns the new U (displacement)
    alphaterm_old =  (alpha_old/2.)**2. * ((x0/R)**2.*(np.exp(-R**2./x0**2.)-1.0) + np.exp(-R**2./x0**2.))
    Pstar = 1./(1.-nu)*(U_old - alphaterm_old)
    alphaterm_new =  (alpha_new/2.)**2. * ((x0/R)**2.*(np.exp(-R**2./x0**2.)-1.0) + np.exp(-R**2./x0**2.))
    U_new = (1.-nu) * Pstar + alphaterm_new                    
    return U_new

def U_displ_constP_changeR(nu,U_old,R_old,x0,R,alph):
    """Return the displacement for a new sample size such that
    the boundary stress of a disc sample of radius R has the same sigma_rr at r=R,
    ie. Pstar, given in units of E"""
    #0.0, 0.2121320343559643, .4242640687119285, 0.7064460135092848
    #returns the new U (displacement)
    alphaterm_old =  (alph/2.)**2. * ((x0/R_old)**2.*(np.exp(-R_old**2./x0**2.)-1.0) + np.exp(-R_old**2./x0**2.))
    Pstar = 1./(1.-nu)*(U_old - alphaterm_old)
    alphaterm_new =  (alph/2.)**2. * ((x0/R)**2.*(np.exp(-R**2./x0**2.)-1.0) + np.exp(-R**2./x0**2.))
    U_new = (1.-nu) * Pstar + alphaterm_new                    
    return U_new

def GB_stress_uDirichlet(alph,x0,R,U,nu,t):
    """computes analytical Stress (Srr,Stt) for Gaussian Bump
    
    Parameters
    ----------
    alph : float
        aspect ratio of the bump
    x0 : float
        the width of the bump
    R : float
        radius of the disc
    U : float
        displacement at the boundary
    nu : float
        Poisson ratio of the sheet
    t : N x 1 array
        radial points on which to evaluate the displacement
        
    Returns
    ----------
    Srr, Stt : N x 1 arrays
        the radial and azimuthal stresses of the sheet
    """
    P = GB_P_uDirichlet(alph,x0,R,U,nu)
    Srr = alph**2/8. * ( (x0/t)**2 * ( 1. - np.exp(-(t/x0)**2) ) + (x0/R)**2 * ( np.exp(-(R/x0)**2) - 1.0 ) ) + P 
    Stt = alph**2/8. * ( (x0/t)**2 * ( np.exp(-(t/x0)**2) - 1. ) + 2. * np.exp(-(t/x0)**2) + (x0/R)**2 * ( np.exp(-(R/x0)**2) - 1.0 ) ) + P
    return Srr, Stt


def GB_displacement_uDirichlet(alph,x0,R,U,nu,t):
    """computes analytical displacement (radial only, since ut = 0) for Gaussian Bump
    
    Parameters
    ----------
    alph : float
        aspect ratio of the bump
    x0 : float
        the width of the bump
    R : float
        radius of the disc
    U : float
        displacement at the boundary
    nu : float
        Poisson ratio of the sheet
    t : N x 1 array
        radial points on which to evaluate the displacement
        
    Returns
    ----------
    ur : N x 1 array
        the radial displacement of the sheet
    """
    Srr, Stt = GB_stress_uDirichlet(alph,x0,R,U,nu,t)
    err = Srr- nu*Stt ;
    ett = Stt- nu*Srr ;
    ur = t * ett;
    return ur

def GB_strain_uDirichlet(alph,x0,R,U,nu,t):
    """computes analytical strain for Gaussian Bump under constant displacement BCs
    
    Parameters
    ----------
    alph : float
        aspect ratio of the bump
    x0 : float
        the width of the bump
    R : float
        radius of the disc
    U : float
        displacement at the boundary
    nu : float
        Poisson ratio of the sheet
    t : N x 1 array
        radial points on which to evaluate the displacement
        
    Returns
    ----------
    ur : N x 1 array
        the radial displacement of the sheet
    """
    Srr, Stt = GB_stress_uDirichlet(alph,x0,R,U,nu,t)
    err = Srr- nu*Stt ;
    ett = Stt- nu*Srr ;
    return err,ett
    

### Sphere stress and strain
def Sphere_sigma_theta(G,R,t): 
    return G/16.*(R**2 - 3*t**2)

def Sphere_sigma_r(G,R,t): 
    return G/16.*(R**2 - t**2)

def Sphere_stress_uDirichlet(G,R,U,nu,t):
    """compute analytical Stress (Srr,Stt) for Sphere, with boundary stress included.
    Assumes azimuthally symmetric sample of radius R.
    
    Parameters
    ----------
    G : float
        Gaussian curvature of the sphere = 1/R^2
    R : float
        radius of the disc
    U : float
        displacement at the boundary
    nu : float
        Poisson ratio of the sheet
    t : N x 1 array
        radial points on which to evaluate the displacement
        
    Returns
    ----------
    Srr, Stt : N x 1 arrays
        the radial and azimuthal stresses of the sheet
    """
    sigmatt_R = G/16.*(R**2 - 3*R**2) 
    P = U/(1-nu) - sigmatt_R
    Srr = G/16.*(R**2 - t**2) + P
    Stt = G/16.*(R**2 - 3*t**2) + P
    return Srr, Stt

def Sphere_strain_uDirichlet(G,R,U,nu,t):
    """computes analytical displacement (radial only, since ut = 0) for disc sample on Sphere 
    
    Parameters
    ----------
    G : float
        Gaussian curvature of the sphere
    R : float
        radius of the disc sample
    U : float
        displacement at the boundary
    nu : float
        Poisson ratio of the sheet
    t : N x 1 array
        radial points on which to evaluate the displacement
        
    Returns
    ----------
    ur : N x 1 array
        the radial displacement of the sheet
    """
    Srr, Stt = Sphere_stress_uDirichlet(G,R,U,nu,t)
    err = Srr- nu*Stt ;
    ett = Stt- nu*Srr ;
    return err,ett    
    
def Sphere_displacement_uDirichlet(G,R,U,nu,t):
    """computes analytical displacement (radial only, since ut = 0) for Sphere
    
    Parameters
    ----------
    alph : float
        aspect ratio of the bump
    x0 : float
        the width of the bump
    R : float
        radius of the disc
    U : float
        displacement at the boundary
    nu : float
        Poisson ratio of the sheet
    t : N x 1 array
        radial points on which to evaluate the displacement
        
    Returns
    ----------
    ur : N x 1 array
        the radial displacement of the sheet
    """
    Srr, Stt = Sphere_stress_uDirichlet(G,R,U,nu,t)
    err = Srr- nu*Stt ;
    ett = Stt- nu*Srr ;
    ur = t * ett;
    return ur
    
### Sphere displ and P functs ###

def Sphere_P_displ(nu, U, G, R):
    """Displacement-based boundary stress calculation"""
    sigmatt_R = Sphere_sigma_theta(G,R,R) 
    return U/(1-nu) - sigmatt_R
    
def Sphere_U_displ_constP_changeG(nu, U_old, G_old, R, G_new):
    """Given displacement and alpha, find new displacement for New alpha such that boundary stress_rr is same.
    Assumes a spherical cap surface with rotationally symmetric BCs.
    Returns the new U (displacement)"""
    U_new = U_old - (1-nu)*(Sphere_sigma_theta(G_old, R, R) - Sphere_sigma_theta(G_new, R, R))
    return U_new

def Sphere_woR_displ(nu, P, G, R):
    """Solve for the radial displacement w/R given the radial stress P"""
    return (1-nu)*P + Sphere_sigma_theta(G, R, R)



##########################################
## 3. Physical Observables
##########################################
def gaussian_curvature(Z, dx=1.):
    """Compute gaussian curvature from 2D grid of z values, sampled at pts with spacing dx.
    NOTE: For now, both dimensions must have the same increment dx."""
    Zy, Zx = np.gradient(Z, dx)                                                     
    Zxy, Zxx = np.gradient(Zx, dx)                                                  
    Zyy, _ = np.gradient(Zy, dx)                                                    
    K = (Zxx * Zyy - (Zxy ** 2)) /  (1 + (Zx ** 2) + (Zy **2)) ** 2             
    return K

def gaussian_curvature_unstructured(x,y,z,dx,N=3):
    """Computes Gaussian curvature of unstructured xyz data (not a mesh)
    by doing cubic interpolation and making a lookup table.
    This was formerly known as gaussian_curavture_unstructured2()
    
    Parameters
    ----------
    dx : float
        distance between grid pts in grid that spans (x,y). Should be similar to actual distance btwn pts in arrays x and y,
        but can be smaller since interpolation is cubic, not linear.
    N : int
        number of evaluation points for meshgrid interpolation of CURVATURE
    
    Returns
    ----------
    K_unstructured : Mx1 array
        Gaussian curvature evaluated at the input data points
    xgrid : NxN array
        the X grid data used to compute the Gaussian curvature, a mesh defined over the input x
    ygrid : NxN array
        the Y grid data used to compute the Gaussian curvature, a mesh defined over the input x
    Kgrid : NxN array
        Gaussian curvature evaluated on the grid xgrid,ygrid
    """
    xg = np.arange(np.nanmin(x)-dx,np.nanmax(x)+dx,dx)
    yg = np.arange(np.nanmin(y)-dx,np.nanmax(y)+dx,dx)
    X,Y = np.meshgrid(xg,yg)
    zgrid = interpolate_onto_mesh(x,y,z, X, Y, mask=False)    
    
    # generate 2 2d grids for the x & y bounds
    xv = X.ravel()
    yv = Y.ravel()
    zv = zgrid.ravel()
    # surface params (s for surface)
    Kgrid = gaussian_curvature(zgrid,dx)
    Kv = Kgrid.ravel()
    xy_pts = np.dstack((x,y))[0]
    #print 'xy_pts=', xy_pts
    lookupXYZ = np.dstack((xv,yv,Kv))[0]
    #print 'lookupXYZ=', lookupXYZ
    K = lookupZ_avgN(lookupXYZ,xy_pts,N=N)[:,2].ravel()
    return K, X, Y, Kgrid


def gaussian_curvature_bspline(z, x, y, N=100):
    """Compute gaussian curvature from 2D grid of z values, sampled at unstructured mesh points x,y.
    Unfortunately, for now, the x and y data is sampled on a SQUARE patch with the same numerical range in values.
    This can be improved in the future by allowing gaussian_curvature(Z) to accomodate rectangular meshes (with dx and dy different).
    See also gaussian_curvature_unstructured for a generally faster performance.
    
    Parameters
    ----------
    z : array of dimension Mx1
        Height of surface
    x : array of dimension Mx1
        x positions of evaluated height
    y : array of dimension Mx1
        y positions of evaluated height
    N : int
        number of evaluation points for meshgrid interpolation of height 
        
    Returns
    ----------
    K_unstructured : Mx1 array
        Gaussian curvature evaluated at the input data points
    xgrid : NxN array
        the X grid data used to compute the Gaussian curvature, a mesh defined over the input x
    ygrid : NxN array
        the Y grid data used to compute the Gaussian curvature, a mesh defined over the input x
    Kgrid : NxN array
        Gaussian curvature evaluated on the grid xgrid,ygrid
    """
    dxx = (np.max(y)-np.min(y))/N
    # generate 2 2d grids for the x & y bounds
    xlin = np.arange(np.min(x), np.max(x) + dxx, dxx)
    ylin = np.arange(np.min(x), np.max(x) + dxx, dxx)
    xgrid, ygrid = np.mgrid[ slice(np.min(x), np.max(x) + dxx, dxx),
                           slice(np.min(x), np.max(x) + dxx, dxx)]
    Zintp = griddata(np.dstack((x,y))[0], z, (xgrid, ygrid), method='cubic')
    Kgrid = gaussian_curvature(Zintp, dxx)
    Kintp = scipy.interpolate.RectBivariateSpline(xlin, ylin, Kgrid)
    K_unstructured = Kintp.ev(x, y)
    return K_unstructured, xgrid, ygrid, Kgrid

def strain2stress_polar(err,ett,ert,nu):
    """Converts strain to stress, assuming plane stress conditions"""
    Srr = 1./(1.-nu**2) * (err + nu*ett)
    Stt = 1./(1.-nu**2) * (ett + nu*err)
    Srt = 1./(1.-nu**2) * ert
    return Srr, Stt, Srt

def stress2displacement(sxx,sxy,syy,x,y,E,nu):
    """should convert stress to displacement, assuming plane stress conditions, up to overall constant... not done"""

    exx = (sxx- nu*syy)/E
    eyy = (syy- nu*sxx)/E 

    'FINISH THIS!'    
    return ux, uy


def bond_length_list(xy,BL):
    """Convert bond list (#bonds x 2) to bond length list (#bonds x 1) for lattice of bonded points. 
    
    Parameters
    ----------
    xy : array of dimension nx2
        2D lattice of points (positions x,y)
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points.
        
    Returns
    ----------
    bL : array of dimension #bonds x 1
        Bond lengths, in order of BL (lowercase denotes 1D array)   
    """  
    bL = np.array([np.sqrt(np.dot(xy[BL[i,1],:]-xy[BL[i,0],:],xy[BL[i,1],:]-xy[BL[i,0],:])) for i in range(len(BL))])
    return bL


##########################################
## 4. Lattice Generation
##########################################

def generate_diskmesh(R,n,steps_azim):
    """Create an array of evenly spaced points in 2D on a disc using Vogel's method.
    
    Parameters
    ----------
    R : float
        radius of the disc
    n : int
        number of points within the disc, distributed by Vogel method
    steps_azim : int
        number of points on the boundary of the disc
        
    Returns
    ---------
    xypts : (n+steps_azim) x 2 array
        the positions of vertices on evenly distributed points on disc
    """
    #steps_azim is the azimuthal NUMBER of steps of the mesh
    # ----> NOT the step size/length
    #Note! The R value is INCLUSIVE!!
    
    #spiral pattern of points using Vogel's method--> Golden Triangles
    #The radius of the ith point is rho_i=R*sqrt(i/n)
    #n = 256
    radius = R*sqrt(arange(n) / float(n))
    golden_angle = pi * (3 - sqrt(5))
    theta = golden_angle * arange(n)
 
    points = zeros((n, 2))
    points[:,0] = cos(theta)
    points[:,1] = sin(theta)
    points *= radius.reshape((n, 1))
    #plt.plot(points[:,0],points[:,1],'b.')

    vals = array([[R*cos(ii*2*pi/steps_azim), R*sin(ii*2*pi/steps_azim) ] for ii in arange(steps_azim)]) #circle points
    vals = reshape(vals, [-1,2]) #[steps**2,2])    
    
    xypts=vstack((points,vals))
    return xypts

def generate_diskmesh_vogelgap(R,n,steps_azim,fraction_edge_gap):
    """Create an array of evenly spaced points in 2D on a disc using Vogel's method, but only up to a smaller radius than the radius of the circle of points with angular density 2pi/steps_azim
    
    Parameters
    ----------
    R : float
        radius of the disc
    n : int
        number of points within the disc, distributed by Vogel method
    steps_azim : int
        number of points on the boundary of the disc
    fraction_edge_gap : float
        difference between Vogel radius and circle radius, as a fraction of radius R.
        
    Returns
    ---------
    xypts : (n+steps_azim) x 2 array
        the positions of vertices on evenly distributed points on disc
    """
    #This includes the Vogel method but only up to a smaller radius than the  
    # radius of the circle of points with angular density 2pi/steps_azim. The 
    # difference between Vogel radius and circle radius is given by 
    # fraction_edge_gap, as a fraction of radius R.
    #steps_azim is the azimuthal NUMBER of steps of the mesh
    # ----> NOT the step size/length
    #Note! The R value is INCLUSIVE!!
    
    #spiral pattern of points using Vogel's method--> Golden Triangles
    #The radius of the ith point is rho_i=R*sqrt(i/n)
    #n = 256
    radius = R*(1-fraction_edge_gap)*np.sqrt(np.arange(n) / float(n))
    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(n)
 
    points = np.zeros((n, 2))
    points[:,0] = np.cos(theta)
    points[:,1] = np.sin(theta)
    points *= radius.reshape((n, 1))
    #plt.plot(points[:,0],points[:,1],'b.')

    vals = np.array([[R*np.cos(ii*2*np.pi/steps_azim), R*np.sin(ii*2*np.pi/steps_azim) ] for ii in np.arange(steps_azim)]) #circle points
    vals = np.reshape(vals, [-1,2]) #[steps**2,2])    
    
    xypts= np.vstack((points,vals))
    return xypts


def generate_lattice(image_shape, lattice_vectors):
    """Creates lattice of positions from arbitrary lattice vectors.
    
    Parameters
    ----------
    image_shape : 2 x 1 list (eg image_shape=[L,L])
        Width and height of the lattice (square)
    lattice_vectors : 2 x 1 list of 2 x 1 lists (eg [[1 ,0 ],[0.5,sqrt(3)/2 ]])
        The two lattice vectors defining the unit cell.
        
    Returns
    ----------
    xy : array of dimension nx2
        2D lattice of points (positions x,y)
    """
    #Generate lattice that lives in 
    #center_pix = np.array(image_shape) // 2
    # Get the lower limit on the cell size.
    dx_cell = max(abs(lattice_vectors[0][0]), abs(lattice_vectors[1][0]))
    dy_cell = max(abs(lattice_vectors[0][1]), abs(lattice_vectors[1][1]))
    # Get an over estimate of how many cells across and up.
    nx = 2*image_shape[0]//dx_cell
    ny = 2*image_shape[1]//dy_cell
    # Generate a square lattice, with too many points.
    # Here I generate a factor of 8 more points than I need, which ensures 
    # coverage for highly sheared lattices.  If your lattice is not highly
    # sheared, than you can generate fewer points.
    x_sq = np.arange(-nx, nx, dtype=float)
    y_sq = np.arange(-ny, nx, dtype=float)
    x_sq.shape = x_sq.shape + (1,)
    y_sq.shape = (1,) + y_sq.shape
    # Now shear the whole thing using the lattice vectors
    #transpose so that row is along x axis
    x_lattice = lattice_vectors[0][1]*x_sq + lattice_vectors[1][1]*y_sq
    y_lattice = lattice_vectors[0][0]*x_sq + lattice_vectors[1][0]*y_sq
    # Trim to fit in box.
    mask = ((x_lattice < image_shape[0]/2.0)
             & (x_lattice > -image_shape[0]/2.0))
    mask = mask & ((y_lattice < image_shape[1]/2.0)
                    & (y_lattice > -image_shape[1]/2.0))
    x_lattice = x_lattice[mask]
    y_lattice = y_lattice[mask]
    # Make output compatible with original version.
    out = np.empty((len(x_lattice), 2), dtype=float)
    out[:, 0] = y_lattice
    out[:, 1] = x_lattice
    i = np.lexsort((out[:,1], out[:,0])) #sort primarily by x, then y
    xy=out[i]
    return xy

def arrow_mesh(x,y,z,dx,dy,dz,rotation_angle=0,tail_width=0.2,head_width=0.5, head_length=0.3, overhang=0.0 ):
    """Creates a mesh arrow (pts,tri) pointing from x,y,z to x+dx,y+dy,z+dz.

    Parameters
    ----------
    x,y,z : floats
        x,y,z position of the tail of the arrow
    dx,dy,dz : floats
        signed distances in x,y,z from the tail to the head of the arrow
    rotation_angle : float
        angle in radians by which arrow rotated about its long axis
    tail_width : float
        width of the arrow tail as fraction of arrow length (tail_width = |(1)-(7)| =|(2)-(6)| )
    head_width : float
        width of the arrow head as fraction of arrow length (head_width = |(3)-(5)|)
    head_length : float
        fraction of the arrow length that is part of the arrow head
    overhang : float
        fraction of the arrow length by which the pointy corners of the head extend behind the head
    """
    #            2|\              
    # 0  _________| \            
    #   |         1  \ 3        
    #   |_________   /          
    # 6         5 | /        
    #           4 |/        
    #
    # Begin by making arrow in the xy plane, with middle of tail at xy, pointing in x dir
    d = np.sqrt(dx**2+dy**2+dz**2)
    pts0 = np.array([[0, d*tail_width*0.5, 0], \
        [d*(1.-head_length), d*tail_width*0.5, 0], \
        [d*(1.-head_length-overhang), d*head_width*0.5, 0], \
        [d, 0, 0], \
        [d*(1.-head_length-overhang), -d*head_width*0.5, 0], \
        [d*(1.-head_length), -d*tail_width*0.5, 0], \
        [0,-d*tail_width*0.5, 0] \
        ])
    # Rotate about axis by rotation_angle
    pts = rotate_vector_xaxis3D(pts0,rotation_angle)
    # Rotate in xy plane
    theta = np.arctan2(dz,np.sqrt(dx**2+dy**2))
    phi = np.arctan2(dy,dx)
    pts = rotate_vector_yaxis3D(pts,-theta)
    pts = rotate_vector_zaxis3D(pts,phi)
    pts+= np.array([x,y,z])
    tri = np.array([ [0,6,1], [6,5,1],\
        [3,2,1], [4,3,5],[3,1,5] ])
    return pts,tri

def rotate_vector_2D(vec,phi):
    """Rotate vector by angle phi in xy plane"""
    if vec.ndim>1:
        """rot is a list of multiple vectors or an array of length >1"""
        rot = np.array([ [x*np.cos(phi)-y*np.sin(phi), \
                          y*np.sin(phi)+y*np.cos(phi) ] for x,y in vec ])
    else:
        rot = np.array([ vec[0]*np.cos(phi)-vec[1]*np.sin(phi), \
                          vec[0]*np.sin(phi)+vec[1]*np.cos(phi) ])
    return rot


def rotate_vector_xaxis3D(vec,phi):
    """Rotate 3D vector(s) by angle phi about x axis --> rotates away from the y axis"""
    if vec.ndim>1:
        rot = np.array([ [x, \
                          y*np.cos(phi)-z*np.sin(phi), \
                          y*np.sin(phi)+z*np.cos(phi) ] for x,y,z in vec ])
    else:
        rot = np.array([ vec[0], \
                         vec[1]*np.cos(phi)-vec[2]*np.sin(phi), \
                         vec[1]*np.sin(phi)+vec[2]*np.cos(phi) ])
    return rot

def rotate_vector_yaxis3D(vec,phi):
    """Rotate 3D vector(s) by angle phi about y axis (in xz plane) --> rotates away from the z axis"""
    if vec.ndim>1:
        rot = np.array([ [x*np.cos(phi)+z*np.sin(phi), \
                          y, \
                          -x*np.sin(phi)+z*np.cos(phi) ] for x,y,z in vec ])
    else:
        rot = np.array([ vec[0]*np.cos(phi)+vec[2]*np.sin(phi), \
                         vec[1],
                         -vec[0]*np.sin(phi)+vec[2]*np.cos(phi) ])
    return rot

def rotate_vector_zaxis3D(vec,phi):
    """Rotate vector by angle phi in xy plane, keeping z value fixed"""
    if vec.ndim>1:
        rot = np.array([ [x*np.cos(phi)-y*np.sin(phi), \
                          x*np.sin(phi)+y*np.cos(phi), z ] for x,y,z in vec ])
    else:
        rot = np.array([ vec[0]*np.cos(phi)-vec[1]*np.sin(phi), \
                        vec[0]*np.sin(phi)+vec[1]*np.cos(phi), vec[2] ])
    return rot



##########################################
## 5. Data Handling
##########################################

def tensor_polar2cartesian2D(Mrr,Mrt,Mtr,Mtt,x,y):
    """converts a Polar tensor into a Cartesian one
    
    Parameters
    ----------
    Mrr, Mtt, Mrt, Mtr : N x 1 arrays
        radial, azimuthal, and shear components of the tensor M
    x : N x 1 array
        the x positions of the points on which M is defined
    y : N x 1 array
        the y positions of the points on which M is defined
        
    Returns
    ----------
    Mxx,Mxy,Myx,Myy : N x 1 arrays
        the cartesian components
    """
    A = Mrr; B = Mrt; C = Mtr; D = Mtt; 
    theta= np.arctan2(y,x);
    ct = np.cos(theta) ;
    st = np.sin(theta) ;
    
    Mxx = ct*(A*ct - B*st) - st*(C*ct - D*st) ;
    Mxy = ct*(B*ct + A*st) - st*(D*ct + C*st) ;
    Myx = st*(A*ct - B*st) + ct*(C*ct - D*st) ;
    Myy = st*(B*ct + A*st) + ct*(D*ct + C*st) ;
    return Mxx, Mxy, Myx, Myy

def tensor_cartesian2polar2D(Mxx,Mxy,Myx,Myy,x,y):
    """converts a Cartesian tensor into a Polar one
    
    Parameters
    ----------
    Mxx,Mxy,Myx,Myy : N x 1 arrays
        cartesian components of the tensor M
    x : N x 1 array
        the x positions of the points on which M is defined
    y : N x 1 array
        the y positions of the points on which M is defined
        
    Returns
    ----------
    Mrr, Mrt, Mtr, Mtt : N x 1 arrays
        radial, shear, and azimuthal components of the tensor M
    """    
    A = Mxx; B = Mxy; C = Myx; D = Myy; 
    theta= np.arctan2(y,x);
    ct = np.cos(theta) ;
    st = np.sin(theta) ;
    
    Mrr = A*ct**2 + (B+C)*ct*st + D*st**2 ;
    Mrt = B*ct**2 + (-A+D)*ct*st - C*st**2 ;
    Mtr = C*ct**2 + (-A+D)*ct*st - B*st**2 ;
    Mtt = D*ct**2 - (B+C)*ct*st + A*st**2 ;
    return Mrr, Mrt, Mtr, Mtt
    
def vectorfield_cartesian2polar(ux,uy,x,y):
    """converts a Cartesian vector field into a Polar one
    
    Parameters
    ----------
    ux,uy : N x 1 arrays
        vector field values along x and y (cartesian)
    x : N x 1 array
        the x positions of the points on which u is defined
    y : N x 1 array
        the y positions of the points on which u is defined
        
    Returns
    ----------
    ur, ut : N x 1 arrays
        radial and azimuthal values of the vector field 
    """
    theta = np.arctan2(y,x)
    ur = ux*np.cos(theta) + uy*np.sin(theta)
    ut = -ux*np.sin(theta) + uy*np.cos(theta)
    return ur, ut
    
def vectorfield_polar2cartesian(ur,ut,x,y):
    """converts a Polar vector field into a Cartesian one
    
    Parameters
    ----------
    ur,ut : N x 1 arrays
        vector field values along r and theta (polar)
    x : N x 1 array
        the x positions of the points on which u is defined
    y : N x 1 array
        the y positions of the points on which u is defined
        
    Returns
    ----------
    ux, uy : N x 1 arrays
        cartesian values of the vector field 
    """
    theta = np.arctan2(y,x)
    beta = theta + np.arctan2(ut,ur)
    umag = np.sqrt(ur**2+ut**2)
    ux = umag*np.cos(beta)
    uy = umag*np.sin(beta)
    return ux, uy
    
  
def flip_orientations_tris(TRI,xyz):
    """Flip triangulations such that their normals are facing upward"""
    for ii in range(len(TRI)):
        V = xyz[TRI[ii,1],:] - xyz[TRI[ii,0],:]
        W = xyz[TRI[ii,2],:] - xyz[TRI[ii,0],:]
        Nz = V[0]*W[1] - V[1]*W[0] 
        if Nz<0:
            temp = TRI[ii,2]
            TRI[ii,2] = TRI[ii,0]
            TRI[ii,0] = temp
    return TRI
    
    
def flip_all_orientations_tris(TRI):
    """Flip triangulations such that their normals are inverted"""
    temp = copy.deepcopy(TRI[:,2])
    TRI[:,2] = TRI[:,0]
    TRI[:,0] = temp
    return TRI
    

def Tri2BL(TRI):
    """Convert triangulation array (#tris x 3) to bond list (#bonds x 2) for 2D lattice of triangulated points. 
    
    Parameters
    ----------
    TRI : array of dimension #tris x 3
        Each row contains indices of the 3 points lying at the vertices of the tri.
        
    Returns
    ----------
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points"""
    BL1 = TRI[:,[0,1]]
    BL2 = np.vstack((BL1,TRI[:,[0,2]]))
    BL3 = np.vstack((BL2,TRI[:,[1,2]]))
    BLt = np.sort(BL3, axis=1);
    #select unique rows of BL
    #BL = np.unique(BLt.view(np.dtype((np.void, BLt.dtype.itemsize*BLt.shape[1])))).view(BLt.dtype).reshape(-1, BLt.shape[1])
    BL = unique_rows(BLt)
    return BL
   
def BL2TRI(BL):
    """Convert bond list (#bonds x 2) to Triangulation array (#tris x 3) (using dictionaries for speedup and scaling)
    
    Parameters
    ----------
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points
        
    Returns
    ----------
    TRI : array of dimension #tris x 3
        Each row contains indices of the 3 points lying at the vertices of the tri.
    """
    d = {}
    tri = np.zeros((len(BL),3), dtype=np.int)
    c = 0
    for i in BL:
        if(i[0] > i[1]):
            t = i[0]
            i[0] = i[1]
            i[1] = t
        if(i[0] in d):
            d[i[0]].append(i[1])
        else:
            d[i[0]] = [i[1]]
    for key in d:
        for n in d[key]:
            for n2 in d[key]:
                if (n>n2) or n not in d:
                    continue
                if (n2 in d[n]):
                    tri[c,:] = [key,n,n2]
                    c += 1
    return tri[0:c]   
   
def BL2TRI_slow(BL,nn):
    """Convert bond list (#bonds x 2) to Triangulation array (#tris x 3)
    
    Parameters
    ----------
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points
    nn : int
        max number of nearest neighbors expected in NL and KL
        
    Returns
    ----------
    TRI : array of dimension #tris x 3
        Each row contains indices of the 3 points lying at the vertices of the tri.
    """
    print('BL2TRI: computing NL...')
    NL, KL = BL2NLandKL(BL,nn)
    #print('BL2TRI: --> max(NL) = '+str(max(NL.ravel()))+'...')
    print('BL2TRI: assembling TRI...')
    ind = 0
    #firstrow = 1
    # for each row in NL, check if both elements of BL (checked over all rows) are neighbors:
    # then the 2 n. neighbors are themselves neighbors, and so a triangle is formed.
    # If so, add lines to TRI with [ BL col 1, BL col 2, row #]
    
    TRItmp = np.zeros((10*len(NL),3),dtype='int')
    # add 1 to all BL values and to all NL values for which KL!=0
    BLp = BL + np.ones(np.shape(BL))
    NLp = copy.deepcopy(NL)
    NLp[KL!=0] +=1
    
    for kk in range(len(NLp)):
        if np.mod(kk,400)==200:
            print('BL2TRI: assembling row '+str(kk)+'/'+str(len(NLp)))
        
        idx = np.logical_and( ismember(BLp[:,0], NLp[kk,:]), ismember(BLp[:,1], NLp[kk,:]) )
        TRIS = BL[idx,:]
        TRItmp[ind:ind+len(TRIS),:]  = np.hstack(( TRIS, kk*np.ones((len(TRIS),1)) ))
        ind = ind+len(TRIS)
        
    TRItmp2 = TRItmp[0:ind,:]
    print(TRItmp2)
    print('BL2TRI: sorting TRI...')
    TRIt = np.sort(TRItmp2, axis=1)
    print('BL2TRI: ensuring unique rows...')
    TRI = unique_rows(TRIt)
    return TRI

def BL2NLandKL(BL,nn=6):
    """Convert bond list (#bonds x 2) to neighbor list (#pts x max# neighbors) for lattice of bonded points. Also returns KL: ones where there is a bond and zero where there is not.
    (Even if you just want NL from BL, you have to compute KL anyway.)
    
    Parameters
    ----------
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points
    nn : int
        maximum number of neighbors
        
    Returns
    ----------
    NL : array of dimension #pts x max(#neighbors)
        The ith row contains indices for the neighbors for the ith point.
    KL :  array of dimension #pts x (max number of neighbors)
        Spring constant list, where 1 corresponds to a true connection while 0 signifies that there is not a connection.
    """
    NL = np.zeros((max(BL.ravel())+1,nn))
    KL = np.zeros((max(BL.ravel())+1,nn))
    for row in BL:
        col = np.where(KL[row[0],:]==0)[0][0]
        NL[row[0],col] = row[1]
        KL[row[0],col] = 1
        col = np.where(KL[row[1],:]==0)[0][0]
        NL[row[1],col] = row[0]
        KL[row[1],col] = 1        
    return NL, KL


def bond_length_list(xy,BL):
    """Convert neighbor list to bond list (#bonds x 2) for lattice of bonded points. 
    
    Parameters
    ----------
    xy : array of dimension nx2
        2D lattice of points (positions x,y)
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points.
        
    Returns
    ----------
    bL : array of dimension #bonds x 1
        Bond lengths, in order of BL (lowercase denotes 1D array)   
    """  
    bL = np.array([np.sqrt(np.dot(xy[BL[i,1],:]-xy[BL[i,0],:],xy[BL[i,1],:]-xy[BL[i,0],:])) for i in range(len(BL))])
    return bL


def cut_bonds(BL,xy,thres):
    """Cuts bonds with lengths greater than threshold value.
    
    Parameters
    ----------
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points
    xy : array of dimension nx2
        2D lattice of points (positions x,y)
    thres : float
        cutoff length between points
                
    Returns
    ----------
    BLtrim : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points, contains no bonds longer than thres"""
    i2cut = (xy[BL[:,0],0]-xy[BL[:,1],0])**2+(xy[BL[:,0],1]-xy[BL[:,1],1])**2< thres**2
    BLtrim = BL[i2cut]
    return BLtrim


def memberIDs(a, b):
    """Return array (c) of indices where elements of a are members of b.
    If ith a elem is member of b, ith elem of c is index of b where a[i] = b[index].
    If ith a elem is not a member of b, ith element of c is 'None'.
    The speed is O(len(a)+len(b)), so it's fast.
    """
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return [bind.get(itm, None) for itm in a]  # None can be replaced by any other "not in b" value

def ismember(a, b):
    """Return logical array (c) testing where elements of a are members of b. 
    The speed is O(len(a)+len(b)), so it's fast.
    """
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = True
    return np.array([bind.get(itm, False) for itm in a])  # None can be replaced by any other "not in b" value


def unique_rows(a):
    """Clean up an array such that all its rows are unique.
    """
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1) 
    return a[ui]


def point_is_on_linesegment_2D_singlept(p, a, b, thres=1e-5):
    """Check if point is on line segment (or vertical line is on plane in 3D, since 3rd dim ignored).
    
    Parameters
    ----------
    p : array or list of length >=2
        The position of the point  
    a : array or list of length >=2
        One end of the line segment
    b : array or list of length >=2
        The other end of the line segment
    thres : float
        How close must the point be to the line segment to be considered to be on it
                
    Returns
    ----------
    Boolean : whether the pt is on the line segment
    """
    # cross product == 0 means the points are aligned (on the line defined by line seg)
    crossproduct = (p[1] - a[1]) * (b[0] - a[0]) - (p[0] - a[0]) * (b[1] - a[1])
    if abs(crossproduct) > thres : return False   # (or != 0 if using integers)

    # dot product must be positive and less than |b-a|^2
    dotproduct = (p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1])*(b[1] - a[1])
    if dotproduct < 0 : return False

    squaredlengthba = (b[0] - a[0])*(b[0] - a[0]) + (b[1] - a[1])*(b[1] - a[1])
    if dotproduct > squaredlengthba: return False

    return True
    
def point_is_on_linesegment_2D(p, a, b,thres=1e-5):
    """Check if point is on line segment (or vertical line is on plane in 3D, since 3rd dim ignored).
    
    Parameters
    ----------
    p : array of dimension #points x >=2
        The points in 2D (or 3D with 3rd dim ignored)
    a : array or list of dimension 1 x >=2
        One end of the line segment
    b : array or list of dimension 1 x >=2
        The other end of the line segment
    thres : float
        How close must the point be to the line segment to be considered to be on it
                
    Returns
    ----------
    Boolean array : whether the pts are on the line segment
    """
    crossproduct = (p[:,1] - a[1]) * (b[0] - a[0]) - (p[:,0] - a[0]) * (b[1] - a[1])
    dotproduct = (p[:,0] - a[0]) * (b[0] - a[0]) + (p[:,1] - a[1])*(b[1] - a[1])
    squaredlengthba = (b[0] - a[0])*(b[0] - a[0]) + (b[1] - a[1])*(b[1] - a[1])
    on_seg = np.logical_and(np.logical_and(abs(crossproduct) < thres, dotproduct > 0 ), dotproduct < squaredlengthba)
    return on_seg


def closest_pts_along_line(pts,endpt1,endpt2):
    """Get point along a line defined by two points (endpts), closest to a point not on the line. Returns p as numpy array.
    
    Parameters
    ----------
    pt : array of length 2
        point near which to find nearest point
    endpt1, endpt2 : arrays of length 2
        x,y positions of points on line as array([[x0,y0],[x1,y1]])
                
    Returns
    ----------
    proj : array of length 2
        the point nearest to pt along line
    """
    # v is vec along line seg
    a = endpt2[0] - endpt1[0]
    b = endpt2[1] - endpt1[1]
    x = pts[:,0] - endpt1[0]
    y = pts[:,1] - endpt1[1]
    
    # the projection of the vector to pt along v
    projv = np.dstack(( a*(a*x+b*y)/(a**2+b**2) , b*(a*x+b*y)/(a**2+b**2) ))[0]
    # add the endpt whose position was subtracted
    p = projv + endpt1*np.ones(projv.shape)  
    return p

def line_pts_are_on_lineseg(p,a,b):
    """Check if an array of points (p) which lie along a line is between two other points (a,b) on that line (ie, is on a line segment)
    
    Parameters
    ----------
    p : array of dim N x 2
        points for which to evaluate if they are on segment
    a,b : arrays or lists of length 2
        x,y positions of line segment endpts 
                
    Returns
    ----------
    True or False: whether pt is between endpts
    """
    # dot product must be positive and less than |b-a|^2
    dotproduct = (p[:,0] - a[0]) * (b[0] - a[0]) + (p[:,1] - a[1])*(b[1] - a[1])
    squaredlengthba = (b[0] - a[0])*(b[0] - a[0]) + (b[1] - a[1])*(b[1] - a[1])
    return np.logical_and(dotproduct > 0, dotproduct < squaredlengthba)
    

def closest_pts_on_lineseg(pts,endpt1,endpt2):
    """Get points on line segment closest to an array of points not on the line;
    the closest point can be an endpt, for example if the linseg is distant from pt.
    
    Parameters
    ----------
    pts : array N x 2
        points near which to find near point
    endpts : array of dimension 2x2
        x,y positions of endpts of line segment as array([[x0,y0],[x1,y1]])
                
    Returns
    ----------
    p : array of length 2
        the point nearest to pt on lineseg
    d : float
        distance from pt to p
    """
    # create output vectors
    pout = np.zeros_like(pts)
    dout = np.zeros_like(pts[:,0])
    
    # Find nearest p along line formed by endpts
    p = closest_pts_along_line(pts,endpt1, endpt2)
    d0 = np.sqrt((p[:,1]-pts[:,1])**2 + (p[:,0]-pts[:,0])**2)
    
    # is p ok?-- are they the line segment? or is out of bounds?
    pok = line_pts_are_on_lineseg(p, endpt1, endpt2)
    
    # Assign those pts and distances for pok indices    
    pout[pok,:] = p[pok,:]
    dout[pok]   = d0[pok]
        
    # For p not on the segment, pick closer endpt
    d1 = (endpt1[1]-pts[:,1])**2 + (endpt1[0]-pts[:,0])**2 
    d2 = (endpt2[1]-pts[:,1])**2 + (endpt2[0]-pts[:,0])**2
    
    nd1 = d1<d2 #nearer to d1
    ntd1 = np.logical_and(~pok, nd1) #nearest to d1
    ntd2 = np.logical_and(~pok,~nd1) #nearest to d2
    
    pout[ ntd1 ,: ] = endpt1
    dout[ ntd1  ]   = np.sqrt( d1[ntd1] )
    pout[ ntd1 ,: ] = endpt2
    dout[ ntd2  ]   = np.sqrt( d2[ntd2] )
    
    return pout, dout
        

def closest_pt_along_line(pt,endpt1,endpt2):
    """Get point along a line defined by two points (endpts), closest to a point not on the line
    
    Parameters
    ----------
    pt : array of length 2
        point near which to find nearest point
    endpt1, endpt2 : arrays of length 2
        x,y positions of points on line as array([[x0,y0],[x1,y1]])
                
    Returns
    ----------
    proj : array of length 2
        the point nearest to pt along line
    """
    #     .pt   /endpt2  
    #          /  
    #        7/proj
    #       //
    #      //endpt1
    #
    # v is vec along line seg
    a = endpt2[0] - endpt1[0]
    b = endpt2[1] - endpt1[1]
    x = pt[0] - endpt1[0]
    y = pt[1] - endpt1[1]

    # the projection of the vector to pt along v (no numpy)
    p = [a*(a*x+b*y)/(a**2+b**2) +endpt1[0] ,  b*(a*x+b*y)/(a**2+b**2)  +endpt1[1]]
    #print 'p (in closest_pt_along...) =', p
    return p

def line_pt_is_on_lineseg(p,a,b):
    """Check if a point (p) which lies along a line is between two other points (a,b) on that line (ie, is on a line segment)
    
    Parameters
    ----------
    p : array or list of length 2
        point near which to find near point
    a,b : arrays or lists of length 2
        x,y positions of endpts of line segment 
                
    Returns
    ----------
    True or False: whether pt is between endpts
    """
    # dot product must be positive and less than |b-a|^2
    dotproduct = (p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1])*(b[1] - a[1])
    if dotproduct < 0 : return False

    squaredlengthba = (b[0] - a[0])*(b[0] - a[0]) + (b[1] - a[1])*(b[1] - a[1])
    if dotproduct > squaredlengthba: return False
    
    return True
    
    
def closest_pt_on_lineseg_dolf(pt,endpt1,endpt2):
    """Get point on line segment closest to a point not on the line; could be an endpt if linseg is distant from pt.
    
    Parameters
    ----------
    pt : array of length 2
        point near which to find near point
    endpts : array of dimension 2x2
        x,y positions of endpts of line segment as array([[x0,y0],[x1,y1]])
                
    Returns
    ----------
    p : array of length 2
        the point nearest to pt on lineseg
    d : float
        distance from pt to p
    """
    p = closest_pt_along_line(pt,endpt1, endpt2)
    # is p ok?-- is it on the line segment? or is out of bounds?
    pok = line_pt_is_on_lineseg(p, endpt1, endpt2)
    if pok:
        # closest point is p, with distance d0
        d0 = dolf.sqrt((p[1]-pt[1])**2 + (p[0]-pt[0])**2)
        return p, d0
    
    else:
        d1 = (endpt1[1]-pt[1])**2 + (endpt1[0]-pt[0])**2 
        d2 = (endpt2[1]-pt[1])**2 + (endpt2[0]-pt[0])**2 
        # p is not on the segment, so pick closer endpt
        if d1 < d2:
            return endpt1, dolf.sqrt(d1)
        else:
            return endpt2, dolf.sqrt(d2)


def closest_pt_on_lineseg(pt,endpt1,endpt2):
    """Get point on line segment closest to a point not on the line; could be an endpt if linseg is distant from pt.
    
    Parameters
    ----------
    pt : array of length 2
        point near which to find near point
    endpts : array of dimension 2x2
        x,y positions of endpts of line segment as array([[x0,y0],[x1,y1]])
                
    Returns
    ----------
    p : array of length 2
        the point nearest to pt on lineseg
    d : float
        distance from pt to p
    """
    p = closest_pt_along_line(pt,endpt1, endpt2)
    # is p ok?-- is it on the line segment? or is out of bounds?
    pok = line_pt_is_on_lineseg(p, endpt1, endpt2)
    if pok:
        # closest point is p, with distance d0
        d0 = np.sqrt((p[1]-pt[1])**2 + (p[0]-pt[0])**2)
        return p, d0
    
    else:
        # p is not on the segment, so pick closer endpt
        d1 = (endpt1[1]-pt[1])**2 + (endpt1[0]-pt[0])**2 
        d2 = (endpt2[1]-pt[1])**2 + (endpt2[0]-pt[0])**2 
        if d1 < d2:
            return endpt1, np.sqrt(d1)
        else:
            return endpt2, np.sqrt(d2)
    
def pt_near_lineseg(x, endpt1, endpt2, W):
    """Determine if pt is within W of line segment"""
    # check if point is anywhere near line before doing calcs
    minx = min(endpt1[0],endpt2[0]) - W
    maxx = max(endpt1[0],endpt2[0]) + W
    if (x[0] < minx) or (x[0] > maxx):
        return False
    else:
        # check y value
        miny = min(endpt1[1],endpt2[1]) - W
        maxy = max(endpt1[1],endpt2[1]) + W
        if (x[1] < miny) or (x[1] > maxy):
            return False
        else:
            # check if point is anywhere near line before doing calcs
            p , dist = closest_pt_on_lineseg( [ x[0],x[1] ],endpt1, endpt2)
            if dist <= W :
                return True
            else:
                return False
            
def pts_are_near_lineseg(x, endpt1, endpt2, W):
    """Determine if pts in array x are within W of line segment"""
    # check if point is anywhere near line before doing calcs
    p , dist = closest_pts_on_lineseg( x ,endpt1, endpt2)
    return dist <= W 


def mindist_from_multiple_linesegs(pts,linesegs):
    """Return the minimum distance between an array of points and any point lying on any of the linesegments
    in the given list of linesegments 'linesegs'.
    
    Parameters
    ----------
    pts : Nx2 array (or list?)
        x,y positions of points
    linsegs : Nx4 array or list
        each row contains x,y of start point, x,y of end point
        
    Returns
    ---------
    dist : float
        minimum distance to any point lying on any of the linesegs
    """
    first = 1
    #print 'pts  = ', pts
    print 'linesegs = ', linesegs
    for row in linesegs:
        endpt1 = [row[0],row[1]]
        endpt2 = [row[2],row[3]]
        p , dist0 = closest_pts_on_lineseg( pts ,endpt1, endpt2)
        if first ==1:
            dist = dist0
            first =0
        else:
            #print 'shape(dist) = ', np.shape(dist)
            #print 'shape(dist0) = ', np.shape(dist0)
            dist = np.min(np.vstack((dist,dist0)), axis=0)
            #print 'dist = ', dist
            #print 'shape(dist) = ', np.shape(dist)
    return dist


def do_kdtree(combined_x_y_arrays,points,k=1):
    """Using kd tree, return indices of nearest points and their distances
    
    Parameters
    ----------
    combined_x_y_arrays : NxD array
        the reference points of which to find nearest ones to 'points' data
    points : MxD array
        data points, finds nearest elements in combined_x_y_arrays to these points.
    
    Returns
    ----------
    indices : Mx1 array
        indices of xyref that are nearest to points
    dist : Mx1 array
        the distances of xyref[indices] from points 
    """
    #usage--> find nearest neighboring point in combined_x_y_arrays for 
    #each point in points.
    # Note: usage for KDTree.query(x, k=1, eps=0, p=2, distance_upper_bound=inf)[source]
    mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
    dist, indexes = mytree.query(points,k=k)
    return indexes, dist

def lookupZ(lookupXYZ,xy_pts):
    """Using kd tree, convert array of xy points to xyz points by lookup up Z values, (based on proximity of xy pts).
    See also lookupZ_avgN().
    
    Parameters
    ----------
    lookupXYZ : Nx3 array 
        the reference points of which to find nearest ones to 'points' data
    xy_pts : MxD array with D>=2
        data points, finds nearest elements in combined_x_y_arrays to these points.
        
    Returns
    ----------
    outXYZpts : Nx3 array
        x,y are from xy_pts, but z values from lookupXYZ
    """
    # print 'xy_pts = ', xy_pts
    # print 'with shape ', np.shape(xy_pts)
    Xtemp=lookupXYZ[:,0]
    Ytemp=lookupXYZ[:,1]
    lookupXY= np.dstack([Xtemp.ravel(), Ytemp.ravel()])[0]
    # Find addZ, the amount to raise the xy_pts in z.
    addZind, distance = do_kdtree(lookupXY,xy_pts)
    addZ = lookupXYZ[addZind,2]
    # print 'addZ = ', addZ
    # print 'with shape ', np.shape(addZ.ravel())
    x= np.ravel(xy_pts[:,0])
    y= np.ravel(xy_pts[:,1])
    # print 'shape of x = ', np.shape(x.ravel())
    outXYZpts= np.dstack([x.ravel(),y.ravel(),addZ.ravel()])[0]
    # View output
    #fig = plt.figure(figsize=(14,6))
    #ax = fig.add_subplot(1, 2, 1, projection='3d')
    #scatter(outXYZpts[:,0],outXYZpts[:,1],outXYZpts[:,2],c='b')
    
    return outXYZpts


def lookupZ_avgN(lookupXYZ,xy_pts,N=5, method='median'):
    """Using kd tree, return array of values for xy_pts given lookupXYZ based on near neighbors.
    Average over N neighbors for the returned value.
    
    Parameters
    ----------
    lookupXYZ : Nx3 array 
        the reference points of which to find nearest ones to 'points' data
    xy_pts : MxD array with D>=2
        data points, finds nearest elements in combined_x_y_arrays to these points.
    N : int
        number of nearby particles over which to average in the lookup evaluation        
        
    Returns
    ----------
    outXYZpts : Nx3 array
        x,y are from xy_pts, but z values from lookupXYZ
    """
    # print 'xy_pts = ', xy_pts
    # print 'with shape ', np.shape(xy_pts)
    Xtemp=lookupXYZ[:,0]
    Ytemp=lookupXYZ[:,1]
    lookupXY= np.dstack([Xtemp.ravel(), Ytemp.ravel()])[0]
    # Find addZ, the amount to raise the xy_pts in z.
    addZind, distance = do_kdtree(lookupXY,xy_pts,k=N)
    #print 'addZind =', addZind
    if isinstance(lookupXYZ,np.ma.core.MaskedArray):
        lookupXYZ = lookupXYZ.data
    if method=='median':
        addZ = np.array([[np.median(lookupXYZ[addZind[ii,:],2])] for ii in range(len(addZind))])
    elif method == 'mean':
        addZ = np.array([[np.nanmean(lookupXYZ[addZind[ii,:],2])] for ii in range(len(addZind))])
    
    # print 'addZ = ', addZ
    # print 'with shape ', np.shape(addZ.ravel())
    x= np.ravel(xy_pts[:,0])
    y= np.ravel(xy_pts[:,1])
    # print 'shape of x = ', np.shape(x.ravel())
    outXYZpts= np.dstack([x.ravel(),y.ravel(),addZ.ravel()])[0]
    # View output
    #fig = plt.figure(figsize=(14,6))
    #ax = fig.add_subplot(1, 2, 1, projection='3d')
    #scatter(outXYZpts[:,0],outXYZpts[:,1],outXYZpts[:,2],c='b')
    
    return outXYZpts

def lookupZ_singlept(lookupXYZ,xy):
    """Using kd tree, return indices of nearest points and their distances using 2D positions.
    
    Parameters
    ----------
    lookupXYZ : Nx3 array 
        the reference points of which to find nearest ones to 'points' data
    xy : list of two floats
        data point, to find nearest element in lookupXYZ
        
    Returns
    ----------
    addZ : float
        z value from lookupXYZ
    """
    Xtemp = lookupXYZ[:,0]
    Ytemp = lookupXYZ[:,1]
    lookupXY = np.dstack([Xtemp.ravel(), Ytemp.ravel()])[0]
    #Find addZ, the amount to raise the xy_pts in z.
    addZind, distance = do_kdtree(lookupXY,np.array([xy[0],xy[1]]))
    addZ = lookupXYZ[addZind,2]
    return addZ


def initphase_linear_slit(x, endpt1, endpt2, W, contour='linear'):
    """Determine initial phase at pt for slit defined by endpts, with width W and fall-off profile given by 'contour' """
    # check if point is anywhere near line before doing calcs
    minx = min(endpt1[0],endpt2[0]) - W
    maxx = max(endpt1[0],endpt2[0]) + W
    #print 'minx = ', minx
    #print 'maxx = ', maxx
    if (x[0] < minx) or (x[0] > maxx):
        val = 1.0
    else:
        # check y value
        miny = min(endpt1[1],endpt2[1]) - W
        maxy = max(endpt1[1],endpt2[1]) + W
        if (x[1] < miny) or (x[1] > maxy):
            val = 1.0
        else:
            # check if point is anywhere near line before doing calcs
            p , dist = closest_pt_on_lineseg( [ x[0],x[1] ],endpt1, endpt2)
            if dist <= W :
                val = (dist / W)*0.95 +0.05 #linear/abs
                #print 'val=', val
            else:
                val = 1.0
    return val

def is_number(s):
    """Check if a string can be represented as a number; works for floats"""
    try:
        float(s)
        return True
    except ValueError:
        return False


def round_thres(a, MinClip):
    """Round a number to the nearest multiple of MinCLip"""
    return round(float(a) / MinClip) * MinClip

def round_thres_numpy(a, MinClip):
    """Round an array of values to the nearest multiple of MinCLip"""
    return np.round(np.array(a,dtype=float) / MinClip) * MinClip

def getVarFromFile(filename):
    """Convert data in a txt file like 'x = 1.5' to a variable x defined as 1.5... this may need work
    http://stackoverflow.com/questions/924700/best-way-to-retrieve-variable-values-from-a-text-file-python-json
    """
    import imp
    f = open(filename)
    data = imp.load_source('data', '', f)
    f.close()
    return data


##########################################
## 6. Loading/Interpolating Data
##########################################
def nearest_gL_fit(lookupdir, beta, rho, fit_mean):
    """Lookup Griffith length for given rho value in table, could be table based on a quadratic fit or of the mean gLs for a given rho.
    Note that for fit_mean==fit, rho = r/R, wherease for fit_mean==mean, rho = r/x0."""
    print('looking for file:')
    print(lookupdir+fit_mean+'_rho_gLmeters_beta'+'{0:.2f}'.format(beta/np.pi).replace('.','p')+'*.txt')
    gLfile  = glob.glob(lookupdir+fit_mean+'_rho_gLmeters_beta'+'{0:.2f}'.format(beta/np.pi).replace('.','p')+'*.txt')[0]
    rhoV, gLV, trsh = np.loadtxt(gLfile, delimiter=',', skiprows=1,usecols=(0, 1,2), unpack=True)
    diff = abs(rhoV - rho)
    IND = np.where(diff== diff.min())
    return gLV[IND][0]

def constP_gL_fit(lookupdir,alph):
    """Lookup Griffith length for given aspect ratio in table of the mean gLs vs aspect ratio, returned in meters"""
    print('looking for file:')
    print(lookupdir+'constP_means_alph_gLinches.txt')
    gLfile  = glob.glob(lookupdir+'constP_means_alph_gLinches.txt')[0]
    alphV, gLV = np.loadtxt(gLfile, delimiter=',', skiprows=1,usecols=(0, 1), unpack=True)
    diff = abs(alphV - alph)
    IND = np.where(diff== diff.min())
    #return in meters
    return float(gLV[IND]/39.3700787)


def interpol_meshgrid(x,y,z,n):
    """Interpolate z on irregular or unordered grid data (x,y) by supplying # points along each dimension.
    Note that this does not guarantee a square mesh, if ranges of x and y differ.
    """
    # define regular grid spatially covering input data
    xg = np.linspace(x.min(),x.max(),n)
    yg = np.linspace(y.min(),y.max(),n)
    X,Y = np.meshgrid(xg,yg)
    
    # interpolate Z values on defined grid
    Z = griddata(np.vstack((x.flatten(),y.flatten())).T, np.vstack(z.flatten()),(X,Y),method='cubic').reshape(X.shape)
    # mask nan values, so they will not appear on plot
    Zm = np.ma.masked_where(np.isnan(Z),Z)
    return X,Y,Zm

def interpolate_onto_mesh(x,y,z,X,Y, mask=True):
    """Interpolate new data x,y,z onto grid data X,Y"""
    # interpolate Z values on defined grid
    Z = griddata(np.vstack((x.flatten(),y.flatten())).T, np.vstack(z.flatten()),(X,Y),method='cubic').reshape(X.shape)
    # mask nan values, so they will not appear on plot
    if mask: Zm = np.ma.masked_where(np.isnan(Z),Z)
    else: Zm =Z
    return Zm

# OLD VERSION 2016-03-04 (new version directly below)
# def load_params(outdir):
#     """Load params from parameters.txt file in outdir."""
#     params = {}
#     with open(outdir+'parameters.txt') as f:
#         for line in f:
#             if '# Parameters' not in line:
#                 (k, val) = line.split('=')
#                 key = k.strip()
#                 #print val
#                 if key == 'date':
#                     val = val[:-1].strip()
#                     print '\n\ndate is specially recognized: date= ', val
#                 elif is_number(val):
#                     # val is a number, so convert to a float
#                     val = float(val[:-1].strip())
#                 else:
#                     try:
#                         # val might be a list, so interpret it as such using ast
#                         #val = ast.literal_eval(val.strip())
#                         exec('val = %s' % (val.strip()) )
#                     except:
#                         # val must be a string
#                         val = val[:-1].strip()
#                         
#                 params[key] = val
#                 #print val
# 
#     return params


def load_params(outdir,paramsfn = 'parameters'):
    """Load params from parameters.txt file in outdir."""
    params = {}
    with open(outdir+paramsfn+'.txt') as f:
        for line in f:
            if '# Parameters' not in line:
                (k, val) = line.split('=')
                key = k.strip()
                #print 'key = ', key
                #print val
                if key == 'date':
                    val = val[:-1].strip()
                    print '\nloading params for: date= ', val
                elif is_number(val):
                    # val is a number, so convert to a float
                    val = float(val[:-1].strip())
                else:
                    try:
                        #print 'trying to make key =', key,' into numpy array...'
                        # If val has both [] and , --> then it is a numpy array
                        # (This is a convention choice.)
                        if '[' in val and ',' in val:
                            make_ndarray = True
                        
                        # val might be a list, so interpret it as such using ast
                        #val = ast.literal_eval(val.strip())
                        exec('val = %s' % (val.strip()) )
                        
                        # Make array if found '[' and ','
                        if make_ndarray:
                            val = np.array(val)
                        print key, ' --> is a numpy array:'
                        print 'val = ', val
                        
                    except:
                        #print 'type(val) = ', type(val)
                        # val must be a string
                        try:
                            # val might be a list of strings?
                            val = val[:-1].strip()
                        except:
                            """val is a list with a single number"""
                            val = val
                            
                        
                params[key] = val
                #print val

    return params


def interp_gL_data(gLfile, beta, rhoval):
    """Read in Griffith lengths (gL) from file, filter by a selected beta, fit to quadratic, and interpolate the result at value(s) rhoval.
    Radial values are measured in units of R."""
    rhoV, gLV, betaV = np.loadtxt(gLfile, delimiter=',', skiprows=1,usecols=(0, 1,2), unpack=True)
    IND = (betaV == beta/np.pi)
    rhodat = rhoV[IND]
    gLdat = gLV[IND]
    #rhodat is in terms of R, ie rhodat = rho/R
    def peval(x, p):
        #return p[0]+p[1]*x+p[2]*x**2
        #gives the griffith length (2a)
        return p[0]+p[1]*x**2
    def residuals(p, gLdat, rhodat):
        #err = p[0]+p[1]*rhodat+p[2]*rhodat**2 - adat
        err = p[0]+p[1]*rhodat**2 - gLdat
        return err
            
    p0 = [0.02, 0.03]
    # gL least squares
    glsq = optimize.leastsq(residuals, p0, args=(gLdat, rhodat))
    p_glsq = glsq[0] #coeffs for lsq soln fitting gL
    return peval(rhoval, p_glsq), p_glsq[0], p_glsq[1]


##########################################
## 7. Saving Data
##########################################
def savexyz(x,y,z,fname,header='x,y,z'):
    """Save 3 Nx1 arrays (for x, y, and z) of point positions as Nx3 txt file"""
    xyz = np.dstack((x,y,z))[0]
    np.savetxt(fname,xyz,delimiter=',',header =header)

    
    
def write_parameters(paramfile, params, padding_var=7):
    """Write text file with parameters (given as a dict) and their values, in a human-readable manner.
    
    Parameters
    ----------
    paramfile : string
        where to store the parameters file (path+name)
    params : dict
        the parameters, with keys as strings
    padding_var : int
        how much white space to leave between key names column and values column
        """
    with open(paramfile, 'w') as myfile:
        myfile.write('# Parameters\n')
    with open(paramfile,'a') as myfile:
        for key in params.keys():
            #print 'Writing param ', str(key)
            #print ' with value ', str(params[key])
            #print ' This param is of type ', type(params[key])
            if isinstance(params[key],str):
                myfile.write('{{0: <{}}}'.format(padding_var).format(key)+\
                             '= '+ params[key] +'\n')
            elif isinstance(params[key],np.ndarray):
                    ##########################
                    # OLD VERSION 2016-03-04
                    #
                    # myfile.write('{{0: <{}}}'.format(padding_var).format(key)+\
                    #              '= '+ str(params[key]).replace('\n','') +'\n')
                    ##########################
                    #print params[key].dtype
                    if key == 'BIND':
                        print 'BIND = ', str(params[key]).replace('\n','')
                    
                    myfile.write('{{0: <{}}}'.format(padding_var).format(key)+\
                                 '= '+ ", ".join(np.array_str(params[key]).split()).replace('[,','[') +'\n')
                    #if params[key].dtype == 'float64':
                    #    myfile.write('{{0: <{}}}'.format(padding_var).format(key)+\
                    #             '= '+ np.array_str(params[key]).replace('\n','').replace('  ',',') +'\n')
                    #elif params[key].dtype == 'int32':
                    #    myfile.write('{{0: <{}}}'.format(padding_var).format(key)+\
                    #             '= '+ str(params[key]).replace('\n','').replace(' ',',') +'\n')
                    #else:
                    #    myfile.write('{{0: <{}}}'.format(padding_var).format(key)+\
                    #             '= '+ str(params[key]).replace('\n','').replace(' ',',') +'\n')
            elif isinstance(params[key],list):
                myfile.write('{{0: <{}}}'.format(padding_var).format(key)+\
                             '= '+ str(params[key]) +'\n')
            else:
                myfile.write('{{0: <{}}}'.format(padding_var).format(key)+\
                             '= '+'{0:.12e}'.format(params[key])+'\n')


def save_images_OUT(OUT,x,y,Bxy,phi,u,params,title,title2,subtext,subsubtext,ind,ptsz=5):
    """Given a dictionary OUT, plot and save images simulation results.
    The structure of OUT must be key--> tuple, where tuple is (logical,name,field parameter for title function)
    """
    for key in OUT:
        if OUT[key][0]:
            name = OUT[key][1]
            print 'Writing ', name, '...'
            title = pe.title_scalar(OUT[key][2],params,t)
            if name == 'phase':
                pe.pf_plot_scatter_scalar(x,y,phiv,imdir+name,name,ind,title,title2,subtext,subsubtext,ptsz=ptsz, vmin=0.,vmax=1.0, shape=params['shape'])
            if name == 'contour':
                pe.pf_contour_unstructured(x,y,phiv,200,0.2,Bxy,imdir+name,name,ind,title,title2,subtext,subsubtext)
            if name == 'gstrain':
                gstrain = project((4*phi**3 - 3*phi**4) * dolf.tr(dolf.sym(nabla_grad(u))+0.5*outer(grad(h),grad(h))), Vf)
                gstrainv = gstrain.vector().array()
                pe.pf_plot_scatter_scalar(x,y,gstrainv,imdir+name,name,ind,title,title2,subtext,subsubtext,ptsz=ptsz, shape=params['shape'])
            if name == 'displacement':
                pe.pf_plot_scatter_2panel(x,y,uxv,uyv,imdir+name,name,'u',ind, title,title2,subtext,subsubtext, shape=params['shape'])
            if name == 'energy':
                En_V = project((4*phi**3 - 3*phi**4) * (E*nu/(2*(1-nu**2))*dolf.tr(dolf.sym(nabla_grad(u))+0.5*outer(grad(h),grad(h)))**2 +\
                               mu*inner(dolf.sym(nabla_grad(u)).T+0.5* outer(grad(h),grad(h)).T,dolf.sym(nabla_grad(u))+0.5* outer(grad(h),grad(h))) ), Vf)
                Env = En_V.vector().array()
                pe.pf_plot_scatter_scalar(x,y,Env,imdir+name,name,ind,title,title2,subtext,subsubtext,vmin=0,vmax=Ec, ptsz=ptsz, shape=params['shape'])
            if name == 'phidiff':
                if ii>0 and firstpass==0:
                    dphi = phiv- phiv_old
                    limC = np.nanmax(np.abs(dphi)) 
                    pe.pf_plot_scatter_scalar(x,y,dphi,imdir+name,name,ind-1,title,title2,subtext,subsubtext,ptsz=ptsz, shape=params['shape'], vmin=-limC, vmax=limC, cmap = cm.coolwarm)
                
                phiv_old = copy.deepcopy(phiv)
                firstpass = 0
            if name == 'gstress':
                epsilon_k = dolf.sym(nabla_grad(u))+0.5*outer(grad(h),grad(h))
                s_V = project((4*phi**3 - 3*phi**4)*( E/(1+nu)* (epsilon_k) + E*nu/((1-nu**2))*dolf.tr(epsilon_k)*dolf.Identity(d) ), Vt)
                sxx = project(s_V[0,0], Vf);           sxxv = sxx.vector().array()
                sxy = project(s_V[0,1], Vf);           sxyv = sxy.vector().array()
                syx = project(s_V[1,0], Vf);           syxv = syx.vector().array()
                syy = project(s_V[1,1], Vf);           syyv = syy.vector().array()
                if vminmax == 'default':
                    vmin = -params['E']*np.sqrt(2*params['Ec']/params['E'])
                    vmax = params['E']*np.sqrt(2*params['Ec']/params['E'])
                elif vminmax == 'auto':
                    vmin = 'auto'
                    vmax = 'auto'
                pe.pf_plot_scatter_4panel(x,y,sxxv,sxyv,syxv,syyv,imdir+name,name,r'\sigma',ind, title,title2,subtext,subsubtext,\
                                          vmin=vmin,vmax=vmax,cmap='seismic', shape=params['shape'], ptsz=np.max(ptsz*0.05,0.1))                     
                    


##########################################
## 8. Plotting 
##########################################

def val2color(vec,cmap=plt.cm.YlGnBu_r):
    """This is such a trivial function, it's purpose here is to really to serve as a reminder of the syntax.
    cmap_name is the object such as matplotlib.pyplot.cm.jet or
    pylab.cm.jet other cmap object.
    If you'd like to use a colormap loaded from an array in a txt file, use
    Cmp = np.loadtxt(cmppath,delimiter=','), and use this object as cmap.
    """
    color_4vector = cmap(vec)
    return color_4vector


def collect_lines(xy,BL,bs,climv):
    """Creates collection of line segments, colored according to an array of values.
    
    Parameters
    ----------
    xy : array of dimension nx2
        2D lattice of points (positions x,y)
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points
    bs : array of dimension #bonds x 1 
        Strain in each bond
    climv : float or tuple
        Color limit for coloring bonds by bs
                
    Returns
    ----------
    line_segments : matplotlib.collections.LineCollection
        Collection of line segments
    """
    lines = [zip(xy[BL[i,:],0], xy[BL[i,:],1]) for i in range(len(BL))]
    line_segments = LineCollection(lines, # Make a sequence of x,y pairs
                                linewidths    = (1.), #could iterate over list
                                linestyles = 'solid',
                                cmap='coolwarm',
                                norm=plt.Normalize(vmin=-climv,vmax=climv))
    line_segments.set_array(bs)
    print(lines)
    return line_segments

def movie_plot_2D(xy, BL, bs, fname, title,xlimv,ylimv,climv=0.1):
    """Plots and saves a 2D image of the lattice with colored bonds.
    
    Parameters
    ----------
    xy : array of dimension nx2
        2D lattice of points (positions x,y)
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points
    bs : array of dimension #bonds x 1 
        Strain in each bond
    fname : string
        Full path including name of the file (.png, etc)
    title : string
        The title of the frame
    climv : float or tuple
        Color limit for coloring bonds by bs
                
    Returns
    ----------
    prints the figure as fname
    """
    #fig = plt.figure()
    fig = plt.gcf()
    plt.clf()
    if len(xy)<10000:
        #if smallish #pts, plot them
        plt.plot(xy[:,0],xy[:,1],'k.')
        lw = (2.)
    else:
        lw=(30/np.sqrt(len(xy)))
        
    # Efficiently plot many lines in a single set of axes using LineCollection
    lines = [zip(xy[BL[i,:],0], xy[BL[i,:],1]) for i in range(len(BL))]
    line_segments = LineCollection(lines, # Make a sequence of x,y pairs
                                linewidths    = lw, #could iterate over list
                                linestyles = 'solid',
                                cmap='coolwarm',
                                norm=plt.Normalize(vmin=-climv,vmax=climv))
    line_segments.set_array(bs)
    ax = plt.axes()
    ax.add_collection(line_segments)
    #set limits
    if isinstance(xlimv, tuple):    
        ax.set_xlim(xlimv)
    else:
        ax.set_xlim(-xlimv,xlimv)
    
    if isinstance(xlimv, tuple):        
        ax.set_ylim(ylimv)
    else:
        ax.set_ylim(-ylimv, ylimv)

    axcb = fig.colorbar(line_segments)
    axcb.set_label('Strain')
    axcb.set_clim(vmin=-climv,vmax=climv)
    ax.set_title(title)        
    plt.savefig(fname)
    plt.close('all')

    
def pf_display_scalar(x,y,C,title,vmin='auto',vmax='auto',ptsz=5,cmap=cm.CMRmap,hilite ='none',axis_on=True,close=True):
    """Display then close a scatter plot of C"""
    fig, ax = plt.subplots(1, 1)
    if isinstance(vmin,str):
        vmin = np.nanmin(C)
    if isinstance(vmax,str):
        vmax = np.nanmax(C)
    #scatter scale (for color scale)
    scsc = ax.scatter(x, y, c=C, s= ptsz, cmap=cmap, edgecolor='', vmin=vmin, vmax=vmax)
    if hilite !='none':
        IND = np.abs(C-hilite) < 1e-9
        ax.scatter(x[IND],y[IND],c=C[IND],s=ptsz*3, marker='d',edgecolor='k')
    ax.set_aspect('equal')
    ax.set_title(title)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(scsc, cax=cbar_ax)
    if not axis_on:
        ax.axis('off')
    plt.show()
    if close:
        plt.close('all')
    
def pf_display_2panel(x,y,C0,C1,title0,title1='',vmin='auto',vmax='auto',ptsz=5,cmap=cm.CMRmap,axis_on=True,close=True):
    """Display then close a scatter plot of two scalar quantities C0, C1"""
    fig, ax = plt.subplots(1,2)
    if isinstance(vmin,str):
        vmin = min(np.nanmin(C0[:]),np.nanmin(C1[:]))
        print 'vmin=',vmin
    if isinstance(vmax,str):
        vmax = max(np.nanmax(C0[:]),np.nanmax(C1[:]))
        print 'vmax=',vmax
    #scatter scale (for color scale)
    scsc0 = ax[0].scatter(x, y, c=C0, s=ptsz, cmap=cmap, edgecolor='', vmin=vmin, vmax=vmax)
    scsc1 = ax[1].scatter(x, y, c=C1, s=ptsz, cmap=cmap, edgecolor='', vmin=vmin, vmax=vmax)
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    ax[0].set_title(title0)
    ax[1].set_title(title1)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    if np.nanmax(C0)>np.nanmax(C1):
        print 'maxC0>maxC1'
        fig.colorbar(scsc0, cax=cbar_ax)
    else:
        fig.colorbar(scsc1, cax=cbar_ax)
    if not axis_on:
        ax[0].axis('off')
        ax[1].axis('off')
    plt.show()
    if close:
        plt.close('all')
    
def pf_display_vector(x,y,C0,C1,varchar,title='',subscripts='cartesian',vmin='auto',vmax='auto',ptsz=5,cmap=cm.CMRmap,axis_on=True,close=True):
    """Display then close a scatter plot of components of vector quantity C
    
    Parameters
    ----------
    x,y : Nx1 arrays
        positions of evaluated points
    C0,C1 : Nx1 arrays
        components of evalauated vector
    varchar : string
        the name of the tensorial variable (raw string works for LaTeX)
    title : string
        additional title above all subplots
    subscripts : string ('cartesian','polar')
        puts subscripts on the subtitles (ie '_x', etc).
        If 'theory', then compares 
    """
    fig, ax = plt.subplots(1,2)
    if isinstance(vmin,str):
        vmin = min(np.nanmin(C0[:]),np.nanmin(C1[:]))
        print 'vmin=',vmin
    if isinstance(vmax,str):
        vmax = max(np.nanmax(C0[:]),np.nanmax(C1[:]))
        print 'vmax=',vmax
    #scatter scale (for color scale)
    scsc0 = ax[0].scatter(x, y, c=C0, s=ptsz, cmap=cmap, edgecolor='', vmin=vmin, vmax=vmax)
    scsc1 = ax[1].scatter(x, y, c=C1, s=ptsz, cmap=cmap, edgecolor='', vmin=vmin, vmax=vmax)
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    if np.nanmax(C0)>np.nanmax(C1):
        print 'maxC0>maxC1'
        fig.colorbar(scsc0, cax=cbar_ax)
    else:
        fig.colorbar(scsc1, cax=cbar_ax)
    if subscripts == 'cartesian':
        ax[0].set_title(r'${}_x$'.format(varchar) ) 
        ax[1].set_title(r'${}_y$'.format(varchar) ) 
    elif subscripts == 'polar':
        ax[0].set_title(r'${}_r$'.format(varchar) ) 
        ax[1].set_title(r'${}_\theta$'.format(varchar) )
    fig.text(0.5,0.975,title,horizontalalignment='center',verticalalignment='top')
    if not axis_on:
        ax[0].axis('off')
        ax[1].axis('off')
    
    plt.show()
    if close:
        plt.close('all')
    

def pf_display_4panel(x,y,C0,C1,C2,C3,title0,title1='',title2='',title3='',vmin='auto',vmax='auto',ptsz=5,cmap=cm.CMRmap,axis_on=True,close=True):
    """Display then close a scatter plot of four scalar quantities C0, C1, C2, C3"""
    fig, ax = plt.subplots(2,2)
    if isinstance(vmin,str):
        vmin = min(np.nanmin(C0[:]),np.nanmin(C1[:]))
        print 'vmin=',vmin
    if isinstance(vmax,str):
        vmax = max(np.nanmax(C0[:]),np.nanmax(C1[:]))
        print 'vmax=',vmax
    #scatter scale (for color scale)
    scsc0 = ax[0,0].scatter(x, y, c=C0, s=ptsz, cmap=cmap, edgecolor='', vmin=vmin, vmax=vmax)
    ax[0,1].scatter(x, y, c=C1, s=ptsz, cmap=cmap, edgecolor='', vmin=vmin, vmax=vmax)
    ax[1,0].scatter(x, y, c=C2, s=ptsz, cmap=cmap, edgecolor='', vmin=vmin, vmax=vmax)
    ax[1,1].scatter(x, y, c=C3, s=ptsz, cmap=cmap, edgecolor='', vmin=vmin, vmax=vmax)
    ax[0,0].set_aspect('equal');    ax[1,0].set_aspect('equal')
    ax[1,0].set_aspect('equal');    ax[0,1].set_aspect('equal')
    ax[0,0].set_title(title0);    ax[0,1].set_title(title1)
    ax[1,0].set_title(title2);    ax[1,1].set_title(title3)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(scsc0, cax=cbar_ax)
    if not axis_on:
        ax[0,0].axis('off')
        ax[0,1].axis('off')
        ax[1,0].axis('off')
        ax[1,1].axis('off')
    plt.show()
    if close:
        plt.close('all')
    
def pf_display_tensor(x,y,C0,C1,C2,C3,varchar,title='',subscripts='cartesian',vmin='auto',vmax='auto',ptsz=5,cmap=cm.CMRmap,axis_on=True,close=True):
    """Display then close a scatter plot of the 2x2 tensor C with components C0,C1,C2,C3.
    
    Parameters
    ----------
    x,y : Nx1 arrays
        positions of evaluated points
    C0,C1,C2,C3 : Nx1 arrays
        components of evalauated tensor
    varchar : string
        the name of the tensorial variable (raw string works for LaTeX)
    subscripts : string ('cartesian','polar','cartesiantensortheory','cartesianvectortheory','polartensortheory','polarvectortheory')
        puts subscripts on the subtitles (ie '_xx', etc)
    title : string
        additional title above all subplots
    """
    fig, ax = plt.subplots(2,2)
    if isinstance(vmin,str):
        vmin = min(np.nanmin(C0[:]),np.nanmin(C1[:]))
        print 'vmin=',vmin
    if isinstance(vmax,str):
        vmax = max(np.nanmax(C0[:]),np.nanmax(C1[:]))
        print 'vmax=',vmax
    #scatter scale (for color scale)
    scsc0 = ax[0,0].scatter(x, y, c=C0, s=ptsz, cmap=cmap, edgecolor='', vmin=vmin, vmax=vmax)
    ax[0,1].scatter(x, y, c=C1, s=ptsz, cmap=cmap, edgecolor='', vmin=vmin, vmax=vmax)
    ax[1,0].scatter(x, y, c=C2, s=ptsz, cmap=cmap, edgecolor='', vmin=vmin, vmax=vmax)
    ax[1,1].scatter(x, y, c=C3, s=ptsz, cmap=cmap, edgecolor='', vmin=vmin, vmax=vmax)
    ax[0,0].set_aspect('equal');    ax[1,0].set_aspect('equal')
    ax[1,0].set_aspect('equal');    ax[0,1].set_aspect('equal')
    
    if subscripts == 'cartesian':
        ax[0,0].set_title(r'${}$'.format(varchar)+r'$_{xx}$')
        ax[0,1].set_title(r'${}$'.format(varchar)+r'$_{xy}$')
        ax[1,0].set_title(r'${}$'.format(varchar)+r'$_{yx}$')
        ax[1,1].set_title(r'${}$'.format(varchar)+r'$_{yy}$')
    elif subscripts == 'polar':
        ax[0,0].set_title(r'${}$'.format(varchar)+r'$_{r r}$')
        ax[0,1].set_title(r'${}$'.format(varchar)+r'$_{r \theta}$')
        ax[1,0].set_title(r'${}$'.format(varchar)+r'$_{\theta r}$')
        ax[1,1].set_title(r'${}$'.format(varchar)+r'$_{\theta\theta}$')
    elif subscripts == 'cartesiantensortheory':
        ax[0,0].set_title(r'${}$'.format(varchar)+r'$_{xx}$')
        ax[0,1].set_title(r'${}$'.format(varchar)+r'$_{yy}$')
        ax[1,0].set_title(r'${}$'.format(varchar)+r'$_{xx}$ theory')
        ax[1,1].set_title(r'${}$'.format(varchar)+r'$_{yy}$ theory')
    elif subscripts == 'cartesianvectortheory':
        ax[0,0].set_title(r'${}$'.format(varchar)+r'$_{x}$')
        ax[0,1].set_title(r'${}$'.format(varchar)+r'$_{y}$')
        ax[1,0].set_title(r'${}$'.format(varchar)+r'$_{x}$ theory')
        ax[1,1].set_title(r'${}$'.format(varchar)+r'$_{y}$ theory')
    elif subscripts == 'polarvectortheory':
        ax[0,0].set_title(r'${}$'.format(varchar)+r'$_{r}$')
        ax[0,1].set_title(r'${}$'.format(varchar)+r'$_{\theta}$')
        ax[1,0].set_title(r'${}$'.format(varchar)+r'$_{r}$ theory')
        ax[1,1].set_title(r'${}$'.format(varchar)+r'$_{\theta}$ theory')
    elif subscripts == 'polartensortheory':
        ax[0,0].set_title(r'${}$'.format(varchar)+r'$_{r r}$')
        ax[0,1].set_title(r'${}$'.format(varchar)+r'$_{\theta\theta}$')
        ax[1,0].set_title(r'${}$'.format(varchar)+r'$_{r r}$ theory')
        ax[1,1].set_title(r'${}$'.format(varchar)+r'$_{\theta\theta}$ theory')
        
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(scsc0, cax=cbar_ax)
    fig.text(0.5,0.975,title,horizontalalignment='center',verticalalignment='top')
    if not axis_on:
        ax[0,0].axis('off')
        ax[0,1].axis('off')
        ax[1,0].axis('off')
        ax[1,1].axis('off')
    plt.show()
    if close:
        plt.close('all')


def pf_plot_pcolormesh_scalar(x,y,C,outdir,name,ind,title,title2='',subtext='',subsubtext='',vmin='auto',vmax='auto',ptsz=10,cmap=cm.CMRmap,shape='circle'):
    """Save a single-panel plot of a scalar quantity C as colored pcolormesh
    
    Parameters
    ----------
    x, y : NxN mesh arrays
        the x and y positions of the points evaluated to Cx, Cy
    C : NxN arrays
        values for the plotted quantity C evaluated at points (x,y)
    outdir : string
        where to save the img
    name : string
        the name of the variable --> file will be saved as name_ind#.png
    varchar : string
        the variable name as a character (could be LaTeX formatted)
    ind : int
        index number for the image
    title : string
    title2 : string
        placed below title
    subtext : string
        placed below plot
    subsubtext : string
        placed at bottom of image
    vmin, vmax : float
        minimum, maximum value of C for colorbar; default is range of values in C
    ptsz : float
        size of colored marker (dot)
    """
    fig, ax = plt.subplots(1, 1)
    if isinstance(vmin,str):
        vmin = np.nanmin(C)
    if isinstance(vmax,str):
        vmax = np.nanmax(C)
    #scatter scale (for color scale)
    scsc = ax.pcolormesh(x, y, C, cmap=cmap, vmin=vmin, vmax=vmax)
    R = x.max()
    if shape == 'circle':
        t = np.arange(0,2*np.pi+0.01,0.01)
        plt.plot(R*np.cos(t),R*np.sin(t), 'k-')
    elif shape == 'square':
        t = np.array([-R,R])
        plt.plot( R*np.array([1,1]), t, 'k-')
        plt.plot( t, R*np.array([1,1]), 'k-')
        plt.plot( t,-R*np.array([1,1]), 'k-')
        plt.plot(-R*np.array([1,1]), t, 'k-')
    elif shape == 'unitsq':
        t = np.array([0,R])
        plt.plot( R*np.array([1,1]), t, 'k-')
        plt.plot( t, R*np.array([0,1]), 'k-')
        plt.plot( t,-R*np.array([0,1]), 'k-')
        plt.plot( np.array([0,0]), t, 'k-')
    elif shape == 'rectangle2x1':
        t = np.array([-R,R])
        plt.plot( R*np.array([1,1]), 2*t, 'k-')
        plt.plot( t, 2*R*np.array([1,1]), 'k-')
        plt.plot( t,-2*R*np.array([1,1]), 'k-')
        plt.plot(-R*np.array([1,1]), 2*t, 'k-')
        
    ax.set_aspect('equal')
    ax.axis('off');
    ax.set_title(title)
    fig.text(0.5,0.12, subtext,horizontalalignment='center' )
    fig.text(0.5,0.05, subsubtext,horizontalalignment='center' )
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(scsc, cax=cbar_ax)
    fig.text(0.5,0.98,title2,horizontalalignment='center',verticalalignment='top')
    savedir = prepdir(outdir)
    plt.savefig(savedir+name+'_'+'{0:06d}'.format(ind)+'.png')
    plt.close('all')

def pf_plot_scatter_scalar(x,y,C,outdir,name,ind,title,title2='',subtext='',subsubtext='',vmin='auto',vmax='auto',ptsz=10,cmap=cm.CMRmap,shape='none', ticks='off'):
    """Save a single-panel plot of a scalar quantity C as colored scatterplot
    
    Parameters
    ----------
    x, y : Nx1 arrays
        the x and y positions of the points evaluated to Cx, Cy
    C : Nx1 arrays
        values for the plotted quantity C evaluated at points (x,y)
    outdir : string
        where to save the img
    name : string
        the name of the variable --> file will be saved as name_ind#.png
    varchar : string
        the variable name as a character (could be LaTeX formatted)
    ind : int
        index number for the image
    title : string
    title2 : string
        placed below title
    subtext : string
        placed below plot
    subsubtext : string
        placed at bottom of image
    vmin, vmax : float
        minimum, maximum value of C for colorbar; default is range of values in C
    ptsz : float
        size of colored marker (dot)
    shape : string ('circle', 'square', 'unitsq', etc)
        characterization of the border to draw, default is 'none' --> no border
    ticks : string ('on' or 'off')
        whether or not to plot the axis (and tick marks)
    """
    fig, ax = plt.subplots(1, 1)
    if isinstance(vmin,str):
        vmin = np.nanmin(C)
    
    if isinstance(vmax,str):
        vmax = np.nanmax(C)
    
    #scatter scale (for color scale)
    scsc = ax.scatter(x, y, c=C, s= ptsz, cmap=cmap, edgecolor='', vmin=vmin, vmax=vmax)
    #scsc = ax.pcolormesh(x, y, C, cmap=cmap, vmin=vmin, vmax=vmax)
    R = x.max()
    if shape == 'circle':
        t = np.arange(0,2*np.pi+0.01,0.01)
        plt.plot(R*np.cos(t),R*np.sin(t), 'k-')
    elif shape == 'square':
        t = np.array([-R,R])
        plt.plot( R*np.array([1,1]), t, 'k-')
        plt.plot( t, R*np.array([1,1]), 'k-')
        plt.plot( t,-R*np.array([1,1]), 'k-')
        plt.plot(-R*np.array([1,1]), t, 'k-')
    elif shape == 'unitsq':
        t = np.array([0,R])
        plt.plot( R*np.array([1,1]), t, 'k-')
        plt.plot( t, R*np.array([1,1]), 'k-')
        plt.plot( t, np.array([0,0]), 'k-')
        plt.plot( np.array([0,0]), t, 'k-')
    elif shape == 'rectangle2x1':
        t = np.array([-R,R])
        plt.plot( R*np.array([1,1]), 2*t, 'k-')
        plt.plot( t, 2*R*np.array([1,1]), 'k-')
        plt.plot( t,-2*R*np.array([1,1]), 'k-')
        plt.plot(-R*np.array([1,1]), 2*t, 'k-')
    elif shape == 'rectangle1x2':
        t = np.array([-R,R])
        plt.plot( 2*R*np.array([1,1]), t, 'k-')
        plt.plot( 2*t, R*np.array([1,1]), 'k-')
        plt.plot( 2*t,-R*np.array([1,1]), 'k-')
        plt.plot(-2*R*np.array([1,1]), t, 'k-')
    
    ax.set_aspect('equal')
    if ticks=='off':
        ax.axis('off');
    ax.set_title(title)
    fig.text(0.5,0.12, subtext,horizontalalignment='center' )
    fig.text(0.5,0.05, subsubtext,horizontalalignment='center' )
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(scsc, cax=cbar_ax)
    fig.text(0.5,0.98,title2,horizontalalignment='center',verticalalignment='top')
    savedir = prepdir(outdir)
    plt.savefig(savedir+name+'_'+'{0:06d}'.format(ind)+'.png')
    plt.close('all')

def pf_plot_scatter_2panel(x,y,C0,C1,outdir,name,varchar,ind, title,title2='',subtext='',subsubtext='',vmin='auto',vmax='auto',ptsz=10, subscripts = 'cartesian', shape='',cmap=cm.CMRmap):
    """Save a two-panel plot of the components of a vector quantity (C0,C1) as colored scatterplot
    
    Parameters
    ----------
    x, y : Nx1 arrays
        the x and y positions of the points evaluated to C0, C1
    C0, C1 : Nx1 arrays
        values for the two plotted quantities evaluated at points (x,y)
    outdir : string
        where to save the img
    name : string
        the name of the variable --> file will be saved as name_ind#.png
    varchar : string
        the variable name as a character (could be LaTeX formatted)
    ind : int
        index number for the image
    title : string
    title2 : string
        placed below title
    subtext : string
        placed below plot
    subsubtext : string
        placed at bottom of image
    vmin : float
        minimum value of Cx or Cy for colorbar. Default is string 'auto', which prompts function to take min of C0
    ptsz : float
        size of colored marker (dot)
    subscripts : str ('cartesian', 'polar', 'theory')
        what coordinate system (and/or subset of elements) to use for naming the subplots; default is 'cartesian' (x,y), can be 'polar' (r,\theta)
    """
    #Plot and save u
    fig, ax = plt.subplots(1, 2)
    if isinstance(vmin,str):
        vmin = np.min( [ np.nanmin(C0),np.nanmin(C1)]  )
    if isinstance(vmax,str):
        vmax = np.max( [ np.nanmax(C0),np.nanmax(C1) ] )
    scu = ax[0].scatter(x, y, c=C0, s= ptsz, edgecolor='', vmin=vmin, vmax=vmax,cmap=cmap )
    ax[1].scatter(x, y, c=C1, s= ptsz, edgecolor='', vmin=vmin, vmax=vmax,cmap=cmap)
    R = x.max()
    if shape == 'circle':
        t = np.arange(0,2*np.pi+0.01,0.01)
        ax[0].plot(R*np.cos(t),R*np.sin(t), 'k-')
        ax[1].plot(R*np.cos(t),R*np.sin(t), 'k-')
    elif shape == 'unitsq':
        t = np.array([0,R])
        ax[0].plot( R*np.array([1,1]), t, 'k-')
        ax[0].plot( t, R*np.array([1,1]), 'k-')
        ax[0].plot( t, np.array([0,0]), 'k-')
        ax[0].plot( np.array([0,0]), t, 'k-')
        ax[1].plot( R*np.array([1,1]), t, 'k-')
        ax[1].plot( t, R*np.array([1,1]), 'k-')
        ax[1].plot( t, np.array([0,0]), 'k-')
        ax[1].plot( np.array([0,0]), t, 'k-')
    elif shape == 'square':
        t = np.array([-R,R])
        ax[0].plot( R*np.array([1,1]), t, 'k-')
        ax[0].plot( t, R*np.array([1,1]), 'k-')
        ax[0].plot( t,-R*np.array([1,1]), 'k-')
        ax[0].plot(-R*np.array([1,1]), t, 'k-')
        ax[1].plot( R*np.array([1,1]), t, 'k-')
        ax[1].plot( t, R*np.array([1,1]), 'k-')
        ax[1].plot( t,-R*np.array([1,1]), 'k-')
        ax[1].plot(-R*np.array([1,1]), t, 'k-')
    elif shape == 'rectangle2x1':
        t = np.array([-R,R])
        ax[0].plot( R*np.array([1,1]), 2*t, 'k-')
        ax[0].plot( t, 2*R*np.array([1,1]), 'k-')
        ax[0].plot( t,-2*R*np.array([1,1]), 'k-')
        ax[0].plot(-R*np.array([1,1]), 2*t, 'k-')        
        ax[1].plot( R*np.array([1,1]), 2*t, 'k-')
        ax[1].plot( t, 2*R*np.array([1,1]), 'k-')
        ax[1].plot( t,-2*R*np.array([1,1]), 'k-')
        ax[1].plot(-R*np.array([1,1]), 2*t, 'k-')
    elif shape == 'rectangle2x1':
        t = np.array([-R,R])
        ax[0].plot( 2*R*np.array([1,1]), t, 'k-')
        ax[0].plot( 2*t, R*np.array([1,1]), 'k-')
        ax[0].plot( 2*t,-R*np.array([1,1]), 'k-')
        ax[0].plot(-2*R*np.array([1,1]), t, 'k-')        
        ax[1].plot( 2*R*np.array([1,1]), t, 'k-')
        ax[1].plot( 2*t, R*np.array([1,1]), 'k-')
        ax[1].plot( 2*t,-R*np.array([1,1]), 'k-')
        ax[1].plot(-2*R*np.array([1,1]), t, 'k-')

    ax[0].set_aspect('equal'); ax[1].set_aspect('equal'); 
    ax[0].axis('off');  ax[1].axis('off');
    if subscripts == 'cartesian':
        ax[0].set_title(r'${}_x$'.format(varchar) ) 
        ax[1].set_title(r'${}_y$'.format(varchar) ) 
    elif subscripts == 'polar':
        ax[0].set_title(r'${}_r$'.format(varchar) ) 
        ax[1].set_title(r'${}_{\theta}$'.format(varchar) )
    elif subscripts == 'theory':
        ax[0].set_title(r'${}$ exprmt'.format(varchar) ) 
        ax[1].set_title(r'${}$ theory'.format(varchar) )
        
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(scu, cax=cbar_ax)
    fig.text(0.5,0.9, title2, horizontalalignment='center',verticalalignment='top')
    fig.text(0.5,0.05, subsubtext, horizontalalignment='center' )
    fig.text(0.5,0.15, subtext, horizontalalignment='center' )
    fig.text(0.5,0.975, title, horizontalalignment='center',verticalalignment='top')
    savedir = prepdir(outdir)
    plt.savefig(savedir+name+'_'+'{0:06d}'.format(ind)+'.png')
    plt.close('all')

def pf_plot_scatter_4panel(x,y,C00,C01,C10,C11,outdir,name,varchar,ind,title,title2,subtext,subsubtext,vmin='auto',vmax='auto',ptsz=3,subscripts='cartesian',cmap=cm.CMRmap,shape='none'):
    """Plot and save a four-panel plot of the components of a tensor quantity as colored scatterplot
    
    Parameters
    ----------
    x, y : Nx1 arrays
        the x and y positions of the points evaluated to C 
    C00, C01, C10, C11 : Nx1 arrays
        values for the plotted vector evaluated at points (x,y)
    outdir : string
        where to save the img
    name : string
        the name of the variable --> file will be saved as name_ind#.png
    varchar : string
        the variable name as a character (could be LaTeX formatted)
    ind : int
        index number for the image
    title : string
    title2 : string
        placed below title
    subtext : string
        placed below plot
    subsubtext : string
        placed at bottom of image
    vmin : float
        minimum value of Cij for colorbar. Default is string 'auto', which prompts function to take min of C00
    ptsz : float
        size of colored marker (dot)
    subscripts : str ('cartesian', 'polar', 'cartesiantheory', 'polartheory', 'cartesianvectortheory')
        what coordinate system (and/or subset of elements) to use for naming the subplots; default is 'cartesian' (x,y), can be 'polar' (r,\theta)
    """
    fig, ax = plt.subplots(2, 2)
    if isinstance(vmin,str):
        vmin = np.min( [ np.nanmin(C00), np.nanmin(C01), np.nanmin(C10), np.nanmin(C11) ] )
    if isinstance(vmax,str):
        vmax = np.max( [ np.nanmax(C00), np.nanmax(C01), np.nanmax(C10), np.nanmax(C11) ] )
    ptsz =10
    sccarte = ax[0,0].scatter(x, y, c=C00, s= ptsz, edgecolor='', vmin=vmin, vmax=vmax, cmap= cmap)
    ax[0,1].scatter(x, y, c=C01, s= ptsz, edgecolor='', vmin=vmin, vmax=vmax, cmap= cmap)
    ax[1,0].scatter(x, y, c=C10, s= ptsz, edgecolor='', vmin=vmin, vmax=vmax, cmap= cmap)
    ax[1,1].scatter(x, y, c=C11, s= ptsz, edgecolor='', vmin=vmin, vmax=vmax, cmap= cmap)
    R = x.max()
    if shape == 'circle':
        t = np.arange(0,2*np.pi+0.01,0.01)
        ax[0,0].plot(R*np.cos(t),R*np.sin(t), 'k-')
        ax[0,1].plot(R*np.cos(t),R*np.sin(t), 'k-')
        ax[1,0].plot(R*np.cos(t),R*np.sin(t), 'k-')
        ax[1,1].plot(R*np.cos(t),R*np.sin(t), 'k-')
    elif shape == 'unitsq':
        t = np.array([0,R])
        ax[0,0].plot( R*np.array([1,1]), t, 'k-')
        ax[0,0].plot( t, R*np.array([1,1]), 'k-')
        ax[0,0].plot( t, np.array([0,0]), 'k-')
        ax[0,0].plot( np.array([0,0]), t, 'k-')
        ax[1,0].plot( R*np.array([1,1]), t, 'k-')
        ax[1,0].plot( t, R*np.array([1,1]), 'k-')
        ax[1,0].plot( t, np.array([0,0]), 'k-')
        ax[1,0].plot( np.array([0,0]), t, 'k-')
        ax[0,1].plot( R*np.array([1,1]), t, 'k-')
        ax[0,1].plot( t, R*np.array([1,1]), 'k-')
        ax[0,1].plot( t, np.array([0,0]), 'k-')
        ax[0,1].plot( np.array([0,0]), t, 'k-')
        ax[1,1].plot( R*np.array([1,1]), t, 'k-')
        ax[1,1].plot( t, R*np.array([1,1]), 'k-')
        ax[1,1].plot( t, np.array([0,0]), 'k-')
        ax[1,1].plot( np.array([0,0]), t, 'k-') 
    elif shape == 'square':
        sfY = 1.0
        sfX = 1.0
    elif shape == 'rectangle2x1':
        sfX = 1.0
        sfY = 2.0
    elif shape == 'rectangle1x2':    
        sfX = 2.0
        sfY = 1.0
    
    if shape == 'square' or shape=='rectangle2x1' or shape == 'rectangle1x2':
        t = np.array([-R,R])
        s = R*np.array([1,1])
        ax[0,0].plot( sfX*s, sfY*t, 'k-')
        ax[0,0].plot( sfX*t, sfY*s, 'k-')
        ax[0,0].plot( sfX*t,-sfY*s, 'k-')
        ax[0,0].plot(-sfX*s, sfY*t, 'k-')        
        ax[0,1].plot( sfX*s, sfY*t, 'k-')
        ax[0,1].plot( sfX*t, sfY*s, 'k-')
        ax[0,1].plot( sfX*t,-sfY*s, 'k-')
        ax[0,1].plot(-sfX*s, sfY*t, 'k-')     
        ax[1,0].plot( sfX*s, sfY*t, 'k-')
        ax[1,0].plot( sfX*t, sfY*s, 'k-')
        ax[1,0].plot( sfX*t,-sfY*s, 'k-')
        ax[1,0].plot(-sfX*s, sfY*t, 'k-')     
        ax[1,1].plot( sfX*s, sfY*t, 'k-')
        ax[1,1].plot( sfX*t, sfY*s, 'k-')
        ax[1,1].plot( sfX*t,-sfY*s, 'k-')
        ax[1,1].plot(-sfX*s, sfY*t, 'k-')
        
    if subscripts == 'cartesian':
        ax[0,0].set_title(r'${}$'.format(varchar)+r'$_{xx}$')
        ax[0,1].set_title(r'${}$'.format(varchar)+r'$_{xy}$')
        ax[1,0].set_title(r'${}$'.format(varchar)+r'$_{yx}$')
        ax[1,1].set_title(r'${}$'.format(varchar)+r'$_{yy}$')
    elif subscripts == 'polar':
        ax[0,0].set_title(r'${}$'.format(varchar)+r'$_{r r}$')
        ax[0,1].set_title(r'${}$'.format(varchar)+r'$_{r \theta}$')
        ax[1,0].set_title(r'${}$'.format(varchar)+r'$_{\theta r}$')
        ax[1,1].set_title(r'${}$'.format(varchar)+r'$_{\theta\theta}$')
    elif subscripts == 'cartesiantensortheory':
        ax[0,0].set_title(r'${}$'.format(varchar)+r'$_{xx}$')
        ax[0,1].set_title(r'${}$'.format(varchar)+r'$_{yy}$')
        ax[1,0].set_title(r'${}$'.format(varchar)+r'$_{xx}$ theory')
        ax[1,1].set_title(r'${}$'.format(varchar)+r'$_{yy}$ theory')
    elif subscripts == 'cartesianvectortheory':
        ax[0,0].set_title(r'${}$'.format(varchar)+r'$_{x}$')
        ax[0,1].set_title(r'${}$'.format(varchar)+r'$_{y}$')
        ax[1,0].set_title(r'${}$'.format(varchar)+r'$_{x}$ theory')
        ax[1,1].set_title(r'${}$'.format(varchar)+r'$_{y}$ theory')
    elif subscripts == 'polarvectortheory':
        ax[0,0].set_title(r'${}$'.format(varchar)+r'$_{r}$')
        ax[0,1].set_title(r'${}$'.format(varchar)+r'$_{\theta}$')
        ax[1,0].set_title(r'${}$'.format(varchar)+r'$_{r}$ theory')
        ax[1,1].set_title(r'${}$'.format(varchar)+r'$_{\theta}$ theory')
    elif subscripts == 'polartensortheory':
        ax[0,0].set_title(r'${}$'.format(varchar)+r'$_{r r}$')
        ax[0,1].set_title(r'${}$'.format(varchar)+r'$_{\theta\theta}$')
        ax[1,0].set_title(r'${}$'.format(varchar)+r'$_{r r}$ theory')
        ax[1,1].set_title(r'${}$'.format(varchar)+r'$_{\theta\theta}$ theory')
        
    ax[0,0].axis('off');  ax[0,1].axis('off');   ax[1,0].axis('off');   ax[1,1].axis('off')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(sccarte, cax=cbar_ax)
    # global title
    fig.text(0.5,0.9, title2, horizontalalignment='center',verticalalignment='top')
    fig.text(0.5,0.05, subsubtext, horizontalalignment='center' )
    fig.text(0.5,0.15, subtext, horizontalalignment='center' )
    fig.text(0.5,0.975, title, horizontalalignment='center',verticalalignment='top')
    savedir = prepdir(outdir)
    plt.savefig(savedir+name+'_'+'{0:06d}'.format(ind)+'.png')
    plt.close('all')

def pf_plot_1D(x,functs,outdir,name,xlab,ylab,title,labels,ptsz=5):
    """Plot and save a 1D plot with any number of curves (columns of numpy array functs).
    
    Parameters
    ----------
    x : Nx1 array
        the x-axis values 
    functs : NxM array
        Each column is a set of values to plot against x
    outdir : string
        where to save the img
    name : string
        the name of the output image --> file will be saved as name.png
    xlab, ylab : strings
        the x and y labels
    title : string
    title2 : string
        placed below title
    subtext : string
        placed below plot
    subsubtext : string
        placed at bottom of image
    labels : dictionary
        The labels for each column of functs, with keys 0,1,2,3...
    ptsz : float
        size of colored marker (dot)
    """
    #style.use('ggplot')
    fig, ax = plt.subplots(1, 1)
    # Colors: red (could use #DD6331 like in publication), green, purple, yellow, blue, orange
    ax.set_color_cycle(['#B32525', '#77AC30', '#7E2F8E','#EFBD46', '#0E7ABF', '#D95419' ]) 
    ind = 0
    for funct in functs.T:
        # print funct
        plt.plot(x,funct,'.',label=labels[ind])
        ind += 1
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    savedir = prepdir(outdir)
    legend = ax.legend(loc='best')
    plt.savefig(savedir+name+'_'+'{0:06d}'.format(ind)+'.png')
    plt.close('all')

def pf_contour(x,y,z,level,Bxy,outdir,name,ind,title,title2,subtext,subsubtext):
    """Plot contour of z on at value=level.
    """
    fig, ax = plt.subplots(1, 1)
    fig.text(0.5,0.96, title2, horizontalalignment='center',verticalalignment='top')
    fig.text(0.5,0.02, subsubtext, horizontalalignment='center' )
    fig.text(0.5,0.05, subtext, horizontalalignment='center' )
    fig.text(0.5,0.99, title, horizontalalignment='center',verticalalignment='top')
    ax.contour(x, y, z, levels=[level],colors='k')
    ax.plot(Bxy[:,0],Bxy[:,1],'k-')
    ax.axis('equal')
    ax.axis('off')
    savedir = prepdir(outdir)
    plt.savefig(savedir+name+'_'+'{0:06d}'.format(ind)+'.png')
    plt.close('all')
    
def pf_contour_unstructured(x,y,z,n,level,Bxy,outdir,name,ind,title,title2,subtext,subsubtext):
    """Interpolate data (x,y,z) onto uniform grid of dim nxn, then plot as contour plot at value=level"""
    X,Y,Z = interpol_meshgrid(x,y,z,n)
    pf_contour(X,Y,Z,level,Bxy,outdir,name,ind,title,title2,subtext,subsubtext)
    
def pf_add_contour_to_plot(x,y,z,n,level,ax,color='k'):
    """Interpolate data (x,y,z) onto uniform grid of dim nxn, then add contour to axis ax (plotting contour of value=level)"""
    X,Y,Z = interpol_meshgrid(x,y,z,n)
    ax.contour(X, Y, Z, levels=[level],colors=color)

    
############################    
# Plot attributes
############################

def pf_titles(field,params,t):
    title= title_scalar(field,params,t)
    if 'BCUP' in params:
        if params['BCUP'] == 'essential':
            if params['surf']=='bump' or params['surf']=='monkeysaddle' or params['surf']=='hyperbolic-paraboloid' or params['surf']=='hyperbolic-paraboloid-transp':
                title2 = params['surf']+r': $x_0$='+'{0:.3f}'.format(params['x0'])+r' $\alpha$='+'{0:.3f}'.format(params['alph'])+r' $U=$'+'{0:.3f}'.format(params['U'])+' BC='+params['BCtype']+r' $R=$'+str(params['L']/2)+\
                         r' $\theta=$'+'{0:.3f}'.format(params['theta']/np.pi)
                HRx0str = r' $H/x_0$='+'{0:.3f}'.format(params['H']/params['x0']) 
            elif params['surf'] == 'flat':
                title2 = r'Flat: $U=$'+'{0:.3f}'.format(params['U'])+' BC='+params['BCtype']+r' $R=$'+str(params['L']/2)+r' $\theta=$'+'{0:.3f}'.format(params['theta']/np.pi)
                HRx0str = r' $H/R$='+'{0:.3f}'.format(params['H']*2./params['L'])
            elif params['surf'] == 'bumps2x1':
                print r' x0=['+'{0:0.2f}'.format(params['x0'][0])+','+'{0:0.2f}'.format(params['x0'][1])+']'
                title2 = r'Corrugated: $U=$'+'{0:.3f}'.format(params['U'])+' BC='+params['BCtype']+r' $R=$'+str(params['L']/2)+\
                        r' arngmt=[['+'{0:.4f}'.format(params['arngmt'][0][0]*2/params['L'])+','+\
                        '{0:.4f}'.format(params['arngmt'][0][1]*2/params['L'])+'],['+\
                        '{0:.4f}'.format(params['arngmt'][1][0]*2/params['L'])+','+\
                        '{0:.4f}'.format(params['arngmt'][1][1]*2/params['L'])+']]'+\
                        '\n\n'+\
                        r'$\alpha=$'+str(params['alph'])+\
                        r' $x_0/R$=['+'{0:0.2f}'.format(params['x0'][0]*2/params['L'])+','+'{0:0.2f}'.format(params['x0'][1]*2/params['L'])+']'+\
                        r' $\theta=$'+'{0:.3f}'.format(params['theta']/np.pi)
                HRx0str = r' $H/R$='+'{0:.3f}'.format(params['H']*2./params['L'])
            elif params['surf'][0:4] == 'bump' :
                print r' x0=['+'{0:0.2f}'.format(params['x0'][0])+',...]'
                title2 = params['surf'] +r': $U=$'+'{0:.3f}'.format(params['U'])+' BC='+params['BCtype']+r' $R=$'+str(params['L']/2)+\
                        r' arngmt=[['+'{0:.4f}'.format(params['arngmt'][0][0]*2/params['L'])+','+\
                        '{0:.4f}'.format(params['arngmt'][0][1]*2/params['L'])+'],['+\
                        '{0:.4f}'.format(params['arngmt'][1][0]*2/params['L'])+','+\
                        '{0:.4f}'.format(params['arngmt'][1][1]*2/params['L'])+'],...]'+\
                        '\n\n'+\
                        r'$\alpha=$'+str(params['alph'])+\
                        r' $x_0/R$=['+'{0:0.2f}'.format(params['x0'][0]*2/params['L'])+','+'{0:0.2f}'.format(params['x0'][1]*2/params['L'])+']'+\
                        r' $\theta=$'+'{0:.3f}'.format(params['theta']/np.pi)
                HRx0str = r' $H/R$='+'{0:.3f}'.format(params['H']*2./params['L'])
            elif params['surf'] == 'saddlebumps2x2':
                print r' x0=['+'{0:0.2f}'.format(params['x0'][0])+','+'{0:0.2f}'.format(params['x0'][1])+']'
                title2 = r'Corrugated: $U=$'+'{0:.3f}'.format(params['U'])+' BC='+params['BCtype']+r' $R=$'+str(params['L']/2)+\
                        r' arngmt=[['+'{0:.4f}'.format(params['arngmt'][0][0]*2/params['L'])+','+\
                        '{0:.4f}'.format(params['arngmt'][0][1]*2/params['L'])+'...]]'+\
                        '\n\n'+\
                        r'$\alpha=$'+str(params['alph'])+\
                        r' $x_0/R$=['+'{0:0.2f}'.format(params['x0'][0]*2/params['L'])+'...]'+\
                        r' $\theta=$'+'{0:.3f}'.format(params['theta']/np.pi)
                HRx0str = r' $H/R$='+'{0:.3f}'.format(params['H']*2./params['L'])
            else:
                title2 = params['surf']+r': $x_0$='+'{0:.3f}'.format(params['x0'])+r' $\alpha$='+'{0:.3f}'.format(params['alph'])+r' $P=$'+'{0:.3f}'.format(params['U'])+'E BC='+params['BCtype']+r' $R=$'+str(params['L']/2)+\
                         r' $\theta=$'+'{0:.3f}'.format(params['theta']/np.pi)
                HRx0str = r' $H/R$='+'{0:.3f}'.format(params['H']*2./params['L'])    
        elif params['BCUP'] == 'natural':
            if params['surf'] == 'bump':
                title2 = r'Bump: $x_0$='+'{0:.3f}'.format(params['x0'])+r' $\alpha$='+'{0:.3f}'.format(params['alph'])+r' $P=$'+'{0:.3f}'.format(params['U'])+'E BC='+params['BCtype']+r' $R=$'+str(params['L']/2)+\
                         r' $\theta=$'+'{0:.3f}'.format(params['theta']/np.pi)
                HRx0str = r' $H/x_0$='+'{0:.3f}'.format(params['H']/params['x0']) 
            elif params['surf'] == 'flat':
                title2 = r'Flat: $P=$'+'{0:.3f}'.format(params['U'])+'E BC='+params['BCtype']+r' $R=$'+str(params['L']/2)+r' $\theta=$'+'{0:.3f}'.format(params['theta']/np.pi)
                HRx0str = r' $H/R$='+'{0:.3f}'.format(params['H']*2./params['L'])
            elif params['surf'] == 'bumps2x1':
                print r' x0=['+'{0:0.2f}'.format(params['x0'][0])+','+'{0:0.2f}'.format(params['x0'][1])+']'
                title2 = r'Corrugated: $P=$'+'{0:.3f}'.format(params['U'])+'E BC='+params['BCtype']+r' $R=$'+str(params['L']/2)+\
                        r' arngmt=[['+'{0:.4f}'.format(params['arngmt'][0][0]*2/params['L'])+','+\
                        '{0:.4f}'.format(params['arngmt'][0][1]*2/params['L'])+'],['+\
                        '{0:.4f}'.format(params['arngmt'][1][0]*2/params['L'])+','+\
                        '{0:.4f}'.format(params['arngmt'][1][1]*2/params['L'])+']]'+\
                        '\n\n'+\
                        r'$\alpha=$'+str(params['alph'])+\
                        r' $x_0/R$=['+'{0:0.2f}'.format(params['x0'][0]*2/params['L'])+','+'{0:0.2f}'.format(params['x0'][1]*2/params['L'])+']'+\
                        r' $\theta=$'+'{0:.3f}'.format(params['theta']/np.pi)
                HRx0str = r' $H/R$='+'{0:.3f}'.format(params['H']*2./params['L'])
            elif params['surf'] == 'saddlebumps2x2':
                print r' x0=['+'{0:0.2f}'.format(params['x0'][0])+','+'{0:0.2f}'.format(params['x0'][1])+']'
                title2 = r'Corrugated: $P=$'+'{0:.3f}'.format(params['U'])+'E BC='+params['BCtype']+r' $R=$'+str(params['L']/2)+\
                        r' arngmt=[['+'{0:.4f}'.format(params['arngmt'][0][0]*2/params['L'])+','+\
                        '{0:.4f}'.format(params['arngmt'][0][1]*2/params['L'])+'...]]'+\
                        '\n\n'+\
                        r'$\alpha=$'+str(params['alph'])+\
                        r' $x_0/R$=['+'{0:0.2f}'.format(params['x0'][0]*2/params['L'])+'...]'+\
                        r' $\theta=$'+'{0:.3f}'.format(params['theta']/np.pi)
                HRx0str = r' $H/R$='+'{0:.3f}'.format(params['H']*2./params['L'])
            else:
                title2 = params['surf']+r': $x_0$='+'{0:.3f}'.format(params['x0'])+r' $\alpha$='+'{0:.3f}'.format(params['alph'])+r' $P=$'+'{0:.3f}'.format(params['U'])+'E BC='+params['BCtype']+r' $R=$'+str(params['L']/2)+\
                         r' $\theta=$'+'{0:.3f}'.format(params['theta']/np.pi)
                HRx0str = r' $H/R$='+'{0:.3f}'.format(params['H']*2./params['L'])    
        elif params['BCUP'] == 'natural-essential':
            if params['surf'] == 'bump':
                title2 = r'Bump: U=0 and $x_0$='+'{0:.3f}'.format(params['x0'])+r' $\alpha$='+'{0:.3f}'.format(params['alph'])+r' $P=$'+'{0:.3f}'.format(params['U'])+'E BC='+params['BCtype']+r' $R=$'+str(params['L']/2)+\
                         r' $\theta=$'+'{0:.3f}'.format(params['theta']/np.pi)
                HRx0str = r' $H/x_0$='+'{0:.3f}'.format(params['H']/params['x0']) 
            elif params['surf'] == 'flat':
                title2 = r'Flat: U=0 and $P=$'+'{0:.3f}'.format(params['U'])+'E BC='+params['BCtype']+r' $R=$'+str(params['L']/2)+r' $\theta=$'+'{0:.3f}'.format(params['theta']/np.pi)
                HRx0str = r' $H/R$='+'{0:.3f}'.format(params['H']*2./params['L'])
            else:       #elif params['surf'] == 'QGP':
                title2 = params['surf']+r': U=0 and $P=$'+'{0:.3f}'.format(params['U'])+'E BC='+params['BCtype']+r' $R=$'\
                            +str(params['L']/2)+r' $\theta=$'+'{0:.3f}'.format(params['theta']/np.pi) +' coldL0='+'{0:2f}'.format(params['coldL0'])
                HRx0str = r' $H/R$='+'{0:.3f}'.format(params['H']*2./params['L'])
    else:
        print 'WARNING: BCUP is not defined in parameters! Continuing anyway...'
        if params['surf']=='bump' or params['surf']=='monkeysaddle' or params['surf']=='hyperbolic-paraboloid' or params['surf']=='hyperbolic-paraboloid-transp':
            title2 = params['surf']+r': $x_0$='+'{0:.3f}'.format(params['x0'])+r' $\alpha$='+'{0:.3f}'.format(params['alph'])+r' $U=$'+'{0:.3f}'.format(params['U'])+' BC='+params['BCtype']+r' $R=$'+str(params['L']/2)+\
                     r' $\theta=$'+'{0:.3f}'.format(params['theta']/np.pi)
            HRx0str = r' $H/x_0$='+'{0:.3f}'.format(params['H']/params['x0']) 
        elif params['surf'] == 'flat':
            title2 = r'Flat: $U=$'+'{0:.3f}'.format(params['U'])+' BC='+params['BCtype']+r' $R=$'+str(params['L']/2)+r' $\theta=$'+'{0:.3f}'.format(params['theta']/np.pi)
            HRx0str = r' $H/R$='+'{0:.3f}'.format(params['H']*2./params['L'])
        elif params['surf'] == 'bumps2x1':
            print r' x0=['+'{0:0.2f}'.format(params['x0'][0])+','+'{0:0.2f}'.format(params['x0'][1])+']'
            title2 = r'Corrugated: $U=$'+'{0:.3f}'.format(params['U'])+' BC='+params['BCtype']+r' $R=$'+str(params['L']/2)+\
                    r' arngmt=[['+'{0:.4f}'.format(params['arngmt'][0][0]*2/params['L'])+','+\
                    '{0:.4f}'.format(params['arngmt'][0][1]*2/params['L'])+'],['+\
                    '{0:.4f}'.format(params['arngmt'][1][0]*2/params['L'])+','+\
                    '{0:.4f}'.format(params['arngmt'][1][1]*2/params['L'])+']]'+\
                    '\n\n'+\
                    r'$\alpha=$'+str(params['alph'])+\
                    r' $x_0/R$=['+'{0:0.2f}'.format(params['x0'][0]*2/params['L'])+','+'{0:0.2f}'.format(params['x0'][1]*2/params['L'])+']'+\
                    r' $\theta=$'+'{0:.3f}'.format(params['theta']/np.pi)
            HRx0str = r' $H/R$='+'{0:.3f}'.format(params['H']*2./params['L'])
        elif params['surf'][0:4] == 'bump' : #this is for channel, bumps1x5, etc
            print r' x0=['+'{0:0.2f}'.format(params['x0'][0])+',...]'
            title2 = params['surf'] +r': $U=$'+'{0:.3f}'.format(params['U'])+' BC='+params['BCtype']+r' $R=$'+str(params['L']/2)+\
                    r' arngmt=[['+'{0:.4f}'.format(params['arngmt'][0][0]*2/params['L'])+','+\
                    '{0:.4f}'.format(params['arngmt'][0][1]*2/params['L'])+'],['+\
                    '{0:.4f}'.format(params['arngmt'][1][0]*2/params['L'])+','+\
                    '{0:.4f}'.format(params['arngmt'][1][1]*2/params['L'])+'],...]'+\
                    '\n\n'+\
                    r'$\alpha=$'+str(params['alph'])+\
                    r' $x_0/R$=['+'{0:0.2f}'.format(params['x0'][0]*2/params['L'])+','+'{0:0.2f}'.format(params['x0'][1]*2/params['L'])+']'+\
                    r' $\theta=$'+'{0:.3f}'.format(params['theta']/np.pi)
            HRx0str = r' $H/R$='+'{0:.3f}'.format(params['H']*2./params['L'])
        elif params['surf'] == 'saddlebumps2x2':
            print r' x0=['+'{0:0.2f}'.format(params['x0'][0])+','+'{0:0.2f}'.format(params['x0'][1])+']'
            title2 = r'Corrugated: $U=$'+'{0:.3f}'.format(params['U'])+' BC='+params['BCtype']+r' $R=$'+str(params['L']/2)+\
                    r' arngmt=[['+'{0:.4f}'.format(params['arngmt'][0][0]*2/params['L'])+','+\
                    '{0:.4f}'.format(params['arngmt'][0][1]*2/params['L'])+'...]]'+\
                    '\n\n'+\
                    r'$\alpha=$'+str(params['alph'])+\
                    r' $x_0/R$=['+'{0:0.2f}'.format(params['x0'][0]*2/params['L'])+'...]'+\
                    r' $\theta=$'+'{0:.3f}'.format(params['theta']/np.pi)
            HRx0str = r' $H/R$='+'{0:.3f}'.format(params['H']*2./params['L'])    
    
    subtext = r'$\xi$='+'{0:3f}'.format(params['xi'])+' $\gamma$='+'{0:3f}'.format(params['gamma'])+r' $\tau$='+'{0:03f}'.format(params['Tau'])
    subsubtext = r'E='+str(params['E'])+r' $\nu$='+'{0:.2f}'.format(params['nu'])+ r' $E_c$='+'{0:.4f}'.format(params['Ec']) + \
                 r' a/$\xi$='+'{0:.3f}'.format(params['a']) + r' W/$\xi$='+'{0:.3f}'.format(params['W']) + \
                 r' $\beta/\pi$='+'{0:.3f}'.format(params['beta']/np.pi) +\
                 HRx0str + \
                 r' dt/$\tau$='+'{0:.3f}'.format(params['dt']/params['Tau'] )
    return title, title2, subtext, subsubtext


def title_scalar(field,params,t):
    t1 = r': t/$\tau$='+'{0:.4f}'.format(t/params['Tau'])
    t2 = r' L/$\xi$='+'{0:.3f}'.format(params['L']/params['xi'])
    t3 = r' n/$\xi$='+'{0:.3f}'.format(params['meshd'])
    if params['eta'] != 0.0:
        t4 = r' $\eta=$'+'{0:.3f}'.format(params['eta'])
    else:
        t4 = ''
    if 'velocity' in params:
        t5 = r' vel$\tau/L$='+'{0:.4f}'.format(params['velocity']*params['Tau']/params['L'])
        if 'restart_refine0' in params:
            #print '\nFound a restart!'
            rrIND_tmp = 0
            while 'restart_refine'+str(rrIND_tmp+1) in params:
                rrIND_tmp +=1
                print '\n rrIND_tmp=', rrIND_tmp
            
            t3 = r' n/$\xi$='+'{0:.3f}'.format(params['restart_refine'+str(rrIND_tmp)+'_meshd'])
    else:
        t5 = ''
    title = field + t1 + t2 + t3 + t4 + t5
    return title

def pf_title_QGP(surf,P,DT,decayL,coldL,L):
    return surf+r' $P=$'+str(P)+r' $\Delta T$='+str(DT)+r' decayL='+str(decayL)+r' coldL='+str(coldL)+r' $L=$'+str(L)


def title_scalar_static(field,params):
    return field+r': L/$\xi$='+'{0:.3f}'.format(params['L']/params['xi']) + r' n/$\xi$='+'{0:.3f}'.format(params['meshd'])+ r' $\eta=$'+'{0:.3f}'.format(params['eta'])

def pf_titles_static(field,params):
    title= title_scalar_static(field,params)
    if params['surf'] == 'bump':
        title2 = r'Bump: $x_0$='+str(params['x0'])+r' $\alpha$='+str(params['alph'])+r' $U=$'+str(params['U'])+' BC='+params['BCtype']+r' $R=$'+str(params['L']/2)
    elif params['surf'] == 'flat':
        title2 = r'Flat: $U=$'+str(params['U'])+' BC='+params['BCtype']+r' $R=$'+str(params['L']/2)
    elif params['surf'] == 'sphere':
        title2 = r'Sphere: $x_0$='+str(params['x0'])+r' $\alpha$='+str(params['alph'])+r' $U=$'+str(params['U'])+' BC='+params['BCtype']+r' $R=$'+str(params['L']/2)
    else:
        title2 = r'$x_0$='+str(params['x0'])+r' $\alpha$='+str(params['alph'])+r' $U=$'+str(params['U'])+' BC='+params['BCtype']+r' $R=$'+str(params['L']/2)
        
    subtext = r'$\xi$='+'{0:3f}'.format(params['xi'])+' $\gamma$='+'{0:3f}'.format(params['gamma'])
    if params['W']/params['xi'] < 1e-3:
        subsubtext = r'E='+str(params['E'])+r' $\nu$='+str(params['nu']) 
    else:
        subsubtext = r'E='+str(params['E'])+r' $\nu$='+str(params['nu'])+ \
                     r' a/$\xi$='+'{0:.3f}'.format(params['a']) + r' W/$\xi$='+'{0:.3f}'.format(params['W']) + \
                     r' $\beta/\pi$='+'{0:.3f}'.format(params['beta']/np.pi) 
    return title, title2, subtext, subsubtext
    

##########################################
## Files, Folders, and Directory Structure
##########################################
def prepdir(dir):
    """Make sure that the (string) variable dir ends with the character '/'.
    This prepares the string dir to be an output directory."""
    if dir[-1]=='/':
        return dir
    else:
        return dir+'/'
      
def ensure_dir(f):
    """Check if directory exists, and make it if not.
    
    Parameters
    ----------
    f : string
        directory path to ensure
                            
    Returns
    ----------
    """
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
        
def find_dir_with_name(name,searchdir):
    """Return a path or list of paths to directories which match the string 'name' (can have wildcards) in searchdir.
    Note that this function returns names with a trailing back slash (/)"""
    if name=='':
        """no name given, no name returned"""
        return []
    else:
        possible_dirs = glob.glob(searchdir+name)
        okdirs = [os.path.isdir(possible_dir) for possible_dir in possible_dirs]
        out = [possible_dirs[i]+'/' for i in range(len(okdirs)) if okdirs[i] ]
        if len(out)==1: return out[0]
        else: return out


    
def find_subsubdirectory(string,maindir):
    """Find subsubdir matching string, in maindir. Return subdir and subsubdir names.
    If there are multiple matching subdirectories, returns list of strings.
    If there are no matches, returns empty lists.
    """
    maindir = prepdir(maindir)
    #print 'maindir = ', maindir
    contents = glob.glob(maindir+'*')
    is_subdir = [os.path.isdir(ii) for ii in contents]
    
    if len(is_subdir) == 0:
        print 'WARNING! Found no matching subdirectory: returning empty list'
        return is_subdir, is_subdir
    else:
        #print 'contents = ', contents
        subdirs = [contents[ii] for ii in np.where(is_subdir)[0].tolist()]
        #print 'subdirs = ', subdirs
    
    found = False
    subsubdir = []
    for ii in subdirs:
        #print 'ii =', ii
        print 'prepdir(ii)+string = ',prepdir(ii)+string
        subcontents = glob.glob(prepdir(ii)+string)
        #print 'glob.glob(',prepdir(ii),string,') = ',subcontents
        is_subsubdir = [os.path.isdir(jj) for jj in subcontents]
        subsubdirs = [subcontents[jj] for jj in np.where(is_subsubdir)[0].tolist()]
        #print 'subsubdirs = ', subsubdirs
        if len(subsubdirs)>0:
            if found == False:            
                if len(subsubdirs) == 1:
                    subdir = prepdir(ii)
                    subsubdir = prepdir(subsubdirs[0])
                    #print 'adding first subdir = ', subdir
                    found = True                
                elif len(subsubdirs) >1:
                    subdir = [prepdir(ii)]*len(subsubdirs)
                    #print 'adding first few subdir = ', subdir
                    found = True
                    subsubdir = [0]*len(subsubdirs)
                    for j in range(len(subsubdirs)):
                        subsubdir[j] = prepdir(subsubdirs[j])
            else:
                # Since already found one, add another
                #print ' Found more subsubdirs'
                #print 'subdir = ', subdir
                
                # Add subdir to list
                if isinstance(subdir,str):
                    subdir = [subdir,prepdir(ii)]
                    print 'adding second to subdir = ', subdir
                    if len(subsubdirs)>1:
                        for kk in range(1,len(subsubdirs)):
                            subdir.append(prepdir(ii))
                        print 'adding second (multiple) to subdir = ', subdir
                else:
                    print 'subsubdirs'
                    for kk in range(1,len(subsubdirs)):
                        subdir.append(prepdir(ii))
                        #print 'subsubdirs = ', subsubdirs
                        print 'adding more to subdir = ', subdir
                # Add subsubdir to list
                for jj in subsubdirs:
                    if isinstance(subsubdir,str):
                        subsubdir = [subsubdir,prepdir(jj)]
                        print 'adding second to subsubdirs = ', subsubdir
                    else:                    
                        subsubdir.append(prepdir(jj))
                        print 'adding more to subsubdirs = ', subsubdir
                
    if found:
        return subdir, subsubdir
    else:
        return '',''
        

##########################################
# Specific Geometric Setups
##########################################

##########################################
## A. Inclined Crack in Uniaxial loading 
##########################################

def ICUL_kink_angle(beta):
    """Compute kink angle for an Inclined Crack in a Uniaxially Loaded plate (ICUL)"""
    eta = np.cos(beta)/ np.sin(beta)
    kink = -2* np.arctan(2*eta/ (1+ np.sqrt(1+8.*eta**2)))
    return kink

##########################################
## B. Quenched Glass Plate (QGP)
##########################################
def Z_Ttube_approx(x,y, decayL=0.2, DT=1.0, P=7.9, coldL=0.0, totLen=0.12, minY = 0.0, L=0.12, polyorder='Quartic4_2xW'):
    """Project x,y pts to surface of cylinder which narrows in a manner that approximates the curvature
    distribution of the glass plate in the limit of one radius of curvature (that of the tube) nearly constant.
    Some arguments are required for maximum efficiency, such as totLen = max(y)-min(y).
    
    Parameters
    ----------
    decayL : fraction of L that is used for decay
    totLen : height of sample, could be 4*R, for instance
    L : 2*R --> width, also radius of cylinder in cold region
    
    """
    # First do zi and pi
    z0 = np.amin( np.dstack((L*np.ones(len(x)),L-L*DT*(1-np.exp(-P*(y-minY-coldL*totLen)))))[0],axis=1)
    cL = coldL*totLen
    dL = decayL*L
    if polyorder == 'Quartic4_2xW':
        # negative y inds
        zi = y < minY+cL-dL
        ni = np.logical_and(y < minY+cL+dL, y >minY+cL-dL)
        pi = y > minY+cL+dL
        #print 'len(y)=', len(y)
        #print zi
        #print 'len(zi)=', np.where(zi)
        #print ni
        #print 'len(ni)=', len(np.where(ni))
        #print pi
        #print 'len(pi)=', len(np.where(pi==True))
        # Replace Heaviside with 3-7th order polynomial
        d = dL
        A = - np.exp(-P*d)*DT*L*(-105.  + \
                105. * np.exp(P*d) -     \
                90.  * P*d -             \
                30.  * P**2*d**2 -       \
                4.   * P**3*d**3)/(48.*d**4) 
        B = np.exp(-P*d) *DT* L *(-42.   +             \
             42. * np.exp(P*d) -              \
             39. * P*d -                      \
             14. * P**2*d**2 -                \
             2.  * P**3*d**3)/(16.*d**5)
        C = - np.exp(-P*d) *DT* L* (-35.  +          \
              35 * np.exp(P*d) -             \
              34 * P*d -                     \
              13 * P**2*d**2 -               \
              2  * P**3*d**3)/(32.*d**6)
        D = np.exp(-P*d) *DT* L * (-15.   +          \
                15. * np.exp(P*d) -                          \
                15. * P*d - 6.  * P**2*d**2 -             \
                P**3*d**3)/(96.*d**7)
        # Offset y --> yni by dL so that effectively polynomial is funct of 2*epsilon (ie 2*decayL)
        yni = y[ni]-minY-cL+dL
        z0[ni] = L+A*yni**4 + B*yni**5 + C*yni**6 + D*yni**7   # Heaviside --> make this quickly decaying polynomial

    f = z0*np.cos(x/z0)
    return f


def Ktinterp(ylin,coldL=2.0,decayL=0.2,alph=1.0, P=7.9, Lscale=1.0, polyorder='Quartic4_2xW'):
    """Return an interpolation of the target curvature for a temperature profile in a QGP.
    Let y=0 be the base of the strip. Usually distances are measured in units of strip halfwidth.
    
    Parameters
    ----------
    ylin : Nx1 array
        linspace over which to interpolate the curvature; must be evenly spaced
    alph : float (default = 1.0)
        coefficient of thermal expansion (overall scaling of G)
    P : float (default = 7.9)
        Peclet number = b*v/D  (halfwidth x velocity / coefficient of thermal diffusion)
    Lscale : float
        Length scale of the half strip width in other units. For ex, in units of the radius of curv of a surface
    """
    xs = sp.Symbol('xs')
    Ts = (1. - sp.exp(-P*(xs-coldL)))
    fTs = sp.lambdify(xs, Ts, 'numpy')
    dy = ylin[2] - ylin[1] #grab one of the difference values --> must all be the same
    T = fTs(ylin)
    if polyorder == 'Quartic4':
        # negative y inds
        zi = ylin < coldL
        ni = np.logical_and(ylin <coldL+decayL, ylin > coldL)
        pi = ylin > coldL+decayL  
        # Replace Heaviside with 3-7th order polynomial
        d = decayL
        A =   np.exp(-P*d)*(-210. + 210.*np.exp(P*d) - 90.*P*d - 15.*P**2*d**2 - P**3*d**3)/(6.*d**4)
        B = - np.exp(-P*d)*(-168. + 168.*np.exp(P*d) - 78.*P*d - 14.*P**2*d**2 - P**3*d**3)/(2.*d**5)
        C =   np.exp(-P*d)*(-140. + 140.*np.exp(P*d) - 68.*P*d - 13.*P**2*d**2 - P**3*d**3)/(2.*d**6)
        D = - np.exp(-P*d)*(-120. + 120.*np.exp(P*d) - 60.*P*d - 12.*P**2*d**2 - P**3*d**3)/(6.*d**7)
        y = ylin[ni]-coldL  
        T[ni] = A*y**4 + B*y**5 + C*y**6 + D*y**7   # Heaviside --> make this quickly decaying polynomial
    elif polyorder == 'Quartic4_2xW':
        # negative y inds
        zi = ylin < coldL - decayL
        ni = np.logical_and(ylin <coldL+decayL, ylin > coldL -decayL)
        pi = ylin > coldL+decayL  
        # Replace Heaviside with 3-7th order polynomial
        d = decayL
        A =   np.exp(-P*d)*(-105. + 105.*np.exp(P*d) - 90.*P*d - 30.*P**2*d**2 - 4.*P**3*d**3)/(48.*d**4)
        B = - np.exp(-P*d)*(- 42. +  42.*np.exp(P*d) - 39.*P*d - 14.*P**2*d**2 - 2.*P**3*d**3)/(16.*d**5)
        C =   np.exp(-P*d)*(- 35. +  35.*np.exp(P*d) - 34.*P*d - 13.*P**2*d**2 - 2.*P**3*d**3)/(32.*d**6)
        D = - np.exp(-P*d)*(- 15. +  15.*np.exp(P*d) - 15.*P*d -  6.*P**2*d**2 - 1.*P**3*d**3)/(96.*d**7)
        y = ylin[ni] - coldL + decayL
        T[ni] = A*y**4 + B*y**5 + C*y**6 + D*y**7   # Heaviside --> make this quickly decaying polynomial
        
    T[zi] = 0.
    Ty = np.gradient(T , dy)
    Tyy= np.gradient(Ty, dy)
    #plt.plot(ylin, T, 'k.', label='T')
    #plt.plot(ylin, Ty, 'g.', label='Ty')
    #plt.plot(ylin, Tyy, 'b.', label='Tyy')
    #plt.legend()
    #plt.title(r'$T$, $\partial_y T$, $\partial_y^2 T$')
    #plt.show()
    Ktinterp = scipy.interpolate.interp1d(ylin,alph*Tyy/Lscale**2)
    return Ktinterp




########
# DEMO #
########
    
if __name__ == "__main__":
    demo_arrow_mesh = True
    demo_tensor = True
    demo_vectfield = True
    demo_linept = True
    demo_gaussiancurvature = True
    demo_gaussiancurvature2 = True
    demo_Ztube = True
    demo_initial_phase_multicrack = True
    
    ptsz =50 #size of dot for scatterplots
    from mpl_toolkits.mplot3d import Axes3D
        

    if demo_arrow_mesh:
        print 'Demonstrating arrow_mesh function: makes custom arrows in 3D'
        fig = plt.figure() #figsize=plt.figaspect(1.0))
        ax = fig.gca(projection='3d')
        x=.5; y=.5; z=1.0;
        # Make a bunch of arrows
        p0,t0 = arrow_mesh(x,y,0,1,0,0)
        p1, t1 = arrow_mesh(x,y,0,1/np.sqrt(2),0,1/np.sqrt(2),rotation_angle=0.0*np.pi) 
        p2, t2 = arrow_mesh(x,y,0,1,0,0,rotation_angle=0.0*np.pi) 
        p3, t3 = arrow_mesh(x,y,z,1.0,0.5,0.5,rotation_angle=0.0*np.pi, head_length=0.1) 
        p4, t4 = arrow_mesh(x,y,z,0.1,-0.5,-0.3,rotation_angle=0.25*np.pi, overhang=0.1 )
        p5, t5 = arrow_mesh(x,y,z,0.0,0.5,-0.5,rotation_angle=0.5*np.pi,tail_width=0.1, overhang=0.2 )
        p6, t6 = arrow_mesh(x,y,z,0.2,0.3,0.5,rotation_angle=0.75*np.pi,head_width=0.8, overhang=0.1 )
        p = [p0,p1,p2,p3,p4,p5,p6]
        for ii in range(len(p)):
            pi = p[ii]
            ax.plot_trisurf(pi[:,0], pi[:,1], pi[:,2], triangles=t0,cmap=cm.jet )    
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        plt.show()
        plt.close('all')
        
    if demo_tensor:
        #Show gaussian bump stress
        print('Demonstrating calculation and easy display of Stress field for Gaussian Bump and conversion to Cartesian coords')
        x = np.linspace(-5,5) 
        y = np.linspace(-5,5) 
        xv, yv = np.meshgrid(x, y)
        x = xv.ravel()
        y = yv.ravel()
        t = np.sqrt(x**2 + y**2)
        
        alph = 0.3; x0 = 1.5; R = 5; U = 0.0; nu = 0.4;
        Srr, Stt = GB_stress_uDirichlet(alph, x0, R, U, nu, t)
        Srt = np.zeros_like(Srr)
        Str = np.zeros_like(Srr)
        
        Sxx, Sxy, Syx, Syy = tensor_polar2cartesian2D(Srr,Srt,Str,Stt,x,y)    
    
        #Polar version
        title0 = r'Polar Stresses for Bump: $x_0$='+str(x0)+r' $\alpha$='+str(alph)+r' $U=$'+str(U)+r' $R=$'+str(R)
        pf_display_tensor(x,y,Srr,Srt,Str,Stt,r'\sigma',title=title0,subscripts='polar',ptsz=20,axis_on=0)
        
        #Cartesian version
        title0 = r'Cartesian Stresses for Bump: $x_0$='+str(x0)+r' $\alpha$='+str(alph)+r' $U=$'+str(U)+r' $R=$'+str(R)
        pf_display_tensor(x,y,Sxx,Sxy,Syx,Syy,r'\sigma',title=title0,subscripts='cartesian',ptsz=20,axis_on=0)
        
    
    if demo_vectfield:
        #Show gaussian bump displacement in r,theta
        print('Demonstrating calculation and easy display of displacement field for Gaussian Bump and sinusoidal field')
        x = np.linspace(-5,5)
        y = np.linspace(-5,5) 
        xv, yv = np.meshgrid(x, y)
        x = xv.ravel()
        y = yv.ravel()
        t = np.sqrt(x**2 + y**2)
        
        alph = 0.3; x0 = 1.5; R = 5; U = 0.02; nu = 0.4; 
        ur = GB_displacement_uDirichlet(alph,x0,R,U,nu,t)
        ut = np.zeros_like(ur)
            
        varchar= r'u'
        title0 = r'Bump: $x_0$='+str(x0)+r' $\alpha$='+str(alph)+r' $U=$'+str(U)+r' $R=$'+str(R)
        pf_display_vector(x,y,ur,ut,varchar,title=title0,subscripts='polar',ptsz=20,axis_on=0)
        
        #Show conversion from displacement field to polar coords
        print('Demonstrating conversion from cartesian displacement field to polar coords and vice versa')
        ux0 = np.cos(x)
        uy0 = np.cos(y)
        ur,ut = vectorfield_cartesian2polar(ux0,uy0,x,y)
        ux,uy = vectorfield_polar2cartesian(ur,ut,x,y)
        pf_display_4panel(x,y,ur,ut,ux,uy,r'Sine field $u_r$',title1=r'Sine field $u_\theta$',\
                          title2=r'Sine field $u_x$',title3=r'Sine field $u_y$', ptsz=20,axis_on=0)
    
    if demo_linept:
        print 'Demo: Define value based on distance from a line segment (used for creating initial state of phase for a crack).'
        print 'This demo focuses on a function that does this for one point at a time (inefficient in numpy but useful in some contexts in FEniCS).'
        pts = np.random.random((4000,2))
        W = .3
        endpt1 = [W,W]
        endpt2 = [1.-W,1.-W]
        
        value = np.zeros_like(pts[:,0])
        ind = 0
        for pt in pts:
            print 'pt=', pt
            x = [pt[0],pt[1]]
            print 'x=', x
            #p, d = closest_pt_on_lineseg(x,endpt1, endpt2)
            value[ind] = initphase_linear_slit(x, endpt1, endpt2, W, contour='linear')
            ind += 1
        pf_display_scalar(pts[:,0],pts[:,1],value,'Phase values near a crack',ptsz=40,axis_on=0)
        plt.show()
        
    if demo_gaussiancurvature:
        print 'Demo: Demonstrating gaussian_curvature_unstructured: measuring the Gaussian curvature of a surface defined by a collection of points'
        X = np.arange(-5, 5, 0.2)
        Y = np.arange(-5, 5, 0.2)
        X, Y = np.meshgrid(X, Y)
        X = X.ravel()
        Y = Y.ravel()
        R = np.sqrt(X**2 + Y**2)
        Z = np.exp(-R**2/(2*np.mean(R)**2))
        K, xgrid, ygrid, Kgrid = gaussian_curvature_bspline(Z,X,Y,N=100)
        fig, ax = plt.subplots(1, 2)
        color = ax[0].scatter(xgrid,ygrid,c=Kgrid,edgecolor='')
        ax[1].scatter(X,Y,c=K,edgecolor='')
        ax[0].set_title('Curvature --mesh pts')
        ax[1].set_title('Curvature --evenly spaced array pts')
        plt.colorbar(color)
        plt.show()
        print np.shape(xgrid), np.shape(ygrid), np.shape(Kgrid)
        print np.shape(X), np.shape(Y), np.shape(K)

        ## Load x,y,z from text file and compute curvature 
        # fname = '/Users/npmitchell/Desktop/data_local/20151022/20151022-1120_QGP_fixbotY_Tri_N10000_dt0p000_HoR0p080_beta0p50/height.txt'
        # X,Y,Z = np.loadtxt(fname, skiprows=1, delimiter=',', unpack=True)
        # K, xgrid, ygrid, Kgrid = gaussian_curvature_unstructured(Z,X,Y,N=100)
        # fig, ax = plt.subplots(1, 2)
        # color = ax[0].scatter(xgrid,ygrid,c=Kgrid,edgecolor='')
        # ax[1].scatter(X,Y,c=K,edgecolor='')
        # plt.colorbar(color)
        # plt.show()
        # print np.shape(xgrid), np.shape(ygrid), np.shape(Kgrid)
        # print np.shape(X), np.shape(Y), np.shape(K)

    if demo_gaussiancurvature2:
        print 'Demo: Demonstrating gaussian_curvature_unstructured2: measuring the Gaussian curvature of a surface defined by a collection of points in another way.'
        x = np.random.random((5000,)).ravel()-0.5
        y = np.random.random((5000,)).ravel()-0.5
        sigma = 0.8
        z = sigma*np.exp(-(x**2+y**2)/(2.*sigma**2))
        
        xy = np.dstack((x,y))[0]
        xyz = np.dstack((x,y,z))[0]
        print 'Triangulating...'
        Triang = Delaunay(xy)
        temp = Triang.vertices
        print 'Flipping orientations...'
        Tri = flip_orientations_tris(temp,xyz)
        #proxy for avg distance between points
        dist = np.mean(np.sqrt((x[Tri[:,0]]-x[Tri[:,1]])**2+(y[Tri[:,0]]-y[Tri[:,1]])**2))
        dx = dist*0.5
        print 'x=', x
        print 'y=', y
        
        K, xgrid, ygrid, Kgrid = gaussian_curvature_unstructured(x,y,z,dx,N=3)
        print 'shape(K)=', np.shape(K)
        fig, ax = plt.subplots(1, 2)
        color = ax[0].scatter(x,y,c=K,edgecolor='')
        ax[1].scatter(xgrid.ravel(),ygrid.ravel(),c=Kgrid.ravel(),edgecolor='')
        plt.colorbar(color)
        fig.text(0.5,0.94,'GCurvature using kd-tree and lookup',horizontalalignment='center')
        plt.show()
    
    if demo_Ztube:
        print 'Demonstrating Z_Ttube_approx: the projection of xy points to a tube with a piecewise defined shape: flat, polynomial, exponential'
        Y = np.arange(-1, 1, 0.005)
        X = np.arange(-1, 1, 0.005) # np.zeros_like(Y) 
        X, Y = np.meshgrid(X, Y)
        x = X.ravel()
        y = Y.ravel()
        decayL =0.05
        DT = 0.15
        coldL= 0.5
        P = 10
        L = 1.0
        z =  Z_Ttube_approx(x,y, decayL=decayL, DT=DT, P=P, coldL=coldL, totLen=2., minY=np.min(y), L=L, polyorder='Quartic4_2xW')
        #Compute Gaussian curvature
        fdir = '/Users/npmitchell/Dropbox/Soft_Matter/PhaseField_Modeling/FEniCS/data_out/static/'
        fname = fdir+'Ztube_GCurvature_P'+'{0:0.2f}'.format(P) \
                +'_DT'+'{0:0.2f}'.format(DT) \
                +'_decayL'+'{0:0.2f}'.format(decayL) \
                +'_coldL'+'{0:0.2f}'.format(coldL) \
                +'_L'+'{0:0.2f}'.format(L) \
                +'.png'
        
        K, xgrid,ygrid,Kgrid = gaussian_curvature_bspline(z,x,y, N=100)
        fig, ax = plt.subplots(1, 2)
        color = ax[0].scatter(xgrid,ygrid,c=Kgrid,edgecolor='',cmap='coolwarm', \
                              vmin=-np.max(np.abs(Kgrid)), vmax=np.max(np.abs(Kgrid)))
        ax[0].set_xlim(np.min(xgrid),np.max(xgrid))
        ax[0].set_ylim(np.min(ygrid),np.max(ygrid))
        ax[0].set_aspect('equal')
        ax[0].set_title(r'$K(x,y)$')
        
        ax[1].plot(y[np.abs(x)<0.1],K[np.abs(x)<0.1],'.')
        ax[1].set_xlim(-0.3,0.3)
        ax[1].set_ylabel(r'$K$')
        ax[1].set_xlabel(r'$y$')
        ax[1].set_title(r'$K(y)$')
        titletext = pf_title_QGP('Ztube',P,DT,decayL,coldL,L)
        fig.text(0.5,0.94,titletext,horizontalalignment='center')
        plt.colorbar(color)
        plt.savefig(fname)
        plt.show()
        
    if demo_initial_phase_multicrack:
        xy = (np.random.random((10000,2))-0.5)*2.0
        H = np.array([-0.2,0.0,0.3])
        Y = np.array([-0.4,0.0,0.4])
        beta =np.array([0.0, 0.25*np.pi, 0.6*np.pi])
        W = 0.1
        a = np.array([0.20, 0.05, 0.5])
        xi = 0.05
        phi = initialPhase_vec(xy, H, Y, beta, W, a, xi, fallofftype = 'linear')
        pf_display_scalar(xy[:,0],xy[:,1],phi,'Phase field for some arbitrary cracks generated by initialPhase_vec()',\
                    cmap=cm.jet)
        
        # Use same function for single crack
        xi = 0.15
        H = 0.1; Y=0.2; beta =0.25*np.pi; W = 0.3; a=0.3
        phi = initialPhase_vec(xy, H, Y, beta, W, a, xi, fallofftype = 'polygauss')
        pf_display_scalar(xy[:,0],xy[:,1],phi,'Demonstrating using same function for single crack: initialPhase_vec()',
                    cmap=cm.jet)
        