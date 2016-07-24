from dolfin import *
import numpy as np

'''Evolve KKL phasefield equations with damage (d phi/dt <0 locally) in a scatter setup:
A thin, flat sheet is conformed to a substrate in the shape of a of a single bump, with a crack at the edge of a
circular (disk-shaped) sample.
To use this example, first make a mesh file that can be loaded, or else set args.shape == 'UnitSquare' (which is a
built-in mesh in FEniCS). Save your mesh as:
'../meshes/'+shape+'Mesh_'+meshtype+add_exten+'_eta'+etastr+'_R'+Rstr+'_N'+str(int(N))+'.xml'
where shape is 'square' or 'circle', meshtype is typically 'Triangular',
add_exten is '_theta'+ '{0:.3f}'.format(theta/np.pi).replace('.','p')+'pi' where theta is the angle in radians that the
created mesh has been rotated, Rstr is '{0:.3f}'.format(R).replace('.','p'), where R is the radius or half-width of the
mesh in coordinate space, and N is the number of vertices in the mesh (which can be specified via args.Npts using the
argparser.
You can create such a mesh using mesh_generateion_xml_fenics.py.

If you intend to use pieces of this code for research or other purposes, please email me at noah.prentice.mitchell@gmail.com.
'''

##############################
# Functions ##################
##############################
def u_boundary(x, on_boundary):
    """define the Dirichlet boundary as pts on boundary"""
    return on_boundary

def pwmin(x, y):
    """Pointwise minimum between two array-like objects
    """
    z = x - y
    z.abs()
    z -= x + y
    z /= -2.0
    return z

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
    p = [a*(a*x+b*y)/(a**2+b**2) + endpt1[0], b*(a*x+b*y)/(a**2+b**2) + endpt1[1]]
    return p


def line_pt_is_on_lineseg(p, a, b):
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
    if dotproduct < 0:
        return False

    squaredlengthba = (b[0] - a[0])*(b[0] - a[0]) + (b[1] - a[1])*(b[1] - a[1])
    if dotproduct > squaredlengthba:
        return False

    return True


def closest_pt_on_lineseg_dolf(pt, endpt1, endpt2):
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
        d0 = sqrt((p[1]-pt[1])**2 + (p[0]-pt[0])**2)
        return p, d0

    else:
        d1 = (endpt1[1]-pt[1])**2 + (endpt1[0]-pt[0])**2
        d2 = (endpt2[1]-pt[1])**2 + (endpt2[0]-pt[0])**2
        # p is not on the segment, so pick closer endpt
        if d1 < d2:
            return endpt1, sqrt(d1)
        else:
            return endpt2, sqrt(d2)


class initialPhase(Expression):
    """Define the initial phase (damage field) for the sheet"""
    def eval(self, value, x):
        C = 0.966
        polysig = W/xi
        polya = -0.392
        polyb = 0.7654
        endpt1 = [ H - a*cos(beta) , Y - a*sin(beta)]
        endpt2 = [ H + a*cos(beta) , Y + a*sin(beta)]
        # check if point is anywhere near line before doing calcs
        minx = min(endpt1[0],endpt2[0]) - 4*W
        maxx = max(endpt1[0],endpt2[0]) + 4*W
        if (x[0] < minx) or (x[0] > maxx):
            value[0] = 1.0
        else:
            # check y value
            miny = min(endpt1[1],endpt2[1]) - 4*W
            maxy = max(endpt1[1],endpt2[1]) + 4*W
            if (x[1] < miny) or (x[1] > maxy):
                value[0] = 1.0
            else:
                # check if point is anywhere near line before doing calcs
                p , dist = closest_pt_on_lineseg_dolf([x[0], x[1]], endpt1, endpt2)
                if dist <= 4*W :
                    # value[0] = (dist / W)*0.95 +0.05 #linear/abs
                    value[0] = 1-C*exp(-(dist/xi)**2/(2*polysig**2))*(1+polya*(dist/xi)**2+polyb*(dist/xi)**4)
                else:
                    value[0] = 1.0


##############################
# System Params ##############
##############################
print 'Defining material parameters...'
# Material/System Parameters
# Note that we rescale all units of meters by variable scale
R = 0.06    # meters
scale = R/0.5
# Young's modulus and poisson ratio, in J/m^3  and unitless, respectively
E, nu = 700000.0 * scale**3, 0.45
# Lame coefficients, in J/m^3
mu, lamb = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))

# energy release rate = 2 * gamma
# gamma = Constant(0.72*sqrt(2*kappa*Ec))    # J/m^2
# Gc = Constant(2*0.72*sqrt(2*kappa*Ec))     # J/m^2
# Ec = Constant(0.005*mu*kappa)              #J/m^3
# xi = Constant(sqrt(kappa/(2*Ec)))          # meters, fracture process zone scale
xi = 0.0025 / scale                          # meters, fracture process zone scale
Gc = 90 * scale**2                           # J/m^2
gamma = Gc * 0.5
psi = 0.72                                   # unitless
kappa = Gc*xi/(2*psi)                        # J/m       = [Es] L^2 / [phi]
chi = 1.0 / scale**3                         # m^3/(J s) = [phi]^2 / ([E] T)
Ec = Gc**2/(8*psi**2*kappa)                  #J/m^3

# characteristic time for dissipation (all energy dissipated inside process zone)
Tau = Constant(1/(mu*chi))

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

# Timing
t = 0.0
dt = 0.05 * float(Tau)
T = 5 * float(Tau)


############################## 
# Mesh Params ################
##############################
# Create mesh and define function space
mesh = UnitSquareMesh(100, 100)

# Continuous Galerkin elements for both Vf and Vv spaces
Vv = VectorFunctionSpace(mesh, "Lagrange", 1)
Vf = FunctionSpace(mesh, "Lagrange", 1)

# Dirichlet BC
U = 0.012
xc, yc = 0., 0.
# Creating biaxial BCs pulling radially outward
u_0 = Expression(('U*sqrt( pow(x[0]-xc,2)+pow(x[1]-yc, 2) )*cos(atan2(x[1]-yc,x[0]-xc))' ,
                  'U*sqrt( pow(x[0]-xc,2)+pow(x[1]-yc, 2) )*sin(atan2(x[1]-yc,x[0]-xc))'), U=U, xc=xc, yc=yc)


###############################
# SURFACE #####################
###############################
# Define the surface
xc, yc = 0.5, 0.5
alph = 0.706446
x0 = (1. / 2.35)
f = Expression("alph*sigma*exp(-0.5*(pow((x[0] - xo)/sigma, 2)) "
               "      -0.5*(pow((x[1] - yo)/sigma, 2)))",
               alph=alph, xo=xc, yo=yc, sigma=x0)
arrangement = np.array([[xc, yc]])
    
h = interpolate(f, Vf)


###############################
# INITIAL CONDITION ###########
###############################
# Writing initial condition for phi
# H, Y, beta are position in x,y and orientation of initial crack
H = .25
Y = 0.5
beta = 0.75 * pi
gL = 0.1

# Define phase size
W = 0.5 * float(xi)
a = 0.5 * gL

# Projecting phi_k
phik0 = initialPhase()
    
phi_k = interpolate(phik0, Vf)
bcu = DirichletBC(Vv, u_0, u_boundary)


##############################
# Define variational problem #
##############################
print 'Defining test and trial functions...'
v = TestFunction(Vf)
tau = TestFunction(Vv)
phi = TrialFunction(Vf)
u = TrialFunction(Vv)


# functions for variational problem
d = u.geometric_dimension()
epsilon = sym(nabla_grad(u))
epsf = 0.5 * outer(grad(h), grad(h)) #curvature contribution to strain
# PLANE STRESS
sigma = E/(1+nu) * epsilon + E*nu/(1 - nu ** 2) * tr(epsilon) * Identity(d)
sigf = E/(1+nu) * epsf + E*nu/(1 - nu ** 2) * tr(epsf) * Identity(d)
Es = E * nu/(2* (1 - nu**2)) * tr(epsilon + epsf) * tr(epsilon + epsf) + mu * inner(epsilon.T+epsf.T, epsilon+epsf)
g = 4 * phi_k**3 - 3 * phi_k**4
gprime = 12 * (phi_k**2 - phi_k**3)


#################
# ITERATIVE
#################
print('Defining force balance...')
# Force balance (a(u,v) = L(v) = Integrate[f*v, dx] )
# Use symmetry properties of sigma
au = -g*inner(sigma, sym(nabla_grad(tau)))*dx + gprime*inner(sigma, outer(grad(phi_k),tau) )*dx
Lu = g*inner(sigf, sym(nabla_grad(tau)))*dx - gprime*inner(sigf, outer(grad(phi_k),tau) )*dx
u = Function(Vv)
solve(au == Lu, u, bcu) 
        
# dphi/dt
print('Setting up Linear problem for Phi-- Implicit Euler')
damage = 'on'
tensile = 'on'
epsilon_k = sym(nabla_grad(u)) + epsf
# Note on usage: conditional(condition, true_value, false_value)
Es_phi = conditional(lt(tr(epsilon_k),0), 0, E/(4*(1-nu))*tr(epsilon_k)**2 ) +\
            E/(2*(1+nu)) * inner( epsilon_k.T-0.5*Identity(d).T*tr(epsilon_k), epsilon_k -0.5*Identity(d)*tr(epsilon_k))

aphi = phi * v * dx + dt * chi * kappa * dot(nabla_grad(phi), nabla_grad(v)) * dx
fphi = phi_k - dt*chi*(gprime * (Es_phi -Ec))
Lphi = fphi * v * dx
phi = Function(Vf)


##############################
# Run Simulation    ##########
##############################
# Get coordinates in x,y of evaluated points for sigma
n = Vf.dim()
dof_coordinates = Vf.dofmap().tabulate_all_coordinates(mesh)
dof_coordinates.resize((n, d))
x = dof_coordinates[:, 0] - xc
y = dof_coordinates[:, 1] - yc

# Allow mapping from one scalar field to another
assigner = FunctionAssigner(Vf, Vf)
vectassigner = FunctionAssigner(Vv, Vv)
u_k = Function(Vv)

ind = 0
while t <= T:
    print 'ind = ', ind 
    print 'time/tau =', t / float(Tau)

    # Solve force balance
    if t != 0:
        # Linear variational problem for u
        solve(au == Lu, u, bcu)
        # or, for more control
        # problem = VariationalProblem(au,Lu,bc_u)
        # problem.parameters["field"] = Value
        # u = problem.solve()

    # Solve phase evolution
    solve(aphi == Lphi, phi) 

    # Take the minimum of phi so only damaging the material, no healing
    phi_min = pwmin(phi.vector(), phi_k.vector())
    phi_k.vector()[:] = phi_min

    if ind % 5 == 0:
        # Plot solution
        plot(phi, interactive=False)

    t += dt
    ind += 1