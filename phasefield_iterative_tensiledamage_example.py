from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import datetime
import phasefield_elasticity as pe
import glob
import os
import argparse

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
def laplacian(f):
    """Calc the laplacian of a scalar field"""
    return div(grad(f))

def Es_f(u_s):
    """Calc the energy density (no curvature)"""
    eps_s = sym(nabla_grad(u_s))
    return 0.5*lamb*tr(eps_s)*tr(eps_s) + mu*inner(eps_s.T,eps_s)

def phi_boundary(x, on_boundary):
    """define the Dirichlet boundary as pts on boundary"""
    return on_boundary or x[1]< 0.06*(-0.95)  

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

class initialPhase(Expression):
    """Define the initial phase (damage field) for the sheet"""
    def eval(self, value, x):
        H, Y, beta = crack_loc()
        W, a = phase_size()
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
                p , dist = pe.closest_pt_on_lineseg_dolf( [ x[0],x[1] ],endpt1, endpt2)
                if dist <= 4*W :
                    # value[0] = (dist / W)*0.95 +0.05 #linear/abs
                    value[0] = 1-C*exp(-(dist/xi)**2/(2*polysig**2))*(1+polya*(dist/xi)**2+polyb*(dist/xi)**4)
                else:
                    value[0] = 1.0

            
def phase_size():
    """Define the size of the initial slit in terms of its halflength a, and its width W.
    """
    W = 0.5 * float(xi) #1.3 * float(xi)
    a = 0.5 * gL          #6 * float(xi) #200-300
    return W, a

def surf_geom():
    """Define the geometry of the surface for the rigid substrate"""
    if surf =='flat':
        alph = 0.000
        x0 = (1./2.35) * R
        f = Constant('0.0')
        arrangement = np.array([[]])
    elif surf=='bump':        
        xc, yc = 0., 0.
        alph = 0.706446
        x0 = (1./2.35) * R
        f = Expression("alph*sigma*exp(-0.5*(pow((x[0] - xo)/sigma, 2)) "
                   "      -0.5*(pow((x[1] - yo)/sigma, 2)))",
                   alph=alph, xo=xc, yo=yc, sigma=x0)
        arrangement = np.array([[xc,yc]])
    return alph, x0, f, arrangement

def crack_loc():
    """Define the initial position and inclination angle of the crack"""
    H = 1.0/2.35*R
    beta = 0.75*pi
    Y = 0.0
    return H, Y, beta



##############################
# Initialize Simulation ######
##############################
parser = argparse.ArgumentParser(description='Options for running phase-field fracture simulation')
parser.add_argument('-shape', '--shape', help='Shape of the mesh to use', type=str, default='circle')
parser.add_argument('-xi', '--xi', help='Size of process zone', type=float, default=0.0025)
parser.add_argument('-psi', '--psi', help='Dimensionless parameter = integral from 0 to 1 of square root of (1-g(phi))',
                    type=float, default=0.72)
parser.add_argument('-Gc', '--Gc', help='Fracture energy in J/m^2', type=float, default=90)
parser.add_argument('-time', '--final_time_tau',
                    help='Duration of the simulation in units of Tau, the characteristic time for a process zone to become damaged',
                    type=float, default=120)
parser.add_argument('-theta', '--theta',
                    help='Rotation angle of the underlying mesh, relative to the mesh loaded, in units of pi radians',
                    type=float, default=0.0)
parser.add_argument('-Npts', '--Npts',
                    help='Number of points in the mesh to load or create, total',
                    type=int, default=40000)
parser.add_argument('-displacement', '--displacement',
                    help='Displacement at the boundary, expressed as a fractional elongation (dR/ R)',
                    type=float, default=0.012)
parser.add_argument('-gstrain_out', '--gstrain_out',
                        help='Whether to output images of the strain multiplied by g(phi) every Nfull_out timesteps',
                        action='store_true')
parser.add_argument('-displacement_out', '--displacement_out',
                        help='Whether to output images of the displacement every Nfull_out timesteps',
                        action='store_true')
parser.add_argument('-phase_out', '--phase_out',
                        help='Whether to output images of the phase every Nfull_out timesteps',
                        action='store_true')
parser.add_argument('-energy_out', '--energy_out',
                        help='Whether to output images of the energy every Nfull_out timesteps',
                        action='store_true')
parser.add_argument('-Nphase_out', '--Nphase_out',help='How many timesteps to run between saving the solution',
                    type=int, default=150)
parser.add_argument('-Nfull_out', '--Nfull_out',help='How many timesteps to run between saving an image of the solution',
                    type=int, default=600)
args = parser.parse_args()

# timestamps
now = datetime.datetime.now()
date = '%02d%02d%02d' % (now.year,now.month, now.day)
hourmin = '%02d%02d' % (now.hour, now.minute)

##############################
# System Params ##############
############################## 
# Material/System Parameters
R = 0.06    # meters
E , nu = 700000.0, 0.45          #J/m^3, unitless
mu, lamb = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu))) #J/m^3, J/m^3

## energy release rate = 2 * gamma 
#gamma = Constant(0.72*sqrt(2*kappa*Ec))    # J/m^2
#Gc = Constant(2*0.72*sqrt(2*kappa*Ec))     # J/m^2
#Ec = Constant(0.005*mu*kappa)              #J/m^3
#xi = Constant(sqrt(kappa/(2*Ec)))          # meters, fracture process zone scale
xi = args.xi                                # meters, fracture process zone scale
Gc = args.Gc                                # J/m^2
gamma = Gc * 0.5
psi = 0.72                                  # unitless
kappa = Gc*xi/(2*psi)                       # J/m       = [Es] L^2 / [phi]
chi = 1.0                                   # m^3/(J s) = [phi]^2 / ([E] T)
Ec = Gc**2/(8*psi**2*kappa)                 #J/m^3
print 'kappa =', kappa
print '0.005*mu/kappa =', 0.005*float(mu)/kappa
print 'Ec = ', Gc**2/(8*psi**2*kappa)  

#characteristic time for dissipation (all energy dissipated inside process zone)
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
T = args.final_time_tau * float(Tau)


############################## 
# Mesh Params ################
##############################
# Create mesh and define function space
N = args.Npts   # points
eta = 0.00   # randomization (jitter)
theta = args.theta * np.pi  # rotation of underlying lattice (counterclockwise)
shape = args.shape #('square' 'circle')
meshtype = 'Triangular' #('UnitSquare' 'SquareLattRight' 'SquareLattRand' 'Vogel' 'Triangular' 'Trisel')
CGCG = 'CG1CG1' # Continuous Galerkin elements for both Vf and Vv spaces

print 'R/xi', '=', R/float(xi)
Rstr = '{0:.3f}'.format(R).replace('.','p')
etastr = '{0:.3f}'.format(eta).replace('.','p')
L = 2.0 * R
if shape == 'square':
    nx = int(np.sqrt(N))
    meshd = nx/L*float(xi)
    if meshtype == 'UnitSquare':
        print 'Creating unit square mesh of ', meshtype, ' lattice topology...'
        mesh = UnitSquareMesh(nx, nx)
    else:
        print 'Creating square-shaped mesh of ', meshtype, ' lattice topology...'
        meshfile = '../meshes/'+shape+'Mesh_'+meshtype+'_eta'+etastr+'_R'+Rstr+'_N'+str(int(N))+'.xml'
        print 'loading meshfile = ', meshfile
        mesh = Mesh(meshfile)
elif shape == 'circle' :
    print 'Creating circle-shaped mesh of ', meshtype, ' lattice topology...'
    meshd = 2*np.sqrt(N/np.pi)*float(xi)/L
    if abs(theta) > 1e-9:
        add_exten = '_theta'+ '{0:.3f}'.format(theta/np.pi).replace('.','p')+'pi'
    else:
        add_exten = ''

    meshfile = '../meshes/'+shape+'Mesh_'+meshtype+add_exten+'_eta'+etastr+'_R'+Rstr+'_N'+str(int(N))+'.xml'
    print 'loading meshfile = ', meshfile
    mesh = Mesh(meshfile)
    
if CGCG == 'CG1CG1':
    print 'Creating Lagrangian function space...'
    Vv = VectorFunctionSpace(mesh, "Lagrange", 1)
    Vf = FunctionSpace(mesh, "Lagrange", 1)

# Dirichlet BC
U = args.displacement # =w/R
xc, yc = 0., 0.
BCtype = 'biaxial' #('uniaxial' 'biaxial' 'shear' 'mixed_u1s1')

if BCtype=='shear':
    print 'Creating shear BCs...'
    u_0 = Expression(('0.0' ,
                      'U*(x[0]-xc)'), U=U, xc=xc)
elif BCtype=='biaxial':
    print 'Creating biaxial BCs...'
    u_0 = Expression(('U*sqrt( pow(x[0]-xc,2)+pow(x[1]-yc, 2) )*cos(atan2(x[1]-yc,x[0]-xc))' ,
                      'U*sqrt( pow(x[0]-xc,2)+pow(x[1]-yc, 2) )*sin(atan2(x[1]-yc,x[0]-xc))'), U=U, xc=xc, yc=yc)
elif BCtype == 'uniaxial':
    print 'Creating uniaxial BCs...'
    u_0 = Expression(('0.0' , 'U*(x[1]-xc)'), U=U, xc=xc, yc=yc)
elif BCtype == 'uniaxialDisc':
    print 'Creating uniaxial BCs on disc...'
    u_0 = Expression(('U*sqrt( pow(x[0]-xc,2)+pow(x[1]-yc, 2) )*cos(atan2(x[1]-yc,x[0]-xc))' ,
                      '0.0'), U=U, xc=xc, yc=yc)
elif BCtype == 'mixed_u1s1' :
    print 'Creating mixed BCs...'
    u_0 = Expression(('U*(x[1]-xc)' ,
                      'U*(x[1]-yc)'), U=U, xc=xc, yc=yc)
    

###############################
# SURFACE #####################
###############################
# Define the surface
surf = 'bump' # ('flat' 'bump' 'bumps2x1')
alph, x0, f, arngmt = surf_geom()
print 'Defined surface: arngmt=', arngmt
    
h = interpolate(f, Vf)
print 'maximum height is ', max(h.vector().array())


###############################
# INITIAL CONDITION ###########
###############################
print 'Writing initial condition for phi...'
H, Y, beta = crack_loc()
print 'H/R=', H/R, ' Y=', Y, ' beta=', beta/np.pi

gL =

W, a = phase_size()
# Projecting phi_k
if W > DOLFIN_EPS:
    phik0 = initialPhase()
else:
    print 'Setting phi=1 everywhere...'
    phik0 = Constant(1.0)
    
phi_k = interpolate(phik0, Vf)
bcu  = DirichletBC(Vv, u_0, u_boundary)


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
epsf = 0.5 * outer(grad(h),grad(h)) #curvature contribution to strain
# PLANE STRESS
sigma = E/(1+nu) * epsilon + E*nu/(1 - nu ** 2) * tr(epsilon)*Identity(d)
sigf = E/(1+nu) * epsf + E*nu/(1 - nu ** 2) * tr(epsf)*Identity(d)
Es = E * nu/(2* (1 - nu**2)) * tr(epsilon + epsf) * tr(epsilon + epsf) + mu * inner(epsilon.T+epsf.T, epsilon+epsf)
g = 4 * phi_k** 3 - 3 * phi_k** 4
gprime = 12 * (phi_k**2 - phi_k**3)


#################
# ITERATIVE
#################
print('Defining force balance...')
# Force balance (a(u,v) = L(v) = Integrate[f*v, dx] )
au = -g*inner(sigma, sym(nabla_grad(tau)))*dx + gprime*inner(sigma, outer(grad(phi_k),tau) )*dx #using symmetry properties of sigma
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
            E/(2*(1+nu)) * inner( epsilon_k.T-0.5*Identity(d).T*tr(epsilon_k) , epsilon_k -0.5*Identity(d)*tr(epsilon_k) )

aphi = phi*v*dx + dt*chi*kappa* dot(nabla_grad(phi), nabla_grad(v)) *dx
fphi = phi_k  - dt*chi*( gprime *(Es_phi -Ec) )
Lphi = fphi*v*dx
phi = Function(Vf)


###############################
# Output Params ###############
###############################
phase_out = args.phase_out
gstrain_out = args.gstrain_out
displacement_out = args.displacement_out
energy_out = args.energy_out

print 'Ensuring output directories exist...'
print 'beta = ', beta
try:
    outdir = '/Users/npmitchell/Desktop/data_local/'+date+'/'+date+'-'+hourmin+'_paths_'+BCtype+'_'+surf+'_'+\
             meshtype[0:3]+\
            add_exten + \
            '_HoR'+'{0:.3f}'.format(H/R).replace('.','p')+\
            '_beta'+'{0:.2f}'.format(beta/np.pi).replace('.','p')+\
            '_N'+'{}'.format(N)+\
            '_dt'+'{0:.3f}'.format(dt).replace('.','p')+'/'
    pe.ensure_dir(outdir)
except:
    outdir = '/Users/labuser/Desktop/data_local/'+date+'/'+date+'-'+hourmin+'_paths_'+BCtype+'_'+surf+'_'+\
             meshtype[0:3]+\
            '_HoR'+'{0:.3f}'.format(H/R).replace('.','p')+\
            '_beta'+'{0:.2f}'.format(beta/np.pi).replace('.','p')+\
            '_N'+'{}'.format(N)+\
            '_dt'+'{0:.3f}'.format(dt).replace('.','p')+'/'
    pe.ensure_dir(outdir)


# Make dictionary with output options 
OUT = {'phase_out': (phase_out,'phase',r'Phase $\phi$'),
            'gstrain_out': (gstrain_out,'gstrain',r'$g(\phi)\epsilon_{kk}$'),
            'gstress_out': (gstress_out,'gstress',r'Phase x Stress $g(\phi) \sigma_{kk}$'),
            'strain_out': (strain_out,'strain',r'Strain Tensor $\epsilon$'),
            'displacement_out': (displacement_out,'displacement',r'Displacement $u$'),
            'displacement_theory_out': (displacement_theory_out,'displacement_theory',r'Displacement $u$'),
            'energy_out': (energy_out,'energy',r'Strain Energy Density $g(\phi)e_s$'),
            'strainpolar_out': (strainpolar_out,'strainpolar',r'Strain $\epsilon$'),
            'phidiff_out': (phidiff_out,'phidiff',r'Difference in $\phi = \phi_{k+1}-\phi_k$'),
            'phinext_out': (phinext_out,'phinext',r'Next $\phi_{k+2}$')
}

# Ensure output directories
for key in OUT:
    if OUT[key][0]:
        pe.ensure_dir(outdir+OUT[key][1]+'/')

## Output parameters file
paramfile = outdir + 'parameters.txt'
params = {'date': date + '-' + hourmin,
                 'BCtype': BCtype,
                 'U'     : U,
                 'E'     : E ,
                 'nu'    : nu ,
                 'mu'    : float(mu) ,
                 'lamb'  : float(lamb) ,
                 'kappa' : kappa ,
                 'chi'   : chi ,
                 'L'     : L ,
                 'Loxi'  : L/float(xi) ,
                 'meshd' : meshd ,
                 'eta'   : eta, 
                 'Ecomu' : float(Ec)/float(mu) ,
                 'Ec'    : float(Ec) ,
                 'W'     : W/float(xi) ,
                 'a'     : a/float(xi) ,
                 'H'     : H ,
                 'Y'     : Y ,
                 'beta'  : beta ,
                 'dt'    : dt ,
                 'xi'    : float(xi) ,
                 'Tau'   : float(Tau) ,
                 'gamma' : float(gamma) ,
                 'CGCG'  : CGCG ,
                 'mesht' : meshtype ,
                 'shape' : shape,
                 'surf'  : surf ,
                 'alph'  : alph ,
                 'x0'    : x0 ,
                 'arngmt': arngmt ,
                 'theta' : theta,
                 'Gc'    : Gc ,
                 'N'     : N  , 
                 'damage': 'on',
                 'tensile': 'on',
                 'method': 'Linear-Implicit',
                 'stress': 'plane-stress',
                 'fit_mean': fit_mean }
    
pe.write_parameters(paramfile, params)
print params

print('Defining Evolution parameters and output files...')
## Evolution Params
ind = 0
ptsz = 4
# How often to output the solution and an images/plots
Nphase_out = args.Nphase_out
Nfull_out = args.Nfull_out
meshout = File(outdir+"mesh.xml") 
meshout << mesh
    
#save initial state
title, title2, subtext, subsubtext = pe.pf_titles(r'Phase Field $\phi$',params,t)

## Get coordinates in x,y of evaluated points for sigma
n = Vf.dim()
dof_coordinates = Vf.dofmap().tabulate_all_coordinates(mesh)
dof_coordinates.resize((n, d))
x = dof_coordinates[:, 0] - xc
y = dof_coordinates[:, 1] - yc

name = 'phase'
title = pe.title_scalar('Phase Field',params,t)
phiv = phi_k.vector().array()
pe.pf_plot_scatter_scalar(x,y,phiv,outdir+name,name,ind,title,title2,subtext,subsubtext,vmin=0,ptsz=ptsz,shape=shape)
name = 'height' 
title = pe.title_scalar('Surface Height',params,t)
hv = h.vector().array()
pe.pf_plot_scatter_scalar(x,y,hv,outdir,name,ind,title,title2,subtext,subsubtext,vmin=0,ptsz=ptsz,shape=shape)

# Allow mapping from one scalar field to another
assigner = FunctionAssigner(Vf, Vf)
vectassigner = FunctionAssigner(Vv, Vv)
u_k = Function(Vv)
    
while t <=T:
    print 'ind = ', ind 
    print 'time/tau =', t / float(Tau), ' date=', date, '-', hourmin
    
    if t!=0:
        # Linear variational problem for u
        print('Force balance:')
        solve(au == Lu, u, bcu) #bcs=bc_u
        # or, for more control
        #problem = VariationalProblem(au,Lu,bc_u)
        #problem.parameters["field"] = Value
        #u = problem.solve()
    
    print('Phase Evolution:')
    solve(aphi == Lphi, phi) 

    # Take the minimum so only damage the material
    phi_min = pwmin(phi.vector(),phi_k.vector())
    phi_k.vector()[:] = phi_min
    
    #print 'phi_k is: ', phi_k
    t += dt
    ind +=1

    if np.mod(ind, Nphase_out) == 0:
        #time_series.store(phi,u, t)
        phiout = File(outdir+'solution/phi_'+'{0:06d}'.format(ind)+'.xml')
        uout = File(outdir+'solution/u_'+'{0:06d}'.format(ind)+'.xml')
        #tout = File(outdir+'solution/t_'+'{0:06d}'.format(ind)+'.xml')
        phiout << phi_k
        vectassigner.assign(u_k,u)
        uout << u_k
     
        
    if np.mod(ind,Nfull_out) == 0:
        if phase_out:
            name = 'phase'
            title = pe.title_scalar('Phase $\phi$',params,t)
            #phiV = project(phi_k, Vf)
            #phiv = phiV.vector().array()
            phiv = phi_k.vector().array()
            pe.pf_plot_scatter_scalar(x,y,phiv,outdir+name,name,ind,title,title2,subtext,subsubtext,vmin=0.,ptsz=ptsz,shape=shape)
        if gstrain_out:
            name = 'gstrain'
            title = pe.title_scalar(r'$g(\phi)\epsilon_{kk}$',params,t)
            gstrain = project((4*phi**3 - 3*phi**4) * tr(sym(nabla_grad(u))+0.5*outer(grad(h),grad(h))), Vf)
            gstrainv = gstrain.vector().array()
            print 'trying to output gstrain to', outdir+name
            pe.pf_plot_scatter_scalar(x,y,gstrainv,outdir+name,name,ind,title,title2,subtext,subsubtext,ptsz=ptsz,shape=shape)
        if displacement_out:
            name = 'displacement'
            title = pe.title_scalar(r'Displacement $u$',params,t)
            u_V = project(u, Vv)
            ux = project(u_V[0], Vf)
            uxv = ux.vector().array()
            uy = project(u_V[1], Vf)
            uyv = uy.vector().array()
            pe.pf_plot_scatter_2panel(x,y,uxv,uyv,outdir+name,name,'u',ind, title,title2,subtext,subsubtext)
        if energy_out:
            name == 'energy'
            title = pe.title_scalar(r'$g(\phi)(e_s)')
            print('Writing file for '+ name+ '...')
            En_V = project((4*phi**3 - 3*phi**4) * (E*nu/(2*(1-nu**2))*tr(sym(nabla_grad(u))+0.5*outer(grad(h),grad(h)))*tr(sym(nabla_grad(u))+0.5* outer(grad(h),grad(h))) +\
                           mu*inner(sym(nabla_grad(u)).T+0.5* outer(grad(h),grad(h)).T,sym(nabla_grad(u))+0.5* outer(grad(h),grad(h))) ), Vf)
            Env = En_V.vector().array()
            pe.pf_plot_scatter_scalar(x,y,Env,outdir+name,name,ind,title,title2,subtext,subsubtext,vmin=0,vmax=E*.025, ptsz=ptsz, shape=shape)