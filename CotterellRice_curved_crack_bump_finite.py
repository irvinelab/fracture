from pylab import *
import numpy as np
from scipy import integrate
import time
import os
import glob
from savitzky_golay import *
timestr = time.strftime("%Y%m%d-%H%M")

'''This code evolves an azimuthal initial slit on the side of a bump, and can be used as a template for implementing the
Cotterell and Rice procedure for predicting crack paths in sheets on curved substrates as laid out in Mitchell et al
2016.

If you intend to use this code for research or other purposes, please email me at noah.prentice.mitchell@gmail.com.
'''


#Stresses###########
def sigma_theta(x,y): #azimuthal stress
    return alpha**2*1./8. * ( -(x*x+y*y)**-1 * (1 - exp(-x*x-y*y)) + 2 * exp(-x*x-y*y) ) + P

def sigma_r(x,y): #radial stress
    return alpha**2*1./8. * (x*x+y*y)**-1 * (1. - exp(-x*x-y*y)) + P
    
def sigma_theta_f(x,y): 
    return alpha**2/8. * ( (x*x+y*y)**-1 * (exp(-x*x-y*y) - 1.) + 2. * exp(-x*x-y*y) + (1./R)**2 * (exp(-(R)**2) - 1.) ) + P

def sigma_r_f(x,y): 
    return alpha**2/8. * ( (x*x+y*y)**-1 * (1. - exp(-x*x-y*y)) + (1./R)**2 * ( exp(-(R)**2) - 1. ) ) + P


    
#Angles###########
def theta(x,y): #azimuth
    return math.atan2(y,x)

def kink_angle(K1,K2):
    eta = K2/K1
    return 2.*math.atan(-2. * eta / (1. + sqrt(1. + 8. * eta**2) ) )
    
def sigma_x(x, y, theta):
    # fixed typo 2016-03-18
    return sigma_r(x,y) * (cos(theta))**2 + sigma_theta(x,y) * (sin(theta))**2

def sigma_y(x, y, theta):
    return sigma_r(x,y) * (sin(theta))**2 + sigma_theta(x,y) * (cos(theta))**2

def sigma_xy(x,y , theta):
    return ( sigma_r(x,y) - sigma_theta(x,y) ) * sin(theta) * cos(theta)

#########
#finite
def sigma_x_f(x, y, theta):
    #return sigma_r_f(t,y) * (cos(theta))**2 + sigma_theta_f(x,y) * (sin(theta))**2
    return sigma_r_f(x,y) * (cos(theta))**2 + sigma_theta_f(x,y) * (sin(theta))**2

def sigma_y_f(x, y, theta):
    return sigma_r_f(x,y) * (sin(theta))**2 + sigma_theta_f(x,y) * (cos(theta))**2

def sigma_xy_f(x,y , theta):
    return ( sigma_r_f(x,y) - sigma_theta_f(x,y) ) * sin(theta) * cos(theta)
#########

def Tn(x,y, gamma):
    return sigma_r(x,y) * (sin(gamma))**2 + sigma_theta(x,y) * (cos(gamma))**2

def Ts(x,y, gamma):
    return ( sigma_r(x,y) - sigma_theta(x,y) ) * sin(gamma) * cos(gamma)

def Ts_prime(x,y,gamma,xp,yp,gammap): #xp, yp, gammap are the previous values needed to calculate derivative
    return ( Ts(x,y,gamma) - Ts(xp,yp,gammap) ) / (x - xp)

def Tn_prime(x,y,gamma,xp,yp,gammap): #xp, yp, gammap are the previous values needed to calculate derivative
    return ( Tn(x,y,gamma) - Tn(xp,yp,gammap) ) / (x - xp)

#########
#finite
def Tn_f(x,y, gamma):
    return sigma_r_f(x,y)*(sin(gamma))**2 + sigma_theta_f(x,y)*(cos(gamma))**2

def Ts_f(x,y, gamma):
    return ( sigma_r_f(x,y) - sigma_theta_f(x,y) ) * sin(gamma)*cos(gamma)

def Ts_prime_f(x,y,gamma,xp,yp,gammap): #xp, yp, gammap are the previous values needed to calculate derivative
    return ( Ts_f(x,y,gamma) - Ts_f(xp,yp,gammap) ) / (x - xp)

def Tn_prime_f(x,y,gamma,xp,yp,gammap): #xp, yp, gammap are the previous values needed to calculate derivative
    return ( Tn_f(x,y,gamma) - Tn_f(xp,yp,gammap) ) / (x - xp)
#########

#lam = lambda in Cotterell-Rice paper.
#lam_prime is essentially omega.
#omega0 is the value of omega at the tip (called omega in cotterell and rice paper)

def lam(y,Htip):
    return y - Htip

def qI(x,y, gamma, omega0,xp,yp,gammap,lam_prime, Htip):
    return Tn(x,y, gamma) - 1.5 * omega0 * Ts(x,y, gamma) + lam(y,Htip) * Ts_prime(x,y,gamma,xp,yp,gammap) + 2. * Ts(x,y, gamma) * lam_prime

def qII(x,y, gamma, omega0, xp,yp,gammap, Htip):
    return Ts(x,y, gamma) + 0.5 * omega0 * Tn(x,y, gamma) + lam(y,Htip) * Tn_prime(x,y,gamma,xp,yp,gammap) #need to add more terms here

#########
#finite
def qI_f(x,y, gamma, omega0,xp,yp,gammap,lam_prime, Htip):
    return Tn_f(x,y, gamma) - 1.5 * omega0 * Ts_f(x,y, gamma) + lam(y,Htip) * Ts_prime_f(x,y,gamma,xp,yp,gammap) + 2. * Ts_f(x,y, gamma) * lam_prime

def qII_f(x,y, gamma, omega0, xp,yp,gammap, Htip):
    return Ts_f(x,y, gamma) + 0.5 * omega0 * Tn_f(x,y, gamma) + lam(y,Htip) * Tn_prime_f(x,y,gamma,xp,yp,gammap) #need to add more terms here
#########

def P_displ(nu,U,alpha,x0,R):
    return (1./(1-nu)) * ( U - 1./4.*alpha**2*( (x0/R)**2 * (e**(-R**2/x0**2)-1) + e**(-R**2/x0**2) ))

def P_displ_infinite(nu,U,alpha,x0):
    return (1./(1-nu)) * ( U )

########
def getRoundedThresholdv(a, MinClip):
    return np.around(np.array(a, dtype=float) / MinClip) * MinClip

def XYdat_uniqueXvals(names, values):
    result_names = np.unique(names)
    result_values = np.empty(result_names.shape)

    for i, name in enumerate(result_names):
        result_values[i] = np.mean(values[names == name])

    return result_names, result_values     



#Choose the type of analysis to perform
finitesample = 1

#Make variable naming for finite/infinite
if finitesample==1:
    fininf = 'finite'
else:
    fininf = 'infinite'

#Make path for output files
fname = './'+timestr+'_'+fininf
if not os.path.exists(fname):
    os.makedirs(fname)
outpath=fname



#pick a few numbers for H, the height / y-location of the initial straight crack
Hstart = 0.0
Hmax = 1.8
H0 = arange(Hstart,Hmax,0.2)

alpha = 0.7064460135092848 # aspect ratio
nu=0.5
U=0.012
R = 2.35
#boundary stress
if finitesample==1:
    P = P_displ(nu,U,alpha,1,R)
else:
    P = P_displ_infinite(nu,U,alpha,1)
    	   

# pick a fine set of values for d
delta_d = 0.005
dmax = 1.0
d = arange(0.0,dmax,delta_d)
# create an empty array/matrix of the same length as a and w
H = zeros([len(d)+1,len(H0)])
omega = zeros([len(d)+1,len(H0)])
K1 = zeros([len(d),len(H0)])
K2 = zeros([len(d),len(H0)])
K1eff = zeros([len(d),len(H0)])
Kcheck = zeros([len(d),len(H0)])
K1_error = zeros([len(d),len(H0)])
K2_error = zeros([len(d),len(H0)])
angle = zeros([len(d),len(H0)])
angle_unphysical = zeros([len(d),len(H0)])

#initial slit lengths <-- get from expt, theory, or constant
a_init=0.2 # initial lenght of straight crack is 2 a_init



#first break the initial straight crack up into segments, such that there is only one type of integration.
if useEXPTainit==1 or useTHEORYanit==1:
    nr_of_segments_list = array([int(a_initlist[ii] / delta_d) for ii in range(len(a_initlist))])
    for j in range(0, len(H0)):
        for aaa in range(0, nr_of_segments_list[j]+1):
            omega[aaa] = 0.0
            H[aaa][j] = H0[j]
else:
    nr_of_segments = int(a_init / delta_d) 
    for j in range(0, len(H0)):
        for aaa in range(0, nr_of_segments+1):
            omega[aaa] = 0.0
            H[aaa][j] = H0[j]
    nr_of_segments_list = ones_like(H0)*nr_of_segments
    
#Save the nr_of_segments list
fname = outpath+'/nr_of_zero_segments_list.txt'
saveM = array([[ H0[i], nr_of_segments_list[i] ] for i in arange(len(nr_of_segments_list))])
np.savetxt(fname, saveM, header='H0 nr_of_zero_segments', comments='')     
    

for j in range(0, len(H0)):
    print H0[j]
    #omega[0][j] = 0.0 #corresponds to the straight semi-infinite part
    #H[0][j] = H0[j] #corresponds to the straight semi-infinite part
    
    #grab the index at which we should start propagating for this value of H0
    #--> this index will be nr_of_segments_list[j]
    
    for i in range(nr_of_segments_list[j],len(d)):
        a=d[i] #crack half-length is given by d. 
        #now calculate K1
        integral_B = zeros(1)
        integral_C = zeros(1)
        for k in range(0, i): #splitting integral up in segments
            if finitesample==1:
                integrand_B = lambda t: qI_f(t, H[k+1][j], math.atan2(H[k+1][j],t) - omega[k+1][j], omega[i][j], t-delta_d, H[k][j],math.atan2(H[k][j],t) - omega[k][j],omega[k+1][j],H[i][j]) * sqrt(1./(pi*a)) * ( sqrt(a + t) / sqrt(a - t) )
            else:
                integrand_B = lambda t: qI(t, H[k+1][j], math.atan2(H[k+1][j],t) - omega[k+1][j], omega[i][j], t-delta_d, H[k][j],math.atan2(H[k][j],t) - omega[k][j],omega[k+1][j],H[i][j]) * sqrt(1./(pi*a)) * ( sqrt(a + t) / sqrt(a - t) )
            
            #gamma = theta - omega
            #omega[i][j] = omega_not i.e. omega at the crack tip. omega[k+1][j] is omega along the crack egde.
            #y=H[k+1][j],   yp=H[k][j] 
            integral_B = integral_B + integrate.quad(integrand_B,d[k],d[k+1],limit=100)[0]
            
            if finitesample==1:
                integrand_C = lambda t: qI_f(t, H[k+1][j], math.atan2(H[k+1][j],t) - omega[k+1][j], omega[i][j], t-delta_d, H[k][j],math.atan2(H[k][j],t) - omega[k][j],omega[k+1][j],H[i][j]) * sqrt(1./(pi*a)) * ( sqrt(a - t) / sqrt(a + t) )
            else: 
                integrand_C = lambda t: qI(t, H[k+1][j], math.atan2(H[k+1][j],t) - omega[k+1][j], omega[i][j], t-delta_d, H[k][j],math.atan2(H[k][j],t) - omega[k][j],omega[k+1][j],H[i][j]) * sqrt(1./(pi*a)) * ( sqrt(a - t) / sqrt(a + t) )
            integral_C = integral_C + integrate.quad(integrand_C,d[k],d[k+1],limit=100)[0]
        temp = integral_B + integral_C
        K1[i][j] = temp 
        
        
        #now calculate K2    
        integral2_B = zeros(1)
        integral2_C = zeros(1)
        for k in range(0, i):
            #print "yes we are in loop k"
            if finitesample==1:
                integrand2_B = lambda t: qII_f(t, H[k+1][j], math.atan2(H[k+1][j],t) - omega[k+1][j], omega[i][j],  t-delta_d, H[k][j], math.atan2(H[k][j],t) - omega[k][j], H[i][j])  * sqrt(1./(pi*a)) * ( sqrt(a + t) / sqrt(a - t) ) #gamma = theta - omega
            else:
                integrand2_B = lambda t: qII(t, H[k+1][j], math.atan2(H[k+1][j],t) - omega[k+1][j], omega[i][j],  t-delta_d, H[k][j], math.atan2(H[k][j],t) - omega[k][j], H[i][j])  * sqrt(1./(pi*a)) * ( sqrt(a + t) / sqrt(a - t) ) #gamma = theta - omega
                
            integral2_B = integral2_B + integrate.quad(integrand2_B,d[k],d[k+1],limit=100)[0]
            
            if finitesample==1:
                integrand2_C = lambda t: qII_f(t, H[k+1][j], math.atan2(H[k+1][j],t) - omega[k+1][j], omega[i][j],  t-delta_d, H[k][j], math.atan2(H[k][j],t) - omega[k][j], H[i][j])  * sqrt(1./(pi*a)) * ( sqrt(a - t) / sqrt(a + t) ) #gamma = theta - omega
            else:
                integrand2_C = lambda t: qII(t, H[k+1][j], math.atan2(H[k+1][j],t) - omega[k+1][j], omega[i][j],  t-delta_d, H[k][j], math.atan2(H[k][j],t) - omega[k][j], H[i][j])  * sqrt(1./(pi*a)) * ( sqrt(a - t) / sqrt(a + t) ) #gamma = theta - omega
                
            integral2_C = integral2_C + integrate.quad(integrand2_C,d[k],d[k+1],limit=100)[0]
        temp =  integral2_B + integral2_C
        K2[i][j] = temp
        #print "The value of K2 is now", temp
        

        #check for straight crack
        #integrandcheck = lambda t: sigma_xy(t, H0[j], math.atan2(H0[j],t)) * sqrt(2/pi) / sqrt(-t+d[i])
        #temp = integrate.quad(integrandcheck,-np.inf,d[i],limit=100)
        #Kcheck[i][j] = temp[0]
        #print "The value of KC is now", temp[0]

        #calculate new 
        angle_unphysical[i][j] = kink_angle(K1[i][j],K2[i][j])
        if K1[i][j] > 0.0:   
            omega[i+1][j] = omega[i][j] + angle_unphysical[i][j] #i+1 because it concerns the omega of the next 'step' 
        elif K1[i][j] == 0.0:
            omega[i+1][j] = omega[i][j] + 2. * math.atan(-copysign(1,K2[i][j])/sqrt(2.))
        else:
            omega[i+1][j] = omega[i][j] + 0.0 #negative K1 is unphysical, should be a break to quit the loop
        #omega[i+1][j] = 0.0    #this line is just a check to see if K reduces to the one for straight crack if I include perturbation
        H[i+1][j] = H[i][j] + omega[i+1][j] * delta_d

        #Calculate K1eff using expression on pg 75 of JG Williams' 'Fracture Mech. of Polymers'
        effk1term = K1[i][j]*(1./2.)*cos(omega[i+1][j]/2.)*(1+cos(omega[i+1][j])) 
        effk2term = K2[i][j]*(3./2.)*sin(omega[i+1][j]/2.)*(1+cos(omega[i+1][j]))
        K1eff[i][j] = effk1term - effk2term
        
        
         
###################################### PLOTTING ################################            
if useEXPTainit==1:
    ainitstr= '_EXPTainit'
elif useTHEORYainit==1:            
    ainitstr= '_THEORYainit'
else:
    ainitstr= '_ainit'+str(a_init)
    
# plot K1 versus d            
figure(1)
for j  in range(0, len(H0)):
    plot(d, K1.transpose()[j][:],label='H='+str(H0[j]))
xlabel(r'$\frac{d}{x_0}$')
ylabel(r'$\frac{K_I}{Y \alpha^2 \sqrt{x_0}}$')
title('Stress intensity factor K1 for a semi-infinite crack in '+fininf+' bump'+r'$\nu=$'+str(nu) )
tight_layout()

fname = outpath+'/'+timestr+'_K1_curved_semi_inf_crack_'+fininf+'_gaussian_bump_alpha' + str(alpha) + ainitstr + '_P'+str(P)+ '_nu'+str(nu)+'.pdf'
legend(loc='best')
savefig(fname)


figure(3)
for j  in range(0, len(H0)):
    plot(d, K2.transpose()[j][:],label='H='+str(H0[j]))
xlabel(r'$\frac{d}{x_0}$')
ylabel(r'$\frac{K_{II}}{Y \alpha^2 \sqrt{x_0}}$')
title('Stress intensity factor K2 for a semi-infinite crack in '+fininf+' bump'+r'$\nu=$'+str(nu) )
tight_layout()
fname = outpath+'/'+timestr+'_K2_curved_semi_inf_crack_'+fininf+'_gaussian_bump_alpha' + str(alpha) + ainitstr + '_P'+str(P)+ '_nu'+str(nu)+'.pdf'
legend(loc='best')
savefig(fname)

#check straight crack
#figure(4)
#for j  in range(0, len(H0)):
#    plot(d, Kcheck.transpose()[j][:],label='H='+str(H0[j]))
#xlabel(r'$\frac{d}{x_0}$')
#ylabel(r'$\frac{K_{II}}{Y \alpha^2 \sqrt{x_0}}$')
#title('Stress intensity factor K2-check for a semi-infinite crack with its tip a distance d from center of bump' )
#fname = 'Kcheck_curved_semi_inf_crack_gaussian_bump_' + timestr + '.pdf'
#legend(loc='best')
savefig(fname)


figure(6)
for j  in range(0, len(H0)):
    plot(d, angle_unphysical.transpose()[j][:],label='H='+str(H0[j]))
xlabel(r'$\frac{d}{x_0}$')
ylabel(r'$\theta$')
title('Kink angle (incl. unphysical) for a semi-infinite crack in '+fininf+' bump'+r'$\nu=$'+str(nu), size = 10)
tight_layout()
fname = outpath+'/'+timestr+'kink_angle_unphys_curved_semi_inf_crack_'+fininf+'_gaussian_bump_alpha'+\
    str(alpha)+ainitstr+'_P'+str(P)+'_nu'+str(nu)+'.pdf'
legend(loc='best')
savefig(fname)


#add element to d
d = append(d, d[len(d)-1]+delta_d)  ## WARNING: now d is 1 row longer!


figure(5)
for j  in range(0, len(H0)):
    plot(d, omega.transpose()[j][:],label='H='+str(H0[j]))
xlabel(r'$\frac{d}{x_0}$')
ylabel(r'$\omega$')
title('Omega for a semi-infinite crack in '+fininf+' bump'+r'$\nu=$'+str(nu), size = 10)
    
tight_layout()
fname = outpath+'/'+timestr+'_Omega_curved_semi_inf_crack_'+fininf+'_gaussian_bump_alpha' + \
    str(alpha) + ainitstr + '_P'+str(P)+'.pdf'
legend(loc='best')
savefig(fname)

figure(7)
for j  in range(0, len(H0)):
    plot(d, H.transpose()[j][:],label='H='+str(H0[j]))
xlabel(r'$\frac{d}{x_0}$')
ylabel(r'$H$')
title('H for a semi-inf crack in '+fininf+' bump'+r'$\nu=$'+str(nu), size = 10)
tight_layout()
fname = outpath+'/'+timestr+'_H_curved_semi_inf_crack_'+fininf+'_gaussian_bump_alpha' + \
    str(alpha) + ainitstr +'_P'+str(P)+'.pdf'
#legend(loc='best')
savefig(fname)

#####SAVING#############
for j  in range(0, len(H0)):
    fname = outpath+'/'+timestr+'_d_K1_K2_curved_semi_inf_crack_'+fininf+'_H'+\
        str(H0[j]) +'_gaussian_bump_alpha'+str(alpha)+ainitstr+ '_P'+str(P)+'_nu'+str(nu)+'.dat'

    fname2 = outpath+'/'+timestr+'_d_K1_K2_curved_semi_inf_crack_'+fininf+'_H'+\
        str(H0[j]) +'_gaussian_bump_alpha'+str(alpha)+ainitstr+ '_P'+str(P)+'_nu'+str(nu)+'.txt'

    savetxt(fname, (d[0:len(d)-1], K1.transpose()[j][:], K2.transpose()[j][:], omega.transpose()[j][0:len(d)-1], angle_unphysical.transpose()[j][0:len(d)-1],H.transpose()[j][0:len(d)-1]))
    #vectors must have the same length, so throw away final datapoint for omega, kink-angle and H
        
    saveM = array([[ d[i], K1.transpose()[j][i], K2.transpose()[j][i], omega.transpose()[j][i], \
        angle_unphysical.transpose()[j][i],H.transpose()[j][i] ] for i in arange(len(d)-1)])
    np.savetxt(fname2, saveM, header='d K1 K2 omega angle_unphysical H')     
    
    #execfile('K1_curved_crack_bump_6may13_3nov14.py')






##### Examine K1eff vs crack length #############

#The following code is for finding K1eff after finding K1,K2, etc sucessfully:
#K1eff = zeros([len(d)-1,len(H0)])
#for j in range(0, len(H0)):
#    print H0[j]
#    #notice that we have added an element to d in the plotting section, so we iterate only up to len(d)-1
#    for i in range(nr_of_segments,len(d)-1): 
#        #Calculate K1eff using expression on pg 75 of JG Williams' 'Fracture Mech. of Polymers'
#        effk1term = K1[i][j] *(1./2.)*cos(omega[i+1][j]/2.)*(1+cos(omega[i+1][j])) 
#        effk2term = K2[i][j]*(3./2.)*sin(omega[i+1][j]/2.)*(1+cos(omega[i+1][j]))
#        K1eff[i][j] = effk1term - effk2term

# Plot K1eff without rescaling
close('all')
clf()
fig = gcf()
for j in range(0, len(H0)):
    print(j)
    plot(d[:-1], K1eff.transpose()[j][:],color=cm.spectral(float(H0[j]/max(H0))), label='H='+str(H0[j]))

legend(loc='best')
ylabel(r'$\frac{K_{I}^{\mathrm{eff}}}{Y}$', rotation=0, size=22)
xlabel(r'$x$', size=22)
fname = outpath+'/'+timestr+'_K1eff_curved_semi_inf_crack_'+fininf+'_gaussian_bump_alpha' + str(alpha) + ainitstr +'_P'+str(P)+'_nu'+str(nu)+'.pdf'
tight_layout()
fig.savefig(fname, transparent=True, pad_inches=0) #bbox_inches='tight', 

# Plot K1eff with rescaling
clf(); fig = gcf()
for j in range(0, len(H0)):
    print(j)
    plot(d[:-1], K1eff.transpose()[j][:]/sqrt(0.0254),color=cm.spectral(float(H0[j]/max(H0))), label='H='+str(H0[j]))

legend(loc='best')
ylabel(r'$\frac{K_{I}^{\mathrm{eff}}}{Y \sqrt{x_0}}$', rotation=0, size=22)
xlabel(r'$x$', size=22);
ax = gca(); ax.yaxis.set_label_coords(-0.11, 0.5)
subplots_adjust(left=0.16, right=0.90, bottom=0.13, top=0.90)
fname = outpath+'/'+timestr+'_K1effscaled_curved_semi_inf_crack_'+fininf+'_gaussian_bump_alpha' + str(alpha) + ainitstr +'_P'+str(P)+'_nu'+str(nu)+'.pdf'
tight_layout()
fig.savefig(fname, transparent=True,  pad_inches=0)









