import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.linalg import logm 
from scipy.linalg import eig
import cmath
import os

##anticonmutador util
def anticonmutador(A,B):
    return np.matmul(A,B) + np.matmul(B,A)

#distribución de Fermi
def fermi(E,mu,beta):
    return 1/(np.exp((E-mu)*beta) + 1)

#Matrices de Pauli y identidad para Jordan Wigner
sigmax = np.array([[0,1],
                   [1,0]])

sigmay = np.array([[0,-1j],
                   [1j,0]])

iden = np.array([[1,0],
                 [0,1]])

sigmaz = np.array([[1,0],
                   [0,-1]])

sigmaup = (sigmax + 1j*sigmay)/2
sigmadown = (sigmax - 1j*sigmay)/2

#Aqui se empieza hacer Jordan Wigner
#Uso la definición que esta en el libro de Schaller, (Open quantum systems Far From equilibrium)
auxdag = np.kron(sigmaz,sigmaup)
aux = np.kron(sigmaz,sigmadown)

auxd = np.kron(sigmaz,sigmaz)
#Jordan-Wigner
dldag = np.kron(sigmaup,np.eye(4))
dl = np.kron(sigmadown,np.eye(4))

drdag = np.kron(auxdag,np.eye(2))
dr = np.kron(aux,np.eye(2))

dddag = np.kron(auxd,sigmaup)
dd = np.kron(auxd,sigmadown)

#operadores de numero 
nd = np.matmul(dddag,dd)
nl = np.matmul(dldag,dl)
nr = np.matmul(drdag,dr)


#here the superoperator that defines the evolution
def Liouvillian( H,Ls, hbar = 1):
    d = len(H)
    superH = -1j/hbar * (np.kron(np.eye(d), H ) - np.kron(H.T,  np.eye(d))   )
    superL = sum( [np.kron(L.conjugate(),L) - 1/2 * (np.kron( np.eye(d), L.conjugate().T.dot(L)) +
                                                     np.kron( L.T.dot(L.conjugate()),np.eye(d) ))
                                                      for L in Ls ] )        
    return superH + superL

#operadores del disipador dn
def Dn(E,U,mun,betan,gamman,dn,dndag,ni,nj):
    d = len(ni)
    auxd1 = np.sqrt( fermi(E,mun,betan)*gamman )*np.matmul( np.matmul((np.eye(d)-ni),(np.eye(d)-nj)),dndag )
    auxd2 = np.sqrt( (1-fermi(E,mun,betan))*gamman )*np.matmul( np.matmul((np.eye(d)-ni),(np.eye(d)-nj)),dn)
    auxd3 = np.sqrt( fermi(E+U,mun,betan)*gamman )*np.matmul( np.matmul((np.eye(d)-ni) ,nj) + np.matmul((np.eye(d)-nj) ,ni) ,dndag )
    auxd4 = np.sqrt( (1-fermi(E+U,mun,betan))*gamman )*np.matmul(np.matmul((np.eye(d)-ni) ,nj) + np.matmul((np.eye(d)-nj) ,ni),dn)
    auxd5 = np.sqrt( fermi(E+ (2*U),mun,betan)*gamman )*np.matmul( np.matmul(ni ,nj),dndag )
    auxd6 = np.sqrt( (1-fermi(E+(2*U),mun,betan))*gamman )*np.matmul(np.matmul(ni ,nj),dn)

    return [auxd1,auxd2,auxd3,auxd4,auxd5,auxd6]

def Dissipator(lambdan,U,mul,mur,mud,betal,betar,betad,gammal,gammar,gammad,nl,nr,nd):
    DR = Dn(lambdan,U,mur,betar,gammar,dr,drdag,nl,nd)
    DL = Dn(lambdan,U,mul,betal,gammal,dl,dldag,nr,nd)
    DD = Dn(lambdan,U,mud,betad,gammad,dd,dddag,nl,nr)

    tot = []
    for l in DL:
        tot.append(l)
    for r in DR:
        tot.append(r)
    for d in DD:
        tot.append(d)    

    return tot

#Funcion auxiliar para calcular los D_{i}\rho

def Dissipate(H,Ls, rho):
    d = len(H)
    superL = np.zeros((d,d) , dtype = np.complex_)
    for L in Ls:
        d = np.matmul( np.matmul( L,rho ),L.transpose().conjugate() )
        e = (1/2)*anticonmutador( np.matmul(L.transpose().conjugate(),L), rho )
        superL += d-e
        
    return superL

def Propagate(rho0,superop,t):
    d = len(rho0)
    propagator = expm (superop *t)
    vec_rho_t = propagator @ np.reshape(rho0,(d**2,1))
    return np.reshape( vec_rho_t, (d,d) )

####Hamiltoniano especifico
def Hamiltonian(El,Er,Ed,U,glr,gld,grd):
    sites = El*nl + Er*nr + Ed*nd
    couplinglr = glr*( np.matmul(dldag,dr) + np.matmul(drdag,dl) )
    couplingld = gld*(np.matmul(dldag,dd) + np.matmul(dddag,dl) )
    couplingrd = grd*(np.matmul(drdag,dd) + np.matmul(dddag,dr) )
    couplingtotal = couplingld+couplinglr+couplingrd
    coulomb = U* (np.matmul(nl,nd) +  np.matmul(nr,nd) + np.matmul(nl,nr)) 
    return sites+couplingtotal+coulomb

def Htd(El,Er,Ed,U):
    Htdl = El*nl + Er*nr + Ed*nd + U*(np.matmul(nr,nd) + np.matmul(nl,nd) + np.matmul(nr,nl)) 
    return Htdl

def coupling(El,Er,Ed,glr,gld,grd):
    mat = np.array([
    [El, glr, gld],
    [glr, Er, grd],
    [gld, grd, Ed]] )
    return mat 

###corrientes de calor, produccion de entropia,flujo de energia,etc..
def currents(Htd,mul,mur,mud,Ll,Lr,Ld,superop,rho0,t):
    Nop = nl + nr + nd
    rhof = Propagate(rho0,superop,t)    
    PD = rhof[0,0].real + rhof[1,1].real 
    P0 = rhof[7,7].real + rhof[6,6].real 

    Dl = Dissipate(Htd,Ll,rhof)
    Dr = Dissipate(Htd,Lr,rhof)
    Dd = Dissipate(Htd,Ld,rhof)

    Qopl = (Htd - mul*Nop)
    Qopr = (Htd - mur*Nop)
    Qopd = (Htd - mud*Nop)

    aux = logm(rhof)
    Ql = np.trace( np.matmul( Dl,Qopl  ) )
    Qr = np.trace( np.matmul( Dr,Qopr  ) )
    Qd = np.trace( np.matmul( Dd,Qopd  ) )

    Sl = -np.trace( np.matmul(Dl,aux) )
    Sr = -np.trace( np.matmul(Dr,aux) )
    Sd = -np.trace( np.matmul(Dd,aux) )

    El = np.trace( np.matmul( Dl,Htd  ) )
    Er = np.trace( np.matmul( Dr,Htd  ) )
    Ed = np.trace( np.matmul(Dd,Htd))

    Wl = mul*np.trace(np.matmul( Dl, Nop ))
    Wr = mur*np.trace(np.matmul( Dr, Nop ))
    Wd = mud*np.trace(np.matmul( Dd, Nop ))

    Nl = np.trace( np.matmul(Dl,Nop ) )
    Nr = np.trace(np.matmul(Dr,Nop ))
    Nd = np.trace(np.matmul( Dd,Nop ))
    return Ql.real, Qr.real, Qd.real, Sl.real, Sr.real,Sd.real, El.real, Er.real, Ed.real, Wl.real,Wr.real,Wd.real,Nl.real,Nr.real,Nd.real
eps = 1E-5
El = 1
Er = 1
Ed = 1
U0 = 40
glr = 5/1000
gld = (5/1000)-eps
grd = (5/1000)+eps
eV = 6.5
mul1 = eV/2
mur1 = -eV/2
mud1 = 2*eV

betar,betad,betal = 1/100,1/100,1/100

gr = (1/100)
gl = 1/100
gd = 1/100

Hcoup = coupling(El,Er,Ed,glr,gld,grd)

eigencoup,eigenvecoup = eig(Hcoup)
print(eigencoup)
E0 = sum(eigencoup)/len(eigencoup)

Ll = Dn(E0,U0,mul1,betal,gl,dl,dldag,nr,nd)
Lr = Dn(E0,U0,mur1,betar,gr,dr,drdag,nl,nd)
Ld = Dn(E0,U0,mud1,betad,gd,dd,dddag,nl,nr)



Ls = Dissipator(E0,U0,mul1,mur1,mud1,betal,betar,betad,gl,gr,gd,nl,nr,nd)
H = Hamiltonian(El,Er,Ed,U0,glr,gld,grd)

superop = Liouvillian(H,Ls)

#revisar la base
#|1,1,1>,|1,1,0>,|1,0,1>,|1,0,0>,|0,1,1>,|0,1,0>,|0,0,1>,|0,0,0>
alp,alp2,alp3,alp4,alp5 = 0.,0.,0.0,0.0,0.
a,b,c,d = 1j*alp,1j*alp2,1j*alp3,1j*alp4

rho0 = np.array([[1/8,0,0,0,0,0,0,0],
                 [0,1/8,a,0,d,0,0,0],
                 [0,-a,1/8,0,0,0,0,0],
                 [0,0,0,1/8,0,0,b,0],
                 [0,-d,0,0,1/8,0,0,0],
                 [0,0,0,0,0,1/8,c,0],
                 [0,0,0,-b,0,-c,1/8,0],
                 [0,0,0,0,0,0,0,1/8]])

rho1 = np.array([[0,0,0,0,0,0,0,0],
                 [0,1/4,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0],
                 [0,0,0,1/4,0,0,0,0],
                 [0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,1/4,0,0],
                 [0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,1/4]])



times = np.linspace(0,1600,1000)
Probnt1 = []
Probnt2 = []
Probnt3 = []
Probnt4 = []
Probnt5 = []
Probnt6 = []
Probnt7 = []
Probnt8 = []
traza = []
cohe = []

for ts in times:
    cal1 = Propagate(rho0,superop,ts)
    auxp = np.matmul(dldag,dr)
    alp = np.trace( np.matmul(auxp,cal1) )
    tot = np.trace(cal1)
    
    traza.append(tot)
    Probnt1.append(cal1[0,0].real )
    Probnt2.append(cal1[1,1].real )
    Probnt3.append(cal1[2,2].real )
    Probnt4.append(cal1[3,3].real )
    Probnt5.append(cal1[4,4].real )
    Probnt6.append(cal1[5,5].real )
    Probnt7.append(cal1[6,6].real ) 
    Probnt8.append(cal1[7,7].real ) 
    cohe.append(abs(cal1[5,3])+ abs(cal1[3,6])+abs(cal1[5,6]) +abs(cal1[1,2])+ abs(cal1[1,4]) + abs(cal1[4,2]) )
 

plt.plot(times,Probnt1,label = "3 ocupados")
plt.plot(times,Probnt2, label = "ocupado r y l")
plt.plot(times,Probnt3, label = "ocupado l y d")
plt.plot(times,Probnt4, label = "ocupado l")
plt.scatter(times,Probnt5, label = "ocupado r y d")
plt.plot(times,Probnt6, label = "ocupado r")
plt.plot(times,Probnt7, label = "ocupado d")
plt.plot(times,Probnt8, label = "vacio total")
plt.legend()
plt.show()


plt.plot(times,cohe)
plt.xlabel(r'$t$', fontsize = 20)
plt.ylabel(r'$\mathcal{C}_{l_{1}}$', fontsize = 20)
plt.show()



values =  Propagate(rho0,superop,3000)

plt.imshow(values.imag)
plt.colorbar()
plt.show()
    
Ufs = np.linspace(7,40,100)
Us = np.linspace(1,6,50)
Num = 200
eVs0 = np.linspace(0,800,Num)
Ql = []
Qr = []
Qd = []
Sls = []
Srs = []
Sds = []
Slr = []
entropf = []
Isl = []
Id = []
Els = []
Ers = []
Eds = []
Erl = []
Qlr = []
Wl = []
Wr = []
Wd = []
Wt = []
Flr = []
Tisl = []
cohes = []
Wdf = []
Qdf = []
Tid = []
Fd = []
Ilf = []
Irf = []
Nls = []
Nrs = []
Nds = []
cohev = []
concv = []
eVs = []
auxff = []
Hcoup0 = coupling(El,Er,Ed,glr,gld,grd)

eigencoup0,eigenvecoup0 = eig(Hcoup0)
E00 = sum(eigencoup0)/len(eigencoup0)
for ev in eVs0:
    mud0 = 2
    U00 = 40 #10
    Ls0 = Dissipator(E00,U00,ev/2,-ev/2,mud0,betal,betar,betad,gl,gr,gd,nl,nr,nd)
    H0 = Hamiltonian(El,Er,Ed,U00,glr,gld,grd)
    superop0 = Liouvillian(H0,Ls0)
    Ll0 = Dn(E00,U00,ev/2,betal,gl,dl,dldag,nr,nd)
    Lr0 = Dn(E00,U00,-ev/2,betar,gr,dr,drdag,nl,nd)
    Ld0 = Dn(E00,U00,mud0,betad,gd,dd,dddag,nl,nr)
    Htd0 =  Htd(El,Er,Ed,U00)
    rhof = Propagate(rho0,superop,40000)
    Ql0,Qr0,Qd0,Sl0,Sr0,Sd0,El0,Er0,Ed0,Wl0,Wr0,Wd0,Nl0,Nr0,Nd0 = currents(Htd0,ev/2,-ev/2,mud0,Ll0,Lr0,Ld0,superop0,rho0,40000)
    Ql.append(Ql0)
    Qr.append(Qr0)
    Qd.append(Qd0)
    #cohev.append(abs(rhof[5,3]) + abs(rhof[4,2]) )   
    sigmal = Sl0 - betal*Ql0
    sigmar = Sr0 - betar*Qr0
    Sls.append(Sl0 - betal*Ql0)
    Srs.append(Sr0 - betar*Qr0)
    Sds.append(Sd0 - betad*Qd0)
    Slr.append( sigmal + sigmar )
    Isl0 = -Sl0 - Sr0
    Id0 = -Sd0
    Il0 = -Sl0
    Ir0 = -Sr0
    Isl.append(Isl0)
    Id.append(-Sd0)
    Els.append(El0)
    Ers.append(Er0)
    Eds.append(Ed0)
    Erl.append(El0 + Er0 )
    Qlr.append(Ql0+Qr0 )
    Flr.append(El0 + Er0 + (1/betal)*Isl0 )
    Tisl.append((1/betal)*Isl0  )
    Wt.append(Wl0 + Wr0 )
    Wdf.append(Wd0)
    Qdf.append(Qd0)
    Fd.append(Ed0 + (1/betad)*Id0 )
    Tid.append((1/betad)*Id0  )
    Ilf.append(-Sl0)
    Irf.append(-Sr0) 
    Nls.append(Nl0)
    Nrs.append(Nr0)
    Nds.append(Nd0)
    eVs.append(ev*betal)
    entropf.append( -betal*(Ql0+Qr0) )
    auxff.append(0)
plt.plot(eVs,Ql,linestyle='--', dashes=(5, 9), color='red',lw = 4,label = r'$J_{L}$')
plt.plot(eVs,Qr,linestyle='--', dashes=(5, 9), color='blue', lw=4,label = r'$J_{R}$') 
plt.plot(eVs,Qd,linestyle='--', dashes=(5, 9), color='black',lw=4,label = r'$J_{d}$')
#plt.plot(eVs,Nls,label = r'$\dot{N}_{L}$')
#plt.plot(eVs,Nrs, label = r'$\dot{N}_{R}$') 
#plt.plot(eVs,Nds,label = r'$\dot{N}_{d}$')
#plt.xscale("log")
plt.xlabel(r'$eV/T$',fontsize = 20)
plt.ylabel(r'$J_{\alpha}$',fontsize = 20)
plt.xticks(fontsize=17)  # X-axis tick labels
plt.yticks(fontsize=17)  # Y-axis tick labels
plt.legend(loc = "lower left",fontsize=15) 
plt.show()   

plt.plot(eVs,Nls,linestyle='--', dashes=(5, 9), color='red',lw=4,label = r'$\dot{N}_{L}$')
plt.plot(eVs,Nrs,linestyle='--', dashes=(5, 9), color='blue',lw=4, label = r'$\dot{N}_{R}$') 
plt.plot(eVs,Nds,linestyle='--', dashes=(5, 9), color='black',lw=4,label = r'$\dot{N}_{d}$')
plt.xticks(fontsize=17)  # X-axis tick labels
plt.yticks(fontsize=17)  # Y-axis tick labels
#plt.xscale("log")
plt.xlabel(r'$eV/T$',fontsize = 20)
#plt.ylim(-0.0018, 0.0018) 
#plt.legend(loc='upper left')  
plt.ylabel(r'$\dot{N}_{\alpha}$',fontsize = 20)
plt.legend(loc = "lower left",fontsize=15) 
plt.show()  

plt.plot(eVs,Qlr, label = r'$\dot{Q}_{rl}$', color = 'b')
plt.plot(eVs,Qd,label = r'$\dot{Q}_{d}$', color = 'r')
plt.legend(fontsize = 15)
plt.xticks(fontsize=17)  # X-axis tick labels
plt.yticks(fontsize=17)
#plt.xscale("log")
plt.show()


plt.plot(eVs,Qd,label = r'$\dot{Q}_{d}$', color = 'b')
plt.plot(eVs,Id, label = r'$\dot{I}_{d}$', color = 'r')
plt.plot(eVs,Sds,label = r'$\dot{\sigma}_{d}$', color = 'k')
#plt.xscale("log")
plt.xlabel(r'$eV/T$',fontsize = 20)
plt.xticks(fontsize=17)  
plt.yticks(fontsize=17)
#plt.ylabel("Heat current",fontsize = 20)
plt.legend(fontsize=15)
plt.show()   

plt.plot(eVs,Sls,linestyle='--', dashes=(5, 9), color='red',lw=2,label = r'$\dot{\sigma}_{L}$')
plt.plot(eVs,Srs,linestyle='--', dashes=(5, 9), color='blue',lw=2, label = r'$\dot{\sigma}_{R}$') 
plt.plot(eVs,Sds,linestyle='--', dashes=(5, 9), color='black',lw=2, label = r'$\dot{\sigma}_{d}$')
plt.xlabel(r'$eV/T$',fontsize = 20)
plt.ylabel(r'$\dot{\sigma}_{\alpha}$',fontsize = 20)
plt.legend(fontsize = 15)
plt.xticks(fontsize=17)  # X-axis tick labels
plt.yticks(fontsize=17)
plt.show()   

plt.plot(eVs,Slr,linestyle='--', dashes=(5, 9), color = 'black',lw = 3)
plt.plot(eVs,auxff,linestyle='--', dashes=(5, 9), color = 'red',lw = 3)
plt.ylabel(r'$\dot{\sigma}_{LR}$',fontsize = 20)
plt.xlabel(r'$eV/T$', fontsize = 20)
#plt.legend(fontsize = 15)
plt.xticks(fontsize=17)  
plt.yticks(fontsize=17)
plt.show()


plt.plot(eVs,Isl, label = r'$\dot{I}_{LR}$', color = 'b')
#plt.plot(eVs,Coher, label = r'$\mathcal{I}_{cohel}$')
#plt.plot(eVs,Cohel, label = r'$\mathcal{I}_{coher}$')
#plt.plot(eVs,Classl, label = r'$\mathcal{I}_{classL}$')
#plt.plot(eVs,Classr, label = r'$\mathcal{I}_{classR}$')
#plt.plot(eVs,cohesum, label = r'$\mathcal{I}_{coheLR}$')
#plt.plot(eVs, sumtot, linestyle='--', color='blue')
plt.plot(eVs,Id, label = r'$\dot{I}_{d}$', color = 'r')
plt.xlabel(r'$eV/T$',fontsize = 20)
plt.legend(fontsize = 15)
plt.xticks(fontsize=17)  # X-axis tick labels
plt.yticks(fontsize=17)
#plt.xscale("log")
plt.legend()
plt.show()


plt.plot(eVs,Ers,linestyle='--', dashes=(5, 9), color='blue',lw=2, label = r'$\dot{E}_{R}$')
plt.plot(eVs,Els,linestyle='--', dashes=(5, 9), color='red',lw=2, label = r'$\dot{E}_{L}$')
plt.plot(eVs,Eds,linestyle='--', dashes=(5, 9), color='black',lw=2, label = r'$\dot{E}_{d}$')
plt.xlabel(r'$eV/T$',fontsize = 20)
plt.ylabel(r'$\dot{E}_{\alpha}$',fontsize = 20)
#plt.xscale("log")
plt.legend(fontsize = 15)
plt.xticks(fontsize=17)  
plt.yticks(fontsize=17)
plt.show()

plt.plot(eVs,Erl, label = r'$\dot{E}_{rl}$')
plt.xlabel(r'$eV/T$',fontsize = 20)
plt.plot(eVs,Qlr, label = r'$\dot{Q}_{rl}$')
plt.xticks(fontsize=17)  
plt.yticks(fontsize=17)
plt.legend(fontsize=15)
#plt.xscale("log")
plt.show()


plt.plot(eVs,Erl,linestyle='--', dashes=(5, 9), color='blue',lw=2, label = r'$\dot{E}_{LR}$')
#plt.plot(eVs,Isl,linestyle='--', dashes=(5, 9), color='red',lw=2, label = r'$\dot{I}_{rl}$')
plt.plot(eVs,Flr,linestyle='--', dashes=(5, 9), color='black',lw=2, label = r'$\dot{F}_{LR}$')
plt.plot(eVs,Tisl,label = r'$T\dot{I}_{LR}$', color = 'g',lw=2)
plt.plot(eVs,Wt,label = r'$\dot{W}_{LR}$', color = 'm',lw=2)
plt.plot(eVs,Qlr,label = r'$J_{LR}$',color = 'gray',lw=2)
plt.xlabel(r'$eV/T$',fontsize = 20)
plt.xticks(fontsize=17)  
plt.yticks(fontsize=17)
plt.legend(fontsize=15)
#plt.xscale("log")
plt.show()

plt.plot(eVs,Eds,linestyle='--', dashes=(5, 9), color='blue',lw=2, label = r'$\dot{E}_{d}$')
#plt.plot(eVs,Isl,linestyle='--', dashes=(5, 9), color='red',lw=2, label = r'$\dot{I}_{rl}$')
plt.plot(eVs,Fd,linestyle='--', dashes=(5, 9), color='black',lw=2, label = r'$\dot{F}_{d}$')
plt.plot(eVs,Tid,label = r'$T_{d}\dot{I}_{d}$', color = 'g',lw=2)
plt.plot(eVs,Wdf,label = r'$\dot{W}_{d}$', color = 'm',lw=2)
#plt.plot(eVs,Qdf,label = r'$J_{d}$',color = "gray",lw=2)
plt.xlabel(r'$eV/T$',fontsize = 20)
plt.xticks(fontsize=17)  
plt.yticks(fontsize=17)
plt.legend(fontsize=15,loc = "lower left")
#plt.xscale("log")
plt.show()



#ojo aqui, bajo eV=200, los puntos L y R parecen estar siendo medidos
#mientras que al superar esa vara L empieza a medir 
plt.plot(eVs,Id,linestyle='--', dashes=(5, 9), color='black',lw=2, label = r'$\dot{I}_{d}$')
plt.plot(eVs,Ilf,linestyle='--', dashes=(5, 9), color='red',lw=2, label = r'$\dot{I}_{l}$')
plt.plot(eVs,Irf,linestyle='--', dashes=(5, 9), color='blue',lw=2, label = r'$\dot{I}_{r}$')
plt.xlabel(r'$eV/T$',fontsize = 20)
plt.ylabel(r'$\dot{I}_{i}$',fontsize = 20)
plt.xticks(fontsize=17)  # X-axis tick labels
plt.yticks(fontsize=17)
plt.legend(fontsize=15)
#plt.xscale("log")
plt.show()



plt.plot(eVs,entropf,linestyle='--', dashes=(5, 9), color = 'black',lw = 3)
plt.plot(eVs,auxff,linestyle='--', dashes=(5, 9), color = 'red',lw = 3)
plt.xlabel(r'$eV/T$',fontsize = 20) 
plt.ylabel(r'$\dot{\sigma}^{o}_{LR}$',fontsize=20)    
plt.xticks(fontsize=17)  
plt.yticks(fontsize=17)
##plt.legend(fontsize=15)
plt.show()


archivo = open("lindbladgamU","w")
decimal_places = 7
total_width = 8
format_str = f"{{:.{decimal_places}f}}" 
#format_str = f"{{:{total_width}.{decimal_places}f}}"
for i in range(Num):
    archivo.write( format_str.format(eVs[i])) #guarda el grado del nodo
    #archivo.write(str(xs[i])) 
    archivo.write(" ") 
    #archivo.write(str(ys[i]))
    archivo.write( format_str.format(Nls[i]))
    archivo.write(" ") 
    #archivo.write(str(ys[i]))
    archivo.write( format_str.format(Id[i]))
    archivo.write("\n")
