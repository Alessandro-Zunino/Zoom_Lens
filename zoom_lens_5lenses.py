import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

plt.rcParams['pdf.fonttype'] = 42

def Lens(f):
    M = sym.Matrix( [[1, 0], [-1/f, 1]] )
    return M

def Space(d):
    M = sym.Matrix( [[1, d], [0, 1]] )
    return M


d1, d2, d3, d4, d5, d6 = sym.symbols('d_1 d_2 d_3 d_4 d_5 d_6')
f1, f2, f3, f4, f5 = sym.symbols('f_1 f_2 f_3 f_4 f_5')

#%% First Lens

M1 = Space(d2)*Lens(f1)*Space(f1)
M1 = sym.simplify(M1)

#%% Afocal system

M2 = Lens(f4)*Space(d4)*Lens(-f3)*Space(d3)*Lens(f2)
M2 = sym.simplify(M2)

S1 = sym.solve(M2[1,0], d4) 

M2 = M2.subs(d4, S1[0])
M2 = sym.simplify(M2)

#%% Last Lens

M3 = Space(f5)*Lens(f5)*Space(d5)
M3 = sym.simplify(M3)

#%% Complete system

M = M3*M2*M1

#%% numerical evaluation

MAG = M[0,0]

mag = sym.lambdify( (f1, f2, f3, f4, f5, d3), MAG, "numpy")
D3 = sym.lambdify( (f2, f3, f4, d3), S1[0], "numpy")

f_1 = 50
f_2 = 100
f_3 = 75
f_4 = 75
f_5 = 50
# f_1 = 50
# f_2 = 75
# f_3 = 50
# f_4 = 75
# f_5 = 75
d_3 = np.linspace(60, 201, num = 900)

Magnification = np.abs( mag(f_1, f_2, f_3, f_4, f_5, d_3) )
d_4 = D3(f_2, f_3, f_4, d_3)

plt.figure(1)

plt.subplot(2,3,1)

plt.xlabel('$d_3$ (mm)')
plt.ylabel('$d_3$ (mm)')
plt.plot(d_3, d_3, 'g')
# plt.xticks( np.arange(0,151,50) )

plt.subplot(2,3,2)

plt.xlabel('$d_3$ (mm)')
plt.ylabel('$d_4$ (mm)')
plt.plot(d_3, d_4, 'b')
# plt.xticks( np.arange(0,151,50) )

plt.subplot(2,3,3)

plt.xlabel('$d_3$ (mm)')
plt.ylabel('Magnification')
plt.plot(d_3, Magnification, 'r')
# plt.xticks( np.arange(0,151,50) )

plt.tight_layout()

# #%% Spot size - wavelength

# def M_vs_wl(M0, wl0, wl):
#     return M0*wl0/wl

# M0 = 1.5
# wl0 = 0.64e-3

# wl = np.linspace(0.488, 0.640, num = 100)*1e-3

# y = M_vs_wl(M0, wl0, wl)

# plt.subplot(3,4,4)
# ax = plt.gca()

# color = 'tab:blue'
# ax.set_xlabel('Wavelength (nm)', color=color)
# ax.set_ylabel('Magnification')
# ax.set_yticks( np.arange(1,3,0.1) )
# ax.plot(wl*1e6, y, color=color)

# #%% Spot size - numerical aperture

# def M_vs_NA(M0, NA0, NA):
#     return M0*NA/NA0

# M0 = 1.5
# NA0 = 1.4

# NA = np.linspace(1, 1.4, num = 100)

# y = M_vs_NA(M0, NA0, NA)

# plt.subplot(3,4,8)
# ax = plt.gca()

# color = 'tab:green'
# ax.set_xlabel('Objective lens NA', color=color)
# ax.set_ylabel('Magnification')
# ax.set_yticks( np.arange(1,3,0.1) )
# ax.plot(NA, y, color=color)

# #%% Spot size - objective magnification
# def M_vs_NA(M0, NA0, MO):
#     return M0*MO0/MO

# M0 = 1.5
# MO0 = 60

# MO = np.linspace(40,100, num = 100)

# y = M_vs_NA(M0, MO0, MO)

# plt.subplot(3,4,12)
# ax = plt.gca()

# color = 'tab:purple'
# ax.set_xlabel('Objective lens magnification', color=color)
# ax.set_ylabel('Magnification')
# ax.set_yticks( np.arange(1,3,0.25) )
# ax.plot(MO, y, color=color)


#%%

plt.subplot(2,1,2)
# plt.gca()
plt.plot(Magnification, d_3+d_4,'k')
plt.ylabel('Encumbrance')
plt.xlabel('Magnification')

ax = plt.gca()
ax.text(1.75,235, '$f_1 = $' + str(f_1) + ' mm')
ax.text(1.75,225, '$f_2 = $' + str(f_2) + ' mm')
ax.text(1.75,215, '$f_3 = -$' + str(f_3) + ' mm')
ax.text(2.25,235, '$f_4 = $' + str(f_4) + ' mm')
ax.text(2.25,225, '$f_5 = $' + str(f_5) + ' mm')

plt.tight_layout()

#%%

K = 60*1.5*644/1.4

M_max = K*1.2/(488*40)
M_min = K*0.1/(644*10)

print('M_max = ', M_max)
print('M_min = ', M_min)

plt.savefig('zoom_lens.pdf')
plt.savefig('zoom_lens.png')
