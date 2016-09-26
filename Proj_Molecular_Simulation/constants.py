from math import pi

# electron charge
q_e = 1.60217662e-19  # Coulomb

# vacuum permittivity
epsilon = 8.8541878176e-12  # F/m

# constant for electron-electron total. Unit: `eV/m`
k_qq = 1 / (4 * pi * epsilon) * q_e  # eV/m
