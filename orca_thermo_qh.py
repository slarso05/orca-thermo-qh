import re
import numpy as np

## ORCA 5.0.4 Quasi-Harmonic Correction Script

# Constants
kB = 1.380658e-23        # J/K
h = 6.62607015e-34       # J·s
R = 0.008314             # kJ/mol/K
Avo = 6.0221367e23       # 1/mol
T = 298.15               # K
amu_to_kg = 1.66053906660e-27  # kg/amu
bohr_to_m = 5.29177210903e-11  # m/bohr

# Config
path = "tester.txt"  # ORCA raw output file

with open(path, "r", encoding="utf-8", errors="ignore") as f:
    output = f.read()

# 1) Final energy
m = re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", output)
if not m:
    raise ValueError("Could not find FINAL SINGLE POINT ENERGY")
Eopt = float(m[-1])

# 2) Vibrational frequencies
freq_block_m = re.search(r"VIBRATIONAL FREQUENCIES.*?\n-+\n(.*?)-+\n", output, flags=re.S)
if not freq_block_m:
    raise ValueError("Could not find VIBRATIONAL FREQUENCIES block")
freq_block = freq_block_m.group(1)

# Grab all floats (incl. negatives), then keep positive ones
Vibfreq_all = [float(x) for x in re.findall(r"-?\d+\.\d+", freq_block)]
Vibfreq = np.array([v for v in Vibfreq_all if v > 0.0], dtype=float)

# Quasi-harmonic min at 150 cm^-1
Vibfreq_QHcorr = np.maximum(Vibfreq, 150.0)

# 3) Moments of inertia from ORCA rotational constants
def get_moments_amu_bohr2_from_orca(text: str):
    m_mhz = re.search(r"Rotational constants\s*in\s*MHz\s*:([^\n]+)", text, flags=re.I)
    if not m_mhz:
        raise ValueError("Could not find rotational constants (MHz).")
    nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", m_mhz.group(1))]
    B_Hz = [x * 1e6 for x in nums]
    I_SI = [h / (8.0 * np.pi**2 * B) for B in B_Hz]             # kg·m²
    conv = 1.0 / (amu_to_kg * bohr_to_m**2)                     # → amu·bohr²
    I1, I2, I3 = (I * conv for I in I_SI)
    return float(I1), float(I2), float(I3)

I1, I2, I3 = get_moments_amu_bohr2_from_orca(output)

# 4) Symmetry number
ms = re.search(r"Symmetry number\s*=\s*(\d+)", output)
sigma = int(ms.group(1)) if ms else 1

# 5) Total mass (amu = g/mol)
mmatch = re.search(r"Total Mass\s+\.\.\.\s+(\d+\.\d+)", output)
if not mmatch:
    raise ValueError("Could not find Total Mass")
mmass = float(mmatch.group(1))

# 6) Thermo corrections
aVibT = 1.43877 * Vibfreq
aEvib = aVibT * (0.5 + 1.0 / (np.exp(aVibT / T) - 1.0))
Evib = R * np.sum(aEvib)              # kJ/mol

Etrans = 1.5 * R * T
Erot = 1.5 * R * T
Ecorr = Evib + Etrans + Erot
Hcorr = Ecorr + R * T

m_kg = (mmass / 1000.0) / Avo
qtrans = ((2.0 * np.pi * m_kg * kB * T) / (h**2))**1.5 * ((kB * T) / 1.01325e5)
Strans = R * (np.log(qtrans) + 1.0 + 1.5)

I1si = I1 * amu_to_kg * bohr_to_m**2
I2si = I2 * amu_to_kg * bohr_to_m**2
I3si = I3 * amu_to_kg * bohr_to_m**2

TempI1 = h**2 / (8.0 * np.pi**2 * I1si * kB)
TempI2 = h**2 / (8.0 * np.pi**2 * I2si * kB)
TempI3 = h**2 / (8.0 * np.pi**2 * I3si * kB)

qrot = (np.pi**0.5 / sigma) * (T**1.5 / (TempI1 * TempI2 * TempI3)**0.5)
Srot = R * (np.log(qrot) + 1.5)

aSvib = ((aVibT / T) / (np.exp(aVibT / T) - 1.0)) - np.log(1.0 - np.exp(-aVibT / T))
Svib = R * np.sum(aSvib)

aVibT_QH = 1.43877 * Vibfreq_QHcorr
aSvib_QH = ((aVibT_QH / T) / (np.exp(aVibT_QH / T) - 1.0)) - np.log(1.0 - np.exp(-aVibT_QH / T))
Svib_QH = R * np.sum(aSvib_QH)

Scorr = Strans + Srot + Svib
Scorr_QHcorr = Strans + Srot + Svib_QH

Gcorr = Hcorr - T * Scorr                  # kJ/mol
Gcorr_QHcorr = Hcorr - T * Scorr_QHcorr    # kJ/mol

# kJ/mol → Hartree
H_Ecorr = Ecorr / 2625.5
H_Hcorr = Hcorr / 2625.5
H_Gcorr = Gcorr / 2625.5
H_Gcorr_QH = Gcorr_QHcorr / 2625.5

E_final = Eopt + H_Ecorr
H_final = Eopt + H_Hcorr
G_final = Eopt + H_Gcorr
qG_final = Eopt + H_Gcorr_QH

print(f"E = {E_final:.10f} Hartree")
print(f"H = {H_final:.10f} Hartree")
print(f"G = {G_final:.10f} Hartree")
print(f"qG (Quasiharmonic corrected Gibbs) = {qG_final:.10f} Hartree")
