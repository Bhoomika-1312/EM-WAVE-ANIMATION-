import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# --- Physical constants ---
c0 = 299792458.0                # speed of light (m/s)
mu0 = 4e-7 * np.pi              # permeability of free space (H/m)
eps0 = 1.0 / (mu0 * c0**2)      # permittivity of free space (F/m)

# --- Helper for 10^x format ---
def sci_notation(num, decimal=4):
    """Return a string of num in 10^x form with specified decimals."""
    if num == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(num))))
    coeff = num / (10**exponent)
    return f"{coeff:.{decimal}f} × 10^{exponent}"

# --- Global variables ---
frequency_hz = None
Ex0 = None
Ey0 = None
Ez0 = None
k_vec = None
k_mag = None
wavelength_medium = None
wavelength_free = None
n = None
eps_r = None
pol_type = None

# --- Function to collect inputs and compute derived quantities ---
def initialize_wave():
    global frequency_hz, Ex0, Ey0, Ez0, k_vec, k_mag, wavelength_medium, wavelength_free, n, eps_r, pol_type

    frequency_hz = float(input("Enter frequency in Hz: "))

    print("\nEnter complex electric field components (format: a+bj)")
    Ex0 = complex(input("Ex0 = "))
    Ey0 = complex(input("Ey0 = "))
    Ez0 = complex(input("Ez0 = "))

    print("\nEnter wave vector components (Kx, Ky, Kz)")
    Kx = float(input("Kx = "))
    Ky = float(input("Ky = "))
    Kz = float(input("Kz = "))

    k_vec = np.array([Kx, Ky, Kz], dtype=float)
    k_mag = np.linalg.norm(k_vec)

    wavelength_medium = 2 * np.pi / k_mag
    wavelength_free = c0 / frequency_hz
    n = wavelength_free / wavelength_medium
    eps_r = n**2

    print("\n--- Derived Parameters ---")
    print(f"Wave number (k) = {sci_notation(k_mag)} rad/m")
    print(f"Wavelength in medium = {sci_notation(wavelength_medium)} m")
    print(f"Free-space wavelength = {sci_notation(wavelength_free)} m")
    print(f"Refractive index (n) = {n:.4f}")
    print(f"Relative Permittivity = {eps_r:.4f}")

    # --- Polarization detection ---
        # --- Polarization detection (improved) ---
    # normalize wave vector
    k_hat = k_vec / np.linalg.norm(k_vec)

    # find two orthonormal transverse unit vectors
    if not np.allclose(k_hat, [0, 0, 1]):
        ref = np.array([0, 0, 1])
    else:
        ref = np.array([0, 1, 0])

    e1 = np.cross(k_hat, ref)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(k_hat, e1)

    # project E0 onto transverse basis
    E0_vec = np.array([Ex0, Ey0, Ez0], dtype=complex)
    E1 = np.vdot(e1, E0_vec)
    E2 = np.vdot(e2, E0_vec)

    A_amp = np.abs(E1)
    B_amp = np.abs(E2)
    phi = np.angle(E2) - np.angle(E1)
    phi_deg = (np.degrees(phi) + 360) % 360

    # classification
    if np.isclose(A_amp, B_amp, rtol=1e-2) and np.isclose(abs(phi_deg - 90) % 180, 0, atol=5):
        pol_type = "Circular"
    elif np.isclose(phi_deg % 180, 0, atol=5) or np.isclose(A_amp, 0, atol=1e-2) or np.isclose(B_amp, 0, atol=1e-2):
        pol_type = "Linear"
    else:
        pol_type = "Elliptical"

    print(f"\nDetected Polarization: {pol_type}")


    # --- Magnetic field phasor ---
    omega = 2 * np.pi * frequency_hz
    E0_vec = np.array([Ex0, Ey0, Ez0], dtype=complex)
    H0_vec = np.cross(k_vec.astype(complex), E0_vec) / (omega * mu0)
    Hx0, Hy0, Hz0 = H0_vec

    print("\n--- Magnetic Field Phasor Components (H0) ---")

    def print_phasor(name, val):
        mag = np.abs(val)
        ang = np.degrees(np.angle(val))
        print(f"{name} = {val.real:+.4e}{val.imag:+.4e}j "
              f"( |{name}| = {mag:.4e}, ∠ = {ang:.2f}° )")

    print_phasor("Hx0", Hx0)
    print_phasor("Hy0", Hy0)
    print_phasor("Hz0", Hz0)

    # Optional: transversality check (k·E ≈ 0 for physical plane waves)
    k_dot_E = np.dot(k_vec, E0_vec)
    if np.abs(k_dot_E) > 1e-6:
        print(f"WARNING: k·E0 = {k_dot_E:.4e} ≠ 0 → not a transverse plane wave.")

# --- Call the initialization ---
initialize_wave()

# --- Common wave parameters for animation ---
k = k_mag
w = 2 * np.pi * frequency_hz
z = np.linspace(0, 20, 200)[::5]  # positions along propagation

fps = 10
duration = 20  # seconds
frames = fps * duration
t_vals = np.linspace(0, duration, frames)

# --- Set up figure ---
# --- Set up figure ---
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

# Define amplitude scale (max of Ex0, Ey0, Ez0)
amp = max(abs(Ex0), abs(Ey0), abs(Ez0), 1e-3)

ax.set_xlim(0, max(z))             # wave propagation
ax.set_ylim(-2*amp, 2*amp)         # Ex / Ey scale
ax.set_zlim(-2*amp, 2*amp)

ax.set_xlabel('z')
ax.set_ylabel('Ex')
ax.set_zlabel('Ey')
ax.set_title(f"{pol_type} Polarization", fontsize=14)

ax.view_init(elev=20, azim=60)

wave_line, = ax.plot([], [], [], 'r', lw=2)
tip_point, = ax.plot([], [], [], 'bo', markersize=6)

# --- Animation update ---
def update(frame):
    t = t_vals[frame]
    z0 = np.linspace(0, 2*np.pi/k, 50)  # a few points along propagation

    # Compute E-field along z
    Ex = np.real(Ex0 * np.exp(-1j*(k*z0 - w*t)))
    Ey = np.real(Ey0 * np.exp(-1j*(k*z0 - w*t)))
    Ez = np.real(Ez0 * np.exp(-1j*(k*z0 - w*t)))

    # Set data for 3D plot
    wave_line.set_data(z0, Ex)
    wave_line.set_3d_properties(Ey)

    # Tip at last point along z
    tip_point.set_data([z0[-1]], [Ex[-1]])
    tip_point.set_3d_properties([Ey[-1]])

    return wave_line, tip_point

  #  Ex = np.real(Ex0 * np.exp(-1j*(k*z - w*t*speed_factor)))
   # Ey = np.real(Ey0 * np.exp(-1j*(k*z - w*t*speed_factor)))
   # Ez = np.real(Ez0 * np.exp(-1j*(k*z - w*t*speed_factor)))

   # wave_line.set_data(z, Ex)
   # wave_line.set_3d_properties(Ey)

  #  tip_point.set_data([z[0]], [Ex[0]])
  #  tip_point.set_3d_properties([Ey[0]])
   # return wave_line, tip_point


ani = FuncAnimation(fig, update, frames=len(t_vals), interval=200, blit=False)
plt.close(fig)

HTML(ani.to_jshtml())
