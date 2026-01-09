import yt
import numpy as np
import matplotlib.pyplot as plt
from yt.units import mh
from scipy.signal import windows
from scipy.ndimage import zoom
from scipy.signal.windows import hann
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft, fftshift

from astropy.timeseries import LombScargle
from scipy.signal import welch

spacing = [12]

time = []
mag_energy = np.array([])

size = 256
grid_x = np.linspace(0, 1, size)
grid_y = np.linspace(0, 1, size)

case_1 = np.zeros((1, size))
case_2 = np.zeros((1, size))
case_3 = np.zeros((1, size))

X,Y = np.meshgrid(grid_x, grid_y)
unit_base={"length_unit": (1.0,"pc"), "time_unit": (1.0,"1.0 * pc / (0.35 * km/s)"), "mass_unit": (3.532e34,"g")}

for space in spacing:
    # For wavelength 2L - I need 2-pc/1.58-km/s to get desired properties - 15uG, 1.58 km/s and 200 cm-3    ?
    # For wavelength L - I need 1-pc/1.0-km/s to get desired properties - 15uG, 1.58 km/s and 200 cm-3      ?
    # For wavelength L - I need 1-pc/1.0-km/s to get desired properties - 15uG, 1.58 km/s and 200 cm-3      ?
    name = 'low_mid_b_dens_grad_high'

    ds_256 = yt.load(f'./results_{name}/data_files_256/LinWave.out2.{space:05d}.athdf', units_override=unit_base)

    left_edge = ds_256.domain_left_edge
    right_edge = ds_256.domain_right_edge
    dims =  ds_256.domain_dimensions

    # print(left_edge, dims)

    data_256 = ds_256.covering_grid(level=0, left_edge=left_edge, dims=dims)
    density = data_256['rho'].to("g/cm**3")[:, :, 0].T

    accurate_number_density_256 = data_256['rho'].to("g/cm**3") / (2.34 * mh)
    number_density = accurate_number_density_256.to("cm**-3")[:, :, 0].T

    time_evolved = ds_256.current_time.to("Myr")
    time.append(time_evolved)

    # vrms = np.sqrt(velocity_x**2 + velocity_y**2)

    velocity_comp_x = data_256[('gas', 'velocity_x')].to("km/s")[:, :, 0].T
    velocity_comp_y = data_256[('gas', 'velocity_y')].to("km/s")[:, :, 0].T

    tolerance = 1e-10
    for x in range(16, size, 16):
            if abs((x / size) - 0.25) < tolerance:
                  case_1 = ((abs(velocity_comp_x[x]) / abs(velocity_comp_y[x])) + (abs(velocity_comp_x[x-1]) / abs(velocity_comp_y[x-1])) 
                            + (abs(velocity_comp_x[x+1]) / abs(velocity_comp_y[x+1]))) / 3.0

            elif abs((x / size) - 0.50) < tolerance:
                  case_2 = ((abs(velocity_comp_x[x]) / abs(velocity_comp_y[x])) + (abs(velocity_comp_x[x-1]) / abs(velocity_comp_y[x-1])) 
                            + (abs(velocity_comp_x[x+1]) / abs(velocity_comp_y[x+1]))) / 3.0

            elif abs((x / size) - 0.75) < tolerance:
                  case_3 = ((abs(velocity_comp_x[x]) / abs(velocity_comp_y[x])) + (abs(velocity_comp_x[x-1]) / abs(velocity_comp_y[x-1])) 
                            + (abs(velocity_comp_x[x+1]) / abs(velocity_comp_y[x+1]))) / 3.0

linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
colors = ['black', '0.3', '0.5', '0.7', '0.85']

plt.figure(figsize=(6, 4), dpi=300)  # High-resolution figure

# ind = np.where(case_3 > 100)
# case_3[ind] = np.mean(case_3)

# Plot 3 cases with consistent styling
plt.plot(grid_x[15:240], case_1[15:240], color="black", lw=0.8, ls=linestyles[0], label=r'$y = 0.25 $ pc')
plt.plot(grid_x[15:240], case_2[15:240], color="tab:blue", lw=0.8, ls=linestyles[1], label=r'$y = 0.50 $ pc')
plt.plot(grid_x[15:240], case_3[15:240], color="tab:red", lw=0.8, ls=linestyles[2], label=r'$y = 0.75 $ pc')

# Labels
plt.xlabel("x [pc]", fontsize=10)
plt.ylabel(r'$v_x / v_y$', fontsize=10)
plt.yscale('log')
curr_time = f'{time[0]:.2f}'
plt.title('time = {}'.format(curr_time))

# Ticks
plt.tick_params(axis='both', which='major', labelsize=10, direction="in", length=4)
plt.tick_params(axis='both', which='minor', direction="in", length=2)
plt.minorticks_on()

# Axes border
# for spine in ['top', 'right']:
#     plt.gca().spines[spine].set_visible(False)

# Legend
plt.legend(handlelength=2.5)
plt.legend(frameon=True, fontsize=8, loc="best")
plt.savefig(f"vel_components_{name}_time_{curr_time}.pdf", bbox_inches="tight")

# Layout
plt.tight_layout()
plt.show()
