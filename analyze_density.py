import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import yt
from yt.units import mh

# def compute_mean_contrast(number_density_2d, cut_axis='x', smoothing_sigma=2.0):
contrasts = []

spacing = [36, 90, 144]

time_step = 0
time = []
size = 256
grid_x = np.linspace(0, 1, size)
grid_y = np.linspace(0, 1, size)

case_1 = np.zeros((int(size / 16), size))
case_2 = np.zeros((int(size / 16), size))
case_3 = np.zeros((int(size / 16), size))

X,Y = np.meshgrid(grid_x, grid_y)
unit_base={"length_unit": (1.0,"pc"), "time_unit": (1.0,"1.0 * pc / (0.35 * km/s)"), "mass_unit": (3.532e34,"g")}

for space in spacing:
    # For wavelength 2L - I need 2-pc/1.58-km/s to get desired properties - 15uG, 1.58 km/s and 200 cm-3    ?
    # For wavelength L - I need 1-pc/1.0-km/s to get desired properties - 15uG,                                                                                                                                                            1.58 km/s and 200 cm-3      ?
    # For wavelength L - I need 1-pc/1.0-km/s to get desired properties - 15uG, 1.58 km/s and 200 cm-3      ?

    name = 'high_b_dens_grad_low'

    ds_256 = yt.load(f'./results_{name}/data_files_256/LinWave.out2.{space:05d}.athdf', units_override=unit_base)

    # print(ds_256.field_list)
    # # print(ds_256.derived_field_list)
    # # print(ds_256.parameters)

    # break

    time_evolved = ds_256.current_time.to("Myr")
    time.append(time_evolved)
    print(time_evolved)

    left_edge = ds_256.domain_left_edge
    right_edge = ds_256.domain_right_edge
    dims =  ds_256.domain_dimensions

    # print(left_edge, dims)

    data_256 = ds_256.covering_grid(level=0, left_edge=left_edge, dims=dims)

    accurate_number_density_256 = data_256['rho'].to("g/cm**3") / (2.34 * mh)
    number_density = accurate_number_density_256.to("cm**-3")[:, :, 0].T
    n_0 = np.mean(number_density[0])

    data = number_density

    # extrema_coords_max = []
    # extrema_coords_min = []

    for x in range(16, size, 16):
        averaged_value = ((number_density[x-1] + number_density[x] + number_density[x+1]) / 3.0) / n_0

        if space == spacing[0]:
            case_1[int(x / 16)] = averaged_value
        elif space == spacing[1]:
            case_2[int(x / 16)] = averaged_value
        else:
            case_3[int(x / 16)] = averaged_value

plt.figure(figsize=(6, 4), dpi=300)  # High-resolution figure

# Plot 3 cases with consistent styling
plt.plot(grid_x, case_1[(int(size / (16*2)))], color="black", lw=0.8, label=f'{time[0]:.2f}')
plt.plot(grid_x, case_2[(int(size / (16*2)))], color="tab:blue", lw=0.8, ls="--", label=f'{time[1]:.2f}')
plt.plot(grid_x, case_3[(int(size / (16*2)))], color="tab:red", lw=0.8, ls="-.", label=f'{time[2]:.2f}')

# Labels
plt.xlabel("x [pc]", fontsize=12)
plt.ylabel(r"$n_{\mathrm{density}}(x)/\,n_0$", fontsize=12)

# Ticks
plt.tick_params(axis='both', which='major', labelsize=10, direction="in", length=4)
plt.tick_params(axis='both', which='minor', direction="in", length=2)
plt.minorticks_on()

# Axes border
# for spine in ['top', 'right']:
#     plt.gca().spines[spine].set_visible(False)

# Legend
plt.legend(frameon=True, fontsize=9, loc="best")
plt.savefig(f"density_{name}_midplane.pdf", bbox_inches="tight")

# Layout
plt.tight_layout()
plt.show()
