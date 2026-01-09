import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import yt
from yt.units import mh

# def compute_mean_contrast(number_density_2d, cut_axis='x', smoothing_sigma=2.0):
def sine1d(x, A, k, phi, C):
    return A * np.sin(k * x + phi) + C

contrasts = []

spacing = [110]

time_step = 0
time = []
size = 256
grid_x = np.linspace(0, 1.58, size)
grid_y = np.linspace(0, 1, size)

case_1 = np.zeros((int(size / 16), size))
case_2 = np.zeros((int(size / 16), size))
case_3 = np.zeros((int(size / 16), size))

X,Y = np.meshgrid(grid_x, grid_y)
unit_base={"length_unit": (1.0,"pc"), "time_unit": (1.0,"1.0 * pc / (0.35 * km/s)"), "mass_unit": (3.532e34,"g")}

for space in spacing:
    # For wavelength 2L - I need 2-pc/1.58-km/s to get desired properties - 15uG, 1.58 km/s and 200 cm-3    ?
    # For wavelength L - I need 1-pc/1.0-km/s to get desired properties - 15uG, 1.58 km/s and 200 cm-3      ?
    # For wavelength L - I need 1-pc/1.0-km/s to get desired properties - 15uG, 1.58 km/s and 200 cm-3      ?

    ds_256 = yt.load(f'./wavelength_analysis/1k_wavelength/data_files_256/LinWave.out2.{space:05d}.athdf', units_override=unit_base)

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

        if space == 108:
            case_1[int(x / 16)] = averaged_value
        # elif space == 180:
        #     case_2[int(x / 16)] = averaged_value
        # else:
        #     case_3[int(x / 16)] = averaged_value

    # ---------------------------
    A = 1.0
    λ = 3.0
    phi = 4.11
    scale = (grid_y.max() - grid_y.min()) * 0.1
    y_pos = grid_y[size * 3 // 4]

    x_dense = np.linspace(grid_x.min(), grid_x.max(), 1000)
    sine_curve = A * np.sin(2 * np.pi * x_dense / λ + phi)

    # Overlay sine (use the same axes, but restore limits afterward)

    # Initial guess
    
    plt.figure(figsize=(6, 5))
    print(X.dtype, Y.dtype, number_density.dtype)
    plt.pcolormesh(X, Y, number_density, cmap='gray', shading='auto', vmin = 100, vmax = 300)
    plt.plot(x_dense, y_pos + sine_curve * scale, 'b--', lw=1.1, label='Manual sine')

    # plt.axhspan(0.3, 1.0, facecolor="red", alpha=0.15,
    #         edgecolor="black", linewidth=1.2)

    # Add label above the box (centered)
    # plt.text(
    #     0.5, 0.4, "Striations",    # x=0.5 (middle), y=0.75 (inside box)
    #     ha="center", va="center",
    #     fontsize=8,
    #     color="black",
    #     bbox=dict(facecolor="white", alpha=0.05, edgecolor="none")  # subtle background
    # )

    plt.title(f'Number Density Map (time={time_evolved:.4f})')
    plt.colorbar(label=r'Number Density (cm$^{-3}$)')
    plt.xlabel('x [pc]')
    plt.ylabel('y [pc]')
    plt.savefig(f'sine_fit_0.5_lambda_density_map_{time_evolved:.4f}.png', dpi=300, bbox_inches='tight')
    plt.close()

# plt.figure(figsize=(6, 4), dpi=300)  # High-resolution figure

# # Plot 3 cases with consistent styling
# plt.plot(grid_x, case_1[(int(size / (16*2)))], color="black", lw=1.5, label=f'{time[0]:.2f}')
# plt.plot(grid_x, case_2[(int(size / (16*2)))], color="tab:blue", lw=1.5, ls="--", label=f'{time[1]:.2f}')
# plt.plot(grid_x, case_3[(int(size / (16*2)))], color="tab:red", lw=1.5, ls="-.", label=f'{time[2]:.2f}')

# # Labels
# plt.xlabel("x [pc]", fontsize=10)
# plt.ylabel(r"$n(x, \, y = 0.5)/\,n_0$", fontsize=10)

# # Ticks
# plt.tick_params(axis='both', which='major', labelsize=10, direction="in", length=4)
# plt.tick_params(axis='both', which='minor', direction="in", length=2)
# plt.minorticks_on()

# # Axes border
# # for spine in ['top', 'right']:
# #     plt.gca().spines[spine].set_visible(False)

# # Legend
# plt.legend(frameon=False, fontsize=9, loc="best")
# plt.savefig("uniform_low_b_density_map_midplane.pdf", bbox_inches="tight")

# # Layout
# plt.tight_layout()
# plt.show()
