import yt
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from yt.units import mh
# import imageio
# from PIL import Image
import glob
import matplotlib.patches as patches
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter


def compute_mean_contrast(smooth_profile):
    peaks, _ = find_peaks(smooth_profile, distance=4)
    troughs, _ = find_peaks(-smooth_profile, distance=4)
    extrema = np.sort(np.concatenate([peaks, troughs]))

    # Compute contrast between each successive extrema pair
    for j in range(len(extrema) - 1):
        a = extrema[j]
        b = extrema[j + 1]
        I1 = smooth_profile[a]
        I2 = smooth_profile[b]

        # Skip zero or negative intensity (can happen due to noise)
        if I1 + I2 == 0:
            continue

        contrast = abs(I1 - I2) * 100 / np.mean(smooth_profile)
        return contrast

spacing = np.linspace(10, 300, 290, dtype=int)
# spacing = [0,36]

time_step = 0
time_in_myr = []

dcf_original = []
dcf_density = []

real = []

size = 256
grid_x = np.linspace(0, 1, size)
grid_y = np.linspace(0, 1, size)

X,Y = np.meshgrid(grid_x, grid_y)
unit_base={"length_unit": (1.0,"pc"), "time_unit": (1.0,"1.0 * pc / (0.35 * km/s)"), "mass_unit": (3.532e34,"g")}

for space in spacing:
    name = 'low_b_dens_grad_high'
    # For wavelength 2L - I need 2-pc/1.58-km/s to get desired properties - 15uG, 1.58 km/s and 200 cm-3    ?
    # For wavelength L - I need 1-pc/1.0-km/s to get desired properties - 15uG, 1.58 km/s and 200 cm-3      ?
    # For wavelength L - I need 1-pc/1.0-km/s to get desired properties - 15uG, 1.58 km/s and 200 cm-3      
    ds = yt.load(f'./results_{name}/data_files_256/LinWave.out2.{space:05d}.athdf', units_override=unit_base)

    left_edge = ds.domain_left_edge
    dims =  ds.domain_dimensions

    grid = ds.covering_grid(level=0, left_edge=left_edge, dims=dims)

    density = grid['rho'].to("g/cm**3")[:, :, 0].T
    accurate_number_density = grid['rho'].to("g/cm**3") / (2.34 * mh)
    number_density = accurate_number_density.to("cm**-3")[:, :, 0].T

    striations = number_density > 0

    physical_magnetic_field_y = grid[('gas', 'magnetic_field_y')].to("uG")[:, :, 0].T
    physical_magnetic_field_x = grid[('gas', 'magnetic_field_x')].to("uG")[:, :, 0].T
    b_total = np.sqrt(physical_magnetic_field_x**2 + physical_magnetic_field_y**2)

    velocity_x = grid[('gas', 'velocity_x')].to("km/s")[:, :, 0].T
    velocity_y = grid[('gas', 'velocity_y')].to("km/s")[:, :, 0].T

    # polarization_angle = np.arctan2(physical_magnetic_field_y, physical_magnetic_field_x)

    smoothening = 10
    mean_magnetic_x = gaussian_filter(physical_magnetic_field_x, smoothening)
    mean_magnetic_y = gaussian_filter(physical_magnetic_field_y, smoothening)

    mean_magnetic_x = yt.YTArray(mean_magnetic_x, 'uG')
    mean_magnetic_y = yt.YTArray(mean_magnetic_y, 'uG')

    mean_angle = np.arctan2(b_total, np.abs(mean_magnetic_x))
    sigma_phi_old = np.std(mean_angle)

    # SIGMA using phi = delta(B_y)/ B_0
    angle = (np.abs(physical_magnetic_field_y - mean_magnetic_y)) / np.mean(b_total)
    net_dispersion = angle - np.mean(angle)
    sigma_phi_new = np.sqrt(np.average(net_dispersion**2, weights= density))

    # SIGMA using density = delta(rho)/ rho_0
    sigma_density_contrasts = []
    for ind in range(16, size, 4):
        average_number_density = (number_density[ind] + number_density[ind-1] + number_density[ind+1]) / 3.0
        mean_density = gaussian_filter(average_number_density, smoothening)
        mean_density = yt.YTArray(mean_density, 'cm**-3')
        density_fluctuations = np.abs(average_number_density - mean_density)
        density_contrast = density_fluctuations / mean_density
        sigma_density_contrasts.append(np.std(density_contrast))
    
    sigma_density = np.mean(sigma_density_contrasts)

    mean_velocity_x = (gaussian_filter(velocity_x * density, smoothening) / gaussian_filter(density, smoothening))
    mean_velocity_y = (gaussian_filter(velocity_y * density, smoothening) / gaussian_filter(density, smoothening))

    mean_velocity_x = yt.YTArray(mean_velocity_x, 'cm/s').to('km/s')
    mean_velocity_y = yt.YTArray(mean_velocity_y, 'cm/s').to('km/s')
    
    rms_speed = np.average(np.sqrt(mean_velocity_x**2 + mean_velocity_y**2))

    print("MEAN velocity", rms_speed)

    # print(mean_velocity_x, mean_velocity_y)

    fluctuating_Vx = (velocity_x - mean_velocity_x).to('km/s')
    fluctuating_Vy = (velocity_y - mean_velocity_y).to('km/s')

    sigma_Vx = np.sqrt(np.average(fluctuating_Vx**2, weights=density)).to('km/s')
    sigma_Vy = np.sqrt(np.average(fluctuating_Vy**2, weights=density)).to('km/s')

    sigma_velocity = (np.sqrt(0.5*(sigma_Vx**2 + sigma_Vy**2)))

    print(sigma_velocity)

    mean_density = np.average(density)

    Q = 0.5

    B_pos_dcf_old = (np.sqrt(4 * np.pi * mean_density)* (rms_speed / (sigma_phi_old))).to("uG")
    B_pos_dcf_den = (np.sqrt(4 * np.pi * mean_density)* (rms_speed / (sigma_density))).to("uG")

    dcf_density.append(B_pos_dcf_den)
    dcf_original.append(B_pos_dcf_old)

    net_magnetic_field = np.sqrt(physical_magnetic_field_x**2 + physical_magnetic_field_y**2)
    real.append(np.average(net_magnetic_field))

    time_evolved = ds.current_time.to("Myr")
    time_in_myr.append(time_evolved)

    # print(B_pos_dcf)
    # print(B_pos_st)
    print(np.average(net_magnetic_field))
    

# Plot the total mass

# plt.figure(figsize=(6, 5))
# # # plt.plot(time_in_myr, mag_x_low_y, label='y~0')
# # # plt.plot(time_in_myr, mag_x_high_y, label='y~1')
# # # plt.plot(time_in_myr, mean_backgroung_mag, label='mean Y')
# plt.plot(time_in_myr, real, 'o-', label=r'B_{simulation}')
# plt.plot(time_in_myr, dcf,'s-', label=r'B_{DCF}')
# plt.plot(time_in_myr, st,'^-', label=r'B_{ST}')
# # # plt.yscale('log')
# plt.legend()
# plt.xlabel(r'Time [MYr]')
# plt.ylabel(r'B [\mu G]')
# plt.show()

linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
colors = ['black', '0.3', '0.5', '0.7', '0.85']

plt.figure(figsize=(6, 4), dpi=300)  # High-resolution figure

# Plot 3 cases with consistent styling
plt.plot(time_in_myr, real, color=colors[0], lw=0.8, ls=linestyles[0], label=r'$B_{0}$')
plt.plot(time_in_myr, dcf_original, color=colors[1], lw=0.8, ls=linestyles[1], label=r'$B_{DCF}, \delta \theta = \frac{\delta B_x}{B_0}$')
plt.plot(time_in_myr, dcf_density, color=colors[3], lw=0.8, ls=linestyles[3], label=r'$B_{DCF}, \delta \theta = \frac{\delta \rho}{\rho_0}$')

# Labels
plt.xlabel("time [Myr]", fontsize=10)
plt.ylabel(r"$B [{\mu G}]$", fontsize=10)

# Ticks
plt.tick_params(axis='both', which='major', labelsize=10, direction="in", length=4)
plt.tick_params(axis='both', which='minor', direction="in", length=2)
plt.minorticks_on()

# Axes border
# for spine in ['top', 'right']:
#     plt.gca().spines[spine].set_visible(False)

# Legend
plt.legend(handlelength=2.5)
plt.legend(frameon=False, fontsize=8, loc="best")
plt.savefig(f"DCF_method_magnetic_field_comparison_{name}.pdf", bbox_inches="tight")

# Layout
plt.tight_layout()
plt.show()
