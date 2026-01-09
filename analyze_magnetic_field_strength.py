import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import yt
from yt.units import mh
from scipy.ndimage import gaussian_filter

# def compute_mean_contrast(number_density_2d, cut_axis='x', smoothing_sigma=2.0):

def get_velocity_dispersion(grid):
    density = grid['rho'].to("g/cm**3")[:, :, 0].T
    velocity_x = grid[('gas', 'velocity_x')].to("km/s")[:, :, 0].T
    velocity_y = grid[('gas', 'velocity_y')].to("km/s")[:, :, 0].T

    smoothening = 10

    mean_velocity_x = (gaussian_filter(velocity_x * density, smoothening) / gaussian_filter(density, smoothening))
    mean_velocity_y = (gaussian_filter(velocity_y * density, smoothening) / gaussian_filter(density, smoothening))

    mean_velocity_x = yt.YTArray(mean_velocity_x, 'cm/s').to('km/s')
    mean_velocity_y = yt.YTArray(mean_velocity_y, 'cm/s').to('km/s')

    # print(mean_velocity_x, mean_velocity_y)

    fluctuating_Vx = (velocity_x - mean_velocity_x).to('km/s')
    fluctuating_Vy = (velocity_y - mean_velocity_y).to('km/s')

    sigma_Vx = np.sqrt(np.average(fluctuating_Vx**2, weights=density)).to('km/s')
    sigma_Vy = np.sqrt(np.average(fluctuating_Vy**2, weights=density)).to('km/s')

    sigma_velocity = (np.sqrt(0.5*(sigma_Vx**2 + sigma_Vy**2)))

    return sigma_velocity

def get_polarization_dispersion(grid):
    density = grid['rho'].to("g/cm**3")[:, :, 0].T
    physical_magnetic_field_y = grid[('gas', 'magnetic_field_y')].to("uG")[:, :, 0].T
    physical_magnetic_field_x = grid[('gas', 'magnetic_field_x')].to("uG")[:, :, 0].T

    polarization_angle = np.arctan2(physical_magnetic_field_y, physical_magnetic_field_x)

    smoothening = 10
    mean_magnetic_x = gaussian_filter(physical_magnetic_field_x, smoothening)
    mean_magnetic_y = gaussian_filter(physical_magnetic_field_y, smoothening)

    mean_angle = np.arctan2(mean_magnetic_y, mean_magnetic_x)
    dispersion_angle = polarization_angle - mean_angle
    dispersion_angle = (dispersion_angle + np.pi/2) % np.pi - np.pi/2

    sigma_phi = np.sqrt(np.average(dispersion_angle**2, weights = density))

    return sigma_phi, density

def calculate_magnetic_field(grid, sigma_phi, sigma_velocity, density):
    physical_magnetic_field_y = grid[('gas', 'magnetic_field_y')].to("uG")[:, :, 0].T
    physical_magnetic_field_x = grid[('gas', 'magnetic_field_x')].to("uG")[:, :, 0].T

    mean_density = np.average(density)

    Q = 0.5

    B_pos_dcf = (Q * np.sqrt(4 * np.pi * mean_density) * (sigma_velocity / sigma_phi)).to("uG")
    B_pos_st = (np.sqrt(2 * np.pi * mean_density)* (sigma_velocity / np.sqrt(sigma_phi))).to("uG")
    B_pos_real = np.mean(np.sqrt(physical_magnetic_field_x**2 + physical_magnetic_field_y**2))

    return B_pos_dcf, B_pos_st, B_pos_real

contrasts = []

spacing = [108]

time_step = 0
time = []
size = 256
grid_x = np.linspace(0, 1, size)
grid_y = np.linspace(0, 1, size)

real = []

case_1 = []
case_2 = []
case_3 = []
case_4 = []

case_5 = []
case_6 = []
case_7 = []
case_8 = []

case_9 = []
case_10 = []
case_11 = []
case_12 = []

X,Y = np.meshgrid(grid_x, grid_y)

iso_sound_speed = 0.35

for space in spacing:
    # For wavelength 2L - I need 2-pc/1.58-km/s to get desired properties - 15uG, 1.58 km/s and 200 cm-3    ?
    # For wavelength L - I need 1-pc/1.0-km/s to get desired properties - 15uG, 1.58 km/s and 200 cm-3      ?
    # For wavelength L - I need 1-pc/1.0-km/s to get desired properties - 15uG, 1.58 km/s and 200 cm-3      ?
    unit_base={"length_unit": (1.0,"pc"), "time_unit": (1.0,"1.0 * pc / (0.35 * km/s)"), "mass_unit": (3.532e34,"g")}

    ds_256_1 = yt.load(f'./results_low_b_dens_grad_high/data_files_256/LinWave.out2.{space:05d}.athdf', units_override=unit_base)
    ds_256_2 = yt.load(f'./results_low_mid_b_dens_grad_high/data_files_256/LinWave.out2.{space:05d}.athdf', units_override=unit_base)
    ds_256_3 = yt.load(f'./results_mid_b_dens_grad_high/data_files_256/LinWave.out2.{space:05d}.athdf', units_override=unit_base)
    ds_256_4 = yt.load(f'./results_high_b_dens_grad_high/data_files_256/LinWave.out2.{space:05d}.athdf', units_override=unit_base)

    ds_256_5 = yt.load(f'./results_low_b_dens_grad_low/data_files_256/LinWave.out2.{space:05d}.athdf', units_override=unit_base)
    ds_256_6 = yt.load(f'./results_low_mid_b_dens_grad_low/data_files_256/LinWave.out2.{space:05d}.athdf', units_override=unit_base)
    ds_256_7 = yt.load(f'./results_mid_b_dens_grad_low/data_files_256/LinWave.out2.{space:05d}.athdf', units_override=unit_base)
    ds_256_8 = yt.load(f'./results_high_b_dens_grad_low/data_files_256/LinWave.out2.{space:05d}.athdf', units_override=unit_base)
    
    ds_256_9 = yt.load(f'./results_low_b_uniform_d/data_files_256/LinWave.out2.{space:05d}.athdf', units_override=unit_base)
    ds_256_10 = yt.load(f'./results_low_mid_b_uniform_d/data_files_256/LinWave.out2.{space:05d}.athdf', units_override=unit_base)
    ds_256_11 = yt.load(f'./results_mid_b_uniform_d/data_files_256/LinWave.out2.{space:05d}.athdf', units_override=unit_base)
    ds_256_12 = yt.load(f'./results_high_b_uniform_d/data_files_256/LinWave.out2.{space:05d}.athdf', units_override=unit_base)

    iso_speed = ds_256_1.quan(iso_sound_speed, "km/s")

    # time_evolved = ds_256.current_time.to("Myr")
    # time.append(time_evolved)
    # print(time_evolved)

    left_edge = ds_256_1.domain_left_edge
    right_edge = ds_256_1.domain_right_edge
    dims =  ds_256_1.domain_dimensions

    # print(left_edge, dims)

    data_256_1 = ds_256_1.covering_grid(level=0, left_edge=left_edge, dims=dims)
    data_256_2 = ds_256_2.covering_grid(level=0, left_edge=left_edge, dims=dims)
    data_256_3 = ds_256_3.covering_grid(level=0, left_edge=left_edge, dims=dims)
    data_256_4 = ds_256_4.covering_grid(level=0, left_edge=left_edge, dims=dims)

    data_256_5 = ds_256_5.covering_grid(level=0, left_edge=left_edge, dims=dims)
    data_256_6 = ds_256_6.covering_grid(level=0, left_edge=left_edge, dims=dims)
    data_256_7 = ds_256_7.covering_grid(level=0, left_edge=left_edge, dims=dims)
    data_256_8 = ds_256_8.covering_grid(level=0, left_edge=left_edge, dims=dims)

    data_256_9 = ds_256_9.covering_grid(level=0, left_edge=left_edge, dims=dims)
    data_256_10 = ds_256_10.covering_grid(level=0, left_edge=left_edge, dims=dims)
    data_256_11 = ds_256_11.covering_grid(level=0, left_edge=left_edge, dims=dims)
    data_256_12 = ds_256_12.covering_grid(level=0, left_edge=left_edge, dims=dims)

    sigma_phi_1, density_1 = get_polarization_dispersion(data_256_1)
    sigma_phi_2, density_2 = get_polarization_dispersion(data_256_2)
    sigma_phi_3, density_3 = get_polarization_dispersion(data_256_3)
    sigma_phi_4, density_4 = get_polarization_dispersion(data_256_4)

    sigma_phi_5, density_5 = get_polarization_dispersion(data_256_5)
    sigma_phi_6, density_6 = get_polarization_dispersion(data_256_6)
    sigma_phi_7, density_7 = get_polarization_dispersion(data_256_7)
    sigma_phi_8, density_8= get_polarization_dispersion(data_256_8)

    sigma_phi_9, density_9 = get_polarization_dispersion(data_256_9)
    sigma_phi_10, density_10 = get_polarization_dispersion(data_256_10)
    sigma_phi_11, density_11 = get_polarization_dispersion(data_256_11)
    sigma_phi_12, density_12 = get_polarization_dispersion(data_256_12)

    sigma_vel_1 = get_velocity_dispersion(data_256_1)
    sigma_vel_2 = get_velocity_dispersion(data_256_2)
    sigma_vel_3 = get_velocity_dispersion(data_256_3)
    sigma_vel_4 = get_velocity_dispersion(data_256_4)

    sigma_vel_5 = get_velocity_dispersion(data_256_5)
    sigma_vel_6 = get_velocity_dispersion(data_256_6)
    sigma_vel_7 = get_velocity_dispersion(data_256_7)
    sigma_vel_8 = get_velocity_dispersion(data_256_8)

    sigma_vel_9 = get_velocity_dispersion(data_256_9)
    sigma_vel_10 = get_velocity_dispersion(data_256_10)
    sigma_vel_11 = get_velocity_dispersion(data_256_11)
    sigma_vel_12 = get_velocity_dispersion(data_256_12)

    dcf_1, st_1, real_1 = calculate_magnetic_field(data_256_1, sigma_phi_1, sigma_vel_1, density_1)
    real.append(real_1)
    dcf_2, st_2, real_2 = calculate_magnetic_field(data_256_2, sigma_phi_2, sigma_vel_2, density_2)
    real.append(real_2)
    dcf_3, st_3, real_3 = calculate_magnetic_field(data_256_3, sigma_phi_3, sigma_vel_3, density_3)
    real.append(real_3)
    dcf_4, st_4, real_4 = calculate_magnetic_field(data_256_4, sigma_phi_4, sigma_vel_4, density_4)
    real.append(real_4)

    dcf_5, st_5, real_5 = calculate_magnetic_field(data_256_5, sigma_phi_5, sigma_vel_5, density_5)
    real.append(real_5)
    dcf_6, st_6, real_6 = calculate_magnetic_field(data_256_6, sigma_phi_6, sigma_vel_6, density_6)
    real.append(real_6)
    dcf_7, st_7, real_7 = calculate_magnetic_field(data_256_7, sigma_phi_7, sigma_vel_7, density_7)
    real.append(real_7)
    dcf_8, st_8, real_8 = calculate_magnetic_field(data_256_8, sigma_phi_8, sigma_vel_8, density_8)
    real.append(real_8)

    dcf_9, st_9, real_9 = calculate_magnetic_field(data_256_9, sigma_phi_9, sigma_vel_9, density_9)
    real.append(real_9)
    dcf_10, st_10, real_10 = calculate_magnetic_field(data_256_10, sigma_phi_10, sigma_vel_10, density_10)
    real.append(real_10)
    dcf_11, st_11, real_11 = calculate_magnetic_field(data_256_11, sigma_phi_11, sigma_vel_11, density_11)
    real.append(real_11)
    dcf_12, st_12, real_12 = calculate_magnetic_field(data_256_12, sigma_phi_12, sigma_vel_12, density_12)
    real.append(real_12)

    case_1.append(sigma_phi_1)
    case_2.append(sigma_phi_2)
    case_3.append(sigma_phi_3)
    case_4.append(sigma_phi_4)

    case_5.append(sigma_phi_5)
    case_6.append(sigma_phi_6)
    case_7.append(sigma_phi_7)
    case_8.append(sigma_phi_8)

    case_9.append(sigma_phi_9)
    case_10.append(sigma_phi_10)
    case_11.append(sigma_phi_11)
    case_12.append(sigma_phi_12)


    # number_density_low_b_high_g = (data_256_1['rho'].to("g/cm**3") / (2.34 * mh)).to("cm**-3")[:, :, 0].T
    # number_density_low_mid_b_high_g = (data_256_2['rho'].to("g/cm**3") / (2.34 * mh)).to("cm**-3")[:, :, 0].T
    # number_density_mid_b_high_g = (data_256_3['rho'].to("g/cm**3") / (2.34 * mh)).to("cm**-3")[:, :, 0].T
    # number_density_high_b_high_g = (data_256_4['rho'].to("g/cm**3") / (2.34 * mh)).to("cm**-3")[:, :, 0].T

    # number_density_low_b_low_g = (data_256_5['rho'].to("g/cm**3") / (2.34 * mh)).to("cm**-3")[:, :, 0].T
    # number_density_low_mid_b_low_g = (data_256_6['rho'].to("g/cm**3") / (2.34 * mh)).to("cm**-3")[:, :, 0].T
    # number_density_mid_b_low_g = (data_256_7['rho'].to("g/cm**3") / (2.34 * mh)).to("cm**-3")[:, :, 0].T
    # number_density_high_b_low_g = (data_256_8['rho'].to("g/cm**3") / (2.34 * mh)).to("cm**-3")[:, :, 0].T

    # number_density_low_b_uni_g = (data_256_9['rho'].to("g/cm**3") / (2.34 * mh)).to("cm**-3")[:, :, 0].T
    # number_density_low_mid_b_uni_g = (data_256_10['rho'].to("g/cm**3") / (2.34 * mh)).to("cm**-3")[:, :, 0].T
    # number_density_mid_b_uni_g = (data_256_11['rho'].to("g/cm**3") / (2.34 * mh)).to("cm**-3")[:, :, 0].T
    # number_density_high_b_uni_g = (data_256_12['rho'].to("g/cm**3") / (2.34 * mh)).to("cm**-3")[:, :, 0].T

    alfven_speed_low_b_high_g = np.mean(data_256_1[('gas', 'alfven_speed')].to("km/s")[:, : , 0].T)
    alfven_speed_low_mid_b_high_g = np.mean(data_256_2[('gas', 'alfven_speed')].to("km/s")[:, : , 0].T)
    alfven_speed_mid_b_high_g = np.mean(data_256_3[('gas', 'alfven_speed')].to("km/s")[:, : , 0].T)
    alfven_speed_high_b_high_g = np.mean(data_256_4[('gas', 'alfven_speed')].to("km/s")[:, : , 0].T)

    alfven_speed_low_b_low_g = np.mean(data_256_5[('gas', 'alfven_speed')].to("km/s")[:, : , 0].T)
    alfven_speed_low_mid_b_low_g = np.mean(data_256_6[('gas', 'alfven_speed')].to("km/s")[:, : , 0].T)
    alfven_speed_mid_b_low_g = np.mean(data_256_7[('gas', 'alfven_speed')].to("km/s")[:, : , 0].T)
    alfven_speed_high_b_low_g = np.mean(data_256_8[('gas', 'alfven_speed')].to("km/s")[:, : , 0].T)

    alfven_speed_low_b_uni_g = np.mean(data_256_9[('gas', 'alfven_speed')].to("km/s")[:, : , 0].T)
    alfven_speed_low_mid_b_uni_g = np.mean(data_256_10[('gas', 'alfven_speed')].to("km/s")[:, : , 0].T)
    alfven_speed_mid_b_uni_g = np.mean(data_256_11[('gas', 'alfven_speed')].to("km/s")[:, : , 0].T)
    alfven_speed_high_b_uni_g = np.mean(data_256_12[('gas', 'alfven_speed')].to("km/s")[:, : , 0].T)

    # n_1 = np.mean(number_density_low_b_high_g[0])
    # n_2 = np.mean(number_density_low_mid_b_high_g[0])
    # n_3 = np.mean(number_density_mid_b_high_g[0])
    # n_4 = np.mean(number_density_high_b_high_g[0])

    # n_5 = np.mean(number_density_low_b_low_g[0])
    # n_6 = np.mean(number_density_low_mid_b_low_g[0])
    # n_7 = np.mean(number_density_mid_b_low_g[0])
    # n_8 = np.mean(number_density_high_b_low_g[0])

    # n_9 = np.mean(number_density_low_b_uni_g[0])
    # n_10 = np.mean(number_density_low_mid_b_uni_g[0])
    # n_11 = np.mean(number_density_mid_b_uni_g[0])
    # n_12 = np.mean(number_density_high_b_uni_g[0])

    # for x in range(16, size, 16):
    #     averaged_value_1 = ((number_density_low_b_high_g[x-1] + number_density_low_b_high_g[x] + number_density_low_b_high_g[x+1]) / 3.0) / n_1
    #     averaged_value_2 = ((number_density_low_mid_b_high_g[x-1] + number_density_low_mid_b_high_g[x] + number_density_low_mid_b_high_g[x+1]) / 3.0) / n_2
    #     averaged_value_3 = ((number_density_mid_b_high_g[x-1] + number_density_mid_b_high_g[x] + number_density_mid_b_high_g[x+1]) / 3.0) / n_4
    #     averaged_value_4 = ((number_density_high_b_high_g[x-1] + number_density_high_b_high_g[x] + number_density_high_b_high_g[x+1]) / 3.0) / n_3

    #     averaged_value_5 = ((number_density_low_b_low_g[x-1] + number_density_low_b_low_g[x] + number_density_low_b_low_g[x+1]) / 3.0) / n_5
    #     averaged_value_6 = ((number_density_low_mid_b_low_g[x-1] + number_density_low_mid_b_low_g[x] + number_density_low_mid_b_low_g[x+1]) / 3.0) / n_6
    #     averaged_value_7 = ((number_density_mid_b_low_g[x-1] + number_density_mid_b_low_g[x] + number_density_mid_b_low_g[x+1]) / 3.0) / n_7
    #     averaged_value_8 = ((number_density_high_b_low_g[x-1] + number_density_high_b_low_g[x] + number_density_high_b_low_g[x+1]) / 3.0) / n_8

    #     averaged_value_9 = ((number_density_low_b_uni_g[x-1] + number_density_low_b_uni_g[x] + number_density_low_b_uni_g[x+1]) / 3.0) / n_9
    #     averaged_value_10 = ((number_density_low_mid_b_uni_g[x-1] + number_density_low_mid_b_uni_g[x] + number_density_low_mid_b_uni_g[x+1]) / 3.0) / n_10
    #     averaged_value_11 = ((number_density_mid_b_uni_g[x-1] + number_density_mid_b_uni_g[x] + number_density_mid_b_uni_g[x+1]) / 3.0) / n_11
    #     averaged_value_12 = ((number_density_high_b_uni_g[x-1] + number_density_high_b_uni_g[x] + number_density_high_b_uni_g[x+1]) / 3.0) / n_12

    #     # contrast_1 = np.abs(np.max(averaged_value_1) - np.min(averaged_value_1)) * 100 / np.mean(averaged_value_1)
    #     # contrast_2 = np.abs(np.max(averaged_value_2) - np.min(averaged_value_2)) * 100 / np.mean(averaged_value_2)
    #     # contrast_3 = np.abs(np.max(averaged_value_3) - np.min(averaged_value_3)) * 100 / np.mean(averaged_value_3)

    #     # contrast_4 = np.abs(np.max(averaged_value_4) - np.min(averaged_value_4)) * 100 / np.mean(averaged_value_4)
    #     # contrast_5 = np.abs(np.max(averaged_value_5) - np.min(averaged_value_5)) * 100 / np.mean(averaged_value_5)
    #     # contrast_6 = np.abs(np.max(averaged_value_6) - np.min(averaged_value_6)) * 100 / np.mean(averaged_value_6)

    #     # contrast_7 = np.abs(np.max(averaged_value_7) - np.min(averaged_value_7)) * 100 / np.mean(averaged_value_7)
    #     # contrast_8 = np.abs(np.max(averaged_value_8) - np.min(averaged_value_8)) * 100 / np.mean(averaged_value_8)
    #     # contrast_9 = np.abs(np.max(averaged_value_9) - np.min(averaged_value_9)) * 100 / np.mean(averaged_value_9)

    #     contrast_1 = compute_mean_contrast(averaged_value_1)
    #     contrast_2 = compute_mean_contrast(averaged_value_2)
    #     contrast_3 = compute_mean_contrast(averaged_value_3)
    #     contrast_4 = compute_mean_contrast(averaged_value_4)

    #     contrast_5 = compute_mean_contrast(averaged_value_5)
    #     contrast_6 = compute_mean_contrast(averaged_value_6)
    #     contrast_7 = compute_mean_contrast(averaged_value_7)
    #     contrast_8 = compute_mean_contrast(averaged_value_8)

    #     contrast_9 = compute_mean_contrast(averaged_value_9)
    #     contrast_10 = compute_mean_contrast(averaged_value_10)
    #     contrast_11 = compute_mean_contrast(averaged_value_11)
    #     contrast_12 = compute_mean_contrast(averaged_value_12)

    

plt.figure(figsize=(6, 4), dpi=300)  # High-resolution figure

# Plot 3 cases with consistent styling
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
colors = ['black', '0.3', '0.5', '0.7', '0.85']
plt.plot(real, real, color=colors[1], lw=0.4, ls=linestyles[0], label=r'$\rm B_{simulation}$')

plt.scatter(real_1, dcf_1, color="tab:blue", marker="o", label=r'$\rm B_{dcf}$')
plt.scatter(real_2, dcf_2,  color="tab:blue", marker="o")
plt.scatter(real_3, dcf_3,  color="tab:blue", marker="o")
plt.scatter(real_4, dcf_4, color="tab:blue", marker="o")

plt.scatter(real_1, st_1, color="tab:green", marker="*", label=r'$\rm B_{st}$')
plt.scatter(real_2, st_2,  color="tab:green", marker="*")
plt.scatter(real_3, st_3,  color="tab:green", marker="*")
plt.scatter(real_4, st_4, color="tab:green", marker="*")

# plt.scatter(real_5, dcf_5, color="black", marker="*", label=f'weak B, g = 0.1')
# plt.scatter(real_6, dcf_6, color="tab:blue", marker="*", label=f'weak-mid B, g = 0.1')
# plt.scatter(real_7, dcf_7,  color="tab:green", marker="*", label=f'mid B, g = 0.1')
# plt.scatter(real_8, dcf_8, color="tab:red", marker="*", label=f'strong B, g = 0.1')

# plt.scatter(real_9, dcf_9, color="black", marker="+", label=f'weak B, uniform')
# plt.scatter(real_10, dcf_10,  color="tab:blue", marker="+", label=f'weak-mid B, uniform')
# plt.scatter(real_11, dcf_11,  color="tab:green", marker="+", label=f'mid B, uniform')
# plt.scatter(real_12, dcf_12, color="tab:red", marker="+", label=f'strong B, uniform')

# Labels
plt.xlabel(r"$\rm B_{simulation} \, [\mu G]$", fontsize=8)
plt.ylabel(r"$\rm B_{predicted} \, [\mu G]$", fontsize=8)

# Ticks
plt.tick_params(axis='both', which='major', labelsize=10, direction="in", length=4)
plt.tick_params(axis='both', which='minor', direction="in", length=2)
plt.minorticks_on()

# Axes border
# for spine in ['top', 'right']:
#     plt.gca().spines[spine].set_visible(False)

# Legend
plt.legend(frameon=True, fontsize=5, loc='upper left', bbox_to_anchor=(1.02, 1.0), markerscale=0.6)
plt.savefig("comparing_magnetic_fields_with_simulation_3Myr.pdf", bbox_inches="tight")

# Layout
plt.tight_layout()
plt.show()
