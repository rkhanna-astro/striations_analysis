import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import yt
from yt.units import mh

# def compute_mean_contrast(number_density_2d, cut_axis='x', smoothing_sigma=2.0):

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


contrasts = []

spacing = [108]

time_step = 0
time = []
size = 256
grid_x = np.linspace(0, 1, size)
grid_y = np.linspace(0, 1, size)

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

    number_density_low_b_high_g = (data_256_1['rho'].to("g/cm**3") / (2.34 * mh)).to("cm**-3")[:, :, 0].T
    number_density_low_mid_b_high_g = (data_256_2['rho'].to("g/cm**3") / (2.34 * mh)).to("cm**-3")[:, :, 0].T
    number_density_mid_b_high_g = (data_256_3['rho'].to("g/cm**3") / (2.34 * mh)).to("cm**-3")[:, :, 0].T
    number_density_high_b_high_g = (data_256_4['rho'].to("g/cm**3") / (2.34 * mh)).to("cm**-3")[:, :, 0].T

    number_density_low_b_low_g = (data_256_5['rho'].to("g/cm**3") / (2.34 * mh)).to("cm**-3")[:, :, 0].T
    number_density_low_mid_b_low_g = (data_256_6['rho'].to("g/cm**3") / (2.34 * mh)).to("cm**-3")[:, :, 0].T
    number_density_mid_b_low_g = (data_256_7['rho'].to("g/cm**3") / (2.34 * mh)).to("cm**-3")[:, :, 0].T
    number_density_high_b_low_g = (data_256_8['rho'].to("g/cm**3") / (2.34 * mh)).to("cm**-3")[:, :, 0].T

    number_density_low_b_uni_g = (data_256_9['rho'].to("g/cm**3") / (2.34 * mh)).to("cm**-3")[:, :, 0].T
    number_density_low_mid_b_uni_g = (data_256_10['rho'].to("g/cm**3") / (2.34 * mh)).to("cm**-3")[:, :, 0].T
    number_density_mid_b_uni_g = (data_256_11['rho'].to("g/cm**3") / (2.34 * mh)).to("cm**-3")[:, :, 0].T
    number_density_high_b_uni_g = (data_256_12['rho'].to("g/cm**3") / (2.34 * mh)).to("cm**-3")[:, :, 0].T

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

    n_1 = np.mean(number_density_low_b_high_g[0])
    n_2 = np.mean(number_density_low_mid_b_high_g[0])
    n_3 = np.mean(number_density_mid_b_high_g[0])
    n_4 = np.mean(number_density_high_b_high_g[0])

    n_5 = np.mean(number_density_low_b_low_g[0])
    n_6 = np.mean(number_density_low_mid_b_low_g[0])
    n_7 = np.mean(number_density_mid_b_low_g[0])
    n_8 = np.mean(number_density_high_b_low_g[0])

    n_9 = np.mean(number_density_low_b_uni_g[0])
    n_10 = np.mean(number_density_low_mid_b_uni_g[0])
    n_11 = np.mean(number_density_mid_b_uni_g[0])
    n_12 = np.mean(number_density_high_b_uni_g[0])

    for x in range(16, size, 16):
        averaged_value_1 = ((number_density_low_b_high_g[x-1] + number_density_low_b_high_g[x] + number_density_low_b_high_g[x+1]) / 3.0) / n_1
        averaged_value_2 = ((number_density_low_mid_b_high_g[x-1] + number_density_low_mid_b_high_g[x] + number_density_low_mid_b_high_g[x+1]) / 3.0) / n_2
        averaged_value_3 = ((number_density_mid_b_high_g[x-1] + number_density_mid_b_high_g[x] + number_density_mid_b_high_g[x+1]) / 3.0) / n_4
        averaged_value_4 = ((number_density_high_b_high_g[x-1] + number_density_high_b_high_g[x] + number_density_high_b_high_g[x+1]) / 3.0) / n_3

        averaged_value_5 = ((number_density_low_b_low_g[x-1] + number_density_low_b_low_g[x] + number_density_low_b_low_g[x+1]) / 3.0) / n_5
        averaged_value_6 = ((number_density_low_mid_b_low_g[x-1] + number_density_low_mid_b_low_g[x] + number_density_low_mid_b_low_g[x+1]) / 3.0) / n_6
        averaged_value_7 = ((number_density_mid_b_low_g[x-1] + number_density_mid_b_low_g[x] + number_density_mid_b_low_g[x+1]) / 3.0) / n_7
        averaged_value_8 = ((number_density_high_b_low_g[x-1] + number_density_high_b_low_g[x] + number_density_high_b_low_g[x+1]) / 3.0) / n_8

        averaged_value_9 = ((number_density_low_b_uni_g[x-1] + number_density_low_b_uni_g[x] + number_density_low_b_uni_g[x+1]) / 3.0) / n_9
        averaged_value_10 = ((number_density_low_mid_b_uni_g[x-1] + number_density_low_mid_b_uni_g[x] + number_density_low_mid_b_uni_g[x+1]) / 3.0) / n_10
        averaged_value_11 = ((number_density_mid_b_uni_g[x-1] + number_density_mid_b_uni_g[x] + number_density_mid_b_uni_g[x+1]) / 3.0) / n_11
        averaged_value_12 = ((number_density_high_b_uni_g[x-1] + number_density_high_b_uni_g[x] + number_density_high_b_uni_g[x+1]) / 3.0) / n_12

        # contrast_1 = np.abs(np.max(averaged_value_1) - np.min(averaged_value_1)) * 100 / np.mean(averaged_value_1)
        # contrast_2 = np.abs(np.max(averaged_value_2) - np.min(averaged_value_2)) * 100 / np.mean(averaged_value_2)
        # contrast_3 = np.abs(np.max(averaged_value_3) - np.min(averaged_value_3)) * 100 / np.mean(averaged_value_3)

        # contrast_4 = np.abs(np.max(averaged_value_4) - np.min(averaged_value_4)) * 100 / np.mean(averaged_value_4)
        # contrast_5 = np.abs(np.max(averaged_value_5) - np.min(averaged_value_5)) * 100 / np.mean(averaged_value_5)
        # contrast_6 = np.abs(np.max(averaged_value_6) - np.min(averaged_value_6)) * 100 / np.mean(averaged_value_6)

        # contrast_7 = np.abs(np.max(averaged_value_7) - np.min(averaged_value_7)) * 100 / np.mean(averaged_value_7)
        # contrast_8 = np.abs(np.max(averaged_value_8) - np.min(averaged_value_8)) * 100 / np.mean(averaged_value_8)
        # contrast_9 = np.abs(np.max(averaged_value_9) - np.min(averaged_value_9)) * 100 / np.mean(averaged_value_9)

        contrast_1 = compute_mean_contrast(averaged_value_1)
        contrast_2 = compute_mean_contrast(averaged_value_2)
        contrast_3 = compute_mean_contrast(averaged_value_3)
        contrast_4 = compute_mean_contrast(averaged_value_4)

        contrast_5 = compute_mean_contrast(averaged_value_5)
        contrast_6 = compute_mean_contrast(averaged_value_6)
        contrast_7 = compute_mean_contrast(averaged_value_7)
        contrast_8 = compute_mean_contrast(averaged_value_8)

        contrast_9 = compute_mean_contrast(averaged_value_9)
        contrast_10 = compute_mean_contrast(averaged_value_10)
        contrast_11 = compute_mean_contrast(averaged_value_11)
        contrast_12 = compute_mean_contrast(averaged_value_12)

        case_1.append(contrast_1)
        case_2.append(contrast_2)
        case_3.append(contrast_3)
        case_4.append(contrast_4)

        case_5.append(contrast_5)
        case_6.append(contrast_6)
        case_7.append(contrast_7)
        case_8.append(contrast_8)

        case_9.append(contrast_9)
        case_10.append(contrast_10)
        case_11.append(contrast_11)
        case_12.append(contrast_12)

plt.figure(figsize=(6, 4), dpi=300)  # High-resolution figure

# Plot 3 cases with consistent styling
plt.scatter(alfven_speed_low_b_high_g / iso_speed, np.mean(case_1), color="black", marker="o", label=f'low B, g = 1.0')
plt.scatter(alfven_speed_low_mid_b_high_g / iso_speed, np.mean(case_2),  color="tab:blue", marker="o", label=f'low mid B, g = 1.0')
plt.scatter(alfven_speed_mid_b_high_g / iso_speed, np.mean(case_3),  color="tab:green", marker="o", label=f'mid B, g = 1.0')
plt.scatter(alfven_speed_high_b_high_g / iso_speed, np.mean(case_4), color="tab:red", marker="o", label=f'high B, g = 1.0')

plt.scatter(alfven_speed_low_b_low_g / iso_speed, np.mean(case_5), color="black", marker="*", label=f'low B, g = 0.1')
plt.scatter(alfven_speed_low_mid_b_low_g / iso_speed, np.mean(case_6), color="tab:blue", marker="*", label=f'low mid B, g = 0.1')
plt.scatter(alfven_speed_mid_b_low_g / iso_speed, np.mean(case_7),  color="tab:green", marker="*", label=f'mid B, g = 0.1')
plt.scatter(alfven_speed_high_b_low_g / iso_speed, np.mean(case_8), color="tab:red", marker="*", label=f'high B, g = 0.1')

plt.scatter(alfven_speed_low_b_uni_g / iso_speed, np.mean(case_9), color="black", marker="+", label=f'low B, uniform')
plt.scatter(alfven_speed_low_mid_b_uni_g / iso_speed, np.mean(case_10),  color="tab:blue", marker="+", label=f'low mid B, uniform')
plt.scatter(alfven_speed_mid_b_uni_g / iso_speed, np.mean(case_11),  color="tab:green", marker="+", label=f'mid B, uniform')
plt.scatter(alfven_speed_high_b_uni_g / iso_speed, np.mean(case_12), color="tab:red", marker="+", label=f'high B, uniform')

# Labels
plt.xlabel(r"$v_a / c_s$", fontsize=8)
plt.ylabel(r"$max \, density \, contrast \, (\%)$", fontsize=8)

# Ticks
plt.tick_params(axis='both', which='major', labelsize=10, direction="in", length=4)
plt.tick_params(axis='both', which='minor', direction="in", length=2)
plt.minorticks_on()

# Axes border
# for spine in ['top', 'right']:
#     plt.gca().spines[spine].set_visible(False)

# Legend
plt.legend(frameon=True, fontsize=5, loc='upper left', bbox_to_anchor=(1.02, 1.0), markerscale=0.6)
plt.savefig("all_mean_density_contrast_all_3Myr.pdf", bbox_inches="tight")

# Layout
plt.tight_layout()
plt.show()
