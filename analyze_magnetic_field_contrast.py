import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import yt
from yt.units import mh

# def compute_mean_contrast(magneitc_field_2d, cut_axis='x', smoothing_sigma=2.0):

def compute_mean_contrast(smooth_profile):
    peaks, _ = find_peaks(smooth_profile, distance=256)
    troughs, _ = find_peaks(-smooth_profile, distance=256)
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
case_a = []

case_4 = []
case_5 = []
case_6 = []
case_b = []

case_7 = []
case_8 = []
case_9 = []
case_c = []

X,Y = np.meshgrid(grid_x, grid_y)

iso_sound_speed = 0.35

for space in spacing:
    # For wavelength 2L - I need 2-pc/1.58-km/s to get desired properties - 15uG, 1.58 km/s and 200 cm-3    ?
    # For wavelength L - I need 1-pc/1.0-km/s to get desired properties - 15uG, 1.58 km/s and 200 cm-3      ?
    # For wavelength L - I need 1-pc/1.0-km/s to get desired properties - 15uG, 1.58 km/s and 200 cm-3      ?
    unit_base={"length_unit": (1.0,"pc"), "time_unit": (1.0,"1.0 * pc / (0.35 * km/s)"), "mass_unit": (3.532e34,"g")}

    ds_256_1 = yt.load(f'./results_low_b_dens_grad_high/data_files_256/LinWave.out2.{space:05d}.athdf', units_override=unit_base)
    ds_256_2 = yt.load(f'./results_low_mid_b_dens_grad_high/data_files_256/LinWave.out2.{space:05d}.athdf', units_override=unit_base)
    ds_256_3 = yt.load(f'./results_high_b_dens_grad_high/data_files_256/LinWave.out2.{space:05d}.athdf', units_override=unit_base)
    ds_256_a = yt.load(f'./results_mid_b_dens_grad_high/data_files_256/LinWave.out2.{space:05d}.athdf', units_override=unit_base)

    ds_256_4 = yt.load(f'./results_low_b_dens_grad_low/data_files_256/LinWave.out2.{space:05d}.athdf', units_override=unit_base)
    ds_256_5 = yt.load(f'./results_low_mid_b_dens_grad_low/data_files_256/LinWave.out2.{space:05d}.athdf', units_override=unit_base)
    ds_256_6 = yt.load(f'./results_high_b_dens_grad_low/data_files_256/LinWave.out2.{space:05d}.athdf', units_override=unit_base)
    ds_256_b = yt.load(f'./results_mid_b_dens_grad_low/data_files_256/LinWave.out2.{space:05d}.athdf', units_override=unit_base)

    ds_256_7 = yt.load(f'./results_low_b_uniform_d/data_files_256/LinWave.out2.{space:05d}.athdf', units_override=unit_base)
    ds_256_8 = yt.load(f'./results_low_mid_b_uniform_d/data_files_256/LinWave.out2.{space:05d}.athdf', units_override=unit_base)
    ds_256_9 = yt.load(f'./results_high_b_uniform_d/data_files_256/LinWave.out2.{space:05d}.athdf', units_override=unit_base)
    ds_256_c = yt.load(f'./results_mid_b_uniform_d/data_files_256/LinWave.out2.{space:05d}.athdf', units_override=unit_base)

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
    data_256_a = ds_256_a.covering_grid(level=0, left_edge=left_edge, dims=dims)

    data_256_4 = ds_256_4.covering_grid(level=0, left_edge=left_edge, dims=dims)
    data_256_5 = ds_256_5.covering_grid(level=0, left_edge=left_edge, dims=dims)
    data_256_6 = ds_256_6.covering_grid(level=0, left_edge=left_edge, dims=dims)
    data_256_b = ds_256_b.covering_grid(level=0, left_edge=left_edge, dims=dims)

    data_256_7 = ds_256_7.covering_grid(level=0, left_edge=left_edge, dims=dims)
    data_256_8 = ds_256_8.covering_grid(level=0, left_edge=left_edge, dims=dims)
    data_256_9 = ds_256_9.covering_grid(level=0, left_edge=left_edge, dims=dims)
    data_256_c = ds_256_c.covering_grid(level=0, left_edge=left_edge, dims=dims)

    magneitc_field_low_b_high_g = (data_256_1[('gas', 'magnetic_field_y')].to("uG"))[:, :, 0].T
    magneitc_field_mid_b_high_g = (data_256_2[('gas', 'magnetic_field_y')].to("uG"))[:, :, 0].T
    magneitc_field_high_b_high_g = (data_256_3[('gas', 'magnetic_field_y')].to("uG"))[:, :, 0].T
    magneitc_field_mid_high_b_high_g = (data_256_a[('gas', 'magnetic_field_y')].to("uG"))[:, :, 0].T

    magneitc_field_low_b_low_g = (data_256_4[('gas', 'magnetic_field_y')].to("uG"))[:, :, 0].T
    magneitc_field_mid_b_low_g = (data_256_5[('gas', 'magnetic_field_y')].to("uG"))[:, :, 0].T
    magneitc_field_high_b_low_g = (data_256_6[('gas', 'magnetic_field_y')].to("uG"))[:, :, 0].T
    magneitc_field_mid_high_b_low_g = (data_256_b[('gas', 'magnetic_field_y')].to("uG"))[:, :, 0].T

    magneitc_field_low_b_uni_g = (data_256_7[('gas', 'magnetic_field_y')].to("uG"))[:, :, 0].T
    magneitc_field_mid_b_uni_g = (data_256_8[('gas', 'magnetic_field_y')].to("uG"))[:, :, 0].T
    magneitc_field_high_b_uni_g = (data_256_9[('gas', 'magnetic_field_y')].to("uG"))[:, :, 0].T
    magneitc_field_mid_high_b_uni_g = (data_256_c[('gas', 'magnetic_field_y')].to("uG"))[:, :, 0].T

    alfven_speed_low_b_high_g = np.mean(data_256_1[('gas', 'alfven_speed')].to("km/s")[:, : , 0].T)
    alfven_speed_mid_b_high_g = np.mean(data_256_2[('gas', 'alfven_speed')].to("km/s")[:, : , 0].T)
    alfven_speed_high_b_high_g = np.mean(data_256_3[('gas', 'alfven_speed')].to("km/s")[:, : , 0].T)
    alfven_speed_mid_high_b_high_g = np.mean(data_256_a[('gas', 'alfven_speed')].to("km/s")[:, : , 0].T)

    alfven_speed_low_b_low_g = np.mean(data_256_4[('gas', 'alfven_speed')].to("km/s")[:, : , 0].T)
    alfven_speed_mid_b_low_g = np.mean(data_256_5[('gas', 'alfven_speed')].to("km/s")[:, : , 0].T)
    alfven_speed_high_b_low_g = np.mean(data_256_6[('gas', 'alfven_speed')].to("km/s")[:, : , 0].T)
    alfven_speed_mid_high_b_low_g = np.mean(data_256_b[('gas', 'alfven_speed')].to("km/s")[:, : , 0].T)

    alfven_speed_low_b_uni_g = np.mean(data_256_7[('gas', 'alfven_speed')].to("km/s")[:, : , 0].T)
    alfven_speed_mid_b_uni_g = np.mean(data_256_8[('gas', 'alfven_speed')].to("km/s")[:, : , 0].T)
    alfven_speed_high_b_uni_g = np.mean(data_256_9[('gas', 'alfven_speed')].to("km/s")[:, : , 0].T)
    alfven_speed_mid_high_b_uni_g = np.mean(data_256_c[('gas', 'alfven_speed')].to("km/s")[:, : , 0].T)

    n_1 = np.mean(magneitc_field_low_b_high_g[0])
    n_2 = np.mean(magneitc_field_mid_b_high_g[0])
    n_3 = np.mean(magneitc_field_high_b_high_g[0])
    n_a = np.mean(magneitc_field_mid_high_b_high_g[0])

    n_4 = np.mean(magneitc_field_low_b_low_g[0])
    n_5 = np.mean(magneitc_field_mid_b_low_g[0])
    n_6 = np.mean(magneitc_field_high_b_low_g[0])
    n_b = np.mean(magneitc_field_mid_high_b_low_g[0])

    n_7 = np.mean(magneitc_field_low_b_uni_g[0])
    n_8 = np.mean(magneitc_field_mid_b_uni_g[0])
    n_9 = np.mean(magneitc_field_high_b_uni_g[0])
    n_c = np.mean(magneitc_field_mid_high_b_uni_g[0])

    for x in range(16, size, 16):
        averaged_value_1 = ((magneitc_field_low_b_high_g[x-1] + magneitc_field_low_b_high_g[x] + magneitc_field_low_b_high_g[x+1]) / 3.0) / n_1
        averaged_value_2 = ((magneitc_field_mid_b_high_g[x-1] + magneitc_field_mid_b_high_g[x] + magneitc_field_mid_b_high_g[x+1]) / 3.0) / n_2
        averaged_value_3 = ((magneitc_field_high_b_high_g[x-1] + magneitc_field_high_b_high_g[x] + magneitc_field_high_b_high_g[x+1]) / 3.0) / n_3
        averaged_value_a = ((magneitc_field_mid_high_b_high_g[x-1] + magneitc_field_mid_high_b_high_g[x] + magneitc_field_mid_high_b_high_g[x+1]) / 3.0) / n_a

        averaged_value_4 = ((magneitc_field_low_b_low_g[x-1] + magneitc_field_low_b_low_g[x] + magneitc_field_low_b_low_g[x+1]) / 3.0) / n_4
        averaged_value_5 = ((magneitc_field_mid_b_low_g[x-1] + magneitc_field_mid_b_low_g[x] + magneitc_field_mid_b_low_g[x+1]) / 3.0) / n_5
        averaged_value_6 = ((magneitc_field_high_b_low_g[x-1] + magneitc_field_high_b_low_g[x] + magneitc_field_high_b_low_g[x+1]) / 3.0) / n_6
        averaged_value_b = ((magneitc_field_mid_high_b_low_g[x-1] + magneitc_field_mid_high_b_low_g[x] + magneitc_field_mid_high_b_low_g[x+1]) / 3.0) / n_b

        averaged_value_7 = ((magneitc_field_low_b_uni_g[x-1] + magneitc_field_low_b_uni_g[x] + magneitc_field_low_b_uni_g[x+1]) / 3.0) / n_7
        averaged_value_8 = ((magneitc_field_mid_b_uni_g[x-1] + magneitc_field_mid_b_uni_g[x] + magneitc_field_mid_b_uni_g[x+1]) / 3.0) / n_8
        averaged_value_9 = ((magneitc_field_high_b_uni_g[x-1] + magneitc_field_high_b_uni_g[x] + magneitc_field_high_b_uni_g[x+1]) / 3.0) / n_9
        averaged_value_c = ((magneitc_field_mid_high_b_uni_g[x-1] + magneitc_field_mid_high_b_uni_g[x] + magneitc_field_mid_high_b_uni_g[x+1]) / 3.0) / n_c

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
        contrast_a = compute_mean_contrast(averaged_value_a)

        contrast_4 = compute_mean_contrast(averaged_value_4)
        contrast_5 = compute_mean_contrast(averaged_value_5)
        contrast_6 = compute_mean_contrast(averaged_value_6)
        contrast_b = compute_mean_contrast(averaged_value_b)

        contrast_7 = compute_mean_contrast(averaged_value_7)
        contrast_8 = compute_mean_contrast(averaged_value_8)
        contrast_9 = compute_mean_contrast(averaged_value_9)
        contrast_c = compute_mean_contrast(averaged_value_c)

        case_1.append(contrast_1)
        case_2.append(contrast_2)
        case_3.append(contrast_3)
        case_a.append(contrast_a)

        case_4.append(contrast_4)
        case_5.append(contrast_5)
        case_6.append(contrast_6)
        case_b.append(contrast_a)

        case_7.append(contrast_7)
        case_8.append(contrast_8)
        case_9.append(contrast_9)
        case_c.append(contrast_c)

plt.figure(figsize=(6, 4), dpi=300)  # High-resolution figure

# Plot 3 cases with consistent styling
plt.scatter(alfven_speed_low_b_high_g / iso_speed, np.mean(case_1), color="black", marker="o", label=f'low B, g = 1.0')
plt.scatter(alfven_speed_mid_b_high_g / iso_speed, np.mean(case_2),  color="tab:blue", marker="o", label=f'low mid B, g = 1.0')
plt.scatter(alfven_speed_high_b_high_g / iso_speed, np.mean(case_3), color="tab:red", marker="o", label=f'high B, g = 1.0')
plt.scatter(alfven_speed_mid_high_b_high_g / iso_speed, np.mean(case_a), color="tab:green", marker="o", label=f'mid B, g = 1.0')

plt.scatter(alfven_speed_low_b_low_g / iso_speed, np.mean(case_4), color="black", marker="*", label=f'low B, g = 0.1')
plt.scatter(alfven_speed_mid_b_low_g / iso_speed, np.mean(case_5),  color="tab:blue", marker="*", label=f'low mid B, g = 0.1')
plt.scatter(alfven_speed_high_b_low_g / iso_speed, np.mean(case_6), color="tab:red", marker="*", label=f'high B, g = 0.1')
plt.scatter(alfven_speed_mid_high_b_low_g / iso_speed, np.mean(case_b), color="tab:green", marker="*", label=f'mid B, g = 0.1')

plt.scatter(alfven_speed_low_b_uni_g / iso_speed, np.mean(case_7), color="black", marker="+", label=f'low B, uniform')
plt.scatter(alfven_speed_mid_b_uni_g / iso_speed, np.mean(case_8),  color="tab:blue", marker="+", label=f'low mid B, uniform')
plt.scatter(alfven_speed_high_b_uni_g / iso_speed, np.mean(case_9), color="tab:red", marker="+", label=f'high B, uniform')
plt.scatter(alfven_speed_mid_high_b_uni_g / iso_speed, np.mean(case_c), color="tab:green", marker="+", label=f'mid B, uniform')

# Labels
plt.xlabel(r"$v_a / c_s$", fontsize=8)
plt.ylabel(r"$max \, \, B_y \, \, contrast \, \, (\%)$", fontsize=8)

# Ticks
plt.tick_params(axis='both', which='major', labelsize=10, direction="in", length=4)
plt.tick_params(axis='both', which='minor', direction="in", length=2)
plt.minorticks_on()

# Axes border
# for spine in ['top', 'right']:
#     plt.gca().spines[spine].set_visible(False)

# Legend
plt.legend(frameon=True, fontsize=5, loc='upper left', bbox_to_anchor=(1.02, 1.0), markerscale=0.6)
plt.savefig("maxz_B_y_contrast_all_3Myr.pdf", bbox_inches="tight")

# Layout
plt.tight_layout()
plt.show()
