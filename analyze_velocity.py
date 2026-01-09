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

spacing = np.linspace(0, 500, 500, dtype=int)

time = []
mag_energy = np.array([])
kin_energy = np.array([])

grid = 256
grid_x = np.linspace(0, 1, grid)
grid_y = np.linspace(0, 1, grid)

X,Y = np.meshgrid(grid_x, grid_y)

for space in spacing:
    # For wavelength 2L - I need 2-pc/1.58-km/s to get desired properties - 15uG, 1.58 km/s and 200 cm-3    ?
    # For wavelength L - I need 1-pc/1.0-km/s to get desired properties - 15uG, 1.58 km/s and 200 cm-3      ?
    # For wavelength L - I need 1-pc/1.0-km/s to get desired properties - 15uG, 1.58 km/s and 200 cm-3      ?
    unit_base={"length_unit": (1.0,"pc"), "time_unit": (1.0,"1.0 * pc / (0.35 * km/s)"), "mass_unit": (3.532e34,"g")}

    ds_256 = yt.load(f'./results_low_mid_b_dens_grad_high/data_files_256/LinWave.out2.{space:05d}.athdf', units_override=unit_base)

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

    velocity_x = data_256[('gas', 'velocity_x')].to("km/s")[:, :, 0].T

    # for x in range(16, size, 16):
    #         averaged_value = (density[x-1] + density[x] + density[x+1]) / 3.0
    #         averaged_number_density[int(x / 16)] = averaged_value
    #         mean_density = np.mean(averaged_value)

    #         averaged_value = (velocity_x[x-1] + velocity_x[x] + velocity_x[x+1]) / 3.0
    #         velocity_plot[int(x / 16)] = averaged_value

    #         magnetic_plot[int(x / 16)] = (magnetic_field_y[x-1] + magnetic_field_y[x] + magnetic_field_y[x+1]) / 3.0
    #         avg_mag_field = (net_magnetic_field[x-1] + net_magnetic_field[x] + net_magnetic_field[x+1]) / 3.0

    #         plot_magnetic_pressure_energy[int(x / 16)] = (np.square(avg_mag_field) / (8*np.pi)).to('erg/cm**3')
    #         # print(plot_magnetic_pressure_energy[int(x / 16)])

    #         plot_velocity_x[int(x / 16)] = ((velocity_x[x-1] + velocity_x[x] + velocity_x[x+1] / 3.0))
    #         avg_vel_field = (velocity_y[x-1] + velocity_y[x] + velocity_y[x+1] / 3.0)
    #         plot_kinetic_energy[int(x / 16)] = (0.5 * mean_density * np.square(avg_vel_field)).to('erg/cm**3')
            # print(plot_kinetic_energy[int(x / 16)])
    # print(averaged_number_density)

    plt.figure(figsize=(6, 5))
    print(X.dtype, Y.dtype, number_density.dtype)
    plt.pcolormesh(X, Y, velocity_x, cmap='gray', shading='auto')

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

    plt.title(f'Velocity X Map (time={time_evolved:.4f})')
    plt.colorbar(label=r'Velocity X Map (km/s)')
    plt.xlabel('x (pc)')
    plt.ylabel('y (pc)')
    plt.savefig(f'velocity_x_map_{time_evolved:.4f}.png', dpi=300, bbox_inches='tight')
    plt.close()



# min_freq = 0.1
# max_freq = 25

# freqs = np.linspace(min_freq, max_freq, 256)

# ls = LombScargle(freqs, averaged_number_density[1])
# density_power = ls.power(freqs)


# for ind in range(16, size, 16):
#     i = int(ind / 16)
#     print(i)

#     signal = averaged_number_density[i]
#     vel_signal = velocity_plot[i]
#     mag_signal = magnetic_plot[i]
    

#     f1, P1 = welch(signal, fs=256, scaling='density')
#     f2, P2 = welch(vel_signal, fs=256, scaling='density')
#     f3, P3 = welch(mag_signal, fs=256, scaling='density')

#     # P_rho_norm = P1 / np.trapezoid(P1, f1)
#     # P_vel_norm = P2 / np.trapezoid(P2, f2)
#     # P_mag_norm = P3 / np.trapezoid(P3, f3)

#     # first_10_indices = np.arange(1, 25)   # k = 1 to 10

#     vx = plot_velocity_x[i]
#     vy = plot_velocity_y[i]

#     # v_mag = np.sqrt(vx + vy)
#     # v_rms = np.sqrt(np.mean(v_mag**2))
#     # dispersion = np.std(v_mag)
#     smoothed_profile_x = gaussian_filter1d(vx, sigma=4.0)
#     smoothed_profile_y = gaussian_filter1d(vy, sigma=4.0)
#     # mean_v = np.mean(v_mag)

#     plt.figure(figsize=(8, 8))
#     # plt.plot(f1[:25], P_rho_norm[:25], label="density")
#     # plt.plot(f2[:25], P_vel_norm[:25], label="velocity")
#     # plt.plot(f3[:25], P_mag_norm[:25], label="magnetic")

#     # plt.legend()
#     # plt.xlabel("k (wavenumber)")
#     # plt.ylabel("Power")
#     # plt.title("1D Power Spectrum of Density, Velocity, Magnetic Field")
#     # plt.grid(True, which="both", ls="--", alpha=0.5)
#     # plt.tight_layout()
#     # # plt.show()
#     # plt.savefig(f'power_spectra_comparison_{i}.png', dpi=300, bbox_inches='tight')
#     avg = np.mean(plot_magnetic_pressure_energy[i])
#     vals = plot_magnetic_pressure_energy[i] > avg
#     print("Average magnetic Energy", avg)
#     print("Average kinetic energy", np.mean(plot_kinetic_energy[i]))
#     plt.plot(time, (avg_kinetic_energy) / (avg_kinetic_energy), label="ratio of ke/be")
#     # plt.plot(grid_x, plot_magnetic_pressure_energy, label="velocity_y")
#     # plt.axhline(v_rms, color='red', linestyle="--", label=f'v_rms = {v_rms:.4f}')
#     # plt.fill_between(grid_x, v_rms - dispersion, v_rms + dispersion, color = 'grey', alpha = 0.3, label ='dispersion region')
#     # plt.plot(grid_x, smoothed_profile_x, label="smooth Vx profile")
#     # plt.plot(grid_x, smoothed_profile_y, label="smooth Vy profile")
#     plt.legend()
#     plt.xlabel("x (pc)")
#     plt.ylabel("Energy")
#     plt.title(f'Energy Ratio (time = {time_evolved:.4f})')
#     plt.grid(True, which="both", ls="--", alpha=0.5)
#     plt.tight_layout()
#     # plt.show()
#     plt.savefig(f'velocity_components_{i}.png', dpi=300, bbox_inches='tight')
#     plt.close()

# energy_ratio = (kin_energy) / (mag_energy)
# avg_ratio = np.mean(energy_ratio)
# plt.plot(time, energy_ratio, label="ratio of ke/be")
# # plt.plot(grid_x, plot_magnetic_pressure_energy, label="velocity_y")
# plt.axhline(avg_ratio, color='red', linestyle="--", label=f'Average Ratio = {avg_ratio:.4f}')
# # plt.fill_between(grid_x, v_rms - dispersion, v_rms + dispersion, color = 'grey', alpha = 0.3, label ='dispersion region')
# # plt.plot(grid_x, smoothed_profile_x, label="smooth Vx profile")
# # plt.plot(grid_x, smoothed_profile_y, label="smooth Vy profile")
# plt.legend()
# plt.xlabel("time (Myr)")
# plt.ylabel("Energy Ratio (Kinetic in Y/ Compressed Magnetic in Y)")
# plt.title(f'Energy Ratio (time = {time_evolved:.4f})')
# plt.grid(True, which="both", ls="--", alpha=0.5)
# plt.tight_layout()
# # plt.show()
# plt.savefig(f'energy_ratio.png', dpi=300, bbox_inches='tight')
# plt.close()


# print(freqs)

# velocity_signal = velocity_plot[1]
# fft_vals = fft(velocity_signal)
# fourier_shifted = fftshift(fft_vals)
# power_spectra = np.abs(fourier_shifted)**2 / 256

# vel_power = power_spectra[positive_freqs]
# print(velocity_signal)
# vel_fft_vals = np.fft.fft(velocity_signal)
# velocity_power_spectra = np.abs(vel_fft_vals[:256//2])**2

# magnetic_signal = magnetic_plot[1]
# mag_fft_vals = np.fft.fft(magnetic_signal)
# magnetic_power_spectra = np.abs(mag_fft_vals[:256//2])**2

# freqs = freqs[first_10_indices]

# den_power = power_spectra[first_10_indices]
# vel_power = velocity_power_spectra[first_10_indices]
# mag_power = magnetic_power_spectra[first_10_indices]

# den_power_norm = den_power / np.trapezoid(den_power, freqs)
# vel_power_norm = vel_power / np.trapezoid(vel_power, freqs)
# mag_power_norm = mag_power / np.trapezoid(mag_power, freqs)