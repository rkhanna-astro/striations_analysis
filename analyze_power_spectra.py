import yt
import numpy as np
import matplotlib.pyplot as plt
from yt.units import mh
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch

spacing = [12, 72, 108]
unit_base={"length_unit": (1.0,"pc"), "time_unit": (1.0,"1.0 * pc / (0.35 * km/s)"), "mass_unit": (3.532e34,"g")}
time_step = 0
size = 256
grid_x = np.linspace(0, 6.32, size)
grid_y = np.linspace(0, 1, size)

inc = 8

velocity_plot = np.zeros((int(size / inc), size))
velocity_plot_2 = np.zeros((int(size / inc), size))
velocity_plot_3 = np.zeros((int(size / inc), size))

averaged_number_density = np.zeros((int(size / inc), size))
averaged_number_density_1 = np.zeros((int(size / inc), size))
averaged_number_density_2 = np.zeros((int(size / inc), size))


magnetic_plot = np.zeros((int(size / inc), size))
magnetic_plot_1 = np.zeros((int(size / inc), size))
magnetic_plot_2 = np.zeros((int(size / inc), size))

time = []

for space in spacing:
    name = 'high_b_dens_grad_high'
    # For wavelength 2L - I need 2-pc/1.58-km/s to get desired properties - 15uG, 1.58 km/s and 200 cm-3    ?
    # For wavelength L - I need 1-pc/1.0-km/s to get desired properties - 15uG, 1.58 km/s and 200 cm-3      ?
    # For wavelength L - I need 1-pc/1.0-km/s to get desired properties - 15uG, 1.58 km/s and 200 cm-3      
    ds_256 = yt.load(f'./results_{name}/data_files_256/LinWave.out2.{space:05d}.athdf', units_override=unit_base)

    left_edge = ds_256.domain_left_edge
    right_edge = ds_256.domain_right_edge
    dims =  ds_256.domain_dimensions

    # print(left_edge, dims)

    data_256 = ds_256.covering_grid(level=0, left_edge=left_edge, dims=dims)
    density = data_256['rho'].to("g/cm**3")[:, :, 0].T

    accurate_number_density_256 = data_256['rho'].to("g/cm**3") / (2.34 * mh)
    number_density = accurate_number_density_256.to("cm**-3")[:, :, 0].T

    velocity_x = data_256[('gas', 'velocity_x')].to("km/s")[:, :, 0].T
    velocity_y = data_256[('gas', 'velocity_y')].to("km/s")[:, :, 0].T

    net_velocity = np.sqrt(velocity_x**2 + velocity_y**2)

    time_evolved = ds_256.current_time.to("Myr")

    # vrms = np.sqrt(velocity_x**2 + velocity_y**2)

    magnetic_field_y = data_256[('gas', 'magnetic_field_y')].to("uG")[:, : , 0].T

    plot_velocity_x = np.zeros((int(size / inc), size))
    plot_velocity_y = np.zeros((int(size / inc), size))

    print(number_density[16])
    print(number_density[208])

    number_density = number_density - np.mean(number_density)
    net_velocity = net_velocity - np.mean(net_velocity)
    magnetic_field_y = magnetic_field_y - np.mean(magnetic_field_y)

    for x in range(inc, size, inc):
            averaged_dens_value = (number_density[x-1] + number_density[x] + number_density[x+1]) / 3.0
            averaged_mag_value = (magnetic_field_y[x-1] + magnetic_field_y[x] + magnetic_field_y[x+1]) / 3.0
            averaged_value = (net_velocity[x-1] + net_velocity[x] + net_velocity[x+1]) / 3.0

            if space == spacing[0]:
                velocity_plot[int(x / inc)] = averaged_value
                averaged_number_density[int(x / inc)] = averaged_dens_value
                magnetic_plot[int(x / inc)] = averaged_mag_value
            elif space == spacing[1]:
                velocity_plot_2[int(x / inc)] = averaged_value
                averaged_number_density_1[int(x / inc)] = averaged_dens_value
                magnetic_plot_1[int(x / inc)] = averaged_mag_value
            else:
                velocity_plot_3[int(x / inc)] = averaged_value
                averaged_number_density_2[int(x / inc)] = averaged_dens_value
                magnetic_plot_2[int(x / inc)] = averaged_mag_value
            
            plot_velocity_x[int(x / inc)] = ((velocity_x[x-1] + velocity_x[x] + velocity_x[x+1] / 3.0))
            plot_velocity_y[int(x / inc)] = ((velocity_y[x-1] + velocity_y[x] + velocity_y[x+1] / 3.0))
    # print(averaged_number_density)
    time_evolved = ds_256.current_time.to("Myr")
    time.append(time_evolved)
    print(time_evolved)



# min_freq = 0.1
# max_freq = 25

# freqs = np.linspace(min_freq, max_freq, 256)

# ls = LombScargle(freqs, averaged_number_density[1])
# density_power = ls.power(freqs)


for ind in range(inc, size, inc):
    i = int(ind / inc)
    print(i)

    den_signal = averaged_number_density[i]
    den_signal_1 = averaged_number_density_1[i]
    den_signal_2 = averaged_number_density_2[i]

    vel_signal = velocity_plot[i]
    vel_signal_1 = velocity_plot_2[i]
    vel_signal_2 = velocity_plot_3[i]

    mag_signal = magnetic_plot[i]
    mag_signal_1 = magnetic_plot_1[i]
    mag_signal_2 = magnetic_plot_2[i]

    f1, P1 = welch(den_signal, fs=256, window= 'hann', scaling='spectrum', nperseg = 256, noverlap = 128)
    f2, P2 = welch(den_signal_1, fs=256, window= 'hann', scaling='spectrum', nperseg = 256, noverlap = 128)
    f3, P3 = welch(den_signal_2, fs=256, window= 'hann', scaling='spectrum', nperseg = 256, noverlap = 128)

    # P_rho_norm = P1 / np.trapezoid(P1, f1)
    # P_vel_norm = P2 / np.trapezoid(P2, f2)
    # P_mag_norm = P3 / np.trapezoid(P3, f3)

    # first_10_indices = np.arange(1, 25)   # k = 1 to 10

    vx = plot_velocity_x[i]
    vy = plot_velocity_y[i]

    # v_mag = np.sqrt(vx + vy)
    # v_rms = np.sqrt(np.mean(v_mag**2))
    # dispersion = np.std(v_mag)
    smoothed_profile_x = gaussian_filter1d(vx, sigma=4.0)
    smoothed_profile_y = gaussian_filter1d(vy, sigma=4.0)
    # mean_v = np.mean(v_mag)

    fig, ax1 = plt.subplots(figsize=(6,4))

    # Primary axis: Density power
    ax1.loglog(f1[:25], P1[:25], color='tab:grey', linewidth=0.7, label=f'{time[0]:.2f}')
    ax1.loglog(f2[:25], P2[:25], color='tab:blue', linewidth=0.7, label=f'{time[1]:.2f}')
    ax1.loglog(f3[:25], P3[:25], color='tab:red', linewidth=0.7, label=f'{time[2]:.2f}')
    ax1.set_xlabel(r'frequency $\nu \,[pc^{-1}]$')
    ax1.set_ylabel(r'$|P_{n_{dens}}|^{2}$', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # # Secondary axis: Velocity power
    # ax2 = ax1.twinx()
    # ax2.plot(f2[:25], P2[:25], color='tab:red', linestyle='--',linewidth=0.7, label='velocity (v)')
    # ax2.set_ylabel(r'$|P_{v}|^{2}$', color='black')
    # ax2.tick_params(axis='y', labelcolor='black')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    # lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(
        loc='upper right',
        frameon=True,           # turn on legend box
        facecolor='white',      # background color
        edgecolor='black',      # border color
        framealpha=0.7          # opacity (0 transparent → 1 solid)
    )

    plt.tight_layout()
    # plt.show()
    plt.savefig(f'mag_{name}_spectra_{(round(ind/256, 3)):.3f}_pc.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    


    # plt.figure(figsize=(8, 8))
    # plt.plot(f1[:25], P1[:25], label="density")   
    # # plt.plot(f2, P2, label="velocity")
    # # plt.plot(f3, P3, label="magnetic field")    

    # plt.legend()
    # plt.xlabel("k (wavenumber)")
    # plt.ylabel("Power")
    # plt.title("1D Power Spectrum of Density, Velocity, Magnetic Field")
    # plt.grid(True, which="both", ls="--", alpha=0.5)

    # plt.plot(grid_x, vx, label="velocity_x")
    # plt.plot(grid_x, vy, label="velocity_y")
    # # plt.axhline(v_rms, color='red', linestyle="--", label=f'v_rms = {v_rms:.4f}')
    # # plt.fill_between(grid_x, v_rms - dispersion, v_rms + dispersion, color = 'grey', alpha = 0.3, label ='dispersion region')
    # # plt.plot(grid_x, smoothed_profile_x, label="smooth Vx profile")
    # # plt.plot(grid_x, smoothed_profile_y, label="smooth Vy profile")
    # plt.legend()
    # plt.xlabel("x (pc)")
    # plt.ylabel("velocity (km/s)")
    # plt.title(f'Velocity variations (time = {time_evolved:.4f})')
    # plt.grid(True, which="both", ls="--", alpha=0.5)
    # 
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