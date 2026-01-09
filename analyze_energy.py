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

# For wavelength 2L - I need 2-pc/1.58-km/s to get desired properties - 15uG, 1.58 km/s and 200 cm-3    ?
    # For wavelength L - I need 1-pc/1.0-km/s to get desired properties - 15uG, 1.58 km/s and 200 cm-3      ?
    # For wavelength L - I need 1-pc/1.0-km/s to get desired properties - 15uG, 1.58 km/s and 200 cm-3      ?

unit_base={"length_unit": (1.0,"pc"), "time_unit": (1.0,"1.0 * pc / (0.35 * km/s)"), "mass_unit": (3.532e34,"g")}
size = 256
grid_x = np.linspace(0, 1, size)
grid_y = np.linspace(0, 1, size)

spacing = np.linspace(100, 200, 100, dtype=int)

ds_256 = yt.load(f'./results_low_mid_b_dens_grad_high/data_files_256/LinWave.out2.{0:05d}.athdf', units_override=unit_base)
left_edge = ds_256.domain_left_edge
right_edge = ds_256.domain_right_edge
dims =  ds_256.domain_dimensions

data_256 = ds_256.covering_grid(level=0, left_edge=left_edge, dims=dims)
mean_By = None
density = None
mean_dens = None

low_time = []
low_mid_time = []
mid_time = []
high_time = []

low_mag_energy_y = np.array([])
low_kin_energy_y = np.array([])

low_mid_mag_energy_y = np.array([])
low_mid_kin_energy_y = np.array([])

mid_mag_energy_y = np.array([])
mid_kin_energy_y = np.array([])

high_mag_energy_y = np.array([])
high_kin_energy_y = np.array([])

mag_field = ['low', 'low_mid', 'mid', 'high']

for mag in mag_field:
    for space in spacing:
        ds_256 = yt.load(f'./results_{mag}_b_uniform_d/data_files_256/LinWave.out2.{space:05d}.athdf', units_override=unit_base)
        data_256 = ds_256.covering_grid(level=0, left_edge=left_edge, dims=dims)

        density = data_256['rho'].to("g/cm**3")
        mean_dens = np.mean(density)
        accurate_number_density_256 = data_256['rho'].to("g/cm**3") / (2.34 * mh)
        number_density = accurate_number_density_256.to("cm**-3")[:, :, 0].T

        striations = number_density > 0
        # print(striations)

        velocity_x = data_256[('gas', 'velocity_x')].to("cm/s")[:, :, 0].T
        velocity_y = data_256[('gas', 'velocity_y')].to("cm/s")[:, :, 0].T

        # if mag == 'low':
        #     alfvne_mach = data_256[('gas', 'mach_alfven')]
        #     print(alfvne_mach)

        time_evolved = ds_256.current_time.to("Myr")
        
        # vrms = np.sqrt(velocity_x**2 + velocity_y**2)

        magnetic_field_y = data_256[('gas', 'magnetic_field_y')].to("uG")[:, : , 0].T
        mean_By = np.mean(magnetic_field_y)
        magnetic_field_x = data_256[('gas', 'magnetic_field_x')].to("uG")[:, : , 0].T
        net_magnetic_field_y = magnetic_field_y - mean_By
        # print(magnetic_field_y)

        avg_kinetic_energy_y = (0.5 * mean_dens * (np.square(velocity_y) + np.square(velocity_x))).to('erg/cm**3').sum()
        avg_magnetic_energy_y = (np.square(net_magnetic_field_y) / (8*np.pi)).to('erg/cm**3').sum()

        if mag == 'low':
            low_kin_energy_y = np.append(low_kin_energy_y, avg_kinetic_energy_y)
            low_time.append(time_evolved)
            # smoothed_kin_profile = gaussian_filter1d(kin_energy, sigma=3.0)

            low_mag_energy_y = np.append(low_mag_energy_y, avg_magnetic_energy_y)
            # smoothed_mag_profile = gaussian_filter1d(mag_energy, sigma=3.0)
        elif mag == 'low_mid':
            low_mid_kin_energy_y = np.append(low_mid_kin_energy_y, avg_kinetic_energy_y)
            # smoothed_kin_profile = gaussian_filter1d(kin_energy, sigma=3.0)
            low_mid_time.append(time_evolved)
            low_mid_mag_energy_y = np.append(low_mid_mag_energy_y, avg_magnetic_energy_y)
            # smoothed_mag_profile = gaussian_filter1d(mag_energy, sigma=3.0)
        elif mag == 'mid':
            mid_kin_energy_y = np.append(mid_kin_energy_y, avg_kinetic_energy_y)
            # smoothed_kin_profile = gaussian_filter1d(kin_energy, sigma=3.0)
            mid_time.append(time_evolved)
            mid_mag_energy_y = np.append(mid_mag_energy_y, avg_magnetic_energy_y)
            # smoothed_mag_profile = gaussian_filter1d(mag_energy, sigma=3.0)
        else:
            high_kin_energy_y = np.append(high_kin_energy_y, avg_kinetic_energy_y)
            # smoothed_kin_profile = gaussian_filter1d(kin_energy, sigma=3.0)
            high_time.append(time_evolved)
            high_mag_energy_y = np.append(high_mag_energy_y, avg_magnetic_energy_y)
            # smoothed_mag_profile = gaussian_filter1d(mag_energy, sigma=3.0)

low_energy_ratio_y =  (low_mag_energy_y) / (low_kin_energy_y) 
low_mid_energy_ratio_y = (low_mid_mag_energy_y) / (low_mid_kin_energy_y) 
mid_energy_ratio_y = (mid_mag_energy_y) / (mid_kin_energy_y)
high_energy_ratio_y = (high_mag_energy_y) / (high_kin_energy_y)

# avg_ratio = np.mean(low_energy_ratio_y)
plt.plot(low_time, np.sqrt(low_energy_ratio_y), linestyle="--", label="low B")
plt.plot(low_mid_time, np.sqrt(low_mid_energy_ratio_y), linestyle="-.", label="low mid B")
plt.plot(mid_time, np.sqrt(mid_energy_ratio_y), linestyle="-", label="mid B")
plt.plot(high_time, np.sqrt(high_energy_ratio_y), linestyle=":", label="high B")

# plt.plot(grid_x, plot_magnetic_pressure_energy, label="velocity_y")
plt.axhline(1.0, color='red', linestyle="--", label=f'Average Ratio = {1.0:.4f}')
plt.legend()
plt.xlabel("time (Myr)")
plt.ylabel(r"$\alpha_{y} \, \, \,(B_y^{2}/ (V_y^{2} + V_x^{2}))$")
# plt.title(f'Energy Ratio (magnetic to kinetic)')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
# plt.show()
plt.savefig(f'uniform_d_kin_total_energy_mag_y_energy_ratio.pdf', dpi=300, bbox_inches='tight')
plt.close()
