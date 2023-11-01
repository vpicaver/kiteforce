import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from scipy.interpolate import interp1d

# Constants
R = 287.05  # Specific gas constant for dry air in J/(kg·K)
Cd = 1.2  # Drag coefficient for a flat plate
A = 1.0  # Area of the flat plate in m^2
KtoC = 273.15 #kelvin to C

# Conversion factor from m/s to knots
ms_to_knots = 1.94384

# Ranges and increments
#temperatures = np.arange(-20, 41, 10)  # Temperature in Celsius
temperatures = np.arange(-40, 41, 1.0)  # Temperature in Celsius
#temperatures = np.arange(13, 16, 0.1)  # Temperature in Celsius
#temperatures = np.arange(15, 16, 10)  # Temperature in Celsius
H = np.arange(0, 4100, 10)  # Altitude in meters
#H = np.arange(0, 100, 0.1)  # Altitude in meters
V = np.arange(0, 60.1 * 1.0/ms_to_knots, 1)  # Wind velocity in m/s

# Initialize 3D arrays to hold the force data
F_given_eq = np.zeros((len(temperatures), len(H), len(V)))

# Calculate Pressure at Altitude using the provided equation
pressure_at_altitude_given_eq = 101325 * (1 - (0.0065 * H / 288.15)) ** 5.2561

# Calculate Air Density using the provided equation
rho_given_eq = pressure_at_altitude_given_eq[:, np.newaxis] / (R * (temperatures[np.newaxis, :] + KtoC))

# Reshape V for broadcasting
V_reshaped = V[np.newaxis, np.newaxis, :]

# Calculate the force using the provided air density calculation
F_given_eq = 0.5 * Cd * A * rho_given_eq[:, :, np.newaxis] * V_reshaped ** 2


# Convert wind velocity from m/s to knots for the x-axis
V_knots = V * ms_to_knots

# Kite sizes
kite_sizes = [6, 8, 11]

# Min and Max kite forces
min_force = 125
max_force = 350


# Generate separate contour plots for each temperature
#for i, T in enumerate(temperatures):
#    fig, ax = plt.subplots(figsize=(6, 5))
#    c = ax.contourf(V_knots, H, F_given_eq[:, i, :], levels=np.linspace(0, 200, 11), cmap='plasma', vmin=0, vmax=200)

#    for kite_size in kite_sizes:
#        kite_force = F_given_eq[:, i, :] * kite_size
#        # Add contour lines for kite force between 100N and 500N
#        c_kite = ax.contour(V_knots, H, kite_force, levels=[min_force, max_force], linewidths=2, colors='lightgray')
#        ax.clabel(c_kite, inline=True, fontsize=8, fmt=f'{kite_size}m kite: %1.0f N')

#    ax.set_title(f'Temperature = {T}°C')
#    ax.set_xlabel('Wind Velocity (knots)')
#    ax.set_ylabel('Altitude (m)')
#    cbar = fig.colorbar(c)
#    cbar.set_label('Force (N)')
#    cbar.set_ticks(np.linspace(0, 200, 11))
#    plt.show()

# Find the index for 0m altitude and 15C temperature
index_H = np.where(np.isclose(H, 0.0, atol=1e-8))[0][0]
index_T = np.where(np.isclose(temperatures, 15.0, atol=1e-8))[0][0]
#index_T_13 = np.where(np.isclose(temperatures, 13.5, atol=1e-8))[0][0]

# Look up the air density at these conditions
rho_at_0m_15C = rho_given_eq[index_H, index_T]

# Calculate air density at standard atmospheric conditions (Altitude = 0, Temperature = 15°C)
rho_standard = 101325 / (R * 288.15)  # Pressure = 101325 Pa, Temperature = 288.15 K

print(f"Air density at 0m and 15°C from rho_given_eq array: {rho_at_0m_15C} kg/m³")
print(f"Air density at standard atmospheric conditions (0m and 15°C): {rho_standard} kg/m³")

# Calculate the percent difference
percent_diff = ((rho_given_eq - rho_standard) / rho_standard) * 100

# Look up the percent difference at 0m and 15°C
percent_diff_at_0m_15C = percent_diff[index_H, index_T]
print(f"Percent difference in air density at 0m and 15°C: {percent_diff_at_0m_15C}%")
#percent_diff_at_0m_13C = percent_diff[index_H, index_T_13]
#print(f"Percent difference in air density at 0m and 13.5°C: {percent_diff_at_0m_13C}%")

def display_lookup_table(data, row_labels, col_labels, row_title, col_title):
    df = pd.DataFrame(data, index=row_labels, columns=col_labels)
    df.index.name = row_title
    df.columns.name = col_title
    return df

## Display the first lookup table
#first_lookup_table = display_lookup_table(percent_diff, H, temperatures, 'Altitude (m)', 'Temperature (°C)')
#first_lookup_table.head()  # Show only the first few rows for demonstration purposes


def plot_contour(data, temperatures, H, title, cbar_label, levels, fmt, filename, cmap='coolwarm'):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create filled contours
    c = ax.contourf(temperatures, H, data, levels=levels, cmap=cmap)

    # Add contour lines
    c_lines = ax.contour(temperatures, H, data, levels=levels, colors='k', linewidths=0.5)

    # Label the contour lines
    ax.clabel(c_lines, inline=True, fontsize=8, fmt=fmt)

    ax.set_title(title)
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Altitude (m)')

    # Loop through and add vertical lines at temperature increments of 10
    for temp in np.arange(np.min(temperatures), np.max(temperatures) + 1, 10):
        ax.axvline(temp, color='gray', linestyle='--', linewidth=0.5)

    # Loop through and add horizontal lines at altitude increments of 500
    for alt in np.arange(np.min(H), np.max(H) + 1, 500):
        ax.axhline(alt, color='gray', linestyle='--', linewidth=0.5)

    cbar = fig.colorbar(c)
    cbar.set_label(cbar_label)
    cbar.set_ticks(levels)

    plt.savefig(filename)

# Use the function for percent_diff
plot_contour(
    data=percent_diff,
    temperatures=temperatures,
    H=H,
    title='Air Density compared to St andard Atmospheric Conditions (15C @ Sea Level 0m)',
    cbar_label='Percent Difference (%)',
    levels=np.linspace(-50, 50, 21),
    fmt='%1.2f%%',
    filename="percent_diff.svg"
)

# Calculate the multiplier
multiplier = rho_standard / rho_given_eq

## Making air density cards ##
# Find the indices that match the subset ranges
subset_temperatures = np.arange(-20, 7, 2)  # From -20°C to 6°C with 2°C increments
subset_H = np.arange(500, 3100, 100)  # From 500m to 3000m with 100m increments

subset_temp_indices = [np.where(np.isclose(temperatures, t, atol=1e-8))[0][0] for t in subset_temperatures]
subset_H_indices = [np.where(np.isclose(H, h, atol=1e-8))[0][0] for h in subset_H]

# Generate the air density multiplier
air_density_multiplier = rho_given_eq / rho_standard

# Extract the subset of data based on the indices
subset_air_density_multiplier = air_density_multiplier[np.ix_(subset_H_indices, subset_temp_indices)]

# Create and display the new subset air_density_multiplier table
subset_air_density_multiplier_table = display_lookup_table(subset_air_density_multiplier, subset_H, subset_temperatures, 'Altitude (m)', 'Temperature (°C)')
subset_air_density_multiplier_table.to_csv("subset_air_density_multiplier_table.csv")

## Create multiplier table
force_at_0m_15C = F_given_eq[index_H, index_T, :]

# Convert existing wind speeds from m/s to km/h for interpolation
V_km_h = V * 3.6

# Create the new wind speed range in km/h
new_V_km_h = np.arange(5, 66, 5)

# Interpolate the existing force data to the new wind speed range
interp_force = interp1d(V_km_h, force_at_0m_15C, kind='linear')
new_force_at_0m_15C = interp_force(new_V_km_h)

# Create the force multiplier range
force_multipliers = np.arange(0.70, 1.105, 0.02)

# Initialize the table to hold the force data
force_multiplier_table = np.zeros((len(force_multipliers), len(new_V_km_h)))

# Populate the table
for i, multiplier_ in enumerate(force_multipliers):
    force_multiplier_table[i, :] = new_force_at_0m_15C * multiplier_

# Create and display the force multiplier table
force_multiplier_table_df = display_lookup_table(force_multiplier_table, force_multipliers, new_V_km_h, 'Force Multiplier', 'Wind Speed (km/h)')
force_multiplier_table_df.to_csv("force_multiplier_table_df.csv")

# Use the function for multiplier
plot_contour(
    data=multiplier,
    temperatures=temperatures,
    H=H,
    title='Kite Size Multiplier Compared to Standard Atmospheric Conditions',
    cbar_label='Kite Size Multiplier',
    levels=np.linspace(0.5, 2, 31),
    fmt='%1.2f',
    filename="kite_size_multiplier.png"
)

################## 2D Graph #####################
# Extract the slice of force data at 0m and 15°C
force_at_0m_15C = F_given_eq[index_H, index_T, :]

# Create a new figure for the line graph
plt.figure(figsize=(8, 6))

# Plot the force vs wind speed
plt.plot(V_knots, force_at_0m_15C, label='Force at 0m and 15°C')

# Add title and labels
plt.title('Force (on a 1m^2 thin plate) vs Wind Speed at 0m and 15°C')
plt.xlabel('Wind Speed (knots)')
plt.ylabel('Force (N)')

# Set tick marks to increment by 5 units
# Create major ticks at every 10 units and minor ticks at every 5 units for x-axis
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
#ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))

# Create major ticks at every 10 units and minor ticks at every 5 units for y-axis
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
#ax.yaxis.set_major_locator(ticker.MultipleLocator(5))

# Add grid lines for both major and minor ticks
plt.grid(which='both')

# Customize grid lines for minor ticks to be gray and less visible
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')

# Add a grid for better readability
plt.grid(True)

# Show legend
plt.legend()

# Save the figure if needed
plt.savefig("force_vs_wind_speed_0m_15C.svg")
plt.savefig("force_vs_wind_speed_0m_15C.png")

########

# Create a new figure for the line graph
plt.figure(figsize=(12, 6))

# Plot the wind speed vs force (note the swapped order of variables)
plt.plot(force_at_0m_15C, V_knots, label='Wind Speed at 0m and 15°C')

# Add title and labels
plt.title('Wind Speed vs Force (on a 1m$^2$ thin plate) at 0m and 15°C')
plt.ylabel('Wind Speed (knots)')
plt.xlabel('Force (N)')

# Set tick marks to increment by 5 units
# Create major ticks at every 10 units and minor ticks at every 5 units for x-axis
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))

# Create major ticks at every 10 units and minor ticks at every 5 units for y-axis
ax.yaxis.set_major_locator(ticker.MultipleLocator(5))

# Add grid lines for both major and minor ticks
plt.grid(which='both')

# Customize grid lines for minor ticks to be gray and less visible
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')

# Add a grid for better readability
plt.grid(True)

# Show legend
plt.legend()

# Save the figure if needed
plt.savefig("wind_speed_vs_force_0m_15C.svg")
plt.savefig("wind_speed_vs_force_0m_15C.png")

########

# Kite sizes and their corresponding wind speed ranges in knots for Flysurfer Hybrid and Flysurfer Peak5
kite_sizes_hybrid = [2.5, 3.5, 5.5, 7.5, 9.5, 11.5]
kite_sizes_peak5 = [2.5, 4, 5, 6, 8, 11, 13]
wind_speed_ranges_hybrid_knots = [(18, 32), (15, 28), (12, 22), (10, 18), (8, 15), (8, 13)]
wind_speed_ranges_peak5_knots = [(18, 32), (15, 28), (12, 22), (10, 18), (8, 15), (6, 11), (5, 10)]

def calculate_forces(kite_sizes, wind_speed_ranges_knots, V, force_at_0m_15C, ms_to_knots):
    # Convert wind speeds from knots to m/s
    wind_speed_ranges = [(min_wind / ms_to_knots, max_wind / ms_to_knots) for min_wind, max_wind in wind_speed_ranges_knots]

    # Initialize lists to store the calculated forces
    min_forces = []
    max_forces = []

    # Calculate the minimum and maximum forces for each kite
    for kite_size, (min_wind, max_wind) in zip(kite_sizes, wind_speed_ranges):
        # Find closest indices for min and max wind speeds in the V array
        min_index = np.argmin(np.abs(V - min_wind))
        max_index = np.argmin(np.abs(V - max_wind))

        # Calculate forces using force_at_0m_15C
        min_force = force_at_0m_15C[min_index] * kite_size
        max_force = force_at_0m_15C[max_index] * kite_size

        min_forces.append(min_force)
        max_forces.append(max_force)

    return min_forces, max_forces

# Kite sizes and wind speed ranges in knots for both kite sets
kite_sizes_hybrid = [2.5, 3.5, 5.5, 7.5, 9.5, 11.5]
wind_speed_ranges_hybrid_knots = [(18, 32), (15, 28), (12, 22), (10, 18), (8, 15), (8, 13)]
kite_sizes_peak5 = [2.5, 4, 5, 6, 8, 11, 13]
wind_speed_ranges_peak5_knots = [(18, 32), (15, 28), (12, 22), (10, 18), (8, 15), (6, 11), (5, 10)]

# Calculate forces for both kite sets
min_forces_hybrid, max_forces_hybrid = calculate_forces(kite_sizes_hybrid, wind_speed_ranges_hybrid_knots, V, force_at_0m_15C, ms_to_knots)
min_forces_peak5, max_forces_peak5 = calculate_forces(kite_sizes_peak5, wind_speed_ranges_peak5_knots, V, force_at_0m_15C, ms_to_knots)


# Create horizontal bar graph
fig, ax = plt.subplots()

# Create an index for each tick position
ind_hybrid = np.arange(len(kite_sizes_hybrid))
ind_peak5 = np.arange(len(kite_sizes_peak5))

# Plot min and max forces as horizontal bars for both kite sets
min_bar_hybrid = ax.barh(ind_hybrid, min_forces_hybrid, 0.35, label='Min Force (Hybrid)', color='dodgerblue', alpha=0.8)
max_bar_hybrid = ax.barh(ind_hybrid, max_forces_hybrid, 0.35, left=min_forces_hybrid, label='Max Force (Hybrid)', color='royalblue', alpha=0.8)
min_bar_peak5 = ax.barh(ind_peak5 + 0.4, min_forces_peak5, 0.35, label='Min Force (Peak5)', color='darkorange', alpha=0.8)
max_bar_peak5 = ax.barh(ind_peak5 + 0.4, max_forces_peak5, 0.35, left=min_forces_peak5, label='Max Force (Peak5)', color='orangered', alpha=0.8)

# Describe the data
ax.set_xlabel('Force (N)')
ax.set_title('Force ranges for different kite sizes at 0m and 15°C')
ax.set_yticks(np.concatenate([ind_hybrid, ind_peak5 + 0.4]))
ax.set_yticklabels([f"{size}m (Hybrid)" for size in kite_sizes_hybrid] + [f"{size}m (Peak5)" for size in kite_sizes_peak5])
ax.legend()

# Add annotations for min and max forces for Flysurfer Hybrid
for i, (min_force, max_force) in enumerate(zip(min_forces_hybrid, max_forces_hybrid)):
    ax.text(min_force - 5, i, f"{min_force:.1f}N", va='center', ha='right', color='white')
    ax.text(min_force + max_force - 5, i, f"{min_force + max_force:.1f}N", va='center', ha='right', color='white')

# Add annotations for min and max forces for Flysurfer Peak5
for i, (min_force, max_force) in enumerate(zip(min_forces_peak5, max_forces_peak5)):
    ax.text(min_force - 5, i + 0.4, f"{min_force:.1f}N", va='center', ha='right', color='white')
    ax.text(min_force + max_force - 5, i + 0.4, f"{min_force + max_force:.1f}N", va='center', ha='right', color='white')

# Add annotations to describe the wind speed range for each kite for Flysurfer Hybrid
for i, (min_wind, max_wind) in enumerate(wind_speed_ranges_hybrid_knots):
    ax.text(min_forces_hybrid[i] + max_forces_hybrid[i] + 5, i, f"{min_wind}-{max_wind} knots", va='center')

# Add annotations to describe the wind speed range for each kite for Flysurfer Peak5
for i, (min_wind, max_wind) in enumerate(wind_speed_ranges_peak5_knots):
    ax.text(min_forces_peak5[i] + max_forces_peak5[i] + 5, i + 0.4, f"{min_wind}-{max_wind} knots", va='center')


plt.savefig("peak_hybrid_force_0m_15C.svg")
plt.savefig("peak_hybrid_force_0m_15C.png")

# Show the plot
plt.show()


