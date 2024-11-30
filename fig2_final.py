import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
NUM_SUBCARRIERS = 4  # N = 4
CELL_RADIUS = 200  # r = 0.2 km (converted to meters)
f_c = 600e6  # Cloudlet CPU frequency in Hz (600 MHz)

# Frequency and Bandwidth
FREQUENCY_CARRIER = 2e9  # 2 GHz
BANDWIDTH_TOTAL = 1e6  # Total bandwidth in Hz (1 MHz)
BANDWIDTH_SC = BANDWIDTH_TOTAL / NUM_SUBCARRIERS  # Bandwidth per subcarrier

# Noise power
NOISE_POWER_DENSITY = -174  # in dBm/Hz
NOISE_POWER = (10 ** (NOISE_POWER_DENSITY / 10)) * 1e-3  # Convert to Watts/Hz
NOISE_POWER *= BANDWIDTH_SC  # Noise power per subcarrier

# Mobile device parameters
p_circuit = 50e-3  # Circuit power in Watts (50 mW)
p_max = 1  # Max transmission power in Watts
kappa = 1e-24  # Energy consumption coefficient
X = 18000  # CPU cycles per bit
F_i_max = 1e9  # Max CPU frequency of mobile devices (1 GHz)

# Job parameters
DATA_SIZE_MIN = 900  # in bits
DATA_SIZE_MAX = 1100  # in bits
DEADLINE_MIN = 50e-3  # 50 ms
DEADLINE_MAX = 150e-3  # 150 ms

# Helper functions
def path_loss(distance):
    # Free-space path loss model
    PL_dB = 20 * np.log10(distance / 1000) + 20 * np.log10(FREQUENCY_CARRIER / 1e6) + 32.45
    PL_linear = 10 ** (-PL_dB / 10)
    return PL_linear

def small_scale_fading():
    # Rayleigh fading
    return np.random.rayleigh(scale=1.0)

def calculate_rate(G_i, P_i):
    SINR = P_i * G_i / NOISE_POWER
    R_i = BANDWIDTH_SC * np.log2(1 + SINR)
    return R_i

def compute_transmission_time(D_i_i, R_i):
    T_i_t = D_i_i / R_i
    return T_i_t

def compute_offloading_energy(P_i, T_i_t):
    E_i_t = (P_i + p_circuit) * T_i_t
    return E_i_t

def compute_remote_execution_time(D_i_i):
    T_i_c = X * D_i_i / f_c
    return T_i_c

def compute_local_energy(D_i_i, T_i_i):
    E_local = kappa * X ** 3 * D_i_i ** 3 / T_i_i ** 2
    return E_local

def bisection_search_power(G_i, D_i_i, T_i_i):
    # Search bounds
    P_min = 0
    P_max = p_max
    tolerance = 1e-6
    max_iterations = 50
    P_i = P_max
    for _ in range(max_iterations):
        P_i = (P_min + P_max) / 2
        R_i = calculate_rate(G_i, P_i)
        T_i_t = compute_transmission_time(D_i_i, R_i)
        if T_i_t > T_i_i:
            # Need higher power to meet deadline
            P_min = P_i
        else:
            # Can try lower power to save energy
            P_max = P_i
        if P_max - P_min < tolerance:
            break
    R_i = calculate_rate(G_i, P_i)
    T_i_t = compute_transmission_time(D_i_i, R_i)
    return P_i, T_i_t

# Algorithm implementations

# Local Execution
def local_execution_total_energy(M, E_local_list):
    return np.sum(E_local_list)

# Minimum Group Allocation (Algorithm 1)
def minimum_group_allocation(M, D_i_list, T_i_list, G, E_local_list):
    alpha = np.zeros(M)  # Offloading decisions
    W = {}  # Subcarrier allocations
    remaining_subcarriers = list(range(NUM_SUBCARRIERS))
    users = list(range(M))
    E_total = 0

    while users and remaining_subcarriers:
        energy_savings = []
        for i in users:
            subcarrier_gains = G[i, remaining_subcarriers]
            if len(subcarrier_gains) == 0:
                continue
            best_subcarrier = remaining_subcarriers[np.argmax(subcarrier_gains)]
            min_group = [best_subcarrier]

            G_i = G[i, min_group][0]
            # Find optimal transmission power
            P_i_opt, T_i_t = bisection_search_power(G_i, D_i_list[i], T_i_list[i])
            T_i_c = compute_remote_execution_time(D_i_list[i])
            E_i_t = compute_offloading_energy(P_i_opt, T_i_t)

            # Check if offloading is beneficial and meets deadline
            if T_i_t + T_i_c <= T_i_list[i] and E_i_t < E_local_list[i]:
                energy_saving = E_local_list[i] - E_i_t
                energy_savings.append((energy_saving, i, min_group, E_i_t))
            else:
                energy_savings.append((0, i, min_group, E_local_list[i]))

        if not energy_savings:
            break

        # Select user with maximum energy saving
        energy_saving, i_selected, min_group_selected, E_i_t_selected = max(energy_savings, key=lambda x: x[0])

        if energy_saving > 0:
            alpha[i_selected] = 1
            W[i_selected] = min_group_selected
            E_total += E_i_t_selected
            for sc in min_group_selected:
                remaining_subcarriers.remove(sc)
            users.remove(i_selected)
        else:
            break

    # Add local execution energy for users who did not offload
    for i in users:
        E_total += E_local_list[i]

    return E_total

# Per-Resource Allocation
def per_resource_allocation(M, D_i_list, T_i_list, G, E_local_list):
    alpha = np.zeros(M)
    W = {}
    remaining_subcarriers = list(range(NUM_SUBCARRIERS))
    users = list(range(M))
    E_total = 0

    # Stage 1: Subcarrier Allocation
    while users and remaining_subcarriers:
        energy_savings = []
        for i in users:
            subcarrier_gains = G[i, remaining_subcarriers]
            if len(subcarrier_gains) == 0:
                continue
            best_subcarrier = remaining_subcarriers[np.argmax(subcarrier_gains)]
            min_group = [best_subcarrier]

            G_i = G[i, min_group][0]
            # Find optimal transmission power
            P_i_opt, T_i_t = bisection_search_power(G_i, D_i_list[i], T_i_list[i])
            T_i_c = compute_remote_execution_time(D_i_list[i])
            E_i_t = compute_offloading_energy(P_i_opt, T_i_t)

            # Assume CPU scheduling is done separately
            if T_i_t + T_i_c <= T_i_list[i] and E_i_t < E_local_list[i]:
                energy_saving = E_local_list[i] - E_i_t
                energy_savings.append((energy_saving, i, min_group, E_i_t))
            else:
                energy_savings.append((0, i, min_group, E_local_list[i]))

        if not energy_savings:
            break

        # Select user with maximum energy saving
        energy_saving, i_selected, min_group_selected, E_i_t_selected = max(energy_savings, key=lambda x: x[0])

        if energy_saving > 0:
            alpha[i_selected] = 1
            W[i_selected] = min_group_selected
            E_total += E_i_t_selected
            for sc in min_group_selected:
                remaining_subcarriers.remove(sc)
            users.remove(i_selected)
        else:
            break

    # Stage 2: CPU Scheduling (simplified)
    t_current = 0
    for i in range(M):
        if alpha[i] == 1:
            T_i_c = compute_remote_execution_time(D_i_list[i])
            t_current += T_i_c  # Accumulated compute time
            if t_current > T_i_list[i]:
                # Deadline missed due to compute resource constraints
                alpha[i] = 0
                E_total += E_local_list[i]  # Need to execute locally
                # Subcarriers allocated to this user are wasted
        else:
            E_total += E_local_list[i]

    return E_total

# Joint Allocation (Algorithm 3)
def joint_allocation(M, D_i_list, T_i_list, G, E_local_list):
    alpha = np.zeros(M)
    W = {}
    schedule = []
    remaining_subcarriers = list(range(NUM_SUBCARRIERS))
    users = list(range(M))
    E_total = 0
    t_current = 0

    while users and remaining_subcarriers:
        energy_savings = []
        for i in users:
            subcarrier_gains = G[i, remaining_subcarriers]
            if len(subcarrier_gains) == 0:
                continue
            best_subcarrier = remaining_subcarriers[np.argmax(subcarrier_gains)]
            min_group = [best_subcarrier]

            G_i = G[i, min_group][0]
            # Find optimal transmission power
            P_i_opt, T_i_t = bisection_search_power(G_i, D_i_list[i], T_i_list[i] - t_current)
            T_i_c = compute_remote_execution_time(D_i_list[i])
            E_i_t = compute_offloading_energy(P_i_opt, T_i_t)

            # Check if offloading meets deadline and is beneficial
            if max(T_i_t, t_current) + T_i_c <= T_i_list[i] and E_i_t < E_local_list[i]:
                energy_saving = E_local_list[i] - E_i_t
                energy_savings.append((energy_saving, i, min_group, T_i_t, T_i_c, E_i_t))
            else:
                continue

        if not energy_savings:
            break

        # Select user with maximum energy saving
        energy_saving, i_selected, min_group_selected, T_i_t_selected, T_i_c_selected, E_i_t_selected = max(energy_savings, key=lambda x: x[0])

        alpha[i_selected] = 1
        W[i_selected] = min_group_selected
        schedule.append(i_selected)
        t_current = max(T_i_t_selected, t_current) + T_i_c_selected
        E_total += E_i_t_selected

        for sc in min_group_selected:
            remaining_subcarriers.remove(sc)
        users.remove(i_selected)

    # Add local execution energy for users who did not offload
    for i in users:
        E_total += E_local_list[i]

    return E_total

# Optimal algorithms are approximated using joint allocation and minimum group allocation

# Main simulation
M_values = np.arange(4, 10, 1)

# Initialize lists to store energy consumption for each algorithm
energy_local_execution = []
energy_min_group = []
energy_optimal_no_cpu = []
energy_per_resource_allocation = []
energy_joint_allocation = []
energy_optimal_with_cpu = []

num_trials = 1000  # Number of trials to average out randomness

for M in M_values:
    E_local_exec_trials = []
    E_min_group_trials = []
    E_opt_no_cpu_trials = []
    E_per_res_alloc_trials = []
    E_joint_alloc_trials = []
    E_opt_with_cpu_trials = []

    for _ in range(num_trials):
        # Generate users' data
        D_i_list = np.random.uniform(DATA_SIZE_MIN, DATA_SIZE_MAX, M)
        T_i_list = np.random.uniform(DEADLINE_MIN, DEADLINE_MAX, M)
        E_local_list = np.array([compute_local_energy(D_i_list[i], T_i_list[i]) for i in range(M)])
        E_local_total = np.sum(E_local_list)
        E_local_exec_trials.append(E_local_total)

        # User positions
        user_positions = np.random.uniform(0, CELL_RADIUS, M)

        # Channel gains matrix G (users x subcarriers)
        G = np.zeros((M, NUM_SUBCARRIERS))
        for i in range(M):
            distance = user_positions[i]
            PL = path_loss(distance)
            for j in range(NUM_SUBCARRIERS):
                ssf = small_scale_fading()
                G[i, j] = PL * ssf

        # Minimum Group Allocation
        E_total_min_group = minimum_group_allocation(M, D_i_list, T_i_list, G, E_local_list)
        E_min_group_trials.append(E_total_min_group)

        # Optimal without CPU Constraint (approximated)
        E_total_opt_no_cpu = E_total_min_group  # Placeholder
        E_opt_no_cpu_trials.append(E_total_opt_no_cpu)

        # Per-Resource Allocation
        E_total_per_resource = per_resource_allocation(M, D_i_list, T_i_list, G, E_local_list)
        E_per_res_alloc_trials.append(E_total_per_resource)

        # Joint Allocation
        E_total_joint = joint_allocation(M, D_i_list, T_i_list, G, E_local_list)
        E_joint_alloc_trials.append(E_total_joint)

        # Optimal with CPU Constraint (approximated)
        E_total_opt_with_cpu = E_total_joint  # Placeholder
        E_opt_with_cpu_trials.append(E_total_opt_with_cpu)

    # Average over trials
    energy_local_execution.append(np.mean(E_local_exec_trials))
    energy_min_group.append(np.mean(E_min_group_trials))
    energy_optimal_no_cpu.append(np.mean(E_opt_no_cpu_trials))
    energy_per_resource_allocation.append(np.mean(E_per_res_alloc_trials))
    energy_joint_allocation.append(np.mean(E_joint_alloc_trials))
    energy_optimal_with_cpu.append(np.mean(E_opt_with_cpu_trials))

# Plotting the results
plt.figure(figsize=(10, 6))

# Local Execution: blue solid line with cross markers (×)
plt.plot(M_values, energy_local_execution, color='blue', marker='x', linestyle='-', label='Local Execution')

# Per-Resource Allocation: blue solid line with square markers (□)
plt.plot(M_values, energy_per_resource_allocation, color='blue', marker='s', linestyle='-', label='Per-Resource Allocation')

# Joint Allocation: red solid line with circle markers (○)
plt.plot(M_values, energy_joint_allocation, color='red', marker='o', linestyle='-', label='Joint Allocation')

# Optimal with CPU Constraint: red solid line with cross markers (×)
plt.plot(M_values, energy_optimal_with_cpu, color='red', marker='x', linestyle='-', label='Optimal with CPU Constraint')

# Minimum Group Allocation: green solid line with circle markers (○)
plt.plot(M_values, energy_min_group, color='green', marker='o', linestyle='-', label='Minimum Group Allocation')

# Optimal without CPU Constraint: green solid line with cross markers (×)
plt.plot(M_values, energy_optimal_no_cpu, color='green', marker='x', linestyle='-', label='Optimal without CPU Constraint')

plt.xlabel('Number of Users (M)')
plt.ylabel('Energy Consumption (Joules)')
plt.title('Energy Consumption w. r. t. Total Number of Users. N = 4, r =0.2 km, f_c = 600 MHz.')
plt.xticks(M_values)
plt.yticks(np.arange(0, 9, 1))  # From 0 to 8 with interval 1
plt.grid(True)
plt.legend()
plt.ylim(0, 8)
plt.xlim(4, 9)

# Save the figure to a file
plt.savefig('figure2.png', dpi=300, bbox_inches='tight')

plt.show()
