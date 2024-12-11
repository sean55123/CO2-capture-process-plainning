import pyomo.environ as pyo
import pyomo.opt as opt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_results(installed_units, operating_units):
    sns.set(style="whitegrid")
    df_installed = pd.DataFrame(list(installed_units.items()), columns=['Process', 'Installed Units'])
    df_operating = pd.DataFrame(list(operating_units.items()), columns=['Process', 'Operating Units'])

    df_merged = pd.merge(df_installed, df_operating, on='Process', how='left').fillna(0)

    df_cost = pd.DataFrame(list(cost.items()), columns=['Process', 'Total Cost (USD)'])
    df_emission = pd.DataFrame(list(emission.items()), columns=['Process', 'Emissions (kg CO2/kg captured)'])

    df_full = pd.merge(df_merged, df_cost, on='Process', how='left')
    df_full = pd.merge(df_full, df_emission, on='Process', how='left')

    throughput_series = pd.Series(throughput)

    df_merged['Process'] = df_merged['Process'].astype(int)
    df_merged['Installed Units Full Throughput'] = df_merged['Installed Units'] * df_merged['Process'].map(throughput_series)
    df_merged['Operating Units Throughput'] = df_merged['Operating Units'] * df_merged['Process'].map(throughput_series)
    df_merged['Process'] = df_merged['Process'].astype(str)

    df_co2 = df_merged[['Process', 'Installed Units Full Throughput', 'Operating Units Throughput']].melt(
        id_vars='Process', 
        var_name='Stage', 
        value_name='CO2 Capture'
    )

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    ax1 = axes[0]
    bar_width = 0.35
    index = range(len(df_merged))

    ax1.bar(index, df_merged['Installed Units'], bar_width, label='Installed Units', color='skyblue')
    ax1.bar([i + bar_width for i in index], df_merged['Operating Units'], bar_width, label='Operating Units', color='salmon')

    ax1.set_xlabel('Process', fontsize=12)
    ax1.set_ylabel('Number of Units', fontsize=12)
    ax1.set_title('Installed vs Operating Units per Process', fontsize=14)
    ax1.set_xticks([i + bar_width / 2 for i in index])
    ax1.set_xticklabels(df_merged['Process'], fontsize=10)
    ax1.legend()

    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    ax2 = axes[1]

    sns.barplot(
        data=df_co2, 
        x='Process', 
        y='CO2 Capture', 
        hue='Stage', 
        palette='Set2', 
        ax=ax2
    )

    ax2.set_xlabel('Process', fontsize=12)
    ax2.set_ylabel('CO$_2$ Capture (kg/hr)', fontsize=12)
    ax2.set_title('CO$_2$ Capture Contribution per Process', fontsize=14)
    ax2.legend(title='Stage')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

# Data Initialization
# Processes
P = [1, 2, 3, 4, 5, 6, 7, 8]

# Parameters for each process 
# Process capital cost
cap = {
    1: 1278.2,  2: 464.48, 3: 1859.76, 4: 1024.88,
    5: 1506.94, 6: 1753.9, 7: 1583.64, 8: 1334.77
}

# Processes' usage of utilities
# Electricity (MJ/hr)
elec = {
    1: 255.6,  2: 9.36,   3: 232.35, 4: 73.239,
    5: 256.36, 6: 402.16, 7: 299.78, 8: 236.27 
}

# Low-pressured steam (MJ/hr)
lps1 = {
    1: 0,       2: 492.3, 3: 0,     4: 168.541,
    5: 447.048, 6: 0,     7: 492.3, 8: 32.8
}

# 2-bar low-temperature steam (MJ/hr)
lps2 = {
    1: 0, 2: 0, 3: 0,    4: 0,
    5: 0, 6: 0, 7: 60.1, 8: 0 
}

# Chilled water (MJ/hr)
cw = {
    1: 0,     2: 0,     3: 424.29, 4: 232.77,
    5: 179.1, 6: 179.1, 7: 59.3,   8: 32.3 
}

# Refrigerant (MJ/hr)
rf = {
    1: 88.95, 2: 0,     3: 0, 4: 0,
    5: 0,     6: 96.84, 7: 0, 8: 95.04
}

# Processes' CO2 capture throughput (kg/hr)
throughput = {
    1: 135.09, 2: 127.84, 3: 133.38, 4: 130.26,
    5: 126.84, 6: 135.09, 7: 129.83, 8: 136.80
}

ut_cost = [16.9, 4.54, 4.11, 4.43, 7.89]  # USD/GJ
ut_emission_factors = [120.06, 72.86, 66.68, 49.4, 66.58]  # kg CO2/GJ

# Utilities usage dictionaries
utilities_usage = {
    'elec': elec,
    'lps1': lps1,
    'lps2': lps2,
    'cw': cw,
    'rf': rf
}

def process_performance(P, utilities_usage, ut_emission_factors, ut_cost, throughput, cap, scenario_elec_price=None, scenario_elec_emission=None):
    co2_input = 142.20
    emissions = {}
    cost = {}
    aoc = {}
    utility_names = ['elec', 'lps1', 'lps2', 'cw', 'rf']

    # Adjust emission factors and costs if scenario parameters for electricity are provided
    scenario_emission_factors = list(ut_emission_factors)
    scenario_costs = list(ut_cost)

    if scenario_elec_price is not None:
        # Electricity is the first in utility_names list
        scenario_costs[0] = scenario_elec_price
    if scenario_elec_emission is not None:
        scenario_emission_factors[0] = scenario_elec_emission

    emission_factors = dict(zip(utility_names, scenario_emission_factors))
    economics_factors = dict(zip(utility_names, scenario_costs))
    
    for p in P:
        total_emission = 0  # kg CO2/hr
        total_operating_cost = 0 # USD/hr
        for utility in utility_names:
            usage_mj_per_hr = utilities_usage[utility][p]
            usage_gj_per_hr = usage_mj_per_hr / 1000
            emission_factor = emission_factors[utility]  # kg CO2/GJ
            indirect_emission = usage_gj_per_hr * emission_factor
            total_emission += indirect_emission
            
            economics_factor = economics_factors[utility]
            operating_cost = usage_gj_per_hr * economics_factor
            total_operating_cost += operating_cost
            
        # Adding with direct emission
        total_emission += (co2_input - throughput[p])

        # Convert operating cost into kUSD/year (assume 8000 hrs/yr)
        annual_operating_cost_kusd_yr = total_operating_cost * 8000 / 1000
        
        # Total cost = Capital cost/3 + Operating cost (kUSD/year)
        total_cost = (cap[p]/3) + annual_operating_cost_kusd_yr
        
        aoc[p] = annual_operating_cost_kusd_yr
        cost[p] = total_cost
        emissions[p] = total_emission / throughput[p]  # kg CO2 emitted per kg captured
    return emissions, cost, aoc

emission, cost, aoc = process_performance(P, utilities_usage, ut_emission_factors, ut_cost, throughput, cap, scenario_elec_emission=None, scenario_elec_price=None)

# Process footprint (m2/unit)
footprint = {
    1: 1.332, 2: 3.097, 3: 53.979, 4: 55.276,
    5: 58.479, 6: 54.148, 7: 44.847, 8: 7.891
}

# Additional Parameters
carbon_tax = 66.70     # USD per ton of CO2 emitted
land_price = 1e4       # USD per unit area
available_land = 6500  # Total available land area
CO2_target = 8333.33   # Target tons of CO2 capture per day (200 ton/day > 8333.33 kg/hr)
U_p = 30               # Maximum units per process
M = U_p                # Big-M value

# Solver
solver = opt.SolverFactory('cbc')

# ===========================================================================================
# First stage
# ===========================================================================================
first = pyo.ConcreteModel()
first.P = P

# Variables
first.y = pyo.Var(first.P, within=pyo.Binary) # Process selection
first.x = pyo.Var(first.P, within=pyo.NonNegativeIntegers) # Number of process installed

# Objective
first.obj = pyo.Objective(
    expr=sum(
        first.x[p] * (
            cost[p] + emission[p] * throughput[p] * carbon_tax + footprint[p] * land_price
        ) for p in first.P
    ),
    sense=pyo.minimize
)

# Constraints
first.selection_limit = pyo.Constraint(expr=sum(first.y[p] for p in first.P) <= 8)

# CO2 Capture Target
first.CO2_target_constraint = pyo.Constraint(
    expr=sum(first.x[p] * throughput[p] for p in first.P) >= CO2_target*1.5 
)

# Land Area Constraint
first.land_constraint = pyo.Constraint(
    expr=sum(first.x[p] * footprint[p] for p in first.P) <= available_land
)

# Process Availability
first.process_availability = pyo.ConstraintList()
for p in first.P:
    first.process_availability.add(first.x[p] <= U_p * first.y[p])

# Solve first stage
print("Solving First Stage...")
result_1 = solver.solve(first, tee=True)
if result_1.solver.termination_condition != pyo.TerminationCondition.optimal:
    print("First stage not optimal or infeasible.")
    exit()

print("First Stage Results:")
selected_processes = [p for p in P if pyo.value(first.y[p]) > 0.5]
installed_units = {p: pyo.value(first.x[p]) for p in selected_processes}
print("Selected Processes:", selected_processes)
print("Installed Units:", installed_units)


# For reproducibility
np.random.seed(42)

# Number of simulation iterations
num_simulations = 8000

# Wind Resource Availability (Normal Distribution)
wind_mean = 8       # m/s
wind_std = 1        # m/s
wind_resource = np.random.normal(wind_mean, wind_std, num_simulations)
wind_resource = np.maximum(wind_resource, 0.1)  # Ensure positive

# Operational Costs (Lognormal Distribution)
operational_mean = 35000    # $
operational_std = 10000     # $
mu = np.log((operational_mean ** 2) / np.sqrt(operational_std ** 2 + operational_mean ** 2))
sigma = np.sqrt(np.log(1 + (operational_std ** 2) / (operational_mean ** 2)))
operational_costs = np.random.lognormal(mu, sigma, num_simulations)

# Regulatory Factors (Binary)
subsidy_probability = 0.3
regulatory_factor = np.random.binomial(1, subsidy_probability, num_simulations)
regulatory_reduction = 0.1  # 10% reduction if subsidy is applied
regulatory_multiplier = 1 - (regulatory_factor * regulatory_reduction)

# Energy Demand (Triangular Distribution)
energy_min = 500    # MW
energy_mode = 1500   # MW
energy_max = 2500    # MW
energy_demand = np.random.triangular(energy_min, energy_mode, energy_max, num_simulations)

# Calculate Wind Energy Price
wind_energy_price = (operational_costs * regulatory_multiplier) / (wind_resource * energy_demand)
wind_energy_price_mwh = wind_energy_price * 1000 / 3.6 # $ per GJ

# Create DataFrame
results = pd.DataFrame({
    'Wind Resource (m/s)': wind_resource,
    'Operational Costs ($)': operational_costs,
    'Regulatory Factor': regulatory_factor,
    'Energy Demand (MW)': energy_demand,
    'Wind Energy Price ($/GJ)': wind_energy_price_mwh
})


# ===========================================================================================
# Second stage
# ===========================================================================================
# Evaluating the affect of renewable energy
scenario_elec_price = 600      # from original 16.72 USD/GJ to 30 USD/GJ
scenario_elec_emission = 0     # from original 120.06 kg CO2/GJ to 0 kg CO2/GJ (Using wind)

# Adjust the ut_cost and ut_emission_factors for the scenario
new_ele_cost = list(ut_cost)
new_ele_cost[0] = scenario_elec_price

new_ele_emission = list(ut_emission_factors)
new_ele_emission[0] = scenario_elec_emission

# Recompute process performance under new electricity conditions
new_emission, new_cost, new_aoc = process_performance(
    P, utilities_usage, new_ele_emission, new_ele_cost, throughput, cap
)

# Build the second stage model
second = pyo.ConcreteModel()
second.P = P

# Variables for second stage:
# z[p]: binary variable indicating whether process p is operated
# x[p]: number of units of process p to be operated
second.z = pyo.Var(second.P, within=pyo.Binary)
second.x = pyo.Var(second.P, within=pyo.NonNegativeIntegers)

# Objective:
# Only consider operating cost and carbon cost
second.obj = pyo.Objective(
    expr=sum(second.x[p] * (new_aoc[p] + (new_emission[p]*throughput[p]*carbon_tax)) 
             for p in second.P),
    sense=pyo.minimize
)

# Constraints:
second.CO2_target_constraint = pyo.Constraint(
    expr=sum(second.x[p]*throughput[p] for p in second.P) >= CO2_target
)

second.land_constraint = pyo.Constraint(
    expr=sum(second.x[p]*footprint[p] for p in second.P) <= available_land
)

second.availability = pyo.ConstraintList()
for p in second.P:
    # Limit to the number of units installed in first stage
    max_units = installed_units[p] if p in installed_units else 0
    # If the process wasn't installed at all in the first stage, max_units = 0
    # Ensure that x[p] can only operate up to the installed capacity if z[p] = 1
    second.availability.add(second.x[p] <= max_units * second.z[p])

second.process_limit = pyo.Constraint(expr=sum(second.z[p] for p in second.P) <= 8)
print("Solving Second Stage with new electricity conditions...")
result_2 = solver.solve(second, tee=True, load_solutions=True)
if (result_2.solver.termination_condition == pyo.TerminationCondition.optimal and
    result_2.solver.status == pyo.SolverStatus.ok):
    operated_processes = [p for p in P if pyo.value(second.z[p]) > 0.5]
else:
    print("Second stage not optimal or infeasible.")
    exit()

print("Second Stage Results (with changed electricity cost and emission):")
operating_units = {p: pyo.value(second.x[p]) for p in P if pyo.value(second.x[p]) > 0}

# Compare first and second stage results
print("First stage installed:", installed_units)
print("Second stage operating decision under new conditions:", operating_units)

plot_results(installed_units, operating_units)