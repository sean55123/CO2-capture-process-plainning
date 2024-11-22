import pyomo.environ as pyo
import pyomo.opt as opt

# Data Initialization

# Processes
P = [1, 2, 3, 4, 5, 6, 7, 8]

# Parameters for each process 
# Minimum required treatment price (USD/kg)
cost = {
    1: 0.978, 2: 0.640, 3: 1.244, 4: 0.875,
    5: 1.172, 6: 1.323, 7: 1.179, 8: 0.988
}

# Global warming potential (kg CO2eq/kg)
emission = {
    1: 0.327, 2: 0.420, 3: 0.433, 4: 0.350,
    5: 0.707, 6: 0.527, 7: 0.724, 8: 0.406
}

# Process footprint (m2/unit)
footprint = {
    1: 1.332, 2: 3.097, 3: 53.979, 4: 55.276,
    5: 58.479, 6: 54.148, 7: 44.847, 8: 7.891
}

throughput = {
    1: 3.244, 2: 3.069, 3: 3.201, 4: 3.126,
    5: 3.044, 6: 3.241, 7: 3.116, 8: 3.283
}

# Additional Parameters
carbon_tax = 50        # $ per ton of CO₂ emitted
land_price = 300       # $ per unit area
available_land = 6500  # Total available land area
CO2_target = 200       # Target tons of CO₂ capture per day
U_p = 30               # Maximum units per process
M = U_p                # Big-M value

# Solver
solver = opt.SolverFactory('cbc')

# Master Problem
master = pyo.ConcreteModel()
master.P = P

# Variables
master.y = pyo.Var(master.P, within=pyo.Binary) # Process selection
master.x = pyo.Var(master.P, within=pyo.NonNegativeIntegers) # Number of process installed

# Objective
master.obj = pyo.Objective(
    expr=sum(
        master.x[p] * (
            cost[p] + emission[p] * throughput[p] * carbon_tax + footprint[p] * land_price
        ) for p in master.P
    ),
    sense=pyo.minimize
)

# Constraints
master.selection_limit = pyo.Constraint(expr=sum(master.y[p] for p in master.P) <= 8)

# CO2 Capture Target
master.CO2_target_constraint = pyo.Constraint(
    expr=sum(master.x[p] * throughput[p] for p in master.P) >= CO2_target
)

# Land Area Constraint
master.land_constraint = pyo.Constraint(
    expr=sum(master.x[p] * footprint[p] for p in master.P) <= available_land
)

# Process Availability
master.process_availability = pyo.ConstraintList()
for p in master.P:
    master.process_availability.add(master.x[p] <= U_p * master.y[p])

# Initialize Benders Cuts list
master.benders_cuts = pyo.ConstraintList()

MAX_ITER = 1000
epsilon = 1e-6
iteration = 0
LB = -float('inf')
UB = float('inf')

while iteration < MAX_ITER and (UB - LB) > epsilon:
    iteration += 1
    print(f"\nIteration {iteration}")
    # Solve Master Problem
    master_result = solver.solve(master, tee=False)
    if (master_result.solver.termination_condition == pyo.TerminationCondition.infeasible):
        print("Master problem is infeasible.")
        break
    
    # Update LB with master objective value
    master_obj = pyo.value(master.obj)
    LB = master_obj
    
    # Scenarios and uncertainties
    scenarios = ['LowDemand', 'HighDemand']
    scenario_throughput = {
        'LowDemand': {p: throughput[p] * 0.95 for p in P},
        'HighDemand': {p: throughput[p] * 1.1 for p in P},
    }

    # Check feasibility under uncertainty in Subproblem
    sub_infeasible = False
    for scenario in scenarios:
        print(f"Checking feasibility under scenario: {scenario}")
        
        # Create a new subproblem for each scenario
        sub = pyo.ConcreteModel()
        sub.P = P

        # Variables
        sub.x = pyo.Var(sub.P, within=pyo.NonNegativeIntegers)

        # Parameters
        sub.y = {p: pyo.value(master.y[p]) for p in P}  # Fixed from master problem
        sub.throughput = scenario_throughput[scenario]

        # Constraints
        # CO2 Capture Target Constraint under the scenario
        sub.CO2_target_constraint = pyo.Constraint(
            expr=sum(sub.x[p] * sub.throughput[p] for p in sub.P) >= CO2_target
        )

        # Land Area Constraint
        sub.land_constraint = pyo.Constraint(
            expr=sum(sub.x[p] * footprint[p] for p in sub.P) <= available_land
        )

        # Process Availability Constraints
        sub.process_availability = pyo.ConstraintList()
        for p in sub.P:
            # Ensure that sub.x[p] does not exceed what was selected in the master problem
            sub.process_availability.add(sub.x[p] <= pyo.value(master.x[p]))
            # Ensure that sub.x[p] does not exceed the maximum units per process
            sub.process_availability.add(sub.x[p] <= U_p * sub.y[p])

        # Total Units Constraint (optional)
        sub.total_units_constraint = pyo.Constraint(
            expr=sum(sub.x[p] for p in sub.P) == sum(pyo.value(master.x[p]) for p in sub.P)
        )

        # Objective (since we're checking feasibility, objective can be zero)
        sub.obj = pyo.Objective(expr=0, sense=pyo.minimize)

        # Solve Subproblem
        sub_result = solver.solve(sub, tee=False)

        if sub_result.solver.termination_condition == pyo.TerminationCondition.infeasible:
            sub_infeasible = True
            print(f"Subproblem infeasible under scenario: {scenario}")
            break  # No need to check other scenarios
        else:
            print(f"Subproblem feasible under scenario: {scenario}")
            # Continue to check the next scenario

    if sub_infeasible:
        print("Subproblem infeasible under uncertainty. Adding feasibility cut.")
        # Add feasibility cut to master problem
        master.benders_cuts.add(
            sum((1 - master.y[p]) if pyo.value(master.y[p]) > 0.5 else master.y[p] for p in P) >= 1
        )
        continue
    else:
        print("Solution is robust under uncertainty.")
        UB = LB
        break  # Converged to a robust solution

print("\nBenders Decomposition Converged")
print(f"Optimal Objective Value: {LB}")
selected_processes = [p for p in P if pyo.value(master.y[p]) > 0.5]
print(f"Selected Processes: {selected_processes}")

# Extract x_p Values
x_values = {p: pyo.value(master.x[p]) for p in P if pyo.value(master.x[p]) > 0}
print(f"Number of Units for Each Selected Process: {x_values}")

# Compute Total Cost
total_cost = pyo.value(master.obj)
print(f"Total Cost: {total_cost}")