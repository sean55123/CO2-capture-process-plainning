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

# Sequestration equipment area (m2 sec/kg)
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
master.y = pyo.Var(master.P, within=pyo.Binary)
master.theta = pyo.Var(bounds=(0, None))  # Lower bound on subproblem's cost

# Objective
master.obj = pyo.Objective(expr=master.theta, sense=pyo.minimize)

# Constraints
# Process Selection Limit
master.selection_limit = pyo.Constraint(expr=sum(master.y[p] for p in master.P) <= 8)

# Initialize Benders Cuts list
master.benders_cuts = pyo.ConstraintList()

# Benders Decomposition Loop
MAX_ITER = 100
epsilon = 1e-6
LB = -float('inf')
UB = float('inf')
iteration = 0

while iteration < MAX_ITER and (UB - LB) > epsilon:
    iteration += 1
    print(f"\nIteration {iteration}")
    # Solve Master Problem
    master_result = solver.solve(master, tee=False)
    if (master_result.solver.termination_condition == pyo.TerminationCondition.infeasible):
        print("Master problem is infeasible.")
        break
    y_values = {p: pyo.value(master.y[p]) for p in P}
    theta_value = pyo.value(master.theta)
    master_obj = pyo.value(master.obj)
    LB = master_obj
    print(f"Master Problem Objective: {master_obj}")
    print(f"Process Selections: {y_values}")
    
    # Subproblem
    sub = pyo.ConcreteModel()
    sub.P = P
    
    # Parameters
    sub.cost = cost
    sub.emission = emission
    sub.footprint = footprint
    sub.throughput = throughput
    sub.carbon_tax = carbon_tax
    sub.land_price = land_price
    sub.CO2_target = CO2_target
    sub.available_land = available_land
    sub.U_p = U_p
    sub.y = y_values  # Fixed from master problem
    
    # Variables
    sub.x = pyo.Var(sub.P, within=pyo.NonNegativeIntegers)
    
    # Objective
    sub.obj = pyo.Objective(
        expr=sum(sub.x[p] * (sub.cost[p] + sub.emission[p] * sub.carbon_tax + sub.footprint[p] * sub.land_price) for p in sub.P),
        sense=pyo.minimize
    )
    
    # Constraints
    # CO2 Capture Target
    sub.CO2_target_constraint = pyo.Constraint(
        expr=sum(sub.x[p] * sub.throughput[p] for p in sub.P) >= sub.CO2_target
    )
    
    # Land Area Constraint
    sub.land_constraint = pyo.Constraint(
        expr=sum(sub.x[p] * sub.footprint[p] for p in sub.P) <= sub.available_land
    )
    
    # Process Availability
    sub.process_availability = pyo.Constraint(sub.P)
    for p in sub.P:
        sub.process_availability[p] = sub.x[p] <= sub.U_p * sub.y[p]
    
    # Solve Subproblem
    sub.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    sub_result = solver.solve(sub, tee=False)
    if (sub_result.solver.termination_condition == pyo.TerminationCondition.infeasible):
        print("Subproblem is infeasible. Adding feasibility cut.")
        # Exclude current infeasible combination
        master.benders_cuts.add(
            sum((1 - master.y[p]) if y_values[p] > 0.5 else master.y[p] for p in P) >= 1
        )
        continue  # Proceed to next iteration
    else:
        # Subproblem is feasible
        sub_obj = pyo.value(sub.obj)
        UB = min(UB, sub_obj)
        print(f"Subproblem Objective: {sub_obj}")
        print(f"Updated Upper Bound: {UB}")
        
        # Get dual variables
        duals = sub.dual
        pi = duals[sub.CO2_target_constraint]  # Dual of CO2 target constraint
        mu = duals[sub.land_constraint]        # Dual of land constraint
        sigma = {}
        for p in sub.P:
            constraint = sub.process_availability[p]
            sigma_p = duals.get(constraint, 0)
            sigma[p] = sigma_p
        
        # Add Optimality Cut to Master Problem
        expr = master.theta >= (
            sum((sub.cost[p] + sub.emission[p] * throughput[p] * sub.carbon_tax + sub.footprint[p] * sub.land_price - sigma[p]) * sub.U_p * master.y[p]
                for p in sub.P)
            + pi * sub.CO2_target + mu * sub.available_land
        )
        master.benders_cuts.add(expr)
        print(f"Added Optimality Cut: {expr}")

print("\nBenders Decomposition Converged")
print(f"Optimal Objective Value: {LB}")
selected_processes = [p for p in P if y_values[p] > 0.5]
print(f"Selected Processes: {selected_processes}")

# Solve Subproblem One More Time to Get x_p Values
final_sub = pyo.ConcreteModel()
final_sub.P = P



# Parameters
final_sub.cost = cost
final_sub.emission = emission
final_sub.footprint = footprint
final_sub.throughput = throughput
final_sub.carbon_tax = carbon_tax
final_sub.land_price = land_price
final_sub.CO2_target = CO2_target
final_sub.available_land = available_land
final_sub.U_p = U_p
final_sub.y = {p: 1 if p in selected_processes else 0 for p in P}

# Variables
final_sub.x = pyo.Var(final_sub.P, within=pyo.NonNegativeIntegers)

# Objective
final_sub.obj = pyo.Objective(
    expr=sum(final_sub.x[p] * (final_sub.cost[p] + final_sub.emission[p] * final_sub.carbon_tax + final_sub.footprint[p] * final_sub.land_price) for p in final_sub.P),
    sense=pyo.minimize
)

# Constraints
# CO2 Capture Target
final_sub.CO2_target_constraint = pyo.Constraint(
    expr=sum(final_sub.x[p] * final_sub.throughput[p] for p in final_sub.P) >= final_sub.CO2_target
)

# Land Area Constraint
final_sub.land_constraint = pyo.Constraint(
    expr=sum(final_sub.x[p] * final_sub.footprint[p] for p in final_sub.P) <= final_sub.available_land
)

# Process Availability
final_sub.process_availability = pyo.ConstraintList()
for p in final_sub.P:
    final_sub.process_availability.add(final_sub.x[p] <= final_sub.U_p * final_sub.y[p])

# Solve the Final Subproblem
solver.solve(final_sub, tee=False)

# Extract x_p Values
x_values = {p: pyo.value(final_sub.x[p]) for p in P if pyo.value(final_sub.x[p]) > 0}
print(f"Number of Units for Each Selected Process: {x_values}")

# Compute Total Cost
total_cost = pyo.value(final_sub.obj)
print(f"Total Cost: {total_cost}")