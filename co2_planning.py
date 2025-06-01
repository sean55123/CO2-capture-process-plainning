import numpy as np
from pyomo.environ import *
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time

class CarbonCapturePlanner:
    def __init__(self, co2_producers, co2_reaction_processes, capture_process_types, 
                 carbon_tax=50, land_price=300, available_land=6500, epsilon=0.01, max_iter=20):
        """
        Initialize the Carbon Capture Process Planner.
        
        Parameters:
        -----------
        co2_producers : dict
            Dictionary with CO2 producer data:
            {'coordinates': [(x1, y1), (x2, y2), ...],
             'production_rate': [a1, a2, ...],
             'cost': [cs1, cs2, ...]}
        
        co2_reaction_processes : dict
            Dictionary with CO2 reaction process data:
            {'coordinates': [(x1, y1), (x2, y2), ...],
             'consumption_rate': [d1, d2, ...]}
        
        capture_process_types : list of dict
            List of dictionaries with CO2 capture process type data:
            [{'count': int,
              'throughput': float,
              'process_cost': float,
              'emission': float,
              'footprint': float,
              'throughput': float}, ...]
        
        carbon_tax : float
            Carbon tax rate in $ per ton of CO₂ emitted
            
        land_price : float
            Land price in $ per unit area
            
        available_land : float
            Total available land area for CO2 capture processes
        
        epsilon : float
            Convergence tolerance
        
        max_iter : int
            Maximum number of iterations
        """
        # Store input data
        self.co2_producers = co2_producers
        self.co2_reaction_processes = co2_reaction_processes
        self.capture_process_types = capture_process_types
        self.carbon_tax = carbon_tax
        self.land_price = land_price
        self.available_land = available_land
        self.epsilon = epsilon
        self.max_iter = max_iter
    
        self.num_producers = len(co2_producers['coordinates'])
        self.num_reactors = len(co2_reaction_processes['coordinates'])
        
        # Calculate total number of potential CO2 capture processes
        self.num_processes = sum(ptype['count'] for ptype in capture_process_types)
        
        # Create process type mapping (which process belongs to which type)
        self.process_type_map = []
        process_idx = 0
        for i, ptype in enumerate(capture_process_types):
            for _ in range(ptype['count']):
                self.process_type_map.append(i)
                process_idx += 1
        
        # Transportation parameters
        self.fixed_trans_cost_producer = 10
        self.var_trans_cost_producer = 0.3
        self.fixed_trans_cost_reactor = 10
        self.var_trans_cost_reactor = 0.3
        
        # Minimum allowed distance between process and producer/reactor
        self.min_distance = 0.5
        
        # Determine the boundaries of the space
        all_x = [coord[0] for coord in co2_producers['coordinates'] + co2_reaction_processes['coordinates']]
        all_y = [coord[1] for coord in co2_producers['coordinates'] + co2_reaction_processes['coordinates']]
        self.x_min, self.x_max = min(all_x), max(all_x)
        self.y_min, self.y_max = min(all_y), max(all_y)
        
        margin = 0.1 * max(self.x_max - self.x_min, self.y_max - self.y_min)
        self.x_min -= margin
        self.x_max += margin
        self.y_min -= margin
        self.y_max += margin
        
        self.iterations = []
        self.lower_bounds = []
        self.upper_bounds = []
        self.best_solution = None
        self.best_objective = float('inf')
        self.model_statistics = []
    
    def get_model_statistics(self, model):
        stats = {
            'variables': {
                'total': 0,
                'binary': 0,
                'integer': 0,
                'continuous': 0,
                'by_component': {}
            },
            'constraints': {
                'total': 0,
                'by_component': {}
            }
        }
        
        for var_name, var_comp in model.component_map(Var).items():
            if var_comp.is_indexed():
                var_count = len(var_comp)
            else:
                var_count = 1
                
            stats['variables']['by_component'][var_name] = var_count
            stats['variables']['total'] += var_count
            
            if var_comp.is_indexed():
                for idx, var in var_comp.items():
                    if var.domain == Binary:
                        stats['variables']['binary'] += 1
                    elif var.domain == Integers:
                        stats['variables']['integer'] += 1
                    else:
                        stats['variables']['continuous'] += 1
            else:
                if var_comp.domain == Binary:
                    stats['variables']['binary'] += 1
                elif var_comp.domain == Integers:
                    stats['variables']['integer'] += 1
                else:
                    stats['variables']['continuous'] += 1
        
        for con_name, con_comp in model.component_map(Constraint).items():
            if con_comp.is_indexed():
                con_count = len(con_comp)
            else:
                con_count = 1
                
            stats['constraints']['by_component'][con_name] = con_count
            stats['constraints']['total'] += con_count
            
        return stats
    
    def create_master_model(self, partitions):
        model = ConcreteModel()
        
        # Sets
        model.I = RangeSet(0, self.num_producers - 1)  # CO2 Producers
        model.J = RangeSet(0, self.num_reactors - 1)   # CO2 Reaction Processes
        model.K = RangeSet(0, self.num_processes - 1)  # CO2 Capture Processes
        model.P = RangeSet(0, len(partitions['midpoints']) - 1)  # Partitions
        model.N = RangeSet(0, len(self.capture_process_types) - 1)  # Process types
        
        # Create sets for process types
        model.Kn = {}
        for n in model.N:
            model.Kn[n] = []
        
        for k in model.K:
            n = self.process_type_map[k]
            model.Kn[n].append(k)
        
        # Variables
        # Binary variables for process selection and links
        model.w = Var(model.K, model.P, domain=Binary)
        model.z_ik = Var(model.I, model.K, model.P, domain=Binary)
        model.z_kj = Var(model.K, model.J, model.P, domain=Binary)
        
        # Flow variables
        model.f_ik = Var(model.I, model.K, model.P, domain=NonNegativeReals)
        model.f_kj = Var(model.K, model.J, model.P, domain=NonNegativeReals)
        model.f_k = Var(model.K, model.P, domain=NonNegativeReals)
        
        # Storage variable
        model.storage_k = Var(model.K, model.P, domain=NonNegativeReals)
        
        # Cost variables
        model.cost_k = Var(model.K, model.P, domain=NonNegativeReals)
        model.cost_ik = Var(model.I, model.K, model.P, domain=NonNegativeReals)
        model.cost_kj = Var(model.K, model.J, model.P, domain=NonNegativeReals)
        
        # Objective function
        model.total_cost = Var(domain=NonNegativeReals)
        
        def obj_rule(model):
            return model.total_cost
        model.obj = Objective(rule=obj_rule, sense=minimize)
        
        # Constraint for total cost calculation
        def total_cost_rule(model):
            # Sum of all pure costs
            base_cost = (
                sum(model.cost_k[k, p]    for k in model.K for p in model.P)
            + sum(model.cost_ik[i, k, p] for i in model.I for k in model.K for p in model.P)
            + sum(model.cost_kj[k, j, p] for k in model.K for j in model.J for p in model.P)
            )
            # Add up revenues separately
            supply_rev   = sum(
                self.co2_producers['cost'][i] * model.f_ik[i, k, p]
                for i in model.I for k in model.K for p in model.P
            )
            reaction_rev = sum(
                self.co2_reaction_processes['profit'][j] * model.f_kj[k, j, p]
                for k in model.K for j in model.J for p in model.P
            )
            # Net cost = total costs minus revenues
            return model.total_cost == base_cost - supply_rev - reaction_rev
        model.total_cost_con = Constraint(rule=total_cost_rule)
        
        # Process cost constraints
        def process_cost_rule(model, k, p):
            n = self.process_type_map[k]
            process_cost = self.capture_process_types[n]['process_cost']
            emission = self.capture_process_types[n]['emission']
            footprint = self.capture_process_types[n]['footprint']
            
            # Fixed costs
            fixed_cost = process_cost * model.w[k, p] + footprint * self.land_price * model.w[k, p]
            
            throughput = self.capture_process_types[n]['throughput']
            emission_cost = emission * throughput * self.carbon_tax * model.w[k, p]
            
            return model.cost_k[k, p] == fixed_cost + emission_cost
        model.process_cost_con = Constraint(model.K, model.P, rule=process_cost_rule)
        
        # Producer to capture process transport cost
        def producer_cost_rule(model, i, k, p):
            ft = self.fixed_trans_cost_producer
            vt = self.var_trans_cost_producer
            min_dist = partitions['min_dist_producer'][i][p]
            return model.cost_ik[i, k, p] == (
                ft * model.z_ik[i, k, p]
            + vt * min_dist * model.f_ik[i, k, p]
            )
        model.producer_cost_con = Constraint(model.I, model.K, model.P, rule=producer_cost_rule)
        
        # Capture process to reaction process transport cost
        def reactor_cost_rule(model, k, j, p):
            ft = self.fixed_trans_cost_reactor
            vt = self.var_trans_cost_reactor
            min_dist = partitions['min_dist_reactor'][j][p]
            return model.cost_kj[k, j, p] == (
                ft * model.z_kj[k, j, p]
            + vt * min_dist * model.f_kj[k, j, p]
            )
        model.reactor_cost_con = Constraint(model.K, model.J, model.P, rule=reactor_cost_rule)
        
        # CO2 production constraints
        def producer_limit_rule(model, i):
            return sum(model.f_ik[i, k, p] for k in model.K for p in model.P) <= self.co2_producers['production_rate'][i]
        model.producer_limit_con = Constraint(model.I, rule=producer_limit_rule)
        
        # CO2 throughput constraint
        def throughput_rule(model, k, p):
            return model.f_k[k, p] == sum(model.f_ik[i, k, p] for i in model.I)
        model.throughput_con = Constraint(model.K, model.P, rule=throughput_rule)
        
        # CO2 balance constraints
        def process_balance_rule(model, k, p):
            return model.f_k[k, p] == sum(model.f_kj[k, j, p] for j in model.J) + model.storage_k[k, p]
        model.process_balance_con = Constraint(model.K, model.P, rule=process_balance_rule)
        
        # CO2 reaction process consumption constraints
        def reactor_consumption_rule(model, j):
            return sum(model.f_kj[k, j, p] for k in model.K for p in model.P) == self.co2_reaction_processes['consumption_rate'][j]
        model.reactor_consumption_con = Constraint(model.J, rule=reactor_consumption_rule)
        
        # CO2 throughput constraint
        def overall_throughput_rule(model):
            total_production = sum(self.co2_producers['production_rate'][i] for i in model.I)
            return sum(model.f_k[k, p] for k in model.K for p in model.P) >= total_production
        model.overall_throughput_con = Constraint(rule=overall_throughput_rule)
        
        # Land area constraint - THIS IS WHERE WE'LL CREATE INFEASIBILITY
        def land_area_rule(model):
            return sum(self.capture_process_types[self.process_type_map[k]]['footprint'] * model.w[k, p] 
                    for k in model.K for p in model.P) <= self.available_land
        model.land_area_con = Constraint(rule=land_area_rule)
        
        # Add the remaining constraints from your original model
        # Logical constraints
        def process_producer_link1_rule(model, i, k, p):
            return model.w[k, p] >= model.z_ik[i, k, p]
        model.process_producer_link1_con = Constraint(model.I, model.K, model.P, rule=process_producer_link1_rule)
        
        def process_producer_link2_rule(model, k, p):
            return sum(model.z_ik[i, k, p] for i in model.I) >= model.w[k, p]
        model.process_producer_link2_con = Constraint(model.K, model.P, rule=process_producer_link2_rule)
        
        def process_reactor_link1_rule(model, k, j, p):
            return model.w[k, p] >= model.z_kj[k, j, p]
        model.process_reactor_link1_con = Constraint(model.K, model.J, model.P, rule=process_reactor_link1_rule)
        
        def process_reactor_link2_rule(model, k, p):
            return sum(model.z_kj[k, j, p] for j in model.J) >= model.w[k, p]
        model.process_reactor_link2_con = Constraint(model.K, model.P, rule=process_reactor_link2_rule)
        
        # Uniqueness constraints
        def process_uniqueness_rule(model, k):
            return sum(model.w[k, p] for p in model.P) <= 1
        model.process_uniqueness_con = Constraint(model.K, rule=process_uniqueness_rule)
        
        def producer_link_uniqueness_rule(model, i, k):
            return sum(model.z_ik[i, k, p] for p in model.P) <= 1
        model.producer_link_uniqueness_con = Constraint(model.I, model.K, rule=producer_link_uniqueness_rule)
        
        def reactor_link_uniqueness_rule(model, k, j):
            return sum(model.z_kj[k, j, p] for p in model.P) <= 1
        model.reactor_link_uniqueness_con = Constraint(model.K, model.J, rule=reactor_link_uniqueness_rule)
        
        # Symmetry breaking constraints
        def symmetry_breaking_rule(model, k1, k2):
            if k1 < k2 and self.process_type_map[k1] == self.process_type_map[k2]:
                return sum(model.w[k1, p] for p in model.P) >= sum(model.w[k2, p] for p in model.P)
            else:
                return Constraint.Skip
        model.symmetry_breaking_con = Constraint(model.K, model.K, rule=symmetry_breaking_rule)
        
        # Throughput capacity constraints
        def throughput_capacity_rule(model, k, p):
            n = self.process_type_map[k]
            capacity = self.capture_process_types[n]['throughput']
            return model.f_k[k, p] <= capacity * model.w[k, p]
        model.throughput_capacity_con = Constraint(model.K, model.P, rule=throughput_capacity_rule)
        
        # Flow bounds
        def producer_flow_bound_rule(model, i, k, p):
            return model.f_ik[i, k, p] <= self.co2_producers['production_rate'][i] * model.z_ik[i, k, p]
        model.producer_flow_bound_con = Constraint(model.I, model.K, model.P, rule=producer_flow_bound_rule)
        
        def reactor_flow_bound_rule(model, k, j, p):
            return model.f_kj[k, j, p] <= self.co2_reaction_processes['consumption_rate'][j] * model.z_kj[k, j, p]
        model.reactor_flow_bound_con = Constraint(model.K, model.J, model.P, rule=reactor_flow_bound_rule)
        
        return model    
    
    def print_model_statistics(self, model, model_name="Model"):
        stats = self.get_model_statistics(model)
        
        print(f"\n{'-'*20} {model_name} STATISTICS {'-'*20}")
        print(f"VARIABLES: {stats['variables']['total']} total")
        print(f"  Binary variables: {stats['variables']['binary']}")
        print(f"  Integer variables: {stats['variables']['integer']}")
        print(f"  Continuous variables: {stats['variables']['continuous']}")
        
        print("\nVariables by component:")
        for comp_name, count in stats['variables']['by_component'].items():
            print(f"  {comp_name}: {count}")
        
        print(f"\nCONSTRAINTS: {stats['constraints']['total']} total")
        print("\nConstraints by component:")
        for comp_name, count in stats['constraints']['by_component'].items():
            print(f"  {comp_name}: {count}")
        
        print(f"{'-'*60}")
        
        return stats
    
    def create_partitioning(self, px, py):
        x_step = (self.x_max - self.x_min) / px
        y_step = (self.y_max - self.y_min) / py
        
        partitions = {
            'midpoints': [],
            'bounds': [],
            'x_step': x_step,
            'y_step': y_step,
            'px': px,
            'py': py
        }
        
        # Calculate midpoints and bounds for all partitions
        for j in range(py):
            for i in range(px):
                # Calculate midpoint
                x_mid = self.x_min + (i + 0.5) * x_step
                y_mid = self.y_min + (j + 0.5) * y_step
                partitions['midpoints'].append((x_mid, y_mid))
                
                # Calculate bounds
                x_lower = self.x_min + i * x_step
                x_upper = self.x_min + (i + 1) * x_step
                y_lower = self.y_min + j * y_step
                y_upper = self.y_min + (j + 1) * y_step
                partitions['bounds'].append(((x_lower, x_upper), (y_lower, y_upper)))
        
        # Calculate minimum distances between producers and partitions
        partitions['min_dist_producer'] = self._calculate_min_distances(
            self.co2_producers['coordinates'], partitions['midpoints'], x_step, y_step)
        
        # Calculate minimum distances between reactors and partitions
        partitions['min_dist_reactor'] = self._calculate_min_distances(
            self.co2_reaction_processes['coordinates'], partitions['midpoints'], x_step, y_step)
        
        return partitions
    
    def _calculate_min_distances(self, points, midpoints, x_step, y_step):
        min_distances = []
        
        for i, (xi, yi) in enumerate(points):
            distances_to_partitions = []
            
            for p, (xp, yp) in enumerate(midpoints):
                # Calculate dx and dy (equations 4a-4d)
                dx = max(abs(xi - xp) - x_step/2, 0)
                dy = max(abs(yi - yp) - y_step/2, 0)
                
                # Calculate minimum distance (equations 4e-4f)
                min_dist = max(np.sqrt(dx**2 + dy**2), self.min_distance)
                distances_to_partitions.append(min_dist)
            
            min_distances.append(distances_to_partitions)
        
        return min_distances
    
    def solve_master_problem(self, partitions):
        model = ConcreteModel()
        
        # Sets
        model.I = RangeSet(0, self.num_producers - 1)  # CO2 Producers
        model.J = RangeSet(0, self.num_reactors - 1)   # CO2 Reaction Processes
        model.K = RangeSet(0, self.num_processes - 1)  # CO2 Capture Processes
        model.P = RangeSet(0, len(partitions['midpoints']) - 1)  # Partitions
        model.N = RangeSet(0, len(self.capture_process_types) - 1)  # Process types
        
        # Create sets for process types
        model.Kn = {}
        for n in model.N:
            model.Kn[n] = []
        
        for k in model.K:
            n = self.process_type_map[k]
            model.Kn[n].append(k)
        
        # Variables
        # Binary variables for process selection and links
        model.w = Var(model.K, model.P, domain=Binary)
        model.z_ik = Var(model.I, model.K, model.P, domain=Binary)
        model.z_kj = Var(model.K, model.J, model.P, domain=Binary)
        
        # Flow variables
        model.f_ik = Var(model.I, model.K, model.P, domain=NonNegativeReals)
        model.f_kj = Var(model.K, model.J, model.P, domain=NonNegativeReals)
        model.f_k = Var(model.K, model.P, domain=NonNegativeReals)
        
        # Storage variable
        model.storage_k = Var(model.K, model.P, domain=NonNegativeReals)
        
        # Cost variables
        model.cost_k = Var(model.K, model.P, domain=NonNegativeReals)
        model.cost_ik = Var(model.I, model.K, model.P, domain=NonNegativeReals)
        model.cost_kj = Var(model.K, model.J, model.P, domain=NonNegativeReals)
        
        # Objective function
        model.total_cost = Var(domain=NonNegativeReals)
        
        def obj_rule(model):
            return model.total_cost
        model.obj = Objective(rule=obj_rule, sense=minimize)
        
        # Constraint for total cost calculation
        def total_cost_rule(model):
            # Sum of all pure costs
            base_cost = (
                sum(model.cost_k[k, p]    for k in model.K for p in model.P)
              + sum(model.cost_ik[i, k, p] for i in model.I for k in model.K for p in model.P)
              + sum(model.cost_kj[k, j, p] for k in model.K for j in model.J for p in model.P)
            )

            supply_rev   = sum(
                self.co2_producers['cost'][i] * model.f_ik[i, k, p]
                for i in model.I for k in model.K for p in model.P
            )
            reaction_rev = sum(
                self.co2_reaction_processes['profit'][j] * model.f_kj[k, j, p]
                for k in model.K for j in model.J for p in model.P
            )
            # Net cost = total costs - revenues
            return model.total_cost == base_cost - supply_rev - reaction_rev
        model.total_cost_con = Constraint(rule=total_cost_rule)
        
        def process_cost_rule(model, k, p):
            n = self.process_type_map[k]
            process_cost = self.capture_process_types[n]['process_cost']
            emission = self.capture_process_types[n]['emission']
            footprint = self.capture_process_types[n]['footprint']
            
            # Fixed costs
            fixed_cost = process_cost * model.w[k, p] + footprint * self.land_price * model.w[k, p]
            
            throughput = self.capture_process_types[n]['throughput']
            emission_cost = emission * throughput * self.carbon_tax * model.w[k, p]
            
            return model.cost_k[k, p] == fixed_cost + emission_cost
        
        model.process_cost_con = Constraint(model.K, model.P, rule=process_cost_rule)
        
        # Producer to capture process transport cost
        def producer_cost_rule(model, i, k, p):
            ft = self.fixed_trans_cost_producer
            vt = self.var_trans_cost_producer
            min_dist = partitions['min_dist_producer'][i][p]
            return model.cost_ik[i, k, p] == (
                ft * model.z_ik[i, k, p]
              + vt * min_dist * model.f_ik[i, k, p]
            )
        model.producer_cost_con = Constraint(model.I, model.K, model.P, rule=producer_cost_rule)
        
        # Capture process to reaction process transport cost
        def reactor_cost_rule(model, k, j, p):
            ft = self.fixed_trans_cost_reactor
            vt = self.var_trans_cost_reactor
            min_dist = partitions['min_dist_reactor'][j][p]
            return model.cost_kj[k, j, p] == (
                ft * model.z_kj[k, j, p]
              + vt * min_dist * model.f_kj[k, j, p]
            )
        model.reactor_cost_con = Constraint(model.K, model.J, model.P, rule=reactor_cost_rule)
        
        # CO2 production constraints
        def producer_limit_rule(model, i):
            return sum(model.f_ik[i, k, p] for k in model.K for p in model.P) <= self.co2_producers['production_rate'][i]
        model.producer_limit_con = Constraint(model.I, rule=producer_limit_rule)
        
        # CO2 throughput constraint - direct throughput without conversion
        def throughput_rule(model, k, p):
            return model.f_k[k, p] == sum(model.f_ik[i, k, p] for i in model.I)
        model.throughput_con = Constraint(model.K, model.P, rule=throughput_rule)
        
        # CO2 balance constraints
        def process_balance_rule(model, k, p):
            return model.f_k[k, p] == sum(model.f_kj[k, j, p] for j in model.J) + model.storage_k[k, p]
        model.process_balance_con = Constraint(model.K, model.P, rule=process_balance_rule)
        
        # CO2 reaction process consumption constraints
        def reactor_consumption_rule(model, j):
            return sum(model.f_kj[k, j, p] for k in model.K for p in model.P) == self.co2_reaction_processes['consumption_rate'][j]
        model.reactor_consumption_con = Constraint(model.J, rule=reactor_consumption_rule)
        
        # CO2 throughput constraint
        def overall_throughput_rule(model):
            total_production = sum(self.co2_producers['production_rate'][i] for i in model.I)
            return sum(model.f_k[k, p] for k in model.K for p in model.P) >= total_production
        model.overall_throughput_con = Constraint(rule=overall_throughput_rule)
        
        # Land area constraint
        def land_area_rule(model):
            return sum(self.capture_process_types[self.process_type_map[k]]['footprint'] * model.w[k, p] 
                    for k in model.K for p in model.P) <= self.available_land
        model.land_area_con = Constraint(rule=land_area_rule)
        
        # Logical constraints
        def process_producer_link1_rule(model, i, k, p):
            return model.w[k, p] >= model.z_ik[i, k, p]
        model.process_producer_link1_con = Constraint(model.I, model.K, model.P, rule=process_producer_link1_rule)
        
        def process_producer_link2_rule(model, k, p):
            return sum(model.z_ik[i, k, p] for i in model.I) >= model.w[k, p]
        model.process_producer_link2_con = Constraint(model.K, model.P, rule=process_producer_link2_rule)
        
        def process_reactor_link1_rule(model, k, j, p):
            return model.w[k, p] >= model.z_kj[k, j, p]
        model.process_reactor_link1_con = Constraint(model.K, model.J, model.P, rule=process_reactor_link1_rule)
        
        def process_reactor_link2_rule(model, k, p):
            return sum(model.z_kj[k, j, p] for j in model.J) >= model.w[k, p]
        model.process_reactor_link2_con = Constraint(model.K, model.P, rule=process_reactor_link2_rule)
        
        # Uniqueness constraints
        def process_uniqueness_rule(model, k):
            return sum(model.w[k, p] for p in model.P) <= 1
        model.process_uniqueness_con = Constraint(model.K, rule=process_uniqueness_rule)
        
        def producer_link_uniqueness_rule(model, i, k):
            return sum(model.z_ik[i, k, p] for p in model.P) <= 1
        model.producer_link_uniqueness_con = Constraint(model.I, model.K, rule=producer_link_uniqueness_rule)
        
        def reactor_link_uniqueness_rule(model, k, j):
            return sum(model.z_kj[k, j, p] for p in model.P) <= 1
        model.reactor_link_uniqueness_con = Constraint(model.K, model.J, rule=reactor_link_uniqueness_rule)
        
        # Symmetry breaking constraints
        def symmetry_breaking_rule(model, k1, k2):
            if k1 < k2 and self.process_type_map[k1] == self.process_type_map[k2]:
                return sum(model.w[k1, p] for p in model.P) >= sum(model.w[k2, p] for p in model.P)
            else:
                return Constraint.Skip
        model.symmetry_breaking_con = Constraint(model.K, model.K, rule=symmetry_breaking_rule)
        
        # Throughput capacity constraints
        def throughput_capacity_rule(model, k, p):
            n = self.process_type_map[k]
            capacity = self.capture_process_types[n]['throughput']
            return model.f_k[k, p] <= capacity * model.w[k, p]
        model.throughput_capacity_con = Constraint(model.K, model.P, rule=throughput_capacity_rule)
        
        # Flow bounds
        def producer_flow_bound_rule(model, i, k, p):
            return model.f_ik[i, k, p] <= self.co2_producers['production_rate'][i] * model.z_ik[i, k, p]
        model.producer_flow_bound_con = Constraint(model.I, model.K, model.P, rule=producer_flow_bound_rule)
        
        def reactor_flow_bound_rule(model, k, j, p):
            return model.f_kj[k, j, p] <= self.co2_reaction_processes['consumption_rate'][j] * model.z_kj[k, j, p]
        model.reactor_flow_bound_con = Constraint(model.K, model.J, model.P, rule=reactor_flow_bound_rule)
        
        # Get model statistics before solving
        master_stats = self.print_model_statistics(model, "MASTER PROBLEM")
        
        # Solve the model
        solver = SolverFactory('gurobi')
        solver.options['mipgap'] = 0.01  # 1% gap
        solver.options['timelimit'] = 300  # 5 minutes time limit
        
        results = solver.solve(model, tee=False)
        
        # Process the results
        if results.solver.termination_condition != TerminationCondition.optimal and \
        results.solver.termination_condition != TerminationCondition.maxTimeLimit:
            print(f"Master problem solver status: {results.solver.status}")
            print(f"Master problem termination condition: {results.solver.termination_condition}")
            raise RuntimeError("Failed to solve the master problem")
        
        # Extract the solution
        solution = {
            'objective': value(model.total_cost),
            'selected_processes': [],
            'process_partitions': {},
            'producer_links': {},
            'reactor_links': {},
            'flows': {
                'producer_process': {},
                'process_reactor': {},
                'process_throughput': {}
            },
            'statistics': master_stats
        }
        
        # Extract selected processes and their partitions
        for k in model.K:
            for p in model.P:
                if value(model.w[k, p]) > 0.5:
                    solution['selected_processes'].append(k)
                    solution['process_partitions'][k] = p
        
        # Extract links and flows
        for k in solution['selected_processes']:
            p = solution['process_partitions'][k]
            
            # Producer links and flows
            solution['producer_links'][k] = []
            solution['flows']['producer_process'][k] = {}
            
            for i in model.I:
                if value(model.z_ik[i, k, p]) > 0.5:
                    solution['producer_links'][k].append(i)
                    solution['flows']['producer_process'][k][i] = value(model.f_ik[i, k, p])
            
            # Reactor links and flows
            solution['reactor_links'][k] = []
            solution['flows']['process_reactor'][k] = {}
            
            for j in model.J:
                if value(model.z_kj[k, j, p]) > 0.5:
                    solution['reactor_links'][k].append(j)
                    solution['flows']['process_reactor'][k][j] = value(model.f_kj[k, j, p])
            
            # Process throughput
            solution['flows']['process_throughput'][k] = value(model.f_k[k, p])
        
        return solution
    
    def solve_subproblem(self, master_solution, partitions):
        # Create the model
        model = ConcreteModel()
        
        # Extract selected processes
        selected_processes = master_solution['selected_processes']
        
        if not selected_processes:
            raise ValueError("No processes selected in the master problem")
        
        # Sets
        model.I = RangeSet(0, self.num_producers - 1) 
        model.J = RangeSet(0, self.num_reactors - 1)  
        model.K = Set(initialize=selected_processes)  
            
        # Parameters - binary decisions fixed from master problem
        model.w = Param(model.K, initialize=lambda m, k: 1)  # All selected processes are active
        
        # Calculate producer and reactor links from master solution
        def z_ik_init(m, i, k):
            return 1 if i in master_solution['producer_links'].get(k, []) else 0
        
        def z_kj_init(m, k, j):
            return 1 if j in master_solution['reactor_links'].get(k, []) else 0
        
        model.z_ik = Param(model.I, model.K, initialize=z_ik_init)
        model.z_kj = Param(model.K, model.J, initialize=z_kj_init)
        
        # Calculate maximum possible distances for bounds
        max_x_dist = self.x_max - self.x_min
        max_y_dist = self.y_max - self.y_min
        max_dist = np.sqrt(max_x_dist**2 + max_y_dist**2)
        
        # Variables
        # Process coordinates
        model.x = Var(model.K, domain=NonNegativeReals)
        model.y = Var(model.K, domain=NonNegativeReals)
        
        # Distances with explicit upper bounds
        model.d_ik = Var(model.I, model.K, domain=NonNegativeReals, bounds=(0, max_dist))
        model.d_kj = Var(model.K, model.J, domain=NonNegativeReals, bounds=(0, max_dist))
        
        # Add auxiliary variables for squared distances to improve numerical stability
        model.d_ik_sq = Var(model.I, model.K, domain=NonNegativeReals, bounds=(0, max_dist**2))
        model.d_kj_sq = Var(model.K, model.J, domain=NonNegativeReals, bounds=(0, max_dist**2))
        
        # Flows
        model.f_ik = Var(model.I, model.K, domain=NonNegativeReals)
        model.f_kj = Var(model.K, model.J, domain=NonNegativeReals)
        model.f_k = Var(model.K, domain=NonNegativeReals)
        
        # Costs
        model.cost_k = Var(model.K, domain=NonNegativeReals)
        model.cost_ik = Var(model.I, model.K, domain=NonNegativeReals)
        model.cost_kj = Var(model.K, model.J, domain=NonNegativeReals)
        
        # Storage variable
        model.storage_k = Var(model.K, domain=NonNegativeReals)
        
        # Objective: minimize net cost consistent with master problem
        def obj_rule(model):
            # Base cost from process and transport
            base_cost = (
                sum(model.cost_k[k] for k in model.K) +
                sum(model.cost_ik[i, k] for i in model.I for k in model.K) +
                sum(model.cost_kj[k, j] for k in model.K for j in model.J)
            )
            # Revenue from CO₂ supply
            supply_rev = sum(
                self.co2_producers['cost'][i] * model.f_ik[i, k] 
                for i in model.I for k in model.K
            )
            # Revenue from CO₂ reaction
            reaction_rev = sum(
                self.co2_reaction_processes['profit'][j] * model.f_kj[k, j] 
                for k in model.K for j in model.J
            )
            # Small regularization term for stability
            reg_term = 1e-6 * (
                sum(model.x[k]**2 for k in model.K) +
                sum(model.y[k]**2 for k in model.K)
            )
            # Net cost = costs minus revenues + reg term
            return base_cost - supply_rev - reaction_rev + reg_term
        
        model.obj = Objective(rule=obj_rule, sense=minimize)
        
        def process_cost_rule(model, k):
            n = self.process_type_map[k]
            process_cost = self.capture_process_types[n]['process_cost']
            emission = self.capture_process_types[n]['emission']
            footprint = self.capture_process_types[n]['footprint']
            
            # Fixed costs
            fixed_cost = process_cost * model.w[k] + footprint * self.land_price * model.w[k]
            
            # Emission cost based on full capacity throughput
            throughput = self.capture_process_types[n]['throughput']
            emission_cost = emission * throughput * self.carbon_tax * model.w[k]
            
            return model.cost_k[k] == fixed_cost + emission_cost
        
        model.process_cost_con = Constraint(model.K, rule=process_cost_rule)
        
        # Producer-process transport cost
        def producer_cost_rule(model, i, k):
            if model.z_ik[i, k] < 0.5:
                return model.cost_ik[i, k] == 0
            else:
                ft = self.fixed_trans_cost_producer
                vt = self.var_trans_cost_producer
                return model.cost_ik[i, k] == (
                    ft * model.z_ik[i, k] + 
                    vt * model.d_ik[i, k] * model.f_ik[i, k]
                )
        model.producer_cost_con = Constraint(model.I, model.K, rule=producer_cost_rule)
        
        # Process-reactor transport cost
        def reactor_cost_rule(model, k, j):
            if model.z_kj[k, j] < 0.5:
                return model.cost_kj[k, j] == 0
            else:
                ft = self.fixed_trans_cost_reactor
                vt = self.var_trans_cost_reactor
                return model.cost_kj[k, j] == (
                    ft * model.z_kj[k, j] + 
                    vt * model.d_kj[k, j] * model.f_kj[k, j]
                )
        model.reactor_cost_con = Constraint(model.K, model.J, rule=reactor_cost_rule)
        
        # Modified distance constraints to handle both active and inactive links
        # Using auxiliary squared distance variables for better numerical stability
        def producer_distance_squared_rule(model, i, k):
            xi, yi = self.co2_producers['coordinates'][i]
            if model.z_ik[i, k] < 0.5:
                # For inactive links, set squared distance to zero
                return model.d_ik_sq[i, k] == 0
            else:
                # For active links, calculate the squared Euclidean distance
                return model.d_ik_sq[i, k] == (model.x[k] - xi)**2 + (model.y[k] - yi)**2
        model.producer_distance_squared_con = Constraint(model.I, model.K, rule=producer_distance_squared_rule)
        
        def reactor_distance_squared_rule(model, k, j):
            xj, yj = self.co2_reaction_processes['coordinates'][j]
            if model.z_kj[k, j] < 0.5:
                # For inactive links, set squared distance to zero
                return model.d_kj_sq[k, j] == 0
            else:
                # For active links, calculate the squared Euclidean distance
                return model.d_kj_sq[k, j] == (model.x[k] - xj)**2 + (model.y[k] - yj)**2
        model.reactor_distance_squared_con = Constraint(model.K, model.J, rule=reactor_distance_squared_rule)
        
        # Connect squared distances to actual distances
        def producer_distance_relation_rule(model, i, k):
            if model.z_ik[i, k] < 0.5:
                return model.d_ik[i, k] == 0
            else:
                # Adding a small epsilon to avoid division by zero issues
                return model.d_ik[i, k] * model.d_ik[i, k] == model.d_ik_sq[i, k] + 1e-10
        model.producer_distance_relation_con = Constraint(model.I, model.K, rule=producer_distance_relation_rule)
        
        def reactor_distance_relation_rule(model, k, j):
            if model.z_kj[k, j] < 0.5:
                return model.d_kj[k, j] == 0
            else:
                # Adding a small epsilon to avoid division by zero issues
                return model.d_kj[k, j] * model.d_kj[k, j] == model.d_kj_sq[k, j] + 1e-10
        model.reactor_distance_relation_con = Constraint(model.K, model.J, rule=reactor_distance_relation_rule)
        
        # CO2 production constraints
        def producer_limit_rule(model, i):
            return sum(model.f_ik[i, k] for k in model.K) <= self.co2_producers['production_rate'][i]
        model.producer_limit_con = Constraint(model.I, rule=producer_limit_rule)
        
        # CO2 throughput - direct throughput without conversion
        def throughput_rule(model, k):
            return sum(model.f_ik[i, k] for i in model.I) == model.f_k[k]
        model.throughput_con = Constraint(model.K, rule=throughput_rule)
        
        # CO2 balance constraints
        def process_balance_rule(model, k):
            return model.f_k[k] == sum(model.f_kj[k, j] for j in model.J) + model.storage_k[k]
        model.process_balance_con = Constraint(model.K, rule=process_balance_rule)
        
        # CO2 reaction process consumption constraints
        def reactor_consumption_rule(model, j):
            return sum(model.f_kj[k, j] for k in model.K) == self.co2_reaction_processes['consumption_rate'][j]
        model.reactor_consumption_con = Constraint(model.J, rule=reactor_consumption_rule)
        
        # CO2 throughput constraint - slightly relaxed for numerical stability
        def overall_throughput_rule(model):
            total_production = sum(self.co2_producers['production_rate'][i] for i in model.I)
            return sum(model.f_k[k] for k in model.K) >= total_production * 0.95  # Relaxed to 95%
        model.overall_throughput_con = Constraint(rule=overall_throughput_rule)
        
        # Throughput capacity constraints
        def throughput_capacity_rule(model, k):
            n = self.process_type_map[k]
            capacity = self.capture_process_types[n]['throughput']
            return model.f_k[k] <= capacity * model.w[k]
        model.throughput_capacity_con = Constraint(model.K, rule=throughput_capacity_rule)
        
        # Coordinate bounds based on selected partitions
        def x_bounds_rule(model, k):
            p = master_solution['process_partitions'][k]
            x_bounds = partitions['bounds'][p][0]
            return (x_bounds[0], x_bounds[1])
        
        def y_bounds_rule(model, k):
            p = master_solution['process_partitions'][k]
            y_bounds = partitions['bounds'][p][1]
            return (y_bounds[0], y_bounds[1])
        
        for k in model.K:
            model.x[k].setub(x_bounds_rule(model, k)[1])
            model.x[k].setlb(x_bounds_rule(model, k)[0])
            model.y[k].setub(y_bounds_rule(model, k)[1])
            model.y[k].setlb(y_bounds_rule(model, k)[0])
        
        # Flow bounds with modified constraints for inactive links
        def producer_flow_bound_rule(model, i, k):
            if model.z_ik[i, k] < 0.5:
                return model.f_ik[i, k] == 0
            else:
                return model.f_ik[i, k] <= self.co2_producers['production_rate'][i]
        model.producer_flow_bound_con = Constraint(model.I, model.K, rule=producer_flow_bound_rule)
        
        def reactor_flow_bound_rule(model, k, j):
            if model.z_kj[k, j] < 0.5:
                return model.f_kj[k, j] == 0
            else:
                return model.f_kj[k, j] <= self.co2_reaction_processes['consumption_rate'][j]
        model.reactor_flow_bound_con = Constraint(model.K, model.J, rule=reactor_flow_bound_rule)
        
        # Minimum distance between processes and fixed points - using squared distances
        def min_dist_producer_rule(model, i, k):
            if model.z_ik[i, k] < 0.5:
                return Constraint.Skip  # Skip if no link
            return model.d_ik_sq[i, k] >= self.min_distance**2
        model.min_dist_producer_con = Constraint(model.I, model.K, rule=min_dist_producer_rule)
        
        def min_dist_reactor_rule(model, k, j):
            if model.z_kj[k, j] < 0.5:
                return Constraint.Skip  # Skip if no link
            return model.d_kj_sq[k, j] >= self.min_distance**2
        model.min_dist_reactor_con = Constraint(model.K, model.J, rule=min_dist_reactor_rule)
        
        # Get model statistics before solving
        sub_stats = self.print_model_statistics(model, "SUBPROBLEM")
        
        # Initialize coordinates to midpoints of their partitions
        for k in model.K:
            p = master_solution['process_partitions'][k]
            x_mid, y_mid = partitions['midpoints'][p]
            model.x[k] = x_mid
            model.y[k] = y_mid
        
        # Multi-attempt solution approach with progressively relaxed settings
        for attempt in range(3):
            try:
                # Configure solver with appropriate options for this attempt
                solver = SolverFactory('ipopt')
                
                # Base options
                solver.options['max_iter'] = 5000
                solver.options['mu_strategy'] = 'adaptive'
                
                # Progressive relaxation based on attempt number
                if attempt == 0:
                    # First attempt: Standard settings
                    solver.options['tol'] = 1e-6
                    solver.options['acceptable_tol'] = 1e-4
                    solver.options['bound_push'] = 0.01
                    solver.options['acceptable_iter'] = 10
                elif attempt == 1:
                    # Second attempt: Relaxed tolerances
                    solver.options['tol'] = 1e-5
                    solver.options['acceptable_tol'] = 1e-3
                    solver.options['bound_push'] = 0.05
                    solver.options['bound_frac'] = 0.05
                    solver.options['acceptable_iter'] = 5
                    solver.options['watchdog_shortened_iter_trigger'] = 10
                else:
                    # Third attempt: Very relaxed settings with automatic scaling
                    solver.options['tol'] = 1e-4
                    solver.options['acceptable_tol'] = 1e-2
                    solver.options['bound_push'] = 0.1
                    solver.options['bound_frac'] = 0.1
                    solver.options['mu_init'] = 0.1
                    solver.options['nlp_scaling_method'] = 'gradient-based'
                    solver.options['acceptable_iter'] = 3
                    solver.options['acceptable_obj_change_tol'] = 1e-2
                    
                print(f"\nSubproblem solve attempt {attempt+1} with tol={solver.options['tol']}, acceptable_tol={solver.options['acceptable_tol']}")
                
                # Print verbose output for later attempts
                results = solver.solve(model, tee=(attempt > 0))
                
                # Check if solution is acceptable
                if (results.solver.status == SolverStatus.ok and 
                    (results.solver.termination_condition == TerminationCondition.optimal or
                    results.solver.termination_condition == TerminationCondition.maxTimeLimit)):
                    
                    # Extract the solution
                    solution = {
                        'objective': value(model.obj),
                        'processes': {},
                        'flows': {
                            'producer_process': {},
                            'process_reactor': {},
                            'process_throughput': {}
                        },
                        'statistics': sub_stats
                    }
                    
                    # Extract process locations and costs
                    for k in model.K:
                        solution['processes'][k] = {
                            'coordinates': (value(model.x[k]), value(model.y[k])),
                            'cost': value(model.cost_k[k]),
                            'producer_links': [],
                            'reactor_links': []
                        }
                        
                        # Extract producer links, flows and costs
                        solution['flows']['producer_process'][k] = {}
                        
                        for i in model.I:
                            if value(model.z_ik[i, k]) > 0.5:
                                solution['processes'][k]['producer_links'].append({
                                    'producer': i,
                                    'distance': value(model.d_ik[i, k]),
                                    'cost': value(model.cost_ik[i, k]),
                                    'flow': value(model.f_ik[i, k])
                                })
                                solution['flows']['producer_process'][k][i] = value(model.f_ik[i, k])
                        
                        # Extract reactor links, flows and costs
                        solution['flows']['process_reactor'][k] = {}
                        
                        for j in model.J:
                            if value(model.z_kj[k, j]) > 0.5:
                                solution['processes'][k]['reactor_links'].append({
                                    'reactor': j,
                                    'distance': value(model.d_kj[k, j]),
                                    'cost': value(model.cost_kj[k, j]),
                                    'flow': value(model.f_kj[k, j])
                                })
                                solution['flows']['process_reactor'][k][j] = value(model.f_kj[k, j])
                        
                        # Process throughput
                        solution['flows']['process_throughput'][k] = value(model.f_k[k])
                    
                    print(f"Subproblem solved successfully on attempt {attempt+1}")
                    return solution
            
            except Exception as e:
                print(f"Attempt {attempt+1} solver exception: {str(e)}")
                if attempt == 2:  # Last attempt failed
                    # Use fallback solution
                    print("All solver attempts failed. Creating an approximate solution...")
                    return self._create_approximate_solution(master_solution, partitions)
        
        # If all attempts fail, use a fallback approach
        print("All solution attempts failed. Creating an approximate solution...")
        return self._create_approximate_solution(master_solution, partitions)

    def _create_approximate_solution(self, master_solution, partitions):
        print("Creating approximate solution based on master problem...")
        approximate_solution = {
            'objective': master_solution['objective'] * 1.05,  # Estimate: 5% worse than master
            'processes': {},
            'flows': {
                'producer_process': master_solution['flows']['producer_process'].copy(),
                'process_reactor': master_solution['flows']['process_reactor'].copy(),
                'process_throughput': master_solution['flows']['process_throughput'].copy()
            },
            'statistics': {'approximated': True}
        }
        
        # Use midpoints of partitions as process locations
        for k in master_solution['selected_processes']:
            p = master_solution['process_partitions'][k]
            coordinates = partitions['midpoints'][p]
            
            # Create approximate process data
            approximate_solution['processes'][k] = {
                'coordinates': coordinates,
                'cost': 0,  
                'producer_links': [],
                'reactor_links': []
            }
            
            # Calculate approximate costs and links
            process_cost = 0
            for i in master_solution['producer_links'].get(k, []):
                producer_coords = self.co2_producers['coordinates'][i]
                dist = np.sqrt((coordinates[0] - producer_coords[0])**2 + 
                            (coordinates[1] - producer_coords[1])**2)
                flow = master_solution['flows']['producer_process'][k].get(i, 0)
                cost = self.fixed_trans_cost_producer + self.var_trans_cost_producer * dist * flow
                
                approximate_solution['processes'][k]['producer_links'].append({
                    'producer': i,
                    'distance': max(dist, self.min_distance),
                    'cost': cost,
                    'flow': flow
                })
                process_cost += cost
                
            for j in master_solution['reactor_links'].get(k, []):
                reactor_coords = self.co2_reaction_processes['coordinates'][j]
                dist = np.sqrt((coordinates[0] - reactor_coords[0])**2 + 
                            (coordinates[1] - reactor_coords[1])**2)
                flow = master_solution['flows']['process_reactor'][k].get(j, 0)
                cost = self.fixed_trans_cost_reactor + self.var_trans_cost_reactor * dist * flow
                
                approximate_solution['processes'][k]['reactor_links'].append({
                    'reactor': j,
                    'distance': max(dist, self.min_distance),
                    'cost': cost,
                    'flow': flow
                })
                process_cost += cost
            
            # Add process fixed costs
            n = self.process_type_map[k]
            process_fixed_cost = self.capture_process_types[n]['process_cost']
            footprint = self.capture_process_types[n]['footprint']
            emission = self.capture_process_types[n]['emission']
            throughput = self.capture_process_types[n]['throughput']
            
            process_cost += process_fixed_cost + footprint * self.land_price + emission * throughput * self.carbon_tax
            approximate_solution['processes'][k]['cost'] = process_cost
        
        # Recalculate total objective
        total_obj = sum(p['cost'] for p in approximate_solution['processes'].values())
        
        # Subtract revenues
        for k in master_solution['selected_processes']:
            for i in master_solution['producer_links'].get(k, []):
                flow = master_solution['flows']['producer_process'][k].get(i, 0)
                if i < len(self.co2_producers['cost']):  # Guard against index errors
                    total_obj -= self.co2_producers['cost'][i] * flow
                
            for j in master_solution['reactor_links'].get(k, []):
                flow = master_solution['flows']['process_reactor'][k].get(j, 0)
                if j < len(self.co2_reaction_processes['profit']):  # Guard against index errors
                    total_obj -= self.co2_reaction_processes['profit'][j] * flow
        
        approximate_solution['objective'] = total_obj
        print(f"Created approximate solution with objective: {total_obj}")
        
        return approximate_solution
    
    def run(self):
        # Initial partitioning
        px, py = 10, 10
        
        # Algorithm iterations
        iter_num = 0
        gap = float('inf')
        # Initialize cumulative timers
        total_master_time = 0.0
        total_sub_time = 0.0
        
        while iter_num < self.max_iter and gap > self.epsilon:
            iter_num += 1
            print(f"\nIteration {iter_num} (px={px}, py={py})")
            
            partitions = self.create_partitioning(px, py)
            
            print("Solving master problem...")
            start_time = time.time()
            master_solution = self.solve_master_problem(partitions)
            master_time = time.time() - start_time
            
            lower_bound = master_solution['objective']
            print(f"Master problem solved in {master_time:.2f} seconds")
            print(f"Lower bound: {lower_bound:.4f}")

            print("Solving subproblem...")
            start_time = time.time()
            sub_solution = self.solve_subproblem(master_solution, partitions)
            sub_time = time.time() - start_time
            
            upper_bound = sub_solution['objective']
            print(f"Subproblem solved in {sub_time:.2f} seconds")
            print(f"Upper bound: {upper_bound:.4f}")
            # Accumulate iteration times
            total_master_time += master_time
            total_sub_time += sub_time
            
            if upper_bound < self.best_objective:
                self.best_solution = sub_solution
                self.best_objective = upper_bound
            
            # Calculate gap
            gap = (upper_bound - lower_bound) / lower_bound
            print(f"Gap: {gap*100:.2f}%")
            
            # Store iteration results with model statistics
            iteration_stats = {
                'iter_num': iter_num,
                'px': px,
                'py': py,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'gap': gap,
                'master_time': master_time,
                'sub_time': sub_time,
                'master_solution': master_solution,
                'sub_solution': sub_solution,
                'master_stats': master_solution.get('statistics', {}),
                'sub_stats': sub_solution.get('statistics', {})
            }
            
            self.iterations.append(iteration_stats)
            self.lower_bounds.append(lower_bound)
            self.upper_bounds.append(upper_bound)
            self.model_statistics.append({
                'iter_num': iter_num,
                'master': master_solution.get('statistics', {}),
                'sub': sub_solution.get('statistics', {})
            })
            
            # Refine partitioning
            px += 5
            py += 5
            
            # Check for convergence
            if gap <= self.epsilon:
                print(f"\nConverged after {iter_num} iterations with gap {gap*100:.2f}%")
                break
        
        # Report total solve times
        print(f"\nTotal master problem solving time: {total_master_time:.2f} seconds")
        print(f"Total subproblem solving time: {total_sub_time:.2f} seconds")
        print(f"Overall solving time: {(total_master_time + total_sub_time):.2f} seconds")
        if iter_num == self.max_iter:
            print(f"\nReached maximum iterations ({self.max_iter}) with gap {gap*100:.2f}%")
        
        # Return the best solution found
        return self.best_solution
    
    def plot_solution(self, solution=None):
        if solution is None:
            if self.best_solution is None:
                raise ValueError("No solution to plot")
            solution = self.best_solution

        # Improved readability plotting
        fig, ax = plt.subplots(figsize=(10, 8))

        # Scatter nodes with distinct markers and colors
        # Producers
        pxs, pys = zip(*self.co2_producers['coordinates'])
        ax.scatter(pxs, pys, s=200, marker='o', edgecolor='k',
                   facecolor='skyblue', label='CO2 Producers', zorder=3)
        for i, (x,y) in enumerate(self.co2_producers['coordinates']):
            ax.annotate(f"P{i}", (x,y), xytext=(0,-12),
                        textcoords="offset points", ha='center',
                        fontsize=12, fontweight='bold', color='navy')

        # Capture processes
        cxs = [solution['processes'][k]['coordinates'][0] for k in solution['processes']]
        cys = [solution['processes'][k]['coordinates'][1] for k in solution['processes']]
        ks  = list(solution['processes'].keys())
        ax.scatter(cxs, cys, s=220, marker='^', edgecolor='k',
                   facecolor='limegreen', label='CO2 Capture Processes', zorder=4)
        for k, x, y in zip(ks, cxs, cys):
            ax.annotate(f"C{k}", (x,y), xytext=(0,10),
                        textcoords="offset points", ha='center',
                        fontsize=12, fontweight='bold', color='darkgreen')

        # Reactors
        rxs, rys = zip(*self.co2_reaction_processes['coordinates'])
        ax.scatter(rxs, rys, s=200, marker='s', edgecolor='k',
                   facecolor='salmon', label='CO2 Reaction Processes', zorder=3)
        for j, (x,y) in enumerate(self.co2_reaction_processes['coordinates']):
            ax.annotate(f"R{j}", (x,y), xytext=(0,10),
                        textcoords="offset points", ha='center',
                        fontsize=12, fontweight='bold', color='darkred')

        # Draw links: Producer -> Capture (blue dashed)
        first = True
        for k in solution['processes']:
            for link in solution['processes'][k]['producer_links']:
                i = link['producer']
                x0,y0 = self.co2_producers['coordinates'][i]
                x1,y1 = solution['processes'][k]['coordinates']
                ax.plot([x0,x1], [y0,y1], linestyle='--', linewidth=1.5,
                        color='dodgerblue', alpha=0.7,
                        label='Prod→Capt' if first else None, zorder=2)
                first = False

        # Draw links: Capture -> Reactor (red solid)
        first = True
        for k in solution['processes']:
            for link in solution['processes'][k]['reactor_links']:
                j = link['reactor']
                x0,y0 = solution['processes'][k]['coordinates']
                x1,y1 = self.co2_reaction_processes['coordinates'][j]
                ax.plot([x0,x1], [y0,y1], linestyle='-', linewidth=2,
                        color='crimson', alpha=0.6,
                        label='Capt→Reac' if first else None, zorder=1)
                first = False

        # Final touches
        ax.set_title(f"CO2 Capture Planning Solution (Total Cost: {solution['objective']:.2f})",
                     fontsize=14)
        ax.set_xlabel("X Coordinate", fontsize=12)
        ax.set_ylabel("Y Coordinate", fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.5)

        # Consolidate legend entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(),
                  loc='upper center', bbox_to_anchor=(0.5, -0.12),
                  ncol=4, frameon=False, fontsize=11)

        plt.tight_layout()
        return fig
    
    def plot_convergence(self):
        if not self.iterations:
            raise ValueError("No iterations to plot")
        
        plt.figure(figsize=(10, 6))
        
        # Plot lower and upper bounds
        iter_nums = [it['iter_num'] for it in self.iterations]
        plt.plot(iter_nums, self.lower_bounds, 'b-o', label='Lower Bound')
        plt.plot(iter_nums, self.upper_bounds, 'r-o', label='Upper Bound')
        
        # Plot gap
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        gaps = [it['gap'] * 100 for it in self.iterations]
        ax2.plot(iter_nums, gaps, 'g--', label='Gap (%)')
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Bound Value')
        ax2.set_ylabel('Gap (%)')
        
        # Add a legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        plt.title('Convergence of the Bilevel Decomposition Algorithm for CO2 Capture Planning')
        plt.grid(True)
        plt.tight_layout()
        
        return plt.gcf()
    
    def print_model_statistics_summary(self):
        if not self.model_statistics:
            print("No model statistics available")
            return
        
        print("\n========== MODEL STATISTICS SUMMARY ==========")
        print("Iteration | Master Problem               | Subproblem")
        print("          | Variables  | Constraints     | Variables | Constraints")
        print("-" * 75)
        
        for stats in self.model_statistics:
            iter_num = stats['iter_num']
            
            # Master problem statistics
            master_vars = stats['master'].get('variables', {}).get('total', 'N/A')
            master_cons = stats['master'].get('constraints', {}).get('total', 'N/A')
            
            # Subproblem statistics
            sub_vars = stats['sub'].get('variables', {}).get('total', 'N/A')
            sub_cons = stats['sub'].get('constraints', {}).get('total', 'N/A')
            
            print(f"{iter_num:9d} | {master_vars:10} | {master_cons:15} | {sub_vars:9} | {sub_cons}")
        
        print("=" * 75)


if __name__ == "__main__":
    # CO2 producer data
    co2_producers = {
        'coordinates': [(0, 0), (0, 5), (3, 3)],
        'production_rate': [120, 120, 160],
        'cost': [20e-3, 22e-3, 28e-3]
    }
    
    # CO2 reaction process data
    co2_reaction_processes = {
        'coordinates': [(5, 0), (5, 5), (2, 1)],
        'consumption_rate': [100, 100, 150],
        'profit': [30e-3, 20e-3, 50e-3]
    }
    
    # CO2 capture process types
    capture_process_types = [
        {
            'count': 2, # Physical adsorption
            'throughput': 135.09,
            'process_cost': 0.978,
            'emission': 0.327,
            'footprint': 1.332,
        },
        {
            'count': 2,  # MEA
            'throughput': 127.84,
            'process_cost': 0.640,
            'emission': 0.420,
            'footprint': 3.097,
        },
        {
            'count': 2, # Physical adsorption + Pressure temperature swing adsorption
            'throughput': 136.80,
            'process_cost': 0.988,
            'emission': 0.406,
            'footprint': 7.891,
        }
    ]
    
    # Create and run the solver with a tight convergence tolerance
    planner = CarbonCapturePlanner(
        co2_producers, 
        co2_reaction_processes, 
        capture_process_types,
        carbon_tax=17.64,       # $ per ton of CO₂
        land_price=300,      # $ per unit area
        available_land=5000, # Total available land
        epsilon=0.001,       # 0.1% tolerance
        max_iter=20
    )
    
    solution = planner.run()
    
    # Print model statistics summary
    planner.print_model_statistics_summary()
    
    # Plot the solution
    planner.plot_solution()
    plt.savefig('co2_capture_solution.png')
    
    # Plot the convergence
    planner.plot_convergence()
    plt.savefig('co2_capture_convergence.png')
    
    print("\nFinal solution:")
    print(f"Total cost: {solution['objective']:.4f}")
    print("CO2 capture process locations:")
    for k, process in solution['processes'].items():
        n = planner.process_type_map[k]
        process_type = planner.capture_process_types[n]
        print(f"Process {k} (Type {n}):")
        print(f"  Location: {process['coordinates']}")