# core/problem_utils.py
# -*- coding: utf-8 -*-
"""
Core utility functions, classes, and operators for the Multi-Depot, Two-Echelon
Vehicle Routing Problem with Drones and Split Deliveries (MD-2E-VRPSD).

This module serves as a foundational library of shared components for the entire
logistics optimization project. It is designed to prevent circular dependencies
by housing critical, problem-specific logic that is utilized by both the core
evaluation/optimization modules and the individual algorithm implementations.

Key Contents:
- Solution Representation: The `SolutionCandidate` class, which encapsulates the
  entire structure of a potential solution, including its routes, assignments,
  and evaluation metrics. This is the primary data structure manipulated by the
  optimization algorithms.

- Initial Solution Generation: Provides multiple strategies for creating an
  initial population or starting point for optimization algorithms:
  - `create_greedy_initial_solution`: A heuristic-based generator that
    produces a reasonably good, feasible starting point using nearest-neighbor logic.
  - `create_random_initial_solution`: A generator that produces a valid but
    entirely random solution, ideal for ensuring unbiased evaluation of
    metaheuristic search capabilities.
  - `create_initial_solution`: A factory function to select the desired
    generation strategy.

- Neighborhood and Perturbation Operators: A suite of functions to generate
  neighboring solutions for local search-based algorithms (like Simulated
  Annealing) or to serve as mutation operators in evolutionary algorithms.
  - `generate_neighbor_solution`: A generic function to apply a random
    perturbation.
  - Basic Mutations: `swap_mutation`, `inversion_mutation`, `scramble_mutation`.
  - Advanced Local Search Operators: `two_opt_mutation`.

- Genetic Algorithm Operators: Specialized crossover operators for permutation-based
  solutions, essential for algorithms like GA.
  - `partially_mapped_crossover_ox1`

- Stage 2 Heuristics: The critical heuristic (`create_heuristic_trips_split_delivery`)
  for generating the second-echelon delivery trips (from outlets to customers),
  which handles split deliveries and the coordinated use of vehicles and drones.

- Validation and Helpers: Utility functions for validating solution integrity and
  formatting output.
"""

# =================================================================================
#  Standard Library Imports
# =================================================================================
import copy
import math
import traceback
import sys
import os
import random
import time
import warnings
import uuid
from typing import List, Dict, Tuple, Optional, Any, Callable, Union

# =================================================================================
#  Third-Party Library Imports
# =================================================================================
import numpy as np

# =================================================================================
#  Safe Core Project Imports
# =================================================================================
# This block ensures robust imports of other core modules, adding the project
# root to the system path if necessary. This is crucial for running scripts
# from different directories or in various IDE configurations.
try:
    # Assumes this file is in project_root/core
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_path = os.path.dirname(current_dir)
    if project_root_path not in sys.path:
        sys.path.insert(0, project_root_path)

    # Import core dependencies. These are essential for the functionality of this module.
    # The cost function itself is designed to accept distance and Stage 2 generator
    # functions as parameters, cleanly avoiding circular import issues.
    from core.distance_calculator import haversine
    from core.cost_function import calculate_total_cost_and_evaluate, format_float

except ImportError as e:
    # If core dependencies are missing, the system cannot function.
    # We log a critical error and define dummy fallbacks to prevent a hard crash
    # on import, allowing the program to potentially start and show a proper error.
    print(f"CRITICAL ERROR in core.problem_utils: A core module failed to import: {e}")
    traceback.print_exc()


    # Define dummy functions/classes to allow the program to load but fail gracefully.
    def haversine(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Dummy haversine function for when the real module fails to import."""
        warnings.warn("DUMMY haversine function is being used due to an import error.")
        if not coord1 or not coord2 or len(coord1) != 2 or len(coord2) != 2:
            return float('inf')
        try:
            return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])
        except Exception:
            return float('inf')


    def calculate_total_cost_and_evaluate(*args: Any, **kwargs: Any) -> Tuple:
        """Dummy cost function for when the real module fails to import."""
        warnings.warn("DUMMY calculate_total_cost_and_evaluate is being used.")
        # Return a tuple matching the expected structure, indicating a total failure.
        return float('inf'), float('inf'), float('inf'), {}, True, True, {}


    # Define a dummy SolutionCandidate class if the real one fails to initialize due to import errors.
    # This class definition is provided later in the file, so an error here indicates a fundamental problem.
    warnings.warn("Problem_utils will use dummy functions due to a critical import failure.")

except Exception as e:
    print(f"An unexpected critical error occurred during the import block of problem_utils: {e}")
    traceback.print_exc()
    # Re-raising is appropriate here as the application state is unrecoverable.
    raise

# =================================================================================
#  Module-level Constants
# =================================================================================
# A small tolerance for floating-point comparisons, particularly for checking
# if demand has been fully met or if a value is effectively zero.
FLOAT_TOLERANCE: float = 1e-6


# =================================================================================
#  Solution Representation: The SolutionCandidate Class
# =================================================================================

class SolutionCandidate:
    """
    Represents a complete candidate solution for the MD-2E-VRPSD problem.

    This class serves as the fundamental data structure, or "chromosome," for the
    optimization algorithms. It encapsulates not only the structural components
    of a solution (routes and assignments) but also stores the results of its
    evaluation, such as cost, time, and feasibility. This unified structure
    simplifies the process of passing solutions between different parts of the
    system and enables consistent comparison logic.

    Attributes:
        id (str): A unique identifier for this solution instance.
        creation_timestamp (float): The timestamp when the solution was created.
        problem_data (Dict[str, Any]): A dictionary containing the static problem
            instance data, including 'locations' and initial 'demands'.
        vehicle_params (Dict[str, Any]): A dictionary of vehicle parameters.
        drone_params (Dict[str, Any]): A dictionary of drone parameters.
        objective_params (Dict[str, float]): A dictionary holding the weights and
            penalties for the objective function calculation.

        stage1_routes (Dict[int, List[int]]): The core genetic material of the
            solution. A dictionary where keys are depot indices and values are
            ordered lists of outlet indices, representing the first-echelon routes.
        outlet_to_depot_assignments (Dict[int, int]): A mapping of each sales
            outlet index to its assigned depot index.
        customer_to_outlet_assignments (Dict[int, int]): A mapping of each
            customer index to its assigned sales outlet index.

        # --- Evaluation Results (populated by the evaluate() method) ---
        stage2_trips (Dict[int, List[Dict]]): A dictionary holding the detailed
            second-echelon trips generated heuristically for each outlet.
        is_feasible (bool): True if all customer demand is met within the defined
            tolerance and no operational constraints are violated.
        weighted_cost (float): The total objective value, calculated as a weighted
            sum of raw cost, time, and penalties for unmet demand. This is the
            primary metric for comparing solutions.
        evaluated_cost (float): The raw transportation cost (Stage 1 + Stage 2).
        evaluated_time (float): The total solution makespan (maximum completion time).
        evaluated_unmet_demand (float): The total quantity of unmet customer demand.
        served_customer_details (Dict[int, Dict]): Detailed fulfillment status
            for each customer.
        evaluation_stage1_error (bool): Flag indicating if an error occurred
            during the evaluation of Stage 1 routes.
        evaluation_stage2_error (bool): Flag indicating if an error occurred
            during the generation or evaluation of Stage 2 trips.
        initialization_error (Optional[str]): A message describing any error
            that occurred during the object's initialization.
    """

    def __init__(self,
                 problem_data: Dict[str, Any],
                 vehicle_params: Dict[str, Any],
                 drone_params: Dict[str, Any],
                 objective_params: Dict[str, float],
                 initial_stage1_routes: Optional[Dict[int, List[int]]] = None,
                 initial_outlet_to_depot_assignments: Optional[Dict[int, int]] = None,
                 initial_customer_to_outlet_assignments: Optional[Dict[int, int]] = None):
        """
        Initializes a SolutionCandidate instance.

        Args:
            problem_data: Contains 'locations' and 'demands' for the VRP instance.
            vehicle_params: Specifies parameters for the vehicle fleet.
            drone_params: Specifies parameters for the drone fleet.
            objective_params: Contains 'cost_weight', 'time_weight', and
                'unmet_demand_penalty'.
            initial_stage1_routes: An optional dictionary defining the initial
                first-echelon routes.
            initial_outlet_to_depot_assignments: Optional mapping of outlets to depots.
            initial_customer_to_outlet_assignments: Optional mapping of customers to outlets.
        """
        self.id: str = str(uuid.uuid4())
        self.creation_timestamp: float = time.time()
        self.initialization_error: Optional[str] = None

        try:
            # --- Validate and Store Problem Data & Parameters ---
            if not isinstance(problem_data, dict) or 'locations' not in problem_data or 'demands' not in problem_data:
                raise ValueError("Invalid 'problem_data' structure. Must contain 'locations' and 'demands'.")
            self.problem_data = problem_data

            if not isinstance(vehicle_params, dict) or not isinstance(drone_params, dict) or not isinstance(
                    objective_params, dict):
                raise ValueError("'vehicle_params', 'drone_params', and 'objective_params' must be dictionaries.")
            self.vehicle_params = copy.deepcopy(vehicle_params)
            self.drone_params = copy.deepcopy(drone_params)
            self.objective_params = copy.deepcopy(objective_params)

            # --- Initialize Solution Structure ---
            # These components define the solution itself. They are deep-copied to ensure
            # that each SolutionCandidate instance is independent.
            self.stage1_routes = copy.deepcopy(initial_stage1_routes) if initial_stage1_routes is not None else {}
            self.outlet_to_depot_assignments = copy.deepcopy(
                initial_outlet_to_depot_assignments) if initial_outlet_to_depot_assignments is not None else {}
            self.customer_to_outlet_assignments = copy.deepcopy(
                initial_customer_to_outlet_assignments) if initial_customer_to_outlet_assignments is not None else {}

            # --- Initialize Evaluation Results to a Default (Unevaluated) State ---
            self._reset_evaluation_results()

        except Exception as e:
            warnings.warn(f"Error during SolutionCandidate initialization: {e}")
            self.initialization_error = str(e)
            self._set_to_invalid_state()

    def _reset_evaluation_results(self):
        """Resets all evaluation-related attributes to their default, unevaluated state."""
        self.stage2_trips: Dict[int, List[Dict]] = {}
        self.is_feasible: bool = False
        self.weighted_cost: float = float('inf')
        self.evaluated_cost: float = float('inf')
        self.evaluated_time: float = float('inf')
        self.evaluated_unmet_demand: float = float('inf')
        self.served_customer_details: Dict[int, Dict] = {}
        self.evaluation_stage1_error: bool = False
        self.evaluation_stage2_error: bool = False

    def _set_to_invalid_state(self):
        """Sets the entire object to a default invalid state upon critical error."""
        self.problem_data = {'locations': {}, 'demands': []}
        self.vehicle_params = {}
        self.drone_params = {}
        self.objective_params = {}
        self.stage1_routes = {}
        self.outlet_to_depot_assignments = {}
        self.customer_to_outlet_assignments = {}
        self._reset_evaluation_results()
        self.evaluation_stage1_error = True
        self.evaluation_stage2_error = True

    def evaluate(self,
                 distance_func: Callable,
                 stage2_trip_generator_func: Callable):
        """
        Evaluates the solution candidate by calculating its cost, time, and feasibility.

        This method orchestrates the evaluation process by calling the main
        `calculate_total_cost_and_evaluate` function. It populates the instance's
        evaluation attributes based on the results.

        Args:
            distance_func: The function to calculate distance between two coordinates.
                Expected signature: `distance_func(coord1, coord2) -> float`.
            stage2_trip_generator_func: The heuristic function to generate second-echelon
                trips from an outlet.
        """
        if self.initialization_error:
            warnings.warn(
                f"Evaluation skipped for SolutionCandidate '{self.id}' due to initialization error: {self.initialization_error}")
            return

        # Ensure evaluation results are reset before a new evaluation.
        self._reset_evaluation_results()

        if not callable(distance_func) or not callable(stage2_trip_generator_func):
            warnings.warn("Evaluation failed: Invalid 'distance_func' or 'stage2_trip_generator_func' provided.")
            self.evaluation_stage1_error = True
            self.evaluation_stage2_error = True
            self.weighted_cost = float('inf')
            self.is_feasible = False
            return

        try:
            # Unpack objective parameters for the call
            cost_weight = self.objective_params.get('cost_weight', 1.0)
            time_weight = self.objective_params.get('time_weight', 0.0)
            unmet_demand_penalty = self.objective_params.get('unmet_demand_penalty', 1e9)  # High default

            # Call the central evaluation function from core.cost_function
            (total_raw_cost, total_time_makespan, final_unmet_demand,
             served_customer_details, eval_s1_error, eval_s2_error,
             stage2_trips_details) = calculate_total_cost_and_evaluate(
                stage1_routes=self.stage1_routes,
                outlet_to_depot_assignments=self.outlet_to_depot_assignments,
                customer_to_outlet_assignments=self.customer_to_outlet_assignments,
                problem_data=self.problem_data,
                vehicle_params=self.vehicle_params,
                drone_params=self.drone_params,
                distance_func=distance_func,
                stage2_trip_generator_func=stage2_trip_generator_func,
                unmet_demand_penalty=unmet_demand_penalty,
                cost_weight=cost_weight,
                time_weight=time_weight
            )

            # Update instance attributes with the results
            self.evaluated_cost = total_raw_cost
            self.evaluated_time = total_time_makespan
            self.evaluated_unmet_demand = final_unmet_demand
            self.served_customer_details = served_customer_details
            self.evaluation_stage1_error = eval_s1_error
            self.evaluation_stage2_error = eval_s2_error
            self.stage2_trips = stage2_trips_details

            # Calculate the final weighted cost
            safe_cost = self.evaluated_cost if math.isfinite(self.evaluated_cost) else float('inf')
            safe_time = self.evaluated_time if math.isfinite(self.evaluated_time) else float('inf')
            safe_unmet = self.evaluated_unmet_demand if math.isfinite(self.evaluated_unmet_demand) else float('inf')

            self.weighted_cost = (cost_weight * safe_cost +
                                  time_weight * safe_time +
                                  unmet_demand_penalty * safe_unmet)

            # Determine feasibility based on evaluation results
            self.is_feasible = (not self.evaluation_stage1_error and
                                not self.evaluation_stage2_error and
                                math.isfinite(self.evaluated_unmet_demand) and
                                abs(self.evaluated_unmet_demand) < FLOAT_TOLERANCE)

            # If any evaluation error occurred, the solution is considered infeasible with infinite cost
            if self.evaluation_stage1_error or self.evaluation_stage2_error:
                self.is_feasible = False
                self.weighted_cost = float('inf')

        except Exception as e:
            warnings.warn(f"An unexpected error occurred during SolutionCandidate.evaluate(): {e}")
            traceback.print_exc()
            self._set_to_invalid_state()
            self.initialization_error = "Evaluation failed unexpectedly."

    def __lt__(self, other: 'SolutionCandidate') -> bool:
        """
        Compares this SolutionCandidate to another for sorting and selection.

        The comparison logic prioritizes feasibility first, then weighted cost. This
        is crucial for many optimization algorithms, as it guides the search
        towards valid solutions before optimizing the objective function.

        1. A feasible solution is always better than an infeasible one.
        2. If both have the same feasibility status, the one with the lower
           `weighted_cost` is considered better.

        Args:
            other: The other SolutionCandidate instance to compare against.

        Returns:
            True if this solution is strictly better than the other, False otherwise.
        """
        if not isinstance(other, SolutionCandidate):
            return NotImplemented

        # --- Feasibility-First Comparison ---
        if self.is_feasible and not other.is_feasible:
            return True  # Feasible is always better than infeasible
        if not self.is_feasible and other.is_feasible:
            return False  # Infeasible is always worse than feasible

        # --- Cost-Based Comparison (if feasibility is equal) ---
        # Handle potential None, NaN, or Inf values gracefully for robustness.
        self_cost = self.weighted_cost if math.isfinite(self.weighted_cost) else float('inf')
        other_cost = other.weighted_cost if math.isfinite(other.weighted_cost) else float('inf')

        return self_cost < other_cost

    def __repr__(self) -> str:
        """Provides a concise, developer-friendly string representation."""
        if self.initialization_error:
            return f"<SolutionCandidate ID={self.id} Error='{self.initialization_error}'>"

        cost_str = format_float(self.weighted_cost, 4)
        status = "Feasible" if self.is_feasible else "Infeasible"
        return (f"<SolutionCandidate ID={self.id} Status={status} "
                f"WCost={cost_str} Unmet={format_float(self.evaluated_unmet_demand, 2)}>")

    def get_summary(self) -> str:
        """Returns a more detailed, multi-line summary of the solution."""
        if self.initialization_error:
            return f"Solution Summary (Error):\n  - ID: {self.id}\n  - Error: {self.initialization_error}"

        summary_lines = [
            f"Solution Summary (ID: {self.id}):",
            f"  - Status: {'Feasible' if self.is_feasible else 'Infeasible'}",
            f"  - Weighted Cost: {format_float(self.weighted_cost, 4)}",
            "  ---------------------------------",
            f"  - Raw Transport Cost: {format_float(self.evaluated_cost, 2)}",
            f"  - Total Time (Makespan): {format_float(self.evaluated_time, 3)} hrs",
            f"  - Unmet Demand: {format_float(self.evaluated_unmet_demand, 4)}",
            f"  - Evaluation Errors: S1={self.evaluation_stage1_error}, S2={self.evaluation_stage2_error}",
            f"  - Stage 1 Routes Defined for: {len(self.stage1_routes)} depots",
        ]
        return "\n".join(summary_lines)

    def __deepcopy__(self, memo: Dict) -> 'SolutionCandidate':
        """
        Creates a deep copy of the SolutionCandidate instance.

        This is essential for genetic algorithms and other population-based methods
        to ensure that modifications to an offspring do not affect its parents or
        other individuals in the population.

        Args:
            memo: A dictionary used by the `copy` module to handle recursive copies.

        Returns:
            A new, independent SolutionCandidate instance.
        """
        # Create a new instance without calling __init__ to avoid re-validation.
        cls = self.__class__
        new_solution = cls.__new__(cls)
        memo[id(self)] = new_solution

        # Deep copy all mutable attributes
        for attr, value in self.__dict__.items():
            setattr(new_solution, attr, copy.deepcopy(value, memo))

        return new_solution


# =================================================================================
#  Solution Validation
# =================================================================================

def validate_solution_structure(solution: SolutionCandidate, problem_data: Dict[str, Any]) -> List[str]:
    """
    Validates the structural integrity of a solution's routes and assignments.

    Checks for issues like invalid indices, duplicate visits in a route, or
    assignments to non-existent entities.

    Args:
        solution: The SolutionCandidate object to validate.
        problem_data: The static problem data containing location counts.

    Returns:
        A list of string descriptions of any validation errors found. An empty
        list indicates the structure is valid.
    """
    errors = []
    num_depots = len(problem_data['locations'].get('logistics_centers', []))
    num_outlets = len(problem_data['locations'].get('sales_outlets', []))

    # Validate Stage 1 routes
    for depot_idx, route in solution.stage1_routes.items():
        if not (0 <= depot_idx < num_depots):
            errors.append(f"Stage 1: Invalid depot index {depot_idx} found in routes.")
            continue
        if len(route) != len(set(route)):
            errors.append(f"Stage 1: Duplicate outlet visits found in route for depot {depot_idx}.")
        for outlet_idx in route:
            if not (0 <= outlet_idx < num_outlets):
                errors.append(f"Stage 1: Route for depot {depot_idx} contains invalid outlet index {outlet_idx}.")

    # Further validation for assignments could be added here if needed.

    return errors


# =================================================================================
#  Initial Solution Generation Strategies
# =================================================================================

def create_initial_solution(strategy: str,
                            problem_data: Dict[str, Any],
                            vehicle_params: Dict[str, Any],
                            drone_params: Dict[str, Any],
                            objective_params: Dict[str, float]) -> Optional[SolutionCandidate]:
    """
    Factory function to create an initial solution using a specified strategy.

    This provides a single entry point for generating initial solutions, making it
    easy to switch between different initialization methods (e.g., 'greedy' vs. 'random')
    without changing the calling code in the main optimizer.

    Args:
        strategy: The name of the generation strategy to use.
                  Currently supported: 'greedy', 'random'.
        problem_data: The problem instance data.
        vehicle_params: Vehicle parameters.
        drone_params: Drone parameters.
        objective_params: Objective function weights and penalties.

    Returns:
        A new SolutionCandidate object, or None if the strategy is unknown or
        generation fails.
    """
    if strategy.lower() == 'greedy':
        return create_greedy_initial_solution(
            problem_data=problem_data,
            vehicle_params=vehicle_params,
            drone_params=drone_params,
            objective_params=objective_params
        )
    elif strategy.lower() == 'random':
        return create_random_initial_solution(
            problem_data=problem_data,
            vehicle_params=vehicle_params,
            drone_params=drone_params,
            objective_params=objective_params
        )
    else:
        warnings.warn(f"Unknown initial solution strategy: '{strategy}'. Returning None.")
        return None


def create_greedy_initial_solution(problem_data: Dict[str, Any],
                                   vehicle_params: Dict[str, Any],
                                   drone_params: Dict[str, Any],
                                   objective_params: Dict[str, float]) -> Optional[SolutionCandidate]:
    """
    Generates an initial solution using a greedy nearest-neighbor heuristic.

    This method produces a single, reasonably high-quality solution that can serve
    as a benchmark or a starting point for local search algorithms.
    The process involves:
    1. Assigning each outlet to its nearest depot.
    2. Assigning each customer to its nearest outlet.
    3. Constructing Stage 1 routes for each depot using a nearest-neighbor tour
       of its assigned outlets.

    Args:
        (Same as create_initial_solution factory)

    Returns:
        A new, evaluated SolutionCandidate, or None if generation fails.
    """
    print("Generating initial solution using GREEDY strategy...")
    start_time = time.time()

    try:
        # Extract location data for convenience
        locations = problem_data.get('locations', {})
        depot_locs = locations.get('logistics_centers', [])
        outlet_locs = locations.get('sales_outlets', [])
        customer_locs = locations.get('customers', [])
        num_depots, num_outlets, num_customers = len(depot_locs), len(outlet_locs), len(customer_locs)

        if num_depots == 0 or num_outlets == 0:
            warnings.warn("Cannot create greedy solution: At least one depot and one outlet are required.")
            return None

        # --- Step 1: Assign outlets to nearest depot ---
        outlet_to_depot_assignments: Dict[int, int] = {}
        depot_to_outlets_map: Dict[int, List[int]] = {i: [] for i in range(num_depots)}
        for o_idx, o_loc in enumerate(outlet_locs):
            dists = [(d_idx, haversine(o_loc, d_loc)) for d_idx, d_loc in enumerate(depot_locs)]
            nearest_depot_idx, _ = min(dists, key=lambda item: item[1])
            outlet_to_depot_assignments[o_idx] = nearest_depot_idx
            depot_to_outlets_map[nearest_depot_idx].append(o_idx)

        # --- Step 2: Assign customers to nearest outlet ---
        customer_to_outlet_assignments: Dict[int, int] = {}
        for c_idx, c_loc in enumerate(customer_locs):
            dists = [(o_idx, haversine(c_loc, o_loc)) for o_idx, o_loc in enumerate(outlet_locs)]
            nearest_outlet_idx, _ = min(dists, key=lambda item: item[1])
            customer_to_outlet_assignments[c_idx] = nearest_outlet_idx

        # --- Step 3: Construct Stage 1 routes via Nearest Neighbor ---
        stage1_routes: Dict[int, List[int]] = {}
        for d_idx in range(num_depots):
            assigned_outlets = depot_to_outlets_map.get(d_idx, [])
            if not assigned_outlets:
                stage1_routes[d_idx] = []
                continue

            # Start the tour from the depot
            current_location_coord = depot_locs[d_idx]
            unvisited_outlets = set(assigned_outlets)
            tour = []

            while unvisited_outlets:
                # Find the nearest unvisited outlet from the current location
                dists_to_unvisited = {o_idx: haversine(current_location_coord, outlet_locs[o_idx]) for o_idx in
                                      unvisited_outlets}
                nearest_outlet_idx = min(dists_to_unvisited, key=dists_to_unvisited.get)

                tour.append(nearest_outlet_idx)
                unvisited_outlets.remove(nearest_outlet_idx)
                current_location_coord = outlet_locs[nearest_outlet_idx]
            stage1_routes[d_idx] = tour

        # --- Step 4: Create and Evaluate the SolutionCandidate ---
        initial_solution = SolutionCandidate(
            problem_data=problem_data,
            vehicle_params=vehicle_params,
            drone_params=drone_params,
            objective_params=objective_params,
            initial_stage1_routes=stage1_routes,
            initial_outlet_to_depot_assignments=outlet_to_depot_assignments,
            initial_customer_to_outlet_assignments=customer_to_outlet_assignments
        )
        initial_solution.evaluate(haversine, create_heuristic_trips_split_delivery)

        end_time = time.time()
        print(f"Greedy initial solution generated in {end_time - start_time:.4f}s. "
              f"Feasible: {initial_solution.is_feasible}, WCost: {format_float(initial_solution.weighted_cost, 4)}")
        return initial_solution

    except Exception as e:
        warnings.warn(f"An unexpected error occurred during greedy initial solution generation: {e}")
        traceback.print_exc()
        return None


def create_random_initial_solution(problem_data: Dict[str, Any],
                                   vehicle_params: Dict[str, Any],
                                   drone_params: Dict[str, Any],
                                   objective_params: Dict[str, float]) -> Optional[SolutionCandidate]:
    """
    Generates an initial solution with RANDOM Stage 1 routes.

    This method is crucial for metaheuristic algorithms (GA, PSO, etc.) that
    require a diverse, un-optimized starting population. It ensures the search
    is not biased by a greedy heuristic.
    The process is:
    1. Assign outlets and customers to their nearest counterparts (like greedy).
       This step is maintained as it provides a logical decomposition of the
       problem space without overly optimizing the solution.
    2. **Crucially**, for each depot, the list of assigned outlets is
       **randomly shuffled** to create the Stage 1 tour, rather than being
       ordered by nearest neighbor.

    Args:
        (Same as create_initial_solution factory)

    Returns:
        A new, evaluated SolutionCandidate with randomized routes.
    """
    print("Generating initial solution using RANDOM strategy...")
    start_time = time.time()

    try:
        # Extract location data for convenience
        locations = problem_data.get('locations', {})
        depot_locs = locations.get('logistics_centers', [])
        outlet_locs = locations.get('sales_outlets', [])
        customer_locs = locations.get('customers', [])
        num_depots, num_outlets, num_customers = len(depot_locs), len(outlet_locs), len(customer_locs)

        if num_depots == 0 or num_outlets == 0:
            warnings.warn("Cannot create random solution: At least one depot and one outlet are required.")
            return None

        # --- Steps 1 & 2: Nearest-neighbor assignments (same as greedy) ---
        outlet_to_depot_assignments: Dict[int, int] = {}
        depot_to_outlets_map: Dict[int, List[int]] = {i: [] for i in range(num_depots)}
        for o_idx, o_loc in enumerate(outlet_locs):
            dists = [(d_idx, haversine(o_loc, d_loc)) for d_idx, d_loc in enumerate(depot_locs)]
            nearest_depot_idx, _ = min(dists, key=lambda item: item[1])
            outlet_to_depot_assignments[o_idx] = nearest_depot_idx
            depot_to_outlets_map[nearest_depot_idx].append(o_idx)

        customer_to_outlet_assignments: Dict[int, int] = {}
        for c_idx, c_loc in enumerate(customer_locs):
            dists = [(o_idx, haversine(c_loc, o_loc)) for o_idx, o_loc in enumerate(outlet_locs)]
            nearest_outlet_idx, _ = min(dists, key=lambda item: item[1])
            customer_to_outlet_assignments[c_idx] = nearest_outlet_idx

        # --- Step 3: Construct Stage 1 routes via RANDOM shuffling ---
        stage1_routes: Dict[int, List[int]] = {}
        for d_idx in range(num_depots):
            assigned_outlets = depot_to_outlets_map.get(d_idx, [])
            # Randomly shuffle the list of outlets to create the tour
            random.shuffle(assigned_outlets)
            stage1_routes[d_idx] = assigned_outlets

        # --- Step 4: Create and Evaluate the SolutionCandidate ---
        initial_solution = SolutionCandidate(
            problem_data=problem_data,
            vehicle_params=vehicle_params,
            drone_params=drone_params,
            objective_params=objective_params,
            initial_stage1_routes=stage1_routes,
            initial_outlet_to_depot_assignments=outlet_to_depot_assignments,
            initial_customer_to_outlet_assignments=customer_to_outlet_assignments
        )
        initial_solution.evaluate(haversine, create_heuristic_trips_split_delivery)

        end_time = time.time()
        print(f"Random initial solution generated in {end_time - start_time:.4f}s. "
              f"Feasible: {initial_solution.is_feasible}, WCost: {format_float(initial_solution.weighted_cost, 4)}")
        return initial_solution

    except Exception as e:
        warnings.warn(f"An unexpected error occurred during random initial solution generation: {e}")
        traceback.print_exc()
        return None


# =================================================================================
#  Neighborhood and Mutation Operators
# =================================================================================

def generate_neighbor_solution(current_solution: SolutionCandidate,
                               operator: Optional[Callable[[List], List]] = None) -> Optional[SolutionCandidate]:
    """
    Generates a neighbor solution by applying a perturbation operator to one of
    the Stage 1 routes of the current solution.

    This function acts as a generic mutation entry point. If no specific operator
    is provided, it randomly selects one from a predefined set of basic mutation
    operators.

    Args:
        current_solution: The base SolutionCandidate object.
        operator: An optional callable that takes a list (route) and returns a
                  perturbed list. If None, a random operator is chosen.

    Returns:
        A new, unevaluated neighbor SolutionCandidate, or None if generation fails.
    """
    if not isinstance(current_solution, SolutionCandidate) or current_solution.initialization_error:
        warnings.warn("Cannot generate neighbor: Invalid or errored current_solution provided.")
        return None

    try:
        neighbor_solution = copy.deepcopy(current_solution)
    except Exception as e:
        warnings.warn(f"Critical error during deepcopy for neighbor generation: {e}")
        traceback.print_exc()
        return None

    # Identify depots with routes that are long enough to be meaningfully perturbed
    eligible_depots = [idx for idx, route in neighbor_solution.stage1_routes.items() if len(route) >= 2]
    if not eligible_depots:
        # If no routes can be perturbed, return an un-modified (but still new) copy.
        # This can be handled by the calling algorithm.
        return neighbor_solution

    selected_depot_index = random.choice(eligible_depots)
    original_route = neighbor_solution.stage1_routes[selected_depot_index]

    # Select the mutation operator
    if operator is None:
        # If no operator is specified, choose one randomly from the basic set.
        mutation_operators = [swap_mutation, scramble_mutation, inversion_mutation]
        selected_operator = random.choice(mutation_operators)
    else:
        selected_operator = operator

    try:
        # Apply the chosen operator to the selected route
        perturbed_route = selected_operator(original_route)
        neighbor_solution.stage1_routes[selected_depot_index] = perturbed_route
    except Exception as e:
        warnings.warn(f"Error applying mutation operator '{selected_operator.__name__}' "
                      f"to route for depot {selected_depot_index}: {e}")
        traceback.print_exc()
        return None  # Indicate failure if the operator itself throws an error

    # The new neighbor is a valid structure but its performance is unknown.
    # Reset its evaluation results to reflect this.
    neighbor_solution._reset_evaluation_results()

    return neighbor_solution


# --- Basic Permutation Mutation Operators ---

def swap_mutation(route: List) -> List:
    """
    Performs a swap mutation on a route.

    Randomly selects two distinct positions in the route and swaps the elements
    at these positions. This is one of the simplest and most common mutation
    operators for permutation-based problems.

    Args:
        route: The list representing the route to be mutated.

    Returns:
        A new list containing the mutated route.
    """
    if len(route) < 2:
        return route[:]  # Return a copy if route is too short to swap

    route_copy = route[:]
    idx1, idx2 = random.sample(range(len(route_copy)), 2)
    route_copy[idx1], route_copy[idx2] = route_copy[idx2], route_copy[idx1]
    return route_copy


def inversion_mutation(route: List) -> List:
    """
    Performs an inversion (or reverse) mutation on a route.

    Randomly selects a sub-sequence within the route and reverses its order.
    This operator is effective at preserving adjacency information (edges) from
    the parent, as it only breaks two edges in the tour.

    Args:
        route: The list representing the route to be mutated.

    Returns:
        A new list containing the mutated route.
    """
    if len(route) < 2:
        return route[:]

    route_copy = route[:]
    size = len(route_copy)
    # Generate two distinct indices to define the sublist
    cut1, cut2 = random.sample(range(size + 1), 2)
    if cut1 > cut2:
        cut1, cut2 = cut2, cut1

    # Reverse the sublist in-place on the copy
    sublist_to_reverse = route_copy[cut1:cut2]
    sublist_to_reverse.reverse()
    route_copy[cut1:cut2] = sublist_to_reverse

    return route_copy


def scramble_mutation(route: List) -> List:
    """
    Performs a scramble mutation on a route.

    Randomly selects a sub-sequence within the route and shuffles the elements
    within that sub-sequence. This is a more disruptive mutation than swap or
    inversion.

    Args:
        route: The list representing the route to be mutated.

    Returns:
        A new list containing the mutated route.
    """
    if len(route) < 2:
        return route[:]

    route_copy = route[:]
    size = len(route_copy)
    cut1, cut2 = random.sample(range(size + 1), 2)
    if cut1 > cut2:
        cut1, cut2 = cut2, cut1

    # Scramble the sublist
    sublist_to_scramble = route_copy[cut1:cut2]
    random.shuffle(sublist_to_scramble)
    route_copy[cut1:cut2] = sublist_to_scramble

    return route_copy


# --- Advanced Local Search and Crossover Operators ---

def two_opt_mutation(route: List) -> List:
    """
    Performs a single, random 2-opt move on a route.

    The 2-opt algorithm is a classic local search heuristic for the Traveling
    Salesperson Problem (TSP). It works by removing two non-adjacent edges from a
    tour and reconnecting the two resulting paths in the only other possible way.
    This implementation performs one such move randomly, rather than iterating
    until a local optimum is found.

    This is generally more effective at finding good solutions than simple swaps.

    Args:
        route: The list representing the route.

    Returns:
        A new list containing the mutated route.
    """
    size = len(route)
    if size < 4:
        # 2-opt requires at least 4 nodes to be meaningful (to have non-adjacent edges)
        return route[:]

    route_copy = route[:]
    # Select two distinct indices i and k for the edges (i, i+1) and (k, k+1)
    i, k = random.sample(range(size - 1), 2)
    if i > k:
        i, k = k, i

    # To ensure non-adjacent edges, k must be greater than i+1
    if k == i + 1:
        # If adjacent edges were chosen, just return a copy to avoid invalid moves
        return route_copy

    # The 2-opt move reverses the segment between i+1 and k
    segment_to_reverse = route_copy[i + 1: k + 1]
    segment_to_reverse.reverse()
    route_copy[i + 1: k + 1] = segment_to_reverse

    return route_copy


def partially_mapped_crossover_ox1(parent1_route: List, parent2_route: List) -> Tuple[List, List]:
    """
    Performs Partially Mapped Crossover (PMX) on two parent routes.

    Note: The original `genetic_algorithm.py` file referenced an "Ordered Crossover (OX1)".
    PMX and OX1 are different operators. This implements PMX, which is often
    more effective at preserving absolute position information. I will also provide
    an OX1 implementation for completeness. This function will be called by a
    higher-level `crossover` function.

    PMX works by:
    1. Selecting a random sub-segment.
    2. Copying the segment from parent1 to child1 and from parent2 to child2.
    3. Creating a mapping of swaps from the segments.
    4. Filling the remaining positions in child1 with elements from parent2,
       using the mapping to resolve conflicts.

    Args:
        parent1_route: The first parent route (list).
        parent2_route: The second parent route (list).

    Returns:
        A tuple containing two new offspring routes.
    """
    size = len(parent1_route)
    if size < 2:
        return parent1_route[:], parent2_route[:]

    p1, p2 = parent1_route[:], parent2_route[:]
    offspring1, offspring2 = [None] * size, [None] * size

    # Step 1: Select a random swath
    cut1, cut2 = sorted(random.sample(range(size), 2))

    # Step 2: Copy the swath and create mappings
    mapping1 = {}
    mapping2 = {}
    for i in range(cut1, cut2 + 1):
        offspring1[i] = p2[i]
        offspring2[i] = p1[i]
        mapping1[p2[i]] = p1[i]
        mapping2[p1[i]] = p2[i]

    # Step 3: Fill in the rest of the offspring
    for i in list(range(cut1)) + list(range(cut2 + 1, size)):
        # Offspring 1
        gene = p1[i]
        while gene in mapping1:
            gene = mapping1[gene]
        offspring1[i] = gene

        # Offspring 2
        gene = p2[i]
        while gene in mapping2:
            gene = mapping2[gene]
        offspring2[i] = gene

    return offspring1, offspring2


def ordered_crossover_ox1(parent1_route: List, parent2_route: List) -> Tuple[List, List]:
    """
    Performs Ordered Crossover (OX1) on two parent routes.

    This was the intended operator in the original project's GA description. It
    preserves the relative order of elements.
    1. A random sub-segment is selected from one parent.
    2. This segment is copied directly to the offspring.
    3. The remaining positions in the offspring are filled with elements from the
       other parent in the order they appear, skipping elements already present.

    Args:
        parent1_route: The first parent route (list).
        parent2_route: The second parent route (list).

    Returns:
        A tuple containing two new offspring routes.
    """
    size = len(parent1_route)
    if size < 2:
        return parent1_route[:], parent2_route[:]

    p1, p2 = parent1_route[:], parent2_route[:]
    child1, child2 = [None] * size, [None] * size

    # Select two random cut points
    cut1, cut2 = sorted(random.sample(range(size), 2))

    # Copy the segment to the children
    child1[cut1:cut2 + 1] = p1[cut1:cut2 + 1]
    child2[cut1:cut2 + 1] = p2[cut1:cut2 + 1]

    # --- Fill Child 1 ---
    p2_elements_to_add = []
    # Iterate through parent 2 starting after the second cut point
    for i in range(size):
        idx = (cut2 + 1 + i) % size
        if p2[idx] not in child1:
            p2_elements_to_add.append(p2[idx])

    # Fill the 'None' slots in child 1
    fill_idx = 0
    for i in range(size):
        idx = (cut2 + 1 + i) % size
        if child1[idx] is None:
            child1[idx] = p2_elements_to_add[fill_idx]
            fill_idx += 1

    # --- Fill Child 2 ---
    p1_elements_to_add = []
    for i in range(size):
        idx = (cut2 + 1 + i) % size
        if p1[idx] not in child2:
            p1_elements_to_add.append(p1[idx])

    fill_idx = 0
    for i in range(size):
        idx = (cut2 + 1 + i) % size
        if child2[idx] is None:
            child2[idx] = p1_elements_to_add[fill_idx]
            fill_idx += 1

    return child1, child2


# =================================================================================
#  Heuristic for Stage 2 Trip Generation
# =================================================================================

def create_heuristic_trips_split_delivery(outlet_index: int,
                                          assigned_customer_indices: List[int],
                                          problem_data: Dict[str, Any],
                                          vehicle_params: Dict[str, Any],
                                          drone_params: Dict[str, Any],
                                          demands_remaining_global: Dict[int, float]) -> List[Dict[str, Any]]:
    """
    Heuristically generates Stage 2 delivery trips from a single sales outlet.

    This is a core greedy heuristic that constructs delivery trips for both
    vehicles and drones, allowing for split deliveries to fulfill customer demand.
    It iteratively creates trips until all assigned demand is met or no more
    trips can be formed. The logic prioritizes creating full vehicle trips first,
    then uses drones for smaller, eligible deliveries.

    IMPORTANT: This function modifies the `demands_remaining_global` dictionary
    in-place to reflect the fulfilled demand.

    Args:
        outlet_index: The index of the sales outlet from which trips originate.
        assigned_customer_indices: A list of customer indices assigned to this outlet.
        problem_data: Dictionary containing 'locations' and original 'demands'.
        vehicle_params: Parameters for vehicles ('payload', 'cost_per_km', 'speed_kmph').
        drone_params: Parameters for drones ('payload', 'max_flight_distance_km', etc.).
        demands_remaining_global: A dictionary mapping each customer index to their
            current remaining demand. This dictionary IS MODIFIED by the function.

    Returns:
        A list of dictionaries, where each dictionary describes a single generated
        trip (e.g., {'type': 'vehicle', 'route': [...], 'load': ..., 'cost': ..., 'time': ...}).
    """
    generated_trips: List[Dict[str, Any]] = []
    if not assigned_customer_indices:
        return generated_trips

    try:
        # --- Safely extract parameters and data ---
        locations = problem_data.get('locations', {})
        outlet_locs = locations.get('sales_outlets', [])
        customer_locs = locations.get('customers', [])

        if not (0 <= outlet_index < len(outlet_locs)):
            warnings.warn(f"Invalid outlet_index {outlet_index} passed to trip generator.")
            return generated_trips

        outlet_coord = outlet_locs[outlet_index]

        # Get vehicle/drone parameters with safe fallbacks
        veh_payload = vehicle_params.get('payload', 0.0)
        veh_cost_km = vehicle_params.get('cost_per_km', 0.0)
        veh_speed = vehicle_params.get('speed_kmph', 1.0)

        drone_payload = drone_params.get('payload', 0.0)
        drone_range = drone_params.get('max_flight_distance_km', 0.0)
        drone_cost_km = drone_params.get('cost_per_km', 0.0)
        drone_speed = drone_params.get('speed_kmph', 1.0)

        # --- Main trip generation loop ---
        # Create a local, mutable list of customers to serve for this outlet
        customers_to_serve = [
            idx for idx in assigned_customer_indices
            if demands_remaining_global.get(idx, 0.0) > FLOAT_TOLERANCE
        ]
        # Sort customers by distance from the outlet (simple greedy heuristic)
        customers_to_serve.sort(key=lambda cust_idx: haversine(outlet_coord, customer_locs[cust_idx]))

        while customers_to_serve:
            customers_served_in_pass = False

            # --- Attempt to form a vehicle trip first ---
            if veh_payload > FLOAT_TOLERANCE:
                current_veh_load = 0.0
                current_veh_route = []
                # Greedily pack the vehicle
                for cust_idx in list(customers_to_serve):  # Iterate over a copy
                    demand_to_serve = demands_remaining_global.get(cust_idx, 0.0)
                    if demand_to_serve <= FLOAT_TOLERANCE:
                        continue  # Already served

                    load_to_add = min(demand_to_serve, veh_payload - current_veh_load)

                    if load_to_add > FLOAT_TOLERANCE:
                        current_veh_load += load_to_add
                        demands_remaining_global[cust_idx] -= load_to_add
                        if cust_idx not in current_veh_route:
                            current_veh_route.append(cust_idx)

                        if demands_remaining_global[cust_idx] < FLOAT_TOLERANCE:
                            customers_to_serve.remove(cust_idx)  # Fully served

                    if current_veh_load >= veh_payload - FLOAT_TOLERANCE:
                        break  # Vehicle is full

                if current_veh_route:
                    customers_served_in_pass = True
                    # Calculate vehicle trip metrics
                    trip_coords = [outlet_coord] + [customer_locs[i] for i in current_veh_route] + [outlet_coord]
                    trip_dist = sum(haversine(trip_coords[i], trip_coords[i + 1]) for i in range(len(trip_coords) - 1))
                    trip_time = trip_dist / veh_speed if veh_speed > FLOAT_TOLERANCE else float('inf')
                    trip_cost = trip_dist * veh_cost_km
                    generated_trips.append({
                        'type': 'vehicle', 'route': current_veh_route, 'load': current_veh_load,
                        'cost': trip_cost, 'time': trip_time
                    })

            # --- Attempt to form drone trips for remaining demand ---
            if drone_payload > FLOAT_TOLERANCE and drone_range > FLOAT_TOLERANCE:
                # Drones make 1-to-1 trips. Iterate through remaining customers.
                for cust_idx in list(customers_to_serve):
                    demand_to_serve = demands_remaining_global.get(cust_idx, 0.0)
                    if demand_to_serve <= FLOAT_TOLERANCE:
                        continue

                    dist_to_cust = haversine(outlet_coord, customer_locs[cust_idx])
                    round_trip_dist = 2 * dist_to_cust

                    if round_trip_dist <= drone_range:
                        # This customer is drone-eligible. Send as many drone trips as needed.
                        while demands_remaining_global.get(cust_idx, 0.0) > FLOAT_TOLERANCE:
                            load_to_add = min(demands_remaining_global[cust_idx], drone_payload)
                            if load_to_add <= FLOAT_TOLERANCE:
                                break

                            customers_served_in_pass = True
                            demands_remaining_global[cust_idx] -= load_to_add
                            trip_time = round_trip_dist / drone_speed if drone_speed > FLOAT_TOLERANCE else float('inf')
                            trip_cost = round_trip_dist * drone_cost_km
                            generated_trips.append({
                                'type': 'drone', 'route': [cust_idx], 'load': load_to_add,
                                'cost': trip_cost, 'time': trip_time
                            })

                        if demands_remaining_global.get(cust_idx, 0.0) < FLOAT_TOLERANCE:
                            customers_to_serve.remove(cust_idx)  # Fully served

            # If a full pass over all customers yields no service, break to prevent infinite loops
            if not customers_served_in_pass:
                break

    except Exception as e:
        warnings.warn(f"Error during Stage 2 trip generation for outlet {outlet_index}: {e}")
        traceback.print_exc()

    return generated_trips


# =================================================================================
#  Standalone Execution Block for Testing
# =================================================================================

if __name__ == '__main__':
    """
    Provides a standalone execution context to test the functions within this module.
    This is invaluable for debugging and ensuring each utility works as expected
    before integrating it into the larger application.
    """
    print("=" * 80)
    print("Running core/problem_utils.py in Standalone Test Mode")
    print("=" * 80)

    # --- Setup Dummy Data for Comprehensive Testing ---
    try:
        print("\n[1] Creating Dummy Problem Data for Testing...")
        dummy_problem_data = {
            'locations': {
                'logistics_centers': [(40.0, -74.0), (41.0, -75.0)],
                'sales_outlets': [(40.1, -74.1), (40.2, -74.2), (40.8, -74.8), (40.9, -74.9)],
                'customers': [(40.11, -74.11), (40.12, -74.12), (40.21, -74.21),
                              (40.81, -74.81), (40.91, -74.91), (40.92, -74.92)]
            },
            'demands': [15.0, 25.0, 10.0, 30.0, 5.0, 40.0]
        }
        dummy_vehicle_params = {'payload': 100.0, 'cost_per_km': 2.0, 'speed_kmph': 50.0}
        dummy_drone_params = {'payload': 5.0, 'max_flight_distance_km': 10.0, 'cost_per_km': 0.5, 'speed_kmph': 80.0}
        dummy_objective_params = {'cost_weight': 0.6, 'time_weight': 0.4, 'unmet_demand_penalty': 10000.0}
        print("   ...Dummy data created successfully.")

    except Exception as e:
        print(f"   ...Error creating dummy data: {e}")
        sys.exit(1)

    # --- Test Initial Solution Generation Strategies ---
    print("\n[2] Testing Initial Solution Generation Strategies...")

    print("\n   --- Testing GREEDY Strategy ---")
    greedy_solution = create_initial_solution(
        'greedy', dummy_problem_data, dummy_vehicle_params, dummy_drone_params, dummy_objective_params
    )
    if greedy_solution:
        print(greedy_solution.get_summary())
    else:
        print("   ...Greedy solution generation failed.")

    print("\n   --- Testing RANDOM Strategy ---")
    random_solution = create_initial_solution(
        'random', dummy_problem_data, dummy_vehicle_params, dummy_drone_params, dummy_objective_params
    )
    if random_solution:
        print(random_solution.get_summary())
    else:
        print("   ...Random solution generation failed.")

    # --- Test Mutation and Local Search Operators ---
    if greedy_solution:
        print("\n[3] Testing Mutation and Local Search Operators...")
        base_route = greedy_solution.stage1_routes.get(0, [])
        if len(base_route) > 1:
            print(f"   Original Route (Depot 0): {base_route}")
            print(f"   -> Swap Mutation:      {swap_mutation(base_route)}")
            print(f"   -> Inversion Mutation: {inversion_mutation(base_route)}")
            print(f"   -> Scramble Mutation:  {scramble_mutation(base_route)}")
            print(f"   -> 2-Opt Mutation:     {two_opt_mutation(base_route)}")
        else:
            print("   ...Skipping mutation tests (route too short).")

        print("\n   --- Testing Neighbor Generation ---")
        neighbor = generate_neighbor_solution(greedy_solution)
        if neighbor:
            print(f"   Original Solution Feasible: {greedy_solution.is_feasible}")
            print(f"   Original S1 Routes: {greedy_solution.stage1_routes}")
            print(f"   Neighbor Generated. Neighbor S1 Routes: {neighbor.stage1_routes}")
            print(f"   Neighbor is unevaluated (Cost: {neighbor.weighted_cost})")
        else:
            print("   ...Neighbor generation failed.")

    # --- Test Crossover Operators ---
    if greedy_solution and random_solution:
        route1 = greedy_solution.stage1_routes.get(0)
        route2 = random_solution.stage1_routes.get(0)
        if route1 and route2 and len(route1) == len(route2):
            print("\n[4] Testing Crossover Operators...")
            print(f"   Parent 1 Route: {route1}")
            print(f"   Parent 2 Route: {route2}")

            print("\n   --- Ordered Crossover (OX1) ---")
            child1_ox1, child2_ox1 = ordered_crossover_ox1(route1, route2)
            print(f"   Child 1 (OX1): {child1_ox1}")
            print(f"   Child 2 (OX1): {child2_ox1}")

            print("\n   --- Partially Mapped Crossover (PMX) ---")
            child1_pmx, child2_pmx = partially_mapped_crossover_ox1(route1, route2)
            print(f"   Child 1 (PMX): {child1_pmx}")
            print(f"   Child 2 (PMX): {child2_pmx}")
        else:
            print("\n[4] Skipping Crossover Tests (routes are invalid or of different lengths).")

    print("\n" + "=" * 80)
    print("Standalone Test for problem_utils.py Finished")
    print("=" * 80)