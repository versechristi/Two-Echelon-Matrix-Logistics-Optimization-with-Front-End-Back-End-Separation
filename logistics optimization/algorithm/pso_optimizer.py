# algorithm/pso_optimizer.py
# -*- coding: utf-8 -*-
"""
An advanced, configurable implementation of Particle Swarm Optimization (PSO)
specifically adapted for the permutation-based nature of the Multi-Depot,
Two-Echelon Vehicle Routing Problem with Drones and Split Deliveries (MD-2E-VRPSD).

This module provides a robust PSO solver that explores the solution space by
simulating the social behavior of a swarm of particles. Each particle represents
a complete candidate solution and adjusts its "trajectory" based on its own
experience and the experience of its neighbors.

Key Architectural Features:
- **Permutation-Based PSO**: The core PSO logic is adapted to handle solutions
  that are represented as permutations (i.e., the sequence of outlet visits
  in Stage 1 routes). This is achieved by defining a particle's velocity as a
  sequence of "swap" operations.
- **Independent Initialization**: The swarm is initialized with a diverse set of
  randomly generated solutions, ensuring an unbiased start to the optimization
  process and allowing for a fair evaluation of the algorithm's search power.
- **Configurable Swarm Topologies**: To control the flow of information within
  the swarm and balance exploration vs. exploitation, this implementation supports
  multiple, selectable neighborhood topologies:
  - **Global Best (g-best)**: All particles are influenced by the single best
    solution found by the entire swarm. This leads to fast convergence but can
    be susceptible to getting trapped in local optima.
  - **Ring (l-best)**: Particles are arranged in a ring and are only influenced
    by the best solution found within their immediate neighborhood (e.g., the
    particle to the left and right). This slows convergence but enhances exploration.
  - **Von Neumann (l-best)**: Particles are arranged on a 2D grid and are
    influenced by their four cardinal neighbors (up, down, left, right). This
    provides a different, often effective, balance of exploration.
- **Inertia Weight Damping**: Implements a linearly decreasing inertia weight, a
  common technique that encourages more global exploration in the initial stages
  of the run and gradually shifts focus to local refinement as the search progresses.
- **Feasibility-Driven Search**: The PSO leverages the `SolutionCandidate` class's
  comparison logic, which inherently prioritizes feasible solutions over
  infeasible ones, guiding the swarm towards valid areas of the search space.
"""

# =================================================================================
#  Standard Library Imports
# =================================================================================
import random
import copy
import time
import math
import traceback
import sys
import os
import warnings
import logging
from typing import List, Dict, Tuple, Optional, Any, Callable, Union

# =================================================================================
#  Third-Party Library Imports
# =================================================================================
import numpy as np

# =================================================================================
#  Logging Configuration
# =================================================================================
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] (PSO) %(message)s"
    )

# =================================================================================
#  Safe Core Project Imports
# =================================================================================
try:
    from core.problem_utils import (
        SolutionCandidate,
        create_initial_solution,
    )
    from core.distance_calculator import haversine
    from core.cost_function import calculate_total_cost_and_evaluate
    from core.problem_utils import create_heuristic_trips_split_delivery

except ImportError as e:
    logger.critical(f"A core module failed to import, which is essential for the PSO. Error: {e}", exc_info=True)


    # Define dummy fallbacks to allow the script to load but fail gracefully
    class SolutionCandidate:
        def __init__(self, *args, **kwargs): self.weighted_cost = float('inf'); self.is_feasible = False

        def evaluate(self, *args, **kwargs): pass

        def __lt__(self, other): return False


    def create_initial_solution(*args, **kwargs):
        return None


    def haversine(c1, c2):
        return 0


    def calculate_total_cost_and_evaluate(*args, **kwargs):
        return float('inf'), float('inf'), float('inf'), {}, True, True, {}


    def create_heuristic_trips_split_delivery(*args, **kwargs):
        return []


    warnings.warn("PSO will use dummy functions due to a critical import failure.")

# =================================================================================
#  Module-level Constants
# =================================================================================
FLOAT_TOLERANCE_PSO = 1e-6


# =================================================================================
#  PSO Operator Functions for Permutations
# =================================================================================

def subtract_permutations(p1: List, p2: List) -> List[Tuple[int, int]]:
    """
    Calculates the "difference" between two permutations, `p2 - p1`, resulting
    in a sequence of swap operations that can transform `p1` into `p2`. This
    swap sequence is the conceptual basis for velocity in permutation-based PSO.

    Args:
        p1: The starting permutation (e.g., a particle's current position).
        p2: The target permutation (e.g., a personal or global best position).

    Returns:
        A list of tuples, where each tuple `(i, j)` represents a swap between
        the elements at index `i` and `j`.
    """
    if len(p1) != len(p2) or set(p1) != set(p2):
        warnings.warn("Cannot subtract permutations of different lengths or with different elements.")
        return []

    p1_copy = list(p1)
    swaps = []
    val_to_idx = {val: i for i, val in enumerate(p1_copy)}

    for i in range(len(p1_copy)):
        if p1_copy[i] != p2[i]:
            # The element that should be at index `i` is `p2[i]`.
            # Find where `p2[i]` is currently located in our evolving `p1_copy`.
            current_pos_of_target = val_to_idx[p2[i]]
            element_to_displace = p1_copy[i]

            # Perform the swap
            p1_copy[i], p1_copy[current_pos_of_target] = p1_copy[current_pos_of_target], p1_copy[i]
            swaps.append((i, current_pos_of_target))

            # Update the index map for the two swapped elements
            val_to_idx[element_to_displace] = current_pos_of_target
            val_to_idx[p2[i]] = i

    return swaps


def apply_velocity_to_permutation(p: List, velocity: List[Tuple[int, int]]) -> List:
    """
    Applies a velocity (a sequence of swaps) to a permutation to produce a new
    permutation, effectively updating the particle's position.

    Args:
        p: The starting permutation (current position).
        velocity: A list of swap operations to apply.

    Returns:
        A new list representing the updated permutation.
    """
    p_new = list(p)
    for i, j in velocity:
        if 0 <= i < len(p_new) and 0 <= j < len(p_new):
            p_new[i], p_new[j] = p_new[j], p_new[i]
        else:
            warnings.warn(
                f"Invalid swap indices ({i}, {j}) encountered for permutation of size {len(p_new)}. Skipping swap.")
    return p_new


def scale_velocity(velocity: List[Tuple[int, int]], factor: float) -> List[Tuple[int, int]]:
    """
    Scales a velocity (swap sequence) by a given factor. This is done by
    randomly selecting a subset of the swaps.

    Args:
        velocity: The full swap sequence to scale.
        factor: A float between 0.0 and 1.0 representing the scaling factor.

    Returns:
        A new, smaller list of swap operations.
    """
    if factor <= 0 or not velocity:
        return []
    if factor >= 1.0:
        return velocity

    num_swaps_to_keep = int(round(len(velocity) * factor))
    return random.sample(velocity, num_swaps_to_keep)


# =================================================================================
#  Swarm Topology Manager
# =================================================================================

class SwarmTopology:
    """
    Manages the neighborhood structure of the PSO swarm.

    This class is responsible for determining which particles can communicate
    with each other. It provides a unified interface to get the "local best"
    solution for any given particle, regardless of the underlying topology.
    """

    def __init__(self, topology_name: str, num_particles: int, grid_dimensions: Optional[Tuple[int, int]] = None):
        """
        Initializes the topology manager.

        Args:
            topology_name: The name of the topology ('global', 'ring', 'von_neumann').
            num_particles: The total number of particles in the swarm.
            grid_dimensions: A tuple (rows, cols) required for 'von_neumann' topology.

        Raises:
            ValueError: If an invalid topology name or configuration is provided.
        """
        self.topology_name = topology_name.lower()
        self.num_particles = num_particles
        self.neighborhoods = self._create_neighborhoods(grid_dimensions)

    def _create_neighborhoods(self, grid_dimensions: Optional[Tuple[int, int]]) -> Dict[int, List[int]]:
        """Builds the neighborhood mapping for each particle."""
        neighborhoods = {}
        if self.topology_name == 'global':
            # In g-best, every particle's neighborhood is the entire swarm.
            all_indices = list(range(self.num_particles))
            for i in range(self.num_particles):
                neighborhoods[i] = all_indices
        elif self.topology_name == 'ring':
            for i in range(self.num_particles):
                # Neighbors are the particle itself and its immediate left and right.
                left_neighbor = (i - 1 + self.num_particles) % self.num_particles
                right_neighbor = (i + 1) % self.num_particles
                neighborhoods[i] = [left_neighbor, i, right_neighbor]
        elif self.topology_name == 'von_neumann':
            if not grid_dimensions or grid_dimensions[0] * grid_dimensions[1] != self.num_particles:
                raise ValueError("Valid grid_dimensions are required for von_neumann topology.")
            rows, cols = grid_dimensions
            for i in range(self.num_particles):
                row, col = divmod(i, cols)
                neighbors = [i]  # Include self
                # Up, Down, Left, Right with wraparound
                neighbors.append(((row - 1 + rows) % rows) * cols + col)  # Up
                neighbors.append(((row + 1) % rows) * cols + col)  # Down
                neighbors.append(row * cols + (col - 1 + cols) % cols)  # Left
                neighbors.append(row * cols + (col + 1) % cols)  # Right
                neighborhoods[i] = list(set(neighbors))  # Use set to handle duplicates at edges
        else:
            raise ValueError(f"Unknown topology: {self.topology_name}")

        return neighborhoods

    def get_local_best_solution(self, particle_index: int,
                                swarm_particles: List['PermutationParticle']) -> SolutionCandidate:
        """
        Finds the best solution within the neighborhood of a given particle.

        Args:
            particle_index: The index of the particle whose neighborhood to check.
            swarm_particles: The entire list of particles in the swarm.

        Returns:
            The `SolutionCandidate` object representing the best personal-best
            solution found by any particle in the specified neighborhood.
        """
        neighborhood_indices = self.neighborhoods.get(particle_index, [])
        if not neighborhood_indices:
            # Fallback to the particle's own p-best if neighborhood is empty
            return swarm_particles[particle_index].personal_best_solution

        # Find the best p-best solution among all neighbors
        best_neighbor_solution = min(
            (swarm_particles[i].personal_best_solution for i in neighborhood_indices),
            key=lambda sol: sol.weighted_cost if math.isfinite(sol.weighted_cost) else float('inf')
        )
        return best_neighbor_solution


# =================================================================================
#  Permutation Particle Class
# =================================================================================

class PermutationParticle:
    """
    Represents a single particle in the permutation-based PSO.

    Each particle encapsulates a complete solution to the VRP, its velocity,
    and its personal best-found solution.
    """

    def __init__(self, initial_solution: SolutionCandidate):
        """
        Initializes a particle.

        Args:
            initial_solution: A `SolutionCandidate` object representing the
                particle's starting position in the solution space.
        """
        if not isinstance(initial_solution, SolutionCandidate):
            raise TypeError("PermutationParticle must be initialized with a valid SolutionCandidate.")

        # --- Position ---
        self.position: SolutionCandidate = initial_solution

        # --- Personal Best ---
        # The best position this specific particle has ever found.
        self.personal_best_solution: SolutionCandidate = copy.deepcopy(initial_solution)

        # --- Velocity ---
        # Represented as a dictionary where each key is a depot index and the value
        # is a list of swap operations for that depot's route.
        # Initial velocity is zero (empty swap lists).
        self.velocity: Dict[int, List[Tuple[int, int]]] = {
            depot_idx: [] for depot_idx in self.position.stage1_routes.keys()
        }

    def update_personal_best(self):
        """
        Updates the particle's personal best solution if its current position is
        an improvement. The comparison correctly prioritizes feasibility.
        """
        if self.position < self.personal_best_solution:
            self.personal_best_solution = copy.deepcopy(self.position)

    def update_velocity(self,
                        local_best_solution: SolutionCandidate,
                        inertia_weight: float,
                        cognitive_weight: float,
                        social_weight: float):
        """
        Updates the particle's velocity based on the standard PSO formula,
        adapted for permutation-based representations.

        The formula is: v(t+1) = w*v(t) + c1*r1*(p_best - x(t)) + c2*r2*(l_best - x(t))
        - `w*v(t)`: Inertia component.
        - `c1*r1*(p_best - x(t))`: Cognitive component (attraction to personal best).
        - `c2*r2*(l_best - x(t))`: Social component (attraction to neighborhood best).

        This update is performed independently for each depot's route permutation.

        Args:
            local_best_solution: The best solution found within this particle's neighborhood.
            inertia_weight (w): The inertia factor.
            cognitive_weight (c1): The cognitive (personal) learning factor.
            social_weight (c2): The social (neighborhood) learning factor.
        """
        r1, r2 = random.random(), random.random()
        new_velocity: Dict[int, List[Tuple[int, int]]] = {}

        for depot_idx, current_route in self.position.stage1_routes.items():
            if not current_route or len(current_route) < 2:
                new_velocity[depot_idx] = []
                continue

            # --- Calculate the three velocity components ---

            # 1. Inertia Component (w * v(t))
            # Scale the current velocity by the inertia weight.
            inertia_component = scale_velocity(self.velocity.get(depot_idx, []), inertia_weight)

            # 2. Cognitive Component (c1*r1 * (p_best - x(t)))
            # Subtract permutations to get the swap sequence toward p-best.
            pbest_route = self.personal_best_solution.stage1_routes.get(depot_idx, [])
            cognitive_diff = subtract_permutations(current_route, pbest_route)
            cognitive_component = scale_velocity(cognitive_diff, cognitive_weight * r1)

            # 3. Social Component (c2*r2 * (l_best - x(t)))
            # Subtract permutations to get the swap sequence toward l-best.
            lbest_route = local_best_solution.stage1_routes.get(depot_idx, [])
            social_diff = subtract_permutations(current_route, lbest_route)
            social_component = scale_velocity(social_diff, social_weight * r2)

            # --- Combine Components ---
            # The new velocity is the concatenation of the swap sequences from each component.
            # The order of application is implicitly handled by `apply_velocity_to_permutation`.
            new_velocity[depot_idx] = inertia_component + cognitive_component + social_component

        self.velocity = new_velocity

    def update_position(self):
        """
        Updates the particle's position by applying its velocity.

        The formula is: x(t+1) = x(t) + v(t+1)
        For permutations, this means applying the sequence of swaps in the velocity
        to the current position's routes.
        """
        new_position_solution = copy.deepcopy(self.position)

        for depot_idx, route_velocity in self.velocity.items():
            if route_velocity:
                current_route = new_position_solution.stage1_routes[depot_idx]
                new_route = apply_velocity_to_permutation(current_route, route_velocity)
                new_position_solution.stage1_routes[depot_idx] = new_route

        # The new position is unevaluated. Reset its metrics.
        new_position_solution._reset_evaluation_results()
        self.position = new_position_solution


# =================================================================================
#  PSO Main Orchestration Function
# =================================================================================

def run_pso_optimizer(
        problem_data: Dict[str, Any],
        vehicle_params: Dict[str, Any],
        drone_params: Dict[str, Any],
        objective_params: Dict[str, float],
        algo_specific_params: Dict[str, Any],
        initial_solution_candidate: Optional[SolutionCandidate] = None  # Ignored, for signature consistency
) -> Dict[str, Any]:
    """
    Executes the Particle Swarm Optimization algorithm.

    This function orchestrates the entire PSO run, including swarm initialization,
    the main iterative loop for updating particle velocities and positions, and
    the final packaging of results. It is configured via the `algo_specific_params`
    dictionary.

    Args:
        problem_data, vehicle_params, drone_params, objective_params: Standard
            problem definition dictionaries.
        algo_specific_params: A dictionary of PSO-specific hyperparameters. Expected keys:
            - `num_particles` (int)
            - `max_iterations` (int)
            - `inertia_weight` (float): Initial inertia weight `w`.
            - `inertia_damping_ratio` (float): Ratio to reduce `w` over iterations.
            - `cognitive_weight` (float): Cognitive learning factor `c1`.
            - `social_weight` (float): Social learning factor `c2`.
            - `topology` (str): 'global', 'ring', or 'von_neumann'.
        initial_solution_candidate: Included for signature consistency with other
            algorithms but is *ignored*. The PSO always generates its own random swarm.

    Returns:
        A dictionary containing the results of the PSO run, including the best
        solution found, its evaluation metrics, and performance history.
    """
    run_start_time = time.time()
    logger.info("--- Particle Swarm Optimization (MD-2E-VRPSD) Started ---")

    # --- 1. Validate and Configure PSO Parameters ---
    logger.info("Configuring PSO parameters...")
    try:
        params = _configure_pso_parameters(algo_specific_params)
        logger.info(f"PSO Configuration: Particles={params['num_particles']}, Iterations={params['max_iterations']}, "
                    f"Topology='{params['topology']}', Inertia(w)={params['inertia_weight']}, "
                    f"Cognitive(c1)={params['cognitive_weight']}, Social(c2)={params['social_weight']}")
    except (ValueError, KeyError) as e:
        error_msg = f"PSO parameter validation failed: {e}"
        logger.error(error_msg, exc_info=True)
        return {'run_error': error_msg}

    # --- 2. Initialize Swarm and Topology ---
    logger.info(f"Initializing swarm of {params['num_particles']} particles...")
    try:
        swarm = _initialize_swarm(
            num_particles=params['num_particles'],
            problem_data=problem_data,
            vehicle_params=vehicle_params,
            drone_params=drone_params,
            objective_params=objective_params
        )
        if not swarm:
            raise RuntimeError("Swarm initialization returned an empty list.")

        topology_manager = SwarmTopology(params['topology'], params['num_particles'])

        # Determine the initial global best solution from the initial swarm
        global_best_solution = min((p.position for p in swarm), key=lambda sol: sol.weighted_cost)

    except Exception as e:
        error_msg = f"Failed to initialize PSO swarm or topology: {e}"
        logger.error(error_msg, exc_info=True)
        return {'run_error': error_msg}

    # --- 3. Main PSO Iteration Loop ---
    logger.info("Starting PSO iteration loop...")
    cost_history = []

    # Initialize inertia weight for damping
    w_initial = params['inertia_weight']
    w_final = params['inertia_weight'] * params['inertia_damping_ratio']

    for i in range(params['max_iterations']):
        # Update inertia weight using linear damping
        current_inertia = w_initial - (w_initial - w_final) * (i / params['max_iterations'])

        # Iterate through each particle in the swarm
        for particle_idx, particle in enumerate(swarm):
            # --- a. Update Particle Velocity ---
            # Get the local best solution for this particle based on the swarm topology
            local_best = topology_manager.get_local_best_solution(particle_idx, swarm)

            particle.update_velocity(
                local_best_solution=local_best,
                inertia_weight=current_inertia,
                cognitive_weight=params['cognitive_weight'],
                social_weight=params['social_weight']
            )

            # --- b. Update Particle Position ---
            particle.update_position()

            # --- c. Evaluate New Position ---
            particle.position.evaluate(haversine, create_heuristic_trips_split_delivery)

            # --- d. Update Personal and Global Bests ---
            particle.update_personal_best()
            if particle.personal_best_solution < global_best_solution:
                global_best_solution = copy.deepcopy(particle.personal_best_solution)

        # --- e. Log Progress ---
        cost_history.append(global_best_solution.weighted_cost)
        if (i + 1) % 10 == 0 or i == params['max_iterations'] - 1:
            logger.info(f"Iteration {i + 1}/{params['max_iterations']} | "
                        f"Global Best Cost: {format_float(global_best_solution.weighted_cost, 2)}")

    # --- 4. Finalization and Result Packaging ---
    run_end_time = time.time()
    logger.info(f"--- PSO Finished in {run_end_time - run_start_time:.2f} seconds ---")

    if global_best_solution:
        logger.info(
            f"Final Best Solution: Feasible={global_best_solution.is_feasible}, Weighted Cost={format_float(global_best_solution.weighted_cost, 4)}")
        # Perform a final, consistent evaluation
        global_best_solution.evaluate(haversine, create_heuristic_trips_split_delivery)
    else:
        logger.warning("PSO run completed, but no valid best solution was found.")

    pso_results = {
        'best_solution': global_best_solution,
        'cost_history': cost_history,
        'total_computation_time': run_end_time - run_start_time,
        'algorithm_name': 'pso_optimizer',
        'algorithm_params': params,
    }

    return pso_results


# =================================================================================
#  Private Helper Functions for PSO
# =================================================================================

def _configure_pso_parameters(user_params: Dict[str, Any]) -> Dict[str, Any]:
    """Validates and configures PSO hyperparameters."""
    defaults = {
        'num_particles': 50,
        'max_iterations': 200,
        'inertia_weight': 0.8,
        'inertia_damping_ratio': 0.99,  # Reduces w to 99% of its value by the end
        'cognitive_weight': 1.5,
        'social_weight': 1.5,
        'topology': 'global',
    }
    params = defaults.copy()
    if isinstance(user_params, dict):
        params.update(user_params)

    # Validation checks
    if not (isinstance(params['num_particles'], int) and params['num_particles'] > 0):
        raise ValueError("`num_particles` must be a positive integer.")
    if not (isinstance(params['max_iterations'], int) and params['max_iterations'] >= 0):
        raise ValueError("`max_iterations` must be a non-negative integer.")
    if not (isinstance(params['inertia_weight'], float) and 0.0 <= params['inertia_weight'] <= 1.5):
        raise ValueError("`inertia_weight` should be a float, typically between 0.4 and 1.2.")
    if not (isinstance(params['cognitive_weight'], float) and params['cognitive_weight'] >= 0):
        raise ValueError("`cognitive_weight` must be a non-negative float.")
    if not (isinstance(params['social_weight'], float) and params['social_weight'] >= 0):
        raise ValueError("`social_weight` must be a non-negative float.")
    if params['topology'] not in ['global', 'ring', 'von_neumann']:
        raise ValueError("`topology` must be one of 'global', 'ring', or 'von_neumann'.")

    return params


def _initialize_swarm(num_particles: int,
                      problem_data: Dict,
                      vehicle_params: Dict,
                      drone_params: Dict,
                      objective_params: Dict) -> List[PermutationParticle]:
    """
    Creates and evaluates the initial swarm of particles.
    """
    swarm = []
    logger.debug(f"Creating swarm of {num_particles} particles with random solutions...")

    for _ in range(num_particles):
        initial_solution = create_initial_solution(
            strategy='random',
            problem_data=problem_data,
            vehicle_params=vehicle_params,
            drone_params=drone_params,
            objective_params=objective_params
        )
        if initial_solution:
            # The initial solution is already evaluated inside the factory function
            particle = PermutationParticle(initial_solution)
            swarm.append(particle)
        else:
            warnings.warn("Failed to create a valid random solution for a particle. Trying again.")

    if len(swarm) != num_particles:
        raise RuntimeError(f"Could not create the required number of particles. "
                           f"Expected {num_particles}, created {len(swarm)}.")

    return swarm


def format_float(value: Any, precision: int = 4) -> str:
    """Safely formats a numerical value for display."""
    if isinstance(value, (int, float)):
        if math.isnan(value): return "NaN"
        if math.isinf(value): return "Infinity" if value > 0 else "-Infinity"
        return f"{value:.{precision}f}"
    return "N/A" if value is None else str(value)


# =================================================================================
#  Standalone Execution Block
# =================================================================================

if __name__ == '__main__':
    """
    Provides a standalone execution context for testing the PSO module.
    """
    print("=" * 80)
    logger.info("Running algorithm/pso_optimizer.py in Standalone Test Mode")
    print("=" * 80)

    # --- Setup Dummy Data for a Test Run ---
    try:
        logger.info("--- [Test] Creating Dummy Problem Data ---")
        dummy_problem_data = {
            'locations': {
                'logistics_centers': [(40.7128, -74.0060)],
                'sales_outlets': [(40.7580, -73.9855), (40.7484, -73.9857), (40.7831, -73.9712), (40.7295, -73.9965)],
                'customers': [
                    (40.76, -73.98), (40.75, -73.99), (40.78, -73.96), (40.74, -73.98),
                    (40.77, -73.97), (40.73, -74.00), (40.79, -73.95), (40.72, -73.99)
                ]
            },
            'demands': [10.0, 20.0, 15.0, 25.0, 12.0, 18.0, 22.0, 30.0]
        }
        dummy_vehicle_params = {'payload': 100.0, 'cost_per_km': 1.8, 'speed_kmph': 40.0}
        dummy_drone_params = {'payload': 5.0, 'max_flight_distance_km': 10.0, 'cost_per_km': 0.7, 'speed_kmph': 60.0}
        dummy_objective_params = {'cost_weight': 0.7, 'time_weight': 0.3, 'unmet_demand_penalty': 10000.0}

        # Test configurations for different topologies
        pso_configs = {
            "Global Topology": {
                'num_particles': 20, 'max_iterations': 20, 'inertia_weight': 0.8, 'inertia_damping_ratio': 0.99,
                'cognitive_weight': 2.0, 'social_weight': 2.0, 'topology': 'global'
            },
            "Ring Topology": {
                'num_particles': 20, 'max_iterations': 20, 'inertia_weight': 0.8, 'inertia_damping_ratio': 0.99,
                'cognitive_weight': 2.0, 'social_weight': 2.0, 'topology': 'ring'
            }
        }
        logger.info("--- [Test] Dummy data and PSO configurations created. ---")

    except Exception as e:
        logger.error(f"--- [Test] Error creating dummy data: {e} ---", exc_info=True)
        sys.exit(1)

    # --- Execute a Test Run for Each Configuration ---
    for name, config in pso_configs.items():
        logger.info(f"\n--- [Test] Running PSO with Configuration: {name} ---")
        try:
            final_results = run_pso_optimizer(
                problem_data=dummy_problem_data,
                vehicle_params=dummy_vehicle_params,
                drone_params=dummy_drone_params,
                objective_params=dummy_objective_params,
                algo_specific_params=config
            )

            print("\n" + "-" * 40 + f" RESULTS FOR: {name} " + "-" * 40)
            if final_results.get('run_error'):
                print(f"  Run failed with error: {final_results['run_error']}")
            else:
                best_sol = final_results.get('best_solution')
                if best_sol:
                    print(f"  Total Runtime: {final_results.get('total_computation_time', 0.0):.4f}s")
                    print(f"  Final Best Solution Status: {'Feasible' if best_sol.is_feasible else 'Infeasible'}")
                    print(f"  Final Best Weighted Cost: {format_float(best_sol.weighted_cost, 4)}")
                    print(f"  Final Best Raw Cost: {format_float(best_sol.evaluated_cost, 2)}")
                    print(f"  Final Best Time (Makespan): {format_float(best_sol.evaluated_time, 2)}")
                    print(f"  Final Best Unmet Demand: {format_float(best_sol.evaluated_unmet_demand, 2)}")
                    print("  Final Best Stage 1 Routes:")
                    for depot, route in best_sol.stage1_routes.items():
                        print(f"    - Depot {depot}: {route}")
                else:
                    print("  Run completed but no best solution was found.")
            print("-" * (82 + len(name)))

        except Exception as e:
            logger.error(f"--- [Test] A critical error occurred during the PSO test run for '{name}'. ---",
                         exc_info=True)

    logger.info("--- Standalone Test for pso_optimizer.py Finished ---")