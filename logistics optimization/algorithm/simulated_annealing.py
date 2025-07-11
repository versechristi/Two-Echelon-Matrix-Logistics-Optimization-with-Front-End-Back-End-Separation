# algorithm/simulated_annealing.py
# -*- coding: utf-8 -*-
"""
An advanced, configurable implementation of the Simulated Annealing (SA)
metaheuristic, tailored for the Multi-Depot, Two-Echelon Vehicle Routing Problem
with Drones and Split Deliveries (MD-2E-VRPSD).

This module provides a robust SA solver that explores the solution space using a
neighborhood search approach combined with a probabilistic acceptance criterion,
mimicking the annealing process in metallurgy. This allows the algorithm to escape
local optima and explore a wider range of solutions.

Key Features:
- **Independent Initialization**: By default, the algorithm begins its search from a
  completely random initial solution, ensuring an unbiased exploration of the
  solution space and providing a true measure of the algorithm's performance.
- **Multiple Annealing Schedules**: To provide fine-grained control over the
  cooling process, this implementation supports several classic schedules:
  - **Exponential (Geometric) Cooling**: The standard `T_new = T_current * alpha`.
  - **Linear Cooling**: Temperature decreases by a fixed amount per iteration.
  - **Logarithmic Cooling**: A very slow schedule, theoretically proven to
    converge to the global optimum given sufficient time.
- **Adaptive Re-annealing (Re-heating)**: Includes a mechanism to detect stagnation
  (when the acceptance rate drops below a threshold) and automatically "re-heat"
  the system by raising the temperature. This allows the search to escape from
  deep local optima.
- **Configurable Neighborhood Operators**: Allows the user to select the specific
  local search operator used to generate neighbor solutions, including 'swap',
  'inversion', 'scramble', and the more powerful '2-opt'.
- **Feasibility-Driven Acceptance**: Leverages the `SolutionCandidate` class's
  `__lt__` method for comparing solutions, which inherently prioritizes feasible
  solutions (all demand met) over infeasible ones.
- **Comprehensive Performance Tracking**: Logs and returns detailed historical data
  from the run, including best cost, current cost, temperature, and acceptance
  rate at each iteration, enabling deep analysis of the annealing process.
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
        format="%(asctime)s [%(levelname)-8s] (SA) %(message)s"
    )

# =================================================================================
#  Safe Core Project Imports
# =================================================================================
try:
    from core.problem_utils import (
        SolutionCandidate,
        create_initial_solution,
        generate_neighbor_solution,
        swap_mutation,
        inversion_mutation,
        scramble_mutation,
        two_opt_mutation
    )
    from core.distance_calculator import haversine
    from core.cost_function import calculate_total_cost_and_evaluate
    from core.problem_utils import create_heuristic_trips_split_delivery

except ImportError as e:
    logger.critical(f"A core module failed to import, which is essential for SA. Error: {e}", exc_info=True)


    # Define dummy fallbacks to allow the script to load but fail gracefully
    class SolutionCandidate:
        def __init__(self, *args, **kwargs): self.weighted_cost = float('inf'); self.is_feasible = False

        def evaluate(self, *args, **kwargs): pass

        def __lt__(self, other): return False


    def create_initial_solution(*args, **kwargs):
        return None


    def generate_neighbor_solution(*args, **kwargs):
        return None


    def swap_mutation(r):
        return r


    def inversion_mutation(r):
        return r


    def scramble_mutation(r):
        return r


    def two_opt_mutation(r):
        return r


    def haversine(c1, c2):
        return 0


    def calculate_total_cost_and_evaluate(*args, **kwargs):
        return float('inf'), float('inf'), float('inf'), {}, True, True, {}


    def create_heuristic_trips_split_delivery(*args, **kwargs):
        return []


    warnings.warn("Simulated Annealing will use dummy functions due to a critical import failure.")

# =================================================================================
#  Module-level Constants
# =================================================================================
FLOAT_TOLERANCE_SA = 1e-6


# =================================================================================
#  Annealing Schedule Manager
# =================================================================================

class AnnealingSchedule:
    """
    A class to manage different temperature cooling (annealing) schedules.

    This provides a structured way to apply various cooling strategies, making
    the SA algorithm more flexible and easier to experiment with.
    """

    def __init__(self, schedule_name: str, initial_temp: float, max_iterations: int, **kwargs):
        """
        Initializes the schedule manager.

        Args:
            schedule_name: The name of the schedule ('exponential', 'linear', 'logarithmic').
            initial_temp: The starting temperature.
            max_iterations: The total number of iterations for the run.
            **kwargs: Additional parameters required by specific schedules (e.g., 'alpha'
                      for exponential, 'min_temp' as a floor).
        """
        self.schedule_name = schedule_name.lower()
        self.initial_temp = initial_temp
        self.current_temp = initial_temp
        self.max_iterations = max_iterations
        self.min_temp = kwargs.get('min_temp', 0.001)
        self.alpha = kwargs.get('alpha', 0.99)  # For exponential cooling

        if self.schedule_name not in ['exponential', 'linear', 'logarithmic']:
            raise ValueError(f"Unknown annealing schedule: '{self.schedule_name}'")
        if self.schedule_name == 'linear':
            # For linear, calculate the amount to decrease at each step
            self.decrement = (self.initial_temp - self.min_temp) / self.max_iterations if self.max_iterations > 0 else 0

    def update_temperature(self, iteration: int) -> float:
        """
        Updates and returns the temperature for the given iteration.

        Args:
            iteration: The current iteration number (starting from 0).

        Returns:
            The new temperature for the next step.
        """
        if self.schedule_name == 'exponential':
            self.current_temp *= self.alpha
        elif self.schedule_name == 'linear':
            self.current_temp -= self.decrement
        elif self.schedule_name == 'logarithmic':
            # Note: Logarithmic cooling is very slow. `c` is a tuning parameter.
            c = 1.0  # This parameter can be tuned.
            self.current_temp = self.initial_temp / (1 + c * math.log(1 + iteration))

        # Ensure temperature does not fall below the minimum threshold.
        self.current_temp = max(self.current_temp, self.min_temp)
        return self.current_temp

    def reheat(self, factor: float = 0.5):
        """
        Re-heats the system by raising the current temperature.

        Args:
            factor: A factor of the initial temperature to set the new temperature to.
                    For example, factor=0.5 sets temp to 50% of initial temp.
        """
        new_temp = self.initial_temp * factor
        logger.warning(f"Re-annealing detected. Re-heating temperature from {self.current_temp:.4f} to {new_temp:.4f}.")
        self.current_temp = new_temp


# =================================================================================
#  Neighborhood Operator Manager
# =================================================================================

def get_neighborhood_operator(operator_name: str) -> Callable[[List], List]:
    """
    Factory function to retrieve a specific neighborhood operator function by name.

    This allows the SA algorithm to be configured to use a specific type of move
    (e.g., always use 2-opt) for generating neighbor solutions.

    Args:
        operator_name: The name of the operator ('swap', 'inversion', 'scramble', '2-opt').

    Returns:
        The callable function corresponding to the chosen operator.
    """
    operator_map = {
        'swap': swap_mutation,
        'inversion': inversion_mutation,
        'scramble': scramble_mutation,
        '2-opt': two_opt_mutation,
    }
    func = operator_map.get(operator_name.lower())
    if func is None:
        logger.warning(f"Unknown neighborhood operator '{operator_name}'. Defaulting to random selection.")
        # Return a lambda that calls the generic generator for random selection
        return lambda sol: generate_neighbor_solution(sol)

    # Return a lambda that calls the generic generator with the *specific* operator
    return lambda sol: generate_neighbor_solution(sol, operator=func)


# =================================================================================
#  SA Main Orchestration Function
# =================================================================================

def run_simulated_annealing(
        problem_data: Dict[str, Any],
        vehicle_params: Dict[str, Any],
        drone_params: Dict[str, Any],
        objective_params: Dict[str, float],
        algo_specific_params: Dict[str, Any],
        initial_solution_candidate: Optional[SolutionCandidate] = None  # Now truly optional
) -> Dict[str, Any]:
    """
    Executes the Simulated Annealing algorithm to solve the MD-2E-VRPSD.

    This function orchestrates the entire SA process. It initializes a solution
    (randomly, by default), then iteratively generates and evaluates neighbor
    solutions, accepting them based on a probabilistic criterion governed by a
    cooling temperature.

    Args:
        problem_data, vehicle_params, drone_params, objective_params: Standard
            problem definition dictionaries.
        algo_specific_params: A dictionary of SA-specific hyperparameters. Expected keys:
            - `initial_temperature` (float)
            - `cooling_rate` (float): Alpha for exponential cooling.
            - `max_iterations` (int)
            - `min_temperature` (float)
            - `annealing_schedule` (str): 'exponential', 'linear', or 'logarithmic'.
            - `neighborhood_operator` (str): 'swap', 'inversion', 'scramble', '2-opt', or 'random'.
            - `enable_reannealing` (bool): Whether to enable adaptive re-heating.
            - `stagnation_threshold` (float): Acceptance rate below which stagnation is detected.
            - `stagnation_window` (int): Number of iterations to check for stagnation.
        initial_solution_candidate: An optional pre-existing solution. If provided,
            the search will start from this point. Otherwise, a random solution is generated.

    Returns:
        A dictionary containing the results of the SA run, including the best
        solution found, its evaluation metrics, and detailed performance history.
    """
    run_start_time = time.time()
    logger.info("--- Simulated Annealing (MD-2E-VRPSD) Started ---")

    # --- 1. Validate and Configure SA Parameters ---
    logger.info("Configuring SA parameters...")
    try:
        params = _configure_sa_parameters(algo_specific_params)
        logger.info(
            f"SA Configuration: InitialTemp={params['initial_temperature']}, MaxIters={params['max_iterations']}, "
            f"Schedule='{params['annealing_schedule']}', Operator='{params['neighborhood_operator']}', "
            f"Re-annealing={params['enable_reannealing']}")
    except (ValueError, KeyError) as e:
        error_msg = f"SA parameter validation failed: {e}"
        logger.error(error_msg, exc_info=True)
        return {'run_error': error_msg}

    # --- 2. Initialize Solution and Annealing Schedule ---
    logger.info("Initializing starting solution and annealing schedule...")
    try:
        if initial_solution_candidate:
            logger.info("Starting SA from a provided initial solution.")
            current_solution = copy.deepcopy(initial_solution_candidate)
            # Ensure it's evaluated with the current run's parameters
            current_solution.evaluate(haversine, create_heuristic_trips_split_delivery)
        else:
            logger.info("No initial solution provided. Generating a random starting solution.")
            current_solution = create_initial_solution(
                strategy='random',
                problem_data=problem_data,
                vehicle_params=vehicle_params,
                drone_params=drone_params,
                objective_params=objective_params
            )

        if not current_solution or current_solution.initialization_error:
            raise RuntimeError("Failed to obtain a valid starting solution for SA.")

        best_solution_overall = copy.deepcopy(current_solution)

        schedule = AnnealingSchedule(
            schedule_name=params['annealing_schedule'],
            initial_temp=params['initial_temperature'],
            max_iterations=params['max_iterations'],
            min_temp=params['min_temperature'],
            alpha=params['cooling_rate']
        )

        neighbor_generator = get_neighborhood_operator(params['neighborhood_operator'])

    except Exception as e:
        error_msg = f"Failed during SA initialization: {e}"
        logger.error(error_msg, exc_info=True)
        return {'run_error': error_msg}

    # --- 3. Main Annealing Loop ---
    logger.info(f"Starting SA loop. Initial Cost: {format_float(current_solution.weighted_cost, 4)}")

    # History tracking lists
    best_cost_history = []
    current_cost_history = []
    temperature_history = []
    acceptance_rate_history = []

    # For re-annealing logic
    acceptance_window = []

    for i in range(params['max_iterations']):
        # --- a. Generate and Evaluate Neighbor ---
        neighbor_solution = neighbor_generator(current_solution)

        if not neighbor_solution:
            warnings.warn(f"Iteration {i + 1}: Failed to generate a neighbor solution. Skipping iteration.")
            # Still need to record history for this step
            best_cost_history.append(best_solution_overall.weighted_cost)
            current_cost_history.append(current_solution.weighted_cost)
            temperature_history.append(schedule.current_temp)
            acceptance_rate_history.append(0.0)
            continue

        neighbor_solution.evaluate(haversine, create_heuristic_trips_split_delivery)

        # --- b. Acceptance Criterion ---
        accepted = False
        if neighbor_solution < current_solution:
            # If the neighbor is better (feasibility-first), always accept it.
            accepted = True
        else:
            # If the neighbor is worse, accept with a probability based on the Boltzmann distribution.
            cost_diff = neighbor_solution.weighted_cost - current_solution.weighted_cost
            if math.isfinite(cost_diff) and cost_diff > 0 and schedule.current_temp > FLOAT_TOLERANCE_SA:
                try:
                    acceptance_prob = math.exp(-cost_diff / schedule.current_temp)
                    if random.random() < acceptance_prob:
                        accepted = True
                except OverflowError:
                    # exp(-large_number) is effectively zero.
                    pass

        # --- c. Update State ---
        if accepted:
            current_solution = neighbor_solution
            # Check if this newly accepted solution is the best one found so far.
            if current_solution < best_solution_overall:
                best_solution_overall = copy.deepcopy(current_solution)

        acceptance_window.append(1 if accepted else 0)

        # --- d. Update History ---
        best_cost_history.append(best_solution_overall.weighted_cost)
        current_cost_history.append(current_solution.weighted_cost)
        temperature_history.append(schedule.current_temp)

        # --- e. Update Temperature (Cooling) ---
        schedule.update_temperature(i)

        # --- f. Adaptive Re-annealing Logic ---
        if params['enable_reannealing'] and len(acceptance_window) >= params['stagnation_window']:
            # Calculate acceptance rate over the last 'window' iterations
            current_acceptance_rate = np.mean(acceptance_window)
            acceptance_rate_history.append(current_acceptance_rate)

            if current_acceptance_rate < params['stagnation_threshold']:
                # Stagnation detected, re-heat the system.
                schedule.reheat(factor=0.5)  # Example: reheat to 50% of initial temp
                # Reset the acceptance window to start fresh after re-heating
                acceptance_window.clear()
            else:
                # Slide the window forward
                acceptance_window.pop(0)

        # --- g. Log Progress ---
        if (i + 1) % (params['max_iterations'] // 20 or 1) == 0:
            logger.info(f"Iter {i + 1:>5}/{params['max_iterations']} | "
                        f"Temp: {format_float(schedule.current_temp, 4):>9s} | "
                        f"Current Cost: {format_float(current_solution.weighted_cost, 2):>12s} | "
                        f"Best Cost: {format_float(best_solution_overall.weighted_cost, 2):>12s} | "
                        f"Accepts/win: {sum(acceptance_window)}/{len(acceptance_window) if acceptance_window else 1}")

    # --- 4. Finalization and Result Packaging ---
    run_end_time = time.time()
    logger.info(f"--- Simulated Annealing Finished in {run_end_time - run_start_time:.2f} seconds ---")

    if best_solution_overall:
        logger.info(
            f"Final Best Solution: Feasible={best_solution_overall.is_feasible}, Weighted Cost={format_float(best_solution_overall.weighted_cost, 4)}")
        # Perform a final, consistent evaluation
        best_solution_overall.evaluate(haversine, create_heuristic_trips_split_delivery)
    else:
        logger.warning("SA run completed, but no valid best solution was found.")

    sa_results = {
        'best_solution': best_solution_overall,
        'cost_history': best_cost_history,
        'current_cost_history': current_cost_history,
        'temperature_history': temperature_history,
        'acceptance_rate_history': acceptance_rate_history,
        'total_computation_time': run_end_time - run_start_time,
        'algorithm_name': 'simulated_annealing',
        'algorithm_params': params,
    }

    return sa_results


# =================================================================================
#  Private Helper Functions for SA
# =================================================================================

def _configure_sa_parameters(user_params: Dict[str, Any]) -> Dict[str, Any]:
    """Validates and configures SA hyperparameters, merging user inputs with defaults."""
    defaults = {
        'initial_temperature': 1000.0,
        'cooling_rate': 0.99,
        'max_iterations': 20000,
        'min_temperature': 0.001,
        'annealing_schedule': 'exponential',
        'neighborhood_operator': 'random',
        'enable_reannealing': False,
        'stagnation_threshold': 0.01,
        'stagnation_window': 100,
    }
    params = defaults.copy()
    if isinstance(user_params, dict):
        params.update(user_params)

    # --- Validation ---
    if not (isinstance(params['initial_temperature'], (int, float)) and params['initial_temperature'] > 0):
        raise ValueError("`initial_temperature` must be a positive number.")
    if not (isinstance(params['cooling_rate'], float) and 0 < params['cooling_rate'] < 1.0):
        raise ValueError("`cooling_rate` (alpha) must be a float between 0 and 1.")
    if not (isinstance(params['max_iterations'], int) and params['max_iterations'] > 0):
        raise ValueError("`max_iterations` must be a positive integer.")
    if params['annealing_schedule'] not in ['exponential', 'linear', 'logarithmic']:
        raise ValueError("`annealing_schedule` must be 'exponential', 'linear', or 'logarithmic'.")
    if params['neighborhood_operator'] not in ['random', 'swap', 'inversion', 'scramble', '2-opt']:
        raise ValueError("`neighborhood_operator` is not a valid selection.")
    if params['enable_reannealing'] and not (
            isinstance(params['stagnation_window'], int) and params['stagnation_window'] > 0):
        raise ValueError("`stagnation_window` must be a positive integer when reannealing is enabled.")

    return params


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
    Provides a standalone execution context for testing the SA module.
    """
    print("=" * 80)
    logger.info("Running algorithm/simulated_annealing.py in Standalone Test Mode")
    print("=" * 80)

    # --- Setup Dummy Data for a Test Run ---
    try:
        logger.info("--- [Test] Creating Dummy Problem Data ---")
        dummy_problem_data = {
            'locations': {
                'logistics_centers': [(40.7128, -74.0060)],
                'sales_outlets': [(40.7580, -73.9855), (40.7484, -73.9857), (40.7831, -73.9712)],
                'customers': [(40.76, -73.98), (40.75, -73.99), (40.78, -73.96), (40.74, -73.98)]
            },
            'demands': [10.0, 20.0, 15.0, 25.0]
        }
        dummy_vehicle_params = {'payload': 50.0, 'cost_per_km': 1.8, 'speed_kmph': 40.0}
        dummy_drone_params = {'payload': 5.0, 'max_flight_distance_km': 8.0, 'cost_per_km': 0.7, 'speed_kmph': 60.0}
        dummy_objective_params = {'cost_weight': 0.7, 'time_weight': 0.3, 'unmet_demand_penalty': 10000.0}

        sa_config = {
            'initial_temperature': 1000.0,
            'cooling_rate': 0.995,
            'max_iterations': 500,  # Keep it short for a test
            'min_temperature': 0.01,
            'annealing_schedule': 'exponential',
            'neighborhood_operator': '2-opt',
            'enable_reannealing': True,
            'stagnation_threshold': 0.02,
            'stagnation_window': 50,
        }
        logger.info("--- [Test] Dummy data and SA configuration created. ---")

    except Exception as e:
        logger.error(f"--- [Test] Error creating dummy data: {e} ---", exc_info=True)
        sys.exit(1)

    # --- Execute the Test Run ---
    logger.info(f"--- [Test] Calling run_simulated_annealing ---")
    try:
        final_results = run_simulated_annealing(
            problem_data=dummy_problem_data,
            vehicle_params=dummy_vehicle_params,
            drone_params=dummy_drone_params,
            objective_params=dummy_objective_params,
            algo_specific_params=sa_config
        )

        print("\n" + "=" * 40 + " FINAL SA SUMMARY " + "=" * 40)
        if final_results.get('run_error'):
            print(f"  Run failed with error: {final_results['run_error']}")
        else:
            best_sol = final_results.get('best_solution')
            if best_sol:
                print(f"  Total Runtime: {final_results.get('total_computation_time', 0.0):.4f}s")
                print(f"  Final Best Solution Status: {'Feasible' if best_sol.is_feasible else 'Infeasible'}")
                print(f"  Final Best Weighted Cost: {format_float(best_sol.weighted_cost, 4)}")
            else:
                print("  Run completed but no best solution was found.")

            # Optional: Plotting the results for visual inspection
            try:
                import matplotlib.pyplot as plt

                fig, ax1 = plt.subplots(figsize=(12, 7))

                ax1.set_xlabel('Iteration')
                ax1.set_ylabel('Cost (Weighted)', color='tab:blue')
                ax1.plot(final_results['best_cost_history'], color='tab:blue', label='Best Cost')
                ax1.plot(final_results['current_cost_history'], color='tab:cyan', alpha=0.5, label='Current Cost')
                ax1.tick_params(axis='y', labelcolor='tab:blue')
                ax1.grid(True, linestyle=':')

                ax2 = ax1.twinx()
                ax2.set_ylabel('Temperature (Log Scale)', color='tab:red')
                ax2.plot(final_results['temperature_history'], color='tab:red', linestyle='--', label='Temperature')
                ax2.set_yscale('log')
                ax2.tick_params(axis='y', labelcolor='tab:red')

                fig.tight_layout()
                plt.title("Simulated Annealing Performance (Standalone Test)")
                fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

                output_dir = "output/sa_standalone_test"
                os.makedirs(output_dir, exist_ok=True)
                test_plot_path = os.path.join(output_dir, "sa_test_run_plot.png")
                plt.savefig(test_plot_path)
                print(f"\n  Performance plot saved to: {test_plot_path}")
                plt.close()
            except ImportError:
                print("\n  Matplotlib not installed. Skipping plot generation.")
            except Exception as plot_e:
                print(f"\n  Error generating plot: {plot_e}")

        print("=" * 94)

    except Exception as e:
        logger.error("--- [Test] A critical, unhandled error occurred during the main SA test execution. ---",
                     exc_info=True)

    logger.info("--- Standalone Test for simulated_annealing.py Finished ---")