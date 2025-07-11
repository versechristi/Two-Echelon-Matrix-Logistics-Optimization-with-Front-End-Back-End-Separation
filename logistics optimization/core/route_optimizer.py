# core/route_optimizer.py
# -*- coding: utf-8 -*-
"""
Main Orchestrator for the Logistics Optimization Engine.

This module serves as the central nervous system for the entire optimization
process. It is responsible for receiving a complete problem definition and a set
of optimization parameters from a high-level interface (such as the GUI), and
then managing the full lifecycle of the optimization run.

The key responsibilities of this orchestrator include:
1.  **Input Validation**: Rigorously validates all incoming data and parameters
    to ensure they are structurally sound and logically consistent before
    commencing any computationally expensive processes.

2.  **Run Initialization**: Sets up the environment for a specific optimization
    run, including creating a unique, timestamped output directory for storing
    all artifacts (logs, results, maps, reports).

3.  **Algorithm Execution**: Iterates through a user-selected list of
    optimization algorithms. For each algorithm, it:
    a.  Creates an appropriate initial solution ('greedy' for baseline,
        'random' for metaheuristics) to ensure fair, independent trials.
    b.  Invokes the algorithm's main execution function.
    c.  Captures and standardizes the results.

4.  **Result Aggregation and Analysis**: After all selected algorithms have
    completed, this module aggregates their results, identifies the best
    overall solution, and determines the best feasible solution found.

5.  **Post-processing and Artifact Generation**: Coordinates the generation of
    all output artifacts for each successful algorithm run, including:
    - Interactive route maps using Folium.
    - Detailed, formatted text-based reports.
    - JSON files containing the raw result data for archival or further analysis.

This module is designed to be completely decoupled from the user interface,
operating solely on the data and parameters provided to it.
"""

# =================================================================================
#  Standard Library Imports
# =================================================================================
import json
import sys
import time
import os
import traceback
import copy
import math
import warnings
import logging
from typing import List, Dict, Tuple, Optional, Any, Callable, Union

from matplotlib import pyplot as plt

# =================================================================================
#  Logging Configuration
# =================================================================================
# Configure a logger for this module to provide detailed, contextual information
# during the optimization process. This is crucial for debugging and tracking.
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] (%(module)s:%(lineno)d) %(message)s"
    )

# =================================================================================
#  Safe Core Project Imports
# =================================================================================
# This section ensures that all necessary components from other core modules are
# available. It uses try-except blocks to provide informative error messages if
# a critical dependency is missing, which is vital for diagnosing setup issues.
try:
    logger.debug("Importing core utilities and functions...")
    from core.distance_calculator import haversine
    from core.cost_function import calculate_total_cost_and_evaluate, format_float
    from core.problem_utils import (
        SolutionCandidate,
        create_initial_solution,  # The new factory function
        create_heuristic_trips_split_delivery,
        FLOAT_TOLERANCE,
    )

    logger.debug("Core utilities and functions imported successfully.")
except ImportError as e:
    logger.critical(f"CRITICAL: Failed to import a core utility module. The application cannot run. Error: {e}",
                    exc_info=True)
    # In a real application, this might trigger a more user-friendly error display,
    # but for a core module, re-raising is appropriate to halt execution.
    raise

# --- Safe Algorithm Imports ---
try:
    logger.debug("Importing optimization algorithm modules...")
    from algorithm import (
        run_genetic_algorithm,
        run_simulated_annealing,
        run_pso_optimizer,
        run_greedy_heuristic,
    )

    # The ALGORITHM_REGISTRY provides a clean, extensible way to map the internal
    # string keys (used by the GUI/config) to the actual callable functions.
    ALGORITHM_REGISTRY: Dict[str, Callable] = {
        "greedy_heuristic": run_greedy_heuristic,
        "genetic_algorithm": run_genetic_algorithm,
        "simulated_annealing": run_simulated_annealing,
        "pso_optimizer": run_pso_optimizer,
    }
    logger.debug(f"Successfully registered algorithms: {list(ALGORITHM_REGISTRY.keys())}")
except ImportError as e:
    logger.critical(f"CRITICAL: Failed to import one or more algorithms from the 'algorithm' package. Error: {e}",
                    exc_info=True)
    ALGORITHM_REGISTRY = {}
    raise

# --- Safe Visualization and Reporting Imports ---
# These components are considered optional. If they fail to import (e.g., due
# to missing dependencies like 'folium'), the optimizer can still run, but the
# corresponding output artifacts will be skipped.
try:
    from visualization.map_generator import generate_folium_map

    _MAP_GENERATION_ENABLED = True
    logger.debug("Map generator imported successfully.")
except ImportError:
    logger.warning("Map generator ('visualization.map_generator') not found. Map generation will be disabled.")


    def generate_folium_map(*args: Any, **kwargs: Any) -> None:
        """Dummy function for when map generation is unavailable."""
        logger.warning("generate_folium_map called, but the module is not available.")
        return None


    _MAP_GENERATION_ENABLED = False

try:
    from utils.report_generator import generate_delivery_report

    _REPORTING_ENABLED = True
    logger.debug("Report generator imported successfully.")
except ImportError:
    logger.warning("Report generator ('utils.report_generator') not found. Report generation will be disabled.")


    def generate_delivery_report(*args: Any, **kwargs: Any) -> str:
        """Dummy function for when report generation is unavailable."""
        logger.warning("generate_delivery_report called, but the module is not available.")
        return "Error: Report generator module is not available."


    _REPORTING_ENABLED = False

try:
    from visualization.plot_generator import PlotGenerator
    _PLOTTING_ENABLED = True
    logger.debug("Plot generator imported successfully.")
except ImportError:
    logger.warning("Plot generator ('visualization.plot_generator') not found. Plot generation will be disabled.")

# =================================================================================
#  Private Helper Functions for Main Orchestrator
# =================================================================================

def _validate_inputs(problem_data: Dict, vehicle_params: Dict, drone_params: Dict,
                     optimization_params: Dict, selected_algorithm_keys: List,
                     objective_weights: Dict) -> List[str]:
    """
    Performs comprehensive validation of all input parameters.

    Args:
        (All arguments from the main `run_optimization` function)

    Returns:
        A list of error strings. An empty list signifies successful validation.
    """
    errors = []
    logger.debug("Performing detailed input validation...")

    # Problem Data Validation
    if not isinstance(problem_data, dict) or 'locations' not in problem_data or 'demands' not in problem_data:
        errors.append("`problem_data` is malformed. It must be a dict with 'locations' and 'demands' keys.")
    else:
        locs = problem_data.get('locations', {})
        if not isinstance(locs, dict) or not all(
                k in locs for k in ['logistics_centers', 'sales_outlets', 'customers']):
            errors.append("'locations' must be a dict with keys 'logistics_centers', 'sales_outlets', 'customers'.")
        elif not locs.get('logistics_centers') or not isinstance(locs.get('logistics_centers'), list):
            errors.append("At least one logistics center must be provided in 'locations'.")

        demands = problem_data.get('demands', [])
        num_customers = len(locs.get('customers', []))
        if not isinstance(demands, list) or len(demands) != num_customers:
            errors.append(
                f"The 'demands' list length ({len(demands)}) must match the number of customers ({num_customers}).")
        elif any(not isinstance(d, (int, float)) or d < 0 for d in demands):
            errors.append("All values in the 'demands' list must be non-negative numbers.")

    # Vehicle and Drone Parameter Validation
    required_veh_keys = {'payload', 'cost_per_km', 'speed_kmph'}
    if not isinstance(vehicle_params, dict) or not required_veh_keys.issubset(vehicle_params):
        errors.append(f"Incomplete `vehicle_params`. Missing keys: {required_veh_keys - set(vehicle_params.keys())}")
    elif not isinstance(vehicle_params.get('speed_kmph', 0), (int, float)) or vehicle_params.get('speed_kmph', 0) <= 0:
        errors.append("Vehicle speed ('speed_kmph') must be a positive number.")

    required_drone_keys = {'payload', 'max_flight_distance_km', 'cost_per_km', 'speed_kmph'}
    if not isinstance(drone_params, dict) or not required_drone_keys.issubset(drone_params):
        errors.append(f"Incomplete `drone_params`. Missing keys: {required_drone_keys - set(drone_params.keys())}")
    elif not isinstance(drone_params.get('speed_kmph', 0), (int, float)) or drone_params.get('speed_kmph', 0) <= 0:
        errors.append("Drone speed ('speed_kmph') must be a positive number.")

    # Optimization and Objective Parameter Validation
    if 'unmet_demand_penalty' not in optimization_params:
        errors.append("`optimization_params` must include 'unmet_demand_penalty'.")

    if 'cost_weight' not in objective_weights or 'time_weight' not in objective_weights:
        errors.append("`objective_weights` must include 'cost_weight' and 'time_weight'.")

    # Algorithm Selection and Parameter Validation
    if not isinstance(selected_algorithm_keys, list) or not selected_algorithm_keys:
        errors.append("`selected_algorithm_keys` must be a non-empty list.")
    else:
        for key in selected_algorithm_keys:
            if key not in ALGORITHM_REGISTRY:
                errors.append(f"Unknown or unavailable algorithm key selected: '{key}'.")
            elif f'{key}_params' not in optimization_params:
                errors.append(f"Missing parameter dictionary '{key}_params' for selected algorithm '{key}'.")

    logger.debug(f"Validation found {len(errors)} errors.")
    return errors


def _initialize_run_environment(base_output_dir: str, timestamp: str) -> Optional[str]:
    """
    Creates the necessary output directories for the optimization run.

    Args:
        base_output_dir: The root directory for all outputs.
        timestamp: The unique timestamp for this specific run.

    Returns:
        The path to the run-specific output directory, or None on failure.
    """
    run_output_dir = os.path.join(base_output_dir, timestamp)
    try:
        # 1. First, create the main directory for this specific run.
        os.makedirs(run_output_dir, exist_ok=True)

        # 2. Then, create all necessary subdirectories inside it.
        #    This also centralizes the creation of the 'charts' directory.
        os.makedirs(os.path.join(run_output_dir, "maps"), exist_ok=True)
        os.makedirs(os.path.join(run_output_dir, "reports"), exist_ok=True)
        os.makedirs(os.path.join(run_output_dir, "results"), exist_ok=True)
        os.makedirs(os.path.join(run_output_dir, "charts"), exist_ok=True) # Add charts dir here

        logger.info(f"Successfully created output directory structure at: {run_output_dir}")
        return run_output_dir
    except OSError as e:
        logger.error(f"Failed to create output directory '{run_output_dir}': {e}")
        return None


def _save_parameters_to_file(run_output_dir: str, params: Dict[str, Any]):
    """Saves the complete set of run parameters to a JSON file for reproducibility."""
    filepath = os.path.join(run_output_dir, "run_parameters.json")
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # We remove the full problem_data to keep the summary file small
            params_to_save = copy.deepcopy(params)
            if 'problem_data' in params_to_save:
                del params_to_save['problem_data']
            json.dump(params_to_save, f, indent=4)
        logger.info(f"Run parameters saved to {filepath}")
    except Exception as e:
        logger.warning(f"Could not save run parameters to file: {e}")


# =================================================================================
#  Main Orchestration Function
# =================================================================================

def run_optimization(
        problem_data: Dict[str, Any],
        vehicle_params: Dict[str, Any],
        drone_params: Dict[str, Any],
        optimization_params: Dict[str, Any],
        selected_algorithm_keys: List[str],
        objective_weights: Dict[str, float],
        task_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main entry point for the optimization process, orchestrating the entire workflow.

    This function serves as the high-level API for running one or more optimization
    algorithms on a given MD-2E-VRPSD problem instance. It ensures that each step,
    from validation to result generation, is handled consistently.

    Args:
        problem_data: A dictionary containing the complete problem definition.
            - `locations`: A dict with keys 'logistics_centers', 'sales_outlets',
              'customers', each holding a list of (latitude, longitude) tuples.
            - `demands`: A list of float values corresponding to each customer's demand.

        vehicle_params: A dictionary of parameters for the vehicle fleet.
            - `payload` (float): Maximum carrying capacity.
            - `cost_per_km` (float): Cost per kilometer of travel.
            - `speed_kmph` (float): Average travel speed.

        drone_params: A dictionary of parameters for the drone fleet.
            - `payload` (float): Maximum carrying capacity.
            - `max_flight_distance_km` (float): Maximum round-trip flight range.
            - `cost_per_km` (float): Cost per kilometer of flight.
            - `speed_kmph` (float): Average flight speed.

        optimization_params: A dictionary containing general optimization settings
            and nested, algorithm-specific parameter dictionaries.
            - `unmet_demand_penalty` (float): Penalty applied per unit of unmet demand.
            - `output_dir` (str): The base directory for saving run outputs.
            - `{algo_key}_params` (Dict): A nested dictionary for each selected
              algorithm (e.g., 'genetic_algorithm_params': {...}).

        selected_algorithm_keys: A list of strings identifying which algorithms to run.
            Each key must correspond to an entry in the `ALGORITHM_REGISTRY`.

        objective_weights: A dictionary defining the weights for the objective function.
            - `cost_weight` (float): The weighting factor for the raw transportation cost.
            - `time_weight` (float): The weighting factor for the solution makespan.

    Returns:
        A comprehensive dictionary containing the results of the entire optimization run.
        The structure is designed to be self-contained and easily parsable by other
        modules (like a GUI or analysis script).
        - `overall_status` (str): High-level status ('Success', 'Validation Failed', etc.).
        - `run_timestamp` (str): The unique timestamp for the run.
        - `output_directory` (str): Path to the dedicated output folder for this run.
        - `results_by_algorithm` (Dict): A nested dictionary where each key is an
          algorithm key, and the value is a detailed summary of its run.
        - `best_algorithm_key` (Optional[str]): Key of the algorithm that found the
          best solution overall (by weighted cost).
        - `fully_served_best_key` (Optional[str]): Key of the algorithm that found the
          best *feasible* solution.
        - `total_computation_time` (float): Total wall-clock time for the entire function call.
        - `parameters_used` (Dict): A deep copy of the input parameters for reproducibility.
    """
    run_start_time = time.time()
    run_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    logger.info(f"--- ROUTE OPTIMIZATION RUN STARTED: {run_timestamp} ---")

    # --- 1. Validate Inputs ---
    validation_errors = _validate_inputs(
        problem_data, vehicle_params, drone_params, optimization_params,
        selected_algorithm_keys, objective_weights
    )
    if validation_errors:
        error_str = "Input validation failed:\n" + "\n".join(f"  - {e}" for e in validation_errors)
        logger.error(error_str)
        return {
            "overall_status": "Validation Failed",
            "error_message": error_str,
            "total_computation_time": time.time() - run_start_time,
            "results_by_algorithm": {},
        }

    # --- 2. Initialize Run Environment ---
    base_output_dir = optimization_params.get("output_dir", "output")
    run_output_dir = _initialize_run_environment(base_output_dir, run_timestamp)
    if not run_output_dir:
        return {
            "overall_status": "Error",
            "error_message": "Failed to create output directories.",
            "total_computation_time": time.time() - run_start_time,
            "results_by_algorithm": {},
        }

    # --- Prepare the main results dictionary ---
    # This structure will be populated as the optimization progresses.
    objective_params = {
        'cost_weight': objective_weights.get('cost_weight', 1.0),
        'time_weight': objective_weights.get('time_weight', 0.0),
        'unmet_demand_penalty': optimization_params.get('unmet_demand_penalty', 1e9),
    }

    overall_results = {
        "overall_status": "Running",
        "error_message": None,
        "run_timestamp": run_timestamp,
        "output_directory": run_output_dir,
        "results_by_algorithm": {},
        "best_algorithm_key": None,
        "best_weighted_cost": float("inf"),
        "fully_served_best_key": None,
        "fully_served_best_cost": float("inf"),
        "total_computation_time": 0.0,
        "parameters_used": {
            "problem_data_summary": {
                "num_logistics_centers": len(problem_data.get('locations', {}).get('logistics_centers', [])),
                "num_sales_outlets": len(problem_data.get('locations', {}).get('sales_outlets', [])),
                "num_customers": len(problem_data.get('locations', {}).get('customers', [])),
                "total_initial_demand": sum(problem_data.get('demands', []))
            },
            "vehicle_params": vehicle_params,
            "drone_params": drone_params,
            "optimization_params": optimization_params,
            "selected_algorithm_keys": selected_algorithm_keys,
            "objective_weights": objective_weights,
        }
    }

    # Save parameters for this run to a file for reproducibility
    _save_parameters_to_file(run_output_dir, overall_results["parameters_used"])

    # --- 3. Execute Selected Algorithms Sequentially ---
    logger.info(f"Beginning execution of selected algorithms: {selected_algorithm_keys}")

    for algo_key in selected_algorithm_keys:
        algo_run_func = ALGORITHM_REGISTRY[algo_key]
        algo_summary = {
            "algorithm_name": algo_key.replace('_', ' ').title(),
            "status": "Pending",
            "run_error": None,
            "computation_time": 0.0,
            "result_data": None,
            "map_path": None,
            "report_path": None,
            "map_generation_error": None,
            "report_generation_error": None,
        }
        overall_results['results_by_algorithm'][algo_key] = algo_summary

        logger.info(f"--- Executing: {algo_summary['algorithm_name']} ---")
        algo_start_time = time.time()

        try:
            # --- Key Change: Generate a specific initial solution for this algorithm ---
            # The Greedy algorithm IS the initial solution generator in its case.
            # Metaheuristics (GA, SA, PSO) should start from a random state.
            init_strategy = 'greedy' if algo_key == 'greedy_heuristic' else 'random'

            logger.info(f"Creating initial solution for '{algo_key}' using '{init_strategy}' strategy...")
            initial_solution = create_initial_solution(
                strategy=init_strategy,
                problem_data=problem_data,
                vehicle_params=vehicle_params,
                drone_params=drone_params,
                objective_params=objective_params
            )

            if not initial_solution:
                raise RuntimeError(f"Failed to create a '{init_strategy}' initial solution.")

            # For iterative algorithms, the generated solution is the starting point.
            # For the greedy algorithm, its generated solution *is* its final result.
            algo_specific_params = optimization_params.get(f'{algo_key}_params', {})

            if algo_key == 'greedy_heuristic':
                # The result from the greedy builder is the final result.
                # We construct a standard result dictionary from it.
                # The evaluation is already done inside the create_greedy_initial_solution.
                final_algo_result = {
                    'best_solution': initial_solution,
                    'cost_history': [initial_solution.weighted_cost],
                    'algorithm_params': algo_specific_params,
                }
            else:
                # Call the metaheuristic algorithm, passing the random initial solution
                final_algo_result = algo_run_func(
                    problem_data=problem_data,
                    vehicle_params=vehicle_params,
                    drone_params=drone_params,
                    objective_params=objective_params,
                    initial_solution_candidate=initial_solution,  # Pass the generated random solution
                    algo_specific_params=algo_specific_params
                )

            if not final_algo_result:
                raise RuntimeError("Algorithm run function returned None.")
            if final_algo_result.get('run_error'):
                raise RuntimeError(f"Algorithm reported a run error: {final_algo_result['run_error']}")

            # --- Standardize and Store Result ---
            best_solution_from_algo = final_algo_result.get('best_solution')
            if not isinstance(best_solution_from_algo, SolutionCandidate):
                raise TypeError("Algorithm did not return a valid SolutionCandidate object in 'best_solution'.")

            # Final, consistent re-evaluation outside the algorithm to ensure uniform metrics
            logger.info("Performing final consistent re-evaluation of the algorithm's best solution...")
            best_solution_from_algo.evaluate(haversine, create_heuristic_trips_split_delivery)

            # Populate the standardized result data dictionary
            algo_summary['result_data'] = {
                'weighted_cost': best_solution_from_algo.weighted_cost,
                'evaluated_cost': best_solution_from_algo.evaluated_cost,
                'evaluated_time': best_solution_from_algo.evaluated_time,
                'evaluated_unmet_demand': best_solution_from_algo.evaluated_unmet_demand,
                'is_feasible': best_solution_from_algo.is_feasible,
                'cost_history': final_algo_result.get('cost_history', []),
                'served_customer_details': best_solution_from_algo.served_customer_details,
                'stage1_routes': best_solution_from_algo.stage1_routes,
                'stage2_trips': best_solution_from_algo.stage2_trips,
            }
            algo_summary['result_data']['total_time'] = best_solution_from_algo.evaluated_time
            algo_summary['status'] = 'Success'
            logger.info(f"Successfully processed result for {algo_summary['algorithm_name']}.")

        except Exception as e:
            error_msg = f"An error occurred during {algo_key} execution: {e}"
            logger.error(error_msg, exc_info=True)
            algo_summary['status'] = 'Failed'
            algo_summary['run_error'] = error_msg

        algo_summary['computation_time'] = time.time() - algo_start_time
        logger.info(f"--- Finished {algo_summary['algorithm_name']} in {algo_summary['computation_time']:.2f}s ---")

    # --- 4. Aggregate Final Results ---
    logger.info("Aggregating all algorithm results...")
    valid_results = [
        (key, summary['result_data'])
        for key, summary in overall_results['results_by_algorithm'].items()
        if summary['status'] == 'Success' and summary.get('result_data')
    ]

    if not valid_results:
        overall_results['overall_status'] = 'No Valid Results'
    else:
        overall_results['overall_status'] = 'Success'
        # Find best overall solution
        best_key, best_data = min(valid_results, key=lambda item: item[1].get('weighted_cost', float('inf')))
        overall_results['best_algorithm_key'] = best_key
        overall_results['best_weighted_cost'] = best_data.get('weighted_cost', float('inf'))

        # Find best feasible solution
        feasible_results = [(k, d) for k, d in valid_results if d.get('is_feasible')]
        if feasible_results:
            best_feasible_key, best_feasible_data = min(feasible_results,
                                                        key=lambda item: item[1].get('weighted_cost', float('inf')))
            overall_results['fully_served_best_key'] = best_feasible_key
            overall_results['fully_served_best_cost'] = best_feasible_data.get('weighted_cost', float('inf'))

    # --- 5. Post-Processing: Generate Artifacts ---
    logger.info("Starting post-processing to generate maps and reports...")
    if _PLOTTING_ENABLED and any(s['status'] == 'Success' for s in overall_results['results_by_algorithm'].values()):
        try:
            logger.info("Generating performance plots...")
            plot_gen = PlotGenerator()
            charts_dir = os.path.join(run_output_dir, "charts")
            os.makedirs(charts_dir, exist_ok=True)

            # 1. Generate iterative curve graph
            fig_iter, ax_iter = plt.subplots(figsize=(10, 6))
            plot_gen.plot_iteration_curves(overall_results['results_by_algorithm'], ax_iter)
            fig_iter.savefig(os.path.join(charts_dir, "iteration_curves.png"), dpi=100)
            plt.close(fig_iter)
            logger.info("Iteration curve plot saved.")

            # 2. Generate performance comparison chart
            fig_comp, (ax_cost, ax_time) = plt.subplots(2, 1, figsize=(10, 12))
            plot_gen.plot_comparison_bars(overall_results['results_by_algorithm'], ax_cost, ax_time)
            fig_comp.tight_layout(pad=3.0)
            fig_comp.savefig(os.path.join(charts_dir, "comparison_chart.png"), dpi=100)
            plt.close(fig_comp)
            logger.info("Comparison bar chart saved.")

        except Exception as e:
            logger.error(f"Failed to generate plots: {e}", exc_info=True)

    for algo_key, summary in overall_results['results_by_algorithm'].items():
        if summary['status'] != 'Success':
            continue

        result_data = summary['result_data']
        # Generate Map
        if _MAP_GENERATION_ENABLED:
            try:
                map_path = os.path.join(run_output_dir, "maps", f"{algo_key}_routes.html")
                generate_folium_map(
                    problem_data=problem_data,
                    solution_structure=result_data,  # Pass the entire result data dict
                    vehicle_params=vehicle_params,
                    drone_params=drone_params,
                    output_path=map_path
                )
                summary['map_path'] = map_path
            except Exception as e:
                summary['map_generation_error'] = str(e)
                logger.error(f"Failed to generate map for {algo_key}: {e}", exc_info=True)

        # Generate Report
        if _REPORTING_ENABLED:
            try:
                report_path = os.path.join(run_output_dir, "reports", f"{algo_key}_report.txt")
                report_content = generate_delivery_report(
                    algorithm_name=summary['algorithm_name'],
                    result_data=result_data,
                    points_data=problem_data['locations'],
                    initial_demands_list=problem_data['demands']
                )
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                summary['report_path'] = report_path
            except Exception as e:
                summary['report_generation_error'] = str(e)
                logger.error(f"Failed to generate report for {algo_key}: {e}", exc_info=True)

        # Save raw result data to JSON
        try:
            result_path = os.path.join(run_output_dir, "results", f"{algo_key}_results.json")
            with open(result_path, 'w', encoding='utf-8') as f:
                # Deepcopy to avoid modifying the original dict during serialization
                json.dump(copy.deepcopy(result_data), f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save JSON results for {algo_key}: {e}", exc_info=True)

    # --- 6. Finalize and Return ---
    overall_results['total_computation_time'] = time.time() - run_start_time
    logger.info(
        f"--- ROUTE OPTIMIZATION RUN FINISHED. Total Time: {overall_results['total_computation_time']:.2f}s ---")

    # Save the final aggregated results object
    final_summary_path = os.path.join(run_output_dir, "run_summary.json")
    try:
        with open(final_summary_path, 'w', encoding='utf-8') as f:
            json.dump(overall_results, f, indent=4)
        logger.info(f"Final run summary saved to {final_summary_path}")
    except Exception as e:
        logger.error(f"Could not save final run summary: {e}")

    return overall_results


# =================================================================================
#  Standalone Execution Block
# =================================================================================

if __name__ == '__main__':
    """
    Provides a standalone execution context for testing the `run_optimization`
    function. This is critical for debugging the orchestration logic without
    needing to run the full GUI application.
    """
    print("=" * 80)
    logger.info("Running core/route_optimizer.py in Standalone Test Mode")
    print("=" * 80)

    # --- Setup Dummy Data for a Test Run ---
    try:
        logger.info("--- [Test] Creating Dummy Problem Data ---")
        dummy_problem_data = {
            'locations': {
                'logistics_centers': [(40.7128, -74.0060)],  # New York
                'sales_outlets': [(40.7580, -73.9855), (40.7484, -73.9857), (40.7831, -73.9712)],
                # Times Sq, Empire State, Central Park
                'customers': [
                    (40.76, -73.98), (40.75, -73.99), (40.78, -73.96),
                    (40.74, -73.98), (40.77, -73.97)
                ]
            },
            'demands': [10.0, 20.0, 15.0, 25.0, 12.0]
        }
        dummy_vehicle_params = {'payload': 50.0, 'cost_per_km': 1.8, 'speed_kmph': 40.0}
        dummy_drone_params = {'payload': 5.0, 'max_flight_distance_km': 8.0, 'cost_per_km': 0.7, 'speed_kmph': 60.0}

        # Use very small iteration counts for a quick test
        dummy_optimization_params = {
            'unmet_demand_penalty': 10000.0,
            'output_dir': 'output/optimizer_standalone_test',
            'genetic_algorithm_params': {'population_size': 10, 'num_generations': 5, 'mutation_rate': 0.2,
                                         'crossover_rate': 0.8, 'elite_count': 1, 'tournament_size': 3},
            'simulated_annealing_params': {'initial_temperature': 200, 'cooling_rate': 0.95, 'max_iterations': 50},
            'pso_optimizer_params': {'num_particles': 10, 'max_iterations': 10, 'inertia_weight': 0.7,
                                     'cognitive_weight': 1.5, 'social_weight': 1.5},
            'greedy_heuristic_params': {},
        }
        dummy_selected_algorithms = ['greedy_heuristic', 'genetic_algorithm', 'simulated_annealing']
        dummy_objective_weights = {'cost_weight': 0.7, 'time_weight': 0.3}
        logger.info("--- [Test] Dummy data and parameters created. ---")

    except Exception as e:
        logger.error(f"--- [Test] Error creating dummy data: {e} ---", exc_info=True)
        sys.exit(1)

    # --- Execute the Test Run ---
    logger.info(f"--- [Test] Calling run_optimization with algorithms: {dummy_selected_algorithms} ---")
    try:
        final_results = run_optimization(
            problem_data=dummy_problem_data,
            vehicle_params=dummy_vehicle_params,
            drone_params=dummy_drone_params,
            optimization_params=dummy_optimization_params,
            selected_algorithm_keys=dummy_selected_algorithms,
            objective_weights=dummy_objective_weights
        )

        # --- Print a Formatted Summary of the Results ---
        print("\n" + "=" * 40 + " FINAL OPTIMIZATION SUMMARY " + "=" * 40)
        print(f"  Overall Status: {final_results.get('overall_status')}")
        if final_results.get('error_message'):
            print(f"  Error Message: {final_results.get('error_message')}")
        print(f"  Output Directory: {final_results.get('output_directory')}")
        print(f"  Total Time: {final_results.get('total_computation_time', 0.0):.4f}s")
        print("-" * 100)
        print(f"  Best Overall Algorithm: {final_results.get('best_algorithm_key', 'N/A')}")
        print(f"  Best Weighted Cost:     {format_float(final_results.get('best_weighted_cost', float('inf')), 4)}")
        print(f"  Best Feasible Algorithm: {final_results.get('fully_served_best_key', 'N/A')}")
        print(
            f"  Best Feasible Cost:      {format_float(final_results.get('fully_served_best_cost', float('inf')), 4)}")
        print("-" * 100)

        for algo_key, summary in final_results.get('results_by_algorithm', {}).items():
            print(f"\n  Algorithm: {summary.get('algorithm_name', 'Unknown')}")
            print(f"    - Status: {summary.get('status', 'Unknown')}")
            print(f"    - Runtime: {summary.get('computation_time', 0.0):.4f}s")
            if summary.get('run_error'):
                print(f"    - ERROR: {summary.get('run_error')}")
            if summary.get('result_data'):
                res_data = summary['result_data']
                print(f"    - Weighted Cost: {format_float(res_data.get('weighted_cost'), 4)}")
                print(f"    - Is Feasible:   {res_data.get('is_feasible')}")
                print(f"    - Map Path:      {summary.get('map_path', 'Not generated')}")
                print(f"    - Report Path:   {summary.get('report_path', 'Not generated')}")
        print("\n" + "=" * 104)

    except Exception as e:
        logger.error("--- [Test] A critical, unhandled error occurred during the main test execution. ---",
                     exc_info=True)

    logger.info("--- Standalone Test for route_optimizer.py Finished ---")