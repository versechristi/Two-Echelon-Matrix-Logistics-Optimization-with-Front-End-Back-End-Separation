# algorithm/genetic_algorithm.py
# -*- coding: utf-8 -*-
"""
An advanced implementation of a Genetic Algorithm (GA) tailored for the
Multi-Depot, Two-Echelon Vehicle Routing Problem with Drones and Split Deliveries
(MD-2E-VRPSD).

This module provides a highly configurable and extensible Genetic Algorithm that
serves as a primary optimization engine for the VRP. It is designed to operate
independently, starting from a randomly generated population to ensure an unbiased
search of the solution space. This approach allows for a fair comparison against
other metaheuristics and provides a true measure of the GA's convergence capabilities.

Key Features:
- **Independent Initialization**: Creates a diverse initial population of random
  solutions, avoiding any bias from pre-optimized or greedy starting points.
- **Rich Operator Suite**: Implements multiple, selectable strategies for each
  core genetic operation, allowing for fine-tuning and experimentation:
  - **Selection**: Tournament, Roulette Wheel, and Rank-Based selection.
  - **Crossover**: Ordered Crossover (OX1), Partially Mapped Crossover (PMX),
    and Cycle Crossover (CX).
- **Memetic Algorithm (MA) Capability**: Integrates an optional local search
  phase (using a 2-Opt heuristic), transforming the GA into a more powerful
  Memetic Algorithm to enhance solution refinement.
- **Adaptive Parameter Control**: Features an adaptive mutation rate mechanism
  that adjusts based on population diversity to balance exploration and
  exploitation dynamically.
- **Comprehensive Tracking**: Monitors and logs key performance indicators
  throughout the run, including best cost, average cost, and population diversity.
- **Feasibility-Driven Evolution**: Utilizes the `SolutionCandidate` class's
  built-in comparison logic, which prioritizes feasible solutions (those with
  all demand met) during selection and evolution, guiding the search toward
  valid and practical outcomes.
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
        format="%(asctime)s [%(levelname)-8s] (GA) %(message)s"
    )

# =================================================================================
#  Safe Core Project Imports
# =================================================================================
# Ensures that all necessary components from other core modules are available.
# This block provides informative error messages if a critical dependency is
# missing, which is vital for diagnosing project setup issues.
try:
    # This assumes the script is run from a context where the project root is in sys.path
    from core.problem_utils import (
        SolutionCandidate,
        create_initial_solution,  # The new factory function
        swap_mutation,
        scramble_mutation,
        inversion_mutation,
        two_opt_mutation,  # For local search (Memetic Algorithm)
        ordered_crossover_ox1,
        partially_mapped_crossover_ox1, generate_neighbor_solution,  # Also aliased as PMX
    # Note: Cycle Crossover (CX) will be implemented locally in this file
    # as it's a classic GA operator.
)
    from core.distance_calculator import haversine
    from core.cost_function import calculate_total_cost_and_evaluate, format_float
    from core.problem_utils import create_heuristic_trips_split_delivery

except ImportError as e:
    logger.critical(f"A core module failed to import, which is essential for the GA. Error: {e}", exc_info=True)


    # Define dummy fallbacks to allow the script to load but fail gracefully
    # if used, preventing a hard crash at the import stage.
    class SolutionCandidate:
        def __init__(self, *args, **kwargs): self.weighted_cost = float('inf'); self.is_feasible = False

        def evaluate(self, *args, **kwargs): pass

        def __lt__(self, other): return False


    def create_initial_solution(*args, **kwargs):
        return None


    def swap_mutation(r):
        return r


    def scramble_mutation(r):
        return r


    def inversion_mutation(r):
        return r


    def two_opt_mutation(r):
        return r


    def ordered_crossover_ox1(p1, p2):
        return p1, p2


    def partially_mapped_crossover_ox1(p1, p2):
        return p1, p2


    def haversine(c1, c2):
        return 0


    def calculate_total_cost_and_evaluate(*args, **kwargs):
        return float('inf'), float('inf'), float('inf'), {}, True, True, {}


    def create_heuristic_trips_split_delivery(*args, **kwargs):
        return []


    warnings.warn("Genetic Algorithm will use dummy functions due to a critical import failure.")

# =================================================================================
#  Module-level Constants
# =================================================================================
FLOAT_TOLERANCE_GA = 1e-6


# =================================================================================
#  Selection Strategy Implementation
# =================================================================================

class SelectionStrategy:
    """
    A class that encapsulates different parent selection strategies for the GA.

    This class provides a structured way to choose and apply various selection
    mechanisms, making the GA more modular and extensible. Each method takes the
    population and the number of parents to select, returning a list of chosen
    parent individuals.
    """

    @staticmethod
    def tournament_selection(population: List[SolutionCandidate], num_parents: int, tournament_size: int) -> List[
        SolutionCandidate]:
        """
        Selects parents using K-way tournament selection.

        In each tournament, `tournament_size` individuals are chosen randomly from
        the population. The one with the best fitness (lowest weighted cost,
        prioritizing feasibility) wins the tournament and is selected as a parent.
        This process is repeated `num_parents` times.

        Args:
            population: The current population of solutions.
            num_parents: The total number of parents to select.
            tournament_size: The number of individuals competing in each tournament.

        Returns:
            A list containing the selected parent individuals.
        """
        if not population:
            return []

        selected_parents = []
        pop_size = len(population)

        # Ensure tournament size is valid
        eff_tournament_size = min(tournament_size, pop_size)
        if eff_tournament_size == 0:
            return []

        for _ in range(num_parents):
            # Randomly select competitors for the tournament
            tournament_competitors = random.sample(population, eff_tournament_size)
            # The winner is the best individual according to the __lt__ method of SolutionCandidate
            winner = min(tournament_competitors)
            selected_parents.append(winner)

        return selected_parents

    @staticmethod
    def roulette_wheel_selection(population: List[SolutionCandidate], num_parents: int) -> List[SolutionCandidate]:
        """
        Selects parents using fitness proportionate selection (Roulette Wheel).

        This method assigns a selection probability to each individual that is
        proportional to its fitness. To handle a minimization problem (lower cost is
        better), fitness values are inverted.

        Warning: This method can perform poorly if fitness values are very close or
        if there are extreme outliers, leading to premature convergence.

        Args:
            population: The current population of solutions.
            num_parents: The total number of parents to select.

        Returns:
            A list containing the selected parent individuals.
        """
        if not population:
            return []

        # Invert costs to create fitness values (higher is better)
        # Add a small epsilon to avoid division by zero for zero-cost solutions
        costs = np.array([ind.weighted_cost for ind in population])
        max_cost = np.max(costs[np.isfinite(costs)]) if np.any(np.isfinite(costs)) else 1.0
        fitness_values = max_cost - costs + FLOAT_TOLERANCE_GA
        fitness_values[~np.isfinite(costs)] = 0.0  # Assign zero fitness to Inf-cost individuals

        total_fitness = np.sum(fitness_values)
        if total_fitness <= FLOAT_TOLERANCE_GA:
            # If all fitness is zero (e.g., all solutions are Inf), select randomly
            return random.choices(population, k=num_parents)

        selection_probs = fitness_values / total_fitness

        # Select parents based on the calculated probabilities
        selected_indices = np.random.choice(len(population), size=num_parents, p=selection_probs)
        selected_parents = [population[i] for i in selected_indices]

        return selected_parents

    @staticmethod
    def rank_based_selection(population: List[SolutionCandidate], num_parents: int) -> List[SolutionCandidate]:
        """
        Selects parents using rank-based selection.

        This method first ranks the individuals based on their fitness (from best
        to worst). The selection probability is then based on the rank, not the
        raw fitness score. This helps to prevent premature convergence caused by
        dominant individuals with exceptionally high fitness.

        Args:
            population: The current population of solutions. The population is assumed
                to be pre-sorted from best to worst.
            num_parents: The total number of parents to select.

        Returns:
            A list containing the selected parent individuals.
        """
        if not population:
            return []

        pop_size = len(population)
        # Ranks are assigned linearly from worst (rank 1) to best (rank pop_size)
        ranks = np.arange(1, pop_size + 1)
        # Reverse to give higher probability to better ranks (lower index in sorted list)
        rank_probs = ranks[::-1] / np.sum(ranks)

        selected_indices = np.random.choice(pop_size, size=num_parents, p=rank_probs)
        selected_parents = [population[i] for i in selected_indices]

        return selected_parents

    @classmethod
    def get_selection_function(cls, strategy_name: str) -> Callable:
        """Factory method to retrieve a selection function by its name."""
        if strategy_name == 'tournament':
            return cls.tournament_selection
        elif strategy_name == 'roulette':
            return cls.roulette_wheel_selection
        elif strategy_name == 'rank':
            return cls.rank_based_selection
        else:
            logger.warning(f"Unknown selection strategy '{strategy_name}'. Defaulting to 'tournament'.")
            return cls.tournament_selection


# =================================================================================
#  Crossover Strategy Implementation
# =================================================================================

def cycle_crossover_cx(parent1_route: List, parent2_route: List) -> Tuple[List, List]:
    """
    Performs Cycle Crossover (CX) on two parent routes.

    CX is a powerful operator that preserves the absolute position of elements
    from the parents. It identifies cycles of elements between the two parents
    and creates offspring by alternating which parent's cycle is copied.

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

    # Track visited indices to handle multiple cycles
    indices_visited = [False] * size
    cycles = []

    for i in range(size):
        if not indices_visited[i]:
            cycle = []
            start_index = i
            current_index = i

            while True:
                cycle.append(current_index)
                indices_visited[current_index] = True
                element_from_p2 = p2[current_index]
                # Find this element's position in p1
                try:
                    current_index = p1.index(element_from_p2)
                except ValueError:
                    # This case indicates inconsistent parent data (elements don't match)
                    warnings.warn("Inconsistent parent data in Cycle Crossover. Falling back to parent copies.")
                    return p1, p2

                if current_index == start_index:
                    break
            cycles.append(cycle)

    # Create offspring by alternating cycles
    for i, cycle in enumerate(cycles):
        if i % 2 == 0:  # Even cycles (including the first) from parent 1 to child 1
            for index in cycle:
                child1[index] = p1[index]
                child2[index] = p2[index]
        else:  # Odd cycles from parent 2 to child 1
            for index in cycle:
                child1[index] = p2[index]
                child2[index] = p1[index]

    return child1, child2


def get_crossover_function(strategy_name: str) -> Callable:
    """Factory method to retrieve a crossover function by its name."""
    if strategy_name == 'ox1':
        return ordered_crossover_ox1
    elif strategy_name == 'pmx':
        return partially_mapped_crossover_ox1
    elif strategy_name == 'cx':
        return cycle_crossover_cx
    else:
        logger.warning(f"Unknown crossover strategy '{strategy_name}'. Defaulting to 'ox1'.")
        return ordered_crossover_ox1


# =================================================================================
#  GA Main Orchestration Function
# =================================================================================

def run_genetic_algorithm(
        problem_data: Dict[str, Any],
        vehicle_params: Dict[str, Any],
        drone_params: Dict[str, Any],
        objective_params: Dict[str, float],
        algo_specific_params: Dict[str, Any],
        initial_solution_candidate: Optional[SolutionCandidate] = None  # Now truly optional
) -> Dict[str, Any]:
    """
    Executes the Genetic Algorithm to solve the MD-2E-VRPSD.

    This function orchestrates the entire GA process, from population
    initialization to the final evolution loop. It is designed to be highly
    configurable through the `algo_specific_params` dictionary, allowing
    for control over population size, generation count, operator selection,
    and advanced features like adaptive mutation and local search.

    Args:
        problem_data: The static VRP instance data.
        vehicle_params: Parameters for the vehicle fleet.
        drone_params: Parameters for the drone fleet.
        objective_params: Weights and penalties for the objective function.
        algo_specific_params: A dictionary of GA-specific hyperparameters. Expected keys:
            - `population_size` (int)
            - `num_generations` (int)
            - `mutation_rate` (float)
            - `crossover_rate` (float)
            - `elite_count` (int)
            - `selection_strategy` (str): 'tournament', 'roulette', or 'rank'.
            - `tournament_size` (int): Required if selection_strategy is 'tournament'.
            - `crossover_strategy` (str): 'ox1', 'pmx', or 'cx'.
            - `use_adaptive_mutation` (bool): Whether to enable adaptive mutation.
            - `use_local_search` (bool): Whether to enable the memetic local search step.
        initial_solution_candidate: An optional, pre-existing solution. If provided,
            it will be injected into the initial population. Otherwise, the population
            is generated randomly from scratch.

    Returns:
        A dictionary containing the results of the GA run, including the best
        solution found, its evaluation metrics, and performance history.
    """
    run_start_time = time.time()
    logger.info("--- Genetic Algorithm (MD-2E-VRPSD) Started ---")

    # --- 1. Validate and Configure GA Parameters ---
    logger.info("Configuring GA parameters...")
    try:
        params = _configure_ga_parameters(algo_specific_params)
        selection_func = SelectionStrategy.get_selection_function(params['selection_strategy'])
        crossover_func = get_crossover_function(params['crossover_strategy'])

        logger.info(f"GA Configuration: PopSize={params['population_size']}, Gens={params['num_generations']}, "
                    f"Selection='{params['selection_strategy']}', Crossover='{params['crossover_strategy']}', "
                    f"MutationRate={params['mutation_rate']}, CrossoverRate={params['crossover_rate']}, "
                    f"Elitism={params['elite_count']}, AdaptiveMutation={params['use_adaptive_mutation']}, "
                    f"MemeticLocalSearch={params['use_local_search']}")

    except (ValueError, KeyError) as e:
        error_msg = f"GA parameter validation failed: {e}"
        logger.error(error_msg, exc_info=True)
        return {'run_error': error_msg}

    # --- 2. Initialize Population ---
    logger.info("Initializing population...")
    try:
        population = _initialize_population(
            pop_size=params['population_size'],
            problem_data=problem_data,
            vehicle_params=vehicle_params,
            drone_params=drone_params,
            objective_params=objective_params,
            seeded_solution=initial_solution_candidate
        )
        if not population:
            raise RuntimeError("Population initialization returned an empty list.")
    except Exception as e:
        error_msg = f"Failed to initialize GA population: {e}"
        logger.error(error_msg, exc_info=True)
        return {'run_error': error_msg}

    # --- 3. Main Evolution Loop ---
    logger.info("Starting GA evolution loop...")
    best_solution_overall: Optional[SolutionCandidate] = None
    best_cost_history: List[float] = []
    avg_cost_history: List[float] = []
    diversity_history: List[float] = []

    current_mutation_rate = params['mutation_rate']

    for generation in range(params['num_generations']):
        # --- a. Evaluate and Sort Population ---
        # The evaluate_population function handles the fitness calculation for each individual.
        _evaluate_population(population, objective_params)

        # Sort the population based on fitness (best solutions first).
        # This is crucial for elitism and rank-based selection.
        population.sort()

        # --- b. Update Best Solution and Track History ---
        current_best_in_gen = population[0]
        if best_solution_overall is None or current_best_in_gen < best_solution_overall:
            best_solution_overall = copy.deepcopy(current_best_in_gen)
            logger.debug(
                f"Gen {generation + 1}: New best solution found! Cost: {format_float(best_solution_overall.weighted_cost, 4)}, Feasible: {best_solution_overall.is_feasible}")

        best_cost_history.append(best_solution_overall.weighted_cost)

        valid_costs = [p.weighted_cost for p in population if math.isfinite(p.weighted_cost)]
        avg_cost_history.append(np.mean(valid_costs) if valid_costs else float('inf'))

        # --- c. Track Diversity and Adapt Mutation Rate ---
        diversity = _calculate_population_diversity(population)
        diversity_history.append(diversity)

        if params['use_adaptive_mutation']:
            current_mutation_rate = _adapt_mutation_rate(
                base_rate=params['mutation_rate'],
                diversity=diversity,
                generation=generation,
                total_generations=params['num_generations']
            )

        # --- d. Log Generation Progress ---
        if (generation + 1) % 10 == 0 or generation == params['num_generations'] - 1:
            logger.info(f"Gen {generation + 1}/{params['num_generations']} | "
                        f"Best Cost: {format_float(best_cost_history[-1], 2)} | "
                        f"Avg Cost: {format_float(avg_cost_history[-1], 2)} | "
                        f"Diversity: {format_float(diversity, 4)} | "
                        f"Mut. Rate: {format_float(current_mutation_rate, 4)}")

        # --- e. Create Next Generation ---
        next_generation = []

        # Elitism: Directly carry over the best individuals.
        if params['elite_count'] > 0:
            elites = [copy.deepcopy(p) for p in population[:params['elite_count']]]
            next_generation.extend(elites)

        # --- f. Perform Selection, Crossover, and Mutation ---
        num_offspring_to_create = params['population_size'] - len(next_generation)

        # Select parents for breeding
        parent_selection_args = {
            'population': population,
            'num_parents': num_offspring_to_create * 2  # Need two parents per crossover
        }
        if params['selection_strategy'] == 'tournament':
            parent_selection_args['tournament_size'] = params['tournament_size']

        parents = selection_func(**parent_selection_args)

        # Create offspring through crossover and mutation
        for i in range(0, num_offspring_to_create, 2):
            if i + 1 >= len(parents): break  # Handle odd number of parents

            parent1, parent2 = parents[i], parents[i + 1]

            # Crossover
            if random.random() < params['crossover_rate']:
                offspring1, offspring2 = _crossover(parent1, parent2, crossover_func)
            else:
                offspring1, offspring2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

            # Mutation
            mutated_offspring1 = _mutate(offspring1, current_mutation_rate)
            mutated_offspring2 = _mutate(offspring2, current_mutation_rate)

            # Optional Local Search (Memetic Algorithm)
            if params['use_local_search']:
                mutated_offspring1 = _local_search_improvement(mutated_offspring1)
                mutated_offspring2 = _local_search_improvement(mutated_offspring2)

            next_generation.extend([mutated_offspring1, mutated_offspring2])

        # Replace the old population with the new generation
        population = next_generation[:params['population_size']]  # Ensure correct size

    # --- 4. Finalization and Result Packaging ---
    run_end_time = time.time()
    logger.info(f"--- Genetic Algorithm Finished in {run_end_time - run_start_time:.2f} seconds ---")

    if best_solution_overall:
        logger.info(
            f"Final Best Solution: Feasible={best_solution_overall.is_feasible}, Weighted Cost={format_float(best_solution_overall.weighted_cost, 4)}")
        # Final, consistent evaluation of the best solution found
        best_solution_overall.evaluate(haversine, create_heuristic_trips_split_delivery)
    else:
        logger.warning("GA run completed, but no valid best solution was found.")

    # Package the comprehensive results into a dictionary
    ga_results = {
        'best_solution': best_solution_overall,
        'cost_history': best_cost_history,
        'avg_cost_history': avg_cost_history,
        'diversity_history': diversity_history,
        'total_computation_time': run_end_time - run_start_time,
        'algorithm_name': 'genetic_algorithm',
        'algorithm_params': params,
    }

    return ga_results


# =================================================================================
#  Private Helper and Core GA Component Functions
# =================================================================================

def _configure_ga_parameters(user_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates and configures GA hyperparameters, merging user inputs with defaults.

    Args:
        user_params: A dictionary of parameters provided by the user/caller.

    Returns:
        A validated and complete dictionary of GA parameters.

    Raises:
        ValueError: If a required parameter is missing or invalid.
    """
    defaults = {
        'population_size': 100,
        'num_generations': 500,
        'mutation_rate': 0.1,
        'crossover_rate': 0.8,
        'elite_count': 2,
        'selection_strategy': 'tournament',
        'tournament_size': 5,
        'crossover_strategy': 'ox1',
        'use_adaptive_mutation': False,
        'use_local_search': False,
    }
    params = defaults.copy()
    if isinstance(user_params, dict):
        params.update(user_params)

    # --- Perform Validation ---
    if not (isinstance(params['population_size'], int) and params['population_size'] > 1):
        raise ValueError("`population_size` must be an integer greater than 1.")
    if not (isinstance(params['num_generations'], int) and params['num_generations'] >= 0):
        raise ValueError("`num_generations` must be a non-negative integer.")
    if not (isinstance(params['mutation_rate'], float) and 0.0 <= params['mutation_rate'] <= 1.0):
        raise ValueError("`mutation_rate` must be a float between 0.0 and 1.0.")
    if not (isinstance(params['crossover_rate'], float) and 0.0 <= params['crossover_rate'] <= 1.0):
        raise ValueError("`crossover_rate` must be a float between 0.0 and 1.0.")
    if not (isinstance(params['elite_count'], int) and 0 <= params['elite_count'] < params['population_size']):
        raise ValueError("`elite_count` must be a non-negative integer less than the population size.")
    if params['selection_strategy'] == 'tournament' and not (
            isinstance(params['tournament_size'], int) and 1 < params['tournament_size'] <= params['population_size']):
        raise ValueError("`tournament_size` must be an integer between 2 and the population size.")

    return params


def _initialize_population(pop_size: int,
                           problem_data: Dict,
                           vehicle_params: Dict,
                           drone_params: Dict,
                           objective_params: Dict,
                           seeded_solution: Optional[SolutionCandidate]) -> List[SolutionCandidate]:
    """
    Creates the initial population for the GA.

    Generates `pop_size` individuals by calling the `create_initial_solution`
    factory with the 'random' strategy. If a `seeded_solution` is provided,
    it is included in the population to ensure it competes.

    Args:
        pop_size: The desired size of the population.
        problem_data, vehicle_params, drone_params, objective_params: Standard
            problem definition dictionaries.
        seeded_solution: An optional pre-existing solution to inject.

    Returns:
        A list of `SolutionCandidate` objects forming the initial population.
    """
    population = []
    logger.debug(f"Creating initial population of size {pop_size}...")

    # If a seed solution is provided, add it to the population
    if seeded_solution:
        population.append(copy.deepcopy(seeded_solution))
        logger.debug("Injected a pre-existing seed solution into the initial population.")

    # Fill the rest of the population with random individuals
    while len(population) < pop_size:
        # Generate a completely new random individual
        individual = create_initial_solution(
            strategy='random',
            problem_data=problem_data,
            vehicle_params=vehicle_params,
            drone_params=drone_params,
            objective_params=objective_params
        )
        if individual and not individual.initialization_error:
            population.append(individual)
        else:
            warnings.warn("Failed to create a valid random individual during population initialization. Trying again.")

    return population


def _evaluate_population(population: List[SolutionCandidate], objective_params: Dict):
    """
    Evaluates the fitness of each individual in the population.

    This function iterates through each `SolutionCandidate` in the population
    and calls its `evaluate` method. It ensures that the evaluation uses the
    correct, consistent objective parameters for the current run.
    """
    cost_weight = objective_params.get('cost_weight', 1.0)
    time_weight = objective_params.get('time_weight', 0.0)
    unmet_penalty = objective_params.get('unmet_demand_penalty', 1e9)

    for individual in population:
        # The `evaluate` method is called on the SolutionCandidate object itself.
        # It handles all the complex calculations internally.
        individual.evaluate(
            distance_func=haversine,
            stage2_trip_generator_func=create_heuristic_trips_split_delivery
        )


def _crossover(parent1: SolutionCandidate, parent2: SolutionCandidate, crossover_func: Callable) -> Tuple[
    SolutionCandidate, SolutionCandidate]:
    """
    Performs crossover on the Stage 1 routes of two parent solutions.

    This function orchestrates the crossover operation. It randomly selects one
    depot's route to perform crossover on, applying the specified `crossover_func`
    (e.g., OX1, PMX, CX) to the corresponding routes from the two parents.
    The offspring inherit the rest of their routes directly from their parents.

    Args:
        parent1: The first parent SolutionCandidate.
        parent2: The second parent SolutionCandidate.
        crossover_func: The specific crossover operator function to use.

    Returns:
        A tuple containing two new, unevaluated offspring SolutionCandidates.
    """
    offspring1 = copy.deepcopy(parent1)
    offspring2 = copy.deepcopy(parent2)

    # Get a list of depots that have routes eligible for crossover (length >= 2)
    eligible_depots = [
        idx for idx, route in parent1.stage1_routes.items()
        if len(route) >= 2 and len(parent2.stage1_routes.get(idx, [])) >= 2
    ]

    if not eligible_depots:
        # If no depots are eligible, return copies of parents
        offspring1._reset_evaluation_results()
        offspring2._reset_evaluation_results()
        return offspring1, offspring2

    # Select one depot's route to apply crossover
    selected_depot = random.choice(eligible_depots)

    p1_route = parent1.stage1_routes[selected_depot]
    p2_route = parent2.stage1_routes[selected_depot]

    try:
        # Apply the chosen crossover function
        c1_route, c2_route = crossover_func(p1_route, p2_route)

        # Assign the new routes to the offspring
        offspring1.stage1_routes[selected_depot] = c1_route
        offspring2.stage1_routes[selected_depot] = c2_route

    except Exception as e:
        warnings.warn(
            f"Error during crossover '{crossover_func.__name__}' for depot {selected_depot}. Offspring will be copies of parents. Error: {e}")
        # On failure, offspring remain copies of parents
        pass

    # Mark offspring as unevaluated
    offspring1._reset_evaluation_results()
    offspring2._reset_evaluation_results()

    return offspring1, offspring2


def _mutate(individual: SolutionCandidate, mutation_rate: float) -> SolutionCandidate:
    """
    Applies a random mutation to the Stage 1 routes of an individual.

    For each depot route in the individual's solution, there is a `mutation_rate`
    chance that a mutation operator (swap, inversion, or scramble) will be applied.

    Args:
        individual: The SolutionCandidate to mutate.
        mutation_rate: The probability of applying mutation to each route.

    Returns:
        The mutated individual (which may be the same as the input if no
        mutation occurred).
    """
    mutated_individual = individual  # No need to copy if mutation is applied in-place to a copy

    mutated = False
    for depot_idx, route in mutated_individual.stage1_routes.items():
        if len(route) >= 2 and random.random() < mutation_rate:
            mutated = True
            # Choose a random mutation operator
            operator = random.choice([swap_mutation, inversion_mutation, scramble_mutation])
            mutated_individual.stage1_routes[depot_idx] = operator(route)

    if mutated:
        # If any mutation occurred, the individual must be re-evaluated.
        mutated_individual._reset_evaluation_results()

    return mutated_individual


def _local_search_improvement(individual: SolutionCandidate, num_iterations: int = 10) -> SolutionCandidate:
    """
    Performs a simple local search on an individual to find nearby improvements.
    This is the core of a Memetic Algorithm.

    It repeatedly applies a 2-Opt mutation and accepts the move only if it
    improves the solution's fitness.

    Args:
        individual: The SolutionCandidate to improve.
        num_iterations: The number of local search attempts to make.

    Returns:
        The improved (or original, if no improvement found) SolutionCandidate.
    """
    improved_individual = individual
    # The individual must be evaluated first to have a baseline cost
    if improved_individual.weighted_cost == float('inf'):
        improved_individual.evaluate(haversine, create_heuristic_trips_split_delivery)

    for _ in range(num_iterations):
        # Generate a neighbor using a specific, powerful local search operator
        neighbor = generate_neighbor_solution(improved_individual, operator=two_opt_mutation)
        if not neighbor:
            continue

        neighbor.evaluate(haversine, create_heuristic_trips_split_delivery)

        # Accept the neighbor only if it's strictly better
        if neighbor < improved_individual:
            improved_individual = neighbor
            # Since we accepted the move, the individual is already evaluated.

    return improved_individual


def _calculate_population_diversity(population: List[SolutionCandidate]) -> float:
    """
    Calculates a measure of diversity for the current population.

    A simple diversity metric is the average pairwise distance between the
    Stage 1 routes of all individuals in the population. This uses a simplified
    "distance" measure between two permutations.

    Args:
        population: The list of SolutionCandidate individuals.

    Returns:
        A float representing the diversity score (higher is more diverse).
    """
    if len(population) < 2:
        return 0.0

    total_distance = 0
    num_pairs = 0

    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            p1 = population[i]
            p2 = population[j]

            # Calculate distance between the routes of the two individuals
            # A simple metric: count differing elements at the same position
            # across all depot routes.
            dist = 0
            all_depots = set(p1.stage1_routes.keys()) | set(p2.stage1_routes.keys())
            for depot_idx in all_depots:
                r1 = p1.stage1_routes.get(depot_idx, [])
                r2 = p2.stage1_routes.get(depot_idx, [])
                if len(r1) != len(r2):
                    dist += abs(len(r1) - len(r2))  # Penalize different lengths
                    continue
                for k in range(len(r1)):
                    if r1[k] != r2[k]:
                        dist += 1

            total_distance += dist
            num_pairs += 1

    return (total_distance / num_pairs) if num_pairs > 0 else 0.0


def _adapt_mutation_rate(base_rate: float, diversity: float, generation: int, total_generations: int) -> float:
    """
    Adapts the mutation rate based on population diversity and search progress.

    - If diversity is low, increase mutation rate to encourage exploration.
    - If diversity is high, decrease it to focus on exploitation.
    - The effect can be modulated over the course of the run.

    Args:
        base_rate: The initial or baseline mutation rate.
        diversity: The current population diversity score.
        generation: The current generation number.
        total_generations: The total number of generations for the run.

    Returns:
        The newly adapted mutation rate.
    """
    # Simple adaptation logic:
    # Define thresholds for low and high diversity. These may need tuning.
    low_diversity_threshold = 1.0
    high_diversity_threshold = 5.0

    if diversity < low_diversity_threshold:
        # Diversity is low, increase mutation to escape local optima
        adapted_rate = base_rate * 2.0
    elif diversity > high_diversity_threshold:
        # Diversity is high, decrease mutation to allow good solutions to converge
        adapted_rate = base_rate * 0.5
    else:
        # Diversity is in a healthy range, use the base rate
        adapted_rate = base_rate

    # Ensure the rate stays within reasonable bounds [0.01, 0.5]
    adapted_rate = max(0.01, min(adapted_rate, 0.5))

    return adapted_rate