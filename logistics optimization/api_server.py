# =======================================================================================
# |                     Advanced Logistics Optimization API Server                      |
# |                                  Version: 2.1                                     |
# =======================================================================================
#
# This script implements a high-performance, asynchronous API server using FastAPI
# to support the Logistics Optimization System. It serves as the backend engine,
# handling computationally intensive tasks, data management, and communication
# with the frontend user interface.
#
# Key Architectural Enhancements in this Version:
#
# 1.  **Critical Bug Fix (Parameter Mismatch)**:
#     - Resolved the "Incomplete params" error by using Pydantic's `Field(alias=...)`
#       in the `VehicleParams` and `DroneParams` models. This allows the API to accept
#       frontend-friendly names (e.g., "max_payload_kg") while providing the core
#       Python logic with the expected internal names (e.g., "payload"), ensuring
#       seamless validation and execution.
#
# 2.  **New Feature (Initial Layout Map)**:
#     - The `/generate-data` endpoint now generates and saves an initial layout map
#       showing the distribution of all generated points (depots, outlets, customers)
#       immediately after data creation.
#     - The endpoint's response has been enhanced to include the URL path to this
#       initial map, enabling the frontend to display a preview instantly.
#
# 3.  **Robust Task Management & State Feedback**:
#     - A dedicated, thread-safe `TaskManager` class manages all background tasks,
#       ensuring data integrity in a multithreaded environment.
#     - The system provides more granular, real-time progress updates to the frontend,
#       offering a transparent view of the entire computation lifecycle.
#
# 4.  **Strict Data Validation & Rich API Documentation**:
#     - Pydantic models are defined with precise validation rules and descriptive
#       titles, which FastAPI uses to build a comprehensive, interactive API documentation.
#
# =======================================================================================

# --- Standard Library Imports ---
import asyncio
import os
import sys
import time
import traceback
import uuid
import threading
import configparser
import io
import logging
import json
from typing import Dict, Any, List, Optional

# --- FastAPI and Pydantic Imports ---
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# --- Configure Centralized Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)-8s] - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Add Project Root to Python Path for Robust Module Imports ---
try:
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    logger.info(f"Project root successfully added to sys.path: {project_root}")
except Exception as e:
    logger.critical(f"FATAL: Failed to configure sys.path. The application cannot start. Error: {e}")
    sys.exit(1)

# --- Import Core Project Logic ---
try:
    from core.route_optimizer import run_optimization
    from data.data_generator import generate_locations, generate_demand
    from visualization.map_generator import generate_folium_map
except ImportError as e:
    logger.critical(
        f"FATAL: Failed to import core modules (e.g., 'run_optimization'). "
        f"Ensure all project modules are correctly structured. Error: {e}"
    )
    sys.exit(1)


# ==============================================================================
#                      Thread-Safe Task Management System
# ==============================================================================

class TaskManager:
    """
    A thread-safe class to manage the state of long-running background tasks.
    This provides a robust, centralized system for tracking task progress,
    results, and errors.
    """

    def __init__(self):
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create_task(self) -> str:
        """Creates a new task entry and returns its unique ID."""
        task_id = str(uuid.uuid4())
        with self._lock:
            self._tasks[task_id] = {
                "status": "starting",
                "progress": 0,
                "message": "Initializing optimization task...",
                "results": None,
                "error": None,
                "logs": []
            }
        logger.info(f"New task created with ID: {task_id}")
        return task_id

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the current state of a task."""
        with self._lock:
            return self._tasks.get(task_id)

    def update_task(self, task_id: str, progress: int, message: str):
        """Updates a task's progress and status message."""
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task["progress"] = max(0, min(100, progress))
                task["message"] = message
                task["logs"].append(f"[{time.strftime('%H:%M:%S')}] {message}")

    def set_completed(self, task_id: str, results: Dict, message: str = "Task completed successfully."):
        """Marks a task as 'completed' and stores the final results."""
        self.update_task(task_id, 100, message)
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id]["status"] = "completed"
                self._tasks[task_id]["results"] = results
        logger.info(f"Task {task_id} has been marked as COMPLETED.")

    def set_failed(self, task_id: str, error_message: str):
        """Marks a task as 'failed' and stores the error details."""
        self.update_task(task_id, 100, f"Error: {error_message}")
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id]["status"] = "failed"
                self._tasks[task_id]["error"] = error_message
        logger.error(f"Task {task_id} has been marked as FAILED. Reason: {error_message}")


task_manager = TaskManager()

# ==============================================================================
#                           FastAPI Application Setup
# ==============================================================================

app = FastAPI(
    title="Advanced Logistics Optimization API",
    description="A high-performance backend API for the MD-2E-VRPSD.",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

output_dir = os.path.join(project_root, "output")
os.makedirs(output_dir, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=output_dir), name="outputs")


# ==============================================================================
#                     Pydantic Models for Data Validation
# ==============================================================================

class DataGenerationParams(BaseModel):
    num_logistics_centers: int = Field(..., ge=1)
    num_sales_outlets: int = Field(..., ge=1)
    num_customers: int = Field(..., ge=1)
    use_solomon_like_distribution: bool = False
    center_latitude: float
    center_longitude: float
    radius_km: float = Field(..., gt=0)
    min_demand: float = Field(..., ge=0)
    max_demand: float = Field(..., ge=0)


class VehicleParams(BaseModel):
    # CRITICAL FIX: Use aliases to match frontend keys while using internal names
    # expected by the core logic.
    payload: float = Field(..., gt=0, alias='max_payload_kg', title="Maximum vehicle payload (kg).")
    cost_per_km: float = Field(..., ge=0, title="Vehicle travel cost per kilometer.")
    speed_kmph: float = Field(..., gt=0, alias='speed_kmh', title="Average vehicle speed (km/h).")


class DroneParams(BaseModel):
    # CRITICAL FIX: Use aliases here as well.
    payload: float = Field(..., gt=0, alias='max_payload_kg', title="Maximum drone payload (kg).")
    cost_per_km: float = Field(..., ge=0, title="Drone travel cost per kilometer.")
    speed_kmph: float = Field(..., gt=0, alias='speed_kmh', title="Average drone speed (km/h).")
    max_flight_distance_km: float = Field(..., gt=0, title="Maximum drone round-trip flight range.")


class ObjectiveParams(BaseModel):
    cost_weight: float = Field(..., ge=0, le=1)
    time_weight: float = Field(..., ge=0, le=1)
    unmet_demand_penalty: float = Field(..., ge=0)

class GAParams(BaseModel):
    population_size: int
    num_generations: int
    mutation_rate: float
    crossover_rate: float
    elite_count: int
    tournament_size: int

class SAParams(BaseModel):
    initial_temperature: float
    cooling_rate: float
    max_iterations: int
    min_temperature: float

class PSOParams(BaseModel):
    num_particles: int
    max_iterations: int
    inertia_weight: float
    cognitive_weight: float
    social_weight: float


class AlgorithmSpecificParams(BaseModel):
    genetic_algorithm: GAParams
    simulated_annealing: SAParams
    pso_optimizer: PSOParams
    greedy_heuristic: Dict[str, Any]


class LocationData(BaseModel):
    logistics_centers: List[List[float]]
    sales_outlets: List[List[float]]
    customers: List[List[float]]


class ProblemData(BaseModel):
    locations: LocationData
    demands: List[float]


# NEW: Define a response model for the data generation endpoint
class DataGenerationResponse(BaseModel):
    problem_data: ProblemData
    initial_map_path: Optional[str] = None


class OptimizationPayload(BaseModel):
    problem_data: ProblemData
    vehicle: VehicleParams  # Pydantic will automatically use the alias for input
    drone: DroneParams  # Pydantic will automatically use the alias for input
    objective: ObjectiveParams
    algorithm_params: AlgorithmSpecificParams
    selected_algorithms: List[str] = Field(..., min_length=1)


class ConfigPayload(BaseModel):
    data_generation: DataGenerationParams
    vehicle: VehicleParams
    drone: DroneParams
    objective: ObjectiveParams
    algorithm_params: AlgorithmSpecificParams


# ==============================================================================
#                      Serve Frontend Static Files
# ==============================================================================

# Mount the 'fronted' directory to serve static files (js, css)
frontend_dir = os.path.join(project_root, "fronted")
app.mount("/fronted", StaticFiles(directory=frontend_dir), name="fronted")

# Set the root endpoint to return the main index.html file
@app.get("/", include_in_schema=False)
async def read_index():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        raise HTTPException(status_code=404, detail="index.html not found")

# ==============================================================================
#                                  API Endpoints
# ==============================================================================

@app.get("/api-health", tags=["General"], summary="API Health Check") # Renamed to avoid conflict
def read_root():
    """Confirms that the API server is online and operational."""
    return {"message": "Advanced Logistics Optimization API is online."}


@app.post("/generate-data", tags=["Data Generation"], response_model=DataGenerationResponse,
          summary="Generate Problem Data and Initial Layout Map")
async def generate_data_endpoint(params: DataGenerationParams):
    """
    Generates a synthetic problem instance. After successful data generation,
    it also creates an initial layout map showing the distribution of all points
    and returns the path to this map along with the problem data.
    """
    logger.info(f"Received data generation request: {params.model_dump(by_alias=True)}")
    try:
        location_generation_args = {
            "num_logistics_centers": params.num_logistics_centers,
            "num_sales_outlets": params.num_sales_outlets,
            "num_customers": params.num_customers,
            "center_latitude": params.center_latitude,
            "center_longitude": params.center_longitude,
            "radius_km": params.radius_km,
            "use_solomon_like_distribution": params.use_solomon_like_distribution
        }
        locations = generate_locations(**location_generation_args)
        if not locations:
            raise HTTPException(status_code=500, detail="Location generation failed on backend.")

        num_actual_customers = len(locations.get('customers', []))
        demands = generate_demand(
            num_customers=num_actual_customers,
            min_demand=params.min_demand,
            max_demand=params.max_demand
        )
        if demands is None:
            raise HTTPException(status_code=500, detail="Demand generation failed on backend.")

        problem_data_dict = {"locations": locations, "demands": demands}

        # --- NEW FEATURE: Generate Initial Layout Map ---
        initial_map_path = None
        try:
            run_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            map_dir = os.path.join(output_dir, run_timestamp, "maps")
            os.makedirs(map_dir, exist_ok=True)
            map_filename = f"initial_layout_{run_timestamp}.html"
            full_map_path = os.path.join(map_dir, map_filename)

            # Call map generator with no solution to plot only points
            generated_path = generate_folium_map(
                problem_data=problem_data_dict,
                solution_structure=None,
                vehicle_params={},  # Not needed for point map
                drone_params={},  # Not needed for point map
                output_path=full_map_path
            )
            if generated_path:
                # Return a relative URL that the frontend can use
                initial_map_path = f"/outputs/{run_timestamp}/maps/{map_filename}"
                logger.info(f"Successfully generated initial layout map at: {initial_map_path}")
        except Exception as map_e:
            logger.error(f"Failed to generate initial layout map, but continuing. Error: {map_e}")

        return {"problem_data": problem_data_dict, "initial_map_path": initial_map_path}

    except Exception as e:
        logger.error(f"An unexpected error in /generate-data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")


@app.post("/run-optimization", tags=["Optimization"], status_code=202, summary="Launch a New Optimization Task")
async def run_optimization_endpoint(payload: OptimizationPayload, background_tasks: BackgroundTasks):
    """
    Accepts a problem definition and launches a long-running optimization task.
    Returns a task ID for status polling.
    """
    task_id = task_manager.create_task()
    background_tasks.add_task(run_optimization_task, task_id, payload)
    logger.info(f"Accepted and launched optimization task with ID: {task_id}")
    return {"task_id": task_id, "status": "accepted"}


@app.get("/check-status/{task_id}", tags=["Optimization"], summary="Poll for Task Status")
async def check_task_status(task_id: str):
    """Allows the client to poll for the real-time status of a task."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task with ID '{task_id}' not found.")
    return task


@app.get("/get-results/{task_id}", tags=["Optimization"], summary="Retrieve Final Task Results")
async def get_task_results(task_id: str):
    """Returns the final results of a successfully completed task."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task with ID '{task_id}' not found.")
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Task not complete or failed. Status: {task['status']}")
    return task.get("results")


@app.post("/load-config", tags=["Configuration"], summary="Load Parameters from .ini File")
async def load_config_endpoint(file: UploadFile = File(...)):
    """
    Receives an uploaded .ini file, parses its contents, and returns them as JSON.
    """
    if not file.filename.endswith('.ini'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .ini file.")
    contents = await file.read()
    config = configparser.ConfigParser()
    try:
        config.read_string(contents.decode('utf-8'))
        config_dict = {s.lower(): dict(config.items(s)) for s in config.sections()}

        if 'algorithm_params' in config_dict:
            for key, value in config_dict['algorithm_params'].items():
                try:
                    config_dict['algorithm_params'][key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Could not parse '{key}' as JSON, keeping as string.")

        # Convert section keys to match Pydantic model names
        if 'vehicle' in config_dict:
            config_dict['vehicle'] = {k.replace('_kg', '').replace('_kmh', '_kmph'): v for k, v in
                                      config_dict['vehicle'].items()}
        if 'drone' in config_dict:
            config_dict['drone'] = {k.replace('_kg', '').replace('_kmh', '_kmph'): v for k, v in
                                    config_dict['drone'].items()}

        return JSONResponse(content=config_dict)
    except Exception as e:
        logger.error(f"Failed to parse config file '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to parse .ini file: {e}")


@app.post("/save-config", tags=["Configuration"], summary="Save Parameters to .ini File")
async def save_config_endpoint(payload: ConfigPayload):
    """
    Receives parameters as JSON and generates a downloadable .ini file.
    """
    config = configparser.ConfigParser()

    config['DATA_GENERATION'] = {k: str(v) for k, v in payload.data_generation.model_dump().items()}
    # Use by_alias=True to write frontend-friendly keys to the file
    config['VEHICLE'] = {k: str(v) for k, v in payload.vehicle.model_dump(by_alias=True).items()}
    config['DRONE'] = {k: str(v) for k, v in payload.drone.model_dump(by_alias=True).items()}
    config['OBJECTIVE'] = {k: str(v) for k, v in payload.objective.model_dump().items()}
    config['ALGORITHM_PARAMS'] = {k: json.dumps(v) for k, v in payload.algorithm_params.model_dump().items()}

    string_io = io.StringIO()
    config.write(string_io)
    file_content = string_io.getvalue()

    return Response(
        content=file_content,
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename=config_{time.strftime('%Y%m%d_%H%M%S')}.ini"}
    )


# ==============================================================================
#                      BACKGROUND TASK IMPLEMENTATION
# ==============================================================================

def run_optimization_task(task_id: str, payload: OptimizationPayload):
    """
    The core background task function. It prepares parameters and calls the
    main optimization engine (`run_optimization`).
    """
    task_manager.update_task(task_id, 5, "Preparing optimization environment...")
    try:
        # The Pydantic model has already handled aliases. We can now convert the
        # validated models to dictionaries with the correct internal keys.
        vehicle_params_dict = payload.vehicle.model_dump()
        drone_params_dict = payload.drone.model_dump()

        optimization_params = {
            "unmet_demand_penalty": payload.objective.unmet_demand_penalty,
            "output_dir": output_dir,
            **{f"{key}_params": value for key, value in payload.algorithm_params.model_dump().items()}
        }
        objective_weights = {
            "cost_weight": payload.objective.cost_weight,
            "time_weight": payload.objective.time_weight,
        }

        task_manager.update_task(task_id, 10,
                                 f"Starting optimization for {len(payload.selected_algorithms)} algorithm(s)...")

        # --- Main, time-consuming call to the core logic ---
        final_results = run_optimization(
            problem_data=payload.problem_data.model_dump(),
            vehicle_params=vehicle_params_dict,
            drone_params=drone_params_dict,
            optimization_params=optimization_params,
            selected_algorithm_keys=payload.selected_algorithms,
            objective_weights=objective_weights,
            # Pass the task_id so the optimizer can provide progress updates if instrumented
            task_id=task_id
        )

        task_manager.update_task(task_id, 95, "Finalizing results and generating artifacts...")
        task_manager.set_completed(task_id, final_results)

    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Task {task_id} FAILED. Error: {e}\nTraceback:\n{error_details}")
        task_manager.set_failed(task_id, f"An unexpected error occurred in the optimization engine: {e}")


# ==============================================================================
#                      Uvicorn Server Startup
# ==============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Uvicorn server for the Advanced Logistics Optimization API...")
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=True)