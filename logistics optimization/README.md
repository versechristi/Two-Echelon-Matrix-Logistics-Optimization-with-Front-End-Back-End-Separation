# Advanced Logistics Optimization System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/) [![Framework](https://img.shields.io/badge/Framework-FastAPI-green.svg)](https://fastapi.tiangolo.com/) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Project Overview

This is a powerful logistics optimization application designed to solve the **Multi-Depot, Two-Echelon Vehicle Routing Problem with Drones and Split Deliveries (MD-2E-VRPSD)**. The system provides an end-to-end solution from problem definition, parameter configuration, algorithm execution, to multi-dimensional result visualization through a web-based Graphical User Interface (GUI).

The core application scenario is as follows: one or more **Depots** are responsible for replenishing multiple **Outlets** (first echelon), which then act as secondary hubs, utilizing both **trucks and drones** to complete the "last-mile" delivery to end customers (second echelon).

## System Interface and Core Features

The system offers a highly integrated and user-friendly graphical interface, combining complex parameter settings, algorithm execution, and result visualization. The left side serves as the control panel, while the right side is a multi-tab panel for displaying results.

### 1. Interactive Route Map

After optimization, the system generates a detailed interactive HTML map using `Folium`. Users can zoom, pan, and click on elements in a web browser to intuitively inspect every detail of the planned routes.

> **Legend:**
> * **Green Square**: Logistics Center (Depot)
> * **Blue Circle**: Sales Outlet
> * **Red Pin**: Customer
> * **Black/Dark Solid Line**: Stage 1 Route (Truck from Depot to Outlet)
> * **Blue Solid Line**: Stage 2 Route (Truck from Outlet to Customer)
> * **Purple Dashed Line**: Stage 2 Route (Drone Delivery)

<img width="1918" height="1005" alt="1817675baba6d6368a2651133f14f357" src="https://github.com/user-attachments/assets/04ba9ae4-b365-45fa-af83-fde4ed98a08b" />

---

### 2. Algorithm Performance Comparison

To facilitate a clear comparison of different algorithms, the system provides intuitive bar charts evaluating results based on key metrics like "Total Cost" and "Total Time" (Makespan). The chart below shows the final scores of various algorithms on the weighted objective function (a combination of transportation cost, time cost, and penalties). A lower cost indicates a higher quality solution.

<img width="1919" height="928" alt="a9bc6ae8d855f75e52d3bb4ed48bc20b" src="https://github.com/user-attachments/assets/95fe6f77-b058-4420-ad1b-1bd32d6ae7a0" />

---

### 3. Algorithm Convergence Curves

This chart illustrates the cost convergence of metaheuristic algorithms (like GA, SA, PSO) during the iteration process. By observing the curves, one can analyze the convergence speed and stability of the algorithms in finding the optimal solution.

<img width="1919" height="927" alt="67f1988e0ef794bec118d911987028fd" src="https://github.com/user-attachments/assets/6d189970-4d74-4973-9dde-4cc3345b18af" />

---

### 4. Detailed Reports and Run Logs

The system generates a structured text report for each algorithm's optimization result, including details on cost, time, feasibility, route specifics, and customer satisfaction. The "Run Log" tab provides real-time logs from the program execution, which is useful for debugging and tracking.

<img width="1918" height="1230" alt="869bf6ba8f00eec399802e9e7aae2749" src="https://github.com/user-attachments/assets/2de2bdbd-6d44-437b-ba5c-9888d5da5411" />

<img width="1918" height="1004" alt="0fefa173cac719df07978a272105ad50" src="https://github.com/user-attachments/assets/bef87f3f-413e-4a4c-9602-3a2ea13233cb" />


## Core Features

* **Advanced Problem Modeling**
    * **Two-Echelon Network**: Fully simulates a two-stage delivery network from a central warehouse to secondary hubs, and then to end customers.
    * **Multi-Depot Support**: Supports starting from multiple logistics centers to optimize more complex regional logistics networks.
    * **Drone Collaboration**: Allows vehicles and drones to work in parallel, with drones dispatched from sales outlets for short-range, small-parcel deliveries.
    * **Split Deliveries**: Enables a single customer's demand to be fulfilled by multiple deliveries (from different trucks or drones) to improve overall load factors and efficiency.

* **Powerful Algorithm Suite**
    The system includes several classic optimization algorithms for users to select, run, and compare:
    * **Genetic Algorithm (GA)**: A global search heuristic based on the principles of biological evolution.
    * **Simulated Annealing (SA)**: A probabilistic local search algorithm based on the physical annealing process.
    * **Particle Swarm Optimization (PSO)**: A population-based optimization algorithm inspired by the social behavior of bird flocking.
    * **Greedy Heuristic**: A fast, constructive heuristic algorithm used to generate high-quality baseline solutions.

* **Full-Fledged Graphical User Interface (GUI)**
    * **B/S Architecture**: The backend is powered by `FastAPI`, providing robust API services, while the frontend is built with `HTML/CSS/JS`, allowing users to access the application via a web browser without any installation.
    * **Asynchronous Task Handling**: Optimization tasks run in a separate background process, ensuring a smooth and responsive user interface, preventing freezes during long computations.
    * **Parametric Configuration**: Users can customize all key parameters through an intuitive accordion menu, including data generation, vehicle/drone attributes, objective function weights, and algorithm-specific hyperparameters.
    * **Configuration Persistence**: Supports saving all current parameter settings to an `.ini` file and loading configurations from a file, facilitating repeated experiments and sharing settings.
    * **Real-time Status Feedback**: A status bar at the bottom provides real-time updates on the current state of the program, such as "Generating data...", "Running optimization...", "Optimization complete".

* **In-depth Result Visualization and Analysis**
    * **Interactive Geographic Maps**: Uses `Folium` to generate interactive HTML maps that can be opened in a browser.
    * **Algorithm Performance Charts**: Uses `Matplotlib` to plot convergence curves and bar charts comparing the final results of different algorithms.
    * **Detailed Text Reports**: Generates structured text reports for each algorithm's optimization results.
    * **Result Export**: All generated charts and reports can be saved locally with a single click.

## Project Structure

The project uses a modular design with clear separation of concerns, making it easy to extend.

```
logistics_optimization/
│
├── api_server.py            # Backend API server (FastAPI), main entry point
├── requirements.txt         # List of dependencies
├── README.md                # Project documentation
│
├── config/                  # Configuration files directory (e.g., default_config.ini)
│
├── core/                    # Core business logic and computation modules
│   ├── cost_function.py     # Objective function calculation
│   ├── distance_calculator.py # Geographic distance (Haversine) utility
│   ├── problem_utils.py     # Core problem utilities (solution class, neighborhood ops)
│   └── route_optimizer.py   # Orchestrator for the optimization process
│
├── data/                    # Data generation and loading modules
│   ├── data_generator.py    # Synthetic data generator
│   └── solomon_parser.py    # Parser for Solomon's benchmark datasets
│
├── algorithm/               # Implementations of optimization algorithms
│   ├── genetic_algorithm.py
│   ├── simulated_annealing.py
│   ├── pso_optimizer.py
│   └── greedy_heuristic.py
│
├── fronted/                 # Frontend resources (HTML/CSS/JS)
│   ├── index.html
│   ├── css/style.css
│   └── js/script.js
│
├── visualization/           # Result visualization modules
│   ├── map_generator.py     # Folium interactive map generator
│   └── plot_generator.py    # Matplotlib chart generator
│
└── output/                  # Default output directory for run results
```


## Tech Stack

* **Backend**: Python, FastAPI, Uvicorn
* **Core Computation**: NumPy, Pandas
* **Visualization**: Matplotlib, Folium
* **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5

## Installation and Setup

1.  **Clone the Repository**
    ```bash
    git clone <your-repository-url>
    cd logistics_optimization
    ```

2.  **Create and Activate a Virtual Environment (Recommended)**
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # Linux/macOS
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    Install all required libraries from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application**
    Start the backend server by running `api_server.py` from the project root directory:
    ```bash
    python api_server.py
    ```

5.  **Access the Application**
    Once the server is running, you can access the application in one of two ways:

    * **Method 1 (Recommended)**: Open your web browser and navigate to:
        ```
        [http://127.0.0.1:8000](http://127.0.0.1:8000)
        ```

    * **Method 2**: Open the `fronted/index.html` file directly in your file explorer. This will also work as it sends requests to the locally running server at `http://127.0.0.1:8000`.

## User Guide

1.  **Configure Parameters**: Use the accordion menu in the "Control Panel" on the left to set parameters for data generation, vehicle/drone attributes, objective function weights, and algorithm hyperparameters.
2.  **Generate Data**: Click the **"Generate New Data"** button to create a new logistics scenario based on your settings. You can also click the **"Load Config"** folder icon to load all parameters from a previously saved `.ini` file.
3.  **Run Optimization**: In the "Algorithm Selection" area, check the box for one or more algorithms you wish to run and compare, then click the **"Run Optimization"** button.
4.  **Analyze Results**: Use the tabs on the right-side panel to view the results:
    * **Route Map**: Select different algorithm results from the dropdown to view their route plans on the map.
    * **Convergence**: View the cost convergence curves for the executed algorithms.
    * **Comparison**: View bar charts comparing the final results of the algorithms.
    * **Report**: Select a result from the dropdown to view its detailed text report.
    * **Run Log**: View the real-time logs from the application's execution.
    * All charts and reports can be saved locally using their respective **"Save"** buttons.

---
*This project was developed by Mr. VerseChristi. For questions or data inquiries, please contact versechristi@gmail.com*
