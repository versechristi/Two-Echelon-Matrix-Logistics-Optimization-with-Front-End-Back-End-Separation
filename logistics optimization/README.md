# 高级物流优化系统 (Advanced Logistics Optimization System)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/) [![Framework](https://img.shields.io/badge/Framework-FastAPI-green.svg)](https://fastapi.tiangolo.com/) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## 项目概述

这是一个功能强大的物流优化桌面应用，专为解决 **带无人机协同与订单拆分的多站点、二级车辆路径问题 (MD-2E-VRPSD)** 而设计。系统通过一个基于Web技术的图形用户界面（GUI），为用户提供了一个从问题定义、参数配置、算法执行到多维度结果可视化的端到端解决方案。

该系统的核心应用场景为：一个或多个**物流中心 (Depot)** 负责向多个**销售网点 (Outlet)** 进行补货（第一级运输），随后，销售网点作为二级枢纽，利用**货车和无人机**共同为终端客户完成“最后一公里”的配送（第二级运输）。

## 系统界面与核心功能演示

系统提供了高度集成且用户友好的图形界面，将复杂参数配置、算法执行与结果可视化融为一体。左侧为控制面板，右侧为多维度的结果展示面板。

### 1. 交互式地理路径图

优化完成后，系统利用 `Folium` 库生成详细的交互式HTML地图，用户可以在浏览器中进行缩放、平移、点击等操作，直观地审查路径规划的每一个细节。

> **图例说明**：
> * **绿色方块**: 一级物流中心 (Depot)
> * **蓝色圆形**: 二级销售网点 (Outlet)
> * **红色图钉**: 终端客户 (Customer)
> * **黑色/深色粗实线**: 第一级运输路径（货车从物流中心到销售网点）
> * **蓝色细实线**: 第二级车辆运输路径（货车从销售网点到客户）
> * **紫色虚线**: 第二级无人机配送路径

![交互式地理路径图](https://i.imgur.com/vHq0u6S.jpeg)

---

### 2. 算法性能对比图

为了方便横向比较不同算法的优劣，系统提供了直观的条形对比图，从“总成本”和“总时间”两个关键维度对结果进行评估。下图展示了各算法在加权目标函数（综合了运输成本、时间成本和惩罚项）上的最终得分。成本越低，代表解决方案质量越高。

![算法性能对比图](https://i.imgur.com/B9B1xAS.png)

---

### 3. 算法收敛曲线

此图表展示了元启发式算法（如遗传算法、模拟退火、粒子群）在迭代过程中的成本收敛情况。通过观察曲线，可以分析算法的收敛速度和寻找最优解的稳定性。

![算法收敛曲线](https://i.imgur.com/XU6d4J8.png)

---

### 4. 详细的文本报告与运行日志

系统为每个算法的优化结果生成了结构化的文本报告，内容包括成本、时间、可行性、各阶段路线和客户满意度等详细信息。同时，“Run Log”选项卡提供了程序运行时的实时日志，便于调试和追踪。

![详细文本报告](https://i.imgur.com/tC6F2f3.png)

![运行日志](https://i.imgur.com/T0bFwY7.png)


## 核心功能特点

* **先进的问题建模**
    * **二级网络 (Two-Echelon)**：完整模拟从中心仓库到二级网点，再到终端客户的两级配送网络。
    * **多仓库支持 (Multi-Depot)**：支持从多个一级物流中心出发，优化更复杂的区域物流网络。
    * **无人机协同 (Drone Collaboration)**：允许车辆与无人机并行工作，无人机从销售网点出发，执行短途、小批量包裹的配送任务。
    * **拆分配送 (Split Deliveries)**：允许一个客户的需求由多次配送（来自不同车辆或无人机）共同满足，以提高整体装载率和效率。

* **强大的算法套件**
    系统内置了多种经典的优化算法，供用户选择、运行和对比：
    * **遗传算法 (Genetic Algorithm, GA)**：基于生物进化原理的全局搜索启发式算法。
    * **模拟退火 (Simulated Annealing, SA)**：基于物理退火过程的概率性局部搜索算法。
    * **粒子群优化 (Particle Swarm Optimization, PSO)**：模拟鸟群觅食行为的群体智能优化算法。
    * **贪心启发式 (Greedy Heuristic)**：一种快速、直接的构造性启发式算法，可用于生成高质量的基准解。

* **全功能的图形用户界面 (GUI)**
    * **B/S 架构**：后端由 `FastAPI` 驱动，提供强大的 API 服务；前端由 `HTML/CSS/JS` 构建，用户通过浏览器即可访问，无需安装桌面应用。
    * **异步任务处理**：优化计算在独立的后台任务中执行，确保了用户界面的流畅和响应性，避免了长时间计算导致的界面卡死。
    * **参数化配置**：通过直观的折叠面板，用户可以自定义所有关键参数，包括数据生成、载具属性、目标函数权重和各算法的超参数。
    * **配置持久化**：支持将当前的所有参数配置保存到 `.ini` 文件中，也支持从文件中加载配置，方便重复实验和分享设置。
    * **实时状态反馈**：通过底部的状态栏和进度条，实时显示系统当前状态，如“数据生成中...”、“正在运行优化...”、“优化完成”等。

* **深度结果可视化与分析**
    * **交互式地理地图**：利用 `Folium` 库生成可在浏览器中打开的交互式HTML地图，直观展示路径规划细节。
    * **算法性能图表**：使用 `Matplotlib` 绘制算法在迭代过程中的目标函数值（成本）收敛曲线，以及多算法在关键指标（总成本、总时间）上的最终表现对比图。
    * **详细文本报告**：为每个算法的优化结果生成结构化的文本报告，包含成本、时间、可行性、各阶段路线等详细信息。
    * **结果导出**：所有生成的图表和报告均可一键保存到本地。

## 项目结构

项目采用模块化的结构设计，职责清晰，易于扩展。

logistics_optimization/
│
├── api_server.py            # 后端API服务器 (FastAPI)，项目主入口
├── requirements.txt         # 项目依赖库列表
├── README.md                # 项目说明文件
│
├── config/                  # 配置文件目录 (e.g., default_config.ini)
│
├── core/                    # 核心业务逻辑与计算模块
│   ├── cost_function.py     # 目标函数（成本、时间、惩罚项）计算
│   ├── distance_calculator.py # 地理距离（Haversine）计算工具
│   ├── problem_utils.py     # 问题核心工具（解决方案定义、邻域操作等）
│   └── route_optimizer.py   # 优化流程编排器，调用算法并整合结果
│
├── data/                    # 数据生成与加载模块
│   ├── data_generator.py    # 合成数据生成器
│   └── solomon_parser.py    # Solomon 标准数据集解析器
│
├── algorithm/               # 优化算法实现
│   ├── genetic_algorithm.py
│   ├── simulated_annealing.py
│   ├── pso_optimizer.py
│   └── greedy_heuristic.py
│
├── fronted/                 # 前端资源目录 (HTML/CSS/JS)
│   ├── index.html
│   ├── css/style.css
│   └── js/script.js
│
├── visualization/           # 结果可视化模块
│   ├── map_generator.py     # Folium 交互式地图生成
│   └── plot_generator.py    # Matplotlib 图表（迭代、对比）生成
│
└── output/                  # 默认输出目录，存放运行结果


## 技术栈

* **后端**: Python, FastAPI, Uvicorn
* **核心计算**: NumPy, Pandas
* **可视化**: Matplotlib, Folium
* **前端**: HTML5, CSS3, JavaScript, Bootstrap 5

## 安装与运行

1.  **克隆仓库**
    ```bash
    git clone <your-repository-url>
    cd logistics_optimization
    ```

2.  **创建并激活虚拟环境 (推荐)**
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # Linux/macOS
    python -m venv venv
    source venv/bin/activate
    ```

3.  **安装依赖**
    根据 `requirements.txt` 文件，安装所有必需的库：
    ```bash
    pip install -r requirements.txt
    ```

4.  **启动程序**
    在项目根目录下运行 `api_server.py` 即可启动后端服务器：
    ```bash
    python api_server.py
    ```

5.  **访问应用**
    服务器启动后，您可以通过以下两种方式访问应用界面：

    * **方式一 (推荐)**：直接在浏览器中打开以下地址：
        ```
        [http://127.0.0.1:8000](http://127.0.0.1:8000)
        ```

    * **方式二**：在文件浏览器中，找到并双击打开 `fronted/index.html` 文件。这种方式同样可以运行，因为它会向已经启动的本地服务器 `http://127.0.0.1:8000` 发送请求。

## 使用指南

1.  **配置参数**: 在左侧的“Control Panel”中，通过折叠面板设置数据生成参数、车辆/无人机属性、目标函数权重以及各算法的超参数。
2.  **生成数据**: 点击 **"Generate New Data"** 按钮，系统将根据您的配置生成一个物流场景。您也可以点击文件夹图标 **"Load Config"** 从 `.ini` 文件加载之前保存的配置。
3.  **运行优化**: 在“Algorithm Selection”区域勾选一个或多个需要运行和对比的算法，然后点击 **"Run Optimization"** 按钮。
4.  **分析结果**: 在右侧的结果面板中：
    * **Route Map**: 通过下拉菜单选择不同算法的路径规划结果，并在地图上查看。
    * **Convergence**: 查看各算法在优化过程中的成本收敛曲线图。
    * **Comparison**: 查看各算法最终结果的条形对比图。
    * **Report**: 查看对应算法的详细文本报告。
    * **Run Log**: 查看程序运行时的实时日志。
    * 所有图表和报告都可以通过相应的 **"Save"** 按钮进行保存。

---
*本项目由Mr VerseChristi编写完成，若有疑问或需要提供数据可以联系 versechristi@gmail.com*