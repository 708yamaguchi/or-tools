#!/usr/bin/env python3
# Copyright 2010-2025 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sat based solver for the RCPSP problems (see rcpsp.proto).

Introduction to the problem:
   https://www.projectmanagement.ugent.be/research/project_scheduling/rcpsp

Data use in flags:
  http://www.om-db.wi.tum.de/psplib/data.html
"""

import collections
import io
import tempfile
import json
from itertools import combinations

from absl import app
from absl import flags

from google.protobuf import text_format
from ortools.sat.python import cp_model
from ortools.scheduling import rcpsp_pb2
from ortools.scheduling.python import rcpsp

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm
import numpy as np

# _INPUT = flags.DEFINE_string("input", "", "Input file to parse and solve.")
_OUTPUT_PROTO = flags.DEFINE_string(
    "output_proto", "", "Output file to write the cp_model proto to."
)
_PARAMS = flags.DEFINE_string("params", "", "Sat solver parameters.")
_USE_INTERVAL_MAKESPAN = flags.DEFINE_bool(
    "use_interval_makespan",
    True,
    "Whether we encode the makespan using an interval or not.",
)
_HORIZON = flags.DEFINE_integer("horizon", -1, "Force horizon.")


# --- Helper Functions for Task ID Calculation ---
# int -> Tuple[int, int, int]
def get_task_ids(task_index):
    """
    Calculates the IDs for placement, work, and retrieval activities for a given task index.
    The task_index is 1-based.
    """
    placement_id = 3 * task_index - 2
    work_id = 3 * task_index - 1
    retrieval_id = 3 * task_index
    return placement_id, work_id, retrieval_id


def generate_rcpsp_max_from_json(input_data, task_combinations, debug_print=False):
    """
    JSON形式の入力データからRCPSP/max形式の文字列を生成します。
    また、ソルバーの解を可視化するために(タスク名, モード番号) -> [リソース名]の
    マッピング辞書も同時に生成して返します。

    --- フォーマット仕様 (場所リソース対応版) ---

    ■ ヘッダー行
    <タスク数(3*N)> <Renewable Resources数> <Reservoir Resources数>

    ■ Renewable Resourcesの定義
    ・数: [ロボット数] + [モジュール数] + [場所数]
    ・意味:
      - 0 ~ (ロボット数-1)番目:
          各種ロボット。タスク実行中のみ専有される。
      - (ロボット数) ~ (ロボット数+モジュール数-1)番目:
          各種モジュール。タスク実行中のみ専有される。(Reservoirとは別枠)
      - (ロボット数+モジュール数) ~ :
          各種場所。その場所でおこなわれるタスク(Placement, Work, Retrieval)の
          実行中に専有される。Capacityは場所ごとに設定可能。

    ■ Reservoir Resourcesの定義
    ・数: [モジュール数]
    ・意味: 各種モジュール。
      - 配置(Placement)タスク開始時に、対応するモジュールリソースを「1消費」する。
      - 回収(Retrieval)タスク開始時に、対応するモジュールリソースを「-1消費」(つまり補充)する。
      - これにより、一度配置されたモジュールは、回収されるまで他のタスクで利用できない。
    """
    # 1. データの解析とパラメータ設定
    tasks = input_data.get('tasks', [])
    N = len(tasks)

    actual_robots = input_data['resources']['robot']
    num_actual_robots = len(actual_robots)
    robot_map = {res['name']: i for i, res in enumerate(actual_robots)}

    actual_modules = input_data['resources']['module']
    num_actual_modules = len(actual_modules)
    module_map = {res['name']: i for i, res in enumerate(actual_modules)}

    # 場所情報の読み込み
    locations = input_data.get('locations', [])
    num_locations = len(locations)
    location_map = {loc['name']: i for i, loc in enumerate(locations)}

    # リソース数の定義を変更
    num_renewable = num_actual_robots + num_actual_modules + num_locations
    num_reservoir = num_actual_modules
    total_resources = num_renewable + num_reservoir

    mode_to_resources_map = {}

    # デバッグ表示用に、リソースIDから名前へのマッピングを作成
    if debug_print:
        id_to_resource_name = {}
        # Renewable Resources
        for name, idx in robot_map.items():
            id_to_resource_name[idx] = f"Robot:{name}"
        for name, idx in module_map.items():
            id_to_resource_name[num_actual_robots + idx] = f"Module(Renewable):{name}"
        for name, idx in location_map.items():
            id_to_resource_name[num_actual_robots + num_actual_modules + idx] = f"Location:{name}"
        # Reservoir Resources
        for name, idx in module_map.items():
            id_to_resource_name[num_renewable + idx] = f"Module(Reservoir):{name}"

    # 2. アクティビティ情報の構築
    activities = {}
    start_successors = [id for n in range(1, N + 1) for id in (3 * n - 2, 3 * n - 1)]
    activities[0] = {
        'cost': 0, 'modes': 1, 'successors': sorted(start_successors),
        'demands': {1: [0] * total_resources}
    }

    for n in range(1, N + 1):
        task_index = n - 1
        task = tasks[task_index]
        task_name = task['name']
        task_duration = task['duration']
        combinations_for_task = task_combinations[task_name]

        # 場所リソースのインデックスを特定
        task_location = task.get('location')
        location_resource_idx = -1
        if task_location and task_location in location_map:
            location_resource_idx = num_actual_robots + num_actual_modules + location_map[task_location]


        placement_id, work_id, retrieval_id = 3*n-2, 3*n-1, 3*n
        final_activity_id = 3 * N + 1

        demands_placement, demands_work, demands_retrieval = {}, {}, {}
        placement_costs_by_mode, retrieval_costs_by_mode = {}, {}

        num_work_modes = len(combinations_for_task) if combinations_for_task else 1
        num_placement_retrieval_modes = num_work_modes * num_actual_robots if combinations_for_task and num_actual_robots > 0 else 1

        if not combinations_for_task:
            # モードがない場合のデフォルト処理
            demands_placement[1] = [0] * total_resources
            demands_work[1] = [0] * total_resources
            demands_retrieval[1] = [0] * total_resources
            placement_costs_by_mode[1] = 0
            retrieval_costs_by_mode[1] = 0
        else:
            # --- 作業(Work)モードのデマンド ---
            for i, combo in enumerate(combinations_for_task):
                mode_num = i + 1
                mode_to_resources_map[(task_name, i)] = sorted(combo)
                demands_w_mode = [0] * total_resources
                robot_is_used = False
                # ロボット/モジュールをRenewableとして専有
                for res_name in combo:
                    if res_name in robot_map:
                        demands_w_mode[robot_map[res_name]] = 1
                        robot_is_used = True
                    elif res_name in module_map:
                        demands_w_mode[num_actual_robots + module_map[res_name]] = 1
                # ロボットが使われる場合のみ、場所リソースを専有する
                if location_resource_idx != -1 and robot_is_used:
                    demands_w_mode[location_resource_idx] = 1
                demands_work[mode_num] = demands_w_mode

            # --- 配置(Placement)・回収(Retrieval)モードのデマンド ---
            mode_num_pr = 0
            for i, combo in enumerate(combinations_for_task):
                cost = 5 if any(res_name in module_map for res_name in combo) else 0
                for j in range(num_actual_robots):
                    mode_num_pr += 1
                    placement_costs_by_mode[mode_num_pr] = cost
                    retrieval_costs_by_mode[mode_num_pr] = cost

                    carrier_robot_name = actual_robots[j]['name']
                    recipe_idx_pr = mode_num_pr - 1
                    mode_to_resources_map[(f"Placement-{task_name}", recipe_idx_pr)] = {'carrier': carrier_robot_name, 'payload': sorted(combo)}
                    mode_to_resources_map[(f"Retrieval-{task_name}", recipe_idx_pr)] = {'carrier': carrier_robot_name, 'payload': sorted(combo)}
                    def _create_placement_retrieval_demand(
                        carrier_robot_idx: int,
                        payload_combo: list[str],
                        demand_sign: int,
                        robot_map: dict,
                        module_map: dict,
                        location_resource_idx: int,
                        num_actual_robots: int,
                        num_renewable: int,
                        total_resources: int
                    ) -> list[int]:
                        demands_mode = [0] * total_resources
                        # 1. 運搬ロボット(Renewable)を専有
                        demands_mode[carrier_robot_idx] = 1
                        # 2. ペイロード内のリソースを専有
                        for res_name in payload_combo:
                            if res_name in robot_map:
                                demands_mode[robot_map[res_name]] = 1
                            elif res_name in module_map:
                                module_idx = module_map[res_name]
                                # Renewableスロットとして専有
                                demands_mode[num_actual_robots + module_idx] = 1
                                # Reservoirとして消費または補充
                                reservoir_idx = num_renewable + module_idx
                                demands_mode[reservoir_idx] = demand_sign
                        # 3. 場所リソースを専有
                        #    (ロボットが関わるタスクであるため、常に専有する)
                        if location_resource_idx != -1:
                            demands_mode[location_resource_idx] = 1
                        return demands_mode

                    # --- 配置(Placement)デマンド ---
                    demands_placement[mode_num_pr] = _create_placement_retrieval_demand(
                        carrier_robot_idx=j,
                        payload_combo=combo,
                        demand_sign=1, # +1で消費
                        robot_map=robot_map,
                        module_map=module_map,
                        location_resource_idx=location_resource_idx,
                        num_actual_robots=num_actual_robots,
                        num_renewable=num_renewable,
                        total_resources=total_resources
                    )

                    # --- 回収(Retrieval)デマンド ---
                    demands_retrieval[mode_num_pr] = _create_placement_retrieval_demand(
                        carrier_robot_idx=j,
                        payload_combo=combo,
                        demand_sign=-1, # -1で補充
                        robot_map=robot_map,
                        module_map=module_map,
                        location_resource_idx=location_resource_idx,
                        num_actual_robots=num_actual_robots,
                        num_renewable=num_renewable,
                        total_resources=total_resources
                    )

        activities[placement_id] = {'modes': num_placement_retrieval_modes, 'successors': [work_id], 'demands': demands_placement, 'costs_by_mode': placement_costs_by_mode}
        activities[work_id] = {'cost': task_duration, 'modes': num_work_modes, 'successors': [retrieval_id], 'demands': demands_work}
        activities[retrieval_id] = {'modes': num_placement_retrieval_modes, 'successors': [final_activity_id], 'demands': demands_retrieval, 'costs_by_mode': retrieval_costs_by_mode}

        if debug_print:
            print(f"\n" + "="*15 + f" DEBUG: Task {n} ({task_name}) " + "="*15)
            print(f"  Location: {task_location}")
            print(f"  Activity IDs: Placement={placement_id}, Work={work_id}, Retrieval={retrieval_id}")

            def get_demands_str(demands_list):
                consumed = []
                for res_id, demand_val in enumerate(demands_list):
                    if demand_val != 0:
                        res_name = id_to_resource_name.get(res_id, f"ID_{res_id}")
                        consumed.append(f"'{res_name}': {demand_val}")
                return ", ".join(consumed) if consumed else "None"

            # --- Placement Activity ---
            print(f"\n  -> Activity: Placement ({num_placement_retrieval_modes} modes)")
            modes_to_show = list(demands_placement.keys())            
            for mode_idx in modes_to_show:
                if mode_idx == '...':
                    print("     ...")
                    continue
                recipe_idx = mode_idx - 1
                combo = mode_to_resources_map.get((f"Placement-{task_name}", recipe_idx), {})
                print(f"     - Mode {mode_idx}:")
                print(f"       Combination: Carrier='{combo.get('carrier')}', Payload={combo.get('payload')}")
                print(f"       Resource Demands: {get_demands_str(demands_placement[mode_idx])}")

            # --- Work Activity ---
            print(f"\n  -> Activity: Work ({num_work_modes} modes)")
            for mode_idx, demands in demands_work.items():
                recipe_idx = mode_idx - 1
                combo = mode_to_resources_map.get((task_name, recipe_idx), [])
                print(f"     - Mode {mode_idx}:")
                print(f"       Combination: {combo}")
                print(f"       Resource Demands: {get_demands_str(demands)}")

            # --- Retrieval Activity ---
            print(f"\n  -> Activity: Retrieval ({num_placement_retrieval_modes} modes)")
            modes_to_show = list(demands_retrieval.keys())
            for mode_idx in modes_to_show:
                if mode_idx == '...':
                    print("     ...")
                    continue
                recipe_idx = mode_idx - 1
                combo = mode_to_resources_map.get((f"Retrieval-{task_name}", recipe_idx), {})
                print(f"     - Mode {mode_idx}:")
                print(f"       Combination: Carrier='{combo.get('carrier')}', Payload={combo.get('payload')}")
                print(f"       Resource Demands: {get_demands_str(demands_retrieval[mode_idx])}")
            print("="*58)

    activities[3*N+1] = {'cost': 0, 'modes': 1, 'successors': [], 'demands': {1: [0] * total_resources}}

    # 3. RCPSP/max 形式の文字列を生成
    output_lines = []
    output_lines.append(f"{3 * N} {num_renewable} {num_reservoir} 0") # ヘッダーを更新

    # 先行関係ブロック (変更なし)
    for i in sorted(activities.keys()):
        act = activities[i]
        num_succ = len(act['successors'])
        line_parts = [str(i), str(act['modes']), str(num_succ)]
        if num_succ > 0:
            line_parts.append(' '.join(map(str, act['successors'])))
            delay_str_parts = []
            for succ_id in act['successors']:
                if succ_id not in activities: continue
                succ_act = activities[succ_id]
                delays = []
                if 'cost' in act:
                    delays = [str(act['cost'])] * (act['modes'] * succ_act['modes'])
                elif 'costs_by_mode' in act:
                    for mode_num in sorted(act['costs_by_mode'].keys()):
                        cost = act['costs_by_mode'][mode_num]
                        delays.extend([str(cost)] * succ_act['modes'])
                delay_str_parts.append(f"[{' '.join(delays)}]")
            line_parts.append(' '.join(delay_str_parts))
        output_lines.append(' '.join(line_parts))

    # リソース消費ブロック (変更なし)
    for i in sorted(activities.keys()):
        act = activities[i]
        for mode_num, demands in sorted(act['demands'].items()):
            demands_str = ' '.join(map(str, demands))
            cost_for_mode = act.get('cost', 0) or act.get('costs_by_mode', {}).get(mode_num, 0)
            if mode_num == 1:
                output_lines.append(f"{i} {mode_num} {cost_for_mode} {demands_str}")
            else:
                output_lines.append(f" {mode_num} {cost_for_mode} {demands_str}")

    # リソース容量定義ブロック
    robot_caps = [res['quantity'] for res in actual_robots]
    module_renewable_caps = [res['quantity'] for res in actual_modules]
    location_caps = [loc['max_robots'] for loc in locations]
    renewable_caps = robot_caps + module_renewable_caps + location_caps
    reservoir_caps = module_renewable_caps

    output_lines.append(' '.join(map(str, renewable_caps + reservoir_caps)))

    return "\n".join(output_lines), mode_to_resources_map


def create_name_mappings(input_data: dict) -> (dict, dict):
    """
    Generates dictionaries to map task IDs and mode numbers to human-readable names.
    """
    tasks = input_data.get('tasks', [])
    N = len(tasks)
    task_id_to_name = {}

    task_id_to_name[0] = "Start"
    task_id_to_name[3 * N + 1] = "Finish"

    for n in range(1, N + 1):
        task_name = tasks[n - 1]['name']
        placement_id, work_id, retrieval_id = get_task_ids(n)

        # Unify terminology from Pre/Post to Placement/Retrieval
        task_id_to_name[placement_id] = f"Placement-{task_name}"
        task_id_to_name[work_id] = f"{task_name}"
        task_id_to_name[retrieval_id] = f"Retrieval-{task_name}"

    mode_to_name = {}
    for i in range(100):
        mode_to_name[i] = f"Mode {i}"

    return task_id_to_name, mode_to_name


def create_resource_name_mappings(input_data):
    """
    入力データと定義に基づき、リソースIDをリソース名にマッピングする辞書を生成します。
    (場所リソースを考慮し、モジュールのRenewableリソースのバグを修正)
    """
    resources = input_data.get('resources', {})
    locations = input_data.get('locations', [])

    renewable_resources = resources.get('robot', []) # Robots
    num_actual_robots = len(renewable_resources)
    reservoir_resources = resources.get('module', []) # Modules
    num_actual_modules = len(reservoir_resources)

    # 1. Renewable Resources のマッピング
    renewable_id_to_name: dict[int, str] = {}
    # Robots (ID: 0 ~ num_actual_robots-1)
    for i, res in enumerate(renewable_resources):
        renewable_id_to_name[i] = res.get('name', f"Robot_{i+1}")
    # Modules (as Renewable) (ID: num_actual_robots ~ num_actual_robots+num_actual_modules-1)
    for i, res in enumerate(reservoir_resources):
        resource_id = num_actual_robots + i
        renewable_id_to_name[resource_id] = f"{res.get('name')} (Renewable Slot)"
    # Locations (ID: num_actual_robots+num_actual_modules ~ )
    for i, loc in enumerate(locations):
        resource_id = num_actual_robots + num_actual_modules + i
        renewable_id_to_name[resource_id] = f"Location: {loc.get('name')}"


    # 2. Reservoir Resources のマッピング (モジュール)
    reservoir_id_to_name: dict[int, str] = {}
    for i, res in enumerate(reservoir_resources):
        reservoir_id_to_name[i] = res.get('name', f"Module_{i+1}")

    return renewable_id_to_name, reservoir_id_to_name


def calculate_optional_tasks(input_data):  # Dict -> Set[int]
    """Determines the set of optional task IDs (placement and retrieval)."""
    tasks = input_data.get('tasks', [])
    optional_tasks = set()
    for n in range(1, len(tasks) + 1):
        placement_id, _, retrieval_id = get_task_ids(n)
        optional_tasks.add(placement_id)
        optional_tasks.add(retrieval_id)
    return optional_tasks


def print_problem_statistics(problem: rcpsp_pb2.RcpspProblem):
    """Display various statistics on the problem."""
    problem_type = (
        "Resource Investment Problem" if problem.is_resource_investment else "RCPSP"
    )
    num_resources = len(problem.resources)
    num_tasks = len(problem.tasks) - 2
    tasks_with_alternatives = 0
    variable_duration_tasks = 0
    tasks_with_delay = 0
    for task in problem.tasks:
        if len(task.recipes) > 1:
            tasks_with_alternatives += 1
            duration_0 = task.recipes[0].duration
            for recipe in task.recipes:
                if recipe.duration != duration_0:
                    variable_duration_tasks += 1
                    break
        if task.successor_delays:
            tasks_with_delay += 1
    if problem.is_rcpsp_max:
        problem_type += "/Max delay"
    if problem.is_consumer_producer:
        print(f"Solving {problem_type} with:")
        print(f"  - {num_resources} reservoir resources")
        print(f"  - {num_tasks} tasks")
    else:
        print(f"Solving {problem_type} with:")
        print(f"  - {num_resources} renewable resources")
        print(f"  - {num_tasks} tasks")
        if tasks_with_alternatives:
            print(f"    - {tasks_with_alternatives} tasks with alternative resources")
        if variable_duration_tasks:
            print(f"    - {variable_duration_tasks} tasks with variable durations")
        if tasks_with_delay:
            print(f"    - {tasks_with_delay} tasks with successor delays")


def print_schedule_by_task(
    solver: cp_model.CpSolver,
    all_active_tasks: list[int],
    executed_tasks: list[int],
    source: int,
    sink: int,
    task_starts: dict,
    task_durations: dict,
    task_ends: dict,
    selected_recipes: dict,
    task_id_to_name: dict,
    mode_to_name: dict,
    mode_to_resources_map: dict,
) -> None:
    """
    Prints the schedule details for each task.
    """
    print("Solution Found:")
    print(f"Optimal Makespan: {solver.objective_value}")
    print("--------------------------------------------------")
    print("--- Schedule by Task (Skipped tasks included) ---")
    name_width = max(len(name) for name in task_id_to_name.values()) + 2
    source_name = task_id_to_name.get(source, f"Task {source}")
    source_start_val = solver.value(task_starts[source])
    print(
        f"{source_name:<{name_width}} "
        f"Start={source_start_val:<3} "
        f"Duration=0   "
        f"End={source_start_val:<3}  (Project Start)"
    )
    for t in sorted(all_active_tasks):
        task_name = task_id_to_name.get(t, f"Task {t}")
        if t in executed_tasks:
            start_val = solver.value(task_starts[t])
            duration_val = solver.value(task_durations[t])
            end_val = solver.value(task_ends[t])

            mode_str = "N/A"
            recipe_index = selected_recipes.get(t)
            if recipe_index is not None:
                resources_used_data = mode_to_resources_map.get((task_name, recipe_index))
                if isinstance(resources_used_data, dict): # Placement/Retrieval
                    res_list = [resources_used_data['carrier']] + resources_used_data['payload']
                    mode_str = f"Res: {res_list}"
                elif resources_used_data: # Work
                    mode_str = f"Res: {resources_used_data}"
                else:
                    mode_str = f"Mode {recipe_index + 1}"

            print(
                f"{task_name:<{name_width}} "
                f"({mode_str}): "
                f"Start={start_val:<3} "
                f"Duration={duration_val:<3} "
                f"End={end_val:<3} "
            )
        else:
            print(f"{task_name:<{name_width}}: --- SKIPPED ---")
    sink_name = task_id_to_name.get(sink, f"Task {sink}")
    sink_start_val = solver.value(task_starts[sink])
    print(
        f"{sink_name:<{name_width}} "
        f"Start={sink_start_val:<3} "
        f"Duration=0   "
        f"End={sink_start_val:<3}  (Project End / Makespan)"
    )


def print_schedule_by_time_step(
    solver: cp_model.CpSolver,
    problem: rcpsp_pb2.RcpspProblem,
    executed_tasks: list[int],
    task_starts: dict,
    task_ends: dict,
    task_to_resource_demands: dict,
    all_resources: range,
    selected_recipes: dict,
    task_id_to_name: dict,
    mode_to_name: dict,
    mode_to_resources_map: dict,
):
    """
    Prints running tasks and resource status for each time step.
    """
    print("\n--- Schedule by Time Step ---")
    makespan = int(solver.objective_value)
    for t in range(makespan + 1):
        running_tasks_info = []
        running_tasks_ids = []
        for task_id in executed_tasks:
            start_time = solver.value(task_starts[task_id])
            end_time = solver.value(task_ends[task_id])
            if start_time <= t < end_time:
                running_tasks_ids.append(task_id)
                task_name = task_id_to_name.get(task_id, f"Task {task_id}")
                mode_str = "N/A"
                recipe_index = selected_recipes.get(task_id)
                if recipe_index is not None:
                    resources_used_data = mode_to_resources_map.get((task_name, recipe_index))
                    if isinstance(resources_used_data, dict):
                         res_list = [resources_used_data['carrier']] + resources_used_data['payload']
                         mode_str = f"Res: {res_list}"
                    elif resources_used_data:
                        mode_str = f"Res: {resources_used_data}"
                    else:
                        mode_str = f"Mode {recipe_index + 1}"
                running_tasks_info.append(f"{task_name}({mode_str})")
        print(f"[Time: {t}]")
        if not running_tasks_info:
            print("  Running Tasks: None")
        else:
            print(f"  Running Tasks: {running_tasks_info}")

        print("  Resource Status:")
        for res_id in all_resources:
            resource = problem.resources[res_id]
            total_capacity = resource.max_capacity
            if total_capacity == -1: continue
            if resource.renewable:
                used_capacity = sum(solver.value(task_to_resource_demands[task_id][res_id]) for task_id in running_tasks_ids if task_id in task_to_resource_demands and len(task_to_resource_demands[task_id]) > res_id)
                remaining_capacity = total_capacity - used_capacity
                print(f"    - (Renewable)   Resource {res_id}: Remaining={remaining_capacity}/{total_capacity} (Used={used_capacity})")
            else:
                consumed_so_far = sum(solver.value(task_to_resource_demands[task_id][res_id]) for task_id in executed_tasks if solver.value(task_starts[task_id]) <= t and task_id in task_to_resource_demands and len(task_to_resource_demands[task_id]) > res_id)
                remaining = total_capacity - consumed_so_far
                print(f"    - (Reservoir)   Resource {res_id}: Remaining={remaining}/{total_capacity} (Consumed={consumed_so_far})")


# --- Visualization Functions ---
def _get_base_task_name(task_name: str) -> str:
    """ Extracts 'kitchen' from 'Placement-kitchen' or 'Retrieval-kitchen' """
    if task_name.startswith("Placement-") or task_name.startswith("Retrieval-"):
        return "-".join(task_name.split("-")[1:])
    return task_name


def _draw_custom_legends(fig, capability_color_map, resource_color_map, input_data, show_symbols=True):
    """
    ### 変更点 ###
    - Resourcesの凡例を「Robots (Renewable)」と「Modules (Reservoir)」に分割
    - それぞれに小見出しを追加して視覚的に分かりやすくした
    """
    # --- Capabilities Legend ---
    fig.text(0.83, 0.90, "Capabilities", fontsize=12, fontweight='bold')
    y_pos = 0.88
    for cap, color in capability_color_map.items():
        ellipse = patches.Ellipse(xy=(0.835, y_pos), width=0.012, height=0.012,
                                  facecolor=color, edgecolor='black',
                                  transform=fig.transFigure, figure=fig)
        fig.patches.append(ellipse)
        fig.text(0.85, y_pos, cap, fontsize=10, va='center')
        y_pos -= 0.03

    # --- Resources Legend ---
    fig.text(0.83, y_pos - 0.02, "Resources", fontsize=12, fontweight='bold')
    y_pos -= 0.05

    # 1. Robots (Renewable) のセクション
    fig.text(0.83, y_pos, "Robots", fontsize=10, fontweight='bold', style='italic', color='dimgray')
    y_pos -= 0.035
    for res in input_data["resources"]["robot"]:
        res_name = res["name"]
        res_color = resource_color_map.get(res_name, "grey")
        capacity = res["quantity"]

        fig.patches.extend([plt.Rectangle((0.83, y_pos - 0.015), 0.01, 0.02,
                                          facecolor=res_color, edgecolor='black',
                                          transform=fig.transFigure, figure=fig)])
        fig.text(0.85, y_pos, res_name, fontsize=10, va='center')
        fig.text(0.85, y_pos - 0.015, f"(Cap: {capacity})", fontsize=8, color='dimgray', va='center')

        x_pos_cap = 0.92
        for cap in res["capabilities"]:
            cap_color = capability_color_map.get(cap, "grey")
            ellipse = patches.Ellipse((x_pos_cap, y_pos), width=0.01, height=0.01,
                                    facecolor=cap_color, edgecolor="black", linewidth=0.5,
                                    transform=fig.transFigure, figure=fig)
            fig.patches.append(ellipse)
            x_pos_cap += 0.012
        y_pos -= 0.045

    # セクション間のスペースを確保
    y_pos -= 0.02

    # 2. Modules (Reservoir) のセクション
    fig.text(0.83, y_pos, "Modules", fontsize=10, fontweight='bold', style='italic', color='dimgray')
    y_pos -= 0.035
    for res in input_data["resources"]["module"]:
        res_name = res["name"]
        res_color = resource_color_map.get(res_name, "grey")
        capacity = res["quantity"]

        fig.patches.extend([plt.Rectangle((0.83, y_pos - 0.015), 0.01, 0.02,
                                          facecolor=res_color, edgecolor='black',
                                          transform=fig.transFigure, figure=fig)])
        fig.text(0.85, y_pos, res_name, fontsize=10, va='center')
        fig.text(0.85, y_pos - 0.015, f"(Cap: {capacity})", fontsize=8, color='dimgray', va='center')

        x_pos_cap = 0.92
        for cap in res["capabilities"]:
            cap_color = capability_color_map.get(cap, "grey")
            ellipse = patches.Ellipse((x_pos_cap, y_pos), width=0.01, height=0.01,
                                    facecolor=cap_color, edgecolor="black", linewidth=0.5,
                                    transform=fig.transFigure, figure=fig)
            fig.patches.append(ellipse)
            x_pos_cap += 0.012
        y_pos -= 0.045

    # --- Symbols Legend ---
    if show_symbols:
        y_pos -= 0.01
        fig.text(0.83, y_pos, "Symbols", fontsize=12, fontweight='bold')
        y_pos -= 0.035
        fig.patches.extend([plt.Rectangle((0.83, y_pos - 0.0075), 0.01, 0.015,
                                          facecolor='lightgrey', edgecolor='black', hatch='//',
                                          transform=fig.transFigure, figure=fig)])
        fig.text(0.85, y_pos, "Robot-led Placement / Retrieval", fontsize=9, va='center')


def _plot_gantt_chart(
    ax, solver, all_task_ids,
    executed_tasks, task_starts, task_durations,
    selected_recipes, task_id_to_name,
    task_name_to_required_caps,
    mode_to_resources_map,
    capability_color_map,
    resource_color_map,
    input_data,
    task_id_to_location
):
    """視覚的に改善されたGanttチャートをmatplotlibのAxesオブジェクトにプロットします。"""
    y_labels = [task_id_to_name.get(t, f"Task {t}") for t in all_task_ids]

    # --- 描画のための準備 ---
    robot_names = {r['name'] for r in input_data["resources"]["robot"]}
    module_names = {r['name'] for r in input_data["resources"]["module"]}

    # --- 階層構造を持つY軸ラベルを生成 ---
    new_y_labels = []
    for i, t_id in enumerate(all_task_ids):
        full_name = y_labels[i]
        base_name = _get_base_task_name(full_name)
        if full_name.startswith("Placement-"):
            new_y_labels.append("  Placement")
        elif full_name.startswith("Retrieval-"):
            new_y_labels.append("  Retrieval")
            if i + 1 < len(all_task_ids):
                 ax.axhline(y=i + 0.5, color='gray', linestyle=':', linewidth=1)
        else:
            new_y_labels.append(base_name)

    ax.set_yticks(range(len(new_y_labels)))
    ax.set_yticklabels(new_y_labels)


    # ★ 場所の区切り線とラベルを描画 ★
    location_info = {loc['name']: loc for loc in input_data.get('locations', [])}
    location_ranges = {}
    current_loc_name = None

    # 場所ごとのタスクの開始・終了インデックスを記録
    for i, t_id in enumerate(all_task_ids):
        loc = task_id_to_location.get(t_id)
        if loc != current_loc_name:
            if current_loc_name is not None:
                location_ranges[current_loc_name]['end'] = i - 1
            if loc is not None:
                location_ranges[loc] = {'start': i, 'end': -1}
            current_loc_name = loc
    if current_loc_name is not None:
        location_ranges[current_loc_name]['end'] = len(all_task_ids) - 1

    # 区切り線とラベルを描画
        for loc, y_range in location_ranges.items():
            # 区切り線
            if y_range['start'] > 0:
                ax.axhline(y=y_range['start'] - 0.5, color='black', linestyle='-', linewidth=1.2)
            # ラベル
            info = location_info.get(loc)
            if info:
                # ラベルのY座標を領域の下端に設定
                y_pos_bottom = y_range['end'] + 0.35
                label = f"{info['name'].upper()} (Cap: {info['max_robots']})"
                ax.text(-7, y_pos_bottom, label,
                        va='bottom',  # 垂直方向の配置基準を 'bottom' に変更
                        ha='left',
                        fontsize=10,
                        fontweight='bold', color='black',
                        bbox=dict(boxstyle="round,pad=0.3", fc='whitesmoke', ec='none', alpha=0.8))

    # 各タスクのバーをプロット
    for i, t in enumerate(all_task_ids):
        task_name_full = task_id_to_name.get(t, f"Task {t}")
        base_task_name = _get_base_task_name(task_name_full)

        # 1. 要求Capabilityの楕円を左側に描画 (Placement/Retrievalでは省略)
        if not (task_name_full.startswith("Placement-") or task_name_full.startswith("Retrieval-")):
            if base_task_name in task_name_to_required_caps:
                required_caps = sorted(task_name_to_required_caps[base_task_name])
                num_caps = len(required_caps)
                start_y = i - (num_caps - 1) * 0.15
                for j, cap in enumerate(required_caps):
                    color = capability_color_map.get(cap, "grey")
                    center_y = start_y + j * 0.3
                    ellipse = patches.Ellipse(
                        xy=(-1.5, center_y),
                        width=1.2,
                        height=0.25,
                        facecolor=color, edgecolor="black", linewidth=0.5,
                        clip_on=False
                    )
                    ax.add_patch(ellipse)

        # 2. タスクバーを描画
        if t in executed_tasks and t in selected_recipes:
            start = solver.value(task_starts[t])
            duration = solver.value(task_durations[t])
            if duration <= 0:
                continue

            recipe_idx = selected_recipes.get(t)
            resources_used_data = mode_to_resources_map.get((task_name_full, recipe_idx), [])

            unique_res_list = []
            carrier_robot = None

            # --- 使用リソースのリストを準備 ---
            if isinstance(resources_used_data, dict): # Placement/Retrieval タスク
                carrier_robot = resources_used_data.get('carrier')
                payload = resources_used_data.get('payload', [])
                combined_resources = ([carrier_robot] if carrier_robot else []) + payload
                unique_res_list = list(set(combined_resources))
            else: # Work タスク
                unique_res_list = list(set(resources_used_data))

            # --- リソースのカスタムソート ---
            def sort_key(res_name):
                if res_name in module_names:
                    return (0, res_name)  # モジュールが先
                elif res_name in robot_names:
                    return (1, res_name)  # ロボットが後
                else:
                    return (2, res_name)  # その他

            unique_res_list.sort(key=sort_key)
            all_res_for_task = unique_res_list

            # --- 描画ロジック ---
            num_resources = len(all_res_for_task)
            if num_resources == 0:
                ax.barh(i, duration, left=start, height=0.6, color="lightgrey", edgecolor="black")
            else:
                total_bar_height = 0.7
                sub_bar_height = total_bar_height / num_resources
                for k, res_name in enumerate(all_res_for_task):
                    color = resource_color_map.get(res_name, "grey")
                    y_pos = (i - total_bar_height / 2) + (sub_bar_height / 2) + k * sub_bar_height
                    hatch_pattern = '//' if res_name == carrier_robot else None
                    ax.barh(y_pos, duration, left=start, height=sub_bar_height,
                          color=color, edgecolor="black", hatch=hatch_pattern)

        elif t not in executed_tasks:
            ax.text(0, i, "--- SKIPPED ---", va='center', ha='left', style='italic', color='lightgrey')

    ax.set_ylabel("Task")
    ax.set_title("Task Schedule Gantt Chart")
    ax.invert_yaxis()
    ax.grid(True, which="major", axis="x", linestyle="--", linewidth=0.5)


def visualize_schedule_only(
    solver,
    all_active_tasks, executed_tasks, task_starts,
    task_durations, selected_recipes,
    task_id_to_name, mode_to_name, title,
    task_name_to_required_caps,
    mode_to_resources_map,
    capability_color_map,
    resource_color_map,
    input_data
):
    """
    Visualizes the scheduling result with the improved Gantt chart.
    (タスクを場所でソートする機能を追加)
    """
    # 1. 場所情報に基づいてタスクをソートするための準備
    locations = input_data.get('locations', [])
    tasks_data = input_data.get('tasks', [])
    location_order = {loc['name']: i for i, loc in enumerate(locations)}
    name_to_task_id = {v: k for k, v in task_id_to_name.items()}

    task_id_to_location = {}
    for task_info in tasks_data:
        base_name = task_info['name']
        location = task_info.get('location')
        if location:
            work_id = name_to_task_id.get(base_name)
            if work_id:
                task_index = (work_id + 1) // 3
                placement_id, work_id, retrieval_id = get_task_ids(int(task_index))
                task_id_to_location[placement_id] = location
                task_id_to_location[work_id] = location
                task_id_to_location[retrieval_id] = location

    def sort_key(task_id):
        location = task_id_to_location.get(task_id)
        order = location_order.get(location, float('inf'))
        return (order, task_id)

    sorted_task_ids = sorted(list(all_active_tasks), key=sort_key)

    # 2. グラフ描画
    makespan = int(solver.objective_value)
    gantt_height = max(5, len(all_active_tasks) * 0.6)
    fig, ax = plt.subplots(figsize=(20, gantt_height))

    # 場所ラベルのスペースを確保するために左マージンを調整
    fig.subplots_adjust(left=0.2, right=0.8)

    _plot_gantt_chart(
        ax, solver, sorted_task_ids, set(executed_tasks),
        task_starts, task_durations, selected_recipes,
        task_id_to_name,
        task_name_to_required_caps,
        mode_to_resources_map,
        capability_color_map,
        resource_color_map,
        input_data,
        task_id_to_location  # 場所情報を描画関数に渡す
    )

    ax.set_xlabel("Time")
    ax.set_xlim(-8, makespan + 5) # ラベル表示用に左側のリミットを調整
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=20))

    _draw_custom_legends(
        fig,
        capability_color_map,
        resource_color_map,
        input_data
    )

    plt.show()


def calculate_all_task_combinations(input_data):
    """
    input_dataを受け取り、各タスクの要求を満たす「既約」な
    リソースの組み合わせを全パターン計算して返します。
    いずれかのタスクで有効な組み合わせが見つからない場合はNoneを返します。
    """
    def _find_irreducible_covers(required_caps, available_resources):
        if not required_caps: return [[]]
        all_valid_covers = []
        for i in range(1, len(available_resources) + 1):
            for combo in combinations(available_resources, i):
                combined_caps = set(c for res in combo for c in res["capabilities"])
                if required_caps.issubset(combined_caps):
                    all_valid_covers.append(list(combo))

        irreducible_solutions = []
        for combo in all_valid_covers:
            is_irreducible = True
            if len(combo) > 1:
                for sub_combo in combinations(combo, len(combo) - 1):
                    sub_caps = set(c for res in sub_combo for c in res["capabilities"])
                    if required_caps.issubset(sub_caps):
                        is_irreducible = False
                        break
            if is_irreducible:
                solution_names = sorted([res["name"] for res in combo])
                if solution_names not in irreducible_solutions:
                    irreducible_solutions.append(solution_names)
        return irreducible_solutions

    all_resources = input_data["resources"]["robot"] + input_data["resources"]["module"]
    all_task_combinations = {}

    for task in input_data["tasks"]:
        task_name, required_capabilities = task["name"], set(task["required_capabilities"])
        combinations_for_task = _find_irreducible_covers(required_capabilities, all_resources)

        # 組み合わせが見つからない場合は即座にエラーを出す。
        if not combinations_for_task:
            raise ValueError("[calculate_all_task_combinations] エラー: 一部のタスクでリソースの組み合わせが見つかりません")

        all_task_combinations[task_name] = combinations_for_task

    return all_task_combinations

def draw_capabilities(ax, capabilities, x_start, y_pos, cap_color_map, patch_size=0.6, patch_margin=0.1, aspect_correction=1.0):
    """
    グラフのアスペクト比による歪みを補正する`aspect_correction`引数を追加。
    楕円の幅(width)をこの値で割ることで、表示上は正円に見えるように調整します。
    """
    sorted_caps = sorted(list(capabilities))
    for i, cap in enumerate(sorted_caps):
        if cap in cap_color_map:
            center_x = x_start + i * (patch_size * 0.3 + patch_margin) + patch_size / 2
            ellipse = patches.Ellipse(
                (center_x, y_pos),
                width=patch_size / aspect_correction,  # 横方向の歪みを補正
                height=patch_size,
                facecolor=cap_color_map[cap],
                edgecolor='gray'
            )
            ax.add_patch(ellipse)


def visualize_task_combinations(input_data, calculated_combinations, cap_color_map, resource_color_map):
    """
    ### 変更 ###
    - `draw_capabilities`に渡すx座標の値を小さくし、楕円をテキスト側に寄せました。
    - タスク名の横に場所(location)情報を表示するようにしました。
    """
    all_resources = input_data["resources"]["robot"] + input_data["resources"]["module"]
    all_capabilities = set(cap for res in all_resources for cap in res["capabilities"])
    sorted_caps = sorted(list(all_capabilities))

    line_count = sum(2.5 + sum(len(combo) + 0.5 for combo in calculated_combinations[task['name']]) + 1.5
                     for task in input_data["tasks"])

    fig, ax = plt.subplots(figsize=(14, line_count * 0.4))

    ax.set_xlim(0, 10)
    ax.set_ylim(0, line_count)
    ax.axis('off')

    fig.suptitle("Task Assignment Options", fontsize=16, fontweight='bold')

    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    fig_width, fig_height = fig.get_size_inches()
    aspect_correction = (y_range / fig_height) / (x_range / fig_width) * 1.4

    y_pos = line_count - 1
    for task in input_data["tasks"]:
        # ★ タスク名と場所を併記
        task_display_name = f"TASK: {task['name']}"
        if 'location' in task:
            task_display_name += f"  @ {task['location']}"
        ax.text(0.5, y_pos, task_display_name, fontsize=14, fontweight='bold', va='center')

        y_pos -= 1.2
        ax.text(1.0, y_pos, "Required:", fontsize=12, va='center')

        draw_capabilities(ax, task['required_capabilities'], 2.0, y_pos, cap_color_map, aspect_correction=aspect_correction)

        y_pos -= 1.5
        combinations_for_task = calculated_combinations[task['name']]
        for i, combo in enumerate(combinations_for_task):
            ax.text(1.5, y_pos, f"Solution {i+1}", fontsize=12, va='center', style='italic', color='navy')
            y_pos -= 1
            for resource_name in combo:
                ax.text(2.0, y_pos, f"• {resource_name}", fontsize=11, va='center')
                resource_data = next((r for r in all_resources if r["name"] == resource_name), None)
                if resource_data:
                    draw_capabilities(ax, resource_data['capabilities'], 3.8, y_pos, cap_color_map, aspect_correction=aspect_correction)
                y_pos -= 1
            y_pos -= 0.5
        y_pos += 1
        if task != input_data["tasks"][-1]:
             ax.hlines(y=y_pos, xmin=0.5, xmax=9.5, colors='lightgray', linestyles='--')
        y_pos -= 2

    _draw_custom_legends(
        fig,
        cap_color_map,
        resource_color_map,
        input_data,
        show_symbols=False
    )
    fig.subplots_adjust(right=0.8, top=0.92)
    plt.show()


def _process_and_display_solution(
    solver, problem,
    all_active_tasks, all_resources, source, sink,
    task_starts, task_ends, task_durations,
    task_to_presence_literals, task_to_resource_demands,
    task_resource_to_fixed_demands, project_name,
    task_id_to_name, mode_to_name, renewable_id_to_name,
    reservoir_id_to_name,
    num_actual_robots, num_actual_modules,
    task_name_to_required_caps,
    mode_to_resources_map,
    capability_color_map,
    resource_color_map,
    irreducible_combinations,
    input_data
):
    """Processes and displays the solution from the solver."""
    executed_tasks = []
    for t in all_active_tasks:
        literals = task_to_presence_literals[t]
        is_mandatory = len(literals) == 1 and isinstance(literals[0], int)
        is_optional_and_chosen = not is_mandatory and sum(solver.value(lit) for lit in literals) == 1
        if is_mandatory or is_optional_and_chosen:
            executed_tasks.append(t)

    selected_recipes = {}
    for t in executed_tasks:
        literals = task_to_presence_literals[t]
        if len(literals) > 1:
            selected_recipes[t] = next(r for r, lit in enumerate(literals) if solver.value(lit))
        else:
            selected_recipes[t] = 0

    print_schedule_by_task(
        solver, all_active_tasks, executed_tasks, source, sink,
        task_starts, task_durations, task_ends, selected_recipes,
        task_id_to_name, mode_to_name,
        mode_to_resources_map=mode_to_resources_map,
    )
    # Commenting out time-step print for brevity, can be re-enabled if needed
    print_schedule_by_time_step(
        solver, problem, executed_tasks, task_starts, task_ends,
        task_to_resource_demands, all_resources, selected_recipes,
        task_id_to_name, mode_to_name,
        mode_to_resources_map=mode_to_resources_map,
    )

    visualize_task_combinations(
        input_data,
        irreducible_combinations,
        capability_color_map,
        resource_color_map)

    visualize_schedule_only(
        solver,
        all_active_tasks, executed_tasks, task_starts,
        task_durations, selected_recipes,
        task_id_to_name, mode_to_name, "Task Schedule Gantt Chart",
        task_name_to_required_caps,
        mode_to_resources_map,
        capability_color_map,
        resource_color_map,
        input_data=input_data
    )


def solve_rcpsp(
    problem: rcpsp_pb2.RcpspProblem,
    proto_file: str,
    params: str,
    active_tasks: set[int],
    source: int,
    sink: int,
    optional_tasks: set[int],
    num_actual_robots: int,
) -> None:
    """Parse and solve a given RCPSP problem in proto format."""
    # Create the model.
    model = cp_model.CpModel()
    model.name = problem.name

    num_resources = len(problem.resources)

    all_active_tasks = list(active_tasks)
    all_active_tasks.sort()
    all_resources = range(num_resources)

    horizon = problem.deadline if problem.deadline != -1 else problem.horizon
    if _HORIZON.value > 0:
        horizon = _HORIZON.value
    elif horizon == -1:  # Naive computation.
        horizon = sum(max(r.duration for r in t.recipes) for t in problem.tasks)
        if problem.is_rcpsp_max:
            for t in problem.tasks:
                for sd in t.successor_delays:
                    for rd in sd.recipe_delays:
                        for d in rd.min_delays:
                            horizon += abs(d)
    print(f"Horizon = {horizon}", flush=True)

    task_starts = {}
    task_ends = {}
    task_durations = {}
    task_intervals = {}
    task_resource_to_energy = {}
    task_to_resource_demands = collections.defaultdict(list)
    task_to_presence_literals = collections.defaultdict(list)
    task_to_recipe_durations = collections.defaultdict(list)
    task_resource_to_fixed_demands = collections.defaultdict(dict)
    task_resource_to_max_energy = collections.defaultdict(int)
    resource_to_sum_of_demand_max = collections.defaultdict(int)
    is_present_literals = {}

    for t in all_active_tasks:
        task = problem.tasks[t]
        num_recipes = len(task.recipes)
        all_recipes = range(num_recipes)
        start_var = model.new_int_var(0, horizon, f"start_of_task_{t}")
        end_var = model.new_int_var(0, horizon, f"end_of_task_{t}")

        if num_recipes > 1:
            literals = [model.new_bool_var(f"is_present_{t}_{r}") for r in all_recipes]
            if t in optional_tasks:
                model.add_at_most_one(literals)
            else:
                model.add_exactly_one(literals)
        else:
            if t in optional_tasks:
                literals = [model.new_bool_var(f"is_present_{t}_0")]
            else:
                literals = [1]

        if len(literals) == 1 and isinstance(literals[0], int):
            is_present = model.new_constant(1)
        else:
            is_present = model.new_bool_var(f"is_present_{t}")
            model.add(is_present == sum(literals))
        is_present_literals[t] = is_present

        demand_matrix = collections.defaultdict(int)

        for recipe_index, recipe in enumerate(task.recipes):
            task_to_recipe_durations[t].append(recipe.duration)
            for demand, resource in zip(recipe.demands, recipe.resources):
                demand_matrix[(resource, recipe_index)] = demand

        duration_var = model.new_int_var_from_domain(
            cp_model.Domain.from_values(task_to_recipe_durations[t]),
            f"duration_of_task_{t}",
        )

        for r in range(num_recipes):
            model.add(duration_var == task_to_recipe_durations[t][r]).only_enforce_if(
                literals[r]
            )

        task_interval = model.new_optional_interval_var(
            start_var, duration_var, end_var, is_present, f"task_interval_{t}"
        )

        task_starts[t] = start_var
        task_ends[t] = end_var
        task_durations[t] = duration_var
        task_intervals[t] = task_interval
        task_to_presence_literals[t] = literals

        for res in all_resources:
            demands = [demand_matrix[(res, recipe)] for recipe in all_recipes]
            task_resource_to_fixed_demands[(t, res)] = demands
            demand_var = model.new_int_var_from_domain(
                cp_model.Domain.from_values(demands), f"demand_{t}_{res}"
            )
            task_to_resource_demands[t].append(demand_var)
            for r in all_recipes:
                model.add(demand_var == demand_matrix[(res, r)]).only_enforce_if(
                    literals[r]
                )
            resource_to_sum_of_demand_max[res] += max(demands)

        for res in all_resources:
            task_resource_to_energy[(t, res)] = sum(
                literals[r]
                * task_to_recipe_durations[t][r]
                * task_resource_to_fixed_demands[(t, res)][r]
                for r in all_recipes
            )
            task_resource_to_max_energy[(t, res)] = max(
                task_to_recipe_durations[t][r]
                * task_resource_to_fixed_demands[(t, res)][r]
                for r in all_recipes
            )

    M = num_actual_robots
    num_main_tasks = (len(problem.tasks) - 2) // 3

    for n in range(1, num_main_tasks + 1):
        placement_id, work_id, retrieval_id = get_task_ids(n)
        if not all(t in active_tasks for t in [placement_id, work_id, retrieval_id]):
            continue

        placement_lits = task_to_presence_literals[placement_id]
        work_lits = task_to_presence_literals[work_id]
        retrieval_lits = task_to_presence_literals[retrieval_id]

        if len(placement_lits) == len(retrieval_lits):
            for k in range(len(placement_lits)):
                model.add(placement_lits[k] == retrieval_lits[k])

        if M > 0 and len(work_lits) > 0:
            num_work_modes = len(work_lits)
            for i in range(num_work_modes):
                corresponding_placement_lits = placement_lits[i * M : (i + 1) * M]
                model.add(work_lits[i] == sum(corresponding_placement_lits))

    makespan = model.new_int_var(0, horizon, "makespan")
    makespan_size = model.new_int_var(1, horizon, "interval_makespan_size")
    interval_makespan = model.new_interval_var(
        makespan,
        makespan_size,
        model.new_constant(horizon + 1),
        "interval_makespan",
    )

    if problem.is_rcpsp_max:
        for task_id in all_active_tasks:
            task = problem.tasks[task_id]
            is_present_t = is_present_literals[task_id]
            for successor_index, next_id in enumerate(task.successors):
                delay_matrix = task.successor_delays[successor_index]
                if next_id == sink:
                    for m1 in range(len(task.recipes)):
                        p1 = task_to_presence_literals[task_id][m1]
                        delay = delay_matrix.recipe_delays[m1].min_delays[0]
                        model.add(task_starts[task_id] + delay <= makespan).only_enforce_if(p1)
                else:
                    is_present_n = is_present_literals[next_id]
                    num_next_modes = len(problem.tasks[next_id].recipes)
                    for m1 in range(len(task.recipes)):
                        s1 = task_starts[task_id]
                        p1 = task_to_presence_literals[task_id][m1]
                        for m2 in range(num_next_modes):
                            delay = delay_matrix.recipe_delays[m1].min_delays[m2]
                            s2 = task_starts[next_id]
                            p2 = task_to_presence_literals[next_id][m2]
                            model.add(s1 + delay <= s2).only_enforce_if([p1, p2])
    else:
        for t in all_active_tasks:
            is_present_t = is_present_literals[t]
            for n in problem.tasks[t].successors:
                if n == sink:
                    model.add(task_ends[t] <= makespan).only_enforce_if(is_present_t)
                elif n in active_tasks:
                    is_present_n = is_present_literals[n]
                    model.add(task_ends[t] <= task_starts[n]).only_enforce_if([is_present_t, is_present_n])

    capacities = []
    max_cost = 0

    for res in all_resources:
        resource = problem.resources[res]
        c = resource.max_capacity
        if c == -1: c = resource_to_sum_of_demand_max[res]
        if problem.is_resource_investment or resource.renewable:
            intervals = [task_intervals[t] for t in all_active_tasks]
            demands = [task_to_resource_demands[t][res] for t in all_active_tasks]
            if problem.is_resource_investment:
                capacity = model.new_int_var(0, c, f"capacity_of_{res}")
                model.add_cumulative(intervals, demands, capacity)
                capacities.append(capacity)
                max_cost += c * resource.unit_cost
            else:
                if _USE_INTERVAL_MAKESPAN.value:
                    intervals.append(interval_makespan)
                    demands.append(c)
                model.add_cumulative(intervals, demands, c)
        else:
            # Reservoir constraint. Multi-mode compatible
            reservoir_times = []
            reservoir_demands = []
            reservoir_actives = []
            total_consumption_terms = []
            for t in all_active_tasks:
                num_recipes_t = len(problem.tasks[t].recipes)
                for r in range(num_recipes_t):
                    demand = task_resource_to_fixed_demands[(t, res)][r]
                    if demand == 0:
                        continue
                    reservoir_times.append(task_starts[t])
                    reservoir_demands.append(demand)
                    is_recipe_r_active = task_to_presence_literals[t][r]
                    reservoir_actives.append(is_recipe_r_active)
                    total_consumption_terms.append(demand * is_recipe_r_active)

            min_capacity = 0
            # Add multi-mode compatible reservoir constraint
            model.AddReservoirConstraintWithActive(
                reservoir_times,
                reservoir_demands,
                reservoir_actives,
                min_capacity,
                resource.max_capacity,
            )
            # Add constraint on the total consumption of the reservoir resource
            model.add(
                cp_model.LinearExpr.sum(total_consumption_terms) == 0
            )

    if problem.is_resource_investment:
        objective = model.new_int_var(0, max_cost, "capacity_costs")
        model.add(objective == sum(problem.resources[i].unit_cost * capacities[i] for i in range(len(capacities))))
    else:
        objective = makespan
    model.minimize(objective)

    task_starts[source] = model.new_constant(0)
    task_ends[source] = model.new_constant(0)
    task_to_presence_literals[0].append(model.new_constant(1))
    is_present_literals[source] = model.new_constant(1)
    task_starts[sink] = makespan
    task_to_presence_literals[sink].append(model.new_constant(1))
    is_present_literals[sink] = model.new_constant(1)

    if proto_file:
        print(f"Writing proto to{proto_file}")
        model.export_to_file(proto_file)

    solver = cp_model.CpSolver()
    if params:
        text_format.Parse(params, solver.parameters)
    solver.parameters.log_search_progress = True
    status = solver.solve(model)

    results = {
        "solver": solver, "problem": problem, "all_active_tasks": all_active_tasks,
        "all_resources": all_resources, "source": source, "sink": sink,
        "task_starts": task_starts, "task_ends": task_ends, "task_durations": task_durations,
        "task_to_presence_literals": task_to_presence_literals,
        "task_to_resource_demands": task_to_resource_demands,
        "task_resource_to_fixed_demands": task_resource_to_fixed_demands,
    }
    return status, results


def create_color_maps(input_data: dict) -> (dict, dict):
    """
    入力データに基づいて、リソースとケイパビリティのカラーマップを生成します。
    - Renewable Resources: 青系の同系色で統一感を出す
    - Reservoir Resources: 主役の情報なので、鮮やかで区別しやすい 'tab10' を割り当て
    - Capabilities: 補助情報なので、ソフトな 'Set3' を割り当て
    """
    renewable_names = [r['name'] for r in input_data["resources"]["robot"]]
    reservoir_names = [r['name'] for r in input_data["resources"]["module"]]
    resource_color_map = {}

    if renewable_names:
        renewable_cmap = cm.get_cmap('Blues')
        points = np.linspace(0.4, 0.9, len(renewable_names))
        colors = renewable_cmap(points)
        for name, color in zip(renewable_names, colors):
            resource_color_map[name] = color
    if reservoir_names:
        reservoir_cmap = cm.get_cmap('tab10')
        colors = [reservoir_cmap((i + 1) % 10) for i in range(len(reservoir_names))]
        for name, color in zip(reservoir_names, colors):
            resource_color_map[name] = color
    all_caps_set = set(cap for task in input_data["tasks"] for cap in task["required_capabilities"])
    for res_type in ["robot", "module"]:
        for res in input_data["resources"][res_type]:
            all_caps_set.update(res["capabilities"])

    all_caps = sorted(list(all_caps_set))
    capability_cmap = cm.get_cmap('Set3')
    capability_color_map = {cap: capability_cmap(i % 12) for i, cap in enumerate(all_caps)}

    return resource_color_map, capability_color_map

def setup_rcpsp_problem(input_data: dict, combinations: dict) -> (rcpsp_pb2.RcpspProblem, dict):
    """
    入力データをRCPSP形式に変換し、ソルバー用の問題オブジェクトをセットアップします。
    """
    rcpsp_data_string, mode_to_resources_map = generate_rcpsp_max_from_json(
        input_data, combinations, debug_print=True)
    print("\n" + "="*25 + " RCPSP/max Data " + "="*25)
    print(rcpsp_data_string)
    print("="*66 + "\n")

    rcpsp_parser = rcpsp.RcpspParser()
    # withステートメントで一時ファイルを安全に扱う
    with tempfile.NamedTemporaryFile(mode='w+', delete=True, suffix='.sch') as temp_f:
        temp_f.write(rcpsp_data_string)
        temp_f.seek(0) # 書き込み後にファイルポインタを先頭に戻す
        rcpsp_parser.parse_file(temp_f.name)

    problem = rcpsp_parser.problem()
    print_problem_statistics(problem)

    return problem, mode_to_resources_map


def main(_):
    input_data = {
        "project_name": "TestTask",
        "locations": [
            {"name": "kitchen", "max_robots": 2},
            {"name": "entrance", "max_robots": 1},
            {"name": "room_center", "max_robots": 1}
        ],
        "resources": {
            "robot": [
                {"name": "r8_robot", "quantity": 1, "capabilities": ["arm", "camera", "gripper"]},
                {"name": "pr2_robot", "quantity": 2, "capabilities": ["camera", "arm"]},
            ],
            "module": [
                {"name": "arm_module", "quantity": 3, "capabilities": ["arm", "camera", "cleaner"]},
                {"name": "temp_module", "quantity": 1, "capabilities": ["temp"]},
                {"name": "camera_module", "quantity": 1, "capabilities": ["camera"]},
                {"name": "gripper_module", "quantity": 1, "capabilities": ["gripper"]},
                {"name": "cleaner_module", "quantity": 1, "capabilities": ["cleaner"]}
            ]
        },
        "tasks": [
            {"name": "cooking", "duration": 30, "required_capabilities": ["arm", "camera", "gripper"], "location": "kitchen"},
            {"name": "IH", "duration": 20, "required_capabilities": ["arm", "camera", "temp"], "location": "kitchen"},
            {"name": "faucet", "duration": 25, "required_capabilities": ["arm", "gripper"], "location": "kitchen"},
            {"name": "fridge", "duration": 15, "required_capabilities": ["arm", "gripper"], "location": "kitchen"},
            {"name": "wall", "duration": 36, "required_capabilities": ["camera", "cleaner"], "location": "entrance"},
            {"name": "table", "duration": 15, "required_capabilities": ["gripper", "cleaner"], "location": "room_center"}
        ]
    }

    # --- 1. データ準備 ---
    task_id_to_name, mode_to_name = create_name_mappings(input_data)
    renewable_id_to_name, reservoir_id_to_name = create_resource_name_mappings(input_data)
    task_name_to_required_caps = {task['name']: task['required_capabilities'] for task in input_data['tasks']}

    # 色設定を専用関数で実行
    resource_color_map, capability_color_map = create_color_maps(input_data)

    try:
        irreducible_combinations = calculate_all_task_combinations(input_data)
    except ValueError as e:
        print(e)
        return

    # --- 2. 問題の構築 ---
    # RCPSP問題のセットアップを専用関数で実行
    problem, mode_to_resources_map = setup_rcpsp_problem(input_data, irreducible_combinations)

    # --- 3. ソルバーの実行 ---
    num_actual_robots = len(input_data["resources"]["robot"])
    last_task = len(problem.tasks) - 1
    status, results = solve_rcpsp(
        problem=problem, proto_file=_OUTPUT_PROTO.value, params=_PARAMS.value,
        active_tasks=set(range(1, last_task)),
        optional_tasks={},
        source=0, sink=last_task,
        num_actual_robots=num_actual_robots
    )

    # --- 4. 結果の表示 ---
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        num_actual_modules = len(input_data["resources"]["module"])
        _process_and_display_solution(
            project_name=input_data["project_name"],
            task_id_to_name=task_id_to_name,
            mode_to_name=mode_to_name,
            renewable_id_to_name=renewable_id_to_name,
            reservoir_id_to_name=reservoir_id_to_name,
            num_actual_robots=num_actual_robots,
            num_actual_modules=num_actual_modules,
            task_name_to_required_caps=task_name_to_required_caps,
            mode_to_resources_map=mode_to_resources_map,
            capability_color_map=capability_color_map,
            resource_color_map=resource_color_map,
            irreducible_combinations=irreducible_combinations,
            input_data=input_data,
            **results,
        )
    elif status == cp_model.INFEASIBLE:
        print("No solution found.")


if __name__ == "__main__":
    app.run(main)
