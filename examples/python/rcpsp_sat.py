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
import matplotlib.patches as patches # patchesを追加
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


def calculate_all_task_combinations(input_data):
    """
    input_dataを受け取り、各タスクの要求を満たす「既約」な
    リソースの組み合わせを全パターン計算して返します。(以前作成した関数)
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
                        is_irreducible = False; break
            if is_irreducible:
                solution_names = sorted([res["name"] for res in combo])
                if solution_names not in irreducible_solutions:
                    irreducible_solutions.append(solution_names)
        return irreducible_solutions

    all_resources = input_data["resources"]["renewable"] + input_data["resources"]["reservoir"]
    all_task_combinations = {}
    for task in input_data["tasks"]:
        task_name, required_capabilities = task["name"], set(task["required_capabilities"])
        combinations_for_task = _find_irreducible_covers(required_capabilities, all_resources)
        all_task_combinations[task_name] = combinations_for_task
    return all_task_combinations


def generate_rcpsp_max_from_json(input_data):
    """
    JSON形式の入力データからRCPSP/max形式の文字列を生成します。
    また、ソルバーの解を可視化するために(タスク名, モード番号) -> [リソース名]の
    マッピング辞書も同時に生成して返します。

    --- フォーマット仕様 (修正版) ---

    ■ ヘッダー行
    <タスク数(3*N)> <Renewable Resourcesの数> <Reservoir Resourcesの数>

    ■ Renewable Resourcesの定義
    ・数: [実際のロボット数] + [実際のモジュール数] + N
    ・意味:
      - 0 ~ (ロボット数-1)番目: 各種「空きロボット数」。
      - (ロボット数) ~ (ロボット数+モジュール数-1)番目: 各種「空きモジュール数」。
      - (ロボット数+モジュール数) ~ : 各「未実行タスク数」。

    ■ Reservoir Resourcesの定義
    ・数: 0 (★修正: ロック機能はソルバー内の制約で実現するため不要)
    """
    # --------------------------------------------------------------------------
    # 1. データの解析と新しい仕様に合わせたパラメータ設定
    # --------------------------------------------------------------------------
    tasks = input_data.get('tasks', [])
    N = len(tasks)
    task_combinations = calculate_all_task_combinations(input_data)

    actual_robots = input_data['resources']['renewable']
    num_actual_robots = len(actual_robots)
    robot_map = {res['name']: i for i, res in enumerate(actual_robots)}

    actual_modules = input_data['resources']['reservoir']
    num_actual_modules = len(actual_modules)
    module_map = {res['name']: i + num_actual_robots for i, res in enumerate(actual_modules)}

    num_renewable = num_actual_robots + num_actual_modules + N
    num_reservoir = 0  # ロック用のReservoirは不要
    total_resources = num_renewable + num_reservoir

    mode_to_resources_map = {}

    # --------------------------------------------------------------------------
    # 2. アクティビティ情報の構築
    # --------------------------------------------------------------------------
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

        placement_id, work_id, retrieval_id = 3*n-2, 3*n-1, 3*n
        final_activity_id = 3 * N + 1

        demands_placement, demands_work, demands_retrieval = {}, {}, {}
        # モードごとのコストを格納する辞書を準備
        placement_costs_by_mode = {}
        retrieval_costs_by_mode = {}

        num_work_modes = len(combinations_for_task) if combinations_for_task else 1
        num_placement_retrieval_modes = num_work_modes * num_actual_robots if combinations_for_task and num_actual_robots > 0 else 1

        # デバッグ用の情報保存
        w_combos_by_mode = {}
        pr_combos_by_mode = {}

        if not combinations_for_task:
            demands_placement[1] = [0] * total_resources
            demands_work[1] = [0] * total_resources
            demands_retrieval[1] = [0] * total_resources
            # デフォルトのコストを設定
            placement_costs_by_mode[1] = 0
            retrieval_costs_by_mode[1] = 0
        else:
            # --- 作業モードのデマンド (Work Activities) ---
            for i, combo in enumerate(combinations_for_task):
                mode_num = i + 1
                w_combos_by_mode[mode_num] = combo
                mode_to_resources_map[(task_name, i)] = sorted(combo)
                demands_w_mode = [0] * total_resources
                for resource_name in combo:
                    if resource_name in robot_map:
                        demands_w_mode[robot_map[resource_name]] = 1
                    elif resource_name in module_map:
                        demands_w_mode[module_map[resource_name]] = 1
                demands_w_mode[num_actual_robots + num_actual_modules + (n-1)] = 1
                demands_work[mode_num] = demands_w_mode

            # --- 配置・回収モードのデマンド (Placement/Retrieval Activities) ---
            mode_num_pr = 0
            for i, combo in enumerate(combinations_for_task):
                # この組み合わせ(combo)にモジュールが含まれるかチェックし、コストを決定
                combo_uses_module = any(res_name in module_map for res_name in combo)
                cost = 5 if combo_uses_module else 0

                for j in range(num_actual_robots): # 配置/回収はどのロボットでもよい
                    mode_num_pr += 1
                    # モード番号に対応するコストを保存
                    placement_costs_by_mode[mode_num_pr] = cost
                    retrieval_costs_by_mode[mode_num_pr] = cost

                    pr_combos_by_mode[mode_num_pr] = (combo, j)
                    recipe_idx_pr = mode_num_pr - 1
                    carrier_robot_name = actual_robots[j]['name']
                    resources_used = sorted(combo + [carrier_robot_name])
                    mode_to_resources_map[(f"Pre-{task_name}", recipe_idx_pr)] = resources_used
                    mode_to_resources_map[(f"Post-{task_name}", recipe_idx_pr)] = resources_used

                    demands_p_mode = [0] * total_resources
                    demands_p_mode[j] = 1
                    for res_name in combo:
                        if res_name in robot_map:
                            demands_p_mode[robot_map[res_name]] = 1
                        if res_name in module_map:
                            demands_p_mode[module_map[res_name]] = 1
                    demands_placement[mode_num_pr] = demands_p_mode

                    demands_r_mode = [0] * total_resources
                    demands_r_mode[j] = 1
                    for res_name in combo:
                        if res_name in robot_map:
                            demands_r_mode[robot_map[res_name]] = 1
                        if res_name in module_map:
                            demands_r_mode[module_map[res_name]] = 1
                    demands_retrieval[mode_num_pr] = demands_r_mode

        # --- ▼▼▼ ここから修正 ▼▼▼ ---
        # モードごとのコスト情報を持つ 'costs_by_mode' を格納する
        activities[placement_id] = {'modes': num_placement_retrieval_modes, 'successors': [work_id], 'demands': demands_placement, 'costs_by_mode': placement_costs_by_mode}
        # 作業タスクはコスト固定なので従来の 'cost' キーを使用
        activities[work_id] = {'cost': task_duration, 'modes': num_work_modes, 'successors': [retrieval_id], 'demands': demands_work}
        activities[retrieval_id] = {'modes': num_placement_retrieval_modes, 'successors': [final_activity_id], 'demands': demands_retrieval, 'costs_by_mode': retrieval_costs_by_mode}
        # --- ▲▲▲ ここまで修正 ▲▲▲ ---

        # デバッグ出力 (変更なし)
        print(f"--- Task {n} ({task['name']}) -----------------")
        # ... (以下、元のデバッグ出力コードは変更なし) ...
        print("-------------------------------------------\n")

    activities[3*N+1] = {'cost': 0, 'modes': 1, 'successors': [], 'demands': {1: [0] * total_resources}}

    # --------------------------------------------------------------------------
    # 3. RCPSP/max 形式の文字列を生成
    # --------------------------------------------------------------------------
    output_lines = []
    output_lines.append(f"{3 * N} {num_renewable} {num_reservoir} 0")

    # 先行関係ブロック
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
                # 現アクティビティが固定コストを持つ場合 (例: work)
                if 'cost' in act:
                    num_delays = act['modes'] * succ_act['modes']
                    delays = [str(act['cost'])] * num_delays
                # 現アクティビティがモード毎にコストを持つ場合 (例: placement)
                elif 'costs_by_mode' in act:
                    for mode_num in sorted(act['costs_by_mode'].keys()):
                        cost = act['costs_by_mode'][mode_num]
                        # 後続アクティビティの全モードに対して同じ遅延を設定
                        delays.extend([str(cost)] * succ_act['modes'])

                delay_str_parts.append(f"[{' '.join(delays)}]")
            line_parts.append(' '.join(delay_str_parts))
        output_lines.append(' '.join(line_parts))

    # リソース消費ブロック
    for i in sorted(activities.keys()):
        act = activities[i]
        for mode_num, demands in sorted(act['demands'].items()):
            demands_str = ' '.join(map(str, demands))

            # モードに応じたコストを取得
            cost_for_mode = 0
            if 'cost' in act: # 固定コストのアクティビティ
                cost_for_mode = act['cost']
            elif 'costs_by_mode' in act: # モード別コストのアクティビティ
                cost_for_mode = act['costs_by_mode'].get(mode_num, 0)

            if mode_num == 1:
                output_lines.append(f"{i} {mode_num} {cost_for_mode} {demands_str}")
            else:
                output_lines.append(f" {mode_num} {cost_for_mode} {demands_str}")

    robot_caps = [res['capacity'] for res in actual_robots]
    module_caps = [res['capacity'] for res in actual_modules]
    task_lock_caps = [1] * N
    renewable_caps = robot_caps + module_caps + task_lock_caps
    reservoir_caps = []

    output_lines.append(' '.join(map(str, renewable_caps + reservoir_caps)))

    return "\n".join(output_lines), mode_to_resources_map


# ... (create_name_mappings, create_resource_name_mappings, calculate_optional_tasks, print_problem_statistics, print_schedule_by_task, print_schedule_by_time_step は変更なし) ...
def create_name_mappings(input_data: dict) -> (dict, dict):
    """
    入力JSONデータから、タスクIDとモード番号を自然言語にマッピングする辞書を生成する。
    """
    tasks = input_data.get('tasks', [])
    N = len(tasks)

    task_id_to_name = {}

    # 特別なタスク (Start/Finish)
    task_id_to_name[0] = "Start"
    task_id_to_name[3 * N + 1] = "Finish"

    # JSONのtasksに基づくタスク
    for n in range(1, N + 1):
        task_name = tasks[n - 1]['name']
        placement_id, work_id, retrieval_id = get_task_ids(n)

        task_id_to_name[placement_id] = f"Pre-{task_name}"
        task_id_to_name[work_id] = f"{task_name}"
        task_id_to_name[retrieval_id] = f"Post-{task_name}"

    mode_to_name = {}
    for i in range(100):
        mode_to_name[i] = f"Mode {i}"

    return task_id_to_name, mode_to_name


def create_resource_name_mappings(input_data):
    """
    入力データと定義に基づき、リソースIDをリソース名にマッピングする辞書を生成します。
    (★注: generate_rcpsp_max_from_json のリソース定義と整合性が取れるように修正)
    """
    tasks = input_data.get('tasks', [])
    resources = input_data.get('resources', {})
    N = len(tasks)

    renewable_resources = resources.get('renewable', []) # Robots
    num_actual_robots = len(renewable_resources)

    reservoir_resources = resources.get('reservoir', []) # Modules
    num_actual_modules = len(reservoir_resources)

    # 1. Renewable Resources のマッピング
    #    generate_rcpsp_max_from_json の定義順 (robots -> modules -> task slots) に合わせる
    renewable_id_to_name: dict[int, str] = {}
    # Robots
    for i in range(num_actual_robots):
        renewable_id_to_name[i] = renewable_resources[i].get('name', f"Robot_{i+1}")
    # ★修正: モジュールはRenewableとして扱われているため、こちらでマッピング
    for i in range(num_actual_modules):
        resource_id = num_actual_robots + i
        renewable_id_to_name[resource_id] = reservoir_resources[i].get('name', f"Module_{i+1}")
    # Task "slots" (Formerly "Task Lock")
    for n in range(N):
        task_name = tasks[n]['name']
        resource_id = num_actual_robots + num_actual_modules + n
        renewable_id_to_name[resource_id] = f"Execution Slot for '{task_name}'"

    # 2. Reservoir Resources のマッピング
    #    ★修正: ロック機能は削除されたため、Reservoirリソースは0個
    reservoir_id_to_name: dict[int, str] = {}

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
    # ★ 引数を追加
    mode_to_resources_map: dict,
) -> None:
    """
    タスクごとにスケジュール（開始、期間、終了時刻）を表示する関数
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

            # すべてのタスクタイプで統一されたロジックでモード情報を表示する
            mode_str = "N/A"
            recipe_index = selected_recipes.get(t)
            if recipe_index is not None:
                # フルネームをキーとしてリソースを検索
                resources_used = mode_to_resources_map.get((task_name, recipe_index))
                if resources_used:
                    mode_str = f"Res: {resources_used}"
                else:
                    # フォールバックとしてモード番号を表示
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
    # ★ 引数を追加
    mode_to_resources_map: dict,
):
    """
    時刻ごとに実行中のタスクとリソースの状態を表示する関数
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

                # # ▼▼▼ モード表示ロジックを修正 ▼▼▼
                # mode_str = "N/A"
                # recipe_index = selected_recipes.get(task_id)
                # if recipe_index is not None:
                #     base_task_name = _get_base_task_name(task_name)
                #     if not task_name.startswith("Pre-") and not task_name.startswith("Post-"):
                #         resources_used = mode_to_resources_map.get((base_task_name, recipe_index))
                #         if resources_used:
                #             mode_str = f"Res: {resources_used}"
                #         else:
                #             mode_str = f"Mode {recipe_index + 1}"
                #     else:
                #          mode_str = f"Mode {recipe_index + 1}"
                # # ▲▲▲ ここまで修正 ▲▲▲
                # すべてのタスクタイプで統一されたロジックでモード情報を表示する
                mode_str = "N/A"
                recipe_index = selected_recipes.get(task_id)
                if recipe_index is not None:
                    # フルネームをキーとしてリソースを検索
                    resources_used = mode_to_resources_map.get((task_name, recipe_index))
                    if resources_used:
                        mode_str = f"Res: {resources_used}"
                    else:
                        # フォールバックとしてモード番号を表示
                        mode_str = f"Mode {recipe_index + 1}"

                running_tasks_info.append(f"{task_name}({mode_str})")

        print(f"[Time: {t}]")
        if not running_tasks_info:
            print("  Running Tasks: None")
        else:
            print(f"  Running Tasks: {running_tasks_info}")

        # リソース状況の表示（この部分は変更なし）
        print("  Resource Status:")
        for res_id in all_resources:
            resource = problem.resources[res_id]
            total_capacity = resource.max_capacity
            if total_capacity == -1:
                print(f"    - (Infinite)    Resource {res_id}: Infinite capacity")
                continue
            if resource.renewable:
                used_capacity = 0
                for task_id in running_tasks_ids:
                    if (
                        task_id in task_to_resource_demands
                        and len(task_to_resource_demands[task_id]) > res_id
                    ):
                        used_capacity += solver.value(
                            task_to_resource_demands[task_id][res_id]
                        )
                remaining_capacity = total_capacity - used_capacity
                if (
                    remaining_capacity < 0
                    or used_capacity < 0
                    or remaining_capacity > total_capacity
                    or used_capacity > total_capacity
                ):
                    raise ValueError(
                        "Renewable resource must be in [min, max] at any time."
                    )
                print(
                    f"    - (Renewable)   Resource {res_id}:"
                    f" Remaining={remaining_capacity}/{total_capacity}".ljust(46),
                    f" (Used={used_capacity})",
                )
            else:
                consumed_so_far = 0
                for task_id in executed_tasks:
                    start_time = solver.value(task_starts[task_id])
                    if start_time <= t:
                        if (
                            task_id in task_to_resource_demands
                            and len(task_to_resource_demands[task_id]) > res_id
                        ):
                            consumed_so_far += solver.value(
                                task_to_resource_demands[task_id][res_id]
                            )
                remaining = total_capacity - consumed_so_far
                if (
                    remaining < 0
                    or consumed_so_far < 0
                    or remaining > total_capacity
                    or consumed_so_far > total_capacity
                ):
                    raise ValueError(
                        "Reservoir resource must be in [min, max] at any time."
                    )
                print(
                    f"    - (Reservoir)   Resource {res_id}:"
                    f" Remaining={remaining}/{total_capacity}".ljust(46),
                    f" (Consumed={consumed_so_far})",
                )


# --- Visualization Functions ---
def _get_base_task_name(task_name: str) -> str:
    """ 'Pre-kitchen' や 'Post-kitchen' から 'kitchen' を抽出する """
    if task_name.startswith("Pre-") or task_name.startswith("Post-"):
        return "-".join(task_name.split("-")[1:])
    return task_name


def _plot_gantt_chart(
    ax, solver, all_task_ids,
    executed_tasks, task_starts, task_durations,
    selected_recipes, task_id_to_name,
    # ★★★ 以下4つの引数を追加 ★★★
    task_name_to_required_caps,
    mode_to_resources_map,
    capability_color_map,
    resource_color_map
):
    """matplotlib Axesオブジェクトにガントチャートをプロットするヘルパー関数。"""
    y_labels = [task_id_to_name.get(t, f"Task {t}") for t in all_task_ids]

    # Y軸のラベルと目盛りを設定
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)

    # 各タスクのバーを描画
    for i, t in enumerate(all_task_ids):
        task_name_full = task_id_to_name.get(t, f"Task {t}")
        base_task_name = _get_base_task_name(task_name_full)

        # 1. ★★★ 要求Capabilityを色付きの円で描画 ★★★
        if base_task_name in task_name_to_required_caps:
            required_caps = sorted(task_name_to_required_caps[base_task_name])
            for j, cap in enumerate(required_caps):
                color = capability_color_map.get(cap, "grey")
                # Y軸ラベルの左側に円を描画
                circle = patches.Circle(
                    xy=(-j * 0.5 - 0.5, i), # X, Y座標
                    radius=0.2,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.5,
                    clip_on=False # プロットエリア外へのはみ出しを許可
                )
                ax.add_patch(circle)

        # 2. ★★★ タスクバーをリソースの色で塗り分けて描画 ★★★
        if t in executed_tasks and t in selected_recipes:
            start = solver.value(task_starts[t])
            duration = solver.value(task_durations[t])
            recipe_idx = selected_recipes.get(t)

            # このタスク・モードで使用されたリソースのリストを取得
            resources_used = mode_to_resources_map.get((task_name_full, recipe_idx), [])

            if duration > 0:
                if not resources_used:
                    # リソース情報がない場合は単色で描画
                    ax.barh(i, duration, left=start, height=0.6, color="grey", edgecolor="black")
                else:
                    # リソースごとにバーを分割して描画
                    bar_width = duration / len(resources_used)
                    for k, res_name in enumerate(resources_used):
                        color = resource_color_map.get(res_name, "grey")
                        ax.barh(
                            i,
                            bar_width,
                            left=start + k * bar_width,
                            height=0.6,
                            color=color,
                            edgecolor="black"
                        )
                        # リソース名をバーの中に書き込む
                        ax.text(
                           start + (k + 0.5) * bar_width, i, res_name,
                           ha='center', va='center', color='white', fontsize=8, fontweight='bold'
                        )
        elif t not in executed_tasks:
            # スキップされたタスク
            ax.text(0, i, "--- SKIPPED ---", va='center', ha='left', style='italic', color='lightgrey')


    ax.set_ylabel("Task")
    ax.set_title("Task Schedule Gantt Chart")
    ax.invert_yaxis()
    ax.grid(True, which="major", axis="x", linestyle="--", linewidth=0.5)

# ... (_plot_resource_usage は変更なし) ...
def _plot_resource_usage(
    ax, solver, problem,
    res_id, executed_tasks, task_starts, task_ends,
    selected_recipes, task_resource_to_fixed_demands,
    renewable_id_to_name, reservoir_id_to_name, makespan
):
    """Helper to plot usage of a single resource on a given matplotlib Axes object."""
    resource = problem.resources[res_id]
    capacity = resource.max_capacity
    time_points = np.arange(makespan + 2)

    num_renewable = sum(1 for r in problem.resources if r.renewable)

    if resource.renewable:
        name = renewable_id_to_name.get(res_id, f"Renewable {res_id}")
        ax.set_title(f"Main Renewable Resource: {name}")
        ax.set_ylabel("Capacity")

        usage_changes = np.zeros(makespan + 2, dtype=int)
        for task_id in executed_tasks:
            recipe_idx = selected_recipes[task_id]
            demand = task_resource_to_fixed_demands.get((task_id, res_id), [])[recipe_idx]
            if demand != 0:
                start, end = solver.value(task_starts[task_id]), solver.value(task_ends[task_id])
                usage_changes[start] += demand
                if end < len(usage_changes):
                    usage_changes[end] -= demand

        usage = np.cumsum(usage_changes)
        ax.step(time_points, capacity - usage, where="post", label="Remaining", linewidth=4)
        ax.axhline(y=0, color="r", linestyle="--", label="Min (0)")
    else: # Reservoir
        name = reservoir_id_to_name.get(res_id - num_renewable, f"Reservoir {res_id}")
        ax.set_title(f"Main Reservoir Resource: {name}")
        ax.set_ylabel("Level")

        level_changes = np.zeros(makespan + 2, dtype=int)
        for task_id in executed_tasks:
            recipe_idx = selected_recipes[task_id]
            demand = task_resource_to_fixed_demands.get((task_id, res_id), [])[recipe_idx]
            if demand != 0:
                level_changes[solver.value(task_starts[task_id])] -= demand

        level = capacity + np.cumsum(level_changes)
        ax.step(time_points, level, where="post", label="Remaining", linewidth=4)
        ax.axhline(y=0, color="r", linestyle="--", label="Min (0)")

    ax.axhline(y=capacity, color="g", linestyle="--", label=f"Max ({capacity})")
    ax.set_ylim(-1, capacity * 1.1 + 1)
    ax.grid(True, which="major", linestyle="--", linewidth=0.5)
    ax.legend(loc='upper right')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

def visualize_schedule_only(
    solver,
    all_active_tasks, executed_tasks, task_starts,
    task_durations, selected_recipes,
    task_id_to_name, mode_to_name, title,
    task_name_to_required_caps,
    mode_to_resources_map,
    capability_color_map,
    resource_color_map
):
    """
    スケジューリング結果をガントチャートのみで可視化します。
    リソース使用量グラフは表示しません。
    """
    makespan = int(solver.objective_value)

    # ガントチャートの描画高さをタスク数に応じて動的に設定
    gantt_height = max(5, len(all_active_tasks) * 0.5)

    # 1. ガントチャート専用の描画領域を1つ作成
    fig, ax = plt.subplots(figsize=(16, gantt_height))
    fig.suptitle(title, fontsize=18)

    # 2. ヘルパー関数を呼び出してガントチャートをプロット
    _plot_gantt_chart(
        ax, solver, sorted(all_active_tasks), set(executed_tasks),
        task_starts, task_durations, selected_recipes,
        task_id_to_name,
        task_name_to_required_caps,
        mode_to_resources_map,
        capability_color_map,
        resource_color_map
    )

    # 3. X軸のラベルと範囲を設定
    ax.set_xlabel("Time")
    ax.set_xlim(-5, makespan + 5) # Y軸ラベルとCapability表示のスペースを確保
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=20))

    # 4. CapabilityとResourceの凡例をプロットの右側に描画
    # --- Capabilityの凡例 ---
    cap_patches = [patches.Patch(color=color, label=cap) for cap, color in capability_color_map.items()]
    legend1 = ax.legend(handles=cap_patches, title="Capabilities",
                             bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    # --- Resourceの凡例 ---
    res_patches = [patches.Patch(color=color, label=res) for res, color in resource_color_map.items()]
    ax.legend(handles=res_patches, title="Resources",
                   bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0.)

    # 複数の凡例を同時に表示するため、最初の凡例を再追加
    ax.add_artist(legend1)

    # 5. レイアウトを調整して表示
    # Y軸ラベルと凡例が図の範囲に収まるように左右の余白を調整
    fig.subplots_adjust(left=0.2, right=0.82, top=0.92)
    plt.show()


def visualize_schedule_and_main_resources(
    solver, problem,
    all_active_tasks, executed_tasks, task_starts,
    task_ends, task_durations, selected_recipes,
    task_resource_to_fixed_demands, task_id_to_name,
    mode_to_name, renewable_id_to_name, reservoir_id_to_name, title,
    num_actual_robots, num_actual_modules,
    # ★★★ 以下5つの引数を追加 ★★★
    task_name_to_required_caps,
    mode_to_resources_map,
    capability_color_map,
    resource_color_map,
    input_data
):
    """ガントチャートと主要リソースの使用状況を1つの図で可視化します。"""
    makespan = int(solver.objective_value)

    main_resources_to_plot = []
    robot_ids = list(range(num_actual_robots))
    main_resources_to_plot.extend(robot_ids)
    num_total_renewable = sum(1 for r in problem.resources if r.renewable)
    module_ids = list(range(num_total_renewable, num_total_renewable + num_actual_modules))
    main_resources_to_plot.extend(module_ids)

    num_plots = 1 + len(main_resources_to_plot)
    tasks_to_display_ids = sorted([t for t in task_id_to_name if t in all_active_tasks or t in {0, len(task_id_to_name) - 1}])
    gantt_height_ratio = max(4, len(tasks_to_display_ids) * 0.4)
    height_ratios = [gantt_height_ratio] + [3] * len(main_resources_to_plot)

    fig, axes = plt.subplots(
        nrows=num_plots, ncols=1, figsize=(16, sum(height_ratios)), # figsizeを調整
        sharex=True, gridspec_kw={'height_ratios': height_ratios}
    )
    axes = [axes] if num_plots == 1 else axes.flatten()
    fig.suptitle(title, fontsize=18)

    # ガントチャートをプロット
    _plot_gantt_chart(
        axes[0], solver, sorted(all_active_tasks), set(executed_tasks),
        task_starts, task_durations, selected_recipes,
        task_id_to_name,
        # ★★★ 追加した引数を渡す ★★★
        task_name_to_required_caps,
        mode_to_resources_map,
        capability_color_map,
        resource_color_map
    )

    # リソース使用量をプロット
    for i, res_id in enumerate(main_resources_to_plot):
        _plot_resource_usage(
            axes[i + 1], solver, problem, res_id, set(executed_tasks),
            task_starts, task_ends, selected_recipes,
            task_resource_to_fixed_demands, renewable_id_to_name,
            reservoir_id_to_name, makespan
        )

    plt.xlabel("Time")
    plt.xlim(-5, makespan + 5) # 左側にスペースを確保
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=20))

    # --- Capabilityの凡例 ---
    cap_patches = [patches.Patch(color=color, label=cap) for cap, color in capability_color_map.items()]
    legend1 = axes[0].legend(handles=cap_patches, title="Capabilities",
                             bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    # --- Resourceの凡例 ---
    res_patches = [patches.Patch(color=color, label=res) for res, color in resource_color_map.items()]
    axes[0].legend(handles=res_patches, title="Resources",
                   bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0.)

    # 最初の凡例を再追加する必要がある
    axes[0].add_artist(legend1)

    fig.subplots_adjust(left=0.15, right=0.85) # Y軸ラベルと凡例のスペースを確保
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
    input_data
):
    """ソルバーからの解を処理し、表示します。"""
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

    # Console output
    print_schedule_by_task(
        solver, all_active_tasks, executed_tasks, source, sink,
        task_starts, task_durations, task_ends, selected_recipes,
        task_id_to_name, mode_to_name,
        mode_to_resources_map=mode_to_resources_map, # 引数を追加
    )
    print_schedule_by_time_step(
        solver, problem, executed_tasks, task_starts, task_ends,
        task_to_resource_demands, all_resources, selected_recipes,
        task_id_to_name, mode_to_name,
        mode_to_resources_map=mode_to_resources_map, # 引数を追加
    )

    visualize_schedule_only(
        solver,
        all_active_tasks, executed_tasks, task_starts,
        task_durations, selected_recipes,
        task_id_to_name, mode_to_name, "title",
        task_name_to_required_caps,
        mode_to_resources_map,
        capability_color_map,
        resource_color_map
        )

    # # Combined Gantt and Main Resource Chart
    # visualize_schedule_and_main_resources(
    #     solver, problem, all_active_tasks, executed_tasks,
    #     task_starts, task_ends, task_durations, selected_recipes,
    #     task_resource_to_fixed_demands, task_id_to_name, mode_to_name,
    #     renewable_id_to_name, reservoir_id_to_name,
    #     title=f"Task Schedule and Resource Usage for '{project_name}'",
    #     num_actual_robots=num_actual_robots,
    #     num_actual_modules=num_actual_modules,
    #     # ★★★ 追加した引数を渡す ★★★
    #     task_name_to_required_caps=task_name_to_required_caps,
    #     mode_to_resources_map=mode_to_resources_map,
    #     capability_color_map=capability_color_map,
    #     resource_color_map=resource_color_map,
    #     input_data=input_data
    # )


# ... (solve_rcpsp は変更なし) ...
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

        # 3つのタスクがすべてアクティブな場合のみ制約を追加
        if not all(t in active_tasks for t in [placement_id, work_id, retrieval_id]):
            continue

        placement_lits = task_to_presence_literals[placement_id]
        work_lits = task_to_presence_literals[work_id]
        retrieval_lits = task_to_presence_literals[retrieval_id]

        # 【制約1】 PlacementとRetrievalで全く同じモードを選択する
        # (例: Placementでモード3を選択したら、Retrievalも必ずモード3を選択する)
        if len(placement_lits) == len(retrieval_lits):
            for k in range(len(placement_lits)):
                model.add(placement_lits[k] == retrieval_lits[k])

        # 【制約2】 Workで選択したリソースの組み合わせと、Placement/Retrievalで
        # 選択したリソースの組み合わせを一致させる
        if M > 0 and len(work_lits) > 0:
            num_work_modes = len(work_lits)
            for i in range(num_work_modes):
                # Workのi番目のリソース組み合わせに対応するPlacementのモード群
                # (運搬ロボットM台分のモード)
                corresponding_placement_lits = placement_lits[i * M : (i + 1) * M]

                # Workのモードiが選択されることと、対応するPlacementモード群の
                # いずれか一つが選択されることは、同値である。
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
                enforcement_lit_t = [is_present_t]
                if next_id == sink:
                    for m1 in range(len(task.recipes)):
                        p1 = task_to_presence_literals[task_id][m1]
                        delay = delay_matrix.recipe_delays[m1].min_delays[0]
                        model.add(task_starts[task_id] + delay <= makespan).only_enforce_if(p1)
                else:
                    is_present_n = is_present_literals[next_id]
                    enforcement_lit_n = [is_present_n]
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
        if c == -1:
            c = resource_to_sum_of_demand_max[res]
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
            if problem.is_consumer_producer:
                reservoir_starts = []
                reservoir_demands = []
                for t in all_active_tasks:
                    if task_resource_to_fixed_demands[(t, res)][0]:
                        reservoir_starts.append(task_starts[t])
                        reservoir_demands.append(
                            task_resource_to_fixed_demands[(t, res)][0]
                        )
                model.add_reservoir_constraint(
                    reservoir_starts,
                    reservoir_demands,
                    resource.min_capacity,
                    resource.max_capacity,
                )
            else:
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
                model.AddReservoirConstraintWithActive(
                    reservoir_times,
                    reservoir_demands,
                    reservoir_actives,
                    min_capacity,
                    resource.max_capacity,
                )
                model.add(
                    cp_model.LinearExpr.sum(total_consumption_terms) == 0
                )

    if problem.is_resource_investment:
        objective = model.new_int_var(0, max_cost, "capacity_costs")
        model.add(
            objective
            == sum(
                problem.resources[i].unit_cost * capacities[i]
                for i in range(len(capacities))
            )
        )
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
    if solver.parameters.num_workers >= 16 and solver.parameters.num_workers < 24:
        solver.parameters.ignore_subsolvers.append("objective_lb_search")
        solver.parameters.extra_subsolvers.append("objective_shaving")
    solver.parameters.push_all_tasks_toward_start = True
    solver.parameters.log_search_progress = True
    status = solver.solve(model)

    results = {
        "solver": solver,
        "problem": problem,
        "all_active_tasks": all_active_tasks,
        "all_resources": all_resources,
        "source": source,
        "sink": sink,
        "task_starts": task_starts,
        "task_ends": task_ends,
        "task_durations": task_durations,
        "task_to_presence_literals": task_to_presence_literals,
        "task_to_resource_demands": task_to_resource_demands,
        "task_resource_to_fixed_demands": task_resource_to_fixed_demands,
    }
    return status, results


def main(_):
    # 1. Define input JSON data
    input_data = {
        "project_name": "TestTask",
        "resources": {
            "renewable": [
                {
                    "name": "r8_robot",
                    "capacity": 1,
                    "capabilities": ["arm", "camera", "gripper"]
                },
                {
                    "name": "pr2_robot",
                    "capacity": 1,
                    "capabilities": ["camera", "arm"]
                },
            ],
            "reservoir": [
                {
                    "name": "arm_module",
                    "capacity": 1,
                    "capabilities": ["arm", "camera", "cleaner"]
                },
                {
                    "name": "temperature_sensor_module",
                    "capacity": 1,
                    "capabilities": ["temperature_sensor"]
                },
                {
                    "name": "camera_module",
                    "capacity": 1,
                    "capabilities": ["camera"]
                },
                {
                    "name": "gripper_module",
                    "capacity": 1,
                    "capabilities": ["gripper"]
                },
                {
                    "name": "cleaner_module",
                    "capacity": 1,
                    "capabilities": ["cleaner"]
                }
            ]
        },
        "tasks": [
            {
                "name": "kitchen",
                "duration": 30,
                "required_capabilities": ["arm", "camera", "gripper"]
            },
            {
                "name": "IH",
                "duration": 20,
                "required_capabilities": ["arm", "camera", "temperature_sensor"]
            },
            {
                "name": "faucet",
                "duration": 25,
                "required_capabilities": ["arm", "gripper"]
            },
            {
                "name": "fridge",
                "duration": 15,
                "required_capabilities": ["arm", "gripper"]
            },
            {
                "name": "wall",
                "duration": 36,
                "required_capabilities": ["camera", "cleaner"]
            },
            {
                "name": "table",
                "duration": 15,
                "required_capabilities": ["gripper", "cleaner"]
            }
        ]
    }

    # 実際のロボットとモジュールの数を取得
    num_actual_robots = len(input_data["resources"]["renewable"])
    num_actual_modules = len(input_data["resources"]["reservoir"])

    # 1.5 マッピングと色分け辞書を生成
    task_id_to_name, mode_to_name = create_name_mappings(input_data)
    renewable_id_to_name, reservoir_id_to_name = create_resource_name_mappings(input_data)

    # Capabilityの色分け辞書
    all_caps = sorted(list(set(cap for task in input_data["tasks"] for cap in task["required_capabilities"])))
    cap_colors = cm.get_cmap('Pastel1', len(all_caps))
    capability_color_map = {cap: cap_colors(i) for i, cap in enumerate(all_caps)}

    # Resourceの色分け辞書
    all_res = [r['name'] for r in input_data["resources"]["renewable"]] + [r['name'] for r in input_data["resources"]["reservoir"]]
    res_colors = cm.get_cmap('tab20b', len(all_res))
    resource_color_map = {res: res_colors(i) for i, res in enumerate(all_res)}

    # Task名と要求Capabilityのマッピング辞書
    task_name_to_required_caps = {task['name']: task['required_capabilities'] for task in input_data['tasks']}

    # 2. Generate RCPSP/max format string from JSON
    rcpsp_data_string = generate_rcpsp_max_from_json(input_data)
    rcpsp_data_string, mode_to_resources_map = generate_rcpsp_max_from_json(input_data)
    print("--- Generated RCPSP/max data ---")
    print(rcpsp_data_string)
    print("---------------------------------")

    # 3. Parse the problem from the generated string
    rcpsp_parser = rcpsp.RcpspParser()
    with tempfile.NamedTemporaryFile(mode='w+', delete=True, suffix='.sch') as temp_f:
        temp_f.write(rcpsp_data_string)
        temp_f.flush()
        rcpsp_parser.parse_file(temp_f.name)
    problem = rcpsp_parser.problem()
    print_problem_statistics(problem)

    # 4. Solve the problem
    last_task = len(problem.tasks) - 1

    status, results = solve_rcpsp(
        problem=problem,
        proto_file=_OUTPUT_PROTO.value,
        params=_PARAMS.value,
        active_tasks=set(range(1, last_task)),
        # optional_tasks=calculate_optional_tasks(input_data),
        optional_tasks={},  # All tasks must be executed once
        source=0,
        sink=last_task,
        num_actual_robots=num_actual_robots # ロボット数を引数として渡す
    )

    # 5. Visualize result
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
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
            input_data=input_data,
            **results,
        )
    elif status == cp_model.INFEASIBLE:
        print("No solution found.")


if __name__ == "__main__":
    app.run(main)
