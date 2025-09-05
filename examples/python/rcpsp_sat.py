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

from absl import app
from absl import flags

from google.protobuf import text_format
from ortools.sat.python import cp_model
from ortools.scheduling import rcpsp_pb2
from ortools.scheduling.python import rcpsp

import matplotlib.pyplot as plt
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


def generate_rcpsp_max_from_json(input_data):
    """
    新しいJSONデータ形式から、RCPSP/max形式の文字列を生成します。

    Args:
        input_data (dict): プロジェクト情報を含むJSONオブジェクト。
                           - resources.renewable[0].capacity: ロボットの台数
                           - resources.reservoir[0].capacity: モジュールの数
                           - tasks: タスクのリスト (name, duration)

    Returns:
        str: RCPSP/max形式にフォーマットされた文字列。
    """
    # --------------------------------------------------------------------------
    # 1. データの解析と基本パラメータの設定
    # --------------------------------------------------------------------------
    tasks = input_data.get('tasks', [])
    N = len(tasks)

    # 再生可能リソース(Renewable)と貯蔵可能リソース(Reservoir)の総数を計算
    num_renewable = 1 + N
    num_reservoir = 1 + (2 * N)

    # --------------------------------------------------------------------------
    # 2. アクティビティ情報の構築 (IDは0から開始)
    # --------------------------------------------------------------------------
    activities = {}

    # ダミーの開始アクティビティ (ID: 0)
    start_successors = [id for n in range(1, N + 1) for id in get_task_ids(n)[:2]]
    activities[0] = {
        'cost': 0, 'modes': 1, 'successors': start_successors,
        'demands': {1: [0] * (num_renewable + num_reservoir)}
    }

    # タスク関連のアクティビティ (ID: 1 から 3N まで)
    for n in range(1, N + 1):
        task_index = n - 1
        task_duration = tasks[task_index]['duration']

        # 各アクティビティのIDを定義
        placement_id, work_id, retrieval_id = get_task_ids(n)
        final_activity_id = 3 * N + 1

        # --- (3n-2): 配置アクティビティ ---
        demands_placement = [0] * (num_renewable + num_reservoir)
        demands_placement[0] = 1                              # Renewable 1
        demands_placement[n] = 0                              # Renewable n+1
        demands_placement[num_renewable] = 1                  # Reservoir 1
        demands_placement[num_renewable + n] = 1              # Reservoir n+1
        demands_placement[num_renewable + N + n] = 1          # Reservoir N+n+1
        activities[placement_id] = {
            'cost': 5, 'modes': 1, 'successors': [work_id],
            'demands': {1: demands_placement}
        }

        # --- (3n-1): 作業アクティビティ ---
        # モード1のリソース消費
        demands_work_m1 = [0] * (num_renewable + num_reservoir)
        demands_work_m1[0] = 1                                # Renewable 1
        demands_work_m1[n] = 1                                # Renewable n+1

        # モード2のリソース消費
        demands_work_m2 = [0] * (num_renewable + num_reservoir)
        demands_work_m2[num_renewable + n] = -1               # Reservoir n+1 (返却)

        activities[work_id] = {
            'cost': task_duration, 'modes': 2, 'successors': [retrieval_id, final_activity_id],
            'demands': {1: demands_work_m1, 2: demands_work_m2}
        }

        # --- (3n): 回収アクティビティ ---
        demands_retrieval = [0] * (num_renewable + num_reservoir)
        demands_retrieval[0] = 1                              # Renewable 1
        demands_retrieval[num_renewable] = -1                 # Reservoir 1 (返却)
        demands_retrieval[num_renewable + N + n] = -1         # Reservoir N+n+1 (返却)
        activities[retrieval_id] = {
            'cost': 5, 'modes': 1, 'successors': [final_activity_id],
            'demands': {1: demands_retrieval}
        }

    # ダミーの終了アクティビティ (ID: 3N+1)
    activities[3 * N + 1] = {
        'cost': 0, 'modes': 1, 'successors': [],
        'demands': {1: [0] * (num_renewable + num_reservoir)}
    }

    # --------------------------------------------------------------------------
    # 3. RCPSP/max 形式の文字列を生成
    # --------------------------------------------------------------------------
    output_lines = []

    # ヘッダー行
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
                succ_act = activities[succ_id]
                num_delays = act['modes'] * succ_act['modes']
                delays = [str(act['cost'])] * num_delays
                delay_str_parts.append(f"[{' '.join(delays)}]")
            line_parts.append(' '.join(delay_str_parts))

        output_lines.append(' '.join(line_parts))

    # リソース消費ブロック
    for i in sorted(activities.keys()):
        act = activities[i]
        for mode_num in sorted(act['demands'].keys()):
            demands = ' '.join(map(str, act['demands'][mode_num]))
            if mode_num == 1:
                output_lines.append(f"{i} {mode_num} {act['cost']} {demands}")
            else:
                output_lines.append(f" {mode_num} {act['cost']} {demands}")

    # リソース上限ブロック
    robot_quantity = input_data['resources']['renewable'][0]['capacity']
    module_quantity = input_data['resources']['reservoir'][0]['capacity']

    renewable_caps = [robot_quantity] + [1] * N
    reservoir_caps = [module_quantity] + [1] * (2 * N)
    all_caps = renewable_caps + reservoir_caps
    output_lines.append(' '.join(map(str, all_caps)))

    return "\n".join(output_lines)


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

    # モードのマッピング
    mode_to_name = {
        1: "by robot",
        2: "by module",
    }

    return task_id_to_name, mode_to_name


from typing import Dict, Tuple

def create_resource_name_mappings(input_data: dict) -> (dict, dict):
    """
    入力データから、リソースIDをリソース名にマッピングする辞書を生成します。

    Args:
        input_data (Dict): プロジェクトのデータが含まれる辞書。

    Returns:
        Tuple[Dict[int, str], Dict[int, str]]:
            1. RenewableリソースのIDと名前のマッピング辞書。
            2. ReservoirリソースのIDと名前のマッピング辞書。
    """
    tasks = input_data.get('tasks', [])
    resources = input_data.get('resources', {})
    N = len(tasks)

    # resourcesセクションからの名前取得（存在しない場合に備える）
    renewable_resources = resources.get('renewable', [])
    robot_name = renewable_resources[0]['name'] if renewable_resources else "Unknown Robot"

    reservoir_resources = resources.get('reservoir', [])
    module_name = reservoir_resources[0]['name'] if reservoir_resources else "Unknown Module"

    # 1. Renewable Resources のマッピング
    renewable_id_to_name: Dict[int, str] = {}

    # 0番目のリソース（ロボット）
    renewable_id_to_name[0] = robot_name

    # 1番目以降のリソース（タスクごとの作業場所）
    for n in range(N):
        task_name = tasks[n]['name']
        resource_id = n + 1
        renewable_id_to_name[resource_id] = f"{task_name} work place for {robot_name}"

    # 2. Reservoir Resources のマッピング
    reservoir_id_to_name: Dict[int, str] = {}

    # 0番目のリソース（モジュール）
    reservoir_id_to_name[0] = module_name

    # 1番目以降のリソース（タスクごとのダミーリソース）
    for n in range(N):
        task_name = tasks[n]['name']
        # 作業用ダミーリソース
        reservoir_id_to_name[2 * n + 1] = f"dummy work resource for {task_name}"
        # 回収用ダミーリソース
        reservoir_id_to_name[2 * n + 2] = f"dummy collect resource for {task_name}"

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

    # Determine problem type.
    problem_type = (
        "Resource Investment Problem" if problem.is_resource_investment else "RCPSP"
    )

    num_resources = len(problem.resources)
    num_tasks = len(problem.tasks) - 2  # 2 sentinels.
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
    # We print 2 less tasks as these are sentinel tasks that are not counted in
    # the description of the rcpsp models.
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
) -> None:
    """
    タスクごとにスケジュール（開始、期間、終了時刻）を表示する関数
    """
    print("Solution Found:")
    print(f"Optimal Makespan: {solver.objective_value}")
    print("--------------------------------------------------")
    print("--- Schedule by Task (Skipped tasks included) ---")

    # 表示幅を定義
    name_width = max(len(name) for name in task_id_to_name.values()) + 2

    # 開始ダミータスクを表示
    source_name = task_id_to_name.get(source, f"Task {source}")
    source_start_val = solver.value(task_starts[source])
    print(
        f"{source_name:<{name_width}} "
        f"Start={source_start_val:<3} "
        f"Duration=0   "
        f"End={source_start_val:<3}  (Project Start)"
    )

    # 全てのアクティブタスクをID順にループ
    for t in sorted(all_active_tasks):
        task_name = task_id_to_name.get(t, f"Task {t}")
        if t in executed_tasks:
            # 実行されたタスクの情報を表示
            start_val = solver.value(task_starts[t])
            duration_val = solver.value(task_durations[t])
            end_val = solver.value(task_ends[t])
            recipe_index = selected_recipes.get(t, "N/A")

            display_mode_num = recipe_index + 1 if isinstance(recipe_index, int) else None
            mode_str = mode_to_name.get(display_mode_num, f"Mode {display_mode_num}") if display_mode_num else "N/A"

            print(
                f"{task_name:<{name_width}} "
                f"({mode_str}): "
                f"Start={start_val:<3} "
                f"Duration={duration_val:<3} "
                f"End={end_val:<3} "
            )
        else:
            # スキップされたタスクの情報を表示
            print(f"{task_name:<{name_width}}: --- SKIPPED ---")


    # 終了ダミータスクを表示
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
):
    """
    時刻ごとに実行中のタスクとリソースの状態を表示する関数
    Reservoirの正しい仕様（初期値0、生産が正）に基づいて表示を修正
    """
    print("\n--- Schedule by Time Step ---")
    makespan = int(solver.objective_value)

    # 時刻を0からMakespanまで1ずつ進める
    for t in range(makespan + 1):
        running_tasks = []  # リソース計算用のタスクIDリスト
        running_tasks_with_mode = []  # 表示用の文字列リスト

        # 各タスクが現在の時刻 t で実行中か確認
        for task_id in executed_tasks:
            start_time = solver.value(task_starts[task_id])
            end_time = solver.value(task_ends[task_id])
            if start_time <= t < end_time:
                running_tasks.append(task_id)
                recipe_index = selected_recipes.get(task_id, "N/A")

                display_mode_num = recipe_index + 1 if isinstance(recipe_index, int) else None
                mode_str = mode_to_name.get(display_mode_num, f"Mode {display_mode_num}") if display_mode_num else "N/A"
                task_name = task_id_to_name.get(task_id, f"Task {task_id}")

                running_tasks_with_mode.append(f"{task_name}({mode_str})")

        print(f"[Time: {t}]")
        if not running_tasks_with_mode:
            print("  Running Tasks: None")
        else:
            print(f"  Running Tasks: {running_tasks_with_mode}")

        # 各リソースの状態を計算して表示
        print("  Resource Status:")
        for res_id in all_resources:
            resource = problem.resources[res_id]
            total_capacity = resource.max_capacity

            if total_capacity == -1:
                print(f"    - (Infinite)    Resource {res_id}: Infinite capacity")
                continue

            # --- リソースの種類に応じて計算を分岐 ---
            if resource.renewable:
                used_capacity = 0
                for task_id in running_tasks:
                    if (
                        task_id in task_to_resource_demands
                        and len(task_to_resource_demands[task_id]) > res_id
                    ):
                        used_capacity += solver.value(
                            task_to_resource_demands[task_id][res_id]
                        )
                remaining_capacity = total_capacity - used_capacity
                print(
                    f"    - (Renewable)   Resource {res_id}:"
                    f" Remaining={remaining_capacity}/{total_capacity}".ljust(46),
                    f" (Used={used_capacity})",
                )
            # 今回は、Renewable ResourceとReservoir Resourceしか扱わない。
            # consumer_producerでもなく、resource_investmentでもない。
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
def _plot_gantt_chart(
    ax, solver, all_task_ids,
    executed_tasks, task_starts, task_durations,
    selected_recipes, task_id_to_name, mode_to_name
):
    """Helper to plot the Gantt chart on a given matplotlib Axes object."""
    y_labels, starts, durations, bar_texts, colors = [], [], [], [], []

    color_map = cm.viridis(np.linspace(0, 1, len(all_task_ids)))

    for i, t in enumerate(all_task_ids):
        task_name = task_id_to_name.get(t, f"Task {t}")
        if t in executed_tasks:
            y_labels.append(task_name)
            starts.append(solver.value(task_starts[t]))
            durations.append(solver.value(task_durations[t]))
            colors.append(color_map[i])

            recipe_idx = selected_recipes.get(t, 0)
            mode_num = recipe_idx + 1
            mode_name = mode_to_name.get(mode_num, "")
            bar_texts.append(mode_name)
        else:
            y_labels.append(f"{task_name} (Skipped)")
            starts.append(0)
            durations.append(0)
            bar_texts.append("")
            colors.append("lightgrey")

    bars = ax.barh(y=y_labels, width=durations, left=starts, edgecolor="black", color=colors, height=0.6)

    for bar, start, duration, text in zip(bars, starts, durations, bar_texts):
        if duration > 0 and text:
            ax.text(
                start + duration / 2, bar.get_y() + bar.get_height() / 2, text,
                va="center", ha="center", color="white", fontweight="bold"
            )

    ax.set_ylabel("Task")
    ax.set_title("Task Schedule Gantt Chart")
    ax.invert_yaxis()
    ax.grid(True, which="major", axis="x", linestyle="--", linewidth=0.5)


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


def visualize_schedule(
    solver, all_active_tasks, executed_tasks, task_starts,
    task_durations, selected_recipes, task_id_to_name,
    mode_to_name, title="Task Schedule Gantt Chart"
):
    """スケジューリング結果をガントチャートで可視化します。"""
    fig, ax = plt.subplots(figsize=(15, len(all_active_tasks) * 0.4 + 2))
    fig.suptitle(title, fontsize=16)

    _plot_gantt_chart(
        ax, solver, sorted(all_active_tasks), set(executed_tasks),
        task_starts, task_durations, selected_recipes,
        task_id_to_name, mode_to_name
    )

    makespan = int(solver.objective_value)
    ax.set_xlabel("Time")
    ax.set_xlim(0, makespan)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=20))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def visualize_resource_usage(
    solver, problem, executed_tasks, task_starts, task_ends,
    selected_recipes, task_resource_to_fixed_demands,
    renewable_id_to_name, reservoir_id_to_name,
    title="Resource Usage Over Time", main_resource_only=False
):
    """時間経過に伴うリソースの使用量を可視化します。"""
    makespan = int(solver.objective_value)

    if main_resource_only:
        resources_to_plot = []
        try:
            resources_to_plot.append(next(i for i, r in enumerate(problem.resources) if r.renewable))
        except StopIteration: pass
        try:
            resources_to_plot.append(next(i for i, r in enumerate(problem.resources) if not r.renewable))
        except StopIteration: pass
    else:
        resources_to_plot = list(range(len(problem.resources)))

    if not resources_to_plot:
        print("Warning: No resources to visualize.")
        return

    fig, axes = plt.subplots(
        nrows=len(resources_to_plot), ncols=1,
        figsize=(12, 3 * len(resources_to_plot)),
        sharex=True, squeeze=False
    )
    axes = axes.flatten()
    fig.suptitle(title, fontsize=16)

    for ax, res_id in zip(axes, resources_to_plot):
        _plot_resource_usage(
            ax, solver, problem, res_id, set(executed_tasks),
            task_starts, task_ends, selected_recipes,
            task_resource_to_fixed_demands, renewable_id_to_name,
            reservoir_id_to_name, makespan
        )

    plt.xlabel("Time")
    plt.xlim(0, makespan)
    if axes.size > 0:
        axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=20))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def visualize_schedule_and_main_resources(
    solver, problem,
    all_active_tasks, executed_tasks, task_starts,
    task_ends, task_durations, selected_recipes,
    task_resource_to_fixed_demands, task_id_to_name,
    mode_to_name, renewable_id_to_name, reservoir_id_to_name, title
):
    """Visualizes the Gantt chart and main resource usage in a single figure."""
    makespan = int(solver.objective_value)

    # 描画対象とする主要リソースのIDを探し、リストに格納
    main_resources_to_plot = []
    try:
        # 1. 最初の「再生可能リソース」を探し、描画対象として追加。二番目以降は追加しない。
        main_resources_to_plot.append(next(i for i, r in enumerate(problem.resources) if r.renewable))
    except StopIteration: pass
    try:
        # 2. 最初の「貯蔵可能リソース」を探し、描画対象として追加。二番目以降は追加しない。
        main_resources_to_plot.append(next(i for i, r in enumerate(problem.resources) if not r.renewable))
    except StopIteration: pass

    num_plots = 1 + len(main_resources_to_plot)
    tasks_to_display_ids = sorted([t for t in task_id_to_name if t in all_active_tasks or t in {0, len(task_id_to_name) - 1}])
    gantt_height_ratio = max(4, len(tasks_to_display_ids) * 0.3)
    height_ratios = [gantt_height_ratio] + [3] * len(main_resources_to_plot)

    fig, axes = plt.subplots(
        nrows=num_plots, ncols=1, figsize=(15, sum(height_ratios)),
        sharex=True, gridspec_kw={'height_ratios': height_ratios}
    )
    axes = [axes] if num_plots == 1 else axes.flatten()
    fig.suptitle(title, fontsize=18)

    # Plot Gantt Chart
    _plot_gantt_chart(
        axes[0], solver, sorted(all_active_tasks), set(executed_tasks),
        task_starts, task_durations, selected_recipes,
        task_id_to_name, mode_to_name
    )

    # Plot Resource Usage
    for i, res_id in enumerate(main_resources_to_plot):
        _plot_resource_usage(
            axes[i + 1], solver, problem, res_id, set(executed_tasks),
            task_starts, task_ends, selected_recipes,
            task_resource_to_fixed_demands, renewable_id_to_name,
            reservoir_id_to_name, makespan
        )

    plt.xlabel("Time")
    plt.xlim(0, makespan)
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=20))
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def _process_and_display_solution(
    solver, problem,
    all_active_tasks, all_resources, source, sink,
    task_starts, task_ends, task_durations,
    task_to_presence_literals, task_to_resource_demands,
    task_resource_to_fixed_demands, project_name,
    task_id_to_name, mode_to_name, renewable_id_to_name,
    reservoir_id_to_name
):
    """Processes and displays the solution from the solver."""
    # Identify which tasks were actually executed
    executed_tasks = []
    for t in all_active_tasks:
        literals = task_to_presence_literals[t]
        # A task is executed if it's mandatory (literals=[1]) or if one of its optional modes was chosen.
        is_mandatory = len(literals) == 1 and isinstance(literals[0], int)
        is_optional_and_chosen = not is_mandatory and sum(solver.value(lit) for lit in literals) == 1
        if is_mandatory or is_optional_and_chosen:
            executed_tasks.append(t)

    # Determine the recipe selected for each executed task
    selected_recipes = {}
    for t in executed_tasks:
        literals = task_to_presence_literals[t]
        if len(literals) > 1:
            selected_recipes[t] = next(r for r, lit in enumerate(literals) if solver.value(lit))
        else:
            selected_recipes[t] = 0 # Only one recipe available

    # Console output
    print_schedule_by_task(
        solver, all_active_tasks, executed_tasks, source, sink,
        task_starts, task_durations, task_ends, selected_recipes,
        task_id_to_name, mode_to_name
    )
    print_schedule_by_time_step(
        solver, problem, executed_tasks, task_starts, task_ends,
        task_to_resource_demands, all_resources, selected_recipes,
        task_id_to_name, mode_to_name
    )

    # --- Graphical output ---
    # The combined view is called by default.
    # You can uncomment the individual charts if needed.

    # # 1. Gantt Chart Only
    # visualize_schedule(
    #     solver, all_active_tasks, executed_tasks, task_starts,
    #     task_durations, selected_recipes, task_id_to_name,
    #     mode_to_name, title=f"Task Schedule for '{project_name}'"
    # )

    # # 2. Resource Usage Chart Only
    # visualize_resource_usage(
    #     solver, problem, executed_tasks, task_starts, task_ends,
    #     selected_recipes, task_resource_to_fixed_demands,
    #     renewable_id_to_name, reservoir_id_to_name,
    #     title=f"Resource Usage for '{project_name}'"
    # )

    # 3. Combined Gantt and Main Resource Chart
    visualize_schedule_and_main_resources(
        solver, problem, all_active_tasks, executed_tasks,
        task_starts, task_ends, task_durations, selected_recipes,
        task_resource_to_fixed_demands, task_id_to_name, mode_to_name,
        renewable_id_to_name, reservoir_id_to_name,
        title=f"Task Schedule and Resource Usage for '{project_name}'"
    )


def solve_rcpsp(
    problem: rcpsp_pb2.RcpspProblem,
    proto_file: str,
    params: str,
    active_tasks: set[int],
    source: int,
    sink: int,
    optional_tasks: set[int],  # optional_tasks を引数として受け取る
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

    # Containers.
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

    # 各タスクが実行されたかを示す代表ブール変数を格納する辞書
    is_present_literals = {}

    # Create task variables.
    for t in all_active_tasks:
        task = problem.tasks[t]
        num_recipes = len(task.recipes)
        all_recipes = range(num_recipes)

        start_var = model.new_int_var(0, horizon, f"start_of_task_{t}")
        end_var = model.new_int_var(0, horizon, f"end_of_task_{t}")

        # optional_tasks に基づいて literals を定義
        if num_recipes > 1:
            literals = [model.new_bool_var(f"is_present_{t}_{r}") for r in all_recipes]
            if t in optional_tasks:
                model.add_at_most_one(literals)
            else:
                model.add_exactly_one(literals)
        else:  # num_recipesが1の場合
            if t in optional_tasks:
                literals = [model.new_bool_var(f"is_present_{t}_0")]
            else:
                literals = [1]

        # タスクtが実行されたことを示す代表変数 is_present を作成
        # if literals == [1]:
        if len(literals) == 1 and isinstance(literals[0], int):
            is_present = model.new_constant(1)
        else:
            # literalsがブール変数のリストの場合
            is_present = model.new_bool_var(f"is_present_{t}")
            model.add(is_present == sum(literals))
        is_present_literals[t] = is_present

        # Temporary data structure to fill in 0 demands.
        demand_matrix = collections.defaultdict(int)

        # Scan recipes and build the demand matrix and the vector of durations.
        for recipe_index, recipe in enumerate(task.recipes):
            task_to_recipe_durations[t].append(recipe.duration)
            for demand, resource in zip(recipe.demands, recipe.resources):
                demand_matrix[(resource, recipe_index)] = demand

        # Create the duration variable from the accumulated durations.
        duration_var = model.new_int_var_from_domain(
            cp_model.Domain.from_values(task_to_recipe_durations[t]),
            f"duration_of_task_{t}",
        )

        # Link the recipe literals and the duration_var.
        for r in range(num_recipes):
            model.add(duration_var == task_to_recipe_durations[t][r]).only_enforce_if(
                literals[r]
            )

        # Create the interval of the task.
        # is_present を使って OptionalIntervalVar に変更
        task_interval = model.new_optional_interval_var(
            start_var, duration_var, end_var, is_present, f"task_interval_{t}"
        )

        # Store task variables.
        task_starts[t] = start_var
        task_ends[t] = end_var
        task_durations[t] = duration_var
        task_intervals[t] = task_interval
        task_to_presence_literals[t] = literals

        # Create the demand variable of the task for each resource.
        for res in all_resources:
            demands = [demand_matrix[(res, recipe)] for recipe in all_recipes]
            task_resource_to_fixed_demands[(t, res)] = demands
            demand_var = model.new_int_var_from_domain(
                cp_model.Domain.from_values(demands), f"demand_{t}_{res}"
            )
            task_to_resource_demands[t].append(demand_var)

            # Link the recipe literals and the demand_var.
            for r in all_recipes:
                model.add(demand_var == demand_matrix[(res, r)]).only_enforce_if(
                    literals[r]
                )

            resource_to_sum_of_demand_max[res] += max(demands)

        # Create the energy expression for (task, resource):
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

    # Create makespan variable
    makespan = model.new_int_var(0, horizon, "makespan")
    makespan_size = model.new_int_var(1, horizon, "interval_makespan_size")
    interval_makespan = model.new_interval_var(
        makespan,
        makespan_size,
        model.new_constant(horizon + 1),
        "interval_makespan",
    )

    # Add precedences.
    if problem.is_rcpsp_max:
        # In RCPSP/Max problem, precedences are given and max delay (possible
        # negative) between the starts of two tasks.
        for task_id in all_active_tasks:
            task = problem.tasks[task_id]
            is_present_t = is_present_literals[task_id]

            for successor_index, next_id in enumerate(task.successors):
                delay_matrix = task.successor_delays[successor_index]

                # Precedence is conditioned on the presence of task_id
                enforcement_lit_t = [is_present_t]

                if next_id == sink:
                    for m1 in range(len(task.recipes)):
                        p1 = task_to_presence_literals[task_id][m1]
                        delay = delay_matrix.recipe_delays[m1].min_delays[0]
                        model.add(task_starts[task_id] + delay <= makespan).only_enforce_if(p1)
                else:
                    is_present_n = is_present_literals[next_id]
                    # Precedence is conditioned on the presence of both tasks
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
        # Normal dependencies (task ends before the start of successors).
        for t in all_active_tasks:
            is_present_t = is_present_literals[t]
            for n in problem.tasks[t].successors:
                if n == sink:
                    # Enforce only if task t is executed
                    model.add(task_ends[t] <= makespan).only_enforce_if(is_present_t)
                elif n in active_tasks:
                    is_present_n = is_present_literals[n]
                    # Enforce only if both t and n are executed
                    model.add(task_ends[t] <= task_starts[n]).only_enforce_if([is_present_t, is_present_n])

    # Containers for resource investment problems.
    capacities = []  # Capacity variables for all resources.
    max_cost = 0  # Upper bound on the investment cost.

    # Create resources.
    for res in all_resources:
        resource = problem.resources[res]
        c = resource.max_capacity
        if c == -1:
            print(f"No capacity: {resource}")
            c = resource_to_sum_of_demand_max[res]

        # RIP problems have only renewable resources, and no makespan.
        if problem.is_resource_investment or resource.renewable:
            # OptionalIntervalVarを渡すため、intervalsも条件付きになる
            intervals = [task_intervals[t] for t in all_active_tasks]
            demands = [task_to_resource_demands[t][res] for t in all_active_tasks]

            if problem.is_resource_investment:
                capacity = model.new_int_var(0, c, f"capacity_of_{res}")
                model.add_cumulative(intervals, demands, capacity)
                capacities.append(capacity)
                max_cost += c * resource.unit_cost
            else:  # Standard renewable resource.
                if _USE_INTERVAL_MAKESPAN.value:
                    intervals.append(interval_makespan)
                    demands.append(c)

                model.add_cumulative(intervals, demands, c)
        else:  # Non empty non renewable resource.
            if problem.is_consumer_producer:  # single mode only
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
            else:  # Reservoir constraint. Multi-mode compatible
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
                # 複数モードに対応したReservoir Resource制約
                model.AddReservoirConstraintWithActive(
                    reservoir_times,
                    reservoir_demands,
                    reservoir_actives,
                    min_capacity,
                    resource.max_capacity,
                )
                # プロジェクト全体でのReservoir Resource消費量に関する制約
                model.add(
                    cp_model.LinearExpr.sum(total_consumption_terms) == 0
                )

    # Objective.
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

    # Add sentinels.
    # These are mandatory and don't need a presence literal in the same way.
    task_starts[source] = model.new_constant(0)
    task_ends[source] = model.new_constant(0)
    task_to_presence_literals[0].append(model.new_constant(1))
    is_present_literals[source] = model.new_constant(1)

    task_starts[sink] = makespan
    task_to_presence_literals[sink].append(model.new_constant(1))
    is_present_literals[sink] = model.new_constant(1)


    # Write model to file.
    if proto_file:
        print(f"Writing proto to{proto_file}")
        model.export_to_file(proto_file)

    # Solve model.
    solver = cp_model.CpSolver()

    if params:
        text_format.Parse(params, solver.parameters)

    if solver.parameters.num_workers >= 16 and solver.parameters.num_workers < 24:
        solver.parameters.ignore_subsolvers.append("objective_lb_search")
        solver.parameters.extra_subsolvers.append("objective_shaving")

    solver.parameters.push_all_tasks_toward_start = True
    solver.parameters.log_search_progress = True

    status = solver.solve(model)

    # 結果にマッピング辞書を追加
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
            # 可変長だが、今は長さ1を想定
            "renewable": [
                {
                    "name": "r8_robot",
                    "capacity": 1
                }
            ],
            # 可変長で、将来的には複数モジュールが入ることを想定
            "reservoir": [
                {
                    "name": "arm_module",
                    "capacity": 3
                }
            ]
        },
        "tasks": [
            {
                "name": "kitchen",
                "duration": 30
            },
            {
                "name": "IH",
                "duration": 20
            },
            {
                "name": "faucet",
                "duration": 25
            },
            {
                "name": "fridge",
                "duration": 15
            },
            {
                "name": "wall",
                "duration": 36
            },
            {
                "name": "table",
                "duration": 15
            }

        ]
    }

    # 1.5 マッピング辞書を生成
    task_id_to_name, mode_to_name = create_name_mappings(input_data)
    renewable_id_to_name, reservoir_id_to_name = create_resource_name_mappings(input_data)

    # 2. Generate RCPSP/max format string from JSON
    rcpsp_data_string = generate_rcpsp_max_from_json(input_data)
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
        optional_tasks=calculate_optional_tasks(input_data),
        source=0,
        sink=last_task,
    )

    # 5. Visualize result
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        _process_and_display_solution(
            project_name=input_data["project_name"],
            task_id_to_name=task_id_to_name,
            mode_to_name=mode_to_name,
            renewable_id_to_name=renewable_id_to_name,
            reservoir_id_to_name=reservoir_id_to_name,
            **results,
        )
    elif status == cp_model.INFEASIBLE:
        print("No solution found.")



if __name__ == "__main__":
    app.run(main)
