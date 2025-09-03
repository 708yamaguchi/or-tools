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

from absl import app
from absl import flags

from google.protobuf import text_format
from ortools.sat.python import cp_model
from ortools.scheduling import rcpsp_pb2
from ortools.scheduling.python import rcpsp

_INPUT = flags.DEFINE_string("input", "", "Input file to parse and solve.")
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
    source: int,
    sink: int,
    task_starts: dict,
    task_durations: dict,
    task_ends: dict,
) -> None:
    """
    タスクごとにスケジュール（開始、期間、終了時刻）を表示する関数
    """
    print("Solution Found:")
    print(f"Optimal Makespan: {solver.objective_value}")
    print("--------------------------------------------------")
    print("--- Schedule by Task ---")

    # 開始ダミータスクを表示
    source_start_val = solver.value(task_starts[source])
    print(
        f"Task {source:2}: "
        f"Start={source_start_val:<3} "
        f"Duration=0   "
        f"End={source_start_val:<3}  (Project Start)"
    )

    # 通常のタスクを表示（IDでソートして見やすくする）
    for t in sorted(all_active_tasks):
        start_val = solver.value(task_starts[t])
        duration_val = solver.value(task_durations[t])
        end_val = solver.value(task_ends[t])
        print(
            f"Task {t:2}: "
            f"Start={start_val:<3} "
            f"Duration={duration_val:<3} "
            f"End={end_val:<3}"
        )

    # 終了ダミータスクを表示
    sink_start_val = solver.value(task_starts[sink])
    print(
        f"Task {sink:2}: "
        f"Start={sink_start_val:<3} "
        f"Duration=0   "
        f"End={sink_start_val:<3}  (Project End / Makespan)"
    )


def print_schedule_by_time_step(
    solver: cp_model.CpSolver,
    problem: rcpsp_pb2.RcpspProblem,
    all_active_tasks: list[int],
    task_starts: dict,
    task_ends: dict,
    task_to_resource_demands: dict,
    all_resources: range,
):
    """
    時刻ごとに実行中のタスクとリソースの状態を表示する関数（再修正版）
    Reservoirの正しい仕様（初期値0、生産が正）に基づいて表示を修正
    """
    print("\n--- Schedule by Time Step ---")
    makespan = int(solver.objective_value)

    # 時刻を0からMakespanまで1ずつ進める
    for t in range(makespan + 1):
        running_tasks = []
        # 各タスクが現在の時刻 t で実行中か確認
        for task_id in all_active_tasks:
            start_time = solver.value(task_starts[task_id])
            end_time = solver.value(task_ends[task_id])
            if start_time <= t < end_time:
                running_tasks.append(task_id)

        print(f"\n[Time: {t}]")
        if not running_tasks:
            print("  Running Tasks: None")
        else:
            print(f"  Running Tasks: {sorted(running_tasks)}")

        # 各リソースの状態を計算して表示
        print("  Resource Status:")
        for res_id in all_resources:
            resource = problem.resources[res_id]
            total_capacity = resource.max_capacity

            if total_capacity == -1:
                print(f"    - (Infinite)    Resource {res_id}: Infinite capacity")
                continue

            # --- リソースの種類に応じて計算を分岐 ---

            # 1. Renewable Resource
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
                    f" Used={used_capacity}/{total_capacity}"
                    f" (Remaining={remaining_capacity})"
                )

            # 2. Reservoir Resource
            elif problem.is_consumer_producer:
                # 仕様に基づき、レベル(累積需要)は0から開始する。
                # 生産が正、消費が負の需要として、時刻tまでに開始したタスクの需要を合計する。
                cumulative_demand = 0
                for task_id in all_active_tasks:
                    start_time = solver.value(task_starts[task_id])
                    if start_time <= t:
                        if (
                            task_id in task_to_resource_demands
                            and len(task_to_resource_demands[task_id]) > res_id
                        ):
                            demand = solver.value(
                                task_to_resource_demands[task_id][res_id]
                            )
                            cumulative_demand += demand
                min_level = resource.min_capacity
                max_level = resource.max_capacity
                print(
                    f"    - (Reservoir)   Resource {res_id}:"
                    f" Level={cumulative_demand:4}"
                    f" (Bounds: [{min_level}, {max_level}])"
                )

            # 3. Simple Non-renewable Resource
            else:
                consumed_so_far = 0
                for task_id in all_active_tasks:
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
                print(
                    f"    - (NonRenewable) Resource {res_id}:"
                    f" Remaining={remaining}/{total_capacity}"
                    f" (Consumed={consumed_so_far})"
                )


def solve_rcpsp(
    problem: rcpsp_pb2.RcpspProblem,
    proto_file: str,
    params: str,
    active_tasks: set[int],
    source: int,
    sink: int,
) -> None:
    """Parse and solve a given RCPSP problem in proto format.

    The model will only look at the tasks {source} + {sink} + active_tasks, and
    ignore all others.

    Args:
      problem: the description of the model to solve in protobuf format
      proto_file: the name of the file to export the CpModel proto to.
      params: the string representation of the parameters to pass to the sat
        solver.
      active_tasks: the set of active tasks to consider.
      source: the source task in the graph. Its end will be forced to 0.
      sink: the sink task of the graph. Its start is the makespan of the problem.

    Returns:
      (lower_bound of the objective, best solution found, assignment)
    """
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

    # Create task variables.
    for t in all_active_tasks:
        task = problem.tasks[t]
        num_recipes = len(task.recipes)
        all_recipes = range(num_recipes)

        start_var = model.new_int_var(0, horizon, f"start_of_task_{t}")
        end_var = model.new_int_var(0, horizon, f"end_of_task_{t}")

        if num_recipes > 1:
            # Create one literal per recipe.
            literals = [model.new_bool_var(f"is_present_{t}_{r}") for r in all_recipes]

            # Exactly one recipe must be performed.
            model.add_exactly_one(literals)

        else:
            literals = [1]

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
        task_interval = model.new_interval_var(
            start_var, duration_var, end_var, f"task_interval_{t}"
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
            num_modes = len(task.recipes)

            for successor_index, next_id in enumerate(task.successors):
                delay_matrix = task.successor_delays[successor_index]
                num_next_modes = len(problem.tasks[next_id].recipes)
                for m1 in range(num_modes):
                    s1 = task_starts[task_id]
                    p1 = task_to_presence_literals[task_id][m1]
                    if next_id == sink:
                        delay = delay_matrix.recipe_delays[m1].min_delays[0]
                        model.add(s1 + delay <= makespan).only_enforce_if(p1)
                    else:
                        for m2 in range(num_next_modes):
                            delay = delay_matrix.recipe_delays[m1].min_delays[m2]
                            s2 = task_starts[next_id]
                            p2 = task_to_presence_literals[next_id][m2]
                            model.add(s1 + delay <= s2).only_enforce_if([p1, p2])
    else:
        # Normal dependencies (task ends before the start of successors).
        for t in all_active_tasks:
            for n in problem.tasks[t].successors:
                if n == sink:
                    model.add(task_ends[t] <= makespan)
                elif n in active_tasks:
                    model.add(task_ends[t] <= task_starts[n])

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
            intervals = [task_intervals[t] for t in all_active_tasks]
            demands = [task_to_resource_demands[t][res] for t in all_active_tasks]

            if problem.is_resource_investment:
                capacity = model.new_int_var(0, c, f"capacity_of_{res}")
                model.add_cumulative(intervals, demands, capacity)
                capacities.append(capacity)
                max_cost += c * resource.unit_cost
            else:  # Standard renewable resource.
                # 自分の場合、Renewableなら、ここが呼ばれる
                if _USE_INTERVAL_MAKESPAN.value:
                    intervals.append(interval_makespan)
                    demands.append(c)

                model.add_cumulative(intervals, demands, c)
        else:  # Non empty non renewable resource. (single mode only)
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
            else:  # No producer-consumer. We just sum the demands.
                # model.add(
                #     cp_model.LinearExpr.sum(
                #         [task_to_resource_demands[t][res] for t in all_active_tasks]
                #     )
                #     <= c
                # )
                #

                # 自分の場合、Renewableでないなら、ここが呼ばれる
                # 変更：NonRenewable Resourcesの代わりに、Reservoir Resourcesを利用。
                # 制約の引数として渡すための3つのリストを準備
                reservoir_times = []
                reservoir_demands = []
                reservoir_actives = []

                # 全てのアクティブなタスクをループ
                for t in all_active_tasks:
                    # タスクtが持つレシピの数を取得
                    num_recipes = len(problem.tasks[t].recipes)
                    # タスクtの各レシピrを、それぞれ独立したオプショナルなイベントとして扱う
                    for r in range(num_recipes):
                        # レシピrの需要量（これは固定の整数値）を取得
                        demand = task_resource_to_fixed_demands[(t, res)][r]
                        # 需要が0のイベントは残量に影響しないため、モデルに追加不要
                        if demand == 0:
                            continue
                        # 1. Times: イベントの発生時刻（タスクtの開始時刻）
                        reservoir_times.append(task_starts[t])
                        # 2. Demands: 資源レベルの変化量（固定値）
                        #    消費（正の値）をそのまま渡す
                        reservoir_demands.append(demand)
                        # 3. Actives: イベントが有効かを示すブール変数
                        #    （タスクtでレシピrが選択された場合にTrueになる変数）
                        is_recipe_r_active = task_to_presence_literals[t][r]
                        reservoir_actives.append(is_recipe_r_active)
                # 全てのタスクの全レシピをイベントとして登録し、Reservoir制約を追加
                resource.min_capacity = 0
                model.AddReservoirConstraintWithActive(
                    reservoir_times,
                    reservoir_demands,
                    reservoir_actives,
                    resource.min_capacity,  # 資源残量の下限 (0。このプログラム内で補完)
                    resource.max_capacity,  # 資源残量の上限 (初期容量。タスクファイルに定義されている)
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
    task_starts[source] = 0
    task_ends[source] = 0
    task_to_presence_literals[0].append(True)
    task_starts[sink] = makespan
    task_to_presence_literals[sink].append(True)

    # Write model to file.
    if proto_file:
        print(f"Writing proto to{proto_file}")
        model.export_to_file(proto_file)

    # Solve model.
    solver = cp_model.CpSolver()

    # Parse user specified parameters.
    if params:
        text_format.Parse(params, solver.parameters)

    # Favor objective_shaving over objective_lb_search.
    if solver.parameters.num_workers >= 16 and solver.parameters.num_workers < 24:
        solver.parameters.ignore_subsolvers.append("objective_lb_search")
        solver.parameters.extra_subsolvers.append("objective_shaving")

    # Experimental: Specify the fact that the objective is a makespan
    solver.parameters.push_all_tasks_toward_start = True

    # Enable logging in the main solve.
    solver.parameters.log_search_progress = True

    # Solve the model.
    status = solver.solve(model)

    # Print Schedule
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Solution Found:")
        print(f"Optimal Makespan: {solver.objective_value}")
        print("--------------------------------------------------")

        # 1. タスクごとのスケジュールを表示
        print_schedule_by_task(
            solver=solver,
            all_active_tasks=all_active_tasks,
            source=source,
            sink=sink,
            task_starts=task_starts,
            task_durations=task_durations,
            task_ends=task_ends,
        )

        # 2. 時刻ごとのスケジュールを表示
        print_schedule_by_time_step(
            solver=solver,
            problem=problem,
            all_active_tasks=all_active_tasks,
            task_starts=task_starts,
            task_ends=task_ends,
            task_to_resource_demands=task_to_resource_demands,
            all_resources=all_resources,
        )

    elif status == cp_model.INFEASIBLE:
        print("No solution found.")


def main(_):
    rcpsp_parser = rcpsp.RcpspParser()
    rcpsp_parser.parse_file(_INPUT.value)

    problem = rcpsp_parser.problem()
    print_problem_statistics(problem)

    last_task = len(problem.tasks) - 1

    solve_rcpsp(
        problem=problem,
        proto_file=_OUTPUT_PROTO.value,
        params=_PARAMS.value,
        active_tasks=set(range(1, last_task)),
        source=0,
        sink=last_task,
    )


if __name__ == "__main__":
    app.run(main)
