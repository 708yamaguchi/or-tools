# rcpsp_helpers.py

import collections
import io
import tempfile
import json
from itertools import combinations
from collections import Counter

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


def generate_rcpsp_max_from_json(input_data, resolved_task_modes, debug_print=False):
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

    module_handling_time = input_data.get("module_handling_time", 5)

    # 場所情報の読み込み
    locations = input_data.get('locations', [])
    num_locations = len(locations)
    location_map = {loc['name']: i for i, loc in enumerate(locations)}

    # リソース数の定義
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

    # --- 先行関係の解析 ---
    task_name_to_ids = {}
    for n in range(1, N + 1):
        task_name = tasks[n - 1]['name']
        p_id, w_id, r_id = get_task_ids(n)
        task_name_to_ids[task_name] = {'placement': p_id, 'work': w_id, 'retrieval': r_id}

    # 各タスクの後続タスクをマッピングする辞書を作成
    # 例: {'task_A': ['task_B'], 'task_C': []}  (BはAの後に実行)
    successors_map = collections.defaultdict(list)
    predecessors_map = collections.defaultdict(list)
    all_task_names = set(task_name_to_ids.keys())
    tasks_with_predecessors = set()

    for task in tasks:
        task_name = task['name']
        for pred_name in task.get('predecessors', []):
            if pred_name in all_task_names:
                successors_map[pred_name].append(task_name)
                predecessors_map[task_name].append(pred_name)
                tasks_with_predecessors.add(task_name)

    # 先行タスクを持たないタスク（プロジェクトの開始点となりうるタスク）を特定
    start_tasks = all_task_names - tasks_with_predecessors

    # 2. アクティビティ情報の構築
    activities = {}

    # スタートノード(ID=0)の後続を、先行関係を持たないタスクのPlacementアクティビティに設定
    start_successors = [task_name_to_ids[name]['placement'] for name in start_tasks]
    activities[0] = {
        'cost': 0, 'modes': 1, 'successors': sorted(start_successors),
        'demands': {1: [0] * total_resources}
    }

    for n in range(1, N + 1):
        task_index = n - 1
        task = tasks[task_index]
        task_name = task['name']
        modes_for_task = resolved_task_modes[task_name]
        combinations_for_task = [mode['resources'] for mode in modes_for_task]

        # 場所リソースのインデックスを特定
        task_location = task.get('location')
        location_resource_idx = -1
        if task_location and task_location in location_map:
            location_resource_idx = num_actual_robots + num_actual_modules + location_map[task_location]


        placement_id, work_id, retrieval_id = get_task_ids(n)
        final_activity_id = 3 * N + 1

        demands_placement, demands_work, demands_retrieval = {}, {}, {}
        placement_costs_by_mode, retrieval_costs_by_mode = {}, {}

        work_costs_by_mode = {}  # Workアクティビティのモード別durationを格納

        num_work_modes = len(combinations_for_task) if combinations_for_task else 1
        num_placement_retrieval_modes = num_work_modes * num_actual_robots if combinations_for_task and num_actual_robots > 0 else 1

        if not combinations_for_task:
            # モードがない場合のデフォルト処理
            demands_placement[1] = [0] * total_resources
            demands_work[1] = [0] * total_resources
            demands_retrieval[1] = [0] * total_resources
            placement_costs_by_mode[1] = 0
            retrieval_costs_by_mode[1] = 0
            work_costs_by_mode[1] = 0
        else:
            # --- 作業(Work)モードのデマンド ---
            for i, combo in enumerate(combinations_for_task):
                mode_num = i + 1
                # 対応するモードのdurationを保存
                work_costs_by_mode[mode_num] = modes_for_task[i]['duration']
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
                cost = module_handling_time if any(res_name in module_map for res_name in combo) else 0
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

        # --- 後続関係の設定 ---
        # 1. Placement -> Work
        placement_successors = [work_id]

        # 2. Work -> Retrieval
        work_successors = [retrieval_id]

        # 3. Retrieval -> 次のタスクのPlacement または 終了ノード
        #    現在のタスク(task_name)の後続タスクを取得
        dependent_task_names = successors_map.get(task_name, [])
        if dependent_task_names:
            # 後続タスクがある場合、それらのPlacementアクティビティをsuccessorとする
            retrieval_successors = [task_name_to_ids[name]['placement'] for name in dependent_task_names]
        else:
            # 後続タスクがない場合、終了ノードをsuccessorとする
            retrieval_successors = [final_activity_id]


        # Workアクティビティに'cost'の代わりに'costs_by_mode'を設定
        activities[placement_id] = {'modes': num_placement_retrieval_modes, 'successors': sorted(placement_successors), 'demands': demands_placement, 'costs_by_mode': placement_costs_by_mode}
        activities[work_id] = {'modes': num_work_modes, 'successors': sorted(work_successors), 'demands': demands_work, 'costs_by_mode': work_costs_by_mode}
        activities[retrieval_id] = {'modes': num_placement_retrieval_modes, 'successors': sorted(retrieval_successors), 'demands': demands_retrieval, 'costs_by_mode': retrieval_costs_by_mode}

        if debug_print:
            print(f"\n" + "="*15 + f" DEBUG: Task {n} ({task_name}) " + "="*15)
            print(f"  Location: {task_location}")
            print(f"  Activity IDs: Placement={placement_id}, Work={work_id}, Retrieval={retrieval_id}")
            print(f"  Successors: P:{activities[placement_id]['successors']} -> W:{activities[work_id]['successors']} -> R:{activities[retrieval_id]['successors']}")

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

    # 先行関係ブロック
    for i in sorted(activities.keys()):
        act = activities[i]
        num_succ = len(act['successors'])
        line_parts = [str(i), str(act['modes']), str(num_succ)]
        if num_succ > 0:
            line_parts.append(' '.join(map(str, act['successors'])))
            delay_str_parts = []
            for succ_id in act['successors']:
                if succ_id not in activities:
                    continue
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

    # リソース消費ブロック
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
    makespan: float,
) -> None:
    """
    Prints the schedule details for each task.
    """
    print("Solution Found:")
    print(f"Makespan: {makespan}")
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
    makespan: float,
):
    """
    Prints running tasks and resource status for each time step.
    """
    print("\n--- Schedule by Time Step ---")
    loop_end = int(makespan)
    for t in range(loop_end + 1):
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
            if total_capacity == -1:
                continue
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


def _draw_custom_legends(fig, capability_color_map, resource_color_map, input_data, show_symbols=True, title_fontsize=12, label_fontsize=10):
    # --- Capabilities Legend ---
    fig.text(0.83, 0.90, "Capabilities", fontsize=title_fontsize, fontweight='bold')
    y_pos = 0.88
    for cap, color in capability_color_map.items():
        ellipse = patches.Ellipse(xy=(0.835, y_pos), width=0.012, height=0.012,
                                  facecolor=color, edgecolor='black',
                                  transform=fig.transFigure, figure=fig)
        fig.patches.append(ellipse)
        fig.text(0.85, y_pos, cap, fontsize=label_fontsize, va='center')
        y_pos -= 0.03

    # --- Resources Legend ---
    fig.text(0.83, y_pos - 0.02, "Resources", fontsize=title_fontsize, fontweight='bold')
    y_pos -= 0.05

    # Helper to generate capability list from either dict or list format
    def get_caps_to_draw(capabilities):
        caps = []
        if isinstance(capabilities, dict):
            for cap, count in sorted(capabilities.items()):
                caps.extend([cap] * count)
        elif isinstance(capabilities, list):
            caps = capabilities # Old format
        return caps

    # 1. Robots (Renewable) のセクション
    fig.text(0.83, y_pos, "Robots", fontsize=label_fontsize, fontweight='bold', style='italic', color='dimgray')
    y_pos -= 0.035
    for res in input_data["resources"]["robot"]:
        res_name = res["name"]
        res_color = resource_color_map.get(res_name, "grey")
        capacity = res["quantity"]

        fig.patches.extend([plt.Rectangle((0.83, y_pos - 0.015), 0.01, 0.02,
                                          facecolor=res_color, edgecolor='black',
                                          transform=fig.transFigure, figure=fig)])
        fig.text(0.85, y_pos, res_name, fontsize=label_fontsize, va='center')
        fig.text(0.85, y_pos - 0.015, f"(Qty. {capacity})", fontsize=label_fontsize - 2, color='dimgray', va='center')

        x_pos_cap = 0.92
        caps_to_draw = get_caps_to_draw(res.get("capabilities", []))
        for cap in caps_to_draw:
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
    fig.text(0.83, y_pos, "Modules", fontsize=label_fontsize, fontweight='bold', style='italic', color='dimgray')
    y_pos -= 0.035
    for res in input_data["resources"]["module"]:
        res_name = res["name"]
        res_color = resource_color_map.get(res_name, "grey")
        capacity = res["quantity"]

        fig.patches.extend([plt.Rectangle((0.83, y_pos - 0.015), 0.01, 0.02,
                                          facecolor=res_color, edgecolor='black',
                                          transform=fig.transFigure, figure=fig)])
        fig.text(0.85, y_pos, res_name, fontsize=label_fontsize, va='center')
        fig.text(0.85, y_pos - 0.015, f"(Qty. {capacity})", fontsize=label_fontsize - 2, color='dimgray', va='center')

        x_pos_cap = 0.92
        caps_to_draw = get_caps_to_draw(res.get("capabilities", []))
        for cap in caps_to_draw:
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
        fig.text(0.83, y_pos, "Symbols", fontsize=title_fontsize, fontweight='bold')
        y_pos -= 0.035
        fig.patches.extend([plt.Rectangle((0.83, y_pos - 0.0075), 0.01, 0.015,
                                          facecolor='lightgrey', edgecolor='black', hatch='//',
                                          transform=fig.transFigure, figure=fig)])
        fig.text(0.85, y_pos, "Robot-led Placement / Retrieval", fontsize=label_fontsize - 1, va='center')


def _plot_gantt_chart(
    ax, solver, all_task_ids,
    executed_tasks, task_starts, task_durations,
    selected_recipes, task_id_to_name,
    recipe_to_caps_map,
    mode_to_resources_map,
    capability_color_map,
    resource_color_map,
    input_data,
    task_id_to_location,
    title,
    title_fontsize=16,
    label_fontsize=12
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
    ax.set_yticklabels(new_y_labels, fontsize=label_fontsize)


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
                label = f"{info['name'].upper()} (Max: {info['max_robots']})"
                ax.text(-7, y_pos_bottom, label,
                        va='bottom',
                        ha='left',
                        fontsize=label_fontsize,
                        fontweight='bold', color='black',
                        bbox=dict(boxstyle="round,pad=0.3", fc='whitesmoke', ec='none', alpha=0.8))

    # 各タスクのバーをプロット
    for i, t in enumerate(all_task_ids):
        task_name_full = task_id_to_name.get(t, f"Task {t}")
        base_task_name = _get_base_task_name(task_name_full)

        # 1. 要求Capabilityの楕円を左側に描画 (Placement/Retrievalでは省略)
        if not (task_name_full.startswith("Placement-") or task_name_full.startswith("Retrieval-")):
            recipe_idx = selected_recipes.get(t)
            # レシピ番号が取得できた場合のみ描画を試みる
            if recipe_idx is not None:
                # 新しいマップから、選択されたレシピに対応する要求機能を取得
                required_caps_dict = recipe_to_caps_map.get((base_task_name, recipe_idx))

                if required_caps_dict:
                    # 辞書から capability-count のペアをリストに展開
                    caps_to_draw = []
                    for cap, count in sorted(required_caps_dict.items()):
                        caps_to_draw.extend([cap] * count)
                    num_caps = len(caps_to_draw)
                    start_y = i - (num_caps - 1) * 0.15
                    for j, cap in enumerate(caps_to_draw):
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
            ax.text(0, i, "--- SKIPPED ---", va='center', ha='left', style='italic', color='lightgrey', fontsize=label_fontsize)

    ax.set_ylabel("Task", fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    ax.invert_yaxis()
    ax.grid(True, which="major", axis="x", linestyle="--", linewidth=0.5)


def visualize_schedule_only(
    solver,
    all_active_tasks, executed_tasks, task_starts,
    task_durations, selected_recipes,
    task_id_to_name, mode_to_name, title,
    recipe_to_caps_map,
    mode_to_resources_map,
    capability_color_map,
    resource_color_map,
    input_data,
    makespan,
    title_fontsize=16,
    label_fontsize=12
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
    gantt_height = max(5, len(all_active_tasks) * 0.6)
    fig, ax = plt.subplots(figsize=(20, gantt_height))

    # 場所ラベルのスペースを確保するために左マージンを調整
    fig.subplots_adjust(left=0.2, right=0.8)

    _plot_gantt_chart(
        ax, solver, sorted_task_ids, set(executed_tasks),
        task_starts, task_durations, selected_recipes,
        task_id_to_name,
        recipe_to_caps_map,
        mode_to_resources_map,
        capability_color_map,
        resource_color_map,
        input_data,
        task_id_to_location,  # 場所情報を描画関数に渡す
        title,
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize
    )

    ax.set_xlabel("Time", fontsize=label_fontsize)
    ax.tick_params(axis='x', labelsize=label_fontsize)
    ax.set_xlim(-8, makespan + 5) # ラベル表示用に左側のリミットを調整
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=20))

    _draw_custom_legends(
        fig,
        capability_color_map,
        resource_color_map,
        input_data,
        title_fontsize=title_fontsize - 2,
        label_fontsize=label_fontsize - 2
    )

    plt.show()


def resolve_task_modes(input_data: dict) -> dict:
    """
    リソースプール全体（ロボット・モジュール）から、要求機能を満たす「既約」な組み合わせをすべて列挙する。
    """

    def _find_all_irreducible_combinations(required_caps_counter, available_resources):
        """
        要求機能を満たす「既約」な組み合わせをすべて探索して返す。
        既約な組み合わせとは、そのセットからどの要素を1つ取り除いても要求を
        満たせなくなるような、無駄のない組み合わせのこと。
        """
        if not required_caps_counter:
            return [[]]

        # STEP 1: 要求機能を満たす有効な組み合わせをすべて見つける
        valid_covers = []
        for i in range(1, len(available_resources) + 1):
            for combo in combinations(available_resources, i):
                provided_caps_counter = Counter()
                for res in combo:
                    provided_caps_counter.update(res.get("capabilities", {}))

                if not (required_caps_counter - provided_caps_counter):
                    valid_covers.append(list(combo))

        # STEP 2: 有効な組み合わせの中から「既約」なものだけを抽出する
        irreducible_covers = []
        for cover in valid_covers:
            is_irreducible = True
            if len(cover) > 1:
                # この組み合わせから要素を1つ減らした部分集合を作る
                for sub_combo in combinations(cover, len(cover) - 1):
                    # 部分集合が有効かどうかをチェック
                    provided_caps_counter = Counter()
                    for res in sub_combo:
                        provided_caps_counter.update(res.get("capabilities", {}))

                    # もし小さい部分集合でも要求を満たせるなら、元のcoverは「既約」ではない
                    if not (required_caps_counter - provided_caps_counter):
                        is_irreducible = False
                        break

            if is_irreducible:
                irreducible_covers.append(cover)

        # STEP 3: 結果をリソース名のリストに変換し、重複を排除して返す
        solutions = []
        for c in irreducible_covers:
            # 同じ種類のモジュールやロボットは区別しない
            solution_names = sorted([res["name"] for res in c])
            if solution_names not in solutions:
                solutions.append(solution_names)

        return solutions

    resolved_modes = {}

    # --- 1. 全リソースを単一の「リソースプール」に統合 ---
    all_resource_instances = []
    all_resource_types = input_data["resources"]["robot"] + input_data["resources"]["module"]
    for res_type in all_resource_types:
        for i in range(res_type.get('quantity', 1)):
            all_resource_instances.append(res_type)

    # --- 2. 各タスクのモードを解決 ---
    for task in input_data["tasks"]:
        task_name = task["name"]
        final_modes_for_task = []

        if "modes" not in task:
            raise ValueError(f"Task '{task_name}' is missing the 'modes' key.")

        for mode_def in task["modes"]:
            duration = mode_def["duration"]
            required_caps = Counter(mode_def["required_capabilities"])

            # --- 3. 全ての既約な組み合わせを探索 ---
            all_combos = _find_all_irreducible_combinations(required_caps, all_resource_instances)

            # --- 4. 見つかった組み合わせをモードとして定義 ---
            for combo in all_combos:
                final_modes_for_task.append({
                    "duration": duration,
                    "resources": combo,
                    "required_caps": required_caps
                })

        if not final_modes_for_task:
            # 実行可能なモードが一つも見つからなかった場合、明確なエラーメッセージと共に例外を発生させる
            raise ValueError(
                f"ERROR: Task '{task_name}' is impossible to execute with the available resources. "
                f"No valid resource combination could be found for any of its defined modes. "
                f"Please check resource capabilities and task requirements."
            )

        resolved_modes[task_name] = final_modes_for_task

    return resolved_modes


def draw_capabilities(ax, capabilities_counter, x_start, y_pos, cap_color_map, patch_size=0.6, patch_margin=0.1, aspect_correction=1.0):
    """
    与えられたCounterに基づき、機能のシンボルを必要な個数だけ描画します。
    """
    # 描画するケイパビリティのリストを作成 (例: {"arm": 2} -> ["arm", "arm"])
    caps_to_draw = []
    for cap, count in sorted(capabilities_counter.items()):
        caps_to_draw.extend([cap] * count)

    for i, cap in enumerate(caps_to_draw):
        if cap in cap_color_map:
            center_x = x_start + i * (patch_size * 0.3 + patch_margin) + patch_size / 2
            ellipse = patches.Ellipse(
                (center_x, y_pos),
                width=patch_size / aspect_correction,
                height=patch_size,
                facecolor=cap_color_map[cap],
                edgecolor='gray'
            )
            ax.add_patch(ellipse)

def visualize_task_combinations(input_data, resolved_task_modes, cap_color_map, resource_color_map, title_fontsize=16, label_fontsize=12):
    all_resources = input_data["resources"]["robot"] + input_data["resources"]["module"]

    # 全てのユニークなケイパビリティを取得
    all_capabilities = set()
    for res in all_resources:
        all_capabilities.update(res["capabilities"])

    # 表示行数の計算ロジックを修正
    line_count = 0
    for task in input_data["tasks"]:
        line_count += 2.5  # Taskヘッダー
        # --- 変更点: "modes"しかないため、計算を簡略化 ---
        if 'modes' in task:
             line_count += (len(task['modes']) * 1.5)

        # 各モードから展開された組み合わせの行数を加算
        modes_for_task = resolved_task_modes[task['name']]
        line_count += sum(len(mode['resources']) + 1.5 for mode in modes_for_task)


    fig, ax = plt.subplots(figsize=(14, line_count * 0.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, line_count)
    ax.axis('off')

    fig.suptitle("Task Assignment Options", fontsize=title_fontsize, fontweight='bold')

    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    fig_width, fig_height = fig.get_size_inches()
    aspect_correction = (y_range / fig_height) / (x_range / fig_width) * 1.4

    y_pos = line_count - 1
    for task in input_data["tasks"]:
        task_display_name = f"TASK: {task['name']}"
        if 'location' in task:
            task_display_name += f"  @ {task['location']}"
        ax.text(0.5, y_pos, task_display_name, fontsize=label_fontsize + 2, fontweight='bold', va='center')
        y_pos -= 1.5

        # --- 変更点: "modes"しかないため、表示ロジックを簡略化 ---
        if 'modes' in task:
            for i, mode_def in enumerate(task['modes']):
                ax.text(1.0, y_pos, f"Mode {i+1} (Duration: {mode_def['duration']}) Requires:", fontsize=label_fontsize-1, va='center')
                draw_capabilities(ax, Counter(mode_def['required_capabilities']), 3.0, y_pos, cap_color_map, aspect_correction=aspect_correction)
                y_pos -= 1.2

        y_pos -= 0.5
        ax.hlines(y=y_pos, xmin=1.0, xmax=9.0, colors='lightblue', linestyles='-')
        y_pos -= 0.8

        # 解決された実行オプションの表示
        modes_for_task = resolved_task_modes[task['name']]
        for i, mode in enumerate(modes_for_task):
            duration = mode['duration']
            ax.text(1.5, y_pos, f"Option {i+1} (Duration: {duration})", fontsize=label_fontsize, va='center', style='italic', color='navy')
            y_pos -= 1
            for resource_name in mode['resources']:
                ax.text(2.0, y_pos, f"  {resource_name}", fontsize=label_fontsize - 1, va='center')
                resource_data = next((r for r in all_resources if r["name"] == resource_name), None)
                if resource_data:
                    resource_caps_counter = Counter(resource_data['capabilities'])
                    draw_capabilities(ax, resource_caps_counter, 3.8, y_pos, cap_color_map, aspect_correction=aspect_correction)
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
        show_symbols=False,
        title_fontsize=title_fontsize - 2,
        label_fontsize=label_fontsize - 2,
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
    recipe_to_caps_map,
    mode_to_resources_map,
    capability_color_map,
    resource_color_map,
    irreducible_combinations,
    input_data,
    module_capacities,
    optimization_mode, # ★変更点: 最適化モードを受け取る
):
    """Processes and displays the solution from the solver."""
    actual_makespan = solver.value(task_starts[sink])

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
        makespan=actual_makespan,
    )
    # Commenting out time-step print for brevity, can be re-enabled if needed
    print_schedule_by_time_step(
        solver, problem, executed_tasks, task_starts, task_ends,
        task_to_resource_demands, all_resources, selected_recipes,
        task_id_to_name, mode_to_name,
        mode_to_resources_map=mode_to_resources_map,
        makespan=actual_makespan,
    )

    # --- 最適化モードに応じて結果サマリーを表示 ---
    print("\nSolution Found:")
    if optimization_mode == 'MINIMIZE_MAKESPAN':
        print(f"Optimal Makespan: {solver.objective_value}")
    elif optimization_mode == 'MINIMIZE_MODULES':
        print(f"Optimal Total Modules Required: {int(solver.objective_value)}")
        print(f"Schedule completed in {solver.value(task_starts[sink])} time units.")
        print("--------------------------------------------------")
        print("--- Required Capacity per Module ---")
        if not module_capacities:
            print("  No modules were used/optimized in this problem.")
        else:
            num_renewable = len(renewable_id_to_name)
            for res_id, cap_var in module_capacities.items():
                module_name = reservoir_id_to_name.get(res_id - num_renewable, f"Unknown_Module_{res_id}")
                print(f"  - {module_name}: {solver.value(cap_var)}")
    print("--------------------------------------------------")

    # Visualize
    title_font_size, label_font_size = 26, 18
    visualize_task_combinations(
        input_data, irreducible_combinations, capability_color_map,
        resource_color_map, title_fontsize=title_font_size,
        label_fontsize=label_font_size)
    visualize_schedule_only(
        solver, all_active_tasks, executed_tasks, task_starts,
        task_durations, selected_recipes, task_id_to_name, mode_to_name,
        "Task Schedule Gantt Chart", recipe_to_caps_map, mode_to_resources_map,
        capability_color_map, resource_color_map, input_data=input_data,
        makespan=actual_makespan,
        title_fontsize=title_font_size, label_fontsize=label_font_size
    )


def add_variable_capacity_reservoir_constraint(
    model: cp_model.CpModel,
    res_id: int,
    num_main_tasks: int,
    active_tasks: set[int],
    task_starts: dict,
    task_ends: dict,
    task_to_presence_literals: dict,
    task_resource_to_fixed_demands: dict,
    horizon: int,
):
    """【新規追加】可変容量を持つReservoirリソース（モジュール）の制約を追加します。

    この関数は、モジュールが「配置完了」から「回収開始」まで専有されるという、
    このRCPSP問題の特性を利用して、必要なモジュールの最小数を計算するための制約を構築します。
    内部では、モジュールの専有期間をインターバル（期間）変数として定義し、
    `AddCumulative`制約を用いて、同時に存在するインターバルの最大数（=必要なモジュール数）を
    求めるモデルを構築できます。これにより、容量自体を最適化変数として扱う
    リソース投資問題を表現できます。

    ■ 元の `add_reservoir_constraint_with_active` との主な違い:
    https://github.com/google/or-tools/blob/93dafb79a0261b5211f4eb17437aeebd7f45e543/ortools/sat/python/cp_model.py#L1304

    1. 容量の扱い:
       - `add_reservoir_constraint_with_active`:
           容量を`max_level`引数として**固定値の整数**で受け取ります。容量を変数にすることはできません。
       - この関数 (`add_variable_capacity_reservoir_constraint`):
           容量を**最適化変数**として内部で生成し、その変数を戻り値として返します。
           これにより、目的関数で容量を最小化できます。

    2. 制約のモデル手法:
       - `add_reservoir_constraint_with_active`:
           **イベントモデル**。指定時刻にレベルが増減する「貯水池」の動きを直接モデル化します。
       - この関数:
           **インターバルモデル**。リソースが専有される「期間」を定義し、その期間の重なり（累積）を計算します。
           エラーの原因となったライブラリの制限を回避し、リソース投資問題を効率的に扱えます。

    3. 汎用性:
       - `add_reservoir_constraint_with_active`:
           汎用的な関数。時刻と変化量のリストを渡せば、様々な問題に利用できます。
       - この関数:
           このプロジェクトの「配置→作業→回収」というタスク構造に**特化**した関数です。
           そのため、引数もタスクの開始・終了時刻など、より具体的な情報を必要とします。

    Args:
        model (cp_model.CpModel): CP-SATモデルのインスタンス。
        res_id (int): 対象となるリソースのID。
        num_main_tasks (int): プロジェクトの主タスクの総数。
        active_tasks (set[int]): 計算対象となるアクティブなタスクIDのセット。
        task_starts (dict): 各タスクの開始時刻を保持する変数ディクショナリ。
        task_ends (dict): 各タスクの終了時刻を保持する変数ディクショナリ。
        task_to_presence_literals (dict): 各タスクのレシピ選択リテラルのディクショナリ。
        task_resource_to_fixed_demands (dict): (タスク, リソース) -> デマンドリスト のディクショナリ。
        horizon (int): 計算のホライゾン（期間上限）。

    Returns:
        IntVar: 計算された最小キャパシティを表す最適化変数。
    """
    placed_intervals = []
    demands_for_cumulative = []

    # 各タスクがこのモジュールリソースを使用する場合、その使用期間をインターバルとして定義
    for n in range(1, num_main_tasks + 1):
        p_id, _, r_id = get_task_ids(n)
        if p_id not in active_tasks or r_id not in active_tasks:
            continue

        # このタスクの各レシピ（モード）がリソース`res_id`を使用するかチェック
        for recipe_idx, usage_lit in enumerate(task_to_presence_literals[p_id]):
            # usage_litが整数(1)の場合、boolvarに変換する
            if isinstance(usage_lit, int):
                if usage_lit == 1:
                    usage_lit = model.new_constant(1)
                else:
                    continue # このレシピは選択されない

            demand = task_resource_to_fixed_demands[(p_id, res_id)][recipe_idx]
            if demand > 0:
                # モジュールの専有期間(start_varからend_var)は、
                # Placementタスクの開始からRetrievalタスクの終了まで
                start_var = task_starts[p_id] # Placementタスクの開始時刻
                end_var = task_ends[r_id]     # Retrievalタスクの終了時刻

                # 期間長を表現する変数
                duration_var = model.new_int_var(0, horizon, f"placed_duration_{n}_{res_id}_{recipe_idx}")
                model.add(start_var + duration_var == end_var).only_enforce_if(usage_lit)

                # OptionalIntervalVarを作成（このレシピが選択された時のみ有効なインターバル）
                interval = model.new_optional_interval_var(
                    start_var, duration_var, end_var, usage_lit,
                    f"placed_interval_{n}_{res_id}_{recipe_idx}"
                )
                placed_intervals.append(interval)
                demands_for_cumulative.append(demand)

    # このモジュールに必要なキャパシティ（最大同時使用数）を表す変数
    upper_bound = sum(d for d in demands_for_cumulative)
    capacity_var = model.new_int_var(0, upper_bound if upper_bound > 0 else 0, f"capacity_module_{res_id}")

    # AddCumulative制約で、同時使用数がキャパシティを超えないようにする
    if placed_intervals:
        model.add_cumulative(placed_intervals, demands_for_cumulative, capacity_var)

    # 計算されたキャパシティ変数を返す
    return capacity_var


def solve_rcpsp(
    problem: rcpsp_pb2.RcpspProblem,
    active_tasks: set[int],
    source: int,
    sink: int,
    num_actual_robots: int,
    optimization_mode: str,
    makespan_limit: int = None,
    proto_file: str = "",
    params: str = "",
    verbose: bool = True,
) -> None:
    """Parse and solve a given RCPSP problem in proto format."""
    model = cp_model.CpModel()
    model.name = problem.name

    num_resources = len(problem.resources)
    all_active_tasks = sorted(list(active_tasks))
    all_resources = range(num_resources)

    # --- 最適化モードに応じてホライゾン（計算範囲）を決定 ---
    horizon = 0
    if optimization_mode == 'MINIMIZE_MODULES':
        if makespan_limit is None:
            raise ValueError("モジュール数最小化モードでは `makespan_limit` の指定が必要です。")
        horizon = makespan_limit
    else:  # 'MINIMIZE_MAKESPAN' モードの場合
        if problem.deadline != -1:
            horizon = problem.deadline
        elif _HORIZON.value > 0:
            horizon = _HORIZON.value
        else:  # Naive computation.
            horizon = sum(max(r.duration for r in t.recipes) for t in problem.tasks)
            if problem.is_rcpsp_max:
                for t in problem.tasks:
                    for sd in t.successor_delays:
                        for rd in sd.recipe_delays:
                            for d in rd.min_delays:
                                horizon += abs(d)
    print(f"Optimization Mode = {optimization_mode}, Horizon = {horizon}", flush=True)

    # --- 変数定義 ---
    task_starts = {}
    task_ends = {}
    task_durations = {}
    task_intervals = {}
    task_to_resource_demands = collections.defaultdict(list)
    task_to_presence_literals = collections.defaultdict(list)
    task_resource_to_fixed_demands = collections.defaultdict(dict)
    is_present_literals = {}

    for t in all_active_tasks:
        task = problem.tasks[t]
        num_recipes = len(task.recipes)
        all_recipes = range(num_recipes)
        start_var = model.new_int_var(0, horizon, f"start_of_task_{t}")
        end_var = model.new_int_var(0, horizon, f"end_of_task_{t}")
        literals = []
        if num_recipes > 1:
            literals = [model.new_bool_var(f"is_present_{t}_{r}") for r in all_recipes]
            model.add_exactly_one(literals)
        else:
            literals = [1]
        is_present = model.new_bool_var(f"is_present_{t}") if num_recipes > 1 else model.new_constant(1)
        if num_recipes > 1:
            model.add(is_present == sum(literals))
        is_present_literals[t] = is_present
        demand_matrix = collections.defaultdict(int)
        recipe_durations = [r.duration for r in task.recipes]
        for recipe_index, recipe in enumerate(task.recipes):
            for demand, resource in zip(recipe.demands, recipe.resources):
                demand_matrix[(resource, recipe_index)] = demand
        duration_var = model.new_int_var_from_domain(cp_model.Domain.from_values(recipe_durations), f"duration_of_task_{t}")
        for r in all_recipes:
            model.add(duration_var == recipe_durations[r]).only_enforce_if(literals[r])
        task_interval = model.new_optional_interval_var(start_var, duration_var, end_var, is_present, f"task_interval_{t}")
        task_starts[t], task_ends[t], task_durations[t], task_intervals[t], task_to_presence_literals[t] = \
            start_var, end_var, duration_var, task_interval, literals
        for res in all_resources:
            demands = [demand_matrix[(res, recipe)] for recipe in all_recipes]
            task_resource_to_fixed_demands[(t, res)] = demands
            demand_var = model.new_int_var_from_domain(cp_model.Domain.from_values(demands), f"demand_{t}_{res}")
            task_to_resource_demands[t].append(demand_var)
            for r in all_recipes:
                model.add(demand_var == demands[r]).only_enforce_if(literals[r])

    # --- モード間連携の制約 (従来と同じ) ---
    M = num_actual_robots
    num_main_tasks = (len(problem.tasks) - 2) // 3
    for n in range(1, num_main_tasks + 1):
        p_id, w_id, r_id = get_task_ids(n)
        if not all(t in active_tasks for t in [p_id, w_id, r_id]):
            continue
        p_lits, w_lits, r_lits = task_to_presence_literals[p_id], task_to_presence_literals[w_id], task_to_presence_literals[r_id]
        if len(p_lits) == len(r_lits):
            for k in range(len(p_lits)):
                model.add(p_lits[k] == r_lits[k])
        if M > 0 and len(w_lits) > 0:
            for i in range(len(w_lits)):
                model.add(w_lits[i] == sum(p_lits[i * M : (i + 1) * M]))

    # --- メイクスパン定義と先行関係制約 (従来と同じ) ---
    makespan = model.new_int_var(0, horizon, "makespan")
    for t in all_active_tasks:
        for n in problem.tasks[t].successors:
            if n == sink:
                model.add(task_ends[t] <= makespan).only_enforce_if(is_present_literals[t])
            elif n in active_tasks:
                model.add(task_ends[t] <= task_starts[n]).only_enforce_if([is_present_literals[t], is_present_literals[n]])

    # モードに応じてリソース制約と目的関数を定義
    module_capacity_vars = {}
    reservoir_resources = [res for res in all_resources if not problem.resources[res].renewable]

    # モードに寄らず、makespanのリミットを定義
    model.add(makespan <= makespan_limit)
    # --- モジュール数最小化モード ---
    if optimization_mode == 'MINIMIZE_MODULES':
        # Renewableリソース(ロボット, 場所)の制約
        for res in all_resources:
            if problem.resources[res].renewable:
                model.add_cumulative([task_intervals[t] for t in all_active_tasks],
                                     [task_to_resource_demands[t][res] for t in all_active_tasks],
                                     problem.resources[res].max_capacity)

        # Reservoirリソース(モジュール)の制約は、新しいヘルパー関数を呼び出す
        for res in reservoir_resources:
            capacity_var = add_variable_capacity_reservoir_constraint(
                model, res, num_main_tasks, active_tasks, task_starts, task_ends,
                task_to_presence_literals, task_resource_to_fixed_demands, horizon
            )
            module_capacity_vars[res] = capacity_var

        # モジュールは必ず回収される、という制約（消費と供給の合計がゼロ）
        for res in reservoir_resources:
            consumption_terms = [d * lit for t in all_active_tasks for r, lit in enumerate(task_to_presence_literals[t]) if (d := task_resource_to_fixed_demands[(t, res)][r]) != 0]
            if consumption_terms:
                model.add(sum(consumption_terms) == 0)

        # 目的関数: モジュールキャパシティの合計
        total_modules_used = model.new_int_var(0, horizon * num_resources, "total_modules_used")
        if module_capacity_vars:
            model.add(total_modules_used == cp_model.LinearExpr.sum(list(module_capacity_vars.values())))
        else:
            model.add(total_modules_used == 0)
        model.minimize(total_modules_used)

    else: # --- メイクスパン最小化モード (デフォルト) ---
        # 従来通りのロジック
        for res in all_resources:
            resource = problem.resources[res]
            if resource.renewable:
                model.add_cumulative([task_intervals[t] for t in all_active_tasks],
                                     [task_to_resource_demands[t][res] for t in all_active_tasks],
                                     resource.max_capacity)
            else:
                times, demands, actives = [], [], []
                for t in all_active_tasks:
                    for r, d in enumerate(task_resource_to_fixed_demands[(t, res)]):
                        if d != 0:
                            times.append(task_starts[t])
                            demands.append(d)
                            actives.append(task_to_presence_literals[t][r])
                if times:
                    model.AddReservoirConstraintWithActive(times, demands, actives, 0, resource.max_capacity)
        model.minimize(makespan)

    # --- 共通の最終設定 (従来と同じ) ---
    task_starts[source] = model.new_constant(0)
    task_ends[source] = model.new_constant(0)
    task_to_presence_literals[0].append(model.new_constant(1))
    is_present_literals[source] = model.new_constant(1)
    task_starts[sink] = makespan
    task_to_presence_literals[sink].append(model.new_constant(1))
    is_present_literals[sink] = model.new_constant(1)

    # --- ソルバー実行と結果の返却 ---
    if proto_file:
        model.export_to_file(proto_file)
    solver = cp_model.CpSolver()
    if params:
        text_format.Parse(params, solver.parameters)
    solver.parameters.log_search_progress = True
    status = solver.solve(model)
    results = { "solver": solver, "problem": problem, "all_active_tasks": all_active_tasks,
        "all_resources": all_resources, "source": source, "sink": sink,
        "task_starts": task_starts, "task_ends": task_ends, "task_durations": task_durations,
        "task_to_presence_literals": task_to_presence_literals,
        "task_to_resource_demands": task_to_resource_demands,
        "task_resource_to_fixed_demands": task_resource_to_fixed_demands,
        "module_capacities": module_capacity_vars }
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
        renewable_cmap = plt.get_cmap('Blues')
        points = np.linspace(0.4, 0.9, len(renewable_names))
        colors = renewable_cmap(points)
        for name, color in zip(renewable_names, colors):
            resource_color_map[name] = color
    if reservoir_names:
        reservoir_cmap = plt.get_cmap('tab10')
        colors = [reservoir_cmap((i + 1) % 10) for i in range(len(reservoir_names))]
        for name, color in zip(reservoir_names, colors):
            resource_color_map[name] = color

    all_caps_set = set()
    for task in input_data["tasks"]:
        if 'modes' in task:
            for mode in task['modes']:
                if 'required_capabilities' in mode:
                    all_caps_set.update(mode["required_capabilities"].keys())
        elif 'required_capabilities' in task:
            all_caps_set.update(task["required_capabilities"].keys())
    for res_type in ["robot", "module"]:
        for res in input_data["resources"][res_type]:
            all_caps_set.update(res["capabilities"])

    all_caps = sorted(list(all_caps_set))
    capability_cmap = cm.get_cmap('Set3')
    capability_color_map = {cap: capability_cmap(i % 12) for i, cap in enumerate(all_caps)}

    return resource_color_map, capability_color_map


def print_rcpsp_precedence_graph(rcpsp_data_string: str, input_data: dict):
    """
    RCPSP/max形式のデータ文字列を解析し、タスクの先行順序（依存関係グラフ）を
    人間が読みやすい形式で表示します。

    Args:
        rcpsp_data_string (str): generate_rcpsp_max_from_jsonから返されたRCPSP/max形式の文字列。
        input_data (dict): タスク名を取得するための元の入力データ。
    """
    print("\n" + "="*20 + " Human-Readable Precedence Graph " + "="*20)
    print("(P): Placement (W) Work (R) Retrieval")

    # 1. まず、IDからタスク名への変換辞書を作成する
    id_to_name = {}
    tasks = input_data.get('tasks', [])
    N = len(tasks)
    id_to_name[0] = "Start"
    for n in range(1, N + 1):
        task_name = tasks[n - 1]['name']
        p_id, w_id, r_id = 3 * n - 2, 3 * n - 1, 3 * n
        id_to_name[p_id] = f"{task_name}(P)"
        id_to_name[w_id] = f"{task_name}(W)" # Workを追加して明確化
        id_to_name[r_id] = f"{task_name}(R)"
    finish_id = 3 * N + 1
    id_to_name[finish_id] = "Finish"

    # 2. RCPSP/max文字列の先行関係ブロックを解析する
    lines = rcpsp_data_string.strip().split('\n')

    # ヘッダー行から総アクティビティ数を取得
    num_activities = int(lines[0].split()[0])

    # 先行関係が定義されているのは、ヘッダーの次の行から (Start, Activities, Finish) の分
    precedence_lines = lines[1 : 1 + num_activities + 2]

    # 3. 解析結果を分かりやすく表示する
    for line in precedence_lines:
        # リソース定義ブロックの行などをスキップ
        if not line.strip() or not line.strip()[0].isdigit():
            continue

        # 遅延情報 `[...]` を除外してパース
        parts = line.split('[')[0].strip().split()
        if len(parts) < 3:
            continue

        task_id = int(parts[0])
        num_successors = int(parts[2])

        task_name = id_to_name.get(task_id, f"Unknown ID {task_id}")

        print(f"■ {task_name} (ID: {task_id})")

        if num_successors > 0:
            successor_ids = [int(s) for s in parts[3 : 3 + num_successors]]
            for succ_id in successor_ids:
                succ_name = id_to_name.get(succ_id, f"Unknown ID {succ_id}")
                print(f"  └─> {succ_name} (ID: {succ_id})")
        else:
            # 通常はFinishノードのみが該当
            print("  └─> (End of Project)")

    print("=" * 67 + "\n")


def setup_rcpsp_problem(input_data: dict, show_debug_prints=False) -> (rcpsp_pb2.RcpspProblem, dict, dict, dict):
    """
    入力データをRCPSP形式に変換し、ソルバー用の問題オブジェクトをセットアップします。
    """
    try:
        resolved_task_modes = resolve_task_modes(input_data)
    except ValueError as e:
        if show_debug_prints:
            print(e)
        return None, None, None, None

    recipe_to_caps_map = {}
    for task_name, modes in resolved_task_modes.items():
        for i, mode in enumerate(modes):
            recipe_to_caps_map[(task_name, i)] = mode['required_caps']

    rcpsp_data_string, mode_to_resources_map = generate_rcpsp_max_from_json(
        input_data, resolved_task_modes, debug_print=show_debug_prints)
    
    if show_debug_prints:
        print("\n" + "="*25 + " RCPSP/max Data " + "="*25)
        print(rcpsp_data_string)
        print("="*66 + "\n")
        # print_rcpsp_precedence_graph(rcpsp_data_string, input_data)

    rcpsp_parser = rcpsp.RcpspParser()
    with tempfile.NamedTemporaryFile(mode='w+', delete=True, suffix='.sch') as temp_f:
        temp_f.write(rcpsp_data_string)
        temp_f.seek(0)
        rcpsp_parser.parse_file(temp_f.name)

    problem = rcpsp_parser.problem()
    if show_debug_prints:
        print_problem_statistics(problem)

    return problem, mode_to_resources_map, resolved_task_modes, recipe_to_caps_map
