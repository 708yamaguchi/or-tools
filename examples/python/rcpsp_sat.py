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

# main_scheduler.py

import copy
from ortools.sat.python import cp_model
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ヘルパーファイルを "h" という短い名前でインポート
import rcpsp_helpers as h


class RcpspScheduler:
    """
    RCPSP問題を管理し、スケジューリングを実行するためのクラス。
    高レベルなビジネスロジックと実行フローをここに集約します。
    """
    def __init__(self, input_data: dict):
        """
        スケジューラのインスタンスを初期化します。
        """
        self.base_input_data = input_data

        # ヘルパー関数を呼び出して、IDと名前のマッピングや表示用のカラーマップを作成
        self.task_id_to_name, self.mode_to_name = h.create_name_mappings(self.base_input_data)
        self.renewable_id_to_name, self.reservoir_id_to_name = h.create_resource_name_mappings(self.base_input_data)
        self.resource_color_map, self.capability_color_map = h.create_color_maps(self.base_input_data)

    def solve(self, optimization_mode: str, makespan_limit: int = None, module_quantities: dict = None, show_results: bool = True) -> dict:
        """
        単一のスケジューリング問題を解きます。
        """
        if show_results:
            print("\n" + "="*25 + f" Starting New Solve Run " + "="*25)
            print(f"Mode: {optimization_mode}, Makespan Limit: {makespan_limit}, Module Overrides: {module_quantities}")

        current_input_data = copy.deepcopy(self.base_input_data)

        if module_quantities:
            for module in current_input_data["resources"]["module"]:
                if module["name"] in module_quantities:
                    module["quantity"] = module_quantities[module["name"]]

        problem, mode_to_resources_map, resolved_task_modes, recipe_to_caps_map = h.setup_rcpsp_problem(
            current_input_data, show_debug_prints=show_results
        )
        if not problem:
            return {"status": "INVALID_SETUP", "makespan": None, "modules": None}

        num_actual_robots = len(current_input_data["resources"]["robot"])
        last_task = len(problem.tasks) - 1

        status, results = h.solve_rcpsp(
            problem=problem,
            active_tasks=set(range(1, last_task)),
            source=0, sink=last_task,
            num_actual_robots=num_actual_robots,
            optimization_mode=optimization_mode,
            makespan_limit=makespan_limit,
            proto_file="",
            params="",
            verbose=show_results
        )

        if show_results and status in (h.cp_model.OPTIMAL, h.cp_model.FEASIBLE):
            num_actual_modules = len(current_input_data["resources"]["module"])
            h._process_and_display_solution(
                project_name=current_input_data["project_name"],
                task_id_to_name=self.task_id_to_name,
                mode_to_name=self.mode_to_name,
                renewable_id_to_name=self.renewable_id_to_name,
                reservoir_id_to_name=self.reservoir_id_to_name,
                num_actual_robots=num_actual_robots,
                num_actual_modules=num_actual_modules,
                recipe_to_caps_map=recipe_to_caps_map,
                mode_to_resources_map=mode_to_resources_map,
                capability_color_map=self.capability_color_map,
                resource_color_map=self.resource_color_map,
                irreducible_combinations=resolved_task_modes,
                input_data=current_input_data,
                optimization_mode=optimization_mode,
                **results,
            )
        elif show_results and status == h.cp_model.INFEASIBLE:
            print("❌ No solution found for the given constraints.")

        if status in (h.cp_model.OPTIMAL, h.cp_model.FEASIBLE):
            solver = results['solver']
            makespan = solver.value(results['task_starts'][results['sink']])
            modules_used = int(solver.objective_value) if optimization_mode == 'MINIMIZE_MODULES' else 0
            return {"status": status, "makespan": makespan, "modules": modules_used}
        else:
            return {"status": status, "makespan": float('inf'), "modules": float('inf')}

    def analyze_tradeoff(self, module_name: str = "arm_m"):
        """
        Makespan（完了時間）と特定モジュールの必要数のトレードオフ関係を分析し、グラフ化します。

        Args:
            module_name (str): 分析対象とするモジュールの名前。
        """
        print("\n" + "="*15 + "    Starting Makespan vs. Module Trade-off Analysis " + "="*15)

        # Step 0: モジュール数上限を取得
        module_upper_limit = len(self.base_input_data.get("tasks", []))
        if module_upper_limit == 0:
            print("  Error: No tasks found in the input data. Aborting analysis.")
            return

        # Step 1: モジュール無制限時の理論上の最短時間を計算
        print("\n[1/4] Calculating minimum possible makespan (with unlimited modules)...")
        res_min_span = self.solve('MINIMIZE_MAKESPAN',
                                   makespan_limit=None, # 上限なしで真の最短時間を探す
                                   module_quantities={module_name: module_upper_limit},
                                   show_results=False)
        if res_min_span["status"] == h.cp_model.INFEASIBLE:
            print("  Error: Could not find a solution even with unlimited modules. Aborting analysis.")
            return
        min_makespan = res_min_span["makespan"]
        print(f"    Minimum makespan: {min_makespan}")

        # Step 2: Step1で得られた最短時間で実行するために必要なモジュール数を確認
        print(f"\n[2/4] Calculating modules needed for the minimum makespan of {int(min_makespan)}...")
        res_max_modules = self.solve('MINIMIZE_MODULES',
                                     makespan_limit=int(min_makespan),
                                     module_quantities={module_name: module_upper_limit},
                                     show_results=False)
        if res_max_modules["status"] == h.cp_model.INFEASIBLE:
            print(f"  Error: Could not find a solution for makespan {int(min_makespan)}. This should not happen. Aborting.")
            return
        max_modules_needed = res_max_modules["modules"]
        print(f"    Max modules needed for minimum makespan: {max_modules_needed}")

        # Step 3: モジュール数を0から順に増やし、makespanを計算 (旧Step3とStep4を統合)
        # 前回のmakespanを次の上限として利用し、探索を効率化
        print(f"\n[3/4] Calculating minimum makespan for each module count (from 0 to {max_modules_needed})...")
        raw_points = []  # (num_modules, makespan) のペアを格納
        upper_bound_makespan = None # 初回の探索では上限は設定しない

        for num_modules in range(max_modules_needed + 1):
            print(f"  - Calculating for {num_modules} module(s)...", end='', flush=True)

            res = self.solve('MINIMIZE_MAKESPAN',
                               makespan_limit=upper_bound_makespan, # 計算済みのmakespanを上限として設定
                               module_quantities={module_name: num_modules},
                               show_results=False)

            if res["status"] in (h.cp_model.OPTIMAL, h.cp_model.FEASIBLE):
                makespan = int(res["makespan"])
                raw_points.append((num_modules, makespan))

                # 見つかったmakespanを次の探索の上限として更新する
                upper_bound_makespan = makespan
                print(f" -> Achieved makespan: {makespan}. Set as new upper bound.")
            else:
                print(" -> No solution found.")
                # 解が見つからない場合でも、上限は維持したまま次のモジュール数へ進む

        if not raw_points:
            print("\n  No feasible solutions found during the analysis. Cannot generate a plot.")
            return

        # Step 4: 描画データの準備とグラフ描画 (旧Step5)
        print("\n[4/4] Preparing data and plotting the results...")

        makespan_to_min_module = {}
        # makespanが小さい順、次にモジュール数が小さい順でソート
        sorted_raw_points = sorted(raw_points, key=lambda x: (x[1], x[0]))

        for num_modules, makespan in sorted_raw_points:
            # 同じmakespanを達成できる、より少ないモジュール数の結果を優先する
            if makespan not in makespan_to_min_module:
                   makespan_to_min_module[makespan] = num_modules

        tradeoff_points = sorted(makespan_to_min_module.items())

        self._plot_tradeoff_graph(
            tradeoff_points,
            module_name,
        )

    def _plot_tradeoff_graph(self, points: list, module_name: str):
        """分析結果をグラフにプロットします。"""
        if not points:
            print("\nNo data points to plot for the trade-off curve.")
            return

        print("\n--- Trade-off Analysis Results ---")
        print("Makespan -> Min Modules")
        for p in reversed(points):
            print(f"  {p[0]:<8} -> {p[1]}")

        points.sort()
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]

        plt.figure(figsize=(12, 7))
        plt.step(x_vals, y_vals, where='post', marker='o', linestyle='-')

        plt.title(f'Trade-off: Makespan vs. Required "{module_name}" Modules', fontsize=16)
        plt.xlabel('Allowed Project Makespan (Time)', fontsize=12)
        plt.ylabel(f'Minimum Required "{module_name}" Modules', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.grid(axis='x', linestyle=':', alpha=0.5)
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.show()


# =============================================================================
# Main Execution Block
# =============================================================================

def main(_):
    # --- 1. Define Input Data ---
    # input_data = {
    #     "project_name": "Cooking",
    #     "makespan_limit": 200,
    #     "module_handling_time": 5,
    #     "locations": [
    #         {"name": "610", "max_robots": 99},
    #     ],
    #     "resources": {
    #         "robot": [
    #             {"name": "r8_r", "quantity": 1, "capabilities": {"arm": 2}},
    #             # {"name": "r8_r", "quantity": 1, "capabilities": {"arm": 0}},
    #         ],
    #         "module": [
    #             {"name": "arm_m", "quantity": 10, "capabilities": {"arm": 1}},
    #         ]
    #     },
    #     "tasks": [
    #         {"name": "cooking XXX", "location": "610",
    #          "modes": [
    #              {"duration": 20, "required_capabilities": {"arm": 3}},
    #              {"duration": 30, "required_capabilities": {"arm": 2}},
    #              {"duration": 45, "required_capabilities": {"arm": 1}},
    #          ]},
    #         {"name": "cooking YYY", "location": "610", "predecessors": ["cooking XXX"],
    #          "modes": [
    #              {"duration": 30, "required_capabilities": {"arm": 2}},
    #          ]},
    #         {"name": "cooking ZZZ", "location": "610", "predecessors": ["cooking YYY"],
    #          "modes": [
    #              {"duration": 10, "required_capabilities": {"arm": 2}},
    #              {"duration": 25, "required_capabilities": {"arm": 1,}},
    #          ]},
    #         {"name": "cooking AAA", "location": "610", "predecessors": ["cooking ZZZ"],
    #          "modes": [
    #              {"duration": 30, "required_capabilities": {"arm": 1}},
    #          ]},
    #         {"name": "cooking BBB", "location": "610", "predecessors": ["cooking AAA"],
    #          "modes": [
    #              {"duration": 30, "required_capabilities": {"arm": 1}},
    #          ]},
    #         {"name": "cooking CCC", "location": "610", # "predecessors": ["cooking BBB"],
    #          "modes": [
    #              {"duration": 30, "required_capabilities": {"arm": 1}},
    #          ]},
    #         {"name": "cooking DDD", "location": "610", "predecessors": ["cooking CCC"],
    #          "modes": [
    #              {"duration": 10, "required_capabilities": {"arm": 1}},
    #          ]},
    #         {"name": "cooking EEE", "location": "610", "predecessors": ["cooking DDD"],
    #          "modes": [
    #              {"duration": 10, "required_capabilities": {"arm": 1}},
    #          ]},
    #         {"name": "cooking FFF", "location": "610", "predecessors": ["cooking EEE"],
    #          "modes": [
    #              {"duration": 10, "required_capabilities": {"arm": 1}},
    #          ]},
    #         {"name": "cooking GGG", "location": "610", "predecessors": ["cooking FFF"],
    #          "modes": [
    #              {"duration": 10, "required_capabilities": {"arm": 1}},
    #          ]},

    #     ]
    # }

    input_data = {
        "project_name": "Clean 602 and 610",
        "makespan_limit": 290,
        "module_handling_time": 5,
        "locations": [
            {"name": "602", "max_robots": 99},
            {"name": "610", "max_robots": 99},
        ],
        "resources": {
            "robot": [
                {"name": "r8_r", "quantity": 1, "capabilities": {"arm": 2}},
            ],
            "module": [
                {"name": "arm_m", "quantity": 10, "capabilities": {"arm": 1}},
            ]
        },
        "tasks": [
            {"name": "clean XXX", "location": "602",
             "modes": [
                 {"duration": 20, "required_capabilities": {"arm": 3}},
                 {"duration": 30, "required_capabilities": {"arm": 2}},
                 {"duration": 45, "required_capabilities": {"arm": 1,}},
             ]},
            {"name": "clean YYY", "location": "610",
             "modes": [
                 {"duration": 30, "required_capabilities": {"arm": 3}},
             ]},
            {"name": "clean ZZZ", "location": "610",
             "modes": [
                 {"duration": 10, "required_capabilities": {"arm": 2}},
                 {"duration": 25, "required_capabilities": {"arm": 1,}},
             ]},
            {"name": "clean AAA", "location": "610",
             "modes": [
                 {"duration": 30, "required_capabilities": {"arm": 1}},
             ]},
            {"name": "clean BBB", "location": "610",
             "modes": [
                 {"duration": 30, "required_capabilities": {"arm": 1}},
             ]},
            {"name": "clean CCC", "location": "610",
             "modes": [
                 {"duration": 30, "required_capabilities": {"arm": 1}},
             ]},
            {"name": "clean DDD", "location": "610",
             "modes": [
                 {"duration": 30, "required_capabilities": {"arm": 1}},
             ]},
            {"name": "clean EEE", "location": "610",
             "modes": [
                 {"duration": 30, "required_capabilities": {"arm": 1}},
             ]},
            {"name": "clean FFF", "location": "610",
             "modes": [
                 {"duration": 30, "required_capabilities": {"arm": 1}},
             ]},
            {"name": "clean GGG", "location": "610",
             "modes": [
                 {"duration": 30, "required_capabilities": {"arm": 1}},
             ]},

        ]
    }


    # --- 2. Initialize Scheduler ---
    scheduler = RcpspScheduler(input_data)

    # --- 3. Choose Execution Mode ---
    # mode = "SINGLE_RUN_MAKESPAN"
    # mode = "SINGLE_RUN_MODULES"
    mode = "TRADEOFF_ANALYSIS"

    # --- 4. Run Selected Mode ---
    if mode == "SINGLE_RUN_MAKESPAN":
        scheduler.solve(optimization_mode='MINIMIZE_MAKESPAN', show_results=True)
    elif mode == "SINGLE_RUN_MODULES":
        scheduler.solve(optimization_mode='MINIMIZE_MODULES', makespan_limit=input_data["makespan_limit"], show_results=True)
    elif mode == "TRADEOFF_ANALYSIS":
        scheduler.analyze_tradeoff(module_name="arm_m")


if __name__ == "__main__":
    # OR-Toolsのフラグなどを処理するためにapp.runを使用
    # この呼び出しはヘルパーファイルからインポートしたappオブジェクトを使います
    h.app.run(main)
