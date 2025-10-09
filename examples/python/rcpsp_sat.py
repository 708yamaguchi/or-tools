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

import argparse
import json

import copy
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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
            print("\n" + "="*25 + " Starting New Solve Run " + "="*25)
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

    def analyze_tradeoff(self, module_name: str = "arm_m", show_results: bool = False):
        """
        Makespan（完了時間）と特定モジュールの必要数のトレードオフ関係を分析し、グラフ化します。

        Args:
            module_name (str): 分析対象とするモジュールの名前。
        """
        print("\n" + "="*15 + "    Starting Makespan vs. Module Trade-off Analysis " + "="*15)

        # Step 0: モジュール数上限を取得
        module_upper_limit = None
        # 'resources'や'module'キーが存在しない場合も考慮し、安全にリストを取得
        modules_list = self.base_input_data.get("resources", {}).get("module", [])
        # モジュールリストをループして、指定された名前のモジュールを探す
        for module in modules_list:
            if module.get("name") == module_name:
                module_upper_limit = module.get("quantity")
                break  # 対象モジュールが見つかったのでループを終了
        # 対象モジュールが入力データに見つからなかった場合はエラーとして処理を中断
        if module_upper_limit is None:
            print(f"  Error: Module '{module_name}' not found in the input data's resources. Aborting analysis.")
            return
        print(f"  Initial quantity for module '{module_name}' is {module_upper_limit}.")

        # Step 1: モジュールを最大限使える場合の理論上の最短時間を計算
        print("\n[1/4] Calculating minimum possible makespan...")
        res_min_span = self.solve('MINIMIZE_MAKESPAN',
                                   makespan_limit=None, # 上限なしで真の最短時間を探す
                                   module_quantities={module_name: module_upper_limit},
                                   show_results=show_results)
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
                                     show_results=show_results)
        if res_max_modules["status"] == h.cp_model.INFEASIBLE:
            print(f"  Error: Could not find a solution for makespan {int(min_makespan)}. This should not happen. Aborting.")
            return
        max_modules_needed = res_max_modules["modules"]
        max_modules_needed = min(max_modules_needed, module_upper_limit)
        print(f"    Max modules needed for minimum makespan: {max_modules_needed}")
        # Step1で設定したモジュール数の上限に達した場合に警告を出力
        if max_modules_needed == module_upper_limit:
            print("\n  ⚠️ WARNING: The number of modules required to achieve the minimum makespan "
                  f"is equal to the initial upper limit ({module_upper_limit}).")
            print("              This may indicate that the 'quantity' for this module is a bottleneck. "
                  "A shorter makespan might be achievable if more modules were available.")

        # Step 3: モジュール数を0から順に増やし、makespanを計算 (旧Step3とStep4を統合)
        # 前回のmakespanを次の上限として利用し、探索を効率化
        print(f"\n[3/4] Calculating minimum makespan for each module count (from 0 to {max_modules_needed})...")
        raw_points = []  # (num_modules, makespan) のペアを格納
        upper_bound_makespan = None # 初回の探索では上限は設定しない
        for num_modules in range(max_modules_needed + 1):
            print(f"  - Calculating for {num_modules} modules with makespan limit {upper_bound_makespan} ... ", end='', flush=True)
            res = self.solve('MINIMIZE_MAKESPAN',
                               makespan_limit=upper_bound_makespan, # 計算済みのmakespanを上限として設定
                               module_quantities={module_name: num_modules},
                               show_results=show_results)
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

        # Step 4: 描画データの準備とグラフ描画
        print("\n[4/4] Preparing data and plotting the results...")

        # モジュール数が小さい順、次にmakespanが小さい順でソート
        sorted_raw_points = sorted(raw_points, key=lambda x: (x[0], x[1]))
        points_for_plot = [(makespan, num_modules) for num_modules, makespan in sorted_raw_points]
        self._plot_tradeoff_graph(points_for_plot, module_name)

    def _plot_tradeoff_graph(self, points: list, module_name: str):
        """
        最適なトレードオフ曲線（ステップ）と全てのデータ点（散布図）を重ねてプロットします。
        """
        if not points:
            print("\nNo data points to plot.")
            return

        # 全てのデータポイントを準備 (散布図用)
        all_x = [p[0] for p in points]
        all_y = [p[1] for p in points]
        # 最適な点のみをフィルタリング (ステッププロット用)
        unique_points = {}
        for x, y in points:
            if x not in unique_points or y < unique_points[x]:
                unique_points[x] = y

        filtered_points = sorted(list(unique_points.items()))
        filtered_x = [p[0] for p in filtered_points]
        filtered_y = [p[1] for p in filtered_points]

        # グラフの描画
        plt.figure(figsize=(12, 7))
        # ステッププロットで「最適なトレードオフ曲線」を描画
        step_plot_lines = plt.step(filtered_x, filtered_y, where='post', linestyle='-', label='Optimal Trade-off')
        # 散布図で「全てのデータ点」を描画。zorder=3 で線より手前に点を表示, s=50でマーカーサイズを調整
        plt.scatter(all_x, all_y, marker='o', zorder=3, s=50, label='All Data Points')
        # グラフの上側と右側に直線を外挿
        min_x = min(p[0] for p in points)
        points_at_min_x = [p for p in points if p[0] == min_x]
        start_point_up = max(points_at_min_x, key=lambda p: p[1])
        max_x = max(p[0] for p in points)
        points_at_max_x = [p for p in points if p[0] == max_x]
        start_point_right = min(points_at_max_x, key=lambda p: p[1])
        line_color = step_plot_lines[0].get_color()
        _, xmax = plt.xlim()
        _, ymax = plt.ylim()
        plt.plot([start_point_up[0], start_point_up[0]], [start_point_up[1], ymax], linestyle='-', color=line_color)
        plt.plot([start_point_right[0], xmax], [start_point_right[1], start_point_right[1]], linestyle='-', color=line_color)

        plt.title(f'Trade-off: Makespan vs. Required "{module_name}" Modules', fontsize=16)
        plt.xlabel('Allowed Project Makespan (Time)', fontsize=12)
        plt.ylabel(f'Minimum Required "{module_name}" Modules', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.grid(axis='x', linestyle=':', alpha=0.5)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.tight_layout()
        plt.show()

# =============================================================================
# Main Execution Block
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="RCPSP Scheduler")
    parser.add_argument("config_file", type=str, help="Path to the input JSON config file.")
    parser.add_argument("mode", type=str, choices=["makespan", "modules", "tradeoff"],
                        help="Execution mode: 'makespan', 'modules', or 'tradeoff'.")
    args = parser.parse_args()

    try:
        with open(args.config_file, 'r') as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{args.config_file}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{args.config_file}'")
        return

    h.calculate_and_print_potential_details(input_data)

    scheduler = RcpspScheduler(input_data)
    if args.mode == "makespan":
        scheduler.solve(optimization_mode='MINIMIZE_MAKESPAN',
                        show_results=True)
    elif args.mode == "modules":
        scheduler.solve(optimization_mode='MINIMIZE_MODULES',
                        show_results=True,
                        makespan_limit=input_data.get("makespan_limit"))
    elif args.mode == "tradeoff":
        scheduler.analyze_tradeoff(module_name="arm_m", show_results=False)


if __name__ == "__main__":
    main()
