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
import numpy as np

import rcpsp_helpers as h

# =============================================================================
# MODIFICATION: Import Fraction for precise fraction arithmetic and math for gcd
# =============================================================================
from fractions import Fraction
import math


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
        
        # =============================================================================
        # MODIFICATION: Calculate the time scaling factor upon initialization
        # =============================================================================
        self.time_scaling_factor = self._calculate_time_scaling_factor(self.base_input_data)

        # ヘルパー関数を呼び出して、IDと名前のマッピングや表示用のカラーマップを作成
        self.task_id_to_name, self.mode_to_name = h.create_name_mappings(self.base_input_data)
        self.renewable_id_to_name, self.reservoir_id_to_name = h.create_resource_name_mappings(self.base_input_data)
        self.resource_color_map, self.capability_color_map = h.create_color_maps(self.base_input_data)

    def _calculate_time_scaling_factor(self, data: dict) -> float:
        """
        入力データ内のすべての時間関連の値を調査し、それらをすべて整数に変換するための
        最適な（最小の）スケーリング倍率を計算します。

        この倍率は、小数を整数にするための倍率（分母の最小公倍数）と、
        全体を縮小するための倍率（分子の最大公約数の逆数）を組み合わせて決定されます。

        例:
        - [10, 20, 50] -> GCDが10なので、0.1倍を返す
        - [5.5, 11] -> 分母のLCMが2なので、2倍を返す
        - [7.5, 12.5, 5] -> 実質的なGCDが2.5なので、1/2.5 = 0.4倍を返す
        """

        def lcm(a, b):
            """2つの整数の最小公倍数を計算する"""
            return abs(a * b) // math.gcd(a, b) if a != 0 and b != 0 else 0

        time_values = []
        # 辞書からNoneでない時間関連の値をすべて集める
        if data.get("module_handling_time") is not None:
            time_values.append(data["module_handling_time"])

        for task in data.get("tasks", []):
            for mode in task.get("modes", []):
                if mode.get("duration") is not None:
                    time_values.append(mode["duration"])

        # 時間の値がなければ、スケーリングは不要
        if not time_values:
            return 1.0

        # すべての値をFraction（分数）オブジェクトに変換
        fractions = [Fraction(str(v)).limit_denominator() for v in time_values]

        # 1. すべての分母の最小公倍数（LCM）を計算
        # これにより、すべての値が整数になるような共通の分母が得られる
        common_denominator = 1
        for f in fractions:
            common_denominator = lcm(common_denominator, f.denominator)

        # 2. 共通の分母を使って、各値の新しい分子を計算
        # 例: [1/2, 1/4] -> common_denominator=4 -> new_numerators=[2, 1]
        numerators = [f.numerator * (common_denominator // f.denominator) for f in fractions]

        # 3. 新しい分子リストの最大公約数（GCD）を計算
        if not numerators:
            return 1.0

        common_numerator_gcd = numerators[0]
        for i in range(1, len(numerators)):
            common_numerator_gcd = math.gcd(common_numerator_gcd, numerators[i])

        # 4. 最適なスケーリング倍率を計算
        # 倍率 = (共通の分母) / (分子の最大公約数)
        # これにより、スケール後の値が可能な限り小さな整数のセットになる
        if common_numerator_gcd == 0:
            return 1.0

        return float(common_denominator) / float(common_numerator_gcd)

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

        # スケーリング倍率が1.0でない場合にのみ、すべての時間関連の値をスケーリングする
        if self.time_scaling_factor != 1.0:
            if show_results:
                # 上方・下方スケーリングの両方に対応できる一般的なメッセージに変更
                print(f"INFO: Scaling all time values by a factor of {self.time_scaling_factor} for the solver.")
            
            # module_handling_timeのスケーリング
            m_time = current_input_data.get("module_handling_time")
            if m_time is not None:
                current_input_data["module_handling_time"] = round(m_time * self.time_scaling_factor)
            
            # 各タスクのdurationのスケーリング
            for task in current_input_data.get("tasks", []):
                for mode in task.get("modes", []):
                    duration = mode.get("duration")
                    if duration is not None:
                        mode["duration"] = round(duration * self.time_scaling_factor)

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
                time_scaling_factor=self.time_scaling_factor,
                **results,
            )
        elif show_results and status == h.cp_model.INFEASIBLE:
            print("❌ No solution found for the given constraints.")

        if status in (h.cp_model.OPTIMAL, h.cp_model.FEASIBLE):
            solver = results['solver']
            scaled_makespan = solver.value(results['task_starts'][results['sink']])
            # =============================================================================
            # MODIFICATION: De-scale the makespan to its original unit
            # =============================================================================
            original_makespan = scaled_makespan / self.time_scaling_factor
            modules_used = int(solver.objective_value) if optimization_mode == 'MINIMIZE_MODULES' else 0
            return {"status": status, "makespan": original_makespan, "modules": modules_used}
        else:
            return {"status": status, "makespan": float('inf'), "modules": float('inf')}


    def _find_analysis_boundaries(self, module_name: str, show_results: bool = False) -> dict:
        """
        分析に必要な境界条件（モジュール上限、最短メイクスパン、その時のモジュール数）を計算します。
        """
        print(f"\nFinding analysis boundaries for module: '{module_name}'")

        # Step 0: モジュール数の上限を取得
        module_upper_limit = next((m.get("quantity") for m in self.base_input_data.get("resources", {}).get("module", []) if m.get("name") == module_name), None)
        if module_upper_limit is None:
            print(f"  Error: Module '{module_name}' not found. Aborting.")
            return {"success": False}
        print(f"  Initial quantity for module '{module_name}' is {module_upper_limit}.")

        # Step 1: モジュールを最大限使える場合の理論上の最短時間を計算
        print("\n[1/2] Calculating minimum possible makespan...")
        res_min_span = self.solve('MINIMIZE_MAKESPAN', module_quantities={module_name: module_upper_limit}, show_results=show_results)
        if res_min_span["status"] == h.cp_model.INFEASIBLE:
            print("  Error: Could not find a solution even with maximum modules. Aborting.")
            return {"success": False}
        min_makespan = res_min_span["makespan"]
        print(f"    -> Minimum makespan: {min_makespan}")

        # Step 2: Step1で得られた最短時間で実行するために必要なモジュール数を確認
        # NOTE: solveメソッドはスケール済みの整数を期待するため、ここでスケーリングと整数変換を行う
        scaled_limit = int(min_makespan * self.time_scaling_factor)
        print(f"\n[2/2] Calculating modules needed for the minimum makespan of {min_makespan}...")
        res_max_modules = self.solve('MINIMIZE_MODULES', makespan_limit=scaled_limit, module_quantities={module_name: module_upper_limit}, show_results=show_results)
        if res_max_modules["status"] == h.cp_model.INFEASIBLE:
            print(f"  Error: Could not find a solution for makespan {min_makespan}. Aborting.")
            return {"success": False}
        max_modules_needed = min(res_max_modules["modules"], module_upper_limit)
        print(f"    -> Max modules needed for minimum makespan: {max_modules_needed}")

        return {
            "success": True,
            "module_upper_limit": module_upper_limit,
            "min_makespan": min_makespan,
            "max_modules_needed": max_modules_needed
        }

    def analyze_potential(self, module_name: str = "arm", show_results: bool = True) -> float:
        """並列化によるメイクスパン短縮率を計算し、その値を返します。"""
        if show_results:
            print("\n" + "="*5 + " 並列化によるメイクスパン短縮率をスケジューリングに基づいて計算 " + "="*5)

        boundaries = self._find_analysis_boundaries(module_name, show_results=False)
        if not boundaries["success"]:
            if show_results:
                print("分析の境界条件を見つけられなかったため、短縮率を計算できません。")
            return None

        # 解が見つかる最小のモジュール数を探索
        if show_results:
            print("\n[+] 実行可能な解を見つけるための最小モジュール数を探索中...")
        min_module_point = None
        for num_modules in range(boundaries["max_modules_needed"] + 1):
            if show_results:
                print(f"  - {num_modules}個のモジュールで確認中... ", end='', flush=True)
            res = self.solve('MINIMIZE_MAKESPAN', module_quantities={module_name: num_modules}, show_results=False)
            if res["status"] in (h.cp_model.OPTIMAL, h.cp_model.FEASIBLE):
                if show_results:
                    print(f"-> 実行可能な解を発見 (メイクスパン: {res['makespan']})")
                min_module_point = (num_modules, res["makespan"])
                break
            elif show_results:
                print("-> 解なし")

        if min_module_point is None:
            if show_results:
                print("\n  - 比較可能なデータ点が2つ未満のため、短縮率は計算できませんでした。")
            return None

        # 短縮率を計算
        min_module_num, makespan_at_min_modules = min_module_point
        max_module_num, makespan_at_max_modules = boundaries["max_modules_needed"], boundaries["min_makespan"]

        reduction_rate = 0.0
        if makespan_at_min_modules > 0 and makespan_at_min_modules > makespan_at_max_modules:
            reduction_rate = (makespan_at_min_modules - makespan_at_max_modules) / makespan_at_min_modules

        if show_results:
            print("\n[+] 分析モデルとの比較用指標を算出...")
            print(f"  - ベースライン時間 (モジュール{min_module_num}台): {makespan_at_min_modules}")
            print(f"  - 短縮後の時間 (モジュール{max_module_num}台): {makespan_at_max_modules}")
            if reduction_rate > 0:
                print(f"\n並列化によるメイクスパン短縮率: {reduction_rate:.4f} ✨")
                print("(この値は、分析モデルの「並列化ポテンシャルスコア」と比較できます)")
            else:
                print("  - 時間短縮が見られなかったため、短縮率は計算しませんでした。")

        return reduction_rate

    def _calculate_tradeoff_points(self, module_name: str, show_results: bool = False) -> list:
        """
        モジュール数とメイクスパンのトレードオフ関係の全データポイントを計算します。
        """
        boundaries = self._find_analysis_boundaries(module_name, show_results)
        if not boundaries["success"]:
            return []

        max_modules_needed = boundaries["max_modules_needed"]

        print(f"\n[3/3] Calculating minimum makespan for each module count (from 0 to {max_modules_needed})...")
        raw_points = []
        upper_bound_makespan = None
        for num_modules in range(max_modules_needed + 1):
            print(f"  - Calculating for {num_modules} modules (limit: {upper_bound_makespan})... ", end='', flush=True)
            # NOTE: solveメソッドはスケール済みの整数を期待するため、ここでスケーリングと整数変換を行う
            scaled_limit = int(upper_bound_makespan * self.time_scaling_factor) if upper_bound_makespan is not None else None
            res = self.solve('MINIMIZE_MAKESPAN', makespan_limit=scaled_limit, module_quantities={module_name: num_modules}, show_results=show_results)
            if res["status"] in (h.cp_model.OPTIMAL, h.cp_model.FEASIBLE):
                makespan = res["makespan"]
                raw_points.append((num_modules, makespan))
                upper_bound_makespan = makespan
                print(f"-> Makespan: {makespan}")
            else:
                print("-> No solution found.")

        return raw_points

    def analyze_tradeoff(self, module_name: str = "arm", show_results: bool = False):
        """
        Makespanと特定モジュールの必要数のトレードオフ関係を分析し、グラフ化します。
        """
        print("\n" + "="*15 + "    Starting Makespan vs. Module Trade-off Analysis " + "="*15)

        raw_points = self._calculate_tradeoff_points(module_name, show_results=show_results)

        if not raw_points:
            print("\n  No feasible solutions found. Cannot generate a plot.")
            return

        print("\n[+] Preparing data and plotting the results...")
        # NOTE: raw_pointsのmakespanは元のスケールに戻されているため、int()をかけると情報が失われる可能性がある。
        # グラフ描画ではfloatのままで問題ない。
        points_for_plot = [(makespan, num_modules) for num_modules, makespan in raw_points]
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
        step_plot_lines = plt.step(filtered_x, filtered_y, where='post', linestyle='-',
                                   label="Trade-off Boundary")
        plt.scatter(all_x, all_y, marker='o', zorder=3, s=50,
                    label="Per-Module Minimum Makespans")
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

        plt.title('Trade-off: Makespan vs. Required Modules', fontsize=32)
        plt.xlabel('Makespan [s]', fontsize=24)
        plt.ylabel('Required Modules', fontsize=24)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.grid(axis='x', linestyle=':', alpha=0.5)
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend(fontsize=18)
        plt.tight_layout()
        plt.show()


    def analyze_correlation(self, module_name: str, arm_counts: list, handling_times: list):
        """
        アーム台数と脱着時間を変更しながら、ポテンシャルスコアとMakespan短縮率の相関を分析します。
        """
        print(f"\n{'='*10} Starting Correlation Analysis {'='*10}")
        print(f"Target module: '{module_name}'")
        print(f"Arm counts to test: {arm_counts}")
        print(f"Handling times to test: {handling_times}")
        print("-" * 50)

        results = []
        total_iterations = len(arm_counts) * len(handling_times)
        current_iteration = 0

        for count in arm_counts:
            for time in handling_times:
                current_iteration += 1
                print(f"[{current_iteration}/{total_iterations}] Analyzing with {count} arms and handling time {time}...")

                # 毎回ベースデータから新しい設定を作成
                current_input_data = copy.deepcopy(self.base_input_data)
                current_input_data["module_handling_time"] = time
                module_found = False
                for module in current_input_data.get("resources", {}).get("module", []):
                    if module.get("name") == module_name:
                        module["quantity"] = count
                        module_found = True
                        break
                if not module_found:
                    print(f"  Error: Module '{module_name}' not found in config. Skipping.")
                    continue

                # 1. ポテンシャルスコアを計算
                potential_score = h.calculate_potential_score(current_input_data, use_physical_arm_limit=True, verbose=False)

                # 2. Makespan短縮率を計算 (新しい設定で一時的なスケジューラを作成)
                # NOTE: 新しいデータでSchedulerを初期化すると、そのデータに基づいて
                #       time_scaling_factorが自動的に再計算される。
                temp_scheduler = RcpspScheduler(current_input_data)
                reduction_rate = temp_scheduler.analyze_potential(module_name=module_name, show_results=False)

                if reduction_rate is not None:
                    results.append((potential_score, reduction_rate))
                    print(f"  -> Potential Score: {potential_score:.4f}, Reduction Rate: {reduction_rate:.4f}")
                else:
                    print("  -> Could not calculate reduction rate. Skipping point.")

        if len(results) < 2:
            print("\nNot enough data points collected (< 2). Cannot generate plot or calculate correlation.")
            return

        potential_scores, reduction_rates = zip(*results)

        # xとyをnumpy配列に変換
        x_scores = np.array(potential_scores)
        y_rates = np.array(reduction_rates)

        print("\n[+] Collected Data Points (Potential Score vs. Reduction Rate):")
        for i, (score, rate) in enumerate(results):
            print(f"  - Point {i+1:2d}: Score={score:.4f}, Reduction Rate={rate:.4f}")

        # --- 評価指標の計算 ---
        metrics = h.calculate_agreement_metrics(x_scores, y_rates)

        print("\n" + "="*20 + " Analysis Metrics " + "="*20)
        print(f"Lin's Concordance Correlation Coefficient (CCC): {metrics['ccc']:.4f}  <- (相関とy=xのズレを両方考慮するので最も適切な指標。1が完全一致)")
        print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}  <- (y=xからの平均的なズレの大きさ。0に近いほど良い)")
        print(f"Pearson Correlation Coefficient: {metrics['pearson_r']:.4f}  <- (参考: 線形関係の強さ。1/-1に近づくほど正/負の相関)")
        print("=" * 62)

        # --- グラフ描画の改善 ---
        plt.figure(figsize=(10, 8))
        plt.scatter(x_scores, y_rates, alpha=0.7, label='Data Points')

        # y=x の理想線を追加
        min_val = min(plt.xlim()[0], plt.ylim()[0], 0) # Ensure line starts from 0 or less
        max_val = max(plt.xlim()[1], plt.ylim()[1], 1) # Ensure line goes to 1 or more
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Agreement (y=x)')


        plt.title('Potential Score vs. Makespan Reduction Rate', fontsize=16)
        plt.xlabel('Parallelization Potential Score (Analysis Model)', fontsize=12)
        plt.ylabel('Makespan Reduction Rate (Scheduler Result)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axis('equal') # 縦横のスケールを合わせる
        plt.legend()

        # グラフに指標を表示
        metrics_text = (
            f"CCC: {metrics['ccc']:.3f}\n"
            f"RMSE: {metrics['rmse']:.3f}\n"
            f"Pearson's r: {metrics['pearson_r']:.3f}"
        )
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7))

        plt.show()

# =============================================================================
# Main Execution Block
# =============================================================================
def setup_arg_parser():
    """コマンドライン引数を定義し、パーサーオブジェクトを返します。"""
    parser = argparse.ArgumentParser(
        description="RCPSP Scheduler",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- グループ1: 基本引数 (全モード共通) ---
    base_group = parser.add_argument_group('Base Arguments (common to all modes)')
    base_group.add_argument("config_file", type=str, help="Path to the input JSON config file.")
    base_group.add_argument("mode", type=str, choices=["makespan", "modules", "tradeoff", "potential", "correlation"],
                            help="""Execution mode:
- makespan: Minimize the total project time (makespan).
- modules: Minimize the number of modules for a given makespan.
- tradeoff: Analyze the trade-off between makespan and modules.
- potential: Compare the analysis model's potential score with the scheduler's result.
- correlation: Analyze the correlation between potential score and makespan reduction rate.""")
    base_group.add_argument("--module-name", type=str, default="arm",
                            help="Specify the target module name (default: 'arm').")

    # --- グループ2: 上書き用引数 (correlationモード以外) ---
    override_group = parser.add_argument_group('Override Options (for all modes EXCEPT correlation)')
    override_group.add_argument("--arm-count", type=int,
                                help="Override the number of arm modules from the config file.")
    # MODIFICATION: Change type to float to allow decimal values
    override_group.add_argument("--handling-time", type=float,
                                help="Override the module handling time from the config file.")

    # --- グループ3: Correlationモード専用引数 ---
    correlation_group = parser.add_argument_group('Correlation Mode Options (correlation mode ONLY)')
    correlation_group.add_argument("--arm-counts", type=int, nargs='+',
                                   help="List of arm counts to iterate over.")
    # MODIFICATION: Change type to float to allow decimal values
    correlation_group.add_argument("--handling-times", type=float, nargs='+',
                                   help="List of module handling times to iterate over.")
    
    return parser


def load_and_prepare_config(args):
    """設定ファイルを読み込み、コマンドライン引数で設定を上書きします。"""
    try:
        with open(args.config_file, 'r') as f:
            config_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{args.config_file}'")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{args.config_file}'")
        return None

    # モジュール名を取得（引数がなければデフォルト値）
    module_name = args.module_name if args.module_name else "arm"

    # アーム台数の上書き
    if args.arm_count is not None:
        print(f"INFO: Overriding arm count with command-line value: {args.arm_count}")
        # 'resources'や'module'キーが存在しない場合も考慮
        module_list = config_data.setdefault('resources', {}).setdefault('module', [])
        found = False
        for mod in module_list:
            if mod.get("name") == module_name:
                mod['quantity'] = args.arm_count
                found = True
                break
        if not found:
            # モジュール定義がない場合は追加する
            module_list.append({"name": module_name, "quantity": args.arm_count, "capabilities": {"arm": 1}})

    # 脱着時間の上書き
    if args.handling_time is not None:
        print(f"INFO: Overriding handling time with command-line value: {args.handling_time}")
        config_data['module_handling_time'] = args.handling_time

    return config_data, module_name


def execute_mode(scheduler, args, module_name, input_data):
    """解析された引数に基づいて、指定されたモードを実行します。"""
    if args.mode in ("makespan", "modules"):
        # 元の値を取得
        unscaled_limit = input_data.get("makespan_limit")
        scaled_limit = None
        # 値が存在する場合、ここでスケーリングと整数変換を行う
        if unscaled_limit is not None:
            scaled_limit = int(unscaled_limit * scheduler.time_scaling_factor)
        if args.mode == "makespan":
            scheduler.solve(optimization_mode='MINIMIZE_MAKESPAN', show_results=True, makespan_limit=scaled_limit)
        elif args.mode == "modules":
            scheduler.solve(optimization_mode='MINIMIZE_MODULES', show_results=True, makespan_limit=scaled_limit)
    elif args.mode == "tradeoff":
        scheduler.analyze_tradeoff(module_name=module_name, show_results=False)
    elif args.mode == "potential":
        h.calculate_potential_score(input_data, use_physical_arm_limit=True, verbose=True)
        scheduler.analyze_potential(module_name=module_name, show_results=True)
    elif args.mode == "correlation":
        if not args.arm_counts or not args.handling_times:
            # This check is already in main, but kept for safety
            parser.error("--arm-counts and --handling-times are REQUIRED for 'correlation' mode.")
        scheduler.analyze_correlation(
            module_name=module_name,
            arm_counts=args.arm_counts,
            handling_times=args.handling_times
        )


def main():
    """プログラムのエントリーポイント。"""
    # 1. 引数の解析
    parser = setup_arg_parser()
    args = parser.parse_args()

    # 2. 引数の組み合わせを検証
    is_correlation_mode = (args.mode == 'correlation')
    # 'correlation'モードでのみ使える引数が、他のモードで使われていないかチェック
    if not is_correlation_mode and (args.arm_counts is not None or args.handling_times is not None):
        parser.error("--arm-counts and --handling-times can only be used with 'correlation' mode.")
    # 'correlation'モード以外で使える引数が、'correlation'モードで使われていないかチェック
    if is_correlation_mode and (args.arm_count is not None or args.handling_time is not None):
        parser.error("--arm-count and --handling-time cannot be used with 'correlation' mode. Use --arm-counts and --handling-times instead.")
    # 'correlation'モードで必須の引数が存在するかチェック
    if is_correlation_mode and (args.arm_counts is None or args.handling_times is None):
        parser.error("--arm-counts and --handling-times are REQUIRED for 'correlation' mode.")

    # 3. 設定の読み込みと準備
    result = load_and_prepare_config(args)
    if result is None:
        return  # 設定ファイルの読み込みに失敗
    input_data, module_name = result

    # 4. モードの実行
    scheduler = RcpspScheduler(input_data)
    execute_mode(scheduler, args, module_name, input_data)

if __name__ == "__main__":
    main()
