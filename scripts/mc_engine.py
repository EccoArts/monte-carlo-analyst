"""
Высокопроизводительный движок вероятностного анализа методом Монте-Карло.
Поддерживает три ключевых режима:
1. flow    - Прогнозирование сроков и пропускной способности (XMR-карты Шухарта + MC-When / MC-How-Many)
2. pert    - Анализ рисков расписания проектов и критического пути (3-точечные оценки PERT, бета-распределение, граф зависимостей)
3. finance - Финансовое моделирование рисков (доходы, расходы, маржинальность, VaR, вероятность кассового разрыва)
"""

import sys
import json
import math
import random
import argparse
from typing import Dict, List, Any, Optional

# Принудительная установка UTF-8 для консольного вывода на Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Попытка импорта numpy для векторного ускорения
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ И СТАТИСТИКА
# ==============================================================================

def calculate_percentiles(data: List[float], percentiles: List[int]) -> Dict[str, float]:
    """Расчет квантилей (P50, P70, P85, P95)"""
    if not data:
        return {}
    if HAS_NUMPY:
        arr = np.array(data)
        return {f"P{p}": float(np.percentile(arr, p)) for p in percentiles}
    sorted_data = sorted(data)
    n = len(sorted_data)
    results = {}
    for p in percentiles:
        idx = int(math.ceil((p / 100.0) * n)) - 1
        idx = max(0, min(n - 1, idx))
        results[f"P{p}"] = float(sorted_data[idx])
    return results


def sample_pert_beta(o: float, m: float, p: float, n_samples: int = 1) -> List[float]:
    """
    Генерация выборки по бета-распределению PERT (Optimistic, Most Likely, Pessimistic).
    alpha = 1 + 4*(m - o)/(p - o)
    beta = 1 + 4*(p - m)/(p - o)
    """
    if p <= o:
        return [o] * n_samples
    if m < o or m > p:
        m = (o + p) / 2.0

    range_span = p - o
    alpha = 1.0 + 4.0 * (m - o) / range_span
    beta = 1.0 + 4.0 * (p - m) / range_span

    if HAS_NUMPY:
        samples = np.random.beta(alpha, beta, size=n_samples)
        return list(o + samples * range_span)
    else:
        results = []
        for _ in range(n_samples):
            val = random.betavariate(alpha, beta)
            results.append(o + val * range_span)
        return results


def sample_distribution(dist_type: str, params: Dict[str, Any], n_samples: int) -> List[float]:
    """Универсальный генератор случайных величин для финансовых и проектных моделей"""
    dist_type = dist_type.lower()
    if dist_type == "constant" or dist_type == "fixed":
        val = float(params.get("value", 0.0))
        return [val] * n_samples

    if dist_type == "pert":
        o = float(params.get("optimistic", params.get("min", 0.0)))
        m = float(params.get("most_likely", params.get("mode", 0.0)))
        p = float(params.get("pessimistic", params.get("max", 0.0)))
        return sample_pert_beta(o, m, p, n_samples)

    if dist_type == "triangular":
        left = float(params.get("min", 0.0))
        mode = float(params.get("mode", (left + float(params.get("max", 1.0))) / 2.0))
        right = float(params.get("max", 1.0))
        if HAS_NUMPY:
            return list(np.random.triangular(left, mode, right, size=n_samples))
        return [random.triangular(left, right, mode) for _ in range(n_samples)]

    if dist_type == "normal":
        mean = float(params.get("mean", 0.0))
        std = float(params.get("std", 1.0))
        if HAS_NUMPY:
            return list(np.random.normal(mean, std, size=n_samples))
        return [random.gauss(mean, std) for _ in range(n_samples)]

    if dist_type == "lognormal":
        mean = float(params.get("mean", 0.0))
        std = float(params.get("std", 1.0))
        if HAS_NUMPY:
            return list(np.random.lognormal(mean, std, size=n_samples))
        return [random.lognormvariate(mean, std) for _ in range(n_samples)]

    if dist_type == "uniform":
        low = float(params.get("min", 0.0))
        high = float(params.get("max", 1.0))
        if HAS_NUMPY:
            return list(np.random.uniform(low, high, size=n_samples))
        return [random.uniform(low, high) for _ in range(n_samples)]

    raise ValueError(f"Неизвестный тип распределения: {dist_type}")


# ==============================================================================
# РЕЖИМ 1: FLOW (XMR-карты + симуляция пропускной способности)
# ==============================================================================

def run_flow_simulation(data: Dict[str, Any], iterations: int = 10000) -> Dict[str, Any]:
    """
    Анализ контрольной карты Шухарта (XMR) и симуляция сроков/объема (MC-When / MC-How-Many)
    """
    throughput_history = [max(0.0, float(x)) for x in data.get("history", [])]
    if len(throughput_history) < 3:
        raise ValueError("Для анализа XMR и потока требуется минимум 3 исторических значения пропускной способности.")
    if sum(throughput_history) <= 0:
        raise ValueError("Сумма исторической пропускной способности должна быть строго больше 0.")

    n_hist = len(throughput_history)
    mean_x = sum(throughput_history) / n_hist

    # Расчет скользящего размаха (Moving Range)
    moving_ranges = [abs(throughput_history[i] - throughput_history[i - 1]) for i in range(1, n_hist)]
    mean_mr = sum(moving_ranges) / len(moving_ranges)

    # Естественные границы процесса (UNPL, LNPL, URL)
    unpl = mean_x + 2.66 * mean_mr
    lnpl = max(0.0, mean_x - 2.66 * mean_mr)
    url = 3.27 * mean_mr

    # Проверка на статистическую предсказуемость процесса
    out_of_control = [
        {"period": i + 1, "value": x, "reason": "Превышение UNPL" if x > unpl else "Ниже LNPL"}
        for i, x in enumerate(throughput_history) if x > unpl or x < lnpl
    ]
    is_predictable = len(out_of_control) == 0

    results = {
        "mode": "flow",
        "iterations": iterations,
        "xmr_chart": {
            "mean_throughput": round(mean_x, 2),
            "mean_moving_range": round(mean_mr, 2),
            "unpl": round(unpl, 2),
            "lnpl": round(lnpl, 2),
            "url": round(url, 2),
            "is_predictable": is_predictable,
            "anomalies": out_of_control
        }
    }

    # Режим MC-When: сколько периодов (дней/спринтов/недель) потребуется для сдачи backlog_items
    backlog_items = data.get("backlog_items")
    if backlog_items is not None and backlog_items > 0:
        periods_needed = []
        if HAS_NUMPY:
            hist_arr = np.array(throughput_history)
            for _ in range(iterations):
                cumsum = 0
                steps = 0
                while cumsum < backlog_items:
                    # Случайный выбор из истории с возвращением (bootstrap)
                    batch = np.random.choice(hist_arr, size=10)
                    for val in batch:
                        cumsum += val
                        steps += 1
                        if cumsum >= backlog_items:
                            break
                periods_needed.append(steps)
        else:
            for _ in range(iterations):
                cumsum = 0
                steps = 0
                while cumsum < backlog_items:
                    cumsum += random.choice(throughput_history)
                    steps += 1
                periods_needed.append(steps)

        when_percentiles = calculate_percentiles(periods_needed, [50, 70, 85, 95])
        results["mc_when"] = {
            "backlog_items": backlog_items,
            "percentiles_periods": when_percentiles,
            "summary": (
                f"С вероятностью 50% задачи будут закрыты за {when_percentiles['P50']} периодов; "
                f"с вероятностью 85% — за {when_percentiles['P85']} периодов; "
                f"с вероятностью 95% — за {when_percentiles['P95']} периодов."
            )
        }

    # Режим MC-How-Many: сколько задач успеем сделать за target_periods
    target_periods = data.get("target_periods")
    if target_periods is not None and target_periods > 0:
        delivered_items = []
        if HAS_NUMPY:
            hist_arr = np.array(throughput_history)
            samples = np.random.choice(hist_arr, size=(iterations, target_periods))
            delivered_items = list(np.sum(samples, axis=1))
        else:
            for _ in range(iterations):
                tot = sum(random.choice(throughput_history) for _ in range(target_periods))
                delivered_items.append(tot)

        # Для гарантии 'не менее N задач с вероятностью C%' берем квантиль (100 - C)
        how_many_percentiles = calculate_percentiles(delivered_items, [5, 15, 30, 50])
        results["mc_how_many"] = {
            "target_periods": target_periods,
            "percentiles_items": {
                "P50": how_many_percentiles["P50"],
                "P85": how_many_percentiles["P15"],
                "P95": how_many_percentiles["P5"],
            },
            "summary": (
                f"За {target_periods} периодов с уверенностью 50% будет сделано ≥ {how_many_percentiles['P50']:.0f} задач; "
                f"с уверенностью 85% — ≥ {how_many_percentiles['P15']:.0f} задач; "
                f"с уверенностью 95% — ≥ {how_many_percentiles['P5']:.0f} задач."
            )
        }

    return results


# ==============================================================================
# РЕЖИМ 2: PERT (Граф зависимостей, расписание, критический путь)
# ==============================================================================

def run_pert_simulation(data: Dict[str, Any], iterations: int = 10000) -> Dict[str, Any]:
    """
    Симуляция расписания проекта по PERT с учетом графа предшественников
    """
    tasks_input = data.get("tasks", [])
    if not tasks_input:
        raise ValueError("Входные данные должны содержать список задач 'tasks'.")

    # Топологическая сортировка задач
    tasks_dict = {t["id"]: t for t in tasks_input}
    visited = {}
    order = []

    def dfs(t_id):
        if visited.get(t_id) == 1:
            raise ValueError(f"Обнаружен циклический граф зависимостей на задаче: {t_id}")
        if visited.get(t_id) == 2:
            return
        visited[t_id] = 1
        for pred_id in tasks_dict[t_id].get("depends_on", []):
            if pred_id in tasks_dict:
                dfs(pred_id)
        visited[t_id] = 2
        order.append(t_id)

    for t in tasks_input:
        if visited.get(t["id"]) != 2:
            dfs(t["id"])

    # Предварительная генерация выборок длительностей для каждой задачи
    task_samples = {}
    task_pert_expected = {}
    task_pert_variance = {}

    for t_id in order:
        t = tasks_dict[t_id]
        o = float(t.get("optimistic", t.get("min", 1.0)))
        m = float(t.get("most_likely", t.get("mode", o)))
        p = float(t.get("pessimistic", t.get("max", m)))
        task_samples[t_id] = sample_pert_beta(o, m, p, iterations)
        te = (o + 4.0 * m + p) / 6.0
        v = ((p - o) / 6.0) ** 2
        task_pert_expected[t_id] = te
        task_pert_variance[t_id] = v

    # Итерации Монте-Карло для поиска времени завершения и критичности задач
    project_durations = [0.0] * iterations
    criticality_counts = {t_id: 0 for t_id in order}

    for it in range(iterations):
        early_start = {}
        early_finish = {}

        for t_id in order:
            preds = tasks_dict[t_id].get("depends_on", [])
            es = max([early_finish[p_id] for p_id in preds], default=0.0)
            dur = task_samples[t_id][it]
            ef = es + dur
            early_start[t_id] = es
            early_finish[t_id] = ef

        proj_len = max(early_finish.values(), default=0.0)
        project_durations[it] = proj_len

        # Трассировка критического пути назад от узлов с максимальным finish
        late_finish = {}
        late_start = {}
        for t_id in reversed(order):
            succs = [s_id for s_id in order if t_id in tasks_dict[s_id].get("depends_on", [])]
            lf = min([late_start[s_id] for s_id in succs], default=proj_len)
            dur = task_samples[t_id][it]
            ls = lf - dur
            late_finish[t_id] = lf
            late_start[t_id] = ls

            # Если резерв времени (slack) близок к 0, задача на критическом пути
            if abs(lf - early_finish[t_id]) < 1e-4:
                criticality_counts[t_id] += 1

    dur_percentiles = calculate_percentiles(project_durations, [50, 70, 85, 95])
    criticality_index = {
        t_id: round((criticality_counts[t_id] / iterations) * 100, 1)
        for t_id in order
    }

    # Сортировка задач по критичности
    top_critical_tasks = sorted(
        [
            {
                "id": t_id,
                "name": tasks_dict[t_id].get("name", t_id),
                "criticality_pct": criticality_index[t_id],
                "pert_expected": round(task_pert_expected[t_id], 2)
            }
            for t_id in order
        ],
        key=lambda x: x["criticality_pct"],
        reverse=True
    )

    p50 = dur_percentiles["P50"]
    p85 = dur_percentiles["P85"]
    contingency_reserve = max(0.0, p85 - p50)

    return {
        "mode": "pert",
        "iterations": iterations,
        "percentiles_duration": dur_percentiles,
        "contingency_reserve_p85": round(contingency_reserve, 2),
        "tasks_criticality": top_critical_tasks,
        "summary": (
            f"Базовый срок (P50) составляет {p50:.2f} ед. "
            f"Для уверенности 85% требуется срок {p85:.2f} ед. (буфер безопасности / резерв: {contingency_reserve:.2f} ед.). "
            f"Максимальный риск задержки несет задача: '{top_critical_tasks[0]['name']}' "
            f"(индекс критичности {top_critical_tasks[0]['criticality_pct']}%)."
        )
    }


# ==============================================================================
# РЕЖИМ 3: FINANCE (Моделирование рисков, кассовых разрывов, VaR)
# ==============================================================================

def run_finance_simulation(data: Dict[str, Any], iterations: int = 10000) -> Dict[str, Any]:
    """
    Симуляция финансовой модели: доходы, расходы, чистая прибыль, VaR, вероятность убытка
    """
    revenue_items = data.get("revenues", [])
    cost_items = data.get("costs", [])

    if not revenue_items and not cost_items:
        raise ValueError("Финансовая модель должна содержать хотя бы один источник в 'revenues' или 'costs'.")

    # Генерация выборок для всех статей доходов
    total_revenues = [0.0] * iterations
    revenue_breakdown = {}
    for item in revenue_items:
        name = item.get("name", "Revenue")
        dist = item.get("distribution", "pert")
        samples = sample_distribution(dist, item, iterations)
        revenue_breakdown[name] = samples
        for i in range(iterations):
            total_revenues[i] += samples[i]

    # Генерация выборок для всех статей расходов
    total_costs = [0.0] * iterations
    cost_breakdown = {}
    for item in cost_items:
        name = item.get("name", "Cost")
        dist = item.get("distribution", "pert")
        samples = sample_distribution(dist, item, iterations)
        cost_breakdown[name] = samples
        for i in range(iterations):
            total_costs[i] += samples[i]

    # Расчет чистой прибыли (Net Profit / Cash Flow)
    net_profit = [total_revenues[i] - total_costs[i] for i in range(iterations)]

    profit_percentiles = calculate_percentiles(net_profit, [5, 15, 50, 85, 95])
    rev_percentiles = calculate_percentiles(total_revenues, [50, 85])
    cost_percentiles = calculate_percentiles(total_costs, [50, 85])

    # Вероятность убытка (P(Profit < 0))
    loss_count = sum(1 for p in net_profit if p < 0)
    prob_loss = (loss_count / iterations) * 100.0

    # Value at Risk (VaR 95%): максимальный вероятный убыток или отклонение на уровне 5-го процентиля
    var_95 = -profit_percentiles["P5"] if profit_percentiles["P5"] < 0 else 0.0

    # Чувствительность параметров (корреляция Пирсона с чистой прибылью)
    def _pearson_corr(x: List[float], y: List[float]) -> Optional[float]:
        if HAS_NUMPY:
            x_arr = np.array(x)
            y_arr = np.array(y)
            if float(np.std(x_arr)) <= 1e-12 or float(np.std(y_arr)) <= 1e-12:
                return None
            c = float(np.corrcoef(x_arr, y_arr)[0, 1])
            return None if math.isnan(c) else c
        n = len(x)
        if n < 2:
            return None
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        var_x = sum((xi - mean_x) ** 2 for xi in x)
        var_y = sum((yi - mean_y) ** 2 for yi in y)
        if var_x <= 1e-12 or var_y <= 1e-12:
            return None
        return cov / math.sqrt(var_x * var_y)

    sensitivity = []
    for name, samples in revenue_breakdown.items():
        corr = _pearson_corr(samples, net_profit)
        if corr is not None:
            sensitivity.append({"parameter": f"[Доход] {name}", "correlation": round(corr, 3)})
    for name, samples in cost_breakdown.items():
        corr = _pearson_corr(samples, net_profit)
        if corr is not None:
            sensitivity.append({"parameter": f"[Расход] {name}", "correlation": round(corr, 3)})
    sensitivity.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    return {
        "mode": "finance",
        "iterations": iterations,
        "net_profit_percentiles": profit_percentiles,
        "revenue_percentiles": rev_percentiles,
        "cost_percentiles": cost_percentiles,
        "probability_of_loss_pct": round(prob_loss, 2),
        "value_at_risk_95": round(var_95, 2),
        "sensitivity": sensitivity,
        "summary": (
            f"Ожидаемая чистая прибыль (P50): {profit_percentiles['P50']:,.2f} руб. "
            f"Вероятность закрытия в убыток: {prob_loss:.1f}%. "
            f"Консервативная оценка риска (P15 / VaR 95%): {profit_percentiles['P15']:,.2f} руб. "
            f"(при худшем 5% сценарии прибыль упадет до {profit_percentiles['P5']:,.2f} руб.)."
        )
    }


# ==============================================================================
# CLI ИНТЕРФЕЙС
# ==============================================================================

def format_markdown_report(result: Dict[str, Any]) -> str:
    """Генерация красивого отчета в Markdown для оркестратора и пользователя"""
    lines = []
    mode = result.get("mode")
    iterations = result.get("iterations", 10000)

    lines.append(f"### 🎲 Результаты анализа методом Монте-Карло ({mode.upper()})")
    lines.append(f"*Количество симуляций:* {iterations:,} итераций | *Движок:* {'NumPy Vectorized' if HAS_NUMPY else 'Pure Python Standard'}\n")

    if mode == "flow":
        xmr = result["xmr_chart"]
        lines.append("#### 📊 Контрольная карта шума процесса (XMR)")
        lines.append(f"- **Средняя пропускная способность (X):** {xmr['mean_throughput']} задач/период")
        lines.append(f"- **Размах шума (Moving Range):** {xmr['mean_moving_range']}")
        lines.append(f"- **Естественные границы процесса (UNPL / LNPL):** [{xmr['lnpl']} … {xmr['unpl']}]")
        status_icon = "✅" if xmr["is_predictable"] else "⚠️"
        lines.append(f"- **Статус процесса:** {status_icon} {'Статистически стабилен и предсказуем' if xmr['is_predictable'] else 'Нестабилен, присутствуют особые причины вариаций'}")

        if "mc_when" in result:
            w = result["mc_when"]
            p = w["percentiles_periods"]
            lines.append(f"\n#### ⏱️ Прогноз сроков (MC-When: {w['backlog_items']} задач)")
            lines.append("| Квантиль надежности | Необходимое число периодов | Уровень риска |")
            lines.append("|---|---|---|")
            lines.append(f"| **P50 (Медиана)** | **{p['P50']:.1f}** | 50% риск срыва |")
            lines.append(f"| **P70 (Умеренный)** | **{p['P70']:.1f}** | 30% риск срыва |")
            lines.append(f"| **P85 (Целевой PM)** | **{p['P85']:.1f}** | 15% риск срыва (рекомендуемый коммитмент) |")
            lines.append(f"| **P95 (Консервативный)** | **{p['P95']:.1f}** | 5% риск срыва (жесткий дедлайн) |")

        if "mc_how_many" in result:
            h = result["mc_how_many"]
            p = h["percentiles_items"]
            lines.append(f"\n#### 📦 Прогноз объема (MC-How-Many: за {h['target_periods']} периодов)")
            lines.append("| Квантиль надежности | Гарантированное число задач |")
            lines.append("|---|---|")
            lines.append(f"| **50% уверенность** | ≥ **{p['P50']:.0f}** задач |")
            lines.append(f"| **85% уверенность** | ≥ **{p['P85']:.0f}** задач |")
            lines.append(f"| **95% уверенность** | ≥ **{p['P95']:.0f}** задач |")

        lines.append(f"\n> **Вывод:** {result.get('mc_when', result.get('mc_how_many', {})).get('summary', '')}")

    elif mode == "pert":
        p = result["percentiles_duration"]
        lines.append("#### ⏳ Оценка расписания проекта (PERT + Critical Path)")
        lines.append("| Квантиль | Длительность проекта | Резерв / буфер | Примечание |")
        lines.append("|---|---|---|---|")
        lines.append(f"| **P50** | **{p['P50']:.2f}** | 0.00 | Сбалансированный срок |")
        lines.append(f"| **P70** | **{p['P70']:.2f}** | +{p['P70'] - p['P50']:.2f} | Оптимальный буфер спринта |")
        lines.append(f"| **P85** | **{p['P85']:.2f}** | +{result['contingency_reserve_p85']:.2f} | Рекомендуемый срок договора/релиза |")
        lines.append(f"| **P95** | **{p['P95']:.2f}** | +{p['P95'] - p['P50']:.2f} | Стресс-сценарий |")

        lines.append("\n#### 🔥 Индекс критичности задач (вероятность попадания на Critical Path)")
        lines.append("| Задача | ID | Критичность | PERT Ожидание |")
        lines.append("|---|---|---|---|")
        for t in result["tasks_criticality"][:7]:
            fire = "🔴" if t["criticality_pct"] > 70 else ("🟡" if t["criticality_pct"] > 30 else "🟢")
            lines.append(f"| {fire} {t['name']} | `{t['id']}` | **{t['criticality_pct']}%** | {t['pert_expected']} |")

        lines.append(f"\n> **Резюме:** {result['summary']}")

    elif mode == "finance":
        p = result["net_profit_percentiles"]
        lines.append("#### 💰 Финансовые риски и маржинальность")
        lines.append(f"- **Вероятность убытка / кассового разрыва:** `{'🔴' if result['probability_of_loss_pct'] > 15 else '🟢'} {result['probability_of_loss_pct']}%`")
        lines.append(f"- **Value at Risk (VaR 95%):** {result['value_at_risk_95']:,.2f} руб.")
        lines.append("\n| Квантиль | Чистая прибыль / Cash Flow | Описание |")
        lines.append("|---|---|---|")
        lines.append(f"| **P95 (Оптимистичный)** | **{p['P95']:,.2f} руб.** | Лучший сценарий (топ-5%) |")
        lines.append(f"| **P50 (Медиана)** | **{p['P50']:,.2f} руб.** | Базовый бизнес-план |")
        lines.append(f"| **P15 (Пессимистичный)** | **{p['P15']:,.2f} руб.** | Консервативная граница |")
        lines.append(f"| **P5 (Стресс / Worst 5%)** | **{p['P5']:,.2f} руб.** | Критический стресс-тест |")

        if result.get("sensitivity"):
            lines.append("\n#### 🎯 Анализ чувствительности (корреляция параметров с результатом)")
            lines.append("| Параметр модели | Коэффициент влияния | Статус |")
            lines.append("|---|---|---|")
            for s in result["sensitivity"][:5]:
                direction = "📈 Повышает прибыль" if s["correlation"] > 0 else "📉 Снижает прибыль"
                lines.append(f"| {s['parameter']} | **{s['correlation']:+.3f}** | {direction} |")

        lines.append(f"\n> **Резюме:** {result['summary']}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo Analytical Engine")
    parser.add_argument("--mode", choices=["flow", "pert", "finance"], required=True, help="Режим симуляции")
    parser.add_argument("--input", type=str, help="Путь к JSON-файлу с входными данными")
    parser.add_argument("--data", type=str, help="Входные данные в виде JSON-строки")
    parser.add_argument("--iterations", type=int, default=10000, help="Количество итераций (по умолчанию 10000)")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown", help="Формат вывода")

    args = parser.parse_args()

    # Чтение данных
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif args.data:
        data = json.loads(args.data)
    else:
        # Чтение из stdin
        data = json.load(sys.stdin)

    if args.mode == "flow":
        res = run_flow_simulation(data, iterations=args.iterations)
    elif args.mode == "pert":
        res = run_pert_simulation(data, iterations=args.iterations)
    elif args.mode == "finance":
        res = run_finance_simulation(data, iterations=args.iterations)
    else:
        raise ValueError(f"Неизвестный режим: {args.mode}")

    if args.format == "json":
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print(format_markdown_report(res))


if __name__ == "__main__":
    main()
