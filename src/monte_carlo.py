"""
Monte Carlo simulator for an Assemble-to-Order (ATO) system.

This module implements a discrete-event Monte Carlo simulator for the
Horng & Lin (2017) ATO system using a continuous-review base-stock policy.

The core function `evaluate_inventory` returns expected profit, average
holding cost, average lost sales, and service level over multiple
replications.
"""

import math
import numpy as np
from typing import Sequence, Tuple


def _sample_erlang(rng: np.random.Generator, rate: float) -> float:
    """Sample an Erlang(shape=2) interarrival time with a given rate."""
    if rate <= 0:
        return math.inf
    return rng.gamma(shape=2.0, scale=1.0 / rate)


def _sample_truncated_normal(rng: np.random.Generator, mean: float, sd: float) -> float:
    """Sample a normal distribution truncated at zero using rejection sampling."""
    if sd <= 0:
        return max(0.0, mean)
    while True:
        value = rng.normal(loc=mean, scale=sd)
        if value >= 0.0:
            return value


def evaluate_inventory(
    x: Sequence[int],
    arrival_rates: Sequence[float],
    production_means: Sequence[float],
    production_stds: Sequence[float],
    profit_per_item: Sequence[float],
    holding_cost_per_item: Sequence[float],
    capacity_per_item: Sequence[int],
    simulation_horizon: float,
    warm_up_period: float,
    n_replications: int,
    random_seed: int = 0,
) -> Tuple[float, float, float, float]:
    """Estimate ATO performance for a base-stock vector x.

    Returns expected_profit, average_holding_cost, average_lost_sales,
    average_service_level across all replications.
    """

    x = np.asarray(x, dtype=int)
    arrival_rates = np.asarray(arrival_rates, dtype=float)
    production_means = np.asarray(production_means, dtype=float)
    production_stds = np.asarray(production_stds, dtype=float)
    profit_per_item = np.asarray(profit_per_item, dtype=float)
    holding_cost_per_item = np.asarray(holding_cost_per_item, dtype=float)
    capacity_per_item = np.asarray(capacity_per_item, dtype=int)

    if x.size != 12:
        raise ValueError("x must have length 12")
    if arrival_rates.size != 8:
        raise ValueError("arrival_rates must have length 8")
    if production_means.size != 12 or production_stds.size != 12:
        raise ValueError("production_means and production_stds must have length 12")
    if profit_per_item.size != 12:
        raise ValueError("profit_per_item must have length 12")
    if holding_cost_per_item.size != 12:
        raise ValueError("holding_cost_per_item must have length 12")
    if capacity_per_item.size != 12:
        raise ValueError("capacity_per_item must have length 12")

    # Bill of Materials from Horng & Lin (2017) Table 1.
    BOM = np.array([
        [0,1,0,0,1,0,1,0,1,1,0,0],  # Product 1
        [1,1,0,1,0,0,1,1,0,1,0,0],  # Product 2
        [0,1,0,0,0,1,0,0,1,1,1,1],  # Product 3
        [0,1,1,0,1,0,0,1,0,1,0,1],  # Product 4
        [1,1,1,1,0,0,1,0,1,0,1,0],  # Product 5
        [0,1,0,0,1,1,0,1,0,1,0,1],  # Product 6
        [0,1,1,0,0,1,1,0,1,0,1,0],  # Product 7
        [1,1,0,1,1,0,0,1,0,1,0,1],  # Product 8
    ], dtype=int)

    key_mask = np.zeros(12, dtype=bool)
    key_mask[:10] = True

    profits = np.zeros(n_replications, dtype=float)
    holding_costs = np.zeros(n_replications, dtype=float)
    lost_sales_arr = np.zeros(n_replications, dtype=float)
    service_levels = np.zeros(n_replications, dtype=float)

    base_rng = np.random.default_rng(random_seed)

    for rep in range(n_replications):
        rng = np.random.default_rng(base_rng.integers(0, 2**31 - 1))

        on_hand = x.copy()
        on_order = np.zeros(12, dtype=int)
        outstanding_orders = []

        next_arrival = np.array([_sample_erlang(rng, rate) for rate in arrival_rates], dtype=float)

        t = 0.0
        last_time = 0.0
        rep_revenue = 0.0
        rep_hold = 0.0
        rep_served = 0
        rep_lost = 0

        while True:
            next_replenishment = min((order[0] for order in outstanding_orders), default=math.inf)
            next_demand = next_arrival.min()
            t_next = min(next_replenishment, next_demand)

            if t_next == math.inf or t_next > simulation_horizon:
                break

            elapsed = t_next - last_time
            if elapsed > 0:
                if last_time >= warm_up_period:
                    rep_hold += elapsed * np.dot(on_hand, holding_cost_per_item)
                elif t_next > warm_up_period:
                    rep_hold += (t_next - warm_up_period) * np.dot(on_hand, holding_cost_per_item)
            last_time = t_next
            t = t_next

            # Process replenishment arrivals at time t
            if outstanding_orders:
                remaining = []
                for arrival_time, item_idx, qty in outstanding_orders:
                    if arrival_time <= t + 1e-12:
                        on_order[item_idx] -= qty
                        on_hand[item_idx] += qty
                    else:
                        remaining.append((arrival_time, item_idx, qty))
                outstanding_orders = remaining

            # Process product arrivals at time t
            arriving_products = np.where(np.abs(next_arrival - t) < 1e-12)[0]
            for prod in arriving_products:
                next_arrival[prod] = t + _sample_erlang(rng, arrival_rates[prod])

                required = BOM[prod].astype(bool)
                required_key = required & key_mask
                if np.any(on_hand[required_key] < 1):
                    if t >= warm_up_period:
                        rep_lost += 1
                    continue

                if t >= warm_up_period:
                    rep_served += 1

                consumed_items = []
                for item_idx in np.where(required)[0]:
                    if on_hand[item_idx] > 0:
                        on_hand[item_idx] -= 1
                        consumed_items.append(item_idx)
                        if t >= warm_up_period:
                            rep_revenue += profit_per_item[item_idx]

                for item_idx in consumed_items:
                    inv_position = on_hand[item_idx] + on_order[item_idx]
                    order_qty = max(0, x[item_idx] - inv_position)
                    max_placeable = capacity_per_item[item_idx] - inv_position
                    order_qty = min(order_qty, max(0, max_placeable))
                    if order_qty > 0:
                        lead_time = _sample_truncated_normal(
                            rng, production_means[item_idx], production_stds[item_idx]
                        )
                        arrival_time = t + lead_time
                        outstanding_orders.append((arrival_time, item_idx, int(order_qty)))
                        on_order[item_idx] += int(order_qty)

        if last_time < simulation_horizon:
            elapsed = simulation_horizon - last_time
            if last_time >= warm_up_period:
                rep_hold += elapsed * np.dot(on_hand, holding_cost_per_item)
            elif simulation_horizon > warm_up_period:
                rep_hold += (simulation_horizon - warm_up_period) * np.dot(on_hand, holding_cost_per_item)

        profits[rep] = rep_revenue - rep_hold
        holding_costs[rep] = rep_hold
        lost_sales_arr[rep] = rep_lost
        service_levels[rep] = (rep_served / (rep_served + rep_lost)) if (rep_served + rep_lost) > 0 else 0.0

    expected_profit = float(np.mean(profits))
    average_holding_cost = float(np.mean(holding_costs))
    average_lost_sales = float(np.mean(lost_sales_arr))
    average_service_level = float(np.mean(service_levels))

    return expected_profit, average_holding_cost, average_lost_sales, average_service_level


if __name__ == "__main__":
    x = [15] * 12
    arrival_rates = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
    production_means = [0.11, 0.09, 0.18, 0.12, 0.15, 0.13, 0.11, 0.22, 0.17, 0.13, 0.21, 0.15]
    production_stds = [0.03, 0.02, 0.02, 0.03, 0.02, 0.01, 0.03, 0.02, 0.01, 0.02, 0.01, 0.02]
    profit_per_item = [1,2,3,4,5,6,7,8,9,10,11,12]
    holding_cost_per_item = [0.7] * 12
    capacity_per_item = [30] * 12
    simulation_horizon = 70
    warm_up_period = 20
    n_replications = 10
    random_seed = 123

    expected_profit, average_holding_cost, average_lost_sales, average_service_level = evaluate_inventory(
        x,
        arrival_rates,
        production_means,
        production_stds,
        profit_per_item,
        holding_cost_per_item,
        capacity_per_item,
        simulation_horizon,
        warm_up_period,
        n_replications,
        random_seed=random_seed,
    )

    print("expected_profit:", expected_profit)
    print("average_holding_cost:", average_holding_cost)
    print("average_lost_sales:", average_lost_sales)
    print("average_service_level:", average_service_level)
