"""
Variable Neighborhood Search (VNS) Algorithm

Implements VNS for integer optimization with best improvement local search
and FE (Function Evaluation) counting. Maximizes the objective function.
"""

import numpy as np
from typing import Callable, Tuple, List


def vns(
    objective_func: Callable,
    max_FE: int,
    iter_max: int,
    k_max: int,
    lb: List[int],
    ub: List[int],
    seed: int = None,
) -> Tuple[np.ndarray, float, int, List[Tuple[int, float]]]:
    """
    Variable Neighborhood Search (VNS) algorithm.
    
    Uses best improvement local search (first neighborhood N1: neighbors differ by ±1 in one dimension).
    
    Args:
        objective_func: Function to maximize. Called as objective_func(x) -> float.
        max_FE: Maximum number of function evaluations allowed.
        iter_max: Maximum number of iterations (main loop).
        k_max: Maximum neighborhood index.
        lb: Lower bounds as list of integers.
        ub: Upper bounds as list of integers.
        seed: Random seed for reproducibility.
    
    Returns:
        (x_best, f_best, n_FE_used, history)
        - x_best: Best solution found (array of integers)
        - f_best: Objective value at best solution
        - n_FE_used: Number of function evaluations used
        - history: List of (iteration, best_fitness) tuples for tracking
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    lb = np.array(lb, dtype=float)
    ub = np.array(ub, dtype=float)
    dim = len(lb)
    
    # Generate initial solution: random feasible point
    x_best = np.random.uniform(lb, ub, dim)
    x_best = np.round(x_best).astype(int)
    
    # Evaluate initial solution
    f_best = objective_func(x_best)
    n_FE = 1
    
    history = [(0, f_best)]
    
    # Main loop
    for iteration in range(iter_max):
        if n_FE >= max_FE:
            break
        
        k = 1
        
        while k <= k_max and n_FE < max_FE:
            # Shake: generate random neighbor in N_k
            # N_k: perturb exactly k random dimensions by ±1
            x_next = x_best.copy()
            indices_to_perturb = np.random.choice(dim, size=min(k, dim), replace=False)
            
            for idx in indices_to_perturb:
                perturbation = np.random.choice([-1, 1])
                x_next[idx] = x_best[idx] + perturbation
            
            # Apply bounds
            x_next = np.clip(x_next, lb, ub).astype(int)
            
            # Evaluate shaken solution
            f_next = objective_func(x_next)
            n_FE += 1
            
            # Local search: best improvement in N1
            improved = True
            x_local = x_next.copy()
            f_local = f_next
            
            while improved and n_FE < max_FE:
                improved = False
                best_neighbor = x_local.copy()
                best_neighbor_f = f_local
                
                # Check all neighbors in N1
                for dim_idx in range(dim):
                    for delta in [-1, 1]:
                        if n_FE >= max_FE:
                            break
                        
                        x_neighbor = x_local.copy()
                        x_neighbor[dim_idx] += delta
                        
                        # Check bounds
                        if x_neighbor[dim_idx] < lb[dim_idx] or x_neighbor[dim_idx] > ub[dim_idx]:
                            continue
                        
                        # Evaluate neighbor
                        f_neighbor = objective_func(x_neighbor)
                        n_FE += 1
                        
                        # Best improvement: accept only if strictly better
                        if f_neighbor > best_neighbor_f:
                            best_neighbor = x_neighbor.copy()
                            best_neighbor_f = f_neighbor
                            improved = True
                
                if improved:
                    x_local = best_neighbor.copy()
                    f_local = best_neighbor_f
            
            # Check if local optimum is better than global best
            if f_local > f_best:
                x_best = x_local.copy()
                f_best = f_local
                k = 1  # Reset k to restart with smaller neighborhoods
            else:
                k += 1  # Try larger neighborhood
        
        history.append((iteration + 1, f_best))
        
        # Check if max FE reached
        if n_FE >= max_FE:
            break
    
    return x_best, f_best, n_FE, history


if __name__ == "__main__":
    # Simple test with sphere function
    def sphere(x):
        return -np.sum(x**2)  # Negative for maximization (minimization becomes maximization)
    
    x_best, f_best, n_FE, history = vns(
        sphere,
        max_FE=1000,
        iter_max=60,
        k_max=5,
        lb=[0]*5,
        ub=[30]*5,
        seed=42
    )
    print(f"Best solution: {x_best}")
    print(f"Best fitness: {f_best}")
    print(f"Function evaluations: {n_FE}")
