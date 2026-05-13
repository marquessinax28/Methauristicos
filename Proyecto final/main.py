"""
main.py
=======
Archivo principal del proyecto de metaheurísticos.

Evalúa TODOS los algoritmos × TODAS las funciones objetivo ×
AMBOS métodos de inicialización × dimensiones 3 y 10.

Por cada combinación se realizan N_RUNS ejecuciones independientes
para calcular las métricas estadísticas (AD, MD, SD).

Al finalizar exporta un archivo Excel con la tabla de resultados.

Uso:
    python main.py

Requisitos adicionales (instalar si hace falta):
    pip install pandas openpyxl
"""

import os
import time
import numpy as np
import pandas as pd

# ── Módulos del proyecto ────────────────────────────────────
from funciones_objetivo import ackley, peaks, sphere, rastrigin
from metodos_inicializacion import inicializacion_lhs, maximin_distance
from algoritmos import (abc, aco, bat, fa, ga, hs, pso,
                        hill_climbing, random_search, local_search)

# ════════════════════════════════════════════════════════════
#  CONFIGURACIÓN GLOBAL
# ════════════════════════════════════════════════════════════
OUTPUT_PATH = r"C:\Users\marqu\Desktop\METAURISTICOS\Proyecto final\resultados_metauristicos.xlsx"

N_RUNS   = 30        # ejecuciones independientes por combinación (para estadísticas)
N_ITER   = 50        # iteraciones internas de cada ejecución
N_POP    = 20        # tamaño de población (≥ 2 requerido)
BOUNDS   = (-3.0, 3.0)
DIMS     = [3, 10]   # dimensiones a evaluar: primero 3D, luego 10D

# ── Registro de algoritmos ───────────────────────────────────
ALGORITHMS = {
    'ABC':                abc,
    'ACO':                aco,
    'BAT':                bat,
    'FA':                 fa,
    'GA':                 ga,
    'HS':                 hs,
    'PSO':                pso,
    'Hill Climbing':      hill_climbing,
    'Busqueda Aleatoria': random_search,
    'Busqueda Local':     local_search,
}

# ── Registro de funciones objetivo ───────────────────────────
FUNCTIONS = {
    'Ackley':    ackley,
    'Peaks':     peaks,
    'Sphere':    sphere,
    'Rastrigin': rastrigin,
}

# ── Métodos de inicialización ────────────────────────────────
INIT_METHODS = {
    'LHS':     inicializacion_lhs,
    'Maximin': maximin_distance,
}


# ════════════════════════════════════════════════════════════
#  FUNCIÓN PRINCIPAL
# ════════════════════════════════════════════════════════════
def main():
    lo, hi = BOUNDS
    rows   = []

    total = len(DIMS) * len(ALGORITHMS) * len(FUNCTIONS) * len(INIT_METHODS)
    done  = 0

    print("=" * 70)
    print("  EVALUACIÓN DE METAHEURÍSTICOS")
    print(f"  Algoritmos   : {len(ALGORITHMS)}")
    print(f"  Funciones    : {len(FUNCTIONS)}")
    print(f"  Inicializac. : {len(INIT_METHODS)}")
    print(f"  Dimensiones  : {DIMS}")
    print(f"  Ejecuciones  : {N_RUNS}  |  Iteraciones internas: {N_ITER}")
    print(f"  Combinaciones: {total}")
    print("=" * 70)

    t_global = time.time()

    for dim in DIMS:
        print(f"\n{'─'*70}")
        print(f"  DIMENSIÓN = {dim}")
        print(f"{'─'*70}")

        for algo_name, algo_fn in ALGORITHMS.items():
            for func_name, obj_fn in FUNCTIONS.items():
                for init_name in INIT_METHODS:

                    best_vals_runs  = []
                    times_runs      = []
                    first_start_pos = None
                    overall_best_pos = None
                    overall_best_val = -np.inf

                    for run in range(N_RUNS):
                        # Semilla diferente en cada ejecución
                        seed = run * 997 + dim * 101
                        rng  = np.random.default_rng(seed)

                        # ── Generar población inicial ───────────────
                        if init_name == 'LHS':
                            init_pos = inicializacion_lhs(
                                N_POP, dim, lo, hi, rng)
                        else:
                            init_pos = maximin_distance(
                                N_POP, dim, lo, hi, rng=rng)

                        # Guardar posición inicial de la primera corrida
                        if run == 0:
                            first_start_pos = init_pos[0].copy()

                        # ── Ejecutar algoritmo ──────────────────────
                        try:
                            t0 = time.time()
                            best_p, best_v, _, _ = algo_fn(
                                obj_fn, dim, BOUNDS, N_ITER, init_pos)
                            elapsed = time.time() - t0
                        except Exception as e:
                            print(f"  [ERROR] {algo_name} | {func_name} | "
                                  f"{init_name} | dim={dim} | run={run}: {e}")
                            best_p   = init_pos[0].copy()
                            best_v   = float(obj_fn(init_pos[0]))
                            elapsed  = 0.0

                        best_vals_runs.append(float(best_v))
                        times_runs.append(float(elapsed))

                        if float(best_v) > overall_best_val:
                            overall_best_val = float(best_v)
                            overall_best_pos  = best_p.copy()

                    # ── Calcular métricas ───────────────────────────
                    bv  = np.array(best_vals_runs)
                    ad  = float(np.mean(bv))
                    md  = float(np.median(bv))
                    sd  = float(np.std(bv, ddof=0))
                    t_p = float(np.mean(times_runs))

                    rows.append({
                        'Algoritmo':
                            algo_name,
                        'Funcion Objetivo':
                            func_name,
                        'Metodo de Inicializacion':
                            init_name,
                        'Dimension':
                            dim,
                        'AD (Media)':
                            round(ad, 6),
                        'MD (Mediana)':
                            round(md, 6),
                        'SD (Desv. Estandar)':
                            round(sd, 6),
                        'Tiempo Promedio (s)':
                            round(t_p, 6),
                        'Posicion Inicial':
                            str(np.round(first_start_pos, 4).tolist()),
                        'Mejor Posicion Encontrada':
                            str(np.round(overall_best_pos, 4).tolist()),
                        'Mejor Valor f(x)':
                            round(overall_best_val, 6),
                    })

                    done += 1
                    elapsed_global = time.time() - t_global
                    eta = (elapsed_global / done) * (total - done) if done > 0 else 0
                    print(f"  [{done:>3}/{total}]  {algo_name:<22} | {func_name:<10} | "
                          f"{init_name:<8} | dim={dim}  "
                          f"→ AD={ad:+.4f}  SD={sd:.4f}  "
                          f"t={t_p:.3f}s  (ETA {eta:.0f}s)")

    # ════════════════════════════════════════════════════════
    #  EXPORTAR A EXCEL
    # ════════════════════════════════════════════════════════
    column_order = [
        'Algoritmo',
        'Funcion Objetivo',
        'Metodo de Inicializacion',
        'Dimension',
        'AD (Media)',
        'MD (Mediana)',
        'SD (Desv. Estandar)',
        'Tiempo Promedio (s)',
        'Posicion Inicial',
        'Mejor Posicion Encontrada',
        'Mejor Valor f(x)',
    ]

    df = pd.DataFrame(rows, columns=column_order)

    # Crear directorio si no existe
    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Exportar con formato de ancho de columnas
    with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Resultados')

        # ── Ajustar ancho de columnas automáticamente ───────
        ws = writer.sheets['Resultados']
        for col_cells in ws.columns:
            max_len = 0
            col_letter = col_cells[0].column_letter
            for cell in col_cells:
                try:
                    cell_len = len(str(cell.value)) if cell.value is not None else 0
                    max_len  = max(max_len, cell_len)
                except Exception:
                    pass
            ws.column_dimensions[col_letter].width = min(max_len + 2, 60)

    total_time = time.time() - t_global
    print(f"\n{'=' * 70}")
    print(f"  ✅  Tabla exportada → {OUTPUT_PATH}")
    print(f"  Filas: {len(df)}   |   Tiempo total: {total_time:.1f}s")
    print(f"{'=' * 70}\n")

    return df


# ════════════════════════════════════════════════════════════
if __name__ == '__main__':
    main()
