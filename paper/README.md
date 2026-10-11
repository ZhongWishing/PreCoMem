# Paper figures

Data and scripts that produce Figures 3, 4 and 5 of the NeurIPS 2026 paper from the aggregated results.

| Script | Figure | Data |
|---|---|---|
| `scripts/1_plot_eval_overview_with_pmv2.py` | Figure 3: backbone generalization (a)–(d) and PersonaMem-v2 system comparison (e)–(f) | `data/backbone_generalization.csv`, `data/personamemv2_baseline.csv` |
| `scripts/2_plot_ablation_dimobench_row.py` | Figure 4: component ablation, DiMoBench adaptation dynamics, operator ablation | `data/component_ablation.csv`, `data/dimobench_dynamics.csv`, `data/dimobench_operator_ablation.csv` |
| `scripts/3_plot_frontier_memory_row.py` | Figure 5: accuracy–efficiency frontier and memory scalability | `data/pareto_frontier_locomo.csv`, `data/memory_scalability_dimo350.csv`, `data/memory_scalability_dimo_xl.csv` |

The CSV files hold the plotted values (means over the three seeds `{13, 42, 1234}` where the paper reports means; standard deviations where error bars are drawn). The scripts only need `numpy` and `matplotlib`:

```bash
pip install matplotlib
python paper/scripts/1_plot_eval_overview_with_pmv2.py
python paper/scripts/2_plot_ablation_dimobench_row.py
python paper/scripts/3_plot_frontier_memory_row.py
```

Outputs are written to `paper/output/` as PDF and PNG. The scripts use Times New Roman when it is installed and fall back to DejaVu Serif otherwise, which changes the look but not the content.
