# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
import numpy as np
from scipy import stats
import math

# --- 1. Вихідні дані для всіх 30 завдань ---
datasets = [
    {"title": "1. Урожайність гречки, ц/га", "data": [[17.2, 16.5, 17.5, 16.3], [18.2, 17.5, 17.9, 18.7], [22.4, 20.9, 21.7, 22.1], [23.4, 23.8, 24.3, 24.5]]},
    {"title": "2. Урожайність гороху, ц/га", "data": [[21.5, 26.3, 24.9, 25.6], [18.9, 17.3, 17.3, 18.4], [22.2, 21.8, 23.5, 23.8], [26.5, 25.8, 25.3, 25.3]]},
    {"title": "3. Кількість рослин, уражених фітофторою, шт", "data": [[12, 15, 17, 16], [22, 23, 27, 24], [25, 28, 27, 28], [23, 24, 25, 24]]},
    {"title": "4. Забур’яненість посівів озимої пшениці, шт/м2", "data": [[75, 84, 93, 77], [88, 98, 103, 105], [112, 121, 127, 127], [64, 58, 37, 47]]},
    {"title": "5. Вміст гумусу у грунті, %", "data": [[2.85, 2.47, 2.56, 2.62], [2.11, 1.99, 2.06, 2.08], [2.33, 2.31, 2.36, 2.42]]},
    {"title": "6. Урожайність озимої пшениці, ц/га", "data": [[44.8, 46.3, 45.8, 47.2], [51.1, 50.3, 50.3, 51.1], [48.7, 48.2, 47.2, 47.1], [42.1, 42.6, 45.3, 43.2]]},
    {"title": "7. Урожайність кукурудзи, ц/га", "data": [[54.6, 55.3, 55.2, 54.1], [48.5, 48.3, 48.7, 47.8], [51.3, 52.7, 53.1, 51.8], [47.3, 45.0, 46.8, 45.5]]},
    {"title": "8. Маса 1000 зерен ярої пшениці, г", "data": [[29.5, 31.4, 30.2, 30.6], [36.4, 36.2, 36.1, 35.8], [42.5, 41.9, 40.9, 41.6], [43.6, 44.0, 44.2, 43.0]]},
    {"title": "9. Число зерен в колосі тритікале, шт.", "data": [[38, 39, 42, 44], [48, 49, 52, 51], [49, 43, 47, 48], [52, 50, 51, 53]]},
    {"title": "10. Урожайність озимого жита, ц/га", "data": [[28.5, 29.8, 28.6, 26.3], [23.5, 24.5, 23.9, 23.8], [25.4, 25.1, 25.9, 26.4], [28.3, 28.9, 29.0, 28.7]]},
    {"title": "11. Вміст олії в насінні сої, %", "data": [[14.2, 14.4, 14.3, 14.7], [15.9, 16.3, 16.3, 16.4], [18.5, 17.7, 18.2, 17.3], [20.3, 20.6, 20.4, 21.0]]},
    {"title": "12. Маса 1000 насінин кормових бобів, г", "data": [[180, 197, 198, 183], [230, 243, 241, 237], [345, 328, 375, 385], [477, 438, 459, 467]]},
    {"title": "13. Вміст крохмалю в бульбах картоплі, %", "data": [[17.0, 17.3, 17.7, 17.4], [18.5, 18.9, 18.2, 18.8], [19.5, 19.1, 18.9, 18.3], [20.2, 19.8, 19.3, 19.5]]},
    {"title": "14. Коефіцієнт транспірації озимого жита", "data": [[270, 284, 285, 273], [300, 296, 311, 310], [321, 319, 327, 326], [330, 335, 333, 340]]},
    {"title": "15. Урожайність кукурудзи, ц/га", "data": [[37.9, 38.4, 38.6, 38.5], [42.5, 43.6, 41.7, 42.0], [43.6, 44.5, 42.8, 43.9], [45.1, 44.9, 44.7, 45.3]]},
    {"title": "16. Урожайність гречки, ц/га", "data": [[17.2, 16.5, 17.5, 16.3], [18.2, 17.5, 17.9, 18.7], [22.4, 20.9, 21.7, 22.1], [23.4, 23.8, 24.3, 24.5]]},
    {"title": "17. Урожайність гороху, ц/га", "data": [[21.5, 26.3, 24.9, 25.6], [18.9, 17.3, 17.3, 18.4], [22.2, 21.8, 23.5, 23.8], [26.5, 25.8, 25.3, 25.3]]},
    {"title": "18. Кількість рослин, уражених фітофторою, шт", "data": [[12, 15, 17, 16], [22, 23, 27, 24], [25, 28, 27, 28], [23, 24, 25, 24]]},
    {"title": "19. Забур’яненість посівів озимої пшениці, шт/м", "data": [[75, 84, 93, 77], [88, 98, 103, 105], [112, 121, 127, 127], [64, 58, 37, 47]]},
    {"title": "20. Вміст гумусу у грунті, %", "data": [[2.55, 2.47, 2.56, 2.62], [2.11, 1.99, 2.06, 2.08], [2.33, 2.31, 2.36, 2.42], [2.90, 2.80, 2.85, 2.84]]},
    {"title": "21. Урожайність озимої пшениці, ц/га", "data": [[44.8, 46.3, 45.8, 47.2], [51.1, 50.3, 50.3, 51.1], [48.7, 48.2, 47.2, 47.1], [42.1, 42.6, 45.3, 43.2]]},
    {"title": "22. Урожайність кукурудзи, ц/га", "data": [[54.6, 55.3, 55.2, 54.1], [48.5, 48.3, 43.7, 47.8], [51.3, 52.7, 53.1, 51.8], [53.5, 54.8, 55.6, 55.0]]},
    {"title": "23. Маса 1000 зерен ярої пшениці, г", "data": [[29.5, 31.4, 30.2, 30.6], [36.4, 36.2, 36.1, 35.8], [42.5, 41.9, 40.9, 41.6], [43.3, 44.0, 43.5, 43.0]]},
    {"title": "24. Число зерен в колосі тритікале, шт.", "data": [[38, 39, 42, 44], [48, 49, 52, 51], [49, 43, 47, 48]]},
    {"title": "25. Урожайність озимого жита, ц/га", "data": [[28.5, 29.8, 28.6, 26.3], [23.5, 24.5, 23.9, 23.8], [25.4, 25.1, 25.9, 26.4], [24.6, 24.9, 25.6, 25.0]]},
    {"title": "26. Вміст олії в насінні сої, %", "data": [[14.2, 14.4, 14.3, 14.7], [15.9, 16.3, 16.3, 16.4], [18.5, 17.7, 18.2, 17.3], [20.3, 20.6, 20.4, 21.0]]},
    {"title": "27. Маса 1000 насінин кормових бобів, г", "data": [[180, 197, 198, 183], [230, 243, 241, 237], [345, 328, 375, 385], [477, 438, 459, 467]]},
    {"title": "28. Вміст крохмалю в бульбах картоплі, %", "data": [[17.0, 17.3, 17.7, 17.4], [18.5, 18.9, 18.2, 18.8], [19.5, 19.1, 18.9, 18.3], [20.0, 20.9, 20.3, 20.6]]},
    {"title": "29. Коефіцієнт транспірації озимого жита", "data": [[270, 284, 285, 273], [300, 296, 311, 310], [321, 319, 327, 326], [330, 341, 336, 340]]},
    {"title": "30. Урожайність кукурудзи, ц/га", "data": [[37.9, 38.4, 38.6, 38.5], [42.5, 43.6, 41.7, 42.0], [43.6, 44.5, 42.8, 43.9], [45.1, 46.2, 46.5, 45.9]]},
]

# --- 2. Функція для виконання аналізу ---
def perform_anova(data_matrix):
    try:
        data = np.array(data_matrix)
        l, n = data.shape
        N = l * n
        variant_sums = np.sum(data, axis=1)
        variant_means = np.mean(data, axis=1)
        A = np.mean(data)
        transformed_data = data - A
        C = (np.sum(transformed_data) ** 2) / N
        Cy = np.sum(transformed_data ** 2) - C
        Cp = np.sum(np.sum(transformed_data, axis=0) ** 2) / l - C
        Cv = np.sum(np.sum(transformed_data, axis=1) ** 2) / n - C
        Cz = Cy - Cv - Cp
        vy, vv, vp, vz = N - 1, l - 1, n - 1, (l - 1) * (n - 1)
        if vv <= 0 or vz <= 0: return {"error": "Недостатньо даних для аналізу."}
        s2v = Cv / vv
        s2z = Cz / vz
        F_fact = s2v / s2z if s2z > 0 else float('inf')
        F_crit = stats.f.ppf(0.95, vv, vz)
        Sx = math.sqrt(s2z / n)
        Sd = math.sqrt(2) * Sx
        t_crit = stats.t.ppf(1 - 0.05 / 2, vz)
        NIR05 = t_crit * Sd
        return {
            "initial_data": data.tolist(), "variant_sums": variant_sums.tolist(),
            "variant_means": variant_means.tolist(),
            "anova_table": [
                ("Загальна (Cy)", f"{Cy:.4f}", vy, ""),
                ("Повторностей (Cp)", f"{Cp:.4f}", vp, ""),
                ("Варіантів (Cv)", f"{Cv:.4f}", vv, f"{s2v:.4f}"),
                ("Похибки (Cz)", f"{Cz:.4f}", vz, f"{s2z:.4f}"),
            ],
            "f_statistic": {"fact": F_fact, "crit": F_crit}, "nir": NIR05, "error": None
        }
    except Exception as e:
        return {"error": f"Сталася помилка: {e}"}

# --- 3. Клас для створення GUI ---
class AnovaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ANOVA Professional Dashboard")
        self.root.geometry("1000x750")

        style = ttkb.Style.get_instance()
        style.configure('TLabelframe.Label', foreground="white")
        
        main_frame = ttkb.Frame(self.root, padding=20)
        main_frame.pack(fill=BOTH, expand=YES)
        
        header_frame = ttkb.Frame(main_frame)
        header_frame.pack(fill=X, pady=(0, 20), anchor=N)
        ttkb.Label(header_frame, text="Дисперсійний аналіз (ANOVA)", font=("Segoe UI", 24, "bold"), bootstyle=PRIMARY).pack(side=LEFT)

        control_frame = ttkb.Frame(main_frame)
        control_frame.pack(fill=X, pady=(0, 20), expand=NO)
        ttkb.Label(control_frame, text="Оберіть експеримент:", font=("Segoe UI", 12)).pack(side=LEFT, padx=(0, 15))
        self.task_titles = [d["title"] for d in datasets]
        self.task_combo = ttkb.Combobox(control_frame, values=self.task_titles, state="readonly", font=("Segoe UI", 12))
        self.task_combo.pack(side=LEFT, fill=X, expand=YES)
        self.task_combo.bind("<<ComboboxSelected>>", self.update_dashboard)

        self.notebook = ttkb.Notebook(main_frame)
        self.notebook.pack(fill=BOTH, expand=YES)
        self.create_tabs()

        self.task_combo.current(0)
        self.update_dashboard()
        
    def create_tabs(self):
        self.tab_summary = ttkb.Frame(self.notebook, padding=20)
        self.tab_data = ttkb.Frame(self.notebook, padding=20)
        self.tab_anova = ttkb.Frame(self.notebook, padding=20)
        self.tab_comparison = ttkb.Frame(self.notebook, padding=20)
        
        self.notebook.add(self.tab_summary, text=" 📊 Головні підсумки ")
        self.notebook.add(self.tab_data, text=" 🔢 Вихідні дані ")
        self.notebook.add(self.tab_anova, text=" 🔬 Таблиця ANOVA ")
        self.notebook.add(self.tab_comparison, text=" 📈 Порівняння варіантів ")

        self.setup_summary_tab()
        self.setup_table_tabs()

    def setup_summary_tab(self):
        conclusion_frame = ttkb.Labelframe(self.tab_summary, text="Ключовий висновок", bootstyle=INFO, padding=15)
        conclusion_frame.pack(fill=X, pady=(0, 20))
        self.lbl_conclusion = ttkb.Label(conclusion_frame, text="", font=("Segoe UI", 16, "bold"))
        self.lbl_conclusion.pack()

        metrics_frame = ttkb.Frame(self.tab_summary)
        metrics_frame.pack(fill=X, expand=YES)
        metrics_frame.columnconfigure((0,1,2), weight=1)

        f_fact_frame = ttkb.Labelframe(metrics_frame, text="F-факт.", bootstyle=SECONDARY, padding=15)
        f_fact_frame.grid(row=0, column=0, padx=10, sticky=NSEW)
        self.lbl_f_fact = ttkb.Label(f_fact_frame, text="", font=("Segoe UI", 22, "bold"), bootstyle=PRIMARY)
        self.lbl_f_fact.pack()

        f_crit_frame = ttkb.Labelframe(metrics_frame, text="F-теор.", bootstyle=SECONDARY, padding=15)
        f_crit_frame.grid(row=0, column=1, padx=10, sticky=NSEW)
        self.lbl_f_crit = ttkb.Label(f_crit_frame, text="", font=("Segoe UI", 22, "bold"), bootstyle=SECONDARY)
        self.lbl_f_crit.pack()

        nir_frame = ttkb.Labelframe(metrics_frame, text="НІР₀₅ (Найменша Істотна Різниця)", bootstyle=SECONDARY, padding=15)
        nir_frame.grid(row=0, column=2, padx=10, sticky=NSEW)
        self.lbl_nir = ttkb.Label(nir_frame, text="", font=("Segoe UI", 22, "bold"), bootstyle=SUCCESS)
        self.lbl_nir.pack()
        
    def setup_table_tabs(self):
        style = ttkb.Style.get_instance()
        style.configure("Treeview.Heading", font=("Segoe UI", 11, "bold"))
        style.configure("Treeview", rowheight=25, font=("Segoe UI", 10))
        self.tree_data = self.create_treeview(self.tab_data, [])
        self.tree_anova = self.create_treeview(self.tab_anova, ["Джерело", "Сума квадратів", "Ст. свободи", "Сер. квадрат"])
        self.tree_comparison = self.create_treeview(self.tab_comparison, ["Варіант", "Середнє", "Відхилення", "Оцінка", "Висновок"])
        
        # --- **ВИПРАВЛЕННЯ КОНТРАСТНОСТІ** ---
        # Встановлюємо фон і БІЛИЙ колір тексту для виділених рядків
        self.tree_comparison.tag_configure("significant", background="#005200", foreground="white")
        self.tree_comparison.tag_configure("not_significant", background="#4a4a00", foreground="white")
        
    def create_treeview(self, parent, columns):
        frame = ttkb.Frame(parent)
        frame.pack(fill=BOTH, expand=YES)
        tree = ttkb.Treeview(frame, columns=columns, show="headings", bootstyle=PRIMARY)
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, anchor=CENTER, width=150)
        vsb = ttkb.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=RIGHT, fill=Y)
        tree.pack(fill=BOTH, expand=YES)
        return tree

    def clear_treeview(self, tree):
        for item in tree.get_children():
            tree.delete(item)

    def update_dashboard(self, event=None):
        idx = self.task_combo.current()
        data = datasets[idx]["data"]
        results = perform_anova(data)
        if results["error"]: return

        f_fact = results["f_statistic"]["fact"]
        f_crit = results["f_statistic"]["crit"]
        is_significant = f_fact > f_crit
        if is_significant:
            self.lbl_conclusion.config(text="Відмінності між варіантами є СТАТИСТИЧНО ЗНАЧУЩИМИ", bootstyle=SUCCESS)
        else:
            self.lbl_conclusion.config(text="Відмінності між варіантами НЕ є статистично значущими", bootstyle=WARNING)
        
        self.lbl_f_fact.config(text=f"{f_fact:.2f}")
        self.lbl_f_crit.config(text=f"{f_crit:.2f}")
        self.lbl_nir.config(text=f"{results['nir']:.4f}")

        self.clear_treeview(self.tree_data)
        num_reps = len(results["initial_data"][0])
        cols = ["Варіант"] + [f"Повт. {i+1}" for i in range(num_reps)] + ["Сума", "Середнє"]
        self.tree_data.config(columns=cols)
        for col in cols: 
            self.tree_data.heading(col, text=col)
            self.tree_data.column(col, anchor=CENTER, width=100)
        for i, row in enumerate(results["initial_data"]):
            values = [i+1] + row + [f"{results['variant_sums'][i]:.2f}", f"{results['variant_means'][i]:.2f}"]
            self.tree_data.insert("", END, values=values)
            
        self.clear_treeview(self.tree_anova)
        for row in results["anova_table"]:
            self.tree_anova.insert("", END, values=row)

        self.clear_treeview(self.tree_comparison)
        means = results["variant_means"]
        nir = results["nir"]
        control_mean = means[0]
        self.tree_comparison.insert("", END, values=["1 (контроль)", f"{control_mean:.2f}", "-", "-", "-"])
        for i in range(1, len(means)):
            diff = means[i] - control_mean
            is_diff_significant = abs(diff) > nir
            tag = "significant" if is_diff_significant else "not_significant"
            conclusion = "Істотне" if is_diff_significant else "Не істотне"
            comparison = f"{abs(diff):.2f} > {nir:.2f}" if is_diff_significant else f"{abs(diff):.2f} <= {nir:.2f}"
            values = [i+1, f"{means[i]:.2f}", f"{diff:.2f}", comparison, conclusion]
            self.tree_comparison.insert("", END, values=values, tags=(tag,))

# --- 4. Запуск програми ---
if __name__ == "__main__":
    app = ttkb.Window(themename="superhero")
    AnovaApp(app)
    app.mainloop()