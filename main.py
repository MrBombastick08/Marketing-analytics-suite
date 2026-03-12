import customtkinter as ctk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog, messagebox
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tabulate import tabulate
import scipy.cluster.hierarchy as sch
import os

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class AnalyticsApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Business Intelligence Tool 2026")
        self.geometry("1400x900")

        self.df = None
        self.scaled_data = None
        self.figures = {}  # Хранилище для графиков, чтобы их сохранять

        # --- Layout ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(self.sidebar, text="📊 DATA ANALYTICS", font=ctk.CTkFont(size=22, weight="bold")).pack(pady=20)

        self.btn_browse = ctk.CTkButton(self.sidebar, text="📂 Открыть CSV", command=self.browse_file)
        self.btn_browse.pack(padx=20, pady=10)

        ctk.CTkLabel(self.sidebar, text="Число кластеров:").pack(pady=(20, 0))
        self.cluster_entry = ctk.CTkEntry(self.sidebar, width=100)
        self.cluster_entry.insert(0, "4")
        self.cluster_entry.pack(pady=10)

        self.btn_run = ctk.CTkButton(self.sidebar, text="🚀 Запустить анализ", command=self.run_full_analysis,
                                     fg_color="#27ae60")
        self.btn_run.pack(padx=20, pady=10)

        self.btn_report = ctk.CTkButton(self.sidebar, text="📄 Создать отчёт", command=self.show_report,
                                        state="disabled")
        self.btn_report.pack(padx=20, pady=10)

        # Main Tabs
        self.tabs = ctk.CTkTabview(self)
        self.tabs.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        self.tab_data = self.tabs.add("📋 Обзор")
        self.tab_elbow = self.tabs.add("📈 Подбор K")
        self.tab_corr = self.tabs.add("🔥 Связи")
        self.tab_cluster = self.tabs.add("🎯 Кластеры")
        self.tab_dendro = self.tabs.add("🌲 Дерево")

    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            try:
                self.df = pd.read_csv(path)
                self.prepare_initial_view()
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось прочитать файл: {e}")

    def prepare_initial_view(self):
        # Очистка старых виджетов
        for tab in [self.tab_data, self.tab_elbow, self.tab_corr, self.tab_cluster, self.tab_dendro]:
            for widget in tab.winfo_children(): widget.destroy()

        # Анализ данных
        null_info = self.df.isnull().sum().to_frame(name='Пропуски').reset_index()
        stats_df = self.df.describe().round(2).reset_index()

        txt = ctk.CTkTextbox(self.tab_data, font=("Courier New", 12))
        txt.pack(fill="both", expand=True, padx=10, pady=10)

        info_str = "=== ПРОВЕРКА ДАННЫХ ===\n"
        info_str += tabulate(null_info, headers='keys', tablefmt='github') + "\n\n"
        info_str += "=== СТАТИСТИКА ===\n"
        info_str += tabulate(stats_df, headers='keys', tablefmt='github')

        txt.insert("0.0", info_str)
        txt.configure(state="disabled")

        # Масштабирование
        num_cols = self.df.select_dtypes(include=[np.number])
        self.scaled_data = StandardScaler().fit_transform(num_cols)

        self.plot_corr()
        self.plot_elbow()

    def save_plot(self, name):
        if name in self.figures:
            path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
            if path:
                self.figures[name].savefig(path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Успех", "График сохранен!")

    def add_plot_to_tab(self, fig, tab, name):
        self.figures[name] = fig
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        btn_save = ctk.CTkButton(tab, text="💾 Сохранить график", command=lambda: self.save_plot(name), width=150)
        btn_save.pack(pady=10)

    def plot_corr(self):
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(self.df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Матрица корреляций")
        self.add_plot_to_tab(fig, self.tab_corr, "correlation")

    def plot_elbow(self):
        wcss = []
        sil = []
        k_range = range(2, 11)
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(self.scaled_data)
            wcss.append(km.inertia_)
            sil.append(silhouette_score(self.scaled_data, km.labels_))

        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(k_range, wcss, 'go-', label='Локоть (WCSS)')
        ax1.set_ylabel('Inertia')

        ax2 = ax1.twinx()
        ax2.plot(k_range, sil, 'ro-', label='Силуэт')
        ax2.set_ylabel('Silhouette Score')

        fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        ax1.set_title("Анализ оптимального числа кластеров")
        self.add_plot_to_tab(fig, self.tab_elbow, "elbow_method")

    def run_full_analysis(self):
        if self.df is None: return

        try:
            k = int(self.cluster_entry.get())
            # K-Means
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            self.df['cluster'] = km.fit_predict(self.scaled_data)

            # Отрисовка Кластеров
            for w in self.tab_cluster.winfo_children(): w.destroy()
            fig, ax = plt.subplots()
            # Берем первые 2 колонки для осей
            cols = self.df.columns
            sns.scatterplot(data=self.df, x=cols[1], y=cols[2], hue='cluster', palette='viridis', s=100, ax=ax)
            ax.set_title(f"Результат кластеризации (K={k})")
            self.add_plot_to_tab(fig, self.tab_cluster, "clusters")

            # Отрисовка Дендрограммы
            for w in self.tab_dendro.winfo_children(): w.destroy()
            fig_d, ax_d = plt.subplots(figsize=(8, 4))
            sch.dendrogram(sch.linkage(self.scaled_data, method='ward'), ax=ax_d)
            ax_d.set_title("Иерархическая дендрограмма")
            self.add_plot_to_tab(fig_d, self.tab_dendro, "dendrogram")

            self.btn_report.configure(state="normal")
            self.tabs.set("🎯 Кластеры")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def show_report(self):
        rep_win = ctk.CTkToplevel(self)
        rep_win.title("Экспорт отчёта")
        rep_win.geometry("900x700")
        rep_win.after(200, lambda: rep_win.focus_force())

        report_df = self.df.groupby('cluster').mean(numeric_only=True).round(2)
        md_table = tabulate(report_df, headers='keys', tablefmt='github')

        report_text = f"# АНАЛИТИЧЕСКИЙ ОТЧЕТ\n\n## Характеристики сегментов\n\n{md_table}\n\n"
        report_text += "## Рекомендации:\n1. Группы с высоким Spending Score — приоритет для маркетинга.\n"
        report_text += "2. Группы с низким доходом, но частыми покупками — сегмент 'Эконом-лояльность'."

        txt = ctk.CTkTextbox(rep_win, font=("Courier New", 14))
        txt.pack(fill="both", expand=True, padx=20, pady=(20, 10))
        txt.insert("0.0", report_text)
        txt.configure(state="disabled")

        def save_report():
            path = filedialog.asksaveasfilename(defaultextension=".md", filetypes=[("Markdown", "*.md")])
            if path:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(report_text)
                messagebox.showinfo("Сохранено", "Отчёт успешно экспортирован!")

        btn_save_rep = ctk.CTkButton(rep_win, text="💾 Сохранить отчёт в .md", command=save_report)
        btn_save_rep.pack(pady=10)


if __name__ == "__main__":
    app = AnalyticsApp()
    app.mainloop()