"""
Функции визуализации результатов моделирования
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'figure.titlesize': 18,
    'legend.fontsize': 11
})


class PeltierPlotter:
    """Класс для создания графиков результатов моделирования"""

    def __init__(self, output_dir='output/plots'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_temperature_profile(self, results_dict, save=True, show=True):
        """
        Построение профиля температуры вдоль ветви.

        Параметры:
        ----------
        results_dict : dict
            Результаты моделирования
        save : bool
            Сохранять ли график в файл
        show : bool
            Показывать ли график
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        x = results_dict['x'] * 1000  # перевод в мм
        T = results_dict['temperature'] - 273.15  # перевод в °C

        ax.plot(x, T, 'b-', linewidth=2, label='Стационарный профиль')

        # Выделение холодного и горячего спаев
        ax.scatter([x[0], x[-1]], [T[0], T[-1]],
                   color=['blue', 'red'], s=100, zorder=5)
        ax.annotate(f'Холодный спай\n{T[0]:.2f} °C',
                    xy=(x[0], T[0]), xytext=(5, -20),
                    textcoords='offset points')
        ax.annotate(f'Горячий спай\n{T[-1]:.2f} °C',
                    xy=(x[-1], T[-1]), xytext=(-60, 10),
                    textcoords='offset points')

        # Настройка графика
        ax.set_xlabel('Координата вдоль ветви, мм')
        ax.set_ylabel('Температура, °C')
        ax.set_title('Профиль температуры в элементе Пельтье')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Информация о параметрах
        params = results_dict['parameters']
        perf = results_dict['performance']
        info_text = f'Ток: {params["current"]} А\n' \
                    f'ΔT: {perf["delta_T"]:.2f} K\n' \
                    f'Q_c: {perf["Q_cold"]:.2f} Вт\n' \
                    f'COP: {perf["COP"]:.3f}'
        ax.text(0.02, 0.98, info_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save:
            filename = f'temp_profile_I{params["current"]:.1f}A.png'
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300)
            print(f"График сохранен: {filepath}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_transient_process(self, results_dict, save=True, show=True):
        """
        Построение переходного процесса.

        Параметры:
        ----------
        results_dict : dict
            Результаты моделирования
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        time = results_dict['time_history']
        T_cold_history = results_dict['temperature_history'][:, 0] - 273.15
        T_hot_history = results_dict['temperature_history'][:, -1] - 273.15

        ax.plot(time, T_cold_history, 'b-', linewidth=2, label='Холодный спай')
        ax.plot(time, T_hot_history, 'r-', linewidth=2, label='Горячий спай')

        # Отметка стационарного состояния
        if results_dict['is_stationary']:
            steady_time = results_dict['performance']['time_to_steady']
            ax.axvline(x=steady_time, color='k', linestyle='--', alpha=0.5,
                       label=f'Стационарный режим ({steady_time:.2f} с)')

        ax.set_xlabel('Время, с')
        ax.set_ylabel('Температура, °C')
        ax.set_title('Переходный процесс в элементе Пельтье')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        if save:
            params = results_dict['parameters']
            filename = f'transient_I{params["current"]:.1f}A.png'
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300)

        if show:
            plt.show()
        else:
            plt.close()

    def plot_performance_curves(self, study_results, save=True, show=True):
        """
        Построение кривых рабочих характеристик.

        Параметры:
        ----------
        study_results : list of dict
            Результаты исследования по разным токам
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        currents = [r['parameters']['current'] for r in study_results]
        Q_cold = [r['performance']['Q_cold'] for r in study_results]
        COP = [r['performance']['COP'] for r in study_results]
        delta_T = [r['performance']['delta_T'] for r in study_results]
        P_total = [r['performance']['P_total'] for r in study_results]

        # График 1: Холодопроизводительность
        ax = axes[0, 0]
        ax.plot(currents, Q_cold, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Ток, А')
        ax.set_ylabel('Холодопроизводительность, Вт')
        ax.set_title('Зависимость Q_c от тока')
        ax.grid(True, alpha=0.3)

        # Отметка максимума
        max_idx = np.argmax(Q_cold)
        ax.plot(currents[max_idx], Q_cold[max_idx], 'r*', markersize=15)
        ax.annotate(f'Максимум\n{currents[max_idx]:.1f} А, {Q_cold[max_idx]:.1f} Вт',
                    xy=(currents[max_idx], Q_cold[max_idx]),
                    xytext=(10, 10), textcoords='offset points')

        # График 2: Коэффициент эффективности
        ax = axes[0, 1]
        ax.plot(currents, COP, 'go-', linewidth=2, markersize=8)
        ax.set_xlabel('Ток, А')
        ax.set_ylabel('COP')
        ax.set_title('Зависимость COP от тока')
        ax.grid(True, alpha=0.3)

        # График 3: Перепад температур
        ax = axes[1, 0]
        ax.plot(currents, delta_T, 'ro-', linewidth=2, markersize=8)
        ax.set_xlabel('Ток, А')
        ax.set_ylabel('ΔT, K')
        ax.set_title('Зависимость ΔT от тока')
        ax.grid(True, alpha=0.3)

        # График 4: Электрическая мощность
        ax = axes[1, 1]
        ax.plot(currents, P_total, 'mo-', linewidth=2, markersize=8)
        ax.set_xlabel('Ток, А')
        ax.set_ylabel('Потребляемая мощность, Вт')
        ax.set_title('Зависимость мощности от тока')
        ax.grid(True, alpha=0.3)

        plt.suptitle('Рабочие характеристики элемента Пельтье', fontsize=16)
        plt.tight_layout()

        if save:
            filepath = self.output_dir / 'performance_curves.png'
            plt.savefig(filepath, dpi=300)

        if show:
            plt.show()
        else:
            plt.close()

    def plot_3d_temperature_surface(self, multi_current_results, save=True, show=True):
        """
        3D визуализация профилей температуры для разных токов.

        Параметры:
        ----------
        multi_current_results : dict
            Словарь {ток: results_dict}
        """
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection='3d')

        currents = sorted(multi_current_results.keys())

        for I in currents:
            results = multi_current_results[I]
            x = results['x'] * 1000  # мм
            T = results['temperature'] - 273.15  # °C

            # Повторяем x для каждой точки z (тока)
            z = np.ones_like(x) * I

            ax.plot(x, z, T, linewidth=2, label=f'I={I} А')

        ax.set_xlabel('Координата, мм')
        ax.set_ylabel('Ток, А')
        ax.set_zlabel('Температура, °C')
        ax.set_title('Профили температуры для разных рабочих токов')
        ax.legend()

        if save:
            filepath = self.output_dir / '3d_temperature_profiles.png'
            plt.savefig(filepath, dpi=300)

        if show:
            plt.show()
        else:
            plt.close()