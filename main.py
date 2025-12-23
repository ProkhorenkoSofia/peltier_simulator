"""
Главный скрипт для управления моделированием элемента Пельтье
"""
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from core.constants import MaterialProperties
from core.model import Peltier1D
from visualization.plotter import PeltierPlotter


def load_config(config_path='config.yaml'):
    """Загрузка конфигурации из YAML файла с преобразованием типов"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Явное преобразование числовых значений
    def convert_numbers(obj):
        if isinstance(obj, dict):
            return {k: convert_numbers(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numbers(item) for item in obj]
        elif isinstance(obj, str):
            # Пробуем преобразовать строку в число, если возможно
            try:
                # Пробуем преобразовать в float
                return float(obj)
            except ValueError:
                try:
                    # Пробуем преобразовать в int
                    return int(obj)
                except ValueError:
                    # Если не получается, оставляем строкой
                    return obj
        else:
            return obj

    return convert_numbers(config)


def run_single_simulation(config, current, output_data=True):
    """
    Выполнение одного расчета для заданного тока.

    Параметры:
    ----------
    config : dict
        Конфигурация моделирования
    current : float
        Рабочий ток
    output_data : bool
        Сохранять ли данные в файл

    Возвращает:
    -----------
    results_dict : dict
        Результаты моделирования
    """
    print(f"\n{'=' * 60}")
    print(f"Моделирование для I = {current:.2f} А")
    print('=' * 60)

    # Инициализация
    material = MaterialProperties(config)
    model = Peltier1D(material, config)

    print(material)

    # Установка тока и начальных условий
    model.set_current(current)
    model.initialize_temperature()

    # Решение нестационарной задачи
    print("Решение нестационарной задачи...")
    time_to_steady = model.solve_transient(
        progress_callback=lambda p: print(f"\rПрогресс: {p * 100:.1f}%", end='')
    )
    print()

    # Расчет характеристик
    performance = model.calculate_performance()

    # Вывод результатов
    print("\nРезультаты:")
    print(f"  Время выхода на стационарный режим: {time_to_steady:.3f} с")
    print(f"  Температура холодного спая: {performance['T_cold'] - 273.15:.2f} °C")
    print(f"  Температура горячего спая: {performance['T_hot'] - 273.15:.2f} °C")
    print(f"  Перепад температур: {performance['delta_T']:.2f} K")
    print(f"  Холодопроизводительность: {performance['Q_cold']:.2f} Вт")
    print(f"  Потребляемая мощность: {performance['P_total']:.2f} Вт")
    print(f"  Коэффициент эффективности (COP): {performance['COP']:.3f}")

    # Сохранение данных
    results = model.get_results()

    if output_data:
        save_simulation_data(results, current)

    return results


def save_simulation_data(results, current):
    """Сохранение результатов в CSV файл"""
    output_dir = Path('output/data')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Сохранение профиля температуры
    df_temp = pd.DataFrame({
        'x_m': results['x'],
        'x_mm': results['x'] * 1000,
        'T_K': results['temperature'],
        'T_C': results['temperature'] - 273.15
    })

    filename = f'temperature_profile_I{current:.1f}A.csv'
    filepath = output_dir / filename
    df_temp.to_csv(filepath, index=False, encoding='utf-8')
    print(f"  Данные сохранены: {filepath}")

    # Сохранение характеристик
    if hasattr(results, 'performance_history'):
        df_perf = pd.DataFrame(results['performance_history'])
        perf_filename = f'performance_history_I{current:.1f}A.csv'
        perf_filepath = output_dir / perf_filename
        df_perf.to_csv(perf_filepath, index=False, encoding='utf-8')


def run_parameter_study(config):
    """
    Исследование зависимости характеристик от тока.

    Параметры:
    ----------
    config : dict
        Конфигурация моделирования

    Возвращает:
    -----------
    study_results : list
        Результаты для всех токов
    """
    study_config = config['study']['current_range']
    currents = np.linspace(study_config['start'],
                           study_config['stop'],
                           study_config['steps'])

    print("\n" + "=" * 60)
    print("ПАРАМЕТРИЧЕСКОЕ ИССЛЕДОВАНИЕ")
    print("Зависимость характеристик от рабочего тока")
    print("=" * 60)

    all_results = []
    performance_table = []

    for I in currents:
        results = run_single_simulation(config, I, output_data=True)
        all_results.append(results)

        perf = results['performance']
        performance_table.append({
            'Ток, А': I,
            'T_хол, °C': perf['T_cold'] - 273.15,
            'T_гор, °C': perf['T_hot'] - 273.15,
            'ΔT, K': perf['delta_T'],
            'Q_хол, Вт': perf['Q_cold'],
            'P_эл, Вт': perf['P_total'],
            'COP': perf['COP']
        })

    # Сохранение сводной таблицы
    df_summary = pd.DataFrame(performance_table)
    summary_path = Path('output/data') / 'performance_summary.csv'
    df_summary.to_csv(summary_path, index=False, encoding='utf-8')
    print(f"\nСводная таблица сохранена: {summary_path}")

    # Вывод таблицы в консоль
    print("\nСводная таблица результатов:")
    print("-" * 80)
    print(df_summary.to_string(index=False))
    print("-" * 80)

    return all_results


def main():
    """Основная функция программы"""
    print("=" * 60)
    print("МОДЕЛИРОВАНИЕ ЭЛЕМЕНТА ПЕЛЬТЬЕ")
    print("Одномерная нестационарная модель")
    print(f"Дата запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Загрузка конфигурации
    try:
        config = load_config()
        print("Конфигурация загружена успешно")
    except FileNotFoundError:
        print("Ошибка: файл config.yaml не найден")
        print("Создайте файл конфигурации или укажите правильный путь")
        return

    # Инициализация визуализатора
    plotter = PeltierPlotter()

    # Выбор режима работы
    print("\nРежимы работы:")
    print("1. Единичный расчет (текущий ток из config.yaml)")
    print("2. Параметрическое исследование (зависимость от тока)")
    print("3. Демонстрационный режим (предустановленные сценарии)")

    try:
        choice = int(input("\nВыберите режим (1-3): "))
    except ValueError:
        print("Неверный ввод. Запускается демонстрационный режим.")
        choice = 3

    if choice == 1:
        # Единичный расчет
        current = config['simulation']['current']
        results = run_single_simulation(config, current, output_data=True)

        # Построение графиков
        plotter.plot_temperature_profile(results, show=config['study']['plot_live'])
        plotter.plot_transient_process(results, show=config['study']['plot_live'])

    elif choice == 2:
        # Параметрическое исследование
        study_results = run_parameter_study(config)

        # Построение графиков
        plotter.plot_performance_curves(study_results,
                                        show=config['study']['plot_live'])

        # Создание словаря для 3D графика
        multi_results = {}
        for res in study_results:
            I = res['parameters']['current']
            multi_results[I] = res

        plotter.plot_3d_temperature_surface(multi_results,
                                            show=config['study']['plot_live'])

    else:
        # Демонстрационный режим
        print("\nДемонстрационный режим: расчет для трех значений тока")
        demo_currents = [2.0, 4.0, 6.0]
        demo_results = []

        for I in demo_currents:
            results = run_single_simulation(config, I, output_data=False)
            demo_results.append(results)
            plotter.plot_temperature_profile(results, show=False)
            plotter.plot_transient_process(results, show=False)

        plotter.plot_performance_curves(demo_results, show=True)

    print("\n" + "=" * 60)
    print("МОДЕЛИРОВАНИЕ ЗАВЕРШЕНО")
    print(f"Результаты сохранены в папке: {Path('output').absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    # Создание необходимых директорий
    Path('output/plots').mkdir(parents=True, exist_ok=True)
    Path('output/data').mkdir(parents=True, exist_ok=True)

    # Запуск программы
    main()