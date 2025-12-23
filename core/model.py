"""
Основной класс математической модели элемента Пельтье
"""
import numpy as np
from typing import Tuple, Optional
from .solver import thomas_algorithm, validate_tridiagonal_system


class Peltier1D:
    """
    Одномерная нестационарная модель элемента Пельтье
    """

    def __init__(self, material_props, config: dict):
        """
        Инициализация модели
        """
        self.mp = material_props
        self.config = config

        # Явное преобразование к float
        self.Nx = int(config['numerics']['nodes'])
        self.dx = float(self.mp.L) / float(self.Nx - 1)
        self.x = np.linspace(0, float(self.mp.L), self.Nx, dtype=float)

        # Параметры времени
        self.dt = float(config['numerics']['time_step'])
        self.max_time = float(config['numerics']['max_time'])
        self.tolerance = float(config['numerics']['tolerance'])

        # Граничные условия
        self.h_cold = float(config['simulation']['h_cold'])
        self.h_hot = float(config['simulation']['h_hot'])
        self.T_env_cold = float(config['simulation']['T_env_cold'])
        self.T_env_hot = float(config['simulation']['T_env_hot'])

        # Рабочие переменные
        self.current = 0.0
        self.j = 0.0  # плотность тока
        self.T = None  # распределение температуры
        self.T_old = None
        self.time_elapsed = 0.0
        self.is_stationary = False

        # Результаты
        self.temperature_history = []
        self.time_history = []
        self.performance = {}

    def set_current(self, current: float):
        """Установка рабочего тока"""
        self.current = current
        self.j = current / self.mp.A  # плотность тока

    def initialize_temperature(self, T0: Optional[float] = None):
        """Инициализация температурного поля"""
        if T0 is None:
            T0 = self.config['simulation']['initial_T']
        self.T = np.ones(self.Nx) * T0
        self.T_old = self.T.copy()

        # Очистка истории
        self.temperature_history = [self.T.copy()]
        self.time_history = [0.0]
        self.is_stationary = False
        self.time_elapsed = 0.0

    def _assemble_system(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Формирование трехдиагональной системы для неявной схемы.
        """
        N = self.Nx
        a = np.zeros(N, dtype=float)  # Явно указываем тип float
        b = np.zeros(N, dtype=float)
        c = np.zeros(N, dtype=float)
        d = np.zeros(N, dtype=float)

        # Явное преобразование параметров к float
        lambda_coeff = float(self.mp.lambda_coeff)
        rho = float(self.mp.rho)
        cp = float(self.mp.cp)
        tau = float(self.mp.tau)
        sigma = float(self.mp.sigma)

        # Коэффициенты для внутренних узлов
        r = lambda_coeff * self.dt / (rho * cp * self.dx ** 2)
        s = self.dt / (rho * cp)
        thomson_coeff = tau * self.j * self.dt / (rho * cp * 2 * self.dx)
        joule_source = (self.j ** 2) / sigma

        # Внутренние узлы (i = 1 .. N-2)
        for i in range(1, N - 1):
            a[i] = -r + thomson_coeff
            b[i] = 1.0 + 2.0 * r
            c[i] = -r - thomson_coeff
            d[i] = self.T_old[i] + s * joule_source

        # Граничные условия (холодный спай, i = 0)
        A_coeff = lambda_coeff / self.dx
        Peltier_coeff_cold = float(self.mp.alpha_pn) * self.current

        a[0] = 0.0
        b[0] = A_coeff + self.h_cold * self.mp.A + Peltier_coeff_cold
        c[0] = -A_coeff
        d[0] = self.h_cold * self.mp.A * self.T_env_cold

        # Граничные условия (горячий спай, i = N-1)
        Peltier_coeff_hot = float(self.mp.alpha_pn) * self.current

        a[-1] = -A_coeff
        b[-1] = A_coeff + self.h_hot * self.mp.A - Peltier_coeff_hot
        c[-1] = 0.0
        d[-1] = self.h_hot * self.mp.A * self.T_env_hot

        return a, b, c, d

    def solve_time_step(self) -> bool:
        """
        Выполнение одного шага по времени.

        Возвращает:
        -----------
        converged : bool
            Флаг достижения стационарного состояния
        """
        if self.is_stationary:
            return True

        # Сохраняем предыдущее решение
        self.T_old = self.T.copy()

        # Формируем и решаем систему
        a, b, c, d = self._assemble_system()
        self.T = thomas_algorithm(a, b, c, d)

        # Проверка сходимости
        max_diff = np.max(np.abs(self.T - self.T_old))

        # Обновление времени и истории
        self.time_elapsed += self.dt
        self.time_history.append(self.time_elapsed)
        self.temperature_history.append(self.T.copy())

        if max_diff < self.tolerance:
            self.is_stationary = True
            print(f"Стационарный режим достигнут за {self.time_elapsed:.3f} с")

        if self.time_elapsed >= self.max_time:
            self.is_stationary = True
            print(f"Достигнуто максимальное время {self.max_time} с")

        return self.is_stationary

    def solve_transient(self, progress_callback=None) -> float:
        """
        Решение нестационарной задачи до стационарного состояния.

        Параметры:
        ----------
        progress_callback : callable, optional
            Функция для отображения прогресса

        Возвращает:
        -----------
        time_to_steady : float
            Время выхода на стационарный режим
        """
        step = 0
        while not self.is_stationary and self.time_elapsed < self.max_time:
            self.solve_time_step()
            step += 1

            if progress_callback and step % 10 == 0:
                progress_callback(self.time_elapsed / self.max_time)

        return self.time_elapsed

    def calculate_performance(self) -> dict:
        """
        Расчет рабочих характеристик по стационарному профилю температуры.

        Возвращает:
        -----------
        performance : dict
            Словарь с характеристиками
        """
        if not self.is_stationary:
            raise RuntimeError("Расчет характеристик возможен только для стационарного режима")

        T_cold = self.T[0]
        T_hot = self.T[-1]

        # Тепловой поток на холодной стороне (формула 2.9)
        dT_dx_cold = (self.T[1] - self.T[0]) / self.dx
        Q_peltier_cold = self.mp.alpha_pn * self.current * T_cold
        Q_cond_cold = self.mp.lambda_coeff * self.mp.A * dT_dx_cold
        Q_cold = Q_peltier_cold - Q_cond_cold

        # Электрическая мощность (формула 2.10)
        P_joule = self.current ** 2 * self.mp.R_total
        P_seebeck = self.mp.alpha_pn * self.current * (T_hot - T_cold)
        P_total = P_joule + P_seebeck

        # Коэффициент эффективности (формула 2.11)
        COP = Q_cold / P_total if P_total > 0 else 0.0

        self.performance = {
            'T_cold': T_cold,
            'T_hot': T_hot,
            'delta_T': T_hot - T_cold,
            'Q_cold': Q_cold,
            'P_joule': P_joule,
            'P_seebeck': P_seebeck,
            'P_total': P_total,
            'COP': COP,
            'time_to_steady': self.time_elapsed
        }

        return self.performance

    def get_results(self) -> dict:
        """Получение всех результатов моделирования"""
        return {
            'x': self.x,
            'temperature': self.T,
            'temperature_history': np.array(self.temperature_history),
            'time_history': np.array(self.time_history),
            'performance': self.performance,
            'is_stationary': self.is_stationary,
            'parameters': {
                'current': self.current,
                'N_pairs': self.mp.N,
                'leg_height': self.mp.L,
                'leg_area': self.mp.A
            }
        }
