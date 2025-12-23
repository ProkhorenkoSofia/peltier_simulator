"""
Физические константы и параметры термоэлектрического модуля
"""
import numpy as np


class PhysicalConstants:
    """Фундаментальные физические константы"""
    BOLTZMANN = 1.380649e-23  # Дж/К
    ELECTRON_CHARGE = 1.602176634e-19  # Кл


class MaterialProperties:
    """Свойства материалов для висмут-теллурида (Bi2Te3)"""

    def __init__(self, config: dict):
        """Инициализация параметров из конфигурации"""
        # Параметры модуля
        self.N = int(config['module']['pairs'])
        self.L = float(config['module']['leg_height'])
        self.A = float(config['module']['leg_area'])

        # Термоэлектрические коэффициенты
        self.alpha_n = float(config['module']['alpha_n'])  # n-тип
        self.alpha_p = float(config['module']['alpha_p'])  # p-тип
        self.alpha_pn = float(self.alpha_p - self.alpha_n)  # дифференциальный

        # Теплофизические свойства (средние для пары)
        self.sigma = float(config['module']['sigma'])
        self.lambda_coeff = float(config['module']['lambda_coeff'])
        self.tau = float(config['module']['tau'])
        self.rho = float(config['module']['rho'])
        self.cp = float(config['module']['cp'])

        # Расчетные величины
        self.R_pair = self.L / (self.sigma * self.A)  # Сопротивление одной ветви
        self.R_total = 2.0 * self.N * self.R_pair  # Полное сопротивление модуля (n+p)

    def __str__(self):
        """Строковое представление параметров"""
        return f"""
Параметры термоэлектрического модуля:
--------------------------------------
Количество пар: {self.N}
Высота ветви: {self.L * 1000:.2f} мм
Сечение ветви: {self.A * 1e6:.2f} мм²
Термо-ЭДС пары (α_pn): {self.alpha_pn * 1e3:.2f} мВ/К
Сопротивление пары: {self.R_pair * 1e3:.2f} мОм
Полное сопротивление модуля: {self.R_total:.3f} Ом
Теплопроводность: {self.lambda_coeff:.2f} Вт/(м·К)
Коэффициент Томсона: {self.tau:.2e} В/К
"""
