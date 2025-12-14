import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from typing import List, Callable

# ==================== 1. Перечисления и константы ====================
class BodyType(Enum):
    CIRCLE = "circle"
    SQUARE = "square"

class Quarter(Enum):
    FIRST = 1
    SECOND = 2
    THIRD = 3
    FOURTH = 4

# ==================== 2. Класс материальной точки ====================
class MaterialPoint:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.trajectory_x = [x]
        self.trajectory_y = [y]

    def update(self, new_x: float, new_y: float):
        self.x = new_x
        self.y = new_y
        self.trajectory_x.append(new_x)
        self.trajectory_y.append(new_y)

# ==================== 3. Класс тела (круг или квадрат) ====================
class Body:
    def __init__(self, body_type: BodyType, size: float, quarter: Quarter, num_points: int = 100):
        self.type = body_type
        self.size = size
        self.quarter = quarter
        self.points = self._generate_points(num_points)

    def _generate_points(self, num_points: int) -> List[MaterialPoint]:
        points = []
        if self.type == BodyType.CIRCLE:
            # Генерация точек окружности во 2-й четверти (x <= 0, y >= 0)
            angles = np.linspace(np.pi/2, np.pi, num_points)
            for angle in angles:
                x = self.size * np.cos(angle)
                y = self.size * np.sin(angle)
                # Сдвигаем вглубь четверти, чтобы не касаться осей
                x -= 0.5
                y += 0.5
                points.append(MaterialPoint(x, y))
        else:  # SQUARE
            # Квадрат стороной size во 2-й четверти
            # Углы квадрата (левый нижний = (-size, 0), правый верхний = (0, size))
            # Сдвигаем от осей
            shift = 0.5
            x_vals = np.linspace(-self.size - shift, -shift, int(np.sqrt(num_points)))
            y_vals = np.linspace(shift, self.size + shift, int(np.sqrt(num_points)))
            for x in x_vals:
                for y in y_vals:
                    points.append(MaterialPoint(x, y))
        return points

# ==================== 4. Класс поля скоростей ====================
class VelocityField:
    def __init__(self, A: Callable[[float], float], B: Callable[[float], float]):
        self.A = A
        self.B = B

    def get_velocity(self, x: float, y: float, t: float) -> tuple[float, float]:
        vx = -self.A(t) * x
        vy = self.B(t) * y
        return vx, vy

# ==================== 5. Интегратор Рунге–Кутты ====================
class ButcherTable:
    def __init__(self, c: np.ndarray, a: np.ndarray, b: np.ndarray):
        self.c = c  # веса для времени
        self.a = a  # матрица коэффициентов
        self.b = b  # веса для результата

# Таблица Бутчера 2.2 (правило средних точек)
BUTCHER_2_2 = ButcherTable(
    c=np.array([0, 1]),
    a=np.array([[0, 0],
                [1, 0]]),
    b=np.array([0.5, 0.5])
)

class RungeKuttaSolver:
    def __init__(self, butcher_table: ButcherTable):
        self.table = butcher_table

    def solve_step(self, field: VelocityField, point: MaterialPoint, t: float, dt: float) -> tuple[float, float]:
        x, y = point.x, point.y
        kx = np.zeros(len(self.table.c))
        ky = np.zeros(len(self.table.c))

        for i, ci in enumerate(self.table.c):
            sum_ax = np.sum([self.table.a[i][j] * kx[j] for j in range(i)])
            sum_ay = np.sum([self.table.a[i][j] * ky[j] for j in range(i)])
            xi = x + dt * sum_ax
            yi = y + dt * sum_ay
            ti = t + dt * ci
            kx[i], ky[i] = field.get_velocity(xi, yi, ti)

        new_x = x + dt * np.dot(self.table.b, kx)
        new_y = y + dt * np.dot(self.table.b, ky)
        return new_x, new_y

# ==================== 6. Основной расчетный класс ====================
class DeformationSimulator:
    def __init__(self, body: Body, field: VelocityField, solver: RungeKuttaSolver):
        self.body = body
        self.field = field
        self.solver = solver

    def simulate(self, t_start: float, t_end: float, dt: float):
        time_points = np.arange(t_start, t_end + dt, dt)
        for t in time_points:
            for point in self.body.points:
                new_x, new_y = self.solver.solve_step(self.field, point, t, dt)
                point.update(new_x, new_y)

# ==================== 7. Визуализация ====================
def plot_results(body: Body, field: VelocityField, t: float, x_lim: tuple, y_lim: tuple):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Траектории всех точек
    ax = axes[0, 0]
    for point in body.points:
        ax.plot(point.trajectory_x, point.trajectory_y, 'b-', alpha=0.3)
    ax.set_title("Траектории материальных точек")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(True)
    ax.axis('equal')

    # 2. Начальная и конечная форма тела
    ax = axes[0, 1]
    initial_x = [p.trajectory_x[0] for p in body.points]
    initial_y = [p.trajectory_y[0] for p in body.points]
    final_x = [p.trajectory_x[-1] for p in body.points]
    final_y = [p.trajectory_y[-1] for p in body.points]
    ax.scatter(initial_x, initial_y, c='blue', label='Начальное положение', alpha=0.5)
    ax.scatter(final_x, final_y, c='red', label='Конечное положение', alpha=0.5)
    ax.set_title("Начальная и конечная форма тела")
    ax.legend()
    ax.grid(True)
    ax.axis('equal')

    # 3. Поле скоростей в момент времени t
    ax = axes[1, 0]
    x_grid, y_grid = np.meshgrid(np.linspace(x_lim[0], x_lim[1], 15),
                                 np.linspace(y_lim[0], y_lim[1], 15))
    vx_grid = np.zeros_like(x_grid)
    vy_grid = np.zeros_like(y_grid)
    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            vx, vy = field.get_velocity(x_grid[i, j], y_grid[i, j], t)
            vx_grid[i, j] = vx
            vy_grid[i, j] = vy
    ax.quiver(x_grid, y_grid, vx_grid, vy_grid, color='green')
    ax.set_title(f"Поле скоростей в t = {t:.2f}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(True)
    ax.axis('equal')

    # 4. Линии тока в момент времени t
    ax = axes[1, 1]
    # Для построения линий тока используем streamplot
    # Нужно более детальное сеточное поле
    x_dense, y_dense = np.meshgrid(np.linspace(x_lim[0], x_lim[1], 30),
                                   np.linspace(y_lim[0], y_lim[1], 30))
    vx_dense = np.zeros_like(x_dense)
    vy_dense = np.zeros_like(y_dense)
    for i in range(x_dense.shape[0]):
        for j in range(x_dense.shape[1]):
            vx, vy = field.get_velocity(x_dense[i, j], y_dense[i, j], t)
            vx_dense[i, j] = vx
            vy_dense[i, j] = vy
    ax.streamplot(x_dense, y_dense, vx_dense, vy_dense, color='purple', density=1.5)
    ax.set_title(f"Линии тока в t = {t:.2f}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(True)
    ax.axis('equal')

    plt.tight_layout()
    plt.show()

# ==================== 8. Основной скрипт ====================
if __name__ == "__main__":
    # Параметры Подгруппы 1
    A_func = lambda t: np.log(t)  # ln(t)
    B_func = lambda t: -np.exp(t)  # -e^t

    # Создаем поле скоростей
    field = VelocityField(A_func, B_func)

    # Создаем тело (круг радиуса 1 во 2-й четверти)
    body = Body(BodyType.CIRCLE, size=1.0, quarter=Quarter.SECOND, num_points=50)

    # Создаем решатель Рунге–Кутты
    solver = RungeKuttaSolver(BUTCHER_2_2)

    # Создаем симулятор
    simulator = DeformationSimulator(body, field, solver)

    # Параметры интегрирования
    t_start = 1.0  # начинаем с t=1, так как ln(t) при t=0 не определён
    t_end = 2.5    # конечное время
    dt = 0.01      # шаг интегрирования

    # Запускаем симуляцию
    simulator.simulate(t_start, t_end, dt)

    # Визуализируем результаты для нескольких моментов времени
    plot_times = [t_start, (t_start + t_end)/2, t_end]
    x_lim = (-3, 1)  # пределы по x (2-я четверть)
    y_lim = (0, 4)   # пределы по y

    for t_plot in plot_times:
        plot_results(body, field, t_plot, x_lim, y_lim)