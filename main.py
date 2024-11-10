import pygame
import numpy as np
import random

# Инициализация Pygame
pygame.init()

# Константы
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SHEEP_SIZE = 40
CAR_SIZE = 50
LANE_WIDTH = 100
NUM_LANES = 4
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9995
RENDER_EVERY = 1000

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
GRAY = (128, 128, 128)
YELLOW = (255, 255, 0)

class Environment:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Овечка переходит дорогу")
        
        # Начальная позиция овечки (слева)
        self.sheep_x = 50
        self.sheep_y = SCREEN_HEIGHT // 2
        
        # Дорожные полосы
        self.lanes = []
        lane_start = (SCREEN_WIDTH - (NUM_LANES * LANE_WIDTH)) // 2
        for i in range(NUM_LANES):
            self.lanes.append(lane_start + i * LANE_WIDTH)
        
        # Список машин для каждой полосы
        self.cars = [[] for _ in range(NUM_LANES)]
        
        # Q-таблица
        self.q_table = {}
        
        # Для отслеживания прогресса обучения
        self.episode_rewards = []
        
        self.render_mode = False  # Добавляем флаг для контроля отрисовки
        
        self.sheep_positions = []  # Добавляем список для х��анения позиций овец
        self.current_episode = 0  # Добавляем счетчик эпизодов
        
        self.last_lane = -1  # Добавляем отслеживание последней полосы
        self.crossed_lanes = set()  # Множество пересеченных полос в текущем эпизоде
    
    def get_state(self):
        state = []
        sheep_lane = self.get_current_lane()
        
        for lane_idx, lane_cars in enumerate(self.cars):
            nearest_car_dist = float('inf')
            for car_y in lane_cars:
                dist = car_y - self.sheep_y
                # Расширяем зону видимости
                if -CAR_SIZE * 3 < dist < CAR_SIZE * 5:
                    if abs(dist) < abs(nearest_car_dist):
                        nearest_car_dist = dist
            
            if nearest_car_dist == float('inf'):
                state.append(0)  # Нет машин
            else:
                # Более детальная дискретизация
                if abs(nearest_car_dist) < CAR_SIZE * 0.5:
                    state.append(1)  # Критически близко
                elif abs(nearest_car_dist) < CAR_SIZE:
                    state.append(2)  # Очень близко
                elif abs(nearest_car_dist) < CAR_SIZE * 2:
                    state.append(3)  # Средняя дистанция
                elif abs(nearest_car_dist) < CAR_SIZE * 3:
                    state.append(4)  # Безопасная дистанция
                else:
                    state.append(5)  # Далеко
            
            # Добавляем направление движения ближайшей машины
            if nearest_car_dist != float('inf'):
                state.append(1 if nearest_car_dist > 0 else -1)
            else:
                state.append(0)
        
        # Добавляем текущую полосу и относительную позицию до цели
        state.append(sheep_lane)
        state.append(min(3, int((SCREEN_WIDTH - self.sheep_x) / (SCREEN_WIDTH / 4))))
        
        return tuple(state)
    
    def get_current_lane(self):
        # Определяем, в какой полосе находится овечка
        for i, lane_x in enumerate(self.lanes):
            if lane_x <= self.sheep_x < lane_x + LANE_WIDTH:
                return i
        return -1 if self.sheep_x < self.lanes[0] else len(self.lanes)
    
    def take_action(self, action):
        # Действия: 0 - влево, 1 - вправо, 2 - в��ерх, 3 - вниз
        old_x = self.sheep_x
        old_y = self.sheep_y
        old_lane = self.get_current_lane()
        
        # Увеличиваем шаг движения для более решительных действий
        step_size = 10  # Было 5
        
        if action == 0:
            self.sheep_x = max(0, self.sheep_x - step_size)
        elif action == 1:
            self.sheep_x = min(SCREEN_WIDTH - SHEEP_SIZE, self.sheep_x + step_size)
        elif action == 2:
            self.sheep_y = max(0, self.sheep_y - step_size)
        elif action == 3:
            self.sheep_y = min(SCREEN_HEIGHT - SHEEP_SIZE, self.sheep_y + step_size)
            
        # Проверка столкновений
        current_lane = self.get_current_lane()
        if 0 <= current_lane < len(self.lanes):
            for car_y in self.cars[current_lane]:
                if abs(self.sheep_y - car_y) < CAR_SIZE:
                    return -100  # Смерть от столкновения
        
        reward = 0
        
        # Увеличиваем награду за пересечение новой полосы
        if current_lane != old_lane and current_lane not in self.crossed_lanes:
            if 0 <= current_lane < len(self.lanes):
                reward += 100  # Было 50
                self.crossed_lanes.add(current_lane)
        
        # Награда за достижение цели
        if self.sheep_x > SCREEN_WIDTH - SHEEP_SIZE:
            return reward + 500  # Увеличиваем награду за достижение цели
            
        # Поощряем движение вправо и наказываем за движение влево
        if self.sheep_x > old_x:
            reward += 30  # Увеличиваем награду за движение вправо
        elif self.sheep_x < old_x:
            reward -= 10  # Увеличиваем штраф за движение влево
        
        # Штраф за нахождение на месте
        if abs(self.sheep_x - old_x) < 1 and abs(self.sheep_y - old_y) < 1:
            reward -= 5
        
        return reward
        
    def update_cars(self):
        # Обновление позиций машин
        for lane_idx in range(len(self.cars)):
            # Удаление машин, выехавших за пределы экрана
            self.cars[lane_idx] = [y - 3 for y in self.cars[lane_idx] if y > -CAR_SIZE]
            
            # Уменьшаем вероятность появления новых машин
            if random.random() < 0.015:  # Было 0.01
                # Проверяем расстояние до последней машины в полосе
                if not self.cars[lane_idx] or SCREEN_HEIGHT - self.cars[lane_idx][-1] > CAR_SIZE * 3:
                    self.cars[lane_idx].append(SCREEN_HEIGHT)
        
    def render(self):
        if not self.render_mode:
            return
        self.screen.fill(WHITE)
        
        # Отрисовка дороги
        road_start = self.lanes[0]
        road_width = LANE_WIDTH * NUM_LANES
        pygame.draw.rect(self.screen, GRAY, 
                        (road_start, 0, road_width, SCREEN_HEIGHT))
        
        # Отрисовка разделительных полос
        for lane_x in self.lanes:
            pygame.draw.line(self.screen, YELLOW, 
                           (lane_x, 0), (lane_x, SCREEN_HEIGHT), 2)
        
        # Отрисовка траекторий предыдущих овец (полупрозрачным серым)
        for positions in self.sheep_positions:
            for pos in positions:
                pygame.draw.rect(self.screen, (200, 200, 200), 
                               (pos[0], pos[1], SHEEP_SIZE, SHEEP_SIZE), 1)
        
        # Отрисовка текущей овечки
        pygame.draw.rect(self.screen, BLACK, 
                        (self.sheep_x, self.sheep_y, SHEEP_SIZE, SHEEP_SIZE))
        
        # Отрисовка машин
        for lane_idx, lane_cars in enumerate(self.cars):
            lane_x = self.lanes[lane_idx]
            for car_y in lane_cars:
                pygame.draw.rect(self.screen, RED,
                               (lane_x + (LANE_WIDTH - CAR_SIZE) // 2, 
                                car_y, CAR_SIZE, CAR_SIZE))
        
        # Отображение номера эпизода
        font = pygame.font.Font(None, 36)
        text = font.render(f'Episode: {self.current_episode}', True, BLACK)
        self.screen.blit(text, (10, 10))
        
        pygame.display.flip()

def train():
    env = Environment()
    episodes = 200000
    epsilon = EPSILON_START
    max_steps = 500  # Уменьшаем максимальное количество шагов
    
    best_reward = float('-inf')
    best_q_table = None
    
    for episode in range(episodes):
        env.current_episode = episode
        env.render_mode = (episode % RENDER_EVERY == 0)
        
        # Сброс для нового эпизода
        env.crossed_lanes = set()  # Очищаем множество пересеченных полос
        env.last_lane = -1
        
        # Очищаем список позиций для новой эпохи
        if env.render_mode:
            env.sheep_positions = []
        
        current_episode_positions = []  # Позиции овцы в текущей эпохе
        
        env.sheep_x = 50
        env.sheep_y = SCREEN_HEIGHT // 2
        state = env.get_state()
        total_reward = 0
        steps = 0
        
        while steps < max_steps:
            if env.render_mode:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                # Сохраняем текущую позицию овцы
                current_episode_positions.append((env.sheep_x, env.sheep_y))
            
            # Выбор действия (оптимизированный)
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                state_actions = env.q_table.get(state, [0] * 4)
                action = np.argmax(state_actions)
            
            # Выполнение действия
            reward = env.take_action(action)
            env.update_cars()
            new_state = env.get_state()
            total_reward += reward
            
            # Оптимизированное Q-learning обновление
            if state not in env.q_table:
                env.q_table[state] = [0] * 4
            if new_state not in env.q_table:
                env.q_table[new_state] = [0] * 4
            
            # Оптимизированная формула Q-learning
            old_value = env.q_table[state][action]
            next_max = max(env.q_table[new_state])  # Используем max вместо np.max
            env.q_table[state][action] = old_value + LEARNING_RATE * (
                reward + DISCOUNT_FACTOR * next_max - old_value)
            
            state = new_state
            steps += 1
            
            # Завершаем эпизод при столкновении или достижении цели
            if reward <= -100 or reward >= 200:  # Смерть или победа
                if env.render_mode:
                    # Добавляем траекторию текущей овцы
                    env.sheep_positions.append(current_episode_positions)
                    # Оставляем только последние 5 траекторий
                    env.sheep_positions = env.sheep_positions[-5:]
                break
            
            # Отрисовка и задержка только при необходимости
            if env.render_mode:
                env.render()
                pygame.time.delay(20)
            
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        env.episode_rewards.append(total_reward)
        
        # Вывод прогресса реже
        if episode % 100 == 0:
            avg_reward = sum(env.episode_rewards[-100:]) / min(100, len(env.episode_rewards))
            print(f"Episode: {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.2f}")
        
        # Сохраняем лучшую Q-таблицу
        if total_reward > best_reward:
            best_reward = total_reward
            best_q_table = env.q_table.copy()
        
        # Каждые 1000 эпизодов восстанавливаем лучшую Q-таблицу
        if episode % 1000 == 0 and best_q_table is not None:
            env.q_table = best_q_table.copy()

if __name__ == "__main__":
    train()