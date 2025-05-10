import random
import pickle
from map import FIXED_MAP

# Constants
ACTIONS = ['up', 'down', 'left', 'right', 'bomb']
LEARNING_RATE = 0.1  
DISCOUNT_FACTOR = 0.9  
EPISODES = 10000 
MAX_STEPS = 100  
EPSILON_START = 1.0 
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01

class Bomb:
    def __init__(self, x, y, timer=3, is_own_bomb=False):
        self.x = x
        self.y = y
        self.timer = timer
        self.is_own_bomb = is_own_bomb 

    def explode(self, map_data):
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < len(map_data) and 0 <= ny < len(map_data[0]):
                if map_data[nx][ny] == 2:  
                    map_data[nx][ny] = 0

class QLearningAgent:
    def __init__(self, map_data):
        self.original_map = [row[:] for row in map_data]
        self.q_table = {}
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE
        self.discount_factor = DISCOUNT_FACTOR
        self.start_positions = [(11, 11), (1, 11), (11, 1)]
        self.best_reward = float('-inf')
        self.best_q_table = None

    def reset(self):
        self.map = [row[:] for row in self.original_map]
        self.start_pos = random.choice(self.start_positions)
        self.pos_x, self.pos_y = self.start_pos
        self.player_pos = (1, 1)
        self.has_bomb = True
        self.bombs = []
        self.players_killed = 0
        self.total_reward = 0

    def get_state(self):
        player_dx = self.player_pos[0] - self.pos_x
        player_dy = self.player_pos[1] - self.pos_y
        
        box_count = sum(1 for i in range(len(self.map)) 
                       for j in range(len(self.map[i])) 
                       if self.map[i][j] == 2)
        
        nearest_bomb = float('inf')
        bomb_direction = 0
        escape_routes = 0
        bomb_sectors = set()
        own_bomb_nearby = False
        enemy_bomb_nearby = False
        
        # Tính toán vùng ảnh hưởng của bom
        for bomb in self.bombs:
            bomb_sectors.add((bomb.x, bomb.y))
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = bomb.x + dx, bomb.y + dy
                if 0 <= nx < len(self.map) and 0 <= ny < len(self.map[0]):
                    if self.map[nx][ny] != 1:
                        bomb_sectors.add((nx, ny))
                        if self.map[nx][ny] == 2:
                            break
        
        # Kiểm tra bom gần nhất và phân biệt bom của mình/địch
        for bomb in self.bombs:
            dist = abs(self.pos_x - bomb.x) + abs(self.pos_y - bomb.y)
            if dist < nearest_bomb:
                nearest_bomb = dist
                if bomb.is_own_bomb:
                    own_bomb_nearby = True
                else:
                    enemy_bomb_nearby = True
                if bomb.x < self.pos_x:
                    bomb_direction = 1
                elif bomb.x > self.pos_x:
                    bomb_direction = 3
                elif bomb.y < self.pos_y:
                    bomb_direction = 4
                else:
                    bomb_direction = 2
        
        # Kiểm tra đường thoát an toàn
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = self.pos_x + dx, self.pos_y + dy
            if 0 <= nx < len(self.map) and 0 <= ny < len(self.map[0]):
                if self.map[nx][ny] == 0:
                    is_safe = True
                    for bx, by in bomb_sectors:
                        if abs(nx - bx) + abs(ny - by) <= 1:
                            is_safe = False
                            break
                    if is_safe:
                        escape_routes += 1
        
        return (self.pos_x, self.pos_y, 
                player_dx, player_dy,
                int(self.has_bomb), 
                int(abs(player_dx) + abs(player_dy) <= 2),
                int(box_count > 0),
                int(nearest_bomb <= 2),
                bomb_direction,
                escape_routes,
                int(own_bomb_nearby),
                int(enemy_bomb_nearby))

    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0] * len(ACTIONS)
        
        # Kiểm tra và né bom
        for bomb in self.bombs:
            bomb_dist = abs(self.pos_x - bomb.x) + abs(self.pos_y - bomb.y)
            if bomb_dist <= 1:  # Tầm bom
                safe_directions = []
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = self.pos_x + dx, self.pos_y + dy
                    if 0 <= nx < len(self.map) and 0 <= ny < len(self.map[0]):
                        if self.map[nx][ny] == 0:
                            is_safe = True
                            for b in self.bombs:
                                if abs(nx - b.x) + abs(ny - b.y) <= 1:
                                    is_safe = False
                                    break
                            if is_safe:
                                if dx == -1: safe_directions.append('up')
                                elif dx == 1: safe_directions.append('down')
                                elif dy == -1: safe_directions.append('left')
                                elif dy == 1: safe_directions.append('right')
                
                if safe_directions:
                    return random.choice(safe_directions)
        
        # Đặt bom khi gần player và có đường thoát
        if abs(self.pos_x - self.player_pos[0]) + abs(self.pos_y - self.player_pos[1]) <= 2 and self.has_bomb:
            has_escape = False
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = self.pos_x + dx, self.pos_y + dy
                if 0 <= nx < len(self.map) and 0 <= ny < len(self.map[0]):
                    if self.map[nx][ny] == 0:
                        is_safe = True
                        for bomb in self.bombs:
                            if abs(nx - bomb.x) + abs(ny - bomb.y) <= 1:
                                is_safe = False
                                break
                        if is_safe:
                            has_escape = True
                            break
            if has_escape:
                return 'bomb'
        
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        return ACTIONS[max(range(len(ACTIONS)), key=lambda i: self.q_table[state][i])]

    def move(self, action):
        if action == 'bomb':
            if self.has_bomb:
                self.bombs.append(Bomb(self.pos_x, self.pos_y, is_own_bomb=True))  
                return True
            return False
        
        x, y = self.pos_x, self.pos_y
        new_x, new_y = x, y
        
        if action == 'up' and x-1 >= 0 and self.map[x-1][y] == 0:
            new_x = x-1
        elif action == 'down' and x+1 < len(self.map) and self.map[x+1][y] == 0:
            new_x = x+1
        elif action == 'left' and y-1 >= 0 and self.map[x][y-1] == 0:
            new_y = y-1
        elif action == 'right' and y+1 < len(self.map[0]) and self.map[x][y+1] == 0:
            new_y = y+1
            
        if (new_x, new_y) != (x, y):
            self.pos_x, self.pos_y = new_x, new_y
            return True
        return False

    def update_bombs(self):
        new_bombs = []
        for bomb in self.bombs:
            bomb.timer -= 1
            if bomb.timer <= 0:
                bomb.explode(self.map)
                if abs(self.player_pos[0] - bomb.x) + abs(self.player_pos[1] - bomb.y) <= 1:
                    self.players_killed += 1
                if abs(self.pos_x - bomb.x) + abs(self.pos_y - bomb.y) <= 1:
                    return -100  # Agent died
            else:
                new_bombs.append(bomb)
        self.bombs = new_bombs
        return 0

    def get_reward(self, moved, action, bomb_result):
        if bomb_result == -100:
            return -1000  
        
        reward = 0
        
        if action == 'bomb':
            if abs(self.player_pos[0] - self.pos_x) + abs(self.player_pos[1] - self.pos_y) <= 2:
                has_escape = False
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = self.pos_x + dx, self.pos_y + dy
                    if 0 <= nx < len(self.map) and 0 <= ny < len(self.map[0]):
                        if self.map[nx][ny] == 0:
                            is_safe = True
                            for bomb in self.bombs:
                                if abs(nx - bomb.x) + abs(ny - bomb.y) <= 1:
                                    is_safe = False
                                    break
                            if is_safe:
                                has_escape = True
                                break
                if has_escape:
                    reward += 300  
                else:
                    reward -= 500  
            else:
                reward -= 200  
        
        if moved:
            new_dist = abs(self.pos_x - self.player_pos[0]) + abs(self.pos_y - self.player_pos[1])
            old_dist = abs(self.pos_x - self.player_pos[0]) + abs(self.pos_y - self.player_pos[1])
            
            if new_dist < old_dist:
                reward += 20
            else:
                reward -= 10
            
            for bomb in self.bombs:
                old_bomb_dist = abs(self.pos_x - bomb.x) + abs(self.pos_y - bomb.y)
                if old_bomb_dist <= 1:
                    new_bomb_dist = abs(self.pos_x - bomb.x) + abs(self.pos_y - bomb.y)
                    if new_bomb_dist > old_bomb_dist:
                        if bomb.is_own_bomb:
                            reward += 200  
                        else:
                            reward += 300  
                    elif new_bomb_dist < old_bomb_dist:
                        reward -= 500  
        
        escape_routes = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = self.pos_x + dx, self.pos_y + dy
            if 0 <= nx < len(self.map) and 0 <= ny < len(self.map[0]):
                if self.map[nx][ny] == 0:
                    is_safe = True
                    for bomb in self.bombs:
                        if abs(nx - bomb.x) + abs(ny - bomb.y) <= 1:
                            is_safe = False
                            break
                    if is_safe:
                        escape_routes += 1
        
        if escape_routes >= 2:
            reward += 100  
        elif escape_routes == 0:
            reward -= 300  
        
        return reward

    def update_q_value(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = [0] * len(ACTIONS)
        if next_state not in self.q_table:
            self.q_table[next_state] = [0] * len(ACTIONS)
        action_idx = ACTIONS.index(action)
        old_value = self.q_table[state][action_idx]
        next_max = max(self.q_table[next_state])
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_max - old_value)
        self.q_table[state][action_idx] = new_value

    def train(self, episodes=EPISODES):
        for episode in range(episodes):
            self.reset()
            steps = 0
            done = False
            while steps < MAX_STEPS and not done:
                state = self.get_state()
                action = self.choose_action(state)
                moved = self.move(action)
                bomb_result = self.update_bombs()
                
                if random.random() < 0.3:
                    px, py = self.player_pos
                    possible_moves = [(px+dx, py+dy) for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]
                                      if 0 <= px+dx < len(self.map) and 0 <= py+dy < len(self.map[0]) and self.map[px+dx][py+dy] == 0]
                    if possible_moves:
                        self.player_pos = random.choice(possible_moves)
                
                next_state = self.get_state()
                reward = self.get_reward(moved, action, bomb_result)
                self.total_reward += reward
                self.update_q_value(state, action, reward, next_state)
                
                if bomb_result == -100 or self.players_killed > 0:
                    done = True
                steps += 1
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            if self.total_reward > self.best_reward:
                self.best_reward = self.total_reward
                self.best_q_table = dict(self.q_table)
            
            if episode % 100 == 0:
                print(f"Episode {episode}, Steps: {steps}, Epsilon: {self.epsilon:.3f}, Kills: {self.players_killed}, Total Reward: {self.total_reward}")
        
        print("Training hoàn thành!")
        print(f"Best Reward: {self.best_reward}")
        print(f"Total Kills: {self.players_killed}")
        self.save_q_table()

    def save_q_table(self, filename='q_table.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

def main():
    map_data = [row[:] for row in FIXED_MAP]
    agent = QLearningAgent(map_data)
    print("Bắt đầu training...")
    agent.train(episodes=EPISODES)
    print("Training xong!")

if __name__ == '__main__':
    main()
