from collections import deque
import heapq
import itertools
import pygame
import random
from bomb import Bomb
from node import Node
from enums.algorithm import Algorithm
from copy import deepcopy
from player import Player
import pickle 

_astar_counter = itertools.count()

class Enemy:

    dire = [[1, 0, 1], [0, 1, 0], [-1, 0, 3], [0, -1, 2]]

    TILE_SIZE = 4

    def __init__(self, x, y, alg):
        self.life = True
        self.path = []
        self.movement_path = []
        self.pos_x = x * Enemy.TILE_SIZE
        self.pos_y = y * Enemy.TILE_SIZE
        self.direction = 0
        self.frame = 0
        self.animation = []
        self.range = 3
        self.bomb_limit = 1
        self.plant = False
        self.algorithm = alg
        self.last_positions = []
        
        if self.algorithm == Algorithm.QLEARNING:
            self.epsilon = 0.1  
            self.q_table = {}
            self.load_q_table()

    def move(self, map, bombs, explosions, enemy):

        if self.direction == 0:
            self.pos_y += 1
        elif self.direction == 1:
            self.pos_x += 1
        elif self.direction == 2:
            self.pos_y -= 1
        elif self.direction == 3:
            self.pos_x -= 1

        if self.pos_x % Enemy.TILE_SIZE == 0 and self.pos_y % Enemy.TILE_SIZE == 0:
            self.movement_path.pop(0)
            self.path.pop(0)
            if len(self.path) > 1:
                grid = self.create_grid(map, bombs, explosions, enemy)
                next = self.path[1]
                if grid[next[0]][next[1]] > 1:
                    self.movement_path.clear()
                    self.path.clear()

        if self.frame == 2:
            self.frame = 0
        else:
            self.frame += 1

        if len(self.last_positions) > 5:
            self.last_positions.pop(0)
        self.last_positions.append((self.pos_x, self.pos_y))

        # Nếu lặp lại vị trí nhiều lần, clear path để chọn hướng khác
        if self.last_positions.count((self.pos_x, self.pos_y)) > 2:
            self.movement_path.clear()
            self.path.clear()

    def make_move(self, map, bombs, explosions, enemy):
        if not self.life:
            return
        if len(self.movement_path) == 0:
            if self.plant:
                bombs.append(self.plant_bomb(map))
                self.plant = False
                map[int(self.pos_x / Enemy.TILE_SIZE)][int(self.pos_y / Enemy.TILE_SIZE)] = 3
                return

            # Nếu thuật toán là Backtracking
            if self.algorithm is Algorithm.BACKTRACKING:
                grid = self.create_grid_weighted(map, bombs, explosions, enemy)
                safe_pos = self.find_safe_plant_position(grid, bombs, steps_left=5)  # Giả sử T = 5
                if safe_pos:
                    self.path = [[int(self.pos_x / Enemy.TILE_SIZE), int(self.pos_y / Enemy.TILE_SIZE)], safe_pos]
                    self.movement_path = []
                    for dx, dy, dir_idx in self.dire:
                        if int(self.pos_x / Enemy.TILE_SIZE) + dx == safe_pos[0] and int(self.pos_y / Enemy.TILE_SIZE) + dy == safe_pos[1]:
                            self.movement_path.append(dir_idx)
                            break
                    self.plant = True
                    return

            # Nếu không tìm được vị trí an toàn, dùng thuật toán khác (BFS, A*, ...)
            if self.algorithm is Algorithm.BFS:
                self.bfs(self.create_grid(map, bombs, explosions, enemy), bombs)
            elif self.algorithm is Algorithm.ASTAR:
                self.astar(self.create_grid_weighted(map, bombs, explosions, enemy))
            elif self.algorithm is Algorithm.BEAM:
                self.beam_search(self.create_grid_weighted(map, bombs, explosions, enemy))
            elif self.algorithm is Algorithm.PARTIAL_OBSERVATION:
                self.partial_observation_search(self.create_grid_weighted(map, bombs, explosions, enemy))
            elif self.algorithm is Algorithm.QLEARNING:
                self.q_learning_search(self.create_grid_weighted(map, bombs, explosions, enemy))
        else:
            self.direction = self.movement_path[0]
            self.move(map, bombs, explosions, enemy)

    def plant_bomb(self, map):
        b = Bomb(self.range, round(self.pos_x / Enemy.TILE_SIZE), round(self.pos_y / Enemy.TILE_SIZE), map, self)
        self.bomb_limit -= 1
        return b

    def check_death(self, exp):

        for e in exp:
            for s in e.sectors:
                if int(self.pos_x / Enemy.TILE_SIZE) == s[0] and int(self.pos_y / Enemy.TILE_SIZE) == s[1]:
                    self.life = False
                    return

    def create_grid(self, map, bombs, explosions, enemys):
        grid = [[0] * len(map) for r in range(len(map))]

        for b in bombs:
            b.get_range(map)
            for x in b.sectors:
                grid[x[0]][x[1]] = 1
            grid[b.pos_x][b.pos_y] = 3

        for e in explosions:
            for s in e.sectors:
                grid[s[0]][s[1]] = 3

        for i in range(len(map)):
            for j in range(len(map[i])):
                if map[i][j] == 1:
                    grid[i][j] = 3
                elif map[i][j] == 2:
                    grid[i][j] = 2

        for x in enemys:
            if x == self:
                continue
            elif not x.life:
                continue
            else:
                grid[int(x.pos_x / Enemy.TILE_SIZE)][int(x.pos_y / Enemy.TILE_SIZE)] = 2

        return grid

    def create_grid_weighted(self, map, bombs, explosions, enemys):
        grid = [[None] * len(map) for r in range(len(map))]

        for i in range(len(map)):
            for j in range(len(map)):
                if map[i][j] == 0:
                    grid[i][j] = Node(i, j, True, 1, 0)
                elif map[i][j] == 2:
                    grid[i][j] = Node(i, j, False, 999, 1)
                elif map[i][j] == 1:
                    grid[i][j] = Node(i, j, False, 999, 2)
                elif map[i][j] == 3:
                    grid[i][j] = Node(i, j, False, 999, 2)

        for b in bombs:
            b.get_range(map)
            for x in b.sectors:
                grid[x[0]][x[1]].weight = 5
                grid[x[0]][x[1]].value = 3
            grid[b.pos_x][b.pos_y].reach = False

        for e in explosions:
            for s in e.sectors:
                grid[s[0]][s[1]].reach = False

        for x in enemys:
            if x == self:
                continue
            elif not x.life:
                continue
            else:
                grid[int(x.pos_x / Enemy.TILE_SIZE)][int(x.pos_y / Enemy.TILE_SIZE)].reach = False
                grid[int(x.pos_x / Enemy.TILE_SIZE)][int(x.pos_y / Enemy.TILE_SIZE)].value = 1
        return grid
    
    def bfs(self, grid, bombs):
        """
        BFS adapted from original DFS/Dijkstra logic:
        - grid: 2D list from create_grid(self, map, bombs, explosions, enemys)
        Values: 0 = safe, 1 = unsafe (bom trong tầm), 2 = crate (destructible), 3 = unreachable
        - Nếu self.bomb_limit > 0: tìm ô gần nhất kề crate để plant bomb
        - Nếu self.bomb_limit == 0: tìm ô safe đầu tiên để né bom
        """

        dirs = self.dire[:]    
        random.shuffle(dirs)

        h, w = len(grid), len(grid[0])
        sx = int(self.pos_x / Enemy.TILE_SIZE)
        sy = int(self.pos_y / Enemy.TILE_SIZE)
        has_bomb = (self.bomb_limit > 0)

        if has_bomb:
            for dx, dy, _ in dirs:
                ax, ay = sx + dx, sy + dy
                if 0 <= ax < h and 0 <= ay < w and grid[ax][ay] == 2:
                    self.path = [[sx, sy]]
                    self.movement_path = []
                    self.plant = True
                    return

        # 3) BFS
        visited = [[False]*w for _ in range(h)]
        parent  = {}
        queue   = deque([(sx, sy)])
        visited[sx][sy] = True

        found = None
        while queue:
            x, y = queue.popleft()

            if (x, y) != (sx, sy):
                if has_bomb:
                    if any(
                        0 <= x+dx < h and 0 <= y+dy < w and grid[x+dx][y+dy] == 2
                        for dx, dy, _ in dirs
                    ):
                        found = (x, y)
                        break
                else:
                    if grid[x][y] == 0:
                        found = (x, y)
                        break

            for dx, dy, _ in dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w \
                and not visited[nx][ny] \
                and grid[nx][ny] in (0, 1):
                    visited[nx][ny] = True
                    parent[(nx, ny)] = (x, y)
                    queue.append((nx, ny))

        if not found:
            # Nếu BFS thất bại, thử backtracking
            alt_path = self.backtracking_path(grid, sx, sy, has_bomb)
            if alt_path:
                self.path = alt_path
                self.movement_path = []
                for i in range(len(alt_path) - 1):
                    x0, y0 = alt_path[i]
                    x1, y1 = alt_path[i + 1]
                    for dx, dy, dir_idx in dirs:
                        if x0 + dx == x1 and y0 + dy == y1:
                            self.movement_path.append(dir_idx)
                            break
                if has_bomb and len(alt_path) == 1:
                    self.plant = True
            else:
                self.path = [[sx, sy]]
                self.movement_path = []
            return

        # 5) Reconstruct path từ found về start
        path = []
        cur = found
        while cur != (sx, sy):
            path.append([cur[0], cur[1]])
            cur = parent[cur]
        path.append([sx, sy])
        path.reverse()
        self.path = path

        # 6) Xây movement_path (danh sách dir_idx)
        self.movement_path = []
        for i in range(len(path) - 1):
            x0, y0 = path[i]
            x1, y1 = path[i + 1]
            for dx, dy, dir_idx in dirs:
                if x0 + dx == x1 and y0 + dy == y1:
                    self.movement_path.append(dir_idx)
                    break

        # 7) Nếu cần plant bomb (đứng yên ngay cạnh crate)
        if has_bomb and len(self.path) == 1:
            self.plant = True

    def backtracking_path(self, grid, sx, sy, has_bomb):
        h, w = len(grid), len(grid[0])
        visited = [[False] * w for _ in range(h)]
        path = []

        def is_valid(x, y):
            return 0 <= x < h and 0 <= y < w and grid[x][y] in (0, 1) and not visited[x][y]

        def backtrack(x, y):
            if has_bomb:
                for dx, dy, _ in self.dire:
                    ax, ay = x + dx, y + dy
                    if 0 <= ax < h and 0 <= ay < w and grid[ax][ay] == 2:
                        path.append((x, y))
                        return True
            else:
                if grid[x][y] == 0 and (x, y) != (sx, sy):
                    path.append((x, y))
                    return True

            visited[x][y] = True
            for dx, dy, _ in self.dire:
                nx, ny = x + dx, y + dy
                if is_valid(nx, ny):
                    path.append((x, y))
                    if backtrack(nx, ny):
                        return True
                    path.pop()
            return False

        if backtrack(sx, sy):
            path.reverse()
            return [[x, y] for (x, y) in path]
        return None

    def astar(self, grid):
        """
        A* với cải tiến:
        - Heuristic tốt hơn cho việc tìm mục tiêu
        - Chi phí an toàn được điều chỉnh
        - Thêm giới hạn số bước tìm kiếm
        """
        print("\n=== A* Debug ===")
        import heapq

        for row in grid:
            for n in row:
                n.weight = float('inf')
                n.parent = None

        h, w = len(grid), len(grid[0])
        sx = int(self.pos_x / Enemy.TILE_SIZE)
        sy = int(self.pos_y / Enemy.TILE_SIZE)
        target_has_bomb = (self.bomb_limit > 0)
        print(f"Start position: ({sx}, {sy})")
        print(f"Has bomb: {target_has_bomb}")

        start = grid[sx][sy]
        start.weight = 0

        def find_nearest_target(x, y):
            min_dist = float('inf')
            target = None
            
            player_pos = None
            for i in range(h):
                for j in range(w):
                    if grid[i][j].value == 4:  
                        player_pos = (i, j)
                        break
            
            if player_pos:
                px, py = player_pos
                dist = abs(x - px) + abs(y - py)
                if dist <= 5:
                    return (px, py)
            
            # Nếu không tìm thấy người chơi gần, mới tìm thùng
            for i in range(h):
                for j in range(w):
                    if grid[i][j].value == 1:  # Thùng
                        dist = abs(x - i) + abs(y - j)
                        if dist < min_dist:
                            min_dist = dist
                            target = (i, j)
            
            return target

        def heuristic(n):
            target = find_nearest_target(n.x, n.y)
            if target:
                return abs(n.x - target[0]) + abs(n.y - target[1])
            return float('inf')

        def is_safe_position(x, y):
            # Kiểm tra xung quanh có bom và có đường thoát
            has_bomb = False
            has_escape = False
            for dx, dy, _ in Enemy.dire:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w:
                    if grid[nx][ny].value == 3:
                        has_bomb = True
                    elif grid[nx][ny].value == 0:
                        has_escape = True
            return not has_bomb or has_escape

        def calculate_cost(node, neighbor):
            base_cost = neighbor.base_weight
            distance_cost = abs(node.x - neighbor.x) + abs(node.y - neighbor.y)
            safety_cost = 0 if is_safe_position(neighbor.x, neighbor.y) else 50  
            return base_cost + distance_cost + safety_cost

        open_set = []
        heapq.heappush(open_set, (heuristic(start), next(_astar_counter), start))
        closed = set()
        goal = None
        max_steps = h * w
        steps = 0

        while open_set and steps < max_steps:
            steps += 1
            _, _, node = heapq.heappop(open_set)
            if node in closed:
                continue
            closed.add(node)
            print(f"Step {steps}: Checking ({node.x}, {node.y})")

            if target_has_bomb:
                if any(
                    0 <= node.x+dx < h and 0 <= node.y+dy < w
                    and grid[node.x+dx][node.y+dy].value == 1
                    and is_safe_position(node.x, node.y)
                    for dx, dy, _ in Enemy.dire
                ):
                    print(f"Found safe spot near crate at ({node.x}, {node.y})")
                    goal = node
                    break
            else:
                if node.value == 0 and is_safe_position(node.x, node.y):
                    print(f"Found safe spot with escape at ({node.x}, {node.y})")
                    goal = node
                    break

            for dx, dy, _ in Enemy.dire:
                nx, ny = node.x + dx, node.y + dy
                if not (0 <= nx < h and 0 <= ny < w):
                    continue
                nbr = grid[nx][ny]
                if not nbr.reach or nbr in closed:
                    continue
                
                new_cost = node.weight + calculate_cost(node, nbr)
                if new_cost < nbr.weight:
                    nbr.weight = new_cost
                    nbr.parent = node
                    heapq.heappush(open_set, (new_cost + heuristic(nbr), next(_astar_counter), nbr))
                    print(f"  Added neighbor ({nx}, {ny}) with cost {new_cost}")

        if goal is None:
            print("No path found!")
            self.path = [[sx, sy]]
            self.movement_path = []
            return

        rev = []
        cur = goal
        while cur:
            rev.append([cur.x, cur.y])
            cur = cur.parent
        rev.reverse()
        self.path = rev
        print(f"Final path: {rev}")

        moves = []
        for i in range(len(rev) - 1):
            x0, y0 = rev[i]
            x1, y1 = rev[i+1]
            for dx, dy, dir_idx in Enemy.dire:
                if x0+dx == x1 and y0+dy == y1:
                    moves.append(dir_idx)
                    break
        self.movement_path = moves
        print(f"Movement path: {moves}")

        if target_has_bomb and len(self.path) == 1:
            self.plant = True
            print("Will plant bomb at current position")

    def beam_search(self, grid, beam_width=5):  
        """
        Beam Search với cải tiến:
        - Tăng beam width để tìm được đường đi tốt hơn
        - Thêm kiểm tra đường thoát
        - Thêm giới hạn số bước tìm kiếm
        """
        print("\n=== Beam Search Debug ===")
        h, w = len(grid), len(grid[0])
        sx = int(self.pos_x / Enemy.TILE_SIZE)
        sy = int(self.pos_y / Enemy.TILE_SIZE)
        target_has_bomb = (self.bomb_limit > 0)
        print(f"Start position: ({sx}, {sy})")
        print(f"Has bomb: {target_has_bomb}")

        def heuristic(x, y):
            # Luôn ưu tiên tìm người chơi trước
            player_pos = None
            for i in range(h):
                for j in range(w):
                    if grid[i][j].value == 4:  
                        player_pos = (i, j)
                        break
            
            if player_pos:
                px, py = player_pos
                dist = abs(x - px) + abs(y - py)
                # Ưu tiên tấn công người chơi nếu khoảng cách <= 5
                if dist <= 5:
                    return -1000 + dist  
            
            # Nếu không tìm thấy người chơi gần, mới tìm thùng
            min_dist = float('inf')
            for i in range(h):
                for j in range(w):
                    if grid[i][j].value == 1:  # Thùng
                        dist = abs(x - i) + abs(y - j)
                        if dist < min_dist:
                            min_dist = dist
            return min_dist

        for row in grid:
            for n in row:
                n.weight = float('inf')
                n.parent = None

        start = grid[sx][sy]
        start.weight = 0
        start.parent = None

        current_level = [(0, sx, sy, start)]
        visited = set([(sx, sy)])
        goal = None
        max_steps = h * w
        steps = 0

        while current_level and steps < max_steps:
            steps += 1
            print(f"\nStep {steps}:")
            print(f"Current level nodes: {len(current_level)}")
            next_level = []
            
            for cost, x, y, node in current_level:
                print(f"  Checking node ({x}, {y}) with cost {cost}")
                
                if target_has_bomb:
                    has_crate = False
                    has_escape = False
                    for dx, dy, _ in Enemy.dire:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w:
                            if grid[nx][ny].value == 1:
                                has_crate = True
                            elif grid[nx][ny].value == 0:
                                has_escape = True
                    if has_crate and has_escape:
                        print(f"Found safe spot near crate at ({x}, {y})")
                        goal = node
                        break
                else:
                    if grid[x][y].value == 0:
                        has_escape = False
                        for dx, dy, _ in Enemy.dire:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < h and 0 <= ny < w and grid[nx][ny].value == 0:
                                has_escape = True
                                break
                        if has_escape:
                            print(f"Found safe spot with escape at ({x}, {y})")
                            goal = node
                            break

                if goal:
                    break

                for dx, dy, _ in Enemy.dire:
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < h and 0 <= ny < w):
                        continue
                    
                    if (nx, ny) in visited:
                        continue
                        
                    neighbor = grid[nx][ny]
                    if not neighbor.reach:
                        continue

                    new_cost = node.weight + neighbor.base_weight
                    if new_cost < neighbor.weight:
                        neighbor.weight = new_cost
                        neighbor.parent = node
                        next_level.append((new_cost + heuristic(nx, ny), nx, ny, neighbor))
                        visited.add((nx, ny))
                        print(f"    Added neighbor ({nx}, {ny}) with cost {new_cost + heuristic(nx, ny)}")

            if goal:
                break

            next_level.sort()
            current_level = next_level[:beam_width]
            print(f"Selected {len(current_level)} best nodes for next level")

        if goal is None:
            print("No path found!")
            self.path = [[sx, sy]]
            self.movement_path = []
            return

        path = []
        cur = goal
        while cur:
            path.append([cur.x, cur.y])
            cur = cur.parent
        path.reverse()
        self.path = path
        print(f"Final path: {path}")

        moves = []
        for i in range(len(path) - 1):
            x0, y0 = path[i]
            x1, y1 = path[i+1]
            for dx, dy, dir_idx in Enemy.dire:
                if x0+dx == x1 and y0+dy == y1:
                    moves.append(dir_idx)
                    break
        self.movement_path = moves
        print(f"Movement path: {moves}")

        if target_has_bomb and len(self.path) == 1:
            self.plant = True
            print("Will plant bomb at current position")

    def partial_observation_search(self, grid, observation_radius=3):
        import random
        print("\n=== Partial Observation Search Debug ===")
        h, w = len(grid), len(grid[0])
        sx = int(self.pos_x / Enemy.TILE_SIZE)
        sy = int(self.pos_y / Enemy.TILE_SIZE)
        target_has_bomb = (self.bomb_limit > 0)

        # 1. Khởi tạo belief_state với -1 là chưa biết
        if not hasattr(self, 'belief_state'):
            self.belief_state = [[-1] * w for _ in range(h)]
            self.observed_areas = set()
            self.memory_decay = 0.1

        def get_observable_area(x, y):
            observable = []
            for i in range(max(0, x - observation_radius), min(h, x + observation_radius + 1)):
                for j in range(max(0, y - observation_radius), min(w, y + observation_radius + 1)):
                    if abs(i - x) + abs(j - y) <= observation_radius:
                        observable.append((i, j))
            return observable

        def update_belief_state(x, y):
            observable = get_observable_area(x, y)
            self.observed_areas.update(observable)
            for i, j in observable:
                self.belief_state[i][j] = grid[i][j].value
            for i in range(h):
                for j in range(w):
                    if (i, j) not in observable and self.belief_state[i][j] == 0:
                        if random.random() < self.memory_decay:
                            self.belief_state[i][j] = -1

        def is_safe_position(x, y):
            if self.belief_state[x][y] != 0:
                return False
            escape_routes = 0
            for dx, dy, _ in Enemy.dire:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and self.belief_state[nx][ny] == 0:
                    escape_routes += 1
            return escape_routes > 0

        def find_nearest_target(x, y):
            min_dist = float('inf')
            target = None
            for i in range(h):
                for j in range(w):
                    if target_has_bomb:
                        if self.belief_state[i][j] == 1:
                            dist = abs(x - i) + abs(y - j)
                            if is_safe_position(i, j) and dist < min_dist:
                                min_dist = dist
                                target = (i, j)
                    else:
                        if self.belief_state[i][j] == 0 and is_safe_position(i, j):
                            dist = abs(x - i) + abs(y - j)
                            if dist < min_dist:
                                min_dist = dist
                                target = (i, j)
            return target

        def find_nearest_unknown(x, y):
            min_dist = float('inf')
            target = None
            for i in range(h):
                for j in range(w):
                    if self.belief_state[i][j] == -1:
                        dist = abs(x - i) + abs(y - j)
                        if dist < min_dist:
                            min_dist = dist
                            target = (i, j)
            return target

        def heuristic(nx, ny):
            t = find_nearest_target(nx, ny)
            if t:
                base_dist = abs(nx - t[0]) + abs(ny - t[1])
                safety_bonus = 10 if is_safe_position(nx, ny) else 0
                return max(base_dist - safety_bonus, 0)  # max(h(x), 0)
            return 9999

        update_belief_state(sx, sy)

        for row in grid:
            for n in row:
                n.weight = float('inf')
                n.parent = None

        start = grid[sx][sy]
        start.weight = 0

        open_set = []
        heapq.heappush(open_set, (heuristic(sx, sy), next(_astar_counter), start))
        closed = set()
        goal = None
        max_steps = h * w // 2
        steps = 0

        while open_set and steps < max_steps:
            steps += 1
            _, _, node = heapq.heappop(open_set)
            if node in closed:
                continue
            closed.add(node)
            update_belief_state(node.x, node.y)

            # 3. Luôn cập nhật lại mục tiêu
            current_target = find_nearest_target(node.x, node.y)

            if target_has_bomb:
                if any(
                    0 <= node.x+dx < h and 0 <= node.y+dy < w
                    and self.belief_state[node.x+dx][node.y+dy] == 1
                    and is_safe_position(node.x, node.y)
                    for dx, dy, _ in Enemy.dire
                ):
                    goal = node
                    break
            else:
                if self.belief_state[node.x][node.y] == 0 and is_safe_position(node.x, node.y):
                    goal = node
                    break

            for dx, dy, _ in Enemy.dire:
                nx, ny = node.x + dx, node.y + dy
                if not (0 <= nx < h and 0 <= ny < w):
                    continue
                nbr = grid[nx][ny]
                if not nbr.reach or nbr in closed:
                    continue
                base_cost = nbr.base_weight
                safety_cost = 0 if is_safe_position(nx, ny) else 50
                new_cost = node.weight + base_cost + safety_cost
                if new_cost < nbr.weight:
                    nbr.weight = new_cost
                    nbr.parent = node
                    heapq.heappush(open_set, (new_cost + heuristic(nx, ny), next(_astar_counter), nbr))

        # 5. Nếu không tìm thấy mục tiêu, chuyển sang chế độ khám phá
        if goal is None:
            unknown = find_nearest_unknown(sx, sy)
            if unknown:
                # Tìm đường tới ô chưa biết gần nhất
                path = [[sx, sy]]
                x, y = sx, sy
                while (x, y) != unknown:
                    dx = 1 if unknown[0] > x else -1 if unknown[0] < x else 0
                    dy = 1 if unknown[1] > y else -1 if unknown[1] < y else 0
                    if dx != 0 and 0 <= x + dx < h and self.belief_state[x + dx][y] != 2:
                        x += dx
                    elif dy != 0 and 0 <= y + dy < w and self.belief_state[x][y + dy] != 2:
                        y += dy
                    else:
                        break
                    path.append([x, y])
                self.path = path
                self.movement_path = []
                for i in range(len(path) - 1):
                    x0, y0 = path[i]
                    x1, y1 = path[i+1]
                    for dx, dy, dir_idx in Enemy.dire:
                        if x0+dx == x1 and y0+dy == y1:
                            self.movement_path.append(dir_idx)
                            break
                return
            else:
                self.path = [[sx, sy]]
                self.movement_path = []
                return

        rev = []
        cur = goal
        while cur:
            rev.append([cur.x, cur.y])
            cur = cur.parent
        rev.reverse()
        self.path = rev

        moves = []
        for i in range(len(rev) - 1):
            x0, y0 = rev[i]
            x1, y1 = rev[i+1]
            for dx, dy, dir_idx in Enemy.dire:
                if x0+dx == x1 and y0+dy == y1:
                    moves.append(dir_idx)
                    break
        self.movement_path = moves

        if target_has_bomb and len(self.path) == 1:
            boxes = 0
            for dx, dy, _ in Enemy.dire:
                nx, ny = sx + dx, sy + dy
                if 0 <= nx < h and 0 <= ny < w and self.belief_state[nx][ny] == 1:
                    boxes += 1
            if boxes > 0 and is_safe_position(sx, sy):
                self.plant = True

    def is_player_vulnerable(self, px, py, grid):
        """Check if player is in a vulnerable position
        Player is vulnerable if:
        1. Has less than 2 escape routes
        2. Distance to enemy <= 3
        3. No boxes blocking direct path
        4. In a corner or narrow space
        """
        if not px or not py:
            return False
        
        h, w = len(grid), len(grid[0])
        
        escape_routes = 0
        for dx, dy, _ in Enemy.dire:
            nx, ny = px + dx, py + dy
            if 0 <= nx < h and 0 <= ny < w and grid[nx][ny].value == 0:
                escape_routes += 1
        
        is_corner = 0
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = px + dx, py + dy
            if 0 <= nx < h and 0 <= ny < w and grid[nx][ny].value in [1, 2]:  # Wall or box
                is_corner += 1
        
        distance = abs(self.pos_x/Enemy.TILE_SIZE - px) + abs(self.pos_y/Enemy.TILE_SIZE - py)
        
        has_clear_path = True
        sx = int(self.pos_x / Enemy.TILE_SIZE)
        sy = int(self.pos_y / Enemy.TILE_SIZE)
        if sx == px or sy == py:
            min_x, max_x = min(sx, px), max(sx, px)
            min_y, max_y = min(sy, py), max(sy, py)
            for i in range(min_x, max_x + 1):
                for j in range(min_y, max_y + 1):
                    if grid[i][j].value == 2:  
                        has_clear_path = False
                        break
        
        return (escape_routes < 2 and distance <= 3 and has_clear_path) or is_corner >= 2

    def find_strategic_box(self, grid, player_pos):
        """Find strategic box to destroy that will:
        1. Open path to player
        2. Create narrow spaces
        3. Create traps to corner player
        """
        h, w = len(grid), len(grid[0])
        px, py = player_pos
        if not px or not py:
            return None

        best_box = None
        best_score = float('-inf')
        
        for bx, by in [(i, j) for i in range(h) for j in range(w) if grid[i][j].value == 2]:
            score = 0
            
            if self.blocks_path_to_player(bx, by, px, py, grid):
                score += 10
            
            narrow_space_score = self.evaluate_narrow_space(bx, by, grid)
            score += narrow_space_score
            
            trap_score = self.evaluate_trap_creation(bx, by, px, py, grid)
            score += trap_score
            
            dist_to_player = abs(bx - px) + abs(by - py)
            score += (10 - dist_to_player)
            
            if score > best_score:
                best_score = score
                best_box = (bx, by)
        
        return best_box

    def blocks_path_to_player(self, bx, by, px, py, grid):
        """Check if box blocks a direct path to player"""
        h, w = len(grid), len(grid[0])
        
        if (bx == px or by == py):  
            if bx == px:  
                min_y, max_y = min(by, py), max(by, py)
                for y in range(min_y, max_y + 1):
                    if grid[bx][y].value == 2 and (bx, y) != (bx, by):
                        return True
            else:  
                min_x, max_x = min(bx, px), max(bx, px)
                for x in range(min_x, max_x + 1):
                    if grid[x][by].value == 2 and (x, by) != (bx, by):
                        return True
        return False

    def evaluate_narrow_space(self, bx, by, grid):
        """Evaluate how much a box contributes to creating narrow spaces"""
        h, w = len(grid), len(grid[0])
        score = 0
        
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = bx + dx, by + dy
            if 0 <= nx < h and 0 <= ny < w:
                if grid[nx][ny].value == 0:  
                    wall_count = 0
                    for ddx, ddy in [(1,0), (-1,0), (0,1), (0,-1)]:
                        nnx, nny = nx + ddx, ny + ddy
                        if 0 <= nnx < h and 0 <= nny < w and grid[nnx][nny].value in [1, 2]:
                            wall_count += 1
                    if wall_count >= 2:  
                        score += 3
        
        return score

    def evaluate_trap_creation(self, bx, by, px, py, grid):
        """Evaluate how much a box contributes to creating traps"""
        h, w = len(grid), len(grid[0])
        score = 0
        
        if self.would_create_corner_trap(bx, by, px, py, grid):
            score += 5
        
        if self.would_reduce_escape_routes(bx, by, px, py, grid):
            score += 4
        
        return score

    def would_create_corner_trap(self, bx, by, px, py, grid):
        """Check if destroying box would create a corner trap near player"""
        h, w = len(grid), len(grid[0])
        
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = px + dx, py + dy
            if 0 <= nx < h and 0 <= ny < w and grid[nx][ny].value == 0:
                wall_count = 0
                for ddx, ddy in [(1,0), (-1,0), (0,1), (0,-1)]:
                    nnx, nny = nx + ddx, ny + ddy
                    if 0 <= nnx < h and 0 <= nny < w and grid[nnx][nny].value in [1, 2]:
                        wall_count += 1
                if wall_count >= 2:  
                    return True
        return False

    def would_reduce_escape_routes(self, bx, by, px, py, grid):
        """Check if destroying box would reduce player's escape routes"""
        h, w = len(grid), len(grid[0])
        
        current_routes = 0
        for dx, dy, _ in Enemy.dire:
            nx, ny = px + dx, py + dy
            if 0 <= nx < h and 0 <= ny < w and grid[nx][ny].value == 0:
                current_routes += 1
        
        grid[bx][by].value = 0
        
        new_routes = 0
        for dx, dy, _ in Enemy.dire:
            nx, ny = px + dx, py + dy
            if 0 <= nx < h and 0 <= ny < w and grid[nx][ny].value == 0:
                new_routes += 1
        
        grid[bx][by].value = 2
        
        return new_routes < current_routes

    def load_q_table(self):
        """Load trained Q-table from file"""
        try:
            with open('q_table.pkl', 'rb') as f:
                self.q_table = pickle.load(f)
        except Exception:
            self.q_table = {}

    def get_state_key(self, grid, bombs=None):
        """Get state key for Q-learning based on current position and environment"""
        h, w = len(grid), len(grid[0])
        ax = int(self.pos_x / Enemy.TILE_SIZE)
        ay = int(self.pos_y / Enemy.TILE_SIZE)
        
        box_near = 0
        for dx, dy, _ in Enemy.dire:
            nx, ny = ax + dx, ay + dy
            if 0 <= nx < h and 0 <= ny < w and grid[nx][ny].value == 1:
                box_near = 1
                
        bomb_near = 0
        if bombs:
            for b in bombs:
                bx, by = b.pos_x, b.pos_y
                if abs(ax - bx) + abs(ay - by) <= b.range:
                    bomb_near = 1
                    break
                    
        safe_escape = 0
        escape_routes = 0
        for dx, dy, _ in Enemy.dire:
            nx, ny = ax + dx, ay + dy
            if 0 <= nx < h and 0 <= ny < w and grid[nx][ny].reach:
                escape_routes += 1
        if escape_routes >= 2:
            safe_escape = 1

        return f'{ax},{ay}|{box_near}|{bomb_near}|{safe_escape}|{self.bomb_limit}'

    def choose_action(self, state):
        """Choose action based on Q-table with epsilon-greedy strategy"""
        if random.random() < self.epsilon:
            return random.choice(Enemy.dire)
        else:
            best_value = float('-inf')
            best_action = None
            for action in Enemy.dire:
                state_action_key = f"{state}|{action[0]},{action[1]},{action[2]}"
                value = self.q_table.get(state_action_key, 0)
                if value > best_value:
                    best_value = value
                    best_action = action
            return best_action if best_action else random.choice(Enemy.dire)

    def q_learning_search(self, grid, bombs=None):
        """Use trained Q-table to make decisions"""
        h, w = len(grid), len(grid[0])
        sx = int(self.pos_x / Enemy.TILE_SIZE)
        sy = int(self.pos_y / Enemy.TILE_SIZE)

        state = self.get_state_key(grid, bombs)
        
        action = self.choose_action(state)
        dx, dy, dir_idx = action
        next_x, next_y = sx + dx, sy + dy

        if not (0 <= next_x < h and 0 <= next_y < w and grid[next_x][next_y].reach):
            valid_moves = []
            for dx, dy, dir_idx in Enemy.dire:
                nx, ny = sx + dx, sy + dy
                if 0 <= nx < h and 0 <= ny < w and grid[nx][ny].reach:
                    valid_moves.append((dx, dy, dir_idx))
            if valid_moves:
                action = random.choice(valid_moves)
                dx, dy, dir_idx = action
                next_x, next_y = sx + dx, sy + dy
            else:
                self.path = [[sx, sy]]
                self.movement_path = []
                return

        self.path = [[sx, sy], [next_x, next_y]]
        self.movement_path = [dir_idx]
        
        if self.bomb_limit > 0:
            has_box = False
            for dx, dy, _ in Enemy.dire:
                nx, ny = sx + dx, sy + dy
                if 0 <= nx < h and 0 <= ny < w and grid[nx][ny].value == 1:
                    has_box = True
                    break
            if has_box:
                self.plant = True
    
    def save_q_table(self):
        """Save Q-table to file"""
        if hasattr(self, 'q_table'):
            with open('q_table.pkl', 'wb') as f:
                pickle.dump(self.q_table, f)

    def heuristic(self, node, goal):
        dx = abs(node.x - goal.x)
        dy = abs(node.y - goal.y)
        traditional_cost = (dx + dy) * 10
        
        state = self.get_state_from_node(node)
        best_action = self.q_learning.get_best_action(state)
        q_value = self.q_learning.get_q_value(state, best_action)
        
        return traditional_cost + 0.5 * q_value

    def update_q_learning(self, path):
        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i+1]
            state = self.get_state_from_node(current_node)
            action = self.get_action_from_nodes(current_node, next_node)
            reward = self.calculate_reward(next_node)
            next_state = self.get_state_from_node(next_node)
            self.q_learning.update(state, action, reward, next_state)

    def load_animations(self, en, scale):
        front = []
        back = []
        left = []
        right = []
        resize_width = scale
        resize_height = scale

        image_path = 'images/enemy/e'
        if en == '':
            image_path = 'images/hero/p'

        f1 = pygame.image.load(image_path + en + 'f0.png')
        f2 = pygame.image.load(image_path + en + 'f1.png')
        f3 = pygame.image.load(image_path + en + 'f2.png')

        f1 = pygame.transform.scale(f1, (resize_width, resize_height))
        f2 = pygame.transform.scale(f2, (resize_width, resize_height))
        f3 = pygame.transform.scale(f3, (resize_width, resize_height))

        front.append(f1)
        front.append(f2)
        front.append(f3)

        r1 = pygame.image.load(image_path + en + 'r0.png')
        r2 = pygame.image.load(image_path + en + 'r1.png')
        r3 = pygame.image.load(image_path + en + 'r2.png')

        r1 = pygame.transform.scale(r1, (resize_width, resize_height))
        r2 = pygame.transform.scale(r2, (resize_width, resize_height))
        r3 = pygame.transform.scale(r3, (resize_width, resize_height))

        right.append(r1)
        right.append(r2)
        right.append(r3)

        b1 = pygame.image.load(image_path + en + 'b0.png')
        b2 = pygame.image.load(image_path + en + 'b1.png')
        b3 = pygame.image.load(image_path + en + 'b2.png')

        b1 = pygame.transform.scale(b1, (resize_width, resize_height))
        b2 = pygame.transform.scale(b2, (resize_width, resize_height))
        b3 = pygame.transform.scale(b3, (resize_width, resize_height))

        back.append(b1)
        back.append(b2)
        back.append(b3)

        l1 = pygame.image.load(image_path + en + 'l0.png')
        l2 = pygame.image.load(image_path + en + 'l1.png')
        l3 = pygame.image.load(image_path + en + 'l2.png')

        l1 = pygame.transform.scale(l1, (resize_width, resize_height))
        l2 = pygame.transform.scale(l2, (resize_width, resize_height))
        l3 = pygame.transform.scale(l3, (resize_width, resize_height))

        left.append(l1)
        left.append(l2)
        left.append(l3)

        self.animation.append(front)
        self.animation.append(right)
        self.animation.append(back)
        self.animation.append(left)