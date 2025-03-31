from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from collections import defaultdict
import pickle
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import copy
from operator import itemgetter
import json

class NumpyEncoder(json.JSONEncoder):
    """用于处理NumPy类型的自定义JSON编码器"""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)



app = Flask(__name__, static_url_path='')
CORS(app)  

app.json_encoder = NumpyEncoder

os.makedirs('models', exist_ok=True)



# Q-Learning 
class TicTacToeAI:
    def __init__(self):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_rate = 0.2
        self.discount_factor = 0.95
        self.epsilon = 0.3
        self.epsilon_decay = 0.9999
        self.min_epsilon = 0.01
        self.model_path = 'models/q_table.pkl'
        self.WINNING_PATTERNS = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8], 
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  
            [0, 4, 8], [2, 4, 6] 
        ]
        self.load_model()

    def analyze_board(self, board):
        empty_spots = self.get_valid_moves(board)
        threats = []
        opportunities = []
        
        for move in empty_spots:
            
            test_board = board.copy()
            test_board[move] = 'O'
            if self.check_winner(test_board) == 'O':
                return {'win_move': move}
                
            
            test_board = board.copy()
            test_board[move] = 'X'
            if self.check_winner(test_board) == 'X':
                threats.append(move)
                
            
            if self.check_fork_opportunity(board, move, 'O'):
                opportunities.append(move)
                
        return {
            'threats': threats,
            'opportunities': opportunities,
            'center': 4 if 4 in empty_spots else None,
            'corners': [pos for pos in [0,2,6,8] if pos in empty_spots],
            'edges': [pos for pos in [1,3,5,7] if pos in empty_spots]
        }

    def check_fork_opportunity(self, board, move, player):
        test_board = board.copy()
        test_board[move] = player
        winning_moves = 0
        
        for pos in self.get_valid_moves(test_board):
            temp_board = test_board.copy()
            temp_board[pos] = player
            if self.check_winner(temp_board) == player:
                winning_moves += 1
                
        return winning_moves >= 2

    def get_strategic_move(self, board):
        analysis = self.analyze_board(board)
        
        
        if 'win_move' in analysis:
            return analysis['win_move'], analysis
            
        
        if analysis['threats']:
            return analysis['threats'][0], analysis
            
        
        if analysis['opportunities']:
            return analysis['opportunities'][0], analysis
            
        
        if analysis['center'] is not None:
            return analysis['center'], analysis
            
        
        if analysis['corners']:
            return random.choice(analysis['corners']), analysis
            
        
        if analysis['edges']:
            return random.choice(analysis['edges']), analysis
            
        return random.choice(self.get_valid_moves(board)), analysis

    def get_best_move(self, board):
        state = self.get_state_key(board)
        valid_moves = self.get_valid_moves(board)
        
        if not valid_moves:
            return -1, None

        if random.random() > self.epsilon:
            strategic_move, analysis = self.get_strategic_move(board)
            q_values = {move: self.q_table[state][move] for move in valid_moves}
            max_q = max(q_values.values())
            
            # 如果策略移动的Q值接近最大Q值，使用策略移动
            if strategic_move in q_values and q_values[strategic_move] >= max_q * 0.8:
                return strategic_move, analysis
                
            # 否则使用Q值最大的移动
            best_moves = [move for move, q in q_values.items() if q == max_q]
            return random.choice(best_moves), analysis
        
        move = random.choice(valid_moves)
        return move, self.analyze_board(board)

    def get_state_key(self, board):
        return ''.join(str(cell) if cell else '_' for cell in board)

    def get_valid_moves(self, board):
        return [i for i, cell in enumerate(board) if not cell]

    def check_winner(self, board):
        for line in self.WINNING_PATTERNS:
            if board[line[0]] and board[line[0]] == board[line[1]] == board[line[2]]:
                return board[line[0]]
        return 'draw' if all(board) else None

    def load_model(self):
        try:
            with open(self.model_path, 'rb') as f:
                self.q_table = defaultdict(lambda: defaultdict(float), pickle.load(f))
            print("成功加载井字棋Q-Learning模型")
        except FileNotFoundError:
            print("未找到已训练的井字棋模型，使用默认模型")

# Minimax AI 代码
class TictactoeState:
    def __init__(self, board, player):
        self.board = board.copy()
        self.player = player
        self.children = []
        self.score = None

    def get_possible_moves(self):
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    moves.append((i, j))
        return moves
    
    def check_over(self):
        # 检查行
        for i in range(3):
            if abs(np.sum(self.board[i, :])) == 3:
                return np.sum(self.board[i, :]) / 3
                
        # 检查列
        for i in range(3):
            if abs(np.sum(self.board[:, i])) == 3:
                return np.sum(self.board[:, i]) / 3
            
        # 检查对角线
        m = self.board[0][0] + self.board[1][1] + self.board[2][2]
        n = self.board[0][2] + self.board[1][1] + self.board[2][0]
        
        if abs(m) == 3:
            return m / 3
            
        if abs(n) == 3:
            return n / 3
            
        # 检查平局
        if np.count_nonzero(self.board) == 9:
            return 0
            
        return None
    
    def generate_tree_with_pruning(self, depth=0, max_depth=9, alpha=-np.inf, beta=np.inf):
        result = self.check_over()
        if result is not None:
            self.score = result
            return self.score
        
        if depth >= max_depth:
            return 0
        
        moves = self.get_possible_moves()

        if self.player == 1:  # 极大节点
            best_score = -np.inf
            for move in moves:
                child_board = self.board.copy()
                child_board[move] = self.player
                child_state = TictactoeState(child_board, -self.player)
                self.children.append(child_state)
                score = child_state.generate_tree_with_pruning(depth+1, max_depth, alpha, beta)
                best_score = max(score, best_score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
        else:  # 极小节点
            best_score = np.inf
            for move in moves:
                child_board = self.board.copy()
                child_board[move] = self.player
                child_state = TictactoeState(child_board, -self.player)
                self.children.append(child_state)
                score = child_state.generate_tree_with_pruning(depth+1, max_depth, alpha, beta)
                best_score = min(score, best_score)
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
                    
        self.score = best_score
        return best_score
    
    def get_best_move(self):
        if not self.children:
            return None
        
        if self.player == 1:
            best_child = max(self.children, key=lambda child: child.score)
        else:
            best_child = min(self.children, key=lambda child: child.score)

        # 找出变化的位置
        for i in range(3):
            for j in range(3):
                if self.board[i][j] != best_child.board[i][j]:
                    return (i, j)
                
        return None

class MinimaxAI:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
    
    def make_move(self, flat_board):
        
        board_2d = np.zeros((3, 3), dtype=int)
        for i in range(9):
            row, col = i // 3, i % 3
            if flat_board[i] == 1:  # X 玩家
                board_2d[row][col] = 1
            elif flat_board[i] == -1:  # O 玩家
                board_2d[row][col] = -1
        
        self.board = board_2d
        
        
        x_count = np.count_nonzero(self.board == 1)
        o_count = np.count_nonzero(self.board == -1)
        
        if x_count > o_count:
            current_player = -1  
        else:
            current_player = 1   
            
    
        root = TictactoeState(self.board, current_player)
        root.generate_tree_with_pruning()
        move = root.get_best_move()
        
        if move:
            evaluation = root.score
            return move, evaluation
        return None, 0


q_learning_ai = TicTacToeAI()
minimax_ai = MinimaxAI()

#==========================================================
# 四子棋相关代码
#==========================================================

def softmax(x):
    """计算softmax值"""
    probs = np.exp(x - np.max(x))
    return probs / np.sum(probs)

# 四子棋棋盘类
class ConnectFourBoard:
    def __init__(self, width=6, height=6, n_in_row=4):
        self.width = width
        self.height = height
        self.n_in_row = n_in_row
        self.players = [1, 2]  # 玩家1(红色)和玩家2(黄色)
        self.states = {}  # 棋盘状态: {位置: 玩家}
        self.availables = list(range(width * height))  # 可用位置列表
        self.last_move = -1  # 最后一步

    def init_board(self, start_player=0):
        """初始化棋盘"""
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception(f'棋盘宽度和高度不能小于 {self.n_in_row}')
        self.current_player = self.players[start_player]  # 先手玩家
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        """将move索引转换为(row, col)坐标"""
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        """将(row, col)坐标转换为move索引"""
        if len(location) != 2:
            return -1
        h, w = location
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def get_available_moves_by_column(self):
        """获取每列的可用位置（最低的空位置）"""
        # 初始化每列的最低可用位置
        column_moves = {}
        
        # 获取所有可用位置
        available_set = set(self.availables)
        
        for col in range(self.width):
            # 从底部向上检查每一列
            for row in range(self.height - 1, -1, -1):
                move = row * self.width + col
                if move in available_set:
                    column_moves[col] = move
                    break
        
        return column_moves

    def do_move(self, move):
        """执行一步落子"""
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = self.players[0] if self.current_player == self.players[1] else self.players[1]
        self.last_move = move

    def has_a_winner(self):
        """检查是否有获胜者"""
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row * 2 - 1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            # 检查水平方向
            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            # 检查垂直方向
            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            # 检查对角线（正斜率）
            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            # 检查对角线（负斜率）
            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """检查游戏是否结束"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1  # 平局
        return False, -1

    def get_current_player(self):
        """获取当前玩家"""
        return self.current_player

    def current_state(self):
        """返回当前玩家视角的棋盘状态（用于神经网络输入）"""
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # 指示当前玩家的颜色
        return square_state[:, ::-1, :]

# 纯MCTS四子棋AI
def rollout_policy_fn(board):
    """快速走子策略，在rollout阶段使用"""
    # 随机rollout
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)

def policy_value_fn(board):
    """一个函数，输入一个棋盘状态，输出(action, probability)元组列表和棋盘状态的得分"""
    # 返回均匀概率和纯MCTS的0分
    action_probs = np.ones(len(board.availables))/len(board.availables)
    return zip(board.availables, action_probs), 0

class TreeNode:
    """MCTS树中的节点"""
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # 从action到TreeNode的映射
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """扩展树节点，创建新的子节点"""
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """在子节点中选择能够提供最大行动价值的节点"""
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """从叶子评估中更新节点值"""
        self._n_visits += 1
        self._Q += (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """递归更新所有祖先节点"""
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """计算并返回此节点的值"""
        self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        return self._Q + self._u

    def is_leaf(self):
        """检查是否是叶子节点"""
        return self._children == {}

    def is_root(self):
        """检查是否是根节点"""
        return self._parent is None

class MCTS_Pure:
    """纯蒙特卡洛树搜索实现"""
    def __init__(self, policy_value_fn, c_puct=5, n_playout=2000):
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self.player = None

    def _playout(self, state):
        """从根到叶子运行单个playout"""
        node = self._root
        while not node.is_leaf():
            # 贪婪地选择下一步动作
            action, node = node.select(self._c_puct)
            state.do_move(action)

        # 使用策略函数评估叶子
        action_probs, _ = self._policy(state)
        
        # 检查游戏结束
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        
        # 随机rollout到游戏结束
        leaf_value = self._evaluate_rollout(state)
        
        # 更新此遍历中节点的值
        node.update_recursive(-leaf_value)

    def _evaluate_rollout(self, state, limit=1000):
        """使用rollout策略一直玩到游戏结束"""
        player = state.get_current_player()
        for i in range(limit):
            end, winner = state.game_end()
            if end:
                break
            action_probs = rollout_policy_fn(state)
            max_action = max(action_probs, key=itemgetter(1))[0]
            state.do_move(max_action)
        else:
            # 如果没有从循环中break，发出警告
            print("警告: rollout达到移动限制")
            
        if winner == -1:  # 平局
            return 0
        else:
            return 1 if winner == player else -1

    def get_move(self, state):
        """顺序运行所有playouts并返回访问次数最多的动作"""
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        return max(self._root._children.items(),
                   key=lambda act_node: act_node[1]._n_visits)[0]

    def get_action(self, board):
        """获取AI的动作"""
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move = self.get_move(board)
            self.update_with_move(-1)
            return move
        else:
            print("警告: 棋盘已满")
            return -1

    def update_with_move(self, last_move):
        """在树中前进一步，保留已知的子树信息"""
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def set_player_ind(self, p):
        """设置玩家编号"""
        self.player = p

    def reset_player(self):
        """重置玩家"""
        self.update_with_move(-1)
        
    def __str__(self):
        return f"MCTS Pure {self.player}"

# AlphaZero神经网络模型（四子棋）
class Net(nn.Module):
    """策略价值网络模块"""
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        # 公共层
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # 动作策略层
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*board_width*board_height,
                                 board_width*board_height)
        # 状态值层
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*board_width*board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # 公共层
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # 动作策略层
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)
        # 状态值层
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*self.board_width*self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val

class PolicyValueNet():
    """策略价值网络"""
    def __init__(self, board_width, board_height, model_file=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # l2惩罚的系数
        
        # 策略值网络模块
        if self.use_gpu:
            self.policy_value_net = Net(board_width, board_height).cuda()
        else:
            self.policy_value_net = Net(board_width, board_height)
            
        if model_file:
            try:
                net_params = torch.load(model_file)
                self.policy_value_net.load_state_dict(net_params)
                print(f"四子棋AlphaZero模型已加载: {model_file}")
            except Exception as e:
                print(f"加载四子棋模型错误: {e}")
                print("使用未训练的模型")

    def policy_value(self, state_batch):
        """输入一批状态，输出一批动作概率和状态值"""
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()

    def policy_value_fn(self, board):
        """输入棋盘，输出动作概率和状态值"""
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height))
        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
            value = value.data.cpu().numpy()[0][0]
        else:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
            value = value.data.numpy()[0][0]
        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs, value

# AlphaZero风格的MCTS
class TreeNode_AlphaZero:
    """AlphaZero MCTS树中的节点"""
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # 从action到TreeNode的映射
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """扩展树节点，创建新的子节点"""
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode_AlphaZero(self, prob)

    def select(self, c_puct):
        """在子节点中选择能够提供最大行动价值的节点"""
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """从叶子评估中更新节点值"""
        self._n_visits += 1
        self._Q += (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """递归更新所有祖先节点"""
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """计算并返回此节点的值"""
        self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        return self._Q + self._u

    def is_leaf(self):
        """检查是否是叶子节点"""
        return self._children == {}

    def is_root(self):
        """检查是否是根节点"""
        return self._parent is None

class MCTS_AlphaZero:
    """AlphaZero风格的蒙特卡洛树搜索实现"""
    def __init__(self, policy_value_fn, c_puct=5, n_playout=2000):
        self._root = TreeNode_AlphaZero(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """从根到叶子运行单个playout"""
        node = self._root
        while not node.is_leaf():
            # 贪婪地选择下一步动作
            action, node = node.select(self._c_puct)
            state.do_move(action)

        # 使用神经网络评估叶子
        action_probs, leaf_value = self._policy(state)
        
        # 检查游戏结束
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            # 对于结束状态，返回"真实"的leaf_value
            if winner == -1:  # 平局
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == state.get_current_player() else -1.0

        # 更新此遍历中节点的值
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """顺序运行所有playouts并返回可用动作及其对应概率"""
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # 根据根节点上的访问计数计算移动概率
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """在树中前进一步，保留已知的子树信息"""
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode_AlphaZero(None, 1.0)

    def __str__(self):
        return "MCTS AlphaZero"

class MCTSPlayer:
    """基于MCTS的AI玩家"""
    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS_AlphaZero(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.player = None

    def set_player_ind(self, p):
        """设置玩家编号"""
        self.player = p

    def reset_player(self):
        """重置玩家"""
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        """获取AI的动作"""
        sensible_moves = board.availables
        move_probs = np.zeros(board.width*board.height)
        
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs
            
            if self._is_selfplay:
                # 添加Dirichlet噪声进行探索
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                self.mcts.update_with_move(move)
            else:
                # 默认temp=1e-3，几乎等同于选择具有最高概率的移动
                move = np.random.choice(acts, p=probs)
                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("警告: 棋盘已满")
            return -1

    def __str__(self):
        return f"AlphaZero {self.player}"

# 初始化四子棋AI
# 创建四子棋棋盘
connect_four_board = ConnectFourBoard(width=6, height=6, n_in_row=4)

# 初始化AlphaZero AI - 加载预训练模型或使用未训练模型
# 统一初始化AlphaZero AI的部分

# 在server.py中找到初始化四子棋AI的代码块，并替换为:

# 初始化四子棋AI
# 创建四子棋棋盘
connect_four_board = ConnectFourBoard(width=6, height=6, n_in_row=4)

# 初始化AlphaZero AI - 加载预训练模型或使用未训练模型
try:
    # 尝试多个可能的模型路径
    model_paths = [
        
        'models/current.model',            # 当前server.py文件指定路径
        
    ]
    
    model_loaded = False
    for model_path in model_paths:
        try:
            if os.path.exists(model_path):
                policy_net = PolicyValueNet(6, 6, model_file=model_path)
                alphazero_ai = MCTSPlayer(policy_net.policy_value_fn, c_puct=5, n_playout=1000)
                print(f"成功加载四子棋AlphaZero模型: {model_path}")
                model_loaded = True
                break
        except Exception as e:
            print(f"尝试加载模型 {model_path} 失败: {e}")
    
    if not model_loaded:
        raise Exception("找不到有效的模型文件")
        
except Exception as e:
    print(f"加载四子棋AlphaZero模型失败: {e}")
    print("使用未训练的模型")
    policy_net = PolicyValueNet(6, 6)
    alphazero_ai = MCTSPlayer(policy_net.policy_value_fn, c_puct=5, n_playout=1000)

# 初始化纯MCTS AI
mcts_pure_ai = MCTS_Pure(policy_value_fn=policy_value_fn, c_puct=5, n_playout=1000)





#==========================================================
# 服务器路由
#==========================================================

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')  # 这里应该返回井字棋页面

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/tictactoe.js')
def serve_tictactoe_js():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(base_dir, 'tictactoe.js')  # 井字棋的JS

@app.route('/connect-four.html')
def serve_connect_four():
    return send_from_directory('.', 'connect-four.html')  # 四子棋的HTML

@app.route('/connectfour.js')
def serve_connect_four_js():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(base_dir, 'connectfour.js')  # 四子棋的JS

# 井字棋AI API端点
@app.route('/make_move', methods=['POST'])
def make_move():
    try:
        data = request.get_json()
        if not data or 'board' not in data:
            return jsonify({'error': '无效的请求数据'}), 400
        
        board = data['board']
        
        ai_board = []
        for cell in board:
            if cell == 'X':
                ai_board.append('X')
            elif cell == 'O':
                ai_board.append('O')
            else:
                ai_board.append('')
        
        move, analysis = q_learning_ai.get_best_move(ai_board)
        if move == -1:
            return jsonify({'error': '没有可用的移动位置'}), 400
            
        return jsonify({
            'move': move,
            'analysis': analysis,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500

@app.route('/minimax_move', methods=['POST'])
def minimax_move():
    try:
        data = request.get_json()
        if not data or 'board' not in data:
            return jsonify({'error': '无效的请求数据'}), 400
        
        flat_board = data['board']
        
        if len(flat_board) != 9:
            return jsonify({'error': '无效的棋盘状态'}), 400
        
        move, evaluation = minimax_ai.make_move(flat_board)
        
        if move is None:
            return jsonify({'error': '没有可用的移动位置'}), 400
        
        return jsonify({
            'move': move,
            'evaluation': float(evaluation),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500

@app.route('/alphazero_move', methods=['POST'])
def alphazero_move():
    try:
        data = request.get_json()
        if not data or 'board' not in data or 'current_player' not in data:
            return jsonify({'error': '无效的请求数据'}), 400
        
        # 转换棋盘格式从JavaScript到Python
        js_board = data['board']
        current_player = data['current_player']
        
        # 重置四子棋棋盘
        connect_four_board.init_board()
        
        # 映射棋盘状态到我们的ConnectFourBoard
        # 转换'red'为玩家1，'yellow'为玩家2
        for row in range(6):
            for col in range(6):
                if js_board[row][col] == 'red':
                    move = row * 6 + col
                    connect_four_board.states[move] = 1
                    connect_four_board.availables.remove(move)
                elif js_board[row][col] == 'yellow':
                    move = row * 6 + col
                    connect_four_board.states[move] = 2
                    connect_four_board.availables.remove(move)
        
        # 设置当前玩家
        connect_four_board.current_player = 1 if current_player == 'red' else 2
        
        # 检查是否有可用的落子位置
        if not connect_four_board.availables:
            return jsonify({'error': '没有可用的落子位置'}), 400
        
        # 获取AlphaZero的动作（整个棋盘的线性索引）
        alphazero_ai.set_player_ind(connect_four_board.current_player)
        move = alphazero_ai.get_action(connect_four_board)
        
        # 将move从线性索引转换为行列坐标
        row = int(move // 6)
        col = int(move % 6)
        
        # 返回具体的行列坐标
        return jsonify({
            'move': {
                'row': row,
                'col': col
            },
            'evaluation': 0.0,  # 可以在这里计算真实的评估
            'status': 'success'
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())  # 打印完整错误堆栈以便调试
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500

@app.route('/mcts_pure_move', methods=['POST'])
def mcts_pure_move():
    try:
        data = request.get_json()
        if not data or 'board' not in data or 'current_player' not in data:
            return jsonify({'error': '无效的请求数据'}), 400
        
        # 转换棋盘格式从JavaScript到Python
        js_board = data['board']
        current_player = data['current_player']
        
        # 重置四子棋棋盘
        connect_four_board.init_board()
        
        # 映射棋盘状态到我们的ConnectFourBoard
        # 转换'red'为玩家1，'yellow'为玩家2
        for row in range(6):
            for col in range(6):
                if js_board[row][col] == 'red':
                    move = row * 6 + col
                    connect_four_board.states[move] = 1
                    connect_four_board.availables.remove(move)
                elif js_board[row][col] == 'yellow':
                    move = row * 6 + col
                    connect_four_board.states[move] = 2
                    connect_four_board.availables.remove(move)
        
        # 设置当前玩家
        connect_four_board.current_player = 1 if current_player == 'red' else 2
        
        # 检查是否有可用的落子位置
        if not connect_four_board.availables:
            return jsonify({'error': '没有可用的落子位置'}), 400
        
        # 获取Pure MCTS的动作（整个棋盘的线性索引）
        mcts_pure_ai.set_player_ind(connect_four_board.current_player)
        move = mcts_pure_ai.get_action(connect_four_board)
        
        # 将move从线性索引转换为行列坐标
        row = int(move // 6)
        col = int(move % 6)
        
        # 返回具体的行列坐标
        return jsonify({
            'move': {
                'row': row,
                'col': col
            },
            'status': 'success'
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())  # 打印完整错误堆栈以便调试
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500

if __name__ == '__main__':
    print("启动游戏服务器 - 访问 http://localhost:5000 开始游戏")
    app.run(host='0.0.0.0', port=5000, debug=True)