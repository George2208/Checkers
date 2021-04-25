from copy import deepcopy
from statistics import median
from threading import Thread
from multiprocessing import Queue
from time import time, sleep
import pygame

RUN = True
PB, KB, PW, KW = -1, -2, 1, 2
enemies = {PB: {PW, KW}, PW: {PB, KB}, KB: {PW, KW}, KW: {PB, KB}}
directions = {
    PB: {(-1, -1), (-1, 1)}, PW: {(1, -1), (1, 1)},
    KB: {(-1, -1), (-1, 1), (1, -1), (1, 1)},
    KW: {(-1, -1), (-1, 1), (1, -1), (1, 1)}
}


def getPossibilities(layout: list[list[int]], whiteNext: bool) -> dict:
    """Get the tree of possible moves or takes for a given layout."""
    def getCaptures():
        def getCapturesFor(board, i, j):
            aux = {}
            for d in directions[board[i][j]]:
                if 0 <= i + 2 * d[0] < 8 and 0 <= j + 2 * d[1] < 8 and not board[i + 2 * d[0]][j + 2 * d[1]] \
                       and board[i + d[0]][j + d[1]] in enemies[board[i][j]]:
                    cpy = deepcopy(board)
                    cpy[i + 2 * d[0]][j + 2 * d[1]] = cpy[i][j]
                    cpy[i][j] = cpy[i + d[0]][j + d[1]] = 0
                    aux[i + 2 * d[0], j + 2 * d[1]] = getCapturesFor(cpy, i + 2 * d[0], j + 2 * d[1])
            return aux
        captures = {}
        for line in range(8):
            for column in range(8):
                if layout[line][column] in {False: {PB, KB}, True: {PW, KW}}[whiteNext]:
                    cap = getCapturesFor(layout, line, column)
                    if cap:
                        captures[(line, column)] = cap
        return captures

    def getMoves():
        moves = {}
        for i in range(8):
            for j in range(8):
                if layout[i][j] in {False: {PB, KB}, True: {PW, KW}}[whiteNext]:
                    aux = []
                    for d in directions[layout[i][j]]:
                        if 0 <= i + d[0] < 8 and 0 <= j + d[1] < 8 and not layout[i + d[0]][j + d[1]]:
                            aux.append((i + d[0], j + d[1]))
                    if aux:
                        moves[(i, j)] = {i: {} for i in aux}
        return moves

    possibilities = getCaptures()
    if not possibilities:
        possibilities = getMoves()
    return possibilities


def getFinalStates(initialBoard: list[list[int]], whiteNext: bool, paths=False) -> list[list[list[int]]] or dict[tuple[tuple[int]], tuple[tuple[int]]]:
    """
    Get all possible outcomes for a given board.
    paths True -> get a list of the layouts
    paths False -> get a dictionary with layouts as keys and the succession of moves as values
    """
    def getMoveSuccessions(tree, path=()):
        pos = []
        for branch in tree:
            if not tree[branch]:
                pos.append(path + (branch,))
            pos.extend(getMoveSuccessions(tree[branch], path + (branch,)))
        return pos

    def getFinalBoard(sc):
        def tryPromotion(x, y):
            if cpy[x][y] == PB and x == 0:
                cpy[x][y] = KB
            if cpy[x][y] == PW and x == 7:
                cpy[x][y] = KW

        def makeMove(selected, target):
            if abs(selected[0] - target[0]) == 1:
                cpy[target[0]][target[1]] = cpy[selected[0]][selected[1]]
                cpy[selected[0]][selected[1]] = 0
            else:
                cpy[target[0]][target[1]] = cpy[selected[0]][selected[1]]
                cpy[selected[0]][selected[1]] = cpy[(target[0]+selected[0])//2][(target[1]+selected[1])//2] = 0
            tryPromotion(*target)

        cpy = deepcopy(initialBoard)
        for seq in range(len(sc)-1):
            makeMove(*sc[seq:seq+2])
        if paths:
            return tuple(tuple(line) for line in cpy)
        return cpy

    if paths:
        return {getFinalBoard(succession): succession for succession in getMoveSuccessions(getPossibilities(initialBoard, whiteNext))}
    return [getFinalBoard(succession) for succession in getMoveSuccessions(getPossibilities(initialBoard, whiteNext))]


class Engine:
    def __init__(self):
        self.board, self.whiteNext = [
            [0,  PW,  0, PW,  0, PW,  0, PW],
            [PW,  0, PW,  0, PW,  0, PW,  0],
            [0,  PW,  0, PW,  0, PW,  0, PW],
            [0,   0,  0,  0,  0,  0,  0,  0],
            [0,   0,  0,  0,  0,  0,  0,  0],
            [PB,  0, PB,  0, PB,  0, PB,  0],
            [0,  PB,  0, PB,  0, PB,  0, PB],
            [PB,  0, PB,  0, PB,  0, PB,  0]
        ], False
        self.validMoves = getPossibilities(self.board, self.whiteNext)

    def move(self, selected, target):
        def tryPromotion(x, y):
            if self.board[x][y] == PB and x == 0:
                self.board[x][y] = KB
                print('Black promoted')
            if self.board[x][y] == PW and x == 7:
                self.board[x][y] = KW
                print('White promoted')

        if selected in self.validMoves and target in self.validMoves[selected]:
            self.board[target[0]][target[1]] = self.board[selected[0]][selected[1]]
            self.board[selected[0]][selected[1]] = 0
            tryPromotion(target[0], target[1])
            if abs(selected[0] - target[0]) == 1:
                self.whiteNext = not self.whiteNext
                self.validMoves = getPossibilities(self.board, self.whiteNext)
                print('White next' if self.whiteNext else 'Black next')
            else:
                self.board[(target[0]+selected[0])//2][(target[1]+selected[1])//2] = 0
                self.validMoves = {target: self.validMoves[selected][target]}
                if list(self.validMoves.values()) == [{}]:
                    self.whiteNext = not self.whiteNext
                    self.validMoves = getPossibilities(self.board, self.whiteNext)
                    print('White next' if self.whiteNext else 'Black next')
            if not self.validMoves:
                print('Black won!' if self.whiteNext else 'White won!')
                return False
        else:
            print('Invalid move')
        return True


def evaluator1(board: list[list[int]]) -> int:
    """Evaluate the board by the sum of all pieces. The value is 1 for normal pieces and 2 for kings"""
    return sum(board[line][column] for line in range(8) for column in range(8))


def evaluator2(board: list[list[int]]) -> int:
    """For every piece multiply its value with the estimated advantage of the square"""
    ___ = 0
    positions = [
        [___, 1.1, ___, 1.1, ___, 1.1, ___, 1.1],
        [1.1, ___, 1.1, ___, 1.1, ___, 1.1, ___],
        [___, 1.0, ___, 1.0, ___, 1.0, ___, 1.1],
        [1.2, ___, 1.0, ___, 1.0, ___, 1.0, ___],
        [___, 1.0, ___, 1.0, ___, 1.0, ___, 1.2],
        [1.1, ___, 1.0, ___, 1.0, ___, 1.0, ___],
        [___, 1.1, ___, 1.0, ___, 1.0, ___, 1.1],
        [1.1, ___, 1.1, ___, 1.1, ___, 1.1, ___]
    ]
    return sum(board[line][column] * positions[line][column] for column in range(8) for line in range(8))


def minMax(initialBoard: list[list[int]], AIWhite: bool, evaluator, initialDepth: int) -> (tuple[(int, int)], int, int):
    def _minMax(board, whiteNext=True, depth=initialDepth):
        if not depth or not RUN:
            return board, evaluator(board)
        nonlocal count
        count += 1
        possibilities = [_minMax(i, not whiteNext, depth-1) for i in getFinalStates(board, whiteNext)]
        if len(possibilities) == 1:
            return possibilities[0], evaluator(possibilities[0])
        if not possibilities:
            return board, -float('inf') if whiteNext else float('inf')
        best = sorted(possibilities, key=lambda x: x[-1])[-1 if whiteNext else 0]
        return (board,) + (*best,)

    count = 0
    mm = _minMax(initialBoard, AIWhite)
    if len(mm) < 3:
        return [{}, {}]
    return getFinalStates(initialBoard, AIWhite, True)[tuple(tuple(line) for line in mm[1])], count, mm[-1]


def alphaBeta(initialBoard: list[list[int]], AIWhite: bool, evaluator, initialDepth: int) -> (tuple[(int, int)], int, int):
    def _alphaBeta(board, whiteNext, depth=initialDepth, alpha=float('-inf'), beta=float('inf')) -> (list[list[int]] or None, float):
        if not depth or not RUN:
            return None, evaluator(board)
        nonlocal count
        count += 1
        possibilities = sorted(getFinalStates(board, whiteNext), key=lambda x: evaluator2(x))
        if len(possibilities) == 1:
            return possibilities[0], evaluator(possibilities[0])
        value = float('-inf') if whiteNext else float('inf')
        table = None
        for i in possibilities:
            pruning = _alphaBeta(i, not whiteNext, depth-1, alpha, beta)[1]
            if (whiteNext and pruning >= value) or (not whiteNext and pruning <= value):
                value = pruning
                table = i
            if whiteNext:
                alpha = max(alpha, value)
            else:
                beta = min(beta, value)
            if alpha >= beta:
                break
        return table, value
    count = 0
    ab = _alphaBeta(initialBoard, AIWhite)
    return getFinalStates(initialBoard, AIWhite, True)[tuple(tuple(line) for line in ab[0])], count, ab[1]


def prettyPrint(matrix: list[list[int]]) -> None:
    pieces = {PW: '\033[91m⬯\033[0m', KW: '\033[91m⬮\033[0m', PB: '\033[94m⬯\033[0m', KB: '\033[94m⬮\033[0m', 0: ' '}
    print('   a   b   c   d   e   f   g   h')
    for line in range(len(matrix)):
        print(line, *[pieces[piece] for piece in matrix[line]])


class Game(Engine):
    def __init__(self):
        def ask(question, true=frozenset(['1', 'y', 'yes', 't', 'true']), false=frozenset(['0', 'n', 'no', 'f', 'false'])):
            while True:
                answer = input(question)
                if answer in true:
                    return True
                if answer in false:
                    return False
                print('\tInvalid answer')

        super().__init__()
        self.queue = Queue()
        self.stop = False
        self.timeSpent = {True: [], False: []}
        self.nrNodes = {True: [], False: []}
        self.startTime = time()
        self.bots = {True: ask('Is white bot? '), False: ask('Is black bot? ')}
        for i in self.bots:
            if self.bots[i]:
                print(f'Options for {"White" if i else "Black"}:')
                algorithm = alphaBeta if ask('\tAlphaBeta(1) or MinMax(2)? ', frozenset(['1', 'alphabeta', 'AlphaBeta', '']), frozenset(['2', 'minmax', 'MinMax'])) else minMax
                evaluator = evaluator1 if ask('\tEvaluator1 or evaluator2? ', frozenset(['1']), frozenset(['2', ''])) else evaluator2
                while True:
                    response = input('\tDifficulty (easy(1) / medium(2) / hard(3)): ')
                    if response in {'easy', '1'}:
                        self.bots[i] = (algorithm, evaluator, 3)
                        break
                    if response in {'medium', '2'}:
                        self.bots[i] = (algorithm, evaluator, 5)
                        break
                    if response in {'hard', '3'}:
                        self.bots[i] = (algorithm, evaluator, 7)
                        break
                    print('\tInvalid answer')
                print(f'\033[32m\tAlgorithm: {algorithm.__name__}, Evaluator: {evaluator.__name__}, Depth: {self.bots[i][2]}\033[0m')
        self.players = {True: 'White(bot)' if self.bots[True] else 'White',
                        False: 'Black(bot)' if self.bots[False] else 'Black'}
        # Thread(target=self.console).start()
        Thread(target=self.graphics).start()
        Thread(target=self.processMoves).start()

    def finalize(self):
        print(f'{self.players[True]}: {len(self.timeSpent[True])} moves')
        print(f'{self.players[False]}: {len(self.timeSpent[False])} moves')
        for i in self.bots:
            if self.bots[i]:
                tStats, nStats = self.timeSpent[i], self.nrNodes[i]
                print(f'{self.players[True]} thought {sum(tStats)} seconds, min {min(tStats)}, max {max(tStats)}, avg {sum(tStats) / len(tStats)}, median {median(tStats)}'
                      f'\n\t\t\tNodes: min {min(nStats)}, max {max(nStats)}, avg {sum(nStats) / len(nStats)}, median {median(nStats)}')
        print(f'Total time: {time()-self.startTime}')

    def processMoves(self):
        prettyPrint(self.board)
        repetitions = set()
        while self.validMoves:
            if str(self.board) in repetitions:
                print('Draw by repetition')
                # self.stop = True
                self.finalize()
                return
            if self.stop:
                self.finalize()
                return
            repetitions.add(str(self.board))
            print(self.players[self.whiteNext] + ' next')
            lastMove = time()
            if self.bots[self.whiteNext]:
                botMoves, nrNodes, estimation = self.bots[self.whiteNext][0](self.board, self.whiteNext, self.bots[self.whiteNext][1], self.bots[self.whiteNext][2])
                self.nrNodes[self.whiteNext].append(nrNodes)
                self.timeSpent[self.whiteNext].append(time() - lastMove)
                print(f'{self.players[self.whiteNext]} thought {self.timeSpent[self.whiteNext][-1]} seconds, {nrNodes} nodes, estimation {estimation}')
                for i in range(len(botMoves) - 1):
                    self.move(botMoves[i], botMoves[i + 1])
                    prettyPrint(self.board)
                    self.nrNodes[self.whiteNext].append(0)
                    self.timeSpent[self.whiteNext].append(0)
                    sleep(.2)
                self.nrNodes[self.whiteNext].pop(-1)
                self.timeSpent[self.whiteNext].pop(-1)
            else:
                nextMove = self.queue.get()
                if nextMove == [None, None]:
                    self.finalize()
                    return
                self.timeSpent[self.whiteNext].append(time() - lastMove)
                print(f'{self.players[self.whiteNext]} thought {self.timeSpent[self.whiteNext][-1]} seconds')
                if nextMove[0] in self.validMoves and nextMove[1] in self.validMoves[nextMove[0]]:
                    self.move(*nextMove)
                    prettyPrint(self.board)
                else:
                    print('invalid move')
        print(self.players[not self.whiteNext] + ' won!')
        self.finalize()

    def console(self):
        while self.validMoves:
            if self.stop:
                self.queue.put((None, None))
                return
            if not self.bots[self.whiteNext]:
                try:
                    inp = input()
                    if self.stop or inp in {'stop', 'exit'}:
                        self.stop = True
                        global RUN
                        RUN = False
                        continue
                    inp = [int(i) for i in inp.split()]
                    if (inp[0], inp[1]) in self.validMoves and (inp[2], inp[3]) in self.validMoves[(inp[0], inp[1])]:
                        self.queue.put(((inp[0], inp[1]), (inp[2], inp[3])))
                except Exception:
                    print("Invalid input")

    def graphics(self):
        def drawBoard():
            RADIUS, CELL = SIZE // 20, SIZE // 8
            WHITE, BLACK = (200, 200, 200), (0, 0, 0)
            COLOR1, COLOR2 = (50, 25, 200), (200, 50, 25)
            ACCENT1, ACCENT2 = (100, 100, 100), (25, 100, 0)
            win.fill(BLACK)
            for i in range(8):
                for j in range(i % 2, 8, 2):
                    pygame.draw.rect(win, WHITE, (i * CELL, j * CELL, CELL, CELL))
            for i in range(8):
                for j in range(8):
                    if (i, j) in self.validMoves and not self.bots[self.whiteNext]:
                        pygame.draw.rect(win, ACCENT1, (j * CELL, i * CELL, CELL, CELL))
                    if (i, j) == selected:
                        pygame.draw.rect(win, ACCENT2, (j * CELL, i * CELL, CELL, CELL))
                    piece = self.board[i][j]
                    if piece == PB:
                        pygame.draw.circle(win, COLOR1, (j * CELL + CELL // 2, i * CELL + CELL // 2), RADIUS)
                    if piece == PW:
                        pygame.draw.circle(win, COLOR2, (j * CELL + CELL // 2, i * CELL + CELL // 2), RADIUS)
                    if piece == KB:
                        pygame.draw.circle(win, COLOR1, (j * CELL + CELL // 2, i * CELL + CELL // 2), RADIUS)
                        pygame.draw.circle(win, ACCENT1, (j * CELL + CELL // 2, i * CELL + CELL // 2), RADIUS // 1.5)
                    if piece == KW:
                        pygame.draw.circle(win, COLOR2, (j * CELL + CELL // 2, i * CELL + CELL // 2), RADIUS)
                        pygame.draw.circle(win, ACCENT1, (j * CELL + CELL // 2, i * CELL + CELL // 2), RADIUS // 1.5)
            if selected in self.validMoves:
                for option in self.validMoves[selected]:
                    pygame.draw.circle(win, ACCENT2, (option[1] * CELL + CELL // 2, option[0] * CELL + CELL // 2), RADIUS // 1.5)

        SIZE = 600
        pygame.init()
        pygame.display.set_caption('Dame')
        win, clock = pygame.display.set_mode((SIZE, SIZE)), pygame.time.Clock()
        selected, previous = None, None
        while self.validMoves:
            if self.stop:
                return
            clock.tick(30)
            drawBoard()
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    self.stop = True
                    self.queue.put((None, None))
                    global RUN
                    RUN = False
                    return
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    previous = selected
                    selected = (pos[1] * 8 // SIZE, pos[0] * 8 // SIZE)
                    if previous in self.validMoves and selected in self.validMoves[previous]:
                        self.queue.put((previous, selected))
                        selected = None


Game()
