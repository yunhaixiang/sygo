const canvas = document.querySelector("#go-board");
const ctx = canvas.getContext("2d");
const boardSizeSelect = document.querySelector("#board-size");
const passButton = document.querySelector("#pass-button");
const undoButton = document.querySelector("#undo-button");
const resetButton = document.querySelector("#reset-button");
const turnLabel = document.querySelector("#turn-label");
const moveCountLabel = document.querySelector("#move-count");
const blackCapturesLabel = document.querySelector("#black-captures");
const whiteCapturesLabel = document.querySelector("#white-captures");
const moveLog = document.querySelector("#move-log");
const lastMessage = document.querySelector("#last-message");

const EMPTY = 0;
const BLACK = 1;
const WHITE = 2;
const STAR_POINTS = {
  9: [2, 4, 6],
  13: [3, 6, 9],
  19: [3, 9, 15],
};

const state = {
  size: 19,
  board: [],
  turn: BLACK,
  captures: { [BLACK]: 0, [WHITE]: 0 },
  history: [],
  hover: null,
};

function createBoard(size) {
  return Array.from({ length: size }, () => Array(size).fill(EMPTY));
}

function reset(size = state.size) {
  state.size = size;
  state.board = createBoard(size);
  state.turn = BLACK;
  state.captures = { [BLACK]: 0, [WHITE]: 0 };
  state.history = [];
  state.hover = null;
  setMessage("Black to play");
  syncUi();
  draw();
}

function opponent(color) {
  return color === BLACK ? WHITE : BLACK;
}

function colorName(color) {
  return color === BLACK ? "Black" : "White";
}

function pointName(row, col) {
  const letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ";
  return `${letters[col]}${state.size - row}`;
}

function inBounds(row, col) {
  return row >= 0 && row < state.size && col >= 0 && col < state.size;
}

function neighbors(row, col) {
  return [
    [row - 1, col],
    [row + 1, col],
    [row, col - 1],
    [row, col + 1],
  ].filter(([r, c]) => inBounds(r, c));
}

function collectGroup(board, row, col) {
  const color = board[row][col];
  const stones = [];
  const liberties = new Set();
  const seen = new Set([`${row},${col}`]);
  const stack = [[row, col]];

  while (stack.length > 0) {
    const [r, c] = stack.pop();
    stones.push([r, c]);

    for (const [nr, nc] of neighbors(r, c)) {
      const value = board[nr][nc];
      if (value === EMPTY) {
        liberties.add(`${nr},${nc}`);
      } else if (value === color) {
        const key = `${nr},${nc}`;
        if (!seen.has(key)) {
          seen.add(key);
          stack.push([nr, nc]);
        }
      }
    }
  }

  return { stones, liberties };
}

function cloneBoard(board) {
  return board.map((row) => row.slice());
}

function playMove(row, col) {
  if (!inBounds(row, col) || state.board[row][col] !== EMPTY) {
    setMessage("Illegal move", true);
    return;
  }

  const board = cloneBoard(state.board);
  const color = state.turn;
  const rival = opponent(color);
  board[row][col] = color;

  const captured = [];
  for (const [nr, nc] of neighbors(row, col)) {
    if (board[nr][nc] !== rival) continue;
    const group = collectGroup(board, nr, nc);
    if (group.liberties.size === 0) {
      for (const [sr, sc] of group.stones) {
        board[sr][sc] = EMPTY;
        captured.push([sr, sc]);
      }
    }
  }

  const ownGroup = collectGroup(board, row, col);
  if (ownGroup.liberties.size === 0) {
    setMessage("Suicide is not allowed", true);
    return;
  }

  state.history.push({
    board: cloneBoard(state.board),
    turn: state.turn,
    captures: { ...state.captures },
    label: `${colorName(color)} ${pointName(row, col)}`,
  });
  state.board = board;
  state.captures[color] += captured.length;
  state.turn = rival;
  setMessage(`${colorName(color)} played ${pointName(row, col)}`);
  syncUi();
  draw();
}

function pass() {
  const color = state.turn;
  state.history.push({
    board: cloneBoard(state.board),
    turn: state.turn,
    captures: { ...state.captures },
    label: `${colorName(color)} pass`,
  });
  state.turn = opponent(color);
  setMessage(`${colorName(color)} passed`);
  syncUi();
  draw();
}

function undo() {
  const previous = state.history.pop();
  if (!previous) {
    setMessage("No move to undo", true);
    return;
  }
  state.board = previous.board;
  state.turn = previous.turn;
  state.captures = previous.captures;
  setMessage("Move undone");
  syncUi();
  draw();
}

function setMessage(message, isInvalid = false) {
  lastMessage.textContent = message;
  lastMessage.classList.toggle("invalid", isInvalid);
}

function syncUi() {
  turnLabel.textContent = colorName(state.turn);
  moveCountLabel.textContent = `${state.history.length}`;
  blackCapturesLabel.textContent = `${state.captures[BLACK]}`;
  whiteCapturesLabel.textContent = `${state.captures[WHITE]}`;
  moveLog.replaceChildren();

  for (const entry of state.history) {
    const li = document.createElement("li");
    li.textContent = entry.label;
    moveLog.append(li);
  }
  moveLog.scrollTop = moveLog.scrollHeight;
}

function boardMetrics() {
  const pixelRatio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const side = Math.max(320, Math.floor(rect.width * pixelRatio));
  if (canvas.width !== side || canvas.height !== side) {
    canvas.width = side;
    canvas.height = side;
  }
  const padding = side * 0.06;
  const gap = (side - padding * 2) / (state.size - 1);
  return { side, padding, gap };
}

function draw() {
  const { side, padding, gap } = boardMetrics();
  ctx.clearRect(0, 0, side, side);
  drawWood(side);
  drawGrid(padding, gap);
  drawStars(padding, gap);
  drawStones(padding, gap);
  drawHover(padding, gap);
}

function drawWood(side) {
  const gradient = ctx.createLinearGradient(0, 0, side, side);
  gradient.addColorStop(0, "#e8c36f");
  gradient.addColorStop(0.5, "#d3a24a");
  gradient.addColorStop(1, "#c28a35");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, side, side);
}

function drawGrid(padding, gap) {
  ctx.strokeStyle = "rgba(56, 35, 8, 0.82)";
  ctx.lineWidth = Math.max(1, gap * 0.018);
  ctx.beginPath();

  for (let i = 0; i < state.size; i += 1) {
    const p = padding + i * gap;
    ctx.moveTo(padding, p);
    ctx.lineTo(padding + gap * (state.size - 1), p);
    ctx.moveTo(p, padding);
    ctx.lineTo(p, padding + gap * (state.size - 1));
  }
  ctx.stroke();
}

function drawStars(padding, gap) {
  const points = STAR_POINTS[state.size] ?? [];
  ctx.fillStyle = "rgba(56, 35, 8, 0.9)";
  for (const row of points) {
    for (const col of points) {
      ctx.beginPath();
      ctx.arc(padding + col * gap, padding + row * gap, Math.max(3, gap * 0.08), 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function drawStones(padding, gap) {
  for (let row = 0; row < state.size; row += 1) {
    for (let col = 0; col < state.size; col += 1) {
      const color = state.board[row][col];
      if (color !== EMPTY) {
        drawStone(padding + col * gap, padding + row * gap, gap * 0.43, color);
      }
    }
  }
}

function drawStone(x, y, radius, color) {
  const gradient = ctx.createRadialGradient(
    x - radius * 0.28,
    y - radius * 0.34,
    radius * 0.12,
    x,
    y,
    radius,
  );
  if (color === BLACK) {
    gradient.addColorStop(0, "#62666d");
    gradient.addColorStop(0.38, "#20242b");
    gradient.addColorStop(1, "#050608");
  } else {
    gradient.addColorStop(0, "#ffffff");
    gradient.addColorStop(0.5, "#ebe7dd");
    gradient.addColorStop(1, "#b9b2a5");
  }

  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = color === BLACK ? "rgba(0, 0, 0, 0.38)" : "rgba(81, 73, 62, 0.36)";
  ctx.lineWidth = Math.max(1, radius * 0.06);
  ctx.stroke();
}

function drawHover(padding, gap) {
  if (!state.hover) return;
  const { row, col } = state.hover;
  if (!inBounds(row, col) || state.board[row][col] !== EMPTY) return;

  ctx.globalAlpha = 0.35;
  drawStone(padding + col * gap, padding + row * gap, gap * 0.43, state.turn);
  ctx.globalAlpha = 1;
}

function eventPoint(event) {
  const rect = canvas.getBoundingClientRect();
  const { padding, gap } = boardMetrics();
  const scale = canvas.width / rect.width;
  const x = (event.clientX - rect.left) * scale;
  const y = (event.clientY - rect.top) * scale;
  const col = Math.round((x - padding) / gap);
  const row = Math.round((y - padding) / gap);
  const snapX = padding + col * gap;
  const snapY = padding + row * gap;
  const distance = Math.hypot(x - snapX, y - snapY);
  return distance <= gap * 0.46 ? { row, col } : null;
}

canvas.addEventListener("click", (event) => {
  const point = eventPoint(event);
  if (point) playMove(point.row, point.col);
});

canvas.addEventListener("mousemove", (event) => {
  state.hover = eventPoint(event);
  draw();
});

canvas.addEventListener("mouseleave", () => {
  state.hover = null;
  draw();
});

boardSizeSelect.addEventListener("change", () => {
  reset(Number(boardSizeSelect.value));
});

passButton.addEventListener("click", pass);
undoButton.addEventListener("click", undo);
resetButton.addEventListener("click", () => reset(state.size));
window.addEventListener("resize", draw);

reset();
