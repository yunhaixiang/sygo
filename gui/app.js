const canvas = document.querySelector("#go-board");
const ctx = canvas.getContext("2d");
const boardSizeSelect = document.querySelector("#board-size");
const passButton = document.querySelector("#pass-button");
const undoButton = document.querySelector("#undo-button");
const resetButton = document.querySelector("#reset-button");
const playSygoButton = document.querySelector("#play-sygo-button");
const stopPlayButton = document.querySelector("#stop-play-button");
const watchButton = document.querySelector("#watch-button");
const stopWatchButton = document.querySelector("#stop-watch-button");
const monitorWorkerSelect = document.querySelector("#monitor-worker");
const boardTitle = document.querySelector("#board-title");
const turnLabel = document.querySelector("#turn-label");
const moveCountLabel = document.querySelector("#move-count");
const blackCapturesLabel = document.querySelector("#black-captures");
const whiteCapturesLabel = document.querySelector("#white-captures");
const komiLabel = document.querySelector("#komi-label");
const moveLog = document.querySelector("#move-log");
const lastMessage = document.querySelector("#last-message");
const monitorStatus = document.querySelector("#monitor-status");

const EMPTY = 0;
const BLACK = 1;
const WHITE = 2;
const STAR_POINTS = {
  9: [2, 4, 6],
  13: [3, 6, 9],
  19: [3, 9, 15],
};
const KOMI_BY_SIZE = {
  9: 7.0,
  13: 7.5,
  19: 7.5,
};
const MONITOR_URL = "../data/selfplay-monitor.json";

const state = {
  size: 9,
  board: [],
  turn: BLACK,
  captures: { [BLACK]: 0, [WHITE]: 0 },
  history: [],
  hover: null,
  moveCount: null,
  monitoring: false,
  apiMode: false,
  apiBusy: false,
  human: BLACK,
  monitorTimer: null,
  monitorUpdatedAt: null,
  monitorWorker: 1,
  modeTitle: "Local Board",
  supportedPlaySizes: [9],
  defaultPlaySize: 9,
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
  state.moveCount = null;
  state.apiMode = false;
  state.apiBusy = false;
  state.modeTitle = "Local Board";
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
  if (state.monitoring) return;
  if (state.apiMode) {
    playApiMove({ row, col });
    return;
  }
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
  if (state.monitoring) return;
  if (state.apiMode) {
    playApiMove({ pass: true });
    return;
  }
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
  if (state.monitoring || state.apiMode) return;
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
  boardTitle.textContent = state.modeTitle;
  turnLabel.textContent = colorName(state.turn);
  moveCountLabel.textContent = `${state.moveCount ?? state.history.length}`;
  blackCapturesLabel.textContent = `${state.captures[BLACK]}`;
  whiteCapturesLabel.textContent = `${state.captures[WHITE]}`;
  komiLabel.textContent = KOMI_BY_SIZE[state.size].toFixed(1);
  moveLog.replaceChildren();

  for (const entry of state.history) {
    const li = document.createElement("li");
    li.textContent = entry.label;
    moveLog.append(li);
  }
  moveLog.scrollTop = moveLog.scrollHeight;
}

function setMonitoring(enabled) {
  state.monitoring = enabled;
  state.modeTitle = enabled ? "Sygo vs. Sygo" : "Local Board";
  boardSizeSelect.disabled = enabled;
  passButton.disabled = enabled;
  undoButton.disabled = enabled;
  resetButton.disabled = enabled;
  playSygoButton.disabled = enabled;
  stopPlayButton.disabled = enabled || !state.apiMode;
  watchButton.disabled = enabled;
  stopWatchButton.disabled = !enabled;
  canvas.style.cursor = enabled ? "default" : "crosshair";
}

function startMonitoring() {
  if (state.monitorTimer) clearInterval(state.monitorTimer);
  state.monitorWorker = Number(monitorWorkerSelect.value);
  setMonitoring(true);
  monitorStatus.textContent = `Watching worker ${state.monitorWorker}`;
  pollMonitor();
  state.monitorTimer = window.setInterval(pollMonitor, 700);
}

function stopMonitoring() {
  if (state.monitorTimer) clearInterval(state.monitorTimer);
  state.monitorTimer = null;
  state.monitorUpdatedAt = null;
  setMonitoring(false);
  monitorStatus.textContent = "Manual board";
  state.modeTitle = "Local Board";
  syncUi();
}

function setApiMode(enabled) {
  state.apiMode = enabled;
  state.modeTitle = enabled ? "User vs. Sygo" : "Local Board";
  boardSizeSelect.disabled = enabled;
  undoButton.disabled = enabled;
  resetButton.disabled = enabled;
  playSygoButton.disabled = enabled;
  stopPlayButton.disabled = !enabled;
  watchButton.disabled = enabled;
  passButton.disabled = false;
  canvas.style.cursor = "crosshair";
  syncUi();
}

async function startSygoGame() {
  stopMonitoring();
  await loadPlayConfig();
  const selectedSize = Number(boardSizeSelect.value);
  const size = state.supportedPlaySizes.includes(selectedSize) ? selectedSize : state.defaultPlaySize;
  boardSizeSelect.value = `${size}`;
  reset(size);
  setMessage("Starting Sygo game");
  try {
    const data = await apiRequest("/api/new", {
      size,
      human: "black",
    });
    setApiMode(true);
    applyApiData(data);
  } catch {
    setMessage("Start with sygo-play, not python -m http.server", true);
  }
}

function stopSygoGame() {
  state.apiBusy = false;
  setApiMode(false);
  reset(state.defaultPlaySize);
  boardSizeSelect.value = `${state.defaultPlaySize}`;
  setMessage("Stopped game");
}

async function loadPlayConfig() {
  try {
    const response = await fetch("/api/config", { cache: "no-store" });
    if (!response.ok) return;
    const data = await response.json();
    const supported = (data.supported_sizes ?? []).map((value) => Number(value));
    if (supported.length > 0) {
      state.supportedPlaySizes = supported;
      state.defaultPlaySize = Number(data.default_size ?? supported[0]);
      setBoardSizeOptions(supported, state.defaultPlaySize);
    }
  } catch {
    state.supportedPlaySizes = [9];
    state.defaultPlaySize = 9;
  }
}

function setBoardSizeOptions(sizes, selectedSize) {
  for (const option of boardSizeSelect.options) {
    const value = Number(option.value);
    option.disabled = !sizes.includes(value);
  }
  boardSizeSelect.value = `${selectedSize}`;
}

async function playApiMove(move) {
  if (state.apiBusy || state.turn !== state.human) return;
  state.apiBusy = true;
  setMessage("Sygo thinking");
  try {
    const data = await apiRequest("/api/play", move);
    applyApiData(data);
  } catch (error) {
    setMessage(error.message || "Move failed", true);
  } finally {
    state.apiBusy = false;
  }
}

async function apiRequest(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "API request failed");
  }
  return data;
}

function applyApiData(data) {
  state.size = Number(data.size);
  state.board = data.board.map((row) => row.map((value) => Number(value)));
  state.turn = data.to_play === "white" ? WHITE : BLACK;
  state.human = data.human === "white" ? WHITE : BLACK;
  state.captures = {
    [BLACK]: Number(data.captures?.black ?? 0),
    [WHITE]: Number(data.captures?.white ?? 0),
  };
  state.moveCount = Number(data.move_count ?? 0);
  state.history = apiHistory(data);
  state.modeTitle = `${data.black_player ?? "Black"} vs. ${data.white_player ?? "White"}`;
  boardSizeSelect.value = `${state.size}`;
  setMessage(data.message ?? (state.turn === state.human ? "Your move" : "Sygo to play"));
  syncUi();
  draw();
}

function apiHistory(data) {
  const entries = [];
  entries.push(...pairedMoveHistory(data.moves ?? []));
  if (data.is_over && typeof data.area_score === "number") {
    entries.push({ label: `Final area score B-W ${data.area_score.toFixed(1)}` });
  }
  return entries;
}

async function pollMonitor() {
  try {
    const response = await fetchMonitor();
    if (!response.ok) {
      monitorStatus.textContent = `Waiting for worker ${state.monitorWorker}`;
      return;
    }
    const data = await response.json();
    if (data.updated_at === state.monitorUpdatedAt) return;
    state.monitorUpdatedAt = data.updated_at;
    applyMonitorData(data);
  } catch {
    monitorStatus.textContent = `Waiting for worker ${state.monitorWorker}`;
  }
}

async function fetchMonitor() {
  const timestamp = Date.now();
  const workerUrl = monitorUrlForWorker(state.monitorWorker);
  const response = await fetch(`${workerUrl}?t=${timestamp}`, { cache: "no-store" });
  if (response.ok || state.monitorWorker !== 1) return response;
  return fetch(`${MONITOR_URL}?t=${timestamp}`, { cache: "no-store" });
}

function monitorUrlForWorker(worker) {
  return `../data/selfplay-monitor-worker${worker}.json`;
}

function applyMonitorData(data) {
  if (!Array.isArray(data.board) || data.board.length === 0) return;
  const size = Number(data.size ?? data.board.length);
  state.size = size;
  state.board = data.board.map((row) => row.map((value) => Number(value)));
  state.turn = data.to_play === "white" ? WHITE : BLACK;
  state.captures = {
    [BLACK]: Number(data.captures?.black ?? 0),
    [WHITE]: Number(data.captures?.white ?? 0),
  };
  state.moveCount = Number(data.move_count ?? 0);
  state.hover = null;
  state.history = monitorHistory(data);
  state.modeTitle = monitorTitle(data);
  boardSizeSelect.value = `${size}`;

  const round = data.round ? `Round ${data.round}${data.rounds ? `/${data.rounds}` : ""}` : null;
  const game = data.game ? `Game ${data.game}${data.games ? `/${data.games}` : ""}` : null;
  const prefix = [round, game].filter(Boolean).join(" ");
  const worker = data.worker ? `Worker ${data.worker}${data.workers ? `/${data.workers}` : ""}` : null;
  const phase = data.is_over ? "finished" : "self-play";
  monitorStatus.textContent = [worker, prefix, phase].filter(Boolean).join(" - ");
  setMessage(monitorMessage(data));
  syncUi();
  draw();
}

function monitorTitle(data) {
  if (data.black_player || data.white_player) {
    return `${data.black_player ?? "Black"} vs. ${data.white_player ?? "White"}`;
  }
  return "Sygo vs. Sygo";
}

function monitorHistory(data) {
  const entries = [];
  const round = data.round ? `Round ${data.round}` : null;
  const game = data.game ? `Game ${data.game}` : null;
  if (round || game) {
    entries.push({ label: [round, game].filter(Boolean).join(", ") });
  }

  if (Array.isArray(data.moves) && data.moves.length > 0) {
    entries.push(...pairedMoveHistory(data.moves, { includeValues: true }));
  } else if (data.move) {
    const player = data.played_by === "white" ? "White" : "Black";
    entries.push({ label: `${player} ${data.move}` });
  } else {
    entries.push({ label: "Starting position" });
  }
  if (data.is_over && typeof data.area_score === "number") {
    entries.push({ label: `Final area score B-W ${data.area_score.toFixed(1)}` });
  }
  return entries;
}

function pairedMoveHistory(moves, options = {}) {
  const includeValues = Boolean(options.includeValues);
  const turns = [];
  const byTurn = new Map();

  for (const move of moves) {
    const number = Number(move.number);
    if (!Number.isFinite(number) || !move.move) continue;

    const turnNumber = Math.floor((number + 1) / 2);
    if (!byTurn.has(turnNumber)) {
      const turn = { number: turnNumber, black: null, white: null };
      byTurn.set(turnNumber, turn);
      turns.push(turn);
    }

    const label = moveLabel(move, includeValues);
    if (move.player === "white") {
      byTurn.get(turnNumber).white = label;
    } else {
      byTurn.get(turnNumber).black = label;
    }
  }

  return turns.map((turn) => {
    const black = turn.black ? `B ${turn.black}` : "B ...";
    const white = turn.white ? `W ${turn.white}` : "";
    return { label: [black, white].filter(Boolean).join("   ") };
  });
}

function moveLabel(move, includeValue) {
  if (!includeValue || typeof move.root_value !== "number") return move.move;
  return `${move.move} (${move.root_value.toFixed(3)})`;
}

function monitorMessage(data) {
  if (data.is_over) {
    return `Game finished, area score B-W ${Number(data.area_score ?? 0).toFixed(1)}`;
  }
  if (data.move) {
    const player = data.played_by === "white" ? "White" : "Black";
    return `${player} played ${data.move}`;
  }
  return `${colorName(state.turn)} to play`;
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
  if (state.monitoring || state.apiMode) return;
  reset(Number(boardSizeSelect.value));
});

passButton.addEventListener("click", pass);
undoButton.addEventListener("click", undo);
resetButton.addEventListener("click", () => {
  if (!state.monitoring && !state.apiMode) reset(state.size);
});
playSygoButton.addEventListener("click", startSygoGame);
stopPlayButton.addEventListener("click", stopSygoGame);
watchButton.addEventListener("click", startMonitoring);
stopWatchButton.addEventListener("click", stopMonitoring);
monitorWorkerSelect.addEventListener("change", () => {
  state.monitorWorker = Number(monitorWorkerSelect.value);
  state.monitorUpdatedAt = null;
  if (state.monitoring) {
    monitorStatus.textContent = `Watching worker ${state.monitorWorker}`;
    pollMonitor();
  }
});
window.addEventListener("resize", draw);

reset();
setMonitoring(false);
