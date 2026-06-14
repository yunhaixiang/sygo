const canvas = document.querySelector("#go-board");
const ctx = canvas.getContext("2d");
const boardSizeSelect = document.querySelector("#board-size");
const modeSelect = document.querySelector("#board-mode");
const checkpointSelect = document.querySelector("#checkpoint-select");
const passButton = document.querySelector("#pass-button");
const undoButton = document.querySelector("#undo-button");
const resetButton = document.querySelector("#reset-button");
const startModeButton = document.querySelector("#start-mode-button");
const stopModeButton = document.querySelector("#stop-mode-button");
const monitorWorkerSelect = document.querySelector("#monitor-worker");
const loadSgfInput = document.querySelector("#load-sgf-input");
const loadSgfButton = document.querySelector("#load-sgf-button");
const saveSgfButton = document.querySelector("#save-sgf-button");
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
  undoStack: [],
  moves: [],
  hover: null,
  moveCount: null,
  monitoring: false,
  apiMode: false,
  apiBusy: false,
  human: BLACK,
  monitorTimer: null,
  monitorUpdatedAt: null,
  monitorWorker: 1,
  mode: "local",
  modeTitle: "Local Board",
  supportedPlaySizes: [9],
  defaultPlaySize: 9,
  checkpoints: [],
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
  state.undoStack = [];
  state.moves = [];
  state.hover = null;
  state.moveCount = null;
  state.apiMode = false;
  state.apiBusy = false;
  state.mode = "local";
  state.modeTitle = "Local Board";
  modeSelect.value = "local";
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
  return pointNameForSize(row, col, state.size);
}

function pointNameForSize(row, col, size) {
  const letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ";
  return `${letters[col]}${size - row}`;
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

function playMove(row, col, options = {}) {
  if (state.monitoring) return false;
  if (state.apiMode && !options.localOnly) {
    playApiMove({ row, col });
    return true;
  }
  if (!inBounds(row, col) || state.board[row][col] !== EMPTY) {
    setMessage("Illegal move", true);
    return false;
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
    return false;
  }

  state.undoStack.push({
    board: cloneBoard(state.board),
    turn: state.turn,
    captures: { ...state.captures },
  });
  state.board = board;
  state.captures[color] += captured.length;
  state.turn = rival;
  recordDisplayedMove(color, { row, col });
  if (!options.silent) {
    setMessage(`${colorName(color)} played ${pointName(row, col)}`);
  }
  syncUi();
  draw();
  return true;
}

function pass(options = {}) {
  if (state.monitoring) return false;
  if (state.apiMode && !options.localOnly) {
    playApiMove({ pass: true });
    return true;
  }
  const color = state.turn;
  state.undoStack.push({
    board: cloneBoard(state.board),
    turn: state.turn,
    captures: { ...state.captures },
  });
  state.turn = opponent(color);
  recordDisplayedMove(color, null);
  if (!options.silent) {
    setMessage(`${colorName(color)} passed`);
  }
  syncUi();
  draw();
  return true;
}

function undo() {
  if (state.monitoring || state.apiMode) return;
  const previous = state.undoStack.pop();
  if (!previous) {
    setMessage("No move to undo", true);
    return;
  }
  state.board = previous.board;
  state.turn = previous.turn;
  state.captures = previous.captures;
  state.moves.pop();
  state.history = pairedMoveHistory(state.moves);
  state.moveCount = state.moves.length;
  setMessage("Move undone");
  syncUi();
  draw();
}

function recordDisplayedMove(color, point) {
  const move = {
    number: state.moves.length + 1,
    player: color === WHITE ? "white" : "black",
    move: point ? pointName(point.row, point.col) : "pass",
  };
  if (point) {
    move.row = point.row;
    move.col = point.col;
  }
  state.moves.push(move);
  state.history = pairedMoveHistory(state.moves);
  state.moveCount = state.moves.length;
}

function setMessage(message, isInvalid = false) {
  lastMessage.textContent = message;
  lastMessage.classList.toggle("invalid", isInvalid);
}

function syncUi() {
  boardTitle.textContent = state.modeTitle;
  turnLabel.textContent = colorName(state.turn);
  moveCountLabel.textContent = `${state.moveCount ?? state.moves.length}`;
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
  refreshControls();
}

function refreshControls() {
  const active = state.monitoring || state.apiMode;
  const watching = state.monitoring;
  const playing = state.apiMode;
  modeSelect.value = state.mode;
  modeSelect.disabled = active;
  boardSizeSelect.disabled = active;
  passButton.disabled = watching || (playing && state.turn !== state.human);
  undoButton.disabled = active;
  resetButton.disabled = active;
  startModeButton.disabled = active;
  stopModeButton.disabled = !active;
  monitorWorkerSelect.disabled = watching;
  checkpointSelect.disabled = active;
  canvas.style.cursor = watching ? "default" : "crosshair";
}

function setMonitoring(enabled) {
  state.monitoring = enabled;
  state.mode = enabled ? "watch" : "local";
  state.modeTitle = enabled ? "Sygo vs. Sygo" : "Local Board";
  refreshControls();
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
  state.mode = enabled ? "play" : "local";
  state.modeTitle = enabled ? "User vs. Sygo" : "Local Board";
  refreshControls();
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
      checkpoint: checkpointSelect.value,
    });
    setApiMode(true);
    applyApiData(data);
  } catch (error) {
    setMessage(error.message || "Start with sygo-play, not python -m http.server", true);
  }
}

function stopSygoGame() {
  state.apiBusy = false;
  setApiMode(false);
  state.modeTitle = "Local Board";
  modeSelect.value = "local";
  setMessage("Stopped game");
  syncUi();
}

async function startSelectedMode() {
  const mode = modeSelect.value;
  if (mode === "play") {
    await startSygoGame();
    return;
  }
  if (mode === "watch") {
    stopSygoGame();
    startMonitoring();
    return;
  }
  stopMonitoring();
  stopSygoGame();
  reset(Number(boardSizeSelect.value));
}

function stopActiveMode() {
  if (state.monitoring) {
    stopMonitoring();
    setMessage("Stopped self-play watch");
    return;
  }
  if (state.apiMode) {
    stopSygoGame();
  }
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
    state.checkpoints = data.checkpoints ?? [];
    setCheckpointOptions(state.checkpoints, data.current_checkpoint ?? "");
  } catch {
    state.supportedPlaySizes = [9];
    state.defaultPlaySize = 9;
    state.checkpoints = [];
    setCheckpointOptions([], "");
  }
}

function setBoardSizeOptions(sizes, selectedSize) {
  for (const option of boardSizeSelect.options) {
    const value = Number(option.value);
    option.disabled = !sizes.includes(value);
  }
  boardSizeSelect.value = `${selectedSize}`;
}

function setCheckpointOptions(checkpoints, selectedCheckpoint) {
  const previous = checkpointSelect.value;
  checkpointSelect.replaceChildren();

  const uniformOption = document.createElement("option");
  uniformOption.value = "";
  uniformOption.textContent = "Uniform test evaluator";
  checkpointSelect.append(uniformOption);

  for (const checkpoint of checkpoints) {
    const option = document.createElement("option");
    option.value = checkpoint.id;
    option.textContent = checkpoint.label;
    checkpointSelect.append(option);
  }

  const ids = new Set(checkpoints.map((checkpoint) => checkpoint.id));
  const nextValue = previous && ids.has(previous) ? previous : selectedCheckpoint;
  checkpointSelect.value = ids.has(nextValue) ? nextValue : "";
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
  state.moves = normalizeMoves(data.moves ?? [], state.size);
  state.history = apiHistory(data);
  state.modeTitle = `${data.black_player ?? "Black"} vs. ${data.white_player ?? "White"}`;
  boardSizeSelect.value = `${state.size}`;
  setMessage(data.message ?? (state.turn === state.human ? "Your move" : "Sygo to play"));
  syncUi();
  draw();
}

function apiHistory(data) {
  const entries = [];
  entries.push(...pairedMoveHistory(state.moves));
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
  state.moves = normalizeMoves(data.moves ?? [], size);
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
    entries.push(...pairedMoveHistory(state.moves, { includeValues: true }));
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

function normalizeMoves(moves, size) {
  return moves
    .map((move, index) => {
      const player = move.player === "white" ? "white" : "black";
      const normalized = {
        number: Number(move.number ?? index + 1),
        player,
        move: `${move.move ?? "pass"}`,
      };
      const parsed = parsePointLabel(normalized.move, size);
      if (parsed) {
        normalized.row = parsed.row;
        normalized.col = parsed.col;
        normalized.move = pointNameForSize(parsed.row, parsed.col, size);
      } else {
        normalized.move = "pass";
      }
      if (typeof move.root_value === "number") {
        normalized.root_value = move.root_value;
      }
      return normalized;
    })
    .filter((move) => Number.isFinite(move.number));
}

function parsePointLabel(label, size) {
  if (!label || label.toLowerCase() === "pass") return null;
  const match = /^([A-HJ-Z])(\d+)$/i.exec(label.trim());
  if (!match) return null;
  const letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ";
  const col = letters.indexOf(match[1].toUpperCase());
  const row = size - Number(match[2]);
  if (!inBoundsForSize(row, col, size)) return null;
  return { row, col };
}

function inBoundsForSize(row, col, size) {
  return row >= 0 && row < size && col >= 0 && col < size;
}

function moveLabel(move, includeValue) {
  if (!includeValue || typeof move.root_value !== "number") return move.move;
  return `${move.move} (${move.root_value.toFixed(3)})`;
}

function saveSgf() {
  const sgf = writeSgf();
  const blob = new Blob([sgf], { type: "application/x-go-sgf;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  link.href = url;
  link.download = `sygo-${state.size}x${state.size}-${timestamp}.sgf`;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  setMessage("SGF saved");
}

function writeSgf() {
  const properties = [
    "GM[1]",
    "FF[4]",
    "CA[UTF-8]",
    "AP[Sygo]",
    `SZ[${state.size}]`,
    `KM[${KOMI_BY_SIZE[state.size].toFixed(1)}]`,
    "RU[CGOS/Tromp-Taylor]",
  ];
  const moves = state.moves
    .map((move) => {
      const color = move.player === "white" ? "W" : "B";
      return `;${color}[${sgfPoint(move)}]`;
    })
    .join("");
  return `(;${properties.join("")}${moves})\n`;
}

function sgfPoint(move) {
  if (move.move === "pass") return "";
  const row = Number.isInteger(move.row) ? move.row : parsePointLabel(move.move, state.size)?.row;
  const col = Number.isInteger(move.col) ? move.col : parsePointLabel(move.move, state.size)?.col;
  if (!inBoundsForSize(row, col, state.size)) return "";
  return `${String.fromCharCode(97 + col)}${String.fromCharCode(97 + row)}`;
}

function promptLoadSgf() {
  loadSgfInput.value = "";
  loadSgfInput.click();
}

async function loadSgfFile() {
  const file = loadSgfInput.files?.[0];
  if (!file) return;
  try {
    const content = await file.text();
    loadSgf(content, file.name);
  } catch (error) {
    setMessage(error.message || "Could not load SGF", true);
  }
}

function loadSgf(content, name = "SGF") {
  const snapshot = snapshotState();
  const record = parseSgf(content);
  try {
    stopMonitoring();
    stopSygoGame();
    reset(record.size);
    boardSizeSelect.value = `${record.size}`;
    state.modeTitle = name.replace(/\.sgf$/i, "") || "Loaded SGF";

    for (const move of record.moves) {
      if (move.player !== (state.turn === WHITE ? "white" : "black")) {
        throw new Error("SGF move order is not supported by this simple loader");
      }
      if (move.pass) {
        if (!pass({ silent: true, localOnly: true })) {
          throw new Error("Could not replay SGF pass");
        }
      } else {
        if (!playMove(move.row, move.col, { silent: true, localOnly: true })) {
          throw new Error(`Could not replay SGF move ${move.move}`);
        }
      }
    }

    state.mode = "local";
    modeSelect.value = "local";
    state.modeTitle = name.replace(/\.sgf$/i, "") || "Loaded SGF";
    setMessage(`Loaded ${name}`);
    syncUi();
    draw();
  } catch (error) {
    restoreState(snapshot);
    throw error;
  }
}

function snapshotState() {
  return {
    size: state.size,
    board: cloneBoard(state.board),
    turn: state.turn,
    captures: { ...state.captures },
    history: state.history.map((entry) => ({ ...entry })),
    undoStack: state.undoStack.map((entry) => ({
      board: cloneBoard(entry.board),
      turn: entry.turn,
      captures: { ...entry.captures },
    })),
    moves: state.moves.map((move) => ({ ...move })),
    hover: state.hover ? { ...state.hover } : null,
    moveCount: state.moveCount,
    monitoring: state.monitoring,
    apiMode: state.apiMode,
    apiBusy: state.apiBusy,
    human: state.human,
    monitorUpdatedAt: state.monitorUpdatedAt,
    monitorWorker: state.monitorWorker,
    mode: state.mode,
    modeTitle: state.modeTitle,
  };
}

function restoreState(snapshot) {
  Object.assign(state, snapshot);
  state.board = cloneBoard(snapshot.board);
  state.captures = { ...snapshot.captures };
  state.history = snapshot.history.map((entry) => ({ ...entry }));
  state.undoStack = snapshot.undoStack.map((entry) => ({
    board: cloneBoard(entry.board),
    turn: entry.turn,
    captures: { ...entry.captures },
  }));
  state.moves = snapshot.moves.map((move) => ({ ...move }));
  state.hover = snapshot.hover ? { ...snapshot.hover } : null;
  boardSizeSelect.value = `${state.size}`;
  modeSelect.value = state.mode;
  syncUi();
  draw();
}

function parseSgf(content) {
  const sizeMatch = /SZ\s*\[\s*(\d+)\s*\]/i.exec(content);
  const size = sizeMatch ? Number(sizeMatch[1]) : 19;
  if (![9, 13, 19].includes(size)) {
    throw new Error(`Unsupported SGF board size: ${size}`);
  }

  const moves = [];
  const movePattern = /;([BW])\[((?:\\.|[^\]])*)\]/gi;
  let match = movePattern.exec(content);
  while (match) {
    const player = match[1].toUpperCase() === "W" ? "white" : "black";
    const rawPoint = unescapeSgfValue(match[2]).trim();
    if (rawPoint === "" || rawPoint.length < 2) {
      moves.push({ player, pass: true });
    } else {
      const col = rawPoint.charCodeAt(0) - 97;
      const row = rawPoint.charCodeAt(1) - 97;
      if (!inBoundsForSize(row, col, size)) {
        throw new Error(`SGF point is outside ${size}x${size}: ${rawPoint}`);
      }
      moves.push({ player, row, col, move: pointNameForSize(row, col, size), pass: false });
    }
    match = movePattern.exec(content);
  }

  return { size, moves };
}

function unescapeSgfValue(value) {
  return value.replace(/\\([\s\S])/g, "$1");
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
startModeButton.addEventListener("click", startSelectedMode);
stopModeButton.addEventListener("click", stopActiveMode);
loadSgfButton.addEventListener("click", promptLoadSgf);
loadSgfInput.addEventListener("change", loadSgfFile);
saveSgfButton.addEventListener("click", saveSgf);
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
loadPlayConfig().finally(() => refreshControls());
