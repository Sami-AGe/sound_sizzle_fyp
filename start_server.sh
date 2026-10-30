#!/bin/bash
# AudioStory Server Starter Script
# Usage: ./start_server.sh          (starts in foreground)
#        ./start_server.sh -d        (starts in background, detached)
#        ./start_server.sh -k        (kills existing server on port 8001 first, then starts)

set -euo pipefail

# ─── Configuration ──────────────────────────────────────────────────────────
PROJECT_ROOT="/Users/syedsunny/sound_sizzle_fyp"
BACKEND_DIR="${PROJECT_ROOT}/backend"
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python3"
PORT=8001
HOST="127.0.0.1"

# ─── Kill existing server if requested ──────────────────────────────────────
if [[ "${1:-}" == "-k" ]]; then
    echo "Killing any existing server on port ${PORT}..."
    PID=$(lsof -ti:"${PORT}" 2>/dev/null || true)
    if [[ -n "${PID}" ]]; then
        kill -9 "${PID}" 2>/dev/null || true
        echo "  → Killed PID ${PID}"
    fi
    sleep 1
fi

# ─── Change to backend directory ────────────────────────────────────────────
cd "${BACKEND_DIR}"

# ─── Load environment variables from .env ───────────────────────────────────
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
fi

# ─── Verify virtual environment ─────────────────────────────────────────────
if [ ! -f "${VENV_PYTHON}" ]; then
    echo "ERROR: Virtual environment not found at ${VENV_PYTHON}"
    echo "       Create it with: python3 -m venv ${PROJECT_ROOT}/.venv"
    echo "       Then install deps: ${VENV_PYTHON} -m pip install -r ${BACKEND_DIR}/requirements.txt"
    exit 1
fi

# ─── Display startup info ───────────────────────────────────────────────────
echo "─────────────────────────────────────────────────"
echo " AudioStory Server"
echo "─────────────────────────────────────────────────"
echo "  Port:        ${HOST}:${PORT}"
echo "  Backend:     ${BACKEND_DIR}"
echo "  Python:      ${VENV_PYTHON}"
echo "  FRAME_DELAY: ${FRAME_DELAY:-8.0 (default)}"
echo "  MAX_FRAMES:  ${MAX_FRAMES:-30 (default)}"
echo "  MODE:        ${GENERATION_MODE:-fast (default)}"
echo "  COMFYUI:     ${COMFYUI_HOST:-not set}"
echo "─────────────────────────────────────────────────"

# ─── Start server ───────────────────────────────────────────────────────────
if [[ "${2:-${1:-}}" == "-d" ]]; then
    echo "Starting in background (detached mode)..."
    nohup "${VENV_PYTHON}" -m uvicorn main:app --reload --host "${HOST}" --port "${PORT}" \
        > "${PROJECT_ROOT}/server.log" 2>&1 &
    echo "  → PID: $!"
    echo "  → Log: ${PROJECT_ROOT}/server.log"
    echo "  → API: http://localhost:${PORT}"
    echo "  → Docs: http://localhost:${PORT}/docs"
    echo "  → Frontend: http://localhost:${PORT}/app/"
else
    # Check if port is already in use
    PID=$(lsof -ti:"${PORT}" 2>/dev/null || true)
    if [[ -n "${PID}" ]]; then
        echo "WARNING: Port ${PORT} is already in use by PID ${PID}"
        echo "         Use './start_server.sh -k' to kill it first, or './start_server.sh -k -d' to restart in background"
        exit 1
    fi

    echo "Starting server (Ctrl+C to stop)..."
    "${VENV_PYTHON}" -m uvicorn main:app --reload --host "${HOST}" --port "${PORT}"
fi
