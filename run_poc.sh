#!/usr/bin/env bash
# End-to-end POC: start the mock llm-d gateway + a LiteLLM proxy with the flow
# control hook, then drive it with both API keys and assert on the headers
# the gateway actually received.
#
#   ./run_poc.sh                    # creates ./.venv on first run
#   VENV=/path/to/venv ./run_poc.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$ROOT/.venv}"
LOGDIR="${LOGDIR:-$ROOT/.poc-logs}"
PROXY_PORT="${PROXY_PORT:-4000}"
MOCK_PORT=8080  # also hardcoded in config.yaml + mock_llmd.py

mkdir -p "$LOGDIR"

# Refuse to run against someone else's proxy: a leftover process on either port
# would answer the requests below and the results would mean nothing.
for port in "$PROXY_PORT" "$MOCK_PORT"; do
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "port $port is already in use -- stop whatever is listening there first" >&2
    exit 1
  fi
done

if [[ ! -x "$VENV/bin/litellm" ]]; then
  echo "==> creating venv at $VENV"
  uv venv "$VENV"
  VIRTUAL_ENV="$VENV" uv pip install 'litellm[proxy]' fastapi uvicorn
fi

pids=()
cleanup() {
  local pid alive
  for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  # Don't return while anything is still holding a port: the next run checks
  # those ports and would refuse to start.
  for _ in $(seq 40); do
    alive=0
    for pid in "${pids[@]:-}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
    [[ $alive -eq 0 ]] && return
    sleep 0.25
  done
  for pid in "${pids[@]:-}"; do kill -9 "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT

echo "==> starting mock llm-d gateway on :$MOCK_PORT"
"$VENV/bin/python" "$ROOT/mock_llmd.py" >"$LOGDIR/mock.log" 2>&1 &
pids+=($!)
disown %% 2>/dev/null || true

echo "==> starting litellm proxy on :$PROXY_PORT"
(cd "$ROOT" && exec "$VENV/bin/litellm" --config "$ROOT/config.yaml" --port "$PROXY_PORT") \
  >"$LOGDIR/proxy.log" 2>&1 &
pids+=($!)
disown %% 2>/dev/null || true

for _ in $(seq 60); do
  if curl -fsS "http://127.0.0.1:$PROXY_PORT/health/readiness" >/dev/null 2>&1; then break; fi
  sleep 1
done
if ! curl -fsS "http://127.0.0.1:$PROXY_PORT/health/readiness" >/dev/null 2>&1; then
  echo "proxy did not come up; see $LOGDIR/proxy.log" >&2
  tail -40 "$LOGDIR/proxy.log" >&2
  exit 1
fi

failures=0

run_case() {
  local name=$1 key=$2 want_objective=$3 want_fairness=$4
  shift 4

  local response headers objective fairness status
  response=$(curl -sS -X POST "http://127.0.0.1:$PROXY_PORT/v1/chat/completions" \
    -H "Authorization: Bearer $key" \
    -H "Content-Type: application/json" \
    "$@" \
    -d '{"model":"gemma-4","messages":[{"role":"user","content":"hi"}]}')

  headers=$(jq -r '.choices[0].message.content // "{}"' <<<"$response" 2>/dev/null || echo '{}')
  objective=$(jq -r '."x-llm-d-inference-objective" // "(none)"' <<<"$headers")
  fairness=$(jq -r '."x-llm-d-inference-fairness-id" // "(none)"' <<<"$headers")

  if [[ "$objective" == "$want_objective" && "$fairness" == "$want_fairness" ]]; then
    status="PASS"
  else
    status="FAIL"
    failures=$((failures + 1))
  fi

  printf '%-4s %-26s %-18s %-18s' "$status" "$name" "$objective" "$fairness"
  if [[ "$status" == "FAIL" ]]; then
    printf '  (wanted %s / %s)' "$want_objective" "$want_fairness"
  fi
  printf '\n'
}

echo
printf '%-4s %-26s %-18s %-18s\n' "" "case" "objective" "fairness-id"
printf '%-4s %-26s %-18s %-18s\n' "----" "--------------------------" "------------------" "------------------"

# key rule: forge is mapped up into the premium band
run_case "key rule: forge" forge-key premium forge

# key rule: lightwell is mapped to the standard band
run_case "key rule: lightwell" lightwell-key standard lightwell

# a standard key claiming the premium band is stripped and re-stamped
run_case "spoof stripped" lightwell-key standard lightwell \
  -H "x-llm-d-inference-objective: premium" \
  -H "x-llm-d-inference-fairness-id: forge"

echo
if [[ $failures -eq 0 ]]; then
  echo "all cases passed"
else
  echo "$failures case(s) failed -- see $LOGDIR/proxy.log"
fi
echo "proxy log: $LOGDIR/proxy.log"
exit $failures
