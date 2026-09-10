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
  local name=$1 key=$2 model=$3 want_objective=$4 want_fairness=$5
  shift 5

  local response headers objective fairness status
  response=$(curl -sS -X POST "http://127.0.0.1:$PROXY_PORT/v1/chat/completions" \
    -H "Authorization: Bearer $key" \
    -H "Content-Type: application/json" \
    "$@" \
    -d '{"model":"'"$model"'","messages":[{"role":"user","content":"hi"}]}')

  headers=$(jq -r '.choices[0].message.content // "{}"' <<<"$response" 2>/dev/null || echo '{}')
  objective=$(jq -r '."x-llm-d-inference-objective" // "(none)"' <<<"$headers")
  fairness=$(jq -r '."x-llm-d-inference-fairness-id" // "(none)"' <<<"$headers")

  if [[ "$objective" == "$want_objective" && "$fairness" == "$want_fairness" ]]; then
    status="PASS"
  else
    status="FAIL"
    failures=$((failures + 1))
  fi

  printf '%-4s %-28s %-12s %-14s %-12s' "$status" "$name" "$model" "$objective" "$fairness"
  if [[ "$status" == "FAIL" ]]; then
    printf '  (wanted %s / %s)' "$want_objective" "$want_fairness"
  fi
  printf '\n'
}

echo
printf '%-4s %-28s %-12s %-14s %-12s\n' "" "case" "model" "objective" "fairness-id"
printf '%-4s %-28s %-12s %-14s %-12s\n' "----" "----------------------------" "------------" "--------------" "------------"

# key -> policy `default`: forge runs in the premium band
run_case "key policy: forge" forge-key gemma-4 premium forge

# key -> policy `async`: lightwell runs in the standard band
run_case "key policy: lightwell" lightwell-key gemma-4 standard lightwell

# a standard key claiming the premium band is stripped and re-stamped
run_case "spoof stripped" lightwell-key gemma-4 standard lightwell \
  -H "x-llm-d-inference-objective: premium" \
  -H "x-llm-d-inference-fairness-id: forge"

# no rule anywhere: drift picks up defaults.policy (`default` -> premium)
run_case "default policy: drift" drift-key gemma-4 premium drift

# token x model override: atlas is `default` (premium), except on the batch
# model where its `models:` map switches it to `async` (standard)
run_case "key policy: atlas" atlas-key gemma-4 premium atlas
run_case "token x model: atlas" atlas-key gemma-4-mini standard atlas

# team policy + team x model override: nomad has no key rule; its team
# (team-photon) is `async`, and drops to best-effort on the batch model
run_case "team policy: nomad" nomad-key gemma-4 standard nomad
run_case "team x model: nomad" nomad-key gemma-4-mini best-effort nomad

# team key_defaults: neither nomad nor ember carries a limit of its own --
# team-photon's `key_defaults: {tpm_limit: 1000}` gives each key its OWN
# per-key counter (unlike a shared team_tpm_limit). The two requests above
# put nomad at 4 of 1000 tokens, so this one crosses the async policy's
# demote_at (0.004) and degrades to best-effort -- while ember, same team,
# same defaults, fresh counter, still runs standard.
run_case "team default: demoted" nomad-key gemma-4 best-effort nomad
run_case "team default: per key" ember-key gemma-4 standard ember

# saturation demotion: meridian's key has a real LiteLLM tpm_limit (1000) and
# the hook reads the v3 rate limiter's own counters. The mock bills 2 tokens
# per request, so saturation runs 0.0000, 0.0020, 0.0040 -- and the third
# request crosses the `default` policy's demote_at: 0.004 and goes out at the
# policy's demote_to (standard), still admitted. All three land inside one 60s
# limiter window (requests take well under that).
run_case "saturation: 1st premium" meridian-key gemma-4 premium meridian
run_case "saturation: 2nd premium" meridian-key gemma-4 premium meridian
run_case "saturation: demoted" meridian-key gemma-4 standard meridian

# per-model enforcement: harbor has a key-wide tpm_limit AND a per-model
# model_tpm_limit on gemma-4, and its policy demotes on the per-model counter
# only (demote_on: [model_per_key]). Three gemma-4 requests saturate that
# counter (third demotes); by then the key-wide counter is past demote_at too
# (6 of 1000 tokens), but gemma-4-mini has no per-model limit, so it stays
# premium -- demotion on one model does not bleed into the other.
run_case "per-model: 1st premium" harbor-key gemma-4 premium harbor
run_case "per-model: 2nd premium" harbor-key gemma-4 premium harbor
run_case "per-model: demoted" harbor-key gemma-4 standard harbor
run_case "per-model: mini untouched" harbor-key gemma-4-mini premium harbor

echo
if [[ $failures -eq 0 ]]; then
  echo "all cases passed"
else
  echo "$failures case(s) failed -- see $LOGDIR/proxy.log"
fi
echo "proxy log: $LOGDIR/proxy.log"
exit $failures
