# LiteLLM -> llm-d flow control POC

llm-d's flow control admits requests by reading two headers at the EPP:

| header | meaning |
| --- | --- |
| `x-llm-d-inference-objective` | the `InferenceObjective` (priority band) to schedule under |
| `x-llm-d-inference-fairness-id` | the flow id used for fair sharing *within* that band |

In LiteLLM, a pre-call hook:
1. **strips** `x-llm-d-*` flow control headers off whatever the client sent, and
2. **injects** them from a static mapping of LiteLLM identity -> objective.

## Layout

```
run_poc.sh            starts the mock gateway + proxy, asserts on headers
config.yaml           proxy config, wired to the mock + static auth
objectives.yaml       the mapping: key alias / user / team -> objective
hooks/flow_control.py the pre-call hook (strip + inject)
mock_llmd.py          stand-in gateway; echoes back the x-llm-d-* it received
static_auth.py        POC-only custom_auth, so no Postgres is needed
poc_keys.yaml         the two fake keys -> identity (never real credentials)
```

## Run it

Needs `uv` and `jq`.

```bash
./run_poc.sh
```

First run builds `./.venv` (~1 min); after that it takes a few seconds.

```
     case                       objective          fairness-id
---- -------------------------- ------------------ ------------------
PASS key rule: forge            premium            forge
PASS key rule: lightwell        standard           lightwell
PASS spoof stripped             standard           lightwell

all cases passed
```

Each row is a real request through the proxy; the objective and fairness id
shown are what the **gateway received**, read back out of the mock's response --
not what the proxy logged it intended to send.

The last row is the one that matters: that request went out with

```
-H "x-llm-d-inference-objective: premium"
-H "x-llm-d-inference-fairness-id: forge"
```

on a key mapped to `standard`, and the gateway still saw `standard` / `lightwell`.
`config.yaml` deliberately sets `forward_client_headers_to_llm_api: true` so
this is tested the hard way -- with that on and the hook removed, the mock sees
the spoofed values verbatim.

## The mapping

`objectives.yaml`, most specific first: **key alias -> user -> team -> default**.
The POC exercises only the `keys` tier, with one key per band:

```yaml
defaults:
  objective: standard
  fairness_id_from: key_alias

keys:
  forge:
    objective: premium
  lightwell:
    objective: standard
```

`users:` and `teams:` sections slot in the same way, keyed by LiteLLM `user_id`
and `team_id`, and are consulted in that order when no key rule matches.

Two knobs on the fairness id, which decides who competes with whom inside a
band:

- `fairness_id_from` picks which identity field becomes the id
  (`key_alias` | `user_id` | `team_id` | `end_user_id` | `api_key`). Per-key is
  the default: one noisy key cannot crowd out the rest of its own team.
- `fairness_id` pins an explicit shared id, so splitting a job across keys does
  not multiply its share.

`models:` at the top of the file scopes injection to the llm-d-backed model
groups, so anything else behind the same proxy is left alone.

The file is re-read when its mtime changes, so a remap takes effect without
restarting the proxy. A malformed edit is logged and the last good mapping is
kept, rather than failing every request.

## What a real deployment would change

Only the two edges. The hook and the mapping are unchanged:

- **Identity.** `static_auth.py` + `poc_keys.yaml` stand in for LiteLLM virtual
  keys, so the POC needs no Postgres. Real keys populate the same
  `key_alias` / `user_id` / `team_id` fields the hook reads, created with the
  aliases `objectives.yaml` expects:
  `litellm keys create --key-alias forge --team-id team-platform`.
- **Upstream.** `api_base` points at `mock_llmd.py` instead of the real
  inference gateway, and `forward_client_headers_to_llm_api` would go back to
  `false` -- the hook strips the two flow control headers either way, but with
  forwarding off no other `x-*` header a client invents reaches the EPP.

## Known gaps

- **The objectives must exist.** `objective:` values have to match an
  `InferenceObjective` in the model's namespace; the proxy does not validate
  them, so a typo is only visible at the EPP.
- **Pass-through routes bypass the hook.** It runs on LiteLLM's common request
  path (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`,
  `/v1/responses`). Anything routed through a LiteLLM pass-through endpoint
  reaches the gateway unfiltered, so don't expose one for the llm-d backends.
- **The proxy has to be the only route to the gateway.** Everything here is
  worthless if a caller can reach the inference gateway directly -- that still
  needs a NetworkPolicy or gateway-level auth.
- **Static mapping only.** No per-key overrides at request time, no time-of-day
  or queue-depth-aware demotion, no spend-based downgrade. The hook is the place
  those would go; the mapping lookup is one function.
