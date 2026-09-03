# LiteLLM -> llm-d flow control POC

llm-d's flow control admits requests by reading two headers at the EPP:

| header | meaning |
| --- | --- |
| `x-llm-d-inference-objective` | the `InferenceObjective` (priority band) to schedule under |
| `x-llm-d-inference-fairness-id` | the flow id used for fair sharing *within* that band |

In LiteLLM, a pre-call hook:
1. **strips** `x-llm-d-*` flow control headers off whatever the client sent,
2. **injects** them from a mapping of LiteLLM identity -> objective, and
3. **demotes** callers who have spent their soft token budget to a lower band
   (see [Soft limits](#soft-limits-demote-dont-429)) -- traffic keeps flowing,
   just at lower priority, instead of a 429.

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
PASS soft limit: 1st premium    premium            meridian
PASS soft limit: 2nd premium    premium            meridian
PASS soft limit: demoted        best-effort        meridian

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

## Soft limits: demote, don't 429

A rule can grant a token budget per window and name the band to fall back to
when it is spent:

```yaml
teams:
  team-a:
    objective: premium
    limits:
      tokens_per_minute: 20000
      tokens_per_hour: 500000
      demote_to: best-effort      # optional; defaults to "best-effort"
```

These are deliberately **not** LiteLLM's `tpm_limit`/`rpm_limit`: those are
hard limits enforced by the built-in parallel request limiter, and exceeding
them rejects the request with a 429. Here the request is always admitted --
what changes is the priority band it is admitted *under*. Inside the budget
the caller runs at its configured objective; once a window's tokens are spent,
requests are stamped with `demote_to` instead, until the window rolls over.
Backpressure then comes from llm-d's own flow control, which is where it
belongs: the gateway knows its queue depth, the proxy doesn't.

Mechanics, all inside the same hook:

- After each response, `async_post_call_success_hook` charges the response's
  actual `usage.total_tokens` against one counter per configured window.
- Counters are keyed by the **matched rule** (`teams:team-a`), not the caller,
  so everyone sharing a rule draws from one budget -- a group grant. Fairness
  ids are unaffected: inside whatever band the group lands in, per-key fair
  sharing still applies.
- Counters live in the proxy's `DualCache` (the same cache LiteLLM's own rate
  limiter uses): in-memory for a single worker, shared via Redis when the
  proxy is configured with one.
- Windows are fixed wall-clock windows (minute / hour), the same scheme as
  LiteLLM's v1 limiter. TTLs garbage-collect old windows.

The POC's `meridian` key runs this live: `tokens_per_minute: 3` against a mock
that bills 2 tokens per request, so the third request in a minute goes out
demoted -- the gateway itself saw `best-effort` -- and the first request of the
next minute is premium again.

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
- **Soft-limit accounting is post-paid.** Usage is charged after the response,
  so a burst of concurrent requests can overshoot a window by whatever was in
  flight when it filled. Fine for tiering; this is not a billing meter. For a
  pre-paid variant, reserve an estimate in the pre-call hook and reconcile
  after, which is what LiteLLM's own v3 limiter does.
- **Streaming responses are not counted.** `async_post_call_success_hook` only
  sees non-streaming responses. A production version would also charge usage
  from the logging callback's completed-stream event
  (`async_log_success_event`), which carries final usage for streams.
- **Demotion needs the demoted objective to exist too.** `demote_to` values
  are `InferenceObjective` names like any other; the EPP decides what an
  unresolvable one means.
- **Still no queue-depth awareness.** Demotion here is quota-driven. Time-of-day
  or saturation-aware logic slots into the same `_spent_window` decision point;
  the mapping lookup is one function.
