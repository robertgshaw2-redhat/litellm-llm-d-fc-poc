# LiteLLM -> llm-d flow control POC

llm-d's flow control admits requests by reading two headers at the EPP:

| header | meaning |
| --- | --- |
| `x-llm-d-inference-objective` | the `InferenceObjective` (priority band) to schedule under |
| `x-llm-d-inference-fairness-id` | the flow id used for fair sharing *within* that band |

In LiteLLM, a pre-call hook:
1. **strips** `x-llm-d-*` flow control headers off whatever the client sent,
2. **injects** them from a set of named **policies** (objective + demotion
   rule) assigned per LiteLLM identity -- key, user or team -- with per-model
   overrides for any of them, and
3. **demotes** callers nearing their LiteLLM rate limits to the policy's
   fallback band
   (see [Saturation demotion](#saturation-demotion-reusing-litellms-rate-limit-counters))
   -- traffic keeps flowing, just at lower priority, instead of a 429.

## Layout

```
run_poc.sh            starts the mock gateway + proxy, asserts on headers
config.yaml           proxy config, wired to the mock + static auth
objectives.yaml       the policies + the mapping: key / user / team -> policy
hooks/flow_control.py the pre-call hook (strip + inject)
mock_llmd.py          stand-in gateway; echoes back the x-llm-d-* it received
static_auth.py        POC-only custom_auth, so no Postgres is needed
poc_keys.yaml         the fake keys -> identity (never real credentials)
```

## Run it

Needs `uv` and `jq`.

```bash
./run_poc.sh
```

First run builds `./.venv` (~1 min); after that it takes a few seconds.

```
     case                         model        objective      fairness-id
---- ---------------------------- ------------ -------------- ------------
PASS key policy: forge            gemma-4      premium        forge
PASS key policy: lightwell        gemma-4      standard       lightwell
PASS spoof stripped               gemma-4      standard       lightwell
PASS default policy: drift        gemma-4      premium        drift
PASS key policy: atlas            gemma-4      premium        atlas
PASS token x model: atlas         gemma-4-mini standard       atlas
PASS team policy: nomad           gemma-4      standard       nomad
PASS team x model: nomad          gemma-4-mini best-effort    nomad
PASS saturation: 1st premium      gemma-4      premium        meridian
PASS saturation: 2nd premium      gemma-4      premium        meridian
PASS saturation: demoted          gemma-4      standard       meridian
PASS per-model: 1st premium       gemma-4      premium        harbor
PASS per-model: 2nd premium       gemma-4      premium        harbor
PASS per-model: demoted           gemma-4      standard       harbor
PASS per-model: mini untouched    gemma-4-mini premium        harbor

all cases passed
```

Each row is a real request through the proxy; the objective and fairness id
shown are what the **gateway received**, read back out of the mock's response --
not what the proxy logged it intended to send.

The `spoof stripped` row is the one that matters: that request went out with

```
-H "x-llm-d-inference-objective: premium"
-H "x-llm-d-inference-fairness-id: forge"
```

on a key mapped to `standard`, and the gateway still saw `standard` / `lightwell`.
`config.yaml` deliberately sets `forward_client_headers_to_llm_api: true` so
this is tested the hard way -- with that on and the hook removed, the mock sees
the spoofed values verbatim.

## Policies and the mapping

`objectives.yaml` defines named **policies** -- an objective plus an optional
demotion rule -- and assigns one per identity:

```yaml
policies:
  default:                  # interactive: premium, degrade to standard near the limit
    objective: premium
    demote_at: 0.8
    demote_to: standard
  async:                    # batch: standard, degrade to best-effort near the limit
    objective: standard
    demote_at: 0.5
    demote_to: best-effort

defaults:
  policy: default
  fairness_id_from: key_alias

keys:
  forge:
    policy: default
  lightwell:
    policy: async
    models:                 # token x model override: policy name or inline mapping
      "gemma-4-mini": best-effort

teams:
  team-photon:
    policy: async
    models:                 # team x model override, same shape
      "gemma-4-mini": best-effort
```

Resolution is most specific first: **key alias -> user -> team -> default**
(the first tier with a rule wins; `users:` is keyed by LiteLLM `user_id`,
`teams:` by `team_id`). Within the matched tier, a `models:` override beats
the tier's base policy -- so the same token runs `default` on its interactive
model and `async` on its batch model. `models:` patterns are fnmatch globs,
first match in declaration order wins.

Everything layers field by field: a rule's policy is laid over the defaults'
policy, and a model override over that, so an override that only changes
`objective` inherits the rule's demotion settings. A rule may reference a
named policy (`policy: async`), set the fields inline, or both (inline wins) --
the pre-policy syntax `objective:` / `demote_at:` directly on a rule still
works unchanged.

Two knobs on the fairness id, which decides who competes with whom inside a
band:

- `fairness_id_from` picks which identity field becomes the id
  (`key_alias` | `user_id` | `team_id` | `end_user_id` | `api_key`). Per-key is
  the default: one noisy key cannot crowd out the rest of its own team.
- `fairness_id` pins an explicit shared id, so splitting a job across keys does
  not multiply its share.

`models:` at the top of the file scopes injection to the llm-d-backed model
groups, so anything else behind the same proxy is left alone.

## Saturation demotion: reusing LiteLLM's rate limit counters

The hook keeps **no counters of its own**. The limits are LiteLLM's ordinary
`tpm_limit` / `rpm_limit` on the key, user or team, and the numbers come from
the v3 parallel request limiter's own tracking -- the same counters it mirrors
into the `x-ratelimit-{descriptor}-{remaining,limit}-{requests,tokens}`
response headers. A policy only adds the demotion rule:

```yaml
policies:
  default:
    objective: premium
    demote_at: 0.004            # fraction of the caller's most-saturated limit
    demote_to: standard         # optional; defaults to "best-effort"
```

Below `demote_at` the caller runs at its configured objective. Past it,
requests are stamped `demote_to` instead -- still admitted, at lower priority,
with backpressure left to llm-d's flow control, which is where it belongs: the
gateway knows its queue depth, the proxy doesn't. The limiter's own hard 429
at 100% remains the backstop, so `demote_at` carves a soft band underneath it
(realistic values are 0.5-0.9; the POC uses 0.004 = 4 of 1000 tokens so three
requests cross it). The computed saturation is also forwarded as
`x-litellm-ratelimit-saturation`, so downstream logic can make its own calls.

### Per-model enforcement

LiteLLM's limits can themselves be scoped per model -- `model_tpm_limit` /
`model_rpm_limit` in key metadata (or team metadata for the team variant) give
each key x model pair its own counter, which the v3 limiter tracks as a
`model_per_key` / `model_per_team` descriptor. By default the hook demotes on
the caller's *most saturated* limit, whichever that is; `demote_on` narrows
the signal to specific descriptor kinds:

```yaml
policies:
  per-model-default:
    objective: premium
    demote_at: 0.8
    demote_to: standard
    demote_on: [model_per_key]   # only the key's per-model counters count

keys:
  harbor:
    policy: per-model-default
```

with the limits on the key itself (real LiteLLM, nothing POC-specific):

```yaml
harbor-key:
  tpm_limit: 1000                # key-wide backstop
  metadata:
    model_tpm_limit:
      gemma-4: 1000              # the counter demotion is driven by
```

The `per-model:` rows in the POC output show the effect: three requests
saturate harbor's `gemma-4` counter and the third is demoted, while the next
request to `gemma-4-mini` -- same key, same window, key-wide counter already
past `demote_at` -- still goes out `premium`, because only the per-model
counters feed the decision. Valid `demote_on` values are the limiter's
descriptor keys: `api_key`, `user`, `team`, `team_member`, `end_user`,
`model_per_key`, `model_per_team`.

How the read works, cheapest path first:

- If the limiter's pre-call already ran for this request, its
  `RateLimitResponse` sits on a ContextVar stash
  (`get_request_stash_for_call`) -- read back with zero cache traffic.
- Otherwise the hook builds the same descriptors the limiter would
  (`_create_rate_limit_descriptors`) and calls
  `should_rate_limit(read_only=True)`: the in-memory tier of the limiter's
  internal `DualCache` answers first (every enforced request writes Redis
  results back into it), Redis fills the gaps. Slightly stale cross-instance,
  which is fine for a priority decision.

In practice the second path is the one taken -- config callbacks register
before the built-in proxy hooks, so this hook runs first -- and that ordering
is the *better* one: it reads **settled** usage only. The limiter's own
numbers (and the `x-ratelimit-*` headers) include the current request's
in-flight token reservation (~hundreds of tokens even for a one-word prompt),
which would make the saturation signal jumpy.

Every failure mode of the read degrades to "no saturation signal, keep the
configured objective" -- never to a failed request.

The file is re-read when its mtime changes, so a remap takes effect without
restarting the proxy. A malformed edit is logged and the last good mapping is
kept, rather than failing every request.

## What a real deployment would change

Only the two edges. The hook and the mapping are unchanged:

- **Identity.** `static_auth.py` + `poc_keys.yaml` stand in for LiteLLM virtual
  keys, so the POC needs no Postgres. Real keys populate the same
  `key_alias` / `user_id` / `team_id` fields the hook reads, created with the
  aliases `objectives.yaml` expects:
  `litellm keys create --key-alias forge --team-id team-platform`. The limits
  are ordinary key/team settings too, per-model ones included
  (`--tpm-limit`, or `model_tpm_limit` in key/team metadata).
- **Upstream.** `api_base` points at `mock_llmd.py` instead of the real
  inference gateway, and `forward_client_headers_to_llm_api` would go back to
  `false` -- the hook strips the two flow control headers either way, but with
  forwarding off no other `x-*` header a client invents reaches the EPP.

## Known gaps

- **The objectives must exist.** `objective:` values have to match an
  `InferenceObjective` in the model's namespace; the proxy does not validate
  them, so a typo is only visible at the EPP.
- **A typo'd policy name only warns.** A `policy:` reference that doesn't
  match anything in `policies:` is logged and contributes nothing -- the
  caller runs on whatever the `defaults:` layer provides.
- **Pass-through routes bypass the hook.** It runs on LiteLLM's common request
  path (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`,
  `/v1/responses`). Anything routed through a LiteLLM pass-through endpoint
  reaches the gateway unfiltered, so don't expose one for the llm-d backends.
- **The proxy has to be the only route to the gateway.** Everything here is
  worthless if a caller can reach the inference gateway directly -- that still
  needs a NetworkPolicy or gateway-level auth.
- **The saturation read leans on private LiteLLM APIs.** `should_rate_limit`,
  `_create_rate_limit_descriptors` and the request stash are internals of the
  v3 limiter, verified against litellm 1.100.1. An upgrade can break the read;
  when it does, the hook logs a warning and keeps injecting the configured
  objective -- demotion silently stops until the call sites are re-checked.
- **Demotion needs limits to exist.** No `tpm_limit`/`rpm_limit` (or their
  per-model `model_tpm_limit`/`model_rpm_limit` variants) on the key, user or
  team means no descriptors, no counters, no saturation signal -- and a
  `demote_on` scope that matches none of the caller's limits is the same as
  no signal. Because the v3 limiter still enforces the hard 429 at 100%,
  `demote_at` must be below 1.0 to buy any soft band at all.
- **Saturation is per LiteLLM identity, not per rule.** The counters belong to
  the key/user/team (or key x model / team x model) that carries the limit. A
  team-wide grant is a `team_tpm_limit`; a `demote_at` on a `teams:` rule then
  demotes members off the shared team counter. `demote_on` picks *which* of
  the caller's counters feed the decision; it cannot conjure counters the
  caller's limits don't create.
- **Demotion needs the demoted objective to exist too.** `demote_to` values
  are `InferenceObjective` names like any other; the EPP decides what an
  unresolvable one means.
- **Still no queue-depth awareness on the proxy side.** Demotion is
  quota-driven; llm-d's flow control supplies the load-aware part inside each
  band. Anything fancier (time-of-day, spend-based) slots into the same
  decision point in `async_pre_call_hook`.
