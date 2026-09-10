"""LiteLLM pre-call hook: map the caller's identity to llm-d flow control headers.

llm-d's flow control layer reads two headers on the way into the EPP:

    x-llm-d-inference-objective    the InferenceObjective (priority band) to admit under
    x-llm-d-inference-fairness-id  the flow id used for fair sharing inside that band

Neither can be trusted from the client -- anything that reaches the gateway could
otherwise promote itself into the latency-critical band and starve everyone else.
So this hook does two things on every request:

    1. strips both headers off whatever the client sent
    2. re-injects them from a static mapping of LiteLLM identity -> policy

The mapping (objectives.yaml) defines named *policies* -- an objective plus an
optional saturation-demotion rule -- and assigns one to each key, user or team.
An assignment may be overridden per model, so a token or team runs one policy on
its interactive model and another on its batch model. Precedence is
most-specific-first: key, then user, then team, then the default; within the
matched tier a `models:` override beats the tier's base policy.

A policy's `demote_at` is a saturation fraction of the caller's normal LiteLLM
rate limits (tpm_limit / rpm_limit on the key, user or team, including the
per-model `model_tpm_limit` / `model_rpm_limit` variants). The hook reads the
v3 parallel request limiter's own counters -- the same numbers it reports as
x-ratelimit-* response headers -- and once the most-saturated limit crosses
`demote_at`, requests go out stamped with a lower objective (`demote_to`,
default "best-effort") instead of being rejected. `demote_on` narrows which
limits count (e.g. only the per-model ones), so demotion on one model does not
bleed into another. The hard 429 at 100% stays, as the limiter still enforces
it; `demote_at` carves a soft band underneath it. No counters of our own: with
Redis configured on the proxy, saturation is cluster-wide for free.

A team rule may also carry `key_defaults:` -- LiteLLM limits (tpm_limit,
rpm_limit, max_parallel_requests, model_tpm_limit, model_rpm_limit) applied to
any member key that doesn't set its own, so every key on a team gets its own
per-key limit without per-key configuration. The v3 limiter enforces and
tracks them as if the key row carried them.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Any

import yaml
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth

OBJECTIVE_HEADER = "x-llm-d-inference-objective"
FAIRNESS_HEADER = "x-llm-d-inference-fairness-id"
SATURATION_HEADER = "x-litellm-ratelimit-saturation"
FLOW_CONTROL_HEADERS = (OBJECTIVE_HEADER, FAIRNESS_HEADER, SATURATION_HEADER)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "objectives.yaml"

# Fields on the authenticated key that a fairness id may be derived from.
# `api_key` is the salted hash of the virtual key, not the key itself.
IDENTITY_FIELDS = ("key_alias", "user_id", "team_id", "end_user_id", "api_key")

DEFAULT_DEMOTE_OBJECTIVE = "best-effort"

# The knobs a policy may set. A rule (or `defaults:`) may reference a named
# policy via `policy:` and/or set these inline; inline wins field by field.
POLICY_FIELDS = (
    "objective",
    "demote_at",
    "demote_to",
    "demote_on",
    "fairness_id",
    "fairness_id_from",
)

# LiteLLM limits a team rule's `key_defaults:` may supply for member keys that
# don't carry their own. The first three are UserAPIKeyAuth fields, the last
# two live in key metadata (that's where the v3 limiter reads them from).
KEY_DEFAULT_FIELDS = ("tpm_limit", "rpm_limit", "max_parallel_requests")
KEY_DEFAULT_METADATA_FIELDS = ("model_tpm_limit", "model_rpm_limit")


class FlowControlHook(CustomLogger):
    def __init__(self, config_path: str | Path | None = None) -> None:
        super().__init__()
        self.config_path = Path(
            os.getenv("LLMD_FLOW_CONTROL_CONFIG") or config_path or DEFAULT_CONFIG_PATH
        )
        self._config: dict[str, Any] = {}
        self._mtime: float | None = None
        self._reload_if_changed()

    # ---------------------------------------------------------------- config

    def _reload_if_changed(self) -> None:
        """Pick up edits to the mapping without a proxy restart.

        The mapping is expected to change under a running proxy, so it is
        re-read whenever its mtime moves. A malformed edit keeps the last good
        config rather than failing every request.
        """
        try:
            mtime = self.config_path.stat().st_mtime
        except OSError as exc:
            if not self._config:
                verbose_proxy_logger.error(
                    "flow_control: cannot read %s (%s) -- no objectives will be injected",
                    self.config_path,
                    exc,
                )
            return

        if mtime == self._mtime:
            return

        try:
            with self.config_path.open() as fh:
                loaded = yaml.safe_load(fh) or {}
        except Exception as exc:  # noqa: BLE001 -- never break traffic on a bad edit
            verbose_proxy_logger.error(
                "flow_control: failed to parse %s (%s) -- keeping previous mapping",
                self.config_path,
                exc,
            )
            return

        self._config = loaded
        self._mtime = mtime
        verbose_proxy_logger.info(
            "flow_control: loaded %s (%d policies, %d key / %d user / %d team rules, default=%s)",
            self.config_path,
            len(loaded.get("policies") or {}),
            len(loaded.get("keys") or {}),
            len(loaded.get("users") or {}),
            len(loaded.get("teams") or {}),
            (loaded.get("defaults") or {}).get("policy")
            or (loaded.get("defaults") or {}).get("objective"),
        )

    @property
    def defaults(self) -> dict[str, Any]:
        return self._config.get("defaults") or {}

    def _applies_to_model(self, model: str | None) -> bool:
        """Only decorate calls routed at llm-d -- other providers reject unknown headers."""
        patterns = self._config.get("models") or ["*"]
        if model is None:
            return False
        return any(fnmatch.fnmatch(model, pattern) for pattern in patterns)

    # ------------------------------------------------------------ resolution

    def _named_policy(self, name: str) -> dict[str, Any]:
        policies = self._config.get("policies") or {}
        policy = policies.get(name)
        if not isinstance(policy, dict):
            verbose_proxy_logger.warning(
                "flow_control: unknown policy %r (defined: %s)",
                name,
                ", ".join(sorted(policies)) or "none",
            )
            return {}
        return policy

    def _object_policy(self, obj: dict[str, Any]) -> dict[str, Any]:
        """The policy an object (a rule, a `models:` override, `defaults:`) carries.

        `policy:` is either a name into the top-level `policies:` section or an
        inline mapping; policy fields set directly on the object override it
        field by field.
        """
        merged: dict[str, Any] = {}
        ref = obj.get("policy")
        if isinstance(ref, str):
            merged.update(self._named_policy(ref))
        elif isinstance(ref, dict):
            merged.update(ref)
        merged.update({k: obj[k] for k in POLICY_FIELDS if obj.get(k) is not None})
        return merged

    def _model_override(
        self, rule: dict[str, Any], model: str | None
    ) -> tuple[dict[str, Any], str] | None:
        """A rule's per-model policy override: `models: {pattern: policy}`.

        Patterns are matched in declaration order, first match wins. The value
        is a policy name or an inline policy mapping.
        """
        overrides = rule.get("models")
        if model is None or not isinstance(overrides, dict):
            return None
        for pattern, ref in overrides.items():
            if not fnmatch.fnmatch(model, str(pattern)):
                continue
            if isinstance(ref, str):
                return self._named_policy(ref), f"model:{pattern}->{ref}"
            if isinstance(ref, dict):
                return self._object_policy(ref), f"model:{pattern}"
            verbose_proxy_logger.warning(
                "flow_control: bad model override %r: %r (expected policy name or mapping)",
                pattern,
                ref,
            )
            return None
        return None

    def _resolve(
        self, user_api_key_dict: UserAPIKeyAuth, model: str | None
    ) -> tuple[dict[str, Any], str]:
        """Return (effective policy, why) for this caller and model.

        Tier precedence is unchanged: key alias > user > team > defaults, first
        tier with a rule wins. The winning tier's policy is layered over the
        defaults' policy, and its `models:` override (if one matches the
        requested model) is layered over that -- so an override that only sets
        `objective` inherits the rule's demotion settings.
        """
        effective = self._object_policy(self.defaults)
        why = "default"
        for section, field in (("keys", "key_alias"), ("users", "user_id"), ("teams", "team_id")):
            identity = getattr(user_api_key_dict, field, None)
            if identity is None:
                continue
            rule = (self._config.get(section) or {}).get(identity)
            if not rule:
                continue
            why = f"{section}:{identity}"
            if isinstance(rule.get("policy"), str):
                why += f" policy={rule['policy']}"
            effective.update(self._object_policy(rule))
            override = self._model_override(rule, model)
            if override is not None:
                policy, label = override
                effective.update(policy)
                why += f" {label}"
            break
        return effective, why

    def _apply_team_key_defaults(self, user_api_key_dict: UserAPIKeyAuth) -> None:
        """Default LiteLLM limits onto a key from its team's `key_defaults:`.

        A team-wide `team_tpm_limit` is one counter shared by the whole team;
        `key_defaults:` instead gives every member key its *own* limit without
        configuring each key. A limit the key already carries always wins --
        these are defaults, not caps.

        This works because config callbacks run before LiteLLM's built-in
        hooks: the fields are set here, then the v3 limiter's pre-call reads
        them off the same UserAPIKeyAuth and enforces + tracks them exactly as
        if the key row carried them. Applied to every request through the
        proxy (not just llm-d-routed models), so accounting stays uniform.
        """
        team_id = getattr(user_api_key_dict, "team_id", None)
        if not team_id:
            return
        rule = (self._config.get("teams") or {}).get(team_id) or {}
        defaults = rule.get("key_defaults")
        if not isinstance(defaults, dict):
            return

        applied = []
        for field in KEY_DEFAULT_FIELDS:
            if defaults.get(field) is None:
                continue
            if getattr(user_api_key_dict, field, None) is None:
                setattr(user_api_key_dict, field, defaults[field])
                applied.append(field)

        meta_defaults = {
            f: defaults[f] for f in KEY_DEFAULT_METADATA_FIELDS if defaults.get(f) is not None
        }
        if meta_defaults:
            metadata = user_api_key_dict.metadata or {}
            for field, value in meta_defaults.items():
                if not metadata.get(field):
                    metadata[field] = value
                    applied.append(field)
            user_api_key_dict.metadata = metadata

        if applied:
            verbose_proxy_logger.debug(
                "flow_control: applied team %s key_defaults to key_alias=%s: %s",
                team_id,
                user_api_key_dict.key_alias,
                ", ".join(applied),
            )

    def _fairness_id(
        self, policy: dict[str, Any], user_api_key_dict: UserAPIKeyAuth
    ) -> str | None:
        """Explicit id wins; otherwise read the configured identity field off the key."""
        explicit = policy.get("fairness_id")
        if explicit:
            return str(explicit)

        source = policy.get("fairness_id_from") or "key_alias"
        if source not in IDENTITY_FIELDS:
            verbose_proxy_logger.warning(
                "flow_control: unknown fairness_id_from %r (expected one of %s)",
                source,
                ", ".join(IDENTITY_FIELDS),
            )
            return None

        value = getattr(user_api_key_dict, source, None)
        if value:
            return str(value)

        # An unauthenticated or aliasless key still needs *a* flow id, else every
        # such request shares one bucket. Fall through the remaining identities.
        for field in IDENTITY_FIELDS:
            value = getattr(user_api_key_dict, field, None)
            if value:
                return str(value)
        return None

    # ------------------------------------------------------------ saturation

    async def _get_saturation(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        data: dict,
        call_type: str,
        scopes: list[str] | None = None,
    ) -> float | None:
        """Max saturation (0.0-1.0+) across the caller's v3 limiter windows.

        `scopes` (a policy's `demote_on`) restricts which limits count, by the
        limiter's own descriptor keys: api_key, user, team, team_member,
        end_user, model_per_key, model_per_team. None means all of them.

        Reuses the v3 parallel request limiter's own counters rather than
        keeping any of our own. Two paths, cheapest first:

        - If the limiter's pre-call already ran for this request, its
          RateLimitResponse is stashed on a ContextVar: read it back, zero
          cache reads. (Those are the exact numbers it later mirrors into the
          x-ratelimit-* response headers.)
        - Otherwise build the same descriptors the limiter would and call
          should_rate_limit(read_only=True): the in-memory tier of its
          internal DualCache answers first (kept warm by every enforced
          request), Redis fills the gaps -- slightly stale cross-instance,
          which is fine for a priority decision.

        These are internal LiteLLM APIs (verified against 1.100.1), so any
        failure degrades to "no saturation signal", never to a failed request.
        """
        try:
            from litellm.proxy.hooks.parallel_request_limiter_v3 import (
                _PROXY_MaxParallelRequestsHandler_v3,
                get_request_stash_for_call,
            )
            from litellm.proxy.proxy_server import proxy_logging_obj

            limiter = proxy_logging_obj.get_proxy_hook("parallel_request_limiter")
            if not isinstance(limiter, _PROXY_MaxParallelRequestsHandler_v3):
                return None

            stash = get_request_stash_for_call(data.get("litellm_call_id"))
            response = stash.rate_limit_response if stash is not None else None

            if response is None:
                metadata = user_api_key_dict.metadata or {}
                descriptors = limiter._create_rate_limit_descriptors(
                    user_api_key_dict=user_api_key_dict,
                    data=data,
                    rpm_limit_type=metadata.get("rpm_limit_type"),
                    tpm_limit_type=metadata.get("tpm_limit_type"),
                    model_has_failures=False,
                    call_type=call_type,
                )
                if not descriptors:
                    return None
                response = await limiter.should_rate_limit(
                    descriptors=descriptors,
                    parent_otel_span=user_api_key_dict.parent_otel_span,
                    read_only=True,
                )

            saturation = None
            for status in response["statuses"]:
                if scopes is not None and status.get("descriptor_key") not in scopes:
                    continue
                limit = status.get("current_limit") or 0
                if limit <= 0:
                    continue
                used = max(limit - status.get("limit_remaining", limit), 0)
                saturation = max(saturation or 0.0, used / limit)
            return saturation
        except Exception as exc:  # noqa: BLE001 -- a broken read must not break traffic
            verbose_proxy_logger.warning("flow_control: saturation read failed: %s", exc)
            return None

    def _demote_objective(self, policy: dict[str, Any]) -> str:
        return str(
            policy.get("demote_to")
            or self.defaults.get("demote_objective")  # legacy spelling
            or DEFAULT_DEMOTE_OBJECTIVE
        )

    # --------------------------------------------------------------- headers

    @staticmethod
    def _strip(headers: Any) -> list[str]:
        """Remove client-supplied flow control headers in place, case-insensitively."""
        if not isinstance(headers, dict):
            return []
        found = [h for h in headers if h.lower() in FLOW_CONTROL_HEADERS]
        for header in found:
            headers.pop(header, None)
        return found

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache,
        data: dict,
        call_type: str,
    ) -> dict:
        self._reload_if_changed()

        # Before anything else -- and before the built-in v3 limiter's own
        # pre-call runs -- fill in LiteLLM limits from the team's key_defaults,
        # so the limiter enforces and tracks them like key-native limits.
        self._apply_team_key_defaults(user_api_key_dict)

        # Strip everywhere the inbound headers can reach the backend or the logs:
        # `headers` is what forward_client_headers_to_llm_api populates,
        # `extra_headers` is the body-level escape hatch, and proxy_server_request
        # is what gets handed to the logging callbacks.
        spoofed: list[str] = []
        spoofed += self._strip(data.get("headers"))
        spoofed += self._strip(data.get("extra_headers"))
        spoofed += self._strip((data.get("proxy_server_request") or {}).get("headers"))
        if spoofed:
            verbose_proxy_logger.warning(
                "flow_control: dropped client-supplied %s from key_alias=%s user=%s",
                ", ".join(sorted(set(spoofed))),
                user_api_key_dict.key_alias,
                user_api_key_dict.user_id,
            )

        model = data.get("model")
        if not self._applies_to_model(model):
            return data

        policy, why = self._resolve(user_api_key_dict, model)
        objective = policy.get("objective")
        if not objective:
            verbose_proxy_logger.debug("flow_control: no objective for %s (%s)", model, why)
            return data

        injected = {OBJECTIVE_HEADER: str(objective)}

        demote_at = policy.get("demote_at")
        if demote_at is not None:
            scopes = policy.get("demote_on")
            if scopes is not None and not isinstance(scopes, list):
                scopes = [scopes]
            saturation = await self._get_saturation(user_api_key_dict, data, call_type, scopes)
            if saturation is not None:
                injected[SATURATION_HEADER] = f"{saturation:.4f}"
                if saturation >= float(demote_at):
                    objective = self._demote_objective(policy)
                    injected[OBJECTIVE_HEADER] = objective
                    verbose_proxy_logger.info(
                        "flow_control: %s at %.0f%% of its rate limits (demote_at=%.0f%%) -> %s",
                        why,
                        saturation * 100,
                        float(demote_at) * 100,
                        objective,
                    )

        fairness_id = self._fairness_id(policy, user_api_key_dict)
        if fairness_id:
            injected[FAIRNESS_HEADER] = fairness_id

        # `extra_headers` wins over `headers` in litellm.completion(), so write
        # there; also update `headers` when header forwarding is turned on so the
        # two never disagree.
        data.setdefault("extra_headers", {}).update(injected)
        if isinstance(data.get("headers"), dict):
            data["headers"].update(injected)

        verbose_proxy_logger.info(
            "flow_control: %s -> objective=%s fairness_id=%s (matched %s)",
            model,
            injected[OBJECTIVE_HEADER],
            fairness_id,
            why,
        )
        return data


# Referenced from config.yaml as `hooks.flow_control.flow_control_hook`.
flow_control_hook = FlowControlHook()
