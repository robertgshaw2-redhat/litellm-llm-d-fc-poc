"""LiteLLM pre-call hook: map the caller's identity to llm-d flow control headers.

llm-d's flow control layer reads two headers on the way into the EPP:

    x-llm-d-inference-objective    the InferenceObjective (priority band) to admit under
    x-llm-d-inference-fairness-id  the flow id used for fair sharing inside that band

Neither can be trusted from the client -- anything that reaches the gateway could
otherwise promote itself into the latency-critical band and starve everyone else.
So this hook does two things on every request:

    1. strips both headers off whatever the client sent
    2. re-injects them from a static mapping of LiteLLM identity -> objective

Precedence is most-specific-first: key alias, then user, then team, then the
default. See objectives.yaml for the mapping itself.

A rule may also carry soft token limits (`limits:`). Those are not LiteLLM's
tpm/rpm limits -- exceeding them never returns a 429. Instead, actual token
usage is charged against per-window counters after each response, and once a
window's budget is spent, subsequent requests are *demoted*: they still go
through, but stamped with a lower objective (`demote_to`, default
"best-effort") until the window rolls over. Counters live in the proxy's
DualCache, so with a Redis cache configured they are shared across workers.
"""

from __future__ import annotations

import fnmatch
import os
import time
from pathlib import Path
from typing import Any

import yaml
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth

OBJECTIVE_HEADER = "x-llm-d-inference-objective"
FAIRNESS_HEADER = "x-llm-d-inference-fairness-id"
FLOW_CONTROL_HEADERS = (OBJECTIVE_HEADER, FAIRNESS_HEADER)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "objectives.yaml"

# Fields on the authenticated key that a fairness id may be derived from.
# `api_key` is the salted hash of the virtual key, not the key itself.
IDENTITY_FIELDS = ("key_alias", "user_id", "team_id", "end_user_id", "api_key")

# Soft limit windows: (field under `limits:`, strftime window label, counter TTL).
# Fixed windows keyed by wall-clock label, same scheme as LiteLLM's own
# parallel request limiter; the TTL only garbage-collects dead windows.
LIMIT_WINDOWS = (
    ("tokens_per_minute", "%Y-%m-%dT%H:%M", 2 * 60),
    ("tokens_per_hour", "%Y-%m-%dT%H", 2 * 3600),
)
DEFAULT_DEMOTE_OBJECTIVE = "best-effort"
CACHE_PREFIX = "llmd-flow-control"


class FlowControlHook(CustomLogger):
    def __init__(self, config_path: str | Path | None = None) -> None:
        super().__init__()
        self.config_path = Path(
            os.getenv("LLMD_FLOW_CONTROL_CONFIG") or config_path or DEFAULT_CONFIG_PATH
        )
        self._config: dict[str, Any] = {}
        self._mtime: float | None = None
        # The proxy's DualCache, captured in async_pre_call_hook: the post-call
        # hook that charges usage is not handed a cache reference by LiteLLM.
        self._cache: Any = None
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
            "flow_control: loaded %s (%d key / %d user / %d team rules, default=%s)",
            self.config_path,
            len(loaded.get("keys") or {}),
            len(loaded.get("users") or {}),
            len(loaded.get("teams") or {}),
            (loaded.get("defaults") or {}).get("objective"),
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

    def _lookup(self, user_api_key_dict: UserAPIKeyAuth) -> tuple[dict[str, Any], str]:
        """Return (rule, why) for this caller, most specific match first."""
        for section, field in (("keys", "key_alias"), ("users", "user_id"), ("teams", "team_id")):
            identity = getattr(user_api_key_dict, field, None)
            if identity is None:
                continue
            rule = (self._config.get(section) or {}).get(identity)
            if rule:
                return rule, f"{section}:{identity}"
        return self.defaults, "default"

    def _fairness_id(self, rule: dict[str, Any], user_api_key_dict: UserAPIKeyAuth) -> str | None:
        """Explicit id wins; otherwise read the configured identity field off the key."""
        explicit = rule.get("fairness_id")
        if explicit:
            return str(explicit)

        source = rule.get("fairness_id_from") or self.defaults.get("fairness_id_from") or "key_alias"
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

    # ---------------------------------------------------------- soft limits

    @staticmethod
    def _window_key(bucket: str, label: str) -> str:
        return f"{CACHE_PREFIX}::{bucket}::{label}::tokens"

    async def _spent_window(
        self, cache: Any, bucket: str, limits: dict[str, Any]
    ) -> tuple[str, float, float] | None:
        """Return (window, used, limit) for the first exhausted window, else None.

        The bucket is the *matched rule* (e.g. "teams:team-a"), so every caller
        sharing that rule draws from one budget -- a group grant, not per-key.
        """
        now = time.gmtime()
        for field, fmt, _ttl in LIMIT_WINDOWS:
            limit = limits.get(field)
            if not limit:
                continue
            used = await cache.async_get_cache(self._window_key(bucket, time.strftime(fmt, now)))
            if used is not None and float(used) >= float(limit):
                return field, float(used), float(limit)
        return None

    def _demote_objective(self, limits: dict[str, Any]) -> str:
        return str(
            limits.get("demote_to")
            or self.defaults.get("demote_objective")
            or DEFAULT_DEMOTE_OBJECTIVE
        )

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
    ) -> None:
        """Charge actual token usage against the matched rule's limit windows.

        Counting happens after the response, so a burst can overshoot a window
        by however much was already in flight -- acceptable for tiering, this
        is not a billing meter. Streaming responses do not pass through this
        hook; a production version would also count them via the logging
        callback's completed-stream event.
        """
        cache = self._cache
        if cache is None or not self._applies_to_model(data.get("model")):
            return None
        rule, why = self._lookup(user_api_key_dict)
        limits = rule.get("limits") or {}
        if not any(limits.get(field) for field, _fmt, _ttl in LIMIT_WINDOWS):
            return None

        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        total = usage.get("total_tokens") if isinstance(usage, dict) else getattr(usage, "total_tokens", None)
        if not total:
            return None

        now = time.gmtime()
        for field, fmt, ttl in LIMIT_WINDOWS:
            if not limits.get(field):
                continue
            await cache.async_increment_cache(
                key=self._window_key(why, time.strftime(fmt, now)),
                value=float(total),
                ttl=ttl,
            )
        return None

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

        rule, why = self._lookup(user_api_key_dict)
        objective = rule.get("objective")
        if not objective:
            verbose_proxy_logger.debug("flow_control: no objective for %s (%s)", model, why)
            return data

        limits = rule.get("limits") or {}
        if limits:
            self._cache = cache
            spent = await self._spent_window(cache, why, limits)
            if spent:
                window, used, limit = spent
                objective = self._demote_objective(limits)
                verbose_proxy_logger.info(
                    "flow_control: %s over %s (%.0f/%.0f tokens) -> demoted to %s",
                    why,
                    window,
                    used,
                    limit,
                    objective,
                )

        injected = {OBJECTIVE_HEADER: str(objective)}
        fairness_id = self._fairness_id(rule, user_api_key_dict)
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
            objective,
            fairness_id,
            why,
        )
        return data


# Referenced from config.yaml as `hooks.flow_control.flow_control_hook`.
flow_control_hook = FlowControlHook()
