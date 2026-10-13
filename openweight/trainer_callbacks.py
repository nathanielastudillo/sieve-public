"""HF Transformers TrainerCallbacks used by the openweight SFT path.

Currently just ``DashboardMetricsCallback``: forwards TRL's per-step log
dicts to the SIEVE dashboard so the live loss chart at
``/openweight`` stays in sync with what ``trl.SFTTrainer`` reports.

The callback is *optional* — it only engages if the env vars
``SIEVE_DASHBOARD_LAUNCH_ID`` (and optionally ``SIEVE_DASHBOARD_URL``)
are set. This means a vanilla ``python -m scripts.sft_openweight_member``
invocation keeps working with no callback and zero network overhead.

POSTs are fire-and-forget on a background daemon thread so a flaky
dashboard never blocks training. We use ``urllib.request`` to avoid
pulling in ``requests`` purely for one HTTP call.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_DASHBOARD_URL = "http://127.0.0.1:8096"


class DashboardMetricsCallback:
    """Forward ``on_log`` dicts from ``trl.SFTTrainer`` to the dashboard.

    Subclasses ``transformers.TrainerCallback`` at construction time so
    this module can be imported without transformers on the path (useful
    for static checks / tests). The import of the base class is deferred
    to ``__new__``.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> "DashboardMetricsCallback":
        from transformers import TrainerCallback

        if not any(isinstance(b, type) and b is TrainerCallback for b in cls.__mro__):
            new_cls = type(cls.__name__, (cls, TrainerCallback), {})
            return object.__new__(new_cls)
        return object.__new__(cls)

    def __init__(
        self,
        launch_id: str,
        dashboard_url: str = DEFAULT_DASHBOARD_URL,
        timeout_s: float = 2.0,
    ) -> None:
        self.launch_id = launch_id
        self.dashboard_url = dashboard_url.rstrip("/")
        self.endpoint = f"{self.dashboard_url}/api/openweight/launch/{launch_id}/metrics"
        self.timeout_s = timeout_s
        self._q: "queue.Queue[Optional[dict]]" = queue.Queue(maxsize=256)
        self._worker = threading.Thread(target=self._run, daemon=True, name="dashboard-metrics")
        self._worker.start()
        self._dropped = 0
        logger.info("DashboardMetricsCallback → %s", self.endpoint)

    # -- TrainerCallback hooks -------------------------------------------------

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: D401, ANN001
        """Called by Trainer every ``logging_steps`` with the latest metrics."""
        if not logs:
            return
        # ``logs`` is a dict with numeric values (loss, grad_norm, lr, entropy,
        # mean_token_accuracy, epoch, num_tokens, …). Coerce to float where we can.
        metrics = {}
        for k, v in logs.items():
            try:
                metrics[k] = float(v)
            except (TypeError, ValueError):
                metrics[k] = v
        point = {
            "step": int(getattr(state, "global_step", 0) or 0),
            "wall_time": time.time(),
            "metrics": metrics,
        }
        try:
            self._q.put_nowait(point)
        except queue.Full:
            self._dropped += 1
            if self._dropped % 50 == 1:
                logger.warning(
                    "metrics queue full, dropped %d points (endpoint=%s)",
                    self._dropped, self.endpoint,
                )

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ANN001
        # Drain by waiting briefly; daemon thread dies with the process anyway.
        self._q.put_nowait(None)
        self._worker.join(timeout=3.0)

    # -- worker ----------------------------------------------------------------

    def _run(self) -> None:
        while True:
            try:
                point = self._q.get()
            except Exception:  # noqa: BLE001
                return
            if point is None:
                return
            try:
                self._post(point)
            except Exception as exc:  # noqa: BLE001
                # Never break training on a bad HTTP call.
                logger.debug("metrics POST failed: %s", exc)

    def _post(self, point: dict) -> None:
        data = json.dumps(point).encode()
        req = urllib.request.Request(
            self.endpoint, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310
            resp.read(1)  # consume + close


def make_callback_from_env() -> Optional[DashboardMetricsCallback]:
    """Instantiate from env vars, or return None if not configured."""
    launch_id = os.environ.get("SIEVE_DASHBOARD_LAUNCH_ID")
    if not launch_id:
        return None
    url = os.environ.get("SIEVE_DASHBOARD_URL", DEFAULT_DASHBOARD_URL)
    try:
        return DashboardMetricsCallback(launch_id=launch_id, dashboard_url=url)
    except Exception as exc:  # noqa: BLE001
        logger.warning("DashboardMetricsCallback init failed: %s", exc)
        return None
