"""Immutable primitives for the SIEVE data pipeline.

A Document flows through a sequence of Stages. Each Stage is a pure function
Document -> Document that appends a StageRecord describing what it did.
Provenance is an append-only log attached to every Document / Paragraph, so
you can reconstruct the full lineage of any piece of text at any time.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional, Union


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StageRecord:
    stage: str
    in_chars: int
    out_chars: int
    wall_ms: int
    notes: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    @property
    def delta_chars(self) -> int:
        return self.out_chars - self.in_chars

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "in_chars": self.in_chars,
            "out_chars": self.out_chars,
            "delta_chars": self.delta_chars,
            "wall_ms": self.wall_ms,
            "notes": dict(self.notes),
            "params": dict(self.params),
        }


@dataclass(frozen=True)
class Provenance:
    doc_id: str
    title: str
    source_path: str
    sha256: str
    stages: tuple = ()

    def with_stage(self, rec: StageRecord) -> "Provenance":
        return replace(self, stages=self.stages + (rec,))

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "source_path": self.source_path,
            "sha256": self.sha256,
            "stages": [s.to_dict() for s in self.stages],
        }


@dataclass(frozen=True)
class Document:
    prov: Provenance
    text: str
    attrs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Paragraph:
    prov: Provenance
    para_id: str
    text: str
    attrs: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "para_id": self.para_id,
            "doc_id": self.prov.doc_id,
            "title": self.prov.title,
            "source_path": self.prov.source_path,
            "text": self.text,
            "attrs": dict(self.attrs),
        }


Stage = Callable[[Document], Document]
StageResult = Union[str, tuple]


def stage(name: str, **params):
    """Decorator turning a text-level function into a Stage.

    The wrapped function takes the document's text and returns either
    ``new_text`` or ``(new_text, notes_dict)``. Timing and char counts are
    recorded automatically.
    """

    def wrap(fn: Callable[[str], StageResult]) -> Stage:
        def runner(doc: Document) -> Document:
            t0 = time.time()
            before = doc.text
            result = fn(before)
            if isinstance(result, tuple):
                after, notes = result
            else:
                after, notes = result, {}
            wall_ms = int((time.time() - t0) * 1000)
            rec = StageRecord(
                stage=name,
                in_chars=len(before),
                out_chars=len(after),
                wall_ms=wall_ms,
                notes=notes,
                params=dict(params),
            )
            return Document(prov=doc.prov.with_stage(rec), text=after, attrs=doc.attrs)

        runner.__stage_name__ = name
        runner.__stage_params__ = dict(params)
        return runner

    return wrap


def pipe(*stages: Stage) -> Stage:
    def run(doc: Document) -> Document:
        d = doc
        for s in stages:
            d = s(d)
        return d

    run.__stage_name__ = "pipe(" + ",".join(getattr(s, "__stage_name__", "?") for s in stages) + ")"
    run.__stages__ = stages
    return run


def run_with_history(doc: Document, stages: list) -> tuple:
    """Apply each stage and return (final_doc, [doc_after_each_stage])."""
    history = [doc]
    d = doc
    for s in stages:
        d = s(d)
        history.append(d)
    return d, history


def make_provenance(source_path: Path, sha: str, title: Optional[str] = None) -> Provenance:
    return Provenance(
        doc_id=sha[:12],
        title=title or source_path.stem,
        source_path=str(source_path),
        sha256=sha,
    )
