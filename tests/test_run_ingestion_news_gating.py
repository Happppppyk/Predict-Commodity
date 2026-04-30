"""run_ingestion._run_news_pipeline_v2 게이팅 + subprocess 격리."""
from __future__ import annotations

import os
import subprocess
from unittest import mock

import pytest

from src.ingestion import run_ingestion


def test_skips_when_env_missing(monkeypatch, capsys):
    monkeypatch.delenv("NEWS_INGESTION_API_KEY", raising=False)
    ok = run_ingestion._run_news_pipeline_v2()
    assert ok is False
    out = capsys.readouterr().out
    assert "스킵" in out or "skip" in out.lower()


def test_skips_when_env_empty(monkeypatch, capsys):
    monkeypatch.setenv("NEWS_INGESTION_API_KEY", "   ")
    ok = run_ingestion._run_news_pipeline_v2()
    assert ok is False


def test_invokes_subprocess_when_env_set(monkeypatch):
    monkeypatch.setenv("NEWS_INGESTION_API_KEY", "1")
    with mock.patch.object(run_ingestion.subprocess, "run") as mrun:
        mrun.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        ok = run_ingestion._run_news_pipeline_v2()
    assert ok is True
    assert mrun.called
    args = mrun.call_args.args[0]
    assert any(str(a).endswith("run_news_pipeline.py") for a in args), args


def test_returns_false_on_subprocess_failure_but_does_not_raise(monkeypatch, capsys):
    monkeypatch.setenv("NEWS_INGESTION_API_KEY", "1")
    with mock.patch.object(run_ingestion.subprocess, "run") as mrun:
        mrun.return_value = subprocess.CompletedProcess(args=[], returncode=2)
        ok = run_ingestion._run_news_pipeline_v2()
    assert ok is False
    out = capsys.readouterr().out
    assert "rc=2" in out or "failed" in out.lower() or "실패" in out
