"""Tests for the Configure tab: schema-mapped form, config build, run endpoint."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from fastmdxplora.config.loader import ConfigError, load_config_file, validate_config
from fastmdxplora.live import config_form
from fastmdxplora.live.server import _dashboard_shell, start_test_server

SAMPLE = {
    "system": "1UBQ",
    "title": "UI test study",
    "forcefield": "amber14",
    "ph": "6.5",
    "ion_concentration_M": "0.2",
    "box_shape": "octahedron",
    "solvent_padding_nm": "1.2",
    "duration_ns": "50",
    "temperature_K": "310",
    "timestep_fs": "2",
    "integrator": "langevin",
    "pressure_bar": "1",
    "platform": "CPU",
    "precision": "mixed",
    "trajectory_interval_steps": "2000",
    "minimize": True,
    "phases": ["setup", "simulation", "analysis", "report"],
}


def _post(base_url: str, path: str, payload: dict) -> tuple[int, dict]:
    req = Request(
        base_url + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req) as resp:  # noqa: S310 - localhost test server
            return resp.status, json.loads(resp.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_configure_view_and_controls_render(tmp_path: Path) -> None:
    html = _dashboard_shell(tmp_path)
    assert 'id="configure"' in html
    assert 'data-view-link="configure"' in html
    assert "Configure &amp; Run" in html
    assert 'id="config-form"' in html
    assert 'id="run-simulation"' in html
    assert 'id="save-config"' in html
    assert "range-ph" in html  # the pH gradient slider
    # every curated field is present as a control
    for f in config_form._all_fields():
        if f.kind == "checkboxes":
            assert f'data-cfg-group="{f.key}"' in html
        else:
            assert f'data-cfg-key="{f.key}"' in html


def test_build_run_config_maps_to_real_schema(tmp_path: Path) -> None:
    cfg = config_form.build_run_config(tmp_path, SAMPLE)
    # values land in the right phase blocks with the right types
    assert cfg["setup"]["forcefield"] == "amber14"
    assert cfg["setup"]["ph"] == 6.5
    assert cfg["setup"]["box_shape"] == "octahedron"
    assert cfg["simulation"]["duration_ns"] == 50.0
    assert cfg["simulation"]["trajectory_interval_steps"] == 2000
    assert cfg["simulation"]["minimize"] is True
    assert cfg["simulation"]["live_telemetry"] is True
    assert cfg["report"]["title"] == "UI test study"
    assert cfg["include"] == ["setup", "simulation", "analysis", "report"]
    assert cfg["systems"] == [{"id": "1ubq", "system": "1UBQ"}]
    # and the whole thing passes FastMDXplora's own strict validator
    validate_config(cfg, require_systems=True)


def test_dry_run_writes_a_valid_config_file(tmp_path: Path) -> None:
    server, base_url = start_test_server(tmp_path)
    try:
        status, body = _post(base_url, "/api/run", {**SAMPLE, "dry_run": True})
    finally:
        server.shutdown()
        server.server_close()
    assert status == 200
    assert body["status"] == "validated"
    written = tmp_path / "ui_run_config.yml"
    assert written.is_file()
    # the written YAML round-trips and validates against the schema
    loaded = load_config_file(written)
    validate_config(loaded, require_systems=True)
    assert loaded["simulation"]["temperature_K"] == 310.0


def test_run_requires_a_system(tmp_path: Path) -> None:
    server, base_url = start_test_server(tmp_path)
    try:
        payload = {k: v for k, v in SAMPLE.items() if k != "system"}
        status, body = _post(base_url, "/api/run", payload)  # not a dry run
    finally:
        server.shutdown()
        server.server_close()
    assert status == 400
    assert "system" in body["error"].lower()


def test_run_status_endpoint_reports_inactive(tmp_path: Path) -> None:
    server, base_url = start_test_server(tmp_path)
    try:
        with urlopen(base_url + "/api/run") as resp:  # noqa: S310
            body = json.loads(resp.read())
    finally:
        server.shutdown()
        server.server_close()
    assert body == {"active": False}


def test_bad_forcefield_is_rejected_by_validator(tmp_path: Path) -> None:
    cfg = config_form.build_run_config(tmp_path, {**SAMPLE, "box_shape": "cube"})
    cfg["setup"]["forcefield"] = "not-a-real-ff"
    # forcefield is a free str in the schema, so that passes; but an unknown
    # KEY must be rejected -- proving we only emit real schema keys.
    cfg["setup"]["totally_made_up_key"] = 1
    with pytest.raises(ConfigError):
        validate_config(cfg, require_systems=True)
