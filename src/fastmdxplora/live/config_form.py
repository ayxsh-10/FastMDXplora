"""Configure tab: a manual configuration form mapped to the real FastMDXplora
config schema, plus request handling to save the config and launch a run.

The form fields are a curated, human-friendly subset of
:mod:`fastmdxplora.config.schema`. Each field records the phase block and key
it maps to, so the posted values are written straight into a valid config YAML
that ``fastmdx explore --config`` consumes.
"""

from __future__ import annotations

import re
import subprocess
import sys
import threading
from dataclasses import dataclass, field as dataclass_field
from html import escape
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FormField:
    key: str
    label: str
    kind: str  # text | number | range | select | toggle | checkboxes
    phase: str | None = None  # "setup" | "simulation" | "report" | None (top-level)
    cast: str = "str"  # str | int | float | bool | list
    default: Any = None
    options: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    unit: str = ""
    placeholder: str = ""
    help: str = ""
    variant: str = ""  # e.g. "ph" for the acidic->alkaline gradient slider


@dataclass(frozen=True)
class FormSection:
    title: str
    description: str
    fields: tuple[FormField, ...] = dataclass_field(default_factory=tuple)


# --- The curated form, every field mapped to a real schema key --------------
FORM_SECTIONS: tuple[FormSection, ...] = (
    FormSection(
        "System",
        "Input structure and study identity.",
        (
            FormField("system", "System (PDB path or PDB ID)", "text", None, "str",
                      placeholder="e.g. 1UBQ or path/to/protein.pdb",
                      help="A local PDB file or an RCSB PDB ID to fetch."),
            FormField("title", "Study title", "text", "report", "str",
                      placeholder="My MD study",
                      help="Recorded in the report (report.title)."),
        ),
    ),
    FormSection(
        "Setup",
        "System preparation: force field, protonation, solvation.",
        (
            FormField("forcefield", "Force field", "select", "setup", "str",
                      default="charmm36",
                      options=("charmm36", "amber14", "amber-fb15"),
                      help="setup.forcefield"),
            FormField("ph", "pH", "range", "setup", "float", default=7.0,
                      minimum=1, maximum=14, step=0.5, variant="ph",
                      help="setup.ph — protonation states."),
            FormField("ion_concentration_M", "Ionic strength", "number", "setup", "float",
                      default=0.15, minimum=0, step=0.01, unit="M",
                      help="setup.ion_concentration_M (physiological ~0.15)."),
            FormField("box_shape", "Box shape", "select", "setup", "str",
                      default="cube",
                      options=("cube", "dodecahedron", "octahedron"),
                      help="setup.box_shape"),
            FormField("solvent_padding_nm", "Solvent padding", "number", "setup", "float",
                      default=1.0, minimum=0.5, step=0.1, unit="nm",
                      help="setup.solvent_padding_nm — solute-to-wall distance."),
        ),
    ),
    FormSection(
        "Simulation",
        "Molecular dynamics production settings.",
        (
            FormField("duration_ns", "Production duration", "range", "simulation", "float",
                      default=2.0, minimum=1, maximum=500, step=1, unit="ns",
                      help="simulation.duration_ns"),
            FormField("temperature_K", "Temperature", "number", "simulation", "float",
                      default=300.0, minimum=0, step=1, unit="K",
                      help="simulation.temperature_K"),
            FormField("timestep_fs", "Timestep", "number", "simulation", "float",
                      default=2.0, minimum=0.5, step=0.5, unit="fs",
                      help="simulation.timestep_fs"),
            FormField("integrator", "Integrator", "select", "simulation", "str",
                      default="langevin_middle",
                      options=("langevin_middle", "langevin", "verlet", "brownian"),
                      help="simulation.integrator"),
            FormField("pressure_bar", "Pressure", "number", "simulation", "float",
                      default=1.0, step=0.1, unit="bar",
                      help="simulation.pressure_bar"),
            FormField("platform", "Compute platform", "select", "simulation", "str",
                      default="auto",
                      options=("auto", "CUDA", "OpenCL", "CPU", "HIP"),
                      help="simulation.platform"),
            FormField("precision", "Precision", "select", "simulation", "str",
                      default="mixed",
                      options=("single", "mixed", "double"),
                      help="simulation.precision"),
            FormField("trajectory_interval_steps", "Frame interval", "number", "simulation", "int",
                      default=1000, minimum=1, step=1, unit="steps",
                      help="simulation.trajectory_interval_steps"),
            FormField("minimize", "Energy minimize first", "toggle", "simulation", "bool",
                      default=True, help="simulation.minimize"),
        ),
    ),
    FormSection(
        "Phases",
        "Which pipeline phases to run.",
        (
            FormField("phases", "Phases to run", "checkboxes", None, "list",
                      default=("setup", "simulation", "analysis", "report"),
                      options=("setup", "simulation", "analysis", "report"),
                      help="Top-level include: subset of phases, in order."),
        ),
    ),
)


def _all_fields() -> list[FormField]:
    return [f for section in FORM_SECTIONS for f in section.fields]


# --- Rendering --------------------------------------------------------------
def render_config_form() -> str:
    sections = "\n".join(_render_section(s) for s in FORM_SECTIONS)
    return f"""
      <form id="config-form" class="config-form" autocomplete="off">
        {sections}
        <div class="form-actions">
          <button type="button" class="button ghost" id="save-config">Save config</button>
          <button type="button" class="button primary" id="run-simulation">Run simulation</button>
        </div>
        <div class="run-result" id="run-result" hidden></div>
      </form>
    """


def _render_section(section: FormSection) -> str:
    fields = "\n".join(_render_field(f) for f in section.fields)
    return (
        '<section class="form-section">'
        f'<div class="form-section-head"><h2>{escape(section.title)}</h2>'
        f'<p class="muted">{escape(section.description)}</p></div>'
        f'<div class="form-grid">{fields}</div>'
        "</section>"
    )


def _render_field(f: FormField) -> str:
    fid = f"cfg-{f.key}"
    label = f'<label for="{fid}">{escape(f.label)}</label>'
    help_html = f'<span class="field-help">{escape(f.help)}</span>' if f.help else ""
    body = ""
    if f.kind == "text":
        body = (
            f'<input type="text" id="{fid}" name="{escape(f.key)}" '
            f'data-cfg-key="{escape(f.key)}" placeholder="{escape(f.placeholder)}">'
        )
    elif f.kind == "number":
        attrs = _num_attrs(f)
        val = "" if f.default is None else escape(str(f.default))
        unit = f'<span class="unit">{escape(f.unit)}</span>' if f.unit else ""
        body = (
            f'<div class="input-row"><input type="number" id="{fid}" '
            f'data-cfg-key="{escape(f.key)}" value="{val}"{attrs}>{unit}</div>'
        )
    elif f.kind == "range":
        attrs = _num_attrs(f)
        val = "" if f.default is None else escape(str(f.default))
        cls = "range" + (" range-ph" if f.variant == "ph" else "")
        scale = ""
        if f.variant == "ph":
            scale = '<div class="range-scale"><span>Acidic</span><span>Neutral</span><span>Alkaline</span></div>'
        body = (
            f'<div class="range-row"><input type="range" class="{cls}" id="{fid}" '
            f'data-cfg-key="{escape(f.key)}" value="{val}"{attrs}>'
            f'<output class="range-out" for="{fid}">{val}{(" " + escape(f.unit)) if f.unit else ""}</output>'
            f"</div>{scale}"
        )
    elif f.kind == "select":
        opts = "".join(
            f'<option value="{escape(o)}"{" selected" if o == f.default else ""}>{escape(o)}</option>'
            for o in f.options
        )
        body = f'<select id="{fid}" data-cfg-key="{escape(f.key)}">{opts}</select>'
    elif f.kind == "toggle":
        checked = " checked" if f.default else ""
        body = (
            f'<label class="toggle"><input type="checkbox" id="{fid}" '
            f'data-cfg-key="{escape(f.key)}"{checked}><span class="track"></span></label>'
        )
    elif f.kind == "checkboxes":
        boxes = "".join(
            f'<label class="chip-check"><input type="checkbox" value="{escape(o)}" '
            f'data-cfg-group="{escape(f.key)}"{" checked" if o in (f.default or ()) else ""}>'
            f"<span>{escape(o)}</span></label>"
            for o in f.options
        )
        body = f'<div class="chip-checks">{boxes}</div>'
    field_class = "field" + (" field-wide" if f.kind == "checkboxes" else "")
    return f'<div class="{field_class}">{label}{body}{help_html}</div>'


def _num_attrs(f: FormField) -> str:
    parts = []
    if f.minimum is not None:
        parts.append(f' min="{_fmt(f.minimum)}"')
    if f.maximum is not None:
        parts.append(f' max="{_fmt(f.maximum)}"')
    if f.step is not None:
        parts.append(f' step="{_fmt(f.step)}"')
    return "".join(parts)


def _fmt(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


# --- Config construction ----------------------------------------------------
def _cast(value: Any, cast: str) -> Any:
    if value is None:
        return None
    if cast == "bool":
        return bool(value)
    if cast == "int":
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None
    if cast == "float":
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    if cast == "list":
        return list(value) if isinstance(value, (list, tuple)) else [value]
    text = str(value).strip()
    return text or None


def build_run_config(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Translate posted form values into a valid FastMDXplora config dict."""
    setup: dict[str, Any] = {}
    simulation: dict[str, Any] = {}
    report: dict[str, Any] = {}
    blocks = {"setup": setup, "simulation": simulation, "report": report}

    for f in _all_fields():
        if f.key in {"system", "title", "phases"}:
            continue
        if f.key not in payload:
            continue
        value = _cast(payload.get(f.key), f.cast)
        if value is None or value == "":
            continue
        block = blocks.get(f.phase or "")
        if block is not None:
            block[f.key] = value

    simulation["live_telemetry"] = True

    config: dict[str, Any] = {"output": root.as_posix()}
    system = str(payload.get("system") or "").strip()
    if system:
        config["systems"] = [{"id": _system_id(system), "system": system}]
    phases = payload.get("phases")
    if isinstance(phases, (list, tuple)) and list(phases):
        config["include"] = list(phases)

    title = str(payload.get("title") or "").strip()
    if title:
        report["title"] = title

    if setup:
        config["setup"] = setup
    if simulation:
        config["simulation"] = simulation
    if report:
        config["report"] = report
    return config


def _system_id(system: str) -> str:
    stem = Path(system).stem or system
    slug = re.sub(r"[^0-9A-Za-z]+", "_", stem).strip("_").lower()
    return slug or "system1"


def _write_yaml(path: Path, config: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# FastMDXplora config generated from the Configure tab.\n")
        yaml.safe_dump(config, fh, sort_keys=False, default_flow_style=False)


# --- Run management ---------------------------------------------------------
_runs: dict[str, subprocess.Popen] = {}
_runs_lock = threading.Lock()


def run_active(root: Path) -> bool:
    with _runs_lock:
        proc = _runs.get(root.as_posix())
        return proc is not None and proc.poll() is None


def _launch(root: Path, config_path: Path, log_path: Path) -> int:
    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "fastmdxplora.cli.main", "explore",
         "--config", str(config_path)],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    with _runs_lock:
        _runs[root.as_posix()] = proc
    return proc.pid


def handle_run_request(root: Path, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Save the config and (unless dry_run) launch ``fastmdx explore``.

    Returns ``(http_status, json_body)``.
    """
    dry_run = bool(payload.get("dry_run"))
    system = str(payload.get("system") or "").strip()
    if not system and not dry_run:
        return 400, {"error": "A system (PDB path or PDB ID) is required to run."}

    try:
        config = build_run_config(root, payload)
    except (ValueError, TypeError) as exc:
        return 400, {"error": f"Invalid configuration: {exc}"}

    config_path = root / "ui_run_config.yml"
    try:
        _write_yaml(config_path, config)
    except OSError as exc:
        return 500, {"error": f"Could not write config: {exc}"}

    if dry_run:
        return 200, {"status": "validated", "config_path": config_path.name, "config": config}

    if run_active(root):
        return 409, {"error": "A run is already in progress for this output directory."}

    try:
        pid = _launch(root, config_path, root / "ui_run.log")
    except OSError as exc:
        return 500, {"error": f"Failed to launch run: {exc}"}
    return 200, {
        "status": "started",
        "pid": pid,
        "config_path": config_path.name,
        "log": "ui_run.log",
    }
