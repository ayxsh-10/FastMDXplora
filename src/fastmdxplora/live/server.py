"""Dependency-free localhost dashboard server."""

from __future__ import annotations

import json
import mimetypes
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from fastmdxplora.live.config_form import handle_run_request, render_config_form, run_active
from fastmdxplora.live.protein_preview import find_structure, protein_preview_payload
from fastmdxplora.live.telemetry import analyze_health, read_events, read_metrics, read_status

PLOT_TITLE_ALIASES = {
    "rmsd": "RMSD",
    "rmsf": "RMSF",
    "rg": "Radius of gyration",
    "hbonds": "Hydrogen bonds",
    "hbond": "Hydrogen bonds",
    "sasa": "SASA",
    "ss": "Secondary structure",
    "secondary_structure": "Secondary structure",
    "dimred_pca": "PCA",
    "pca": "PCA",
    "dimred_mds": "MDS",
    "mds": "MDS",
    "dimred_tsne": "t-SNE",
    "tsne": "t-SNE",
    "cluster_kmeans": "KMeans trajectory scatter",
    "kmeans_trajectory": "KMeans trajectory scatter",
    "kmeans_population": "KMeans population",
    "cluster_kmeans_counts": "KMeans population",
    "cluster_hierarchical": "Hierarchical trajectory scatter",
    "hierarchical_trajectory": "Hierarchical trajectory scatter",
    "hierarchical_population": "Hierarchical population",
    "cluster_hierarchical_counts": "Hierarchical population",
    "cluster_hierarchical_dendrogram": "Hierarchical dendrogram",
    "hierarchical_dendrogram": "Hierarchical dendrogram",
    "cluster_dbscan": "DBSCAN trajectory scatter",
    "dbscan_trajectory": "DBSCAN trajectory scatter",
    "dbscan_population": "DBSCAN population",
    "cluster_dbscan_counts": "DBSCAN population",
    "qvalue": "Native contacts",
    "native_contacts": "Native contacts",
    "dihedrals": "Dihedrals",
    "ramachandran": "Ramachandran plot",
}

DASHBOARD_ASSET_TITLES = {
    "rmsd_dashboard": "RMSD",
    "rmsf_dashboard": "RMSF",
    "rg_dashboard": "Radius of gyration",
    "hbonds_dashboard": "Hydrogen bonds",
    "sasa_dashboard": "SASA",
    "pca_dashboard": "PCA",
    "mds_dashboard": "MDS",
    "tsne_dashboard": "t-SNE",
    "kmeans_trajectory_dashboard": "KMeans trajectory scatter",
    "kmeans_population_dashboard": "KMeans population",
    "hierarchical_trajectory_dashboard": "Hierarchical trajectory scatter",
    "hierarchical_population_dashboard": "Hierarchical population",
    "hierarchical_dendrogram_dashboard": "Hierarchical dendrogram",
    "cluster_hierarchical_dendrogram_dashboard": "Hierarchical dendrogram",
    "dbscan_trajectory_dashboard": "DBSCAN trajectory scatter",
    "dbscan_population_dashboard": "DBSCAN population",
    "ss_dashboard": "Secondary structure",
    "qvalue_dashboard": "Native contacts",
    "dihedrals_dashboard": "Dihedrals",
}

PLOT_CATEGORY_BY_TITLE = {
    "RMSD": "Core Metrics",
    "RMSF": "Core Metrics",
    "Radius of gyration": "Core Metrics",
    "Hydrogen bonds": "Core Metrics",
    "KMeans trajectory scatter": "Clustering",
    "KMeans population": "Clustering",
    "Hierarchical trajectory scatter": "Clustering",
    "Hierarchical population": "Clustering",
    "Hierarchical dendrogram": "Clustering",
    "DBSCAN trajectory scatter": "Clustering",
    "DBSCAN population": "Clustering",
    "PCA": "Additional Analysis",
    "MDS": "Additional Analysis",
    "t-SNE": "Additional Analysis",
    "SASA": "Additional Analysis",
    "Secondary structure": "Additional Analysis",
    "Native contacts": "Additional Analysis",
    "Dihedrals": "Additional Analysis",
    "Ramachandran plot": "Additional Analysis",
}

KEY_PLOT_TITLES = {"RMSD", "RMSF", "Radius of gyration", "Hydrogen bonds", "PCA", "SASA"}


@dataclass
class DashboardSession:
    """Background local dashboard server session."""

    server: ThreadingHTTPServer
    thread: threading.Thread
    root: Path
    host: str
    port: int
    requested_port: int

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def port_was_changed(self) -> bool:
        return self.port != self.requested_port

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def wait_forever(self) -> None:
        while self.thread.is_alive():
            time.sleep(0.2)


def make_handler(project_root: str | Path) -> type[BaseHTTPRequestHandler]:
    root = Path(project_root).resolve()

    class LiveDashboardHandler(BaseHTTPRequestHandler):
        server_version = "FastMDXLive/1.0"

        def do_GET(self) -> None:  # noqa: N802 - stdlib API
            parsed = urlparse(self.path)
            path = parsed.path
            if path in {"/", "/results", "/live"}:
                self._send_html(_dashboard_shell(root))
                return
            if path == "/api/status":
                status = read_status(root)
                metrics = read_metrics(root)
                payload = {"status": status, "health": analyze_health(status, metrics)}
                self._send_json(payload)
                return
            if path == "/api/metrics":
                self._send_json({"metrics": read_metrics(root)})
                return
            if path == "/api/events":
                self._send_json({"events": read_events(root)})
                return
            if path == "/api/artifacts":
                self._send_json({"artifacts": _artifact_records(root)})
                return
            if path == "/api/results":
                self._send_json(_results_payload(root))
                return
            if path == "/api/protein-preview":
                regenerate = parsed.query == "regenerate=1"
                self._send_json(protein_preview_payload(root, regenerate=regenerate))
                return
            if path == "/api/run":
                self._send_json({"active": run_active(root)})
                return
            if path == "/structure/topology.pdb":
                self._send_structure(root)
                return
            if path.startswith("/static/"):
                self._send_static_asset(path.removeprefix("/static/"))
                return
            if path.startswith("/artifacts/"):
                self._send_artifact(root, path.removeprefix("/artifacts/"))
                return
            self.send_error(404, "Not found")

        def do_POST(self) -> None:  # noqa: N802 - stdlib API
            parsed = urlparse(self.path)
            if parsed.path == "/api/run":
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    raw = self.rfile.read(length) if length else b"{}"
                    payload = json.loads(raw or b"{}")
                    if not isinstance(payload, dict):
                        raise ValueError("payload must be a JSON object")
                except (ValueError, json.JSONDecodeError) as exc:
                    self._send_json({"error": f"Invalid request: {exc}"}, status=400)
                    return
                status, body = handle_run_request(root, payload)
                self._send_json(body, status=status)
                return
            self.send_error(404, "Not found")

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_artifact(self, root: Path, raw_rel: str) -> None:
            try:
                target = (root / unquote(raw_rel)).resolve()
                target.relative_to(root)
            except ValueError:
                self.send_error(403, "Artifact path is outside the output directory")
                return
            if not target.is_file():
                self.send_error(404, "Artifact not found")
                return
            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            data = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_structure(self, root: Path) -> None:
            target = find_structure(root)
            if target is None or not target.is_file():
                self.send_error(404, "Structure not found")
                return
            data = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "chemical/x-pdb; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_static_asset(self, raw_name: str) -> None:
            name = Path(unquote(raw_name)).name
            static_root = Path(__file__).with_name("static")
            target = (static_root / name).resolve()
            try:
                target.relative_to(static_root.resolve())
            except ValueError:
                self.send_error(403, "Static asset path is outside the asset directory")
                return
            if not target.is_file():
                self.send_error(404, "Static asset not found")
                return
            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            data = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return LiveDashboardHandler


def serve_dashboard(
    *,
    output: str | Path,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    session = start_dashboard_session(output=output, host=host, port=port)
    print(f"Live dashboard running at {session.url}")
    if session.port_was_changed:
        print(
            f"Requested port {session.requested_port} was busy, "
            f"so FastMDXplora used {session.port}."
        )
    print(f"Watching: {session.root}")
    print("Press Ctrl+C to stop.")
    try:
        session.wait_forever()
    except KeyboardInterrupt:
        pass
    finally:
        session.stop()


def start_dashboard_session(
    *,
    output: str | Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    max_port_tries: int = 50,
) -> DashboardSession:
    """Start the local dashboard server in a background thread."""
    root = Path(output).resolve()
    handler = make_handler(root)
    requested_port = int(port)
    candidates = [0] if requested_port == 0 else range(requested_port, requested_port + max_port_tries)
    last_error: OSError | None = None
    for candidate in candidates:
        try:
            server = ThreadingHTTPServer((host, int(candidate)), handler)
        except OSError as exc:
            last_error = exc
            continue
        actual_port = int(server.server_address[1])
        thread = threading.Thread(
            target=server.serve_forever,
            name=f"FastMDXLiveDashboard:{actual_port}",
            daemon=True,
        )
        thread.start()
        return DashboardSession(
            server=server,
            thread=thread,
            root=root,
            host=host,
            port=actual_port,
            requested_port=requested_port,
        )
    if last_error is not None:
        raise last_error
    raise OSError("No dashboard ports were available")


def start_test_server(project_root: str | Path) -> tuple[ThreadingHTTPServer, str]:
    handler = make_handler(project_root)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _artifact_records(root: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    if not root.exists():
        return records
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            continue
        if "__pycache__" in path.parts:
            continue
        records.append(
            {
                "path": rel,
                "name": path.name,
                "href": f"/artifacts/{rel}?v={int(path.stat().st_mtime)}",
                "size": str(path.stat().st_size),
                "mtime": str(path.stat().st_mtime),
                "display_path": _compact_path(rel),
            }
        )
    return records


def _compact_path(path: str, *, keep: int = 2) -> str:
    parts = Path(path).parts
    if len(parts) <= keep:
        return path
    return ".../" + "/".join(parts[-keep:])


def _results_payload(root: Path) -> dict[str, Any]:
    artifacts = _artifact_records(root)
    by_path = {record["path"]: record for record in artifacts}
    manifest = _load_json(root / "manifest.json")
    analysis_manifest = _load_json(root / "analysis" / "analysis_manifest.json")
    sim_manifest = _load_json(root / "simulation" / "simulation_parameters.json")
    plots = _plot_records(artifacts)
    report_paths = [
        "report/report.md",
        "report/slides.pptx",
        "report/project_bundle.zip",
        "report/dashboard.html",
        "analysis/analysis_manifest.json",
    ]
    reports = [by_path[path] for path in report_paths if path in by_path]
    dashboard = by_path.get("report/dashboard.html")
    return {
        "refreshed_at": _iso_now(),
        "has_analysis": any(record["path"].startswith("analysis/") for record in artifacts),
        "has_report": any(record["path"].startswith("report/") for record in artifacts),
        "summary": _summary_records(root, manifest, analysis_manifest, sim_manifest),
        "system": _system_info(root, manifest, analysis_manifest, sim_manifest),
        "phases": _phase_records(manifest),
        "dashboard": dashboard,
        "plots": plots,
        "key_plots": [plot for plot in plots if plot["title"] in KEY_PLOT_TITLES][:6],
        "reports": reports,
        "artifacts": artifacts,
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _summary_records(
    root: Path,
    manifest: dict[str, Any],
    analysis_manifest: dict[str, Any],
    sim_manifest: dict[str, Any],
) -> list[dict[str, str]]:
    status = str(manifest.get("status") or "not available")
    frames = analysis_manifest.get("n_frames") or sim_manifest.get("n_production_frames")
    atoms = analysis_manifest.get("n_atoms")
    sim_time = sim_manifest.get("duration_ns_actual")
    live_status = read_status(root)
    temperature = live_status.get("target_temperature_K") or _last_metric_value(root, "temperature")
    latest = live_status.get("status") or status
    return [
        {"label": "Project status", "value": status},
        {"label": "Latest status", "value": str(latest or "not available")},
        {"label": "Frames", "value": str(frames or "not available")},
        {"label": "Atoms", "value": str(atoms or "not available")},
        {"label": "Simulation time", "value": f"{sim_time} ns" if sim_time is not None else "not available"},
        {"label": "Temperature", "value": f"{temperature} K" if temperature not in (None, "") else "not available"},
    ]


def _last_metric_value(root: Path, field: str) -> str | None:
    metrics = read_metrics(root, limit=1)
    if not metrics:
        return None
    value = metrics[-1].get(field)
    return str(value) if value not in (None, "") else None


def _system_info(
    root: Path,
    manifest: dict[str, Any],
    analysis_manifest: dict[str, Any],
    sim_manifest: dict[str, Any],
) -> dict[str, str]:
    live_status = read_status(root)
    return {
        "system": str(manifest.get("system") or "not available"),
        "output_folder": root.as_posix(),
        "atoms": str(analysis_manifest.get("n_atoms") or "not available"),
        "frames": str(analysis_manifest.get("n_frames") or "not available"),
        "platform": str(live_status.get("platform") or sim_manifest.get("platform_used") or "not available"),
        "timestep_fs": str(live_status.get("timestep_fs") or "not available"),
        "checkpoint": str(live_status.get("current_checkpoint_path") or "not available"),
    }


def _phase_records(manifest: dict[str, Any]) -> list[dict[str, str]]:
    seen: dict[str, str] = {}
    for phase in manifest.get("phases", []):
        if isinstance(phase, dict):
            name = str(phase.get("name") or "").lower()
            if name:
                seen[name] = str(phase.get("status") or "unknown")
    records = []
    for name in ("setup", "simulation", "analysis", "report"):
        status = seen.get(name, "not run")
        records.append({"name": name.title() if name != "simulation" else "Simulation", "status": status})
    return records


def _plot_records(artifacts: list[dict[str, str]]) -> list[dict[str, str]]:
    image_suffixes = {".png", ".jpg", ".jpeg", ".gif", ".svg"}
    candidates = [
        record for record in artifacts
        if Path(record["path"]).suffix.lower() in image_suffixes
        and (
            record["path"].startswith("report/dashboard_assets/")
            or record["path"].startswith("analysis/")
        )
    ]
    chosen: dict[str, dict[str, str]] = {}
    for record in candidates:
        title = _plot_title(record["path"])
        if not title:
            continue
        current = chosen.get(title)
        if current is None or _plot_priority(record["path"]) < _plot_priority(current["path"]):
            mode = "dashboard view" if record["path"].startswith("report/dashboard_assets/") else "artifact fallback"
            enriched = dict(record)
            enriched.update(
                {
                    "title": title,
                    "category": PLOT_CATEGORY_BY_TITLE.get(title, _plot_category(record["path"])),
                    "mode": mode,
                }
            )
            chosen[title] = enriched
    return sorted(chosen.values(), key=lambda item: (_category_order(item["category"]), item["title"]))


def _plot_priority(path: str) -> int:
    return 0 if path.startswith("report/dashboard_assets/") else 1


def _plot_title(path: str) -> str | None:
    stem = Path(path).stem
    if stem in DASHBOARD_ASSET_TITLES:
        return DASHBOARD_ASSET_TITLES[stem]
    if stem in PLOT_TITLE_ALIASES:
        return PLOT_TITLE_ALIASES[stem]
    lower = stem.lower()
    for key, title in PLOT_TITLE_ALIASES.items():
        if lower == key or lower.endswith(f"_{key}") or key in lower:
            return title
    return _humanize_stem(stem)


def _humanize_stem(stem: str) -> str:
    return stem.replace("_", " ").replace("-", " ").strip().title()


def _plot_category(path: str) -> str:
    parts = Path(path).parts
    if "cluster" in parts or "dashboard_assets" in parts and "cluster" in path:
        return "Clustering"
    if any(part in {"rmsd", "rmsf", "rg", "hbonds", "hbond"} for part in parts):
        return "Core Metrics"
    if any(part in {"sasa", "ss", "dimred", "dihedrals", "qvalue"} for part in parts):
        return "Additional Analysis"
    return "Other"


def _category_order(category: str) -> int:
    order = {"Core Metrics": 0, "Additional Analysis": 1, "Clustering": 2, "Other": 3}
    return order.get(category, 99)


def _iso_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _dashboard_shell(root: Path) -> str:
    output_label = root.as_posix()
    config_form_html = render_config_form()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FastMDXplora Live Dashboard</title>
  <style>
    @font-face {{ font-family:"Bricolage Grotesque"; font-style:normal; font-weight:200 800; font-display:swap; src:url("/static/bricolage.woff2") format("woff2"); }}
    @font-face {{ font-family:"Hanken Grotesk"; font-style:normal; font-weight:100 900; font-display:swap; src:url("/static/hanken.woff2") format("woff2"); }}
    :root {{
      color-scheme: light dark;
      --bg:#f7f8fa; --sidebar:#ffffff; --panel:#ffffff; --panel-frame:#f8f9fb; --line:#ececf1; --line-soft:#f0f1f4;
      --text:#1a1c22; --muted:#6a6f7a; --faint:#9aa0ab; --accent:#5a57d6; --accent-ink:#4a47c4;
      --ok:#2f9e64; --warn:#bd8420; --bad:#d14343;
      --shadow:rgba(20,22,40,.06); --pill-bg:rgba(90,87,214,.10); --pill-border:rgba(90,87,214,.22);
      --sans:"Hanken Grotesk", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
      --display:"Bricolage Grotesque", "Hanken Grotesk", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
      --mono:ui-monospace, "Cascadia Code", "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg:#0e0f13; --sidebar:#16181e; --panel:#16181e; --panel-frame:#12141a; --line:#24262f; --line-soft:#20222a;
        --text:#eceef2; --muted:#9ba1ad; --faint:#6b7280; --accent:#8c88f5; --accent-ink:#a6a3ff;
        --ok:#46b87c; --warn:#d8a23f; --bad:#e06b6b;
        --shadow:rgba(0,0,0,.4); --pill-bg:rgba(140,136,245,.15); --pill-border:rgba(140,136,245,.30);
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font-family:var(--sans); -webkit-font-smoothing:antialiased; }}
    .layout {{ display:grid; grid-template-columns:260px minmax(0,1fr); min-height:100vh; }}
    aside {{ border-right:1px solid var(--line); background:var(--sidebar); padding:22px 16px; }}
    main {{ padding:24px; width:min(1480px, 100%); }}
    .brand-row {{ display:grid; grid-template-columns:40px 1fr; gap:11px; align-items:center; margin-bottom:6px; }}
    .brand-logo {{ width:40px; height:40px; border-radius:11px; object-fit:cover; box-shadow:0 1px 2px var(--shadow); }}
    .brand {{ font-family:var(--display); font-weight:650; font-size:1rem; letter-spacing:-.01em; }}
    .subtle, .muted {{ color:var(--muted); }}
    .nav-heading {{ color:var(--faint); font-size:.7rem; font-weight:700; letter-spacing:.07em; text-transform:uppercase; margin:20px 0 8px; }}
    .nav-link {{ display:flex; gap:9px; align-items:center; padding:8px 10px; border-radius:8px; color:var(--text); text-decoration:none; }}
    .nav-link.active, .nav-link:hover {{ background:var(--pill-bg); color:var(--accent-ink); }}
    .view {{ display:none; }}
    .view.active {{ display:block; }}
    .header {{ display:flex; justify-content:space-between; gap:18px; align-items:flex-end; border-bottom:1px solid var(--line); padding-bottom:18px; margin-bottom:22px; }}
    h1 {{ margin:0 0 6px; font-family:var(--display); font-size:1.6rem; font-weight:650; letter-spacing:-.02em; }}
    h2 {{ font-family:var(--display); font-weight:600; letter-spacing:-.01em; }}
    .badge {{ display:inline-flex; align-items:center; gap:8px; padding:7px 10px; border:1px solid var(--line); border-radius:999px; background:var(--panel-frame); }}
    .dot {{ width:9px; height:9px; border-radius:99px; background:var(--warn); }}
    .dot.ok {{ background:var(--ok); }} .dot.warning {{ background:var(--warn); }} .dot.failed {{ background:var(--bad); }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:14px; margin-bottom:18px; }}
    .card, .panel {{ border:1px solid var(--line); border-radius:14px; background:var(--panel); box-shadow:0 1px 2px var(--shadow); padding:16px; }}
    .card .label {{ color:var(--faint); font-size:.72rem; font-weight:600; letter-spacing:.04em; text-transform:uppercase; margin-bottom:8px; }} .card .value {{ font-family:var(--mono); font-variant-numeric:tabular-nums; font-size:1.2rem; font-weight:550; overflow-wrap:anywhere; }}
    .progress-track {{ height:13px; border-radius:99px; background:var(--panel-frame); border:1px solid var(--line); overflow:hidden; }}
    .progress-fill {{ height:100%; width:0%; background:linear-gradient(90deg,var(--accent),var(--ok)); transition:width .3s ease; }}
    .grid {{ display:grid; grid-template-columns:minmax(0,1.1fr) minmax(280px,.9fr); gap:14px; align-items:start; }}
    .live-grid {{
      display:grid;
      grid-template-columns:minmax(0,2fr) minmax(320px,1fr);
      gap:18px;
      align-items:start;
    }}
    .live-main-column {{ display:grid; gap:14px; min-width:0; }}
    .live-bottom-grid {{
      display:grid;
      grid-template-columns:minmax(0,1fr) minmax(300px,.8fr);
      gap:14px;
      align-items:start;
      margin-top:14px;
    }}
    canvas {{ width:100%; height:230px; background:var(--panel-frame); border:1px solid var(--line); border-radius:8px; }}
    .events {{ max-height:320px; overflow:auto; display:grid; gap:8px; }}
    .event {{ padding:8px 10px; border:1px solid var(--line); border-radius:7px; background:var(--panel-frame); }}
    .artifact-list {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:10px; }}
    .artifact {{ color:inherit; text-decoration:none; border:1px solid var(--line); border-radius:8px; padding:10px; background:var(--panel-frame); min-width:0; max-width:100%; overflow:hidden; display:grid; gap:3px; }}
    .artifact-title, .artifact-subtitle, .artifact-path, .plot-meta, .plot-path, .card-subtitle {{
      min-width:0;
      max-width:100%;
      overflow-wrap:anywhere;
      word-break:break-word;
      line-height:1.35;
    }}
    .artifact-title {{ font-weight:700; color:var(--text); }}
    .artifact-subtitle, .artifact-path, .card-subtitle {{ color:var(--muted); font-size:.82rem; }}
    .artifact-path, .plot-path {{
      display:-webkit-box;
      -webkit-line-clamp:2;
      -webkit-box-orient:vertical;
      overflow:hidden;
    }}
    .toolbar {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; }}
    .button {{ border:1px solid var(--line); border-radius:7px; background:var(--panel-frame); color:var(--text); padding:8px 11px; cursor:pointer; }}
    .button:hover {{ border-color:var(--pill-border); background:var(--pill-bg); }}
    .result-state {{ margin-bottom:14px; }}
    .plot-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:14px; }}
    .plot-card {{ border:1px solid var(--line); border-radius:8px; padding:12px; background:var(--panel-frame); min-width:0; max-width:100%; overflow:hidden; display:flex; flex-direction:column; gap:10px; }}
    .plot-card.card-sm {{ grid-column:span 1; }}
    .plot-card.card-md {{ grid-column:span 1; }}
    .plot-card.card-lg {{ grid-column:span 2; }}
    .plot-card.card-wide {{ grid-column:span 2; }}
    .plot-card h3, .plot-title {{ margin:0 0 8px; font-size:1rem; overflow-wrap:anywhere; line-height:1.15; }}
    .plot-title-row {{ display:flex; gap:8px; justify-content:space-between; align-items:flex-start; }}
    .size-controls {{ display:flex; gap:4px; flex-wrap:wrap; }}
    .size-button {{ border:1px solid var(--line); background:var(--panel-frame); color:var(--muted); border-radius:5px; padding:2px 5px; font-size:.68rem; cursor:pointer; }}
    .size-button:hover, .size-button.active {{ color:white; border-color:var(--pill-border); }}
    .plot-frame {{ height:210px; display:flex; align-items:center; justify-content:center; border:1px solid var(--line); border-radius:8px; background:var(--panel-frame); overflow:hidden; }}
    .plot-card.card-lg .plot-frame {{ height:390px; }}
    .plot-card.card-wide .plot-frame {{ height:260px; }}
    .plot-frame img {{ width:100%; height:100%; object-fit:contain; display:block; }}
    .preview-toolbar {{ display:flex; gap:7px; align-items:center; flex-wrap:wrap; margin-top:10px; }}
    .preview-button {{ border:1px solid var(--line); border-radius:7px; background:var(--panel-frame); color:var(--text); padding:6px 9px; cursor:pointer; font-size:.84rem; }}
    .preview-button:hover, .preview-button.active {{ border-color:var(--pill-border); background:var(--pill-bg); }}
    .preview-button:disabled {{ opacity:.45; cursor:not-allowed; }}
    .preview-note {{ margin-top:8px; color:var(--muted); font-size:.86rem; }}
    .protein-preview-card .plot-title-row,
    .protein-preview-card .preview-toolbar,
    .protein-preview-card .preview-note {{
      position:relative;
      z-index:2;
    }}
    .protein-preview-card {{
      max-height:520px;
      overflow:hidden;
      min-width:0;
    }}
    .protein-preview-frame {{
      height:340px;
      max-height:340px;
      position:relative;
      z-index:1;
      display:flex;
      align-items:center;
      justify-content:center;
      overflow:hidden;
      border:1px solid var(--line);
      border-radius:8px;
      background:var(--panel-frame);
      margin-top:12px;
    }}
    .protein-preview-frame img {{
      width:100%;
      height:100%;
      object-fit:contain;
      display:block;
    }}
    .protein-view {{ width:100%; height:340px; max-height:340px; }}
    .protein-view.hidden {{ display:none; }}
    #proteinViewer, #protein-viewer, #protein-3dmol-viewer {{ width:100%; height:340px; max-height:340px; }}
    .viewer-3dmol {{ width:100%; height:340px; max-height:340px; display:block; background:#08111f; }}
    .viewer-canvas {{ width:100%; height:340px; max-height:340px; display:block; cursor:grab; background:radial-gradient(circle at 50% 42%, #162a42, #071321 70%); }}
    .viewer-canvas.dragging {{ cursor:grabbing; }}
    .protein-preview-card.expanded {{
      position:fixed;
      inset:24px;
      z-index:30;
      max-height:none;
      overflow:auto;
      box-shadow:0 26px 80px rgba(0,0,0,.55);
    }}
    .protein-preview-card.expanded .protein-preview-frame {{
      height:min(72vh,720px);
      max-height:min(72vh,720px);
    }}
    .protein-preview-card.expanded .protein-view,
    .protein-preview-card.expanded #protein-viewer,
    .protein-preview-card.expanded #protein-3dmol-viewer,
    .protein-preview-card.expanded .viewer-3dmol,
    .protein-preview-card.expanded .viewer-canvas {{
      height:100%;
      max-height:none;
    }}
    .protein-preview-card:not(.expanded) #protein-collapse {{ display:none; }}
    body.protein-preview-expanded {{ overflow:hidden; }}
    .plot-footer {{
      margin-top:auto;
      display:grid;
      gap:4px;
      min-width:0;
      max-width:100%;
      color:var(--muted);
      font-size:.82rem;
      line-height:1.35;
    }}
    .plot-summary, .plot-category {{
      min-width:0;
      max-width:100%;
      overflow-wrap:anywhere;
    }}
    .plot-card.card-sm .plot-summary,
    .plot-card.card-sm .plot-category {{
      display:none;
    }}
    .phase-list {{ display:grid; gap:10px; }}
    .phase-row {{ display:grid; grid-template-columns:1fr auto; gap:12px; align-items:center; padding:10px 12px; border:1px solid var(--line); border-radius:8px; background:var(--panel-frame); }}
    .table {{ display:grid; gap:8px; }}
    .table-row {{ display:grid; grid-template-columns:minmax(130px,.35fr) minmax(0,1fr); gap:12px; padding:9px 0; border-bottom:1px solid var(--line-soft); }}
    .category-heading {{ margin:18px 0 10px; color:var(--muted); font-size:.82rem; text-transform:uppercase; letter-spacing:.06em; }}
    .quick-links {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:10px; }}
    .config-form {{ display:grid; gap:16px; max-width:1100px; }}
    .form-section {{ border:1px solid var(--line); border-radius:14px; background:var(--panel); box-shadow:0 1px 2px var(--shadow); padding:18px 18px 20px; }}
    .form-section-head {{ margin-bottom:14px; }}
    .form-section-head h2 {{ margin:0 0 2px; font-size:1rem; }}
    .form-section-head p {{ margin:0; font-size:.84rem; }}
    .form-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:16px 18px; }}
    .field {{ display:grid; gap:7px; align-content:start; min-width:0; }}
    .field-wide {{ grid-column:1/-1; }}
    .field > label {{ font-size:.82rem; font-weight:600; color:var(--text); }}
    .field-help {{ font-size:.74rem; color:var(--faint); }}
    .field input[type=text], .field input[type=number], .field select {{
      font:inherit; color:var(--text); background:var(--panel-frame); border:1px solid var(--line);
      border-radius:9px; padding:9px 11px; width:100%; min-width:0;
    }}
    .field input[type=number] {{ font-family:var(--mono); }}
    .field input:focus, .field select:focus {{ outline:none; border-color:var(--accent); box-shadow:0 0 0 3px var(--pill-bg); }}
    .input-row {{ display:flex; align-items:center; gap:8px; }}
    .input-row .unit {{ color:var(--faint); font-size:.8rem; white-space:nowrap; }}
    .range-row {{ display:flex; align-items:center; gap:12px; }}
    .range {{ -webkit-appearance:none; appearance:none; height:6px; border-radius:99px; background:var(--line); flex:1 1 auto; }}
    .range::-webkit-slider-thumb {{ -webkit-appearance:none; width:18px; height:18px; border-radius:50%; background:var(--accent); border:2px solid #fff; box-shadow:0 1px 3px var(--shadow); cursor:pointer; }}
    .range::-moz-range-thumb {{ width:18px; height:18px; border-radius:50%; background:var(--accent); border:2px solid #fff; cursor:pointer; }}
    .range-ph {{ background:linear-gradient(90deg,#e8663c,#e6b93c,#3fa653,#3f7bd6,#7b5cd6); }}
    .range-out {{ font-family:var(--mono); font-variant-numeric:tabular-nums; font-size:.86rem; min-width:64px; text-align:right; color:var(--text); }}
    .range-scale {{ display:flex; justify-content:space-between; font-size:.68rem; color:var(--faint); margin-top:2px; }}
    .toggle {{ display:inline-flex; align-items:center; cursor:pointer; }}
    .toggle input {{ position:absolute; opacity:0; }}
    .toggle .track {{ width:42px; height:24px; border-radius:99px; background:var(--line); position:relative; transition:background .18s ease; }}
    .toggle .track::after {{ content:""; position:absolute; top:3px; left:3px; width:18px; height:18px; border-radius:50%; background:#fff; box-shadow:0 1px 2px var(--shadow); transition:transform .18s ease; }}
    .toggle input:checked + .track {{ background:var(--accent); }}
    .toggle input:checked + .track::after {{ transform:translateX(18px); }}
    .chip-checks {{ display:flex; gap:8px; flex-wrap:wrap; }}
    .chip-check {{ display:inline-flex; align-items:center; gap:7px; border:1px solid var(--line); border-radius:99px; padding:6px 12px; cursor:pointer; font-size:.84rem; background:var(--panel-frame); }}
    .chip-check input {{ accent-color:var(--accent); }}
    .form-actions {{ display:flex; gap:10px; justify-content:flex-end; align-items:center; }}
    .button.primary {{ background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600; }}
    .button.primary:hover {{ background:var(--accent-ink); border-color:var(--accent-ink); }}
    .button.ghost {{ background:transparent; }}
    .button:disabled {{ opacity:.5; cursor:not-allowed; }}
    .run-result {{ border:1px solid var(--line); border-radius:10px; padding:12px 14px; font-size:.86rem; }}
    .run-result.ok {{ border-color:var(--pill-border); background:var(--pill-bg); }}
    .run-result.bad {{ border-color:rgba(209,67,67,.4); background:rgba(209,67,67,.08); }}
    .run-result pre {{ margin:8px 0 0; font-family:var(--mono); font-size:.78rem; white-space:pre-wrap; overflow-wrap:anywhere; }}
    @media (max-width: 1000px) {{
      .live-grid {{ grid-template-columns:1fr; }}
      .live-bottom-grid {{ grid-template-columns:1fr; }}
      .protein-preview-card {{ max-height:500px; }}
      .protein-preview-frame {{ height:320px; max-height:320px; }}
      .protein-view, #proteinViewer, #protein-viewer, #protein-3dmol-viewer, .viewer-3dmol, .viewer-canvas {{ height:320px; max-height:320px; }}
    }}
    @media (max-width: 800px) {{ .layout {{ grid-template-columns:1fr; }} aside {{ border-right:0; border-bottom:1px solid var(--line); }} .grid {{ grid-template-columns:1fr; }} .header {{ display:block; }} }}
  </style>
</head>
<body>
<div class="layout">
  <aside>
    <div class="brand-row">
      <img class="brand-logo" alt="AAI Research Lab" src="/static/aai_logo.png">
      <div><div class="brand">FastMDXplora</div><div class="subtle" style="font-size:.72rem">Live dashboard</div></div>
    </div>
    <div class="nav-heading">Setup</div>
    <a href="#configure" class="nav-link" data-view-link="configure">Configure &amp; Run</a>
    <div class="nav-heading">Overview</div>
    <a href="#dashboard" class="nav-link active" data-view-link="dashboard">Dashboard</a>
    <a href="#live" class="nav-link" data-view-link="live">Live Simulation</a>
    <a href="#analysis-plots" class="nav-link" data-view-link="analysis-plots">Analysis Plots</a>
    <a href="#generated-files" class="nav-link" data-view-link="generated-files">Generated Files</a>
    <a href="#system-info" class="nav-link" data-view-link="system-info">System Info</a>
    <a href="#run-status" class="nav-link" data-view-link="run-status">Run Status</a>
  </aside>
  <main>
    <section id="configure" class="view">
      <div class="header">
        <div><h1>Configure &amp; Run</h1><div class="subtle">Set parameters mapped to FastMDXplora's config, save, and launch a run.</div></div>
      </div>
      <div class="panel result-state">Values map directly to the FastMDXplora YAML schema. Save writes <code>ui_run_config.yml</code>; Run launches <code>fastmdx explore</code> with live telemetry.</div>
      {config_form_html}
    </section>
    <section id="dashboard" class="view active">
      <div class="header">
        <div><h1>Dashboard</h1><div class="subtle">Completed results and generated files</div></div>
        <div class="toolbar">
          <button class="button" type="button" id="refresh-results">Refresh</button>
          <span class="muted">Last refreshed: <span id="results-refreshed">not available</span></span>
        </div>
      </div>
      <div class="panel result-state" id="results-state">Analysis outputs not available yet. They will appear here after analysis/report finishes.</div>
      <div class="cards" id="summary-cards"></div>
      <div class="grid">
        <div class="panel"><h2>Phase Summary</h2><div id="dashboard-phase-list" class="phase-list"></div></div>
        <div class="panel"><h2>Live Status Summary</h2><div id="dashboard-live-summary" class="table"></div></div>
      </div>
      <div class="panel" style="margin-top:14px">
        <h2>Quick Actions</h2>
        <div id="overview-links" class="quick-links"></div>
      </div>
    </section>
    <section id="live" class="view">
      <div class="header">
        <div><h1>Live Simulation</h1><div class="subtle">Watching {output_label}</div></div>
        <div class="badge"><span id="status-dot" class="dot"></span><span id="status-text">not available</span></div>
      </div>
      <div class="cards">
        <div class="card"><div class="label">Current stage</div><div class="value" id="stage">not available</div></div>
        <div class="card"><div class="label">Current step</div><div class="value" id="step">not available</div></div>
        <div class="card"><div class="label">Frames</div><div class="value" id="frames">not available</div></div>
        <div class="card"><div class="label">Simulation time</div><div class="value" id="sim-time">not available</div></div>
        <div class="card"><div class="label">Elapsed wall time</div><div class="value" id="elapsed">not available</div></div>
        <div class="card"><div class="label">Platform</div><div class="value" id="platform">not available</div></div>
      </div>
      <div class="live-grid">
        <div class="live-main-column">
          <div class="panel">
            <h2>Progress</h2><div class="progress-track"><div id="progress-fill" class="progress-fill"></div></div>
            <div class="muted" id="progress-label">not available</div>
          </div>
          <div class="panel"><h2>Energy / Temperature Trends</h2><canvas id="metrics-chart" width="900" height="260"></canvas><div class="muted" id="chart-empty">not available yet</div></div>
        </div>
        <div class="panel protein-preview-card" id="protein-preview-card">
        <div class="plot-title-row">
          <div><h2>Protein Preview</h2><span class="badge" id="protein-preview-mode">not available</span></div>
          <div class="toolbar">
            <button class="button" type="button" id="protein-expand" onclick="setProteinPreviewExpanded(true)">Expand</button>
            <button class="button" type="button" id="protein-collapse" onclick="setProteinPreviewExpanded(false)">Close</button>
          </div>
        </div>
        <div class="preview-toolbar" aria-label="Protein preview controls">
          <button class="preview-button active" type="button" id="protein-view-static">PyMOL Preview</button>
          <button class="preview-button" type="button" id="protein-view-3d">Interactive 3D</button>
          <button class="preview-button" type="button" id="protein-spin">Spin</button>
          <button class="preview-button" type="button" id="protein-reset">Reset view</button>
          <button class="preview-button" type="button" id="regenerate-preview">Regenerate preview</button>
        </div>
        <div class="preview-note" id="protein-preview-note">Interactive 3Dmol viewer will appear when a structure is available.</div>
        <div class="protein-preview-frame" id="protein-preview-frame">
          <div class="protein-view" id="protein-viewer-wrap">
            <div class="viewer-3dmol" id="protein-3dmol-viewer" aria-label="Interactive 3Dmol protein viewer"></div>
          </div>
          <div class="protein-view hidden" id="protein-fallback-wrap">
            <canvas class="viewer-canvas" id="protein-viewer" width="900" height="520" aria-label="Interactive protein viewer"></canvas>
          </div>
          <div class="protein-view hidden" id="protein-static-wrap">
            <p class="muted" id="protein-preview-message">No topology/PDB found yet.</p>
          </div>
        </div>
        </div>
      </div>
      <div class="live-bottom-grid">
        <div class="panel"><h2>Health</h2><p id="health-message">not available</p><p class="muted" id="health-explanation">not available</p></div>
        <div class="panel"><h2>Recent events</h2><div id="events" class="events"></div></div>
      </div>
    </section>
    <section id="analysis-plots" class="view">
      <div class="header"><div><h1>Analysis Plots</h1><div class="subtle">Dashboard-native assets are used when available.</div></div></div>
      <div id="analysis-plot-groups"></div>
    </section>
    <section id="generated-files" class="view">
      <div class="header"><div><h1>Generated Files</h1><div class="subtle">Report outputs, manifests, and important artifacts</div></div></div>
      <div class="panel" style="margin-bottom:14px"><h2>Report Artifacts</h2><div id="report-artifact-list" class="artifact-list"></div></div>
      <div class="panel"><h2>All Generated Files</h2><div id="artifact-list" class="artifact-list"></div></div>
    </section>
    <section id="system-info" class="view">
      <div class="header"><h1>System Info</h1></div>
      <div class="panel"><div class="table" id="system-info-panel"></div></div>
    </section>
    <section id="run-status" class="view">
      <div class="header"><h1>Run Status</h1></div>
      <div class="panel"><h2>Manifest Phases</h2><div id="run-status-panel" class="phase-list"></div></div>
      <div class="panel" style="margin-top:14px"><h2>Live Telemetry Status</h2><div id="live-status-panel" class="table"></div></div>
    </section>
  </main>
</div>
<script src="/static/3Dmol-min.js"></script>
<script>
const fmt = (v, fallback="not available") => (v === null || v === undefined || v === "" ? fallback : String(v));
const setText = (id, value) => {{ document.getElementById(id).textContent = fmt(value); }};
function showView(name) {{
  document.querySelectorAll(".view").forEach(el => el.classList.toggle("active", el.id === name));
  document.querySelectorAll("[data-view-link]").forEach(el => el.classList.toggle("active", el.dataset.viewLink === name));
}}
document.querySelectorAll("[data-view-link]").forEach(el => el.addEventListener("click", event => {{ event.preventDefault(); showView(el.dataset.viewLink); }}));
async function fetchJson(url) {{ const res = await fetch(url, {{cache:"no-store"}}); return await res.json(); }}
function progressFrom(status, metrics) {{
  const latest = metrics.length ? metrics[metrics.length - 1] : {{}};
  let pct = Number(latest.progress_percent ?? status.progress_percent);
  if (!Number.isFinite(pct) && status.current_step && status.total_planned_steps) pct = Number(status.current_step) / Number(status.total_planned_steps) * 100;
  return Number.isFinite(pct) ? Math.max(0, Math.min(100, pct)) : null;
}}
function updateCards(status, health, metrics) {{
  const dot = document.getElementById("status-dot");
  dot.className = "dot " + fmt(health.state, "unknown");
  setText("status-text", health.state || status.status);
  setText("stage", status.stage);
  setText("step", status.current_step && status.total_planned_steps ? `${{status.current_step}} / ${{status.total_planned_steps}}` : status.current_step);
  setText("frames", status.current_frame_count && status.planned_frame_count ? `${{status.current_frame_count}} / ${{status.planned_frame_count}}` : status.current_frame_count);
  setText("sim-time", status.simulation_time_completed_ns ? `${{status.simulation_time_completed_ns}} ns` : null);
  setText("elapsed", status.elapsed_wall_time_s ? `${{status.elapsed_wall_time_s}} s` : null);
  setText("platform", status.platform);
  const pct = progressFrom(status, metrics);
  document.getElementById("progress-fill").style.width = pct === null ? "0%" : pct.toFixed(1) + "%";
  setText("progress-label", pct === null ? null : pct.toFixed(1) + "% complete");
  setText("health-message", health.message);
  setText("health-explanation", health.explanation);
  document.getElementById("live-status-panel").innerHTML = tableRows({{
    status: status.status,
    stage: status.stage,
    platform: status.platform,
    current_step: status.current_step,
    checkpoint: status.current_checkpoint_path,
    last_update: status.last_update_timestamp
  }});
  document.getElementById("dashboard-live-summary").innerHTML = tableRows({{
    status: status.status,
    stage: status.stage,
    current_step: status.current_step,
    platform: status.platform,
    health: health.state
  }});
}}
function drawChart(metrics) {{
  const canvas = document.getElementById("metrics-chart");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!metrics.length) {{ document.getElementById("chart-empty").style.display = "block"; return; }}
  document.getElementById("chart-empty").style.display = "none";
  const series = [
    {{field:"total_energy", color:"#39b7c9"}},
    {{field:"temperature", color:"#efb35e"}}
  ].map(spec => ({{...spec, values: metrics.map(r => Number(r[spec.field])).filter(Number.isFinite)}})).filter(s => s.values.length);
  if (!series.length) {{ document.getElementById("chart-empty").style.display = "block"; return; }}
  const all = series.flatMap(s => s.values);
  const min = Math.min(...all), max = Math.max(...all), span = max === min ? 1 : max - min;
  ctx.strokeStyle = "var(--line)"; ctx.lineWidth = 1;
  for (let y=30; y<canvas.height-25; y+=45) {{ ctx.beginPath(); ctx.moveTo(42,y); ctx.lineTo(canvas.width-16,y); ctx.stroke(); }}
  for (const s of series) {{
    ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.beginPath();
    s.values.forEach((v, i) => {{
      const x = 42 + (canvas.width - 64) * (i / Math.max(1, s.values.length - 1));
      const y = 18 + (canvas.height - 46) * (1 - ((v - min) / span));
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }});
    ctx.stroke();
  }}
}}
async function poll() {{
  try {{
    const [statusPayload, metricsPayload, eventsPayload] = await Promise.all([
      fetchJson("/api/status"), fetchJson("/api/metrics"), fetchJson("/api/events")
    ]);
    const status = statusPayload.status || {{}};
    const metrics = metricsPayload.metrics || [];
    updateCards(status, statusPayload.health || {{}}, metrics);
    drawChart(metrics);
    document.getElementById("events").innerHTML = (eventsPayload.events || []).slice(-20).reverse().map(e => `<div class="event"><strong>${{e.level}}</strong> ${{e.message}}<div class="muted">${{e.timestamp}}</div></div>`).join("") || '<p class="muted">not available</p>';
  }} catch (err) {{
    setText("health-message", "Could not read live dashboard data");
    setText("health-explanation", String(err));
  }}
}}
async function refreshProteinPreview(regenerate = false) {{
  const payload = await fetchJson("/api/protein-preview" + (regenerate ? "?regenerate=1" : ""));
  proteinPayload = payload || {{}};
  const staticWrap = document.getElementById("protein-static-wrap");
  const viewerButton = document.getElementById("protein-view-3d");
  const staticButton = document.getElementById("protein-view-static");
  const imageUrl = payload.static_image_url || payload.image_url || payload.href;
  if (payload.static_available && imageUrl) {{
    staticButton.disabled = false;
    staticWrap.innerHTML = '<a href="' + imageUrl + '"><img src="' + imageUrl + '" alt="PyMOL protein preview"></a>';
  }} else {{
    staticButton.disabled = true;
    staticWrap.innerHTML = '<p class="muted" id="protein-preview-message">' + (payload.message || "PyMOL preview unavailable. Showing schematic fallback.") + '</p>';
  }}
  if (payload.viewer_available && payload.viewer_mode === "3dmol" && payload.structure_url) {{
    viewerButton.disabled = false;
    await load3DmolViewer(payload.structure_url);
  }} else if (payload.fallback_available && payload.structure_url) {{
    viewerButton.disabled = false;
    await loadSchematicViewer(payload.structure_url);
  }} else {{
    viewerButton.disabled = true;
  }}
  const activeMode = currentProteinPreviewMode();
  if (payload.viewer_available && payload.viewer_mode === "3dmol" && (activeMode === "empty" || regenerate)) {{
    showProteinPreviewMode("viewer");
  }} else if (payload.static_available && imageUrl && (activeMode === "empty" || regenerate)) {{
    showProteinPreviewMode("static");
  }} else if (!payload.static_available && payload.fallback_available) {{
    showProteinPreviewMode("fallback");
  }} else {{
    showProteinPreviewMode(activeMode === "empty" ? "static" : activeMode);
  }}
}}
function currentProteinPreviewMode() {{
  if (!document.getElementById("protein-viewer-wrap").classList.contains("hidden")) return "viewer";
  if (!document.getElementById("protein-fallback-wrap").classList.contains("hidden")) return "fallback";
  if (!document.getElementById("protein-static-wrap").classList.contains("hidden")) return "static";
  return "empty";
}}
function proteinPreviewLabel(mode) {{
  if (mode === "viewer") {{
    if (proteinPayload.viewer_mode === "3dmol") return "Interactive 3Dmol viewer";
    if (proteinPayload.viewer_mode === "ngl") return "NGL viewer";
    return "interactive viewer";
  }}
  if (mode === "fallback") {{
    return "schematic fallback";
  }}
  if (proteinPayload.static_available) {{
    return proteinPayload.static_mode === "pymol" ? "PyMOL render" : "static preview";
  }}
  return "not available";
}}
function proteinPreviewNote(mode) {{
  if (mode === "viewer") {{
    return proteinPayload.viewer_mode === "3dmol"
      ? "Interactive 3Dmol cartoon viewer. Drag to rotate, scroll to zoom, or use Spin."
      : "Interactive molecular viewer.";
  }}
  if (mode === "fallback") {{
    return "Schematic fallback CA/backbone trace. This is not a PyMOL render or full molecular viewer.";
  }}
  if (proteinPayload.static_available) {{
    return proteinPayload.static_mode === "pymol"
      ? "Showing the PyMOL-rendered cartoon/ribbon PNG."
      : "Showing the available static protein preview image.";
  }}
  return proteinPayload.message || "PyMOL preview unavailable. Showing schematic fallback.";
}}
function showProteinPreviewMode(mode) {{
  const viewerWrap = document.getElementById("protein-viewer-wrap");
  const fallbackWrap = document.getElementById("protein-fallback-wrap");
  const staticWrap = document.getElementById("protein-static-wrap");
  const view3d = document.getElementById("protein-view-3d");
  const viewStatic = document.getElementById("protein-view-static");
  const requestedViewer = mode === "viewer";
  const requestedFallback = mode === "fallback";
  const useViewer = requestedViewer && !view3d.disabled && proteinPayload.viewer_mode === "3dmol";
  const useFallback = (requestedFallback || requestedViewer) && !useViewer && !view3d.disabled && proteinPayload.fallback_available;
  const useStatic = !useViewer && !useFallback;
  viewerWrap.classList.toggle("hidden", !useViewer);
  fallbackWrap.classList.toggle("hidden", !useFallback);
  staticWrap.classList.toggle("hidden", !useStatic);
  view3d.classList.toggle("active", useViewer || useFallback);
  viewStatic.classList.toggle("active", useStatic);
  const active = useViewer ? "viewer" : useFallback ? "fallback" : "static";
  setText("protein-preview-mode", proteinPreviewLabel(active));
  setText("protein-preview-note", proteinPreviewNote(active));
  if (useViewer) {{
    render3DmolViewer();
  }} else if (useFallback) {{
    drawSchematicViewer();
  }}
}}
let proteinPayload = {{}};
let proteinViewerSource = "";
let proteinViewer3Dmol = null;
let proteinViewer3DmolSource = "";
let proteinViewer3DmolPdb = "";
let proteinViewerSpinning = false;
const proteinFallbackViewer = {{
  points: [],
  angleX: -0.4,
  angleY: 0.7,
  zoom: 1,
  dragging: false,
  lastX: 0,
  lastY: 0,
  animation: null
}};
async function load3DmolViewer(url) {{
  if (proteinViewer3DmolSource === url && proteinViewer3Dmol) {{
    render3DmolViewer();
    return;
  }}
  const res = await fetch(url, {{cache:"no-store"}});
  proteinViewer3DmolPdb = await res.text();
  proteinViewer3DmolSource = url;
  if (!window.$3Dmol || !proteinViewer3DmolPdb.trim()) {{
    proteinPayload.viewer_available = false;
    proteinPayload.viewer_mode = null;
    if (proteinPayload.fallback_available) await loadSchematicViewer(url);
    return;
  }}
  const element = document.getElementById("protein-3dmol-viewer");
  element.innerHTML = "";
  proteinViewer3Dmol = $3Dmol.createViewer(element, {{backgroundColor:"#08111f"}});
  proteinViewer3Dmol.addModel(proteinViewer3DmolPdb, "pdb");
  proteinViewer3Dmol.setStyle({{}}, {{cartoon: {{color:"spectrum"}}}});
  proteinViewer3Dmol.zoomTo();
  proteinViewer3Dmol.render();
}}
function render3DmolViewer() {{
  if (!proteinViewer3Dmol) return;
  proteinViewer3Dmol.resize();
  proteinViewer3Dmol.zoomTo();
  proteinViewer3Dmol.render();
}}
async function loadSchematicViewer(url) {{
  if (proteinViewerSource === url && proteinFallbackViewer.points.length) {{
    drawSchematicViewer();
    return;
  }}
  const res = await fetch(url, {{cache:"no-store"}});
  const text = await res.text();
  const points = parsePdbTrace(text);
  proteinViewerSource = url;
  proteinFallbackViewer.points = points;
  if (!points.length) {{
    document.getElementById("protein-view-3d").disabled = true;
    if (!proteinPayload.static_available) {{
      document.getElementById("protein-static-wrap").innerHTML = '<p class="muted">Structure loaded, but no CA atoms were found for the schematic viewer.</p>';
    }}
    return;
  }}
  drawSchematicViewer();
}}
function parsePdbTrace(text) {{
  const points = [];
  for (const line of text.split(/\\r?\\n/)) {{
    if (!line.startsWith("ATOM") && !line.startsWith("HETATM")) continue;
    const atom = line.slice(12, 16).trim();
    if (atom !== "CA") continue;
    const x = Number(line.slice(30, 38));
    const y = Number(line.slice(38, 46));
    const z = Number(line.slice(46, 54));
    if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) points.push({{x, y, z}});
  }}
  return points;
}}
function drawSchematicViewer() {{
  const canvas = document.getElementById("protein-viewer");
  if (!canvas || !proteinFallbackViewer.points.length) return;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const width = rect.width;
  const height = rect.height;
  ctx.clearRect(0, 0, width, height);
  const projected = projectProteinPoints(width, height);
  ctx.lineWidth = 3;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  for (let i = 1; i < projected.length; i += 1) {{
    const prev = projected[i - 1];
    const curr = projected[i];
    ctx.strokeStyle = residueColor(i, projected.length);
    ctx.beginPath();
    ctx.moveTo(prev.x, prev.y);
    ctx.lineTo(curr.x, curr.y);
    ctx.stroke();
  }}
  for (let i = 0; i < projected.length; i += 1) {{
    const point = projected[i];
    ctx.fillStyle = residueColor(i, projected.length);
    ctx.beginPath();
    ctx.arc(point.x, point.y, Math.max(2.2, 4.5 * point.depth), 0, Math.PI * 2);
    ctx.fill();
  }}
}}
function projectProteinPoints(width, height) {{
  const pts = proteinFallbackViewer.points;
  const center = pts.reduce((acc, p) => ({{x: acc.x + p.x, y: acc.y + p.y, z: acc.z + p.z}}), {{x:0, y:0, z:0}});
  center.x /= pts.length; center.y /= pts.length; center.z /= pts.length;
  const sinX = Math.sin(proteinFallbackViewer.angleX), cosX = Math.cos(proteinFallbackViewer.angleX);
  const sinY = Math.sin(proteinFallbackViewer.angleY), cosY = Math.cos(proteinFallbackViewer.angleY);
  const rotated = pts.map(p => {{
    const x0 = p.x - center.x, y0 = p.y - center.y, z0 = p.z - center.z;
    const x1 = x0 * cosY + z0 * sinY;
    const z1 = -x0 * sinY + z0 * cosY;
    const y1 = y0 * cosX - z1 * sinX;
    const z2 = y0 * sinX + z1 * cosX;
    return {{x:x1, y:y1, z:z2}};
  }});
  const maxRadius = Math.max(1, ...rotated.map(p => Math.hypot(p.x, p.y)));
  const scale = Math.min(width, height) * 0.38 * proteinFallbackViewer.zoom / maxRadius;
  const zMin = Math.min(...rotated.map(p => p.z));
  const zMax = Math.max(...rotated.map(p => p.z));
  const zSpan = zMax === zMin ? 1 : zMax - zMin;
  return rotated.map(p => ({{
    x: width / 2 + p.x * scale,
    y: height / 2 - p.y * scale,
    depth: 0.55 + 0.45 * ((p.z - zMin) / zSpan)
  }}));
}}
function residueColor(index, total) {{
  const hue = Math.round(210 + (index / Math.max(1, total - 1)) * 130);
  return `hsl(${{hue}}, 84%, 62%)`;
}}
function animateProteinSpin() {{
  if (!proteinViewerSpinning) return;
  proteinFallbackViewer.angleY += 0.012;
  drawSchematicViewer();
  proteinFallbackViewer.animation = requestAnimationFrame(animateProteinSpin);
}}
function setProteinSpin(active) {{
  proteinViewerSpinning = active;
  document.getElementById("protein-spin").classList.toggle("active", active);
  if (proteinViewer3Dmol && currentProteinPreviewMode() === "viewer") {{
    proteinViewer3Dmol.spin(active);
    proteinViewer3Dmol.render();
    return;
  }}
  if (active) animateProteinSpin();
  else if (proteinFallbackViewer.animation) cancelAnimationFrame(proteinFallbackViewer.animation);
}}
function resetProteinViewer() {{
  if (proteinViewer3Dmol && currentProteinPreviewMode() === "viewer") {{
    proteinViewer3Dmol.spin(false);
    proteinViewerSpinning = false;
    document.getElementById("protein-spin").classList.remove("active");
    proteinViewer3Dmol.zoomTo();
    proteinViewer3Dmol.render();
    return;
  }}
  proteinFallbackViewer.angleX = -0.4;
  proteinFallbackViewer.angleY = 0.7;
  proteinFallbackViewer.zoom = 1;
  drawSchematicViewer();
}}
function bindProteinViewerControls() {{
  const canvas = document.getElementById("protein-viewer");
  canvas.addEventListener("mousedown", event => {{
    proteinFallbackViewer.dragging = true;
    proteinFallbackViewer.lastX = event.clientX;
    proteinFallbackViewer.lastY = event.clientY;
    canvas.classList.add("dragging");
  }});
  window.addEventListener("mouseup", () => {{ proteinFallbackViewer.dragging = false; canvas.classList.remove("dragging"); }});
  window.addEventListener("mousemove", event => {{
    if (!proteinFallbackViewer.dragging) return;
    proteinFallbackViewer.angleY += (event.clientX - proteinFallbackViewer.lastX) * 0.01;
    proteinFallbackViewer.angleX += (event.clientY - proteinFallbackViewer.lastY) * 0.01;
    proteinFallbackViewer.lastX = event.clientX;
    proteinFallbackViewer.lastY = event.clientY;
    drawSchematicViewer();
  }});
  canvas.addEventListener("wheel", event => {{
    event.preventDefault();
    proteinFallbackViewer.zoom = Math.max(0.45, Math.min(3, proteinFallbackViewer.zoom + (event.deltaY < 0 ? 0.08 : -0.08)));
    drawSchematicViewer();
  }}, {{passive:false}});
  document.getElementById("protein-view-3d").addEventListener("click", () => showProteinPreviewMode("viewer"));
  document.getElementById("protein-view-static").addEventListener("click", () => showProteinPreviewMode("static"));
  document.getElementById("protein-spin").addEventListener("click", () => setProteinSpin(!proteinViewerSpinning));
  document.getElementById("protein-reset").addEventListener("click", resetProteinViewer);
  document.getElementById("protein-expand").addEventListener("click", () => setProteinPreviewExpanded(true));
  document.getElementById("protein-collapse").addEventListener("click", () => setProteinPreviewExpanded(false));
}}
function setProteinPreviewExpanded(expanded) {{
  const card = document.getElementById("protein-preview-card");
  card.classList.toggle("expanded", expanded);
  document.body.classList.toggle("protein-preview-expanded", expanded);
  document.getElementById("protein-expand").textContent = expanded ? "Expanded" : "Expand";
  window.setTimeout(() => {{
    if (currentProteinPreviewMode() === "viewer") render3DmolViewer();
    if (currentProteinPreviewMode() === "fallback") drawSchematicViewer();
  }}, 50);
}}
function artifactLink(a) {{
  return `<a class="artifact" href="${{a.href}}" title="${{a.path}}"><span class="artifact-title">${{a.name}}</span><span class="artifact-path">${{a.display_path || a.path}}</span></a>`;
}}
function plotCard(p) {{
  return `<article class="plot-card card-md" data-card-key="${{p.path}}"><div class="plot-title-row"><div><h3 class="plot-title">${{p.title}}</h3><span class="badge">${{p.mode}}</span></div><div class="size-controls" aria-label="Card size"><button type="button" class="size-button" data-card-size="sm">S</button><button type="button" class="size-button active" data-card-size="md">M</button><button type="button" class="size-button" data-card-size="lg">L</button><button type="button" class="size-button" data-card-size="wide">Wide</button></div></div><div class="plot-frame"><a href="${{p.href}}"><img src="${{p.href}}" alt="${{p.title}}"></a></div><div class="plot-footer"><div class="plot-summary">${{p.mode}}</div><div class="plot-category">${{p.category}}</div><a class="artifact-path plot-path" href="${{p.href}}" title="${{p.path}}">${{p.display_path || p.path}}</a></div></article>`;
}}
function summaryCard(card) {{
  return `<article class="card"><div class="label">${{card.label}}</div><div class="value">${{card.value}}</div></article>`;
}}
function phaseRow(row) {{
  return `<div class="phase-row"><strong>${{row.name}}</strong><span class="badge">${{row.status}}</span></div>`;
}}
function tableRows(obj) {{
  return Object.entries(obj || {{}}).map(([key, value]) => `<div class="table-row"><strong>${{key.replaceAll("_", " ")}}</strong><span class="muted">${{fmt(value)}}</span></div>`).join("") || '<p class="muted">not available</p>';
}}
function groupedPlotSections(plots) {{
  if (!plots.length) return '<div class="panel">Analysis outputs not available yet. They will appear here after analysis/report finishes.</div>';
  const groups = {{}};
  for (const plot of plots) {{
    const category = plot.category || "Other";
    if (!groups[category]) groups[category] = [];
    groups[category].push(plot);
  }}
  return Object.entries(groups).map(([category, items]) => `<section><h2 class="category-heading">${{category}}</h2><div class="plot-grid">${{items.map(plotCard).join("")}}</div></section>`).join("");
}}
function bindPlotSizing() {{
  const key = "fastmdx-live-card-layout:" + window.location.pathname;
  let saved = {{}};
  try {{ saved = JSON.parse(localStorage.getItem(key) || "{{}}"); }} catch (_) {{ saved = {{}}; }}
  for (const card of document.querySelectorAll(".plot-card[data-card-key]")) {{
    const stored = saved[card.dataset.cardKey] || "md";
    applyCardSize(card, stored, false);
    for (const button of card.querySelectorAll("[data-card-size]")) {{
      button.onclick = () => applyCardSize(card, button.dataset.cardSize, true);
    }}
  }}
  function applyCardSize(card, size, persist) {{
    for (const name of ["sm", "md", "lg", "wide"]) card.classList.remove("card-" + name);
    card.classList.add("card-" + (size || "md"));
    for (const button of card.querySelectorAll("[data-card-size]")) {{
      button.classList.toggle("active", button.dataset.cardSize === size);
    }}
    if (persist) {{
      saved[card.dataset.cardKey] = size;
      try {{ localStorage.setItem(key, JSON.stringify(saved)); }} catch (_) {{}}
    }}
  }}
}}
function overviewLinks(reports) {{
  const byPath = Object.fromEntries((reports || []).map(item => [item.path, item]));
  const internal = [
    {{label:"Live Simulation", href:"#live", view:"live"}},
    {{label:"Analysis Plots", href:"#analysis-plots", view:"analysis-plots"}},
    {{label:"Generated Files", href:"#generated-files", view:"generated-files"}},
  ];
  const outputLinks = [
    ["Static Dashboard HTML", "report/dashboard.html"],
    ["Markdown Report", "report/report.md"],
    ["Slides PPTX", "report/slides.pptx"],
    ["Bundle", "report/project_bundle.zip"],
  ].map(([label, path]) => byPath[path] ? {{label, href: byPath[path].href, path}} : null).filter(Boolean);
  return [...internal, ...outputLinks].map(link => `<a class="artifact" href="${{link.href}}" data-view-target="${{link.view || ""}}" title="${{link.path || link.label}}"><span class="artifact-title">${{link.label}}</span><span class="artifact-subtitle">${{link.path || "Open tab"}}</span></a>`).join("");
}}
async function refreshResults() {{
  const payload = await fetchJson("/api/results");
  const artifacts = payload.artifacts || [];
  const plots = payload.plots || [];
  const reports = payload.reports || [];
  setText("results-refreshed", payload.refreshed_at);
  const state = document.getElementById("results-state");
  if (!payload.has_analysis && !payload.has_report) {{
    state.style.display = "block";
    state.textContent = "Analysis outputs not available yet. They will appear here after analysis/report finishes.";
  }} else {{
    state.style.display = "none";
  }}
  document.getElementById("summary-cards").innerHTML = (payload.summary || []).map(summaryCard).join("");
  document.getElementById("dashboard-phase-list").innerHTML = (payload.phases || []).map(phaseRow).join("");
  document.getElementById("analysis-plot-groups").innerHTML = groupedPlotSections(plots);
  bindPlotSizing();
  document.getElementById("overview-links").innerHTML = overviewLinks(reports);
  for (const link of document.querySelectorAll("[data-view-target]")) {{
    if (link.dataset.viewTarget) link.onclick = (event) => {{ event.preventDefault(); showView(link.dataset.viewTarget); }};
  }}
  document.getElementById("report-artifact-list").innerHTML = reports.map(artifactLink).join("") || '<p class="muted">Report artifacts not available yet.</p>';
  document.getElementById("artifact-list").innerHTML = artifacts.map(artifactLink).join("") || '<p class="muted">No generated files found yet.</p>';
  document.getElementById("system-info-panel").innerHTML = tableRows(payload.system || {{}});
  document.getElementById("run-status-panel").innerHTML = (payload.phases || []).map(phaseRow).join("") || '<p class="muted">not available</p>';
}}
document.getElementById("refresh-results").addEventListener("click", () => refreshResults().catch(err => setText("results-state", String(err))));
document.getElementById("regenerate-preview").addEventListener("click", () => refreshProteinPreview(true).catch(err => setText("protein-preview-message", String(err))));
bindProteinViewerControls();
poll(); refreshResults(); refreshProteinPreview();
setInterval(poll, 3000);
setInterval(() => refreshResults().catch(() => {{}}), 5000);
setInterval(() => refreshProteinPreview().catch(() => {{}}), 5000);

// --- Configure tab: range displays + save/run ---
(function initConfigForm() {{
  const form = document.getElementById("config-form");
  if (!form) return;
  form.querySelectorAll(".range").forEach(range => {{
    const out = form.querySelector('output[for="' + range.id + '"]');
    const unit = out ? (out.textContent.replace(/[\\d.\\s-]/g, "") ) : "";
    const sync = () => {{ if (out) out.textContent = range.value + (unit ? " " + unit : ""); }};
    range.addEventListener("input", sync); sync();
  }});
  function collectConfig() {{
    const payload = {{}};
    form.querySelectorAll("[data-cfg-key]").forEach(el => {{
      const key = el.dataset.cfgKey;
      if (el.type === "checkbox") payload[key] = el.checked;
      else payload[key] = el.value;
    }});
    const phases = [];
    form.querySelectorAll("[data-cfg-group]").forEach(el => {{ if (el.checked) phases.push(el.value); }});
    if (phases.length) payload.phases = phases;
    return payload;
  }}
  const result = document.getElementById("run-result");
  function showResult(kind, title, detail) {{
    result.hidden = false;
    result.className = "run-result " + kind;
    result.innerHTML = "<strong>" + title + "</strong>" + (detail ? "<pre>" + detail + "</pre>" : "");
  }}
  async function submit(dryRun, button) {{
    const payload = collectConfig();
    payload.dry_run = dryRun;
    const buttons = form.querySelectorAll(".form-actions .button");
    buttons.forEach(b => b.disabled = true);
    const original = button.textContent;
    button.textContent = dryRun ? "Saving..." : "Launching...";
    try {{
      const res = await fetch("/api/run", {{
        method: "POST", headers: {{"Content-Type": "application/json"}},
        body: JSON.stringify(payload),
      }});
      const body = await res.json();
      if (!res.ok) {{ showResult("bad", "Could not " + (dryRun ? "save" : "start") + ": " + (body.error || res.status)); }}
      else if (body.status === "validated") {{ showResult("ok", "Config saved to " + body.config_path, JSON.stringify(body.config, null, 2)); }}
      else if (body.status === "started") {{ showResult("ok", "Run started (pid " + body.pid + "). Watch the Live Simulation tab.", "config: " + body.config_path + "  •  log: " + body.log); showView("live"); }}
      else {{ showResult("ok", "Done", JSON.stringify(body, null, 2)); }}
    }} catch (err) {{ showResult("bad", "Request failed: " + err); }}
    finally {{ buttons.forEach(b => b.disabled = false); button.textContent = original; }}
  }}
  document.getElementById("save-config").addEventListener("click", ev => submit(true, ev.currentTarget));
  document.getElementById("run-simulation").addEventListener("click", ev => submit(false, ev.currentTarget));
}})();
</script>
</body>
</html>"""
