"""Admin catalog management. Requirement 2: a product added here must land in
BOTH the main database and the vector database, and stay in sync as it changes.

Every route below writes through vectors.upsert_product() / delete_product(),
which is the single path that touches all three stores. There is deliberately
no second way to write a product.
"""

import json
import logging

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app import auth, config, db, vectors
from app.routes.pages import render, rows_to_products

log = logging.getLogger(__name__)
router = APIRouter(prefix="/admin")


def _langsmith_client():
    from langsmith import Client
    return Client(api_key=config.LANGSMITH_API_KEY)


def _langsmith_runs(limit: int = 50) -> tuple[list[dict], str | None]:
    """Root traces for the configured project, newest first. Empty + note if
    LangSmith is not configured or the API call fails."""
    if not config.LANGSMITH_API_KEY:
        return [], "Set LANGSMITH_API_KEY to see traces from LangSmith."
    try:
        client = _langsmith_client()
        raw = list(client.list_runs(
            project_name=config.LANGSMITH_PROJECT,
            is_root=True,
            limit=limit,
        ))
        raw.sort(key=lambda r: r.start_time or 0, reverse=True)
        out = []
        for r in raw:
            ms = None
            if r.start_time and r.end_time:
                ms = int((r.end_time - r.start_time).total_seconds() * 1000)
            out.append({
                "id": str(r.id),
                "name": r.name or "trace",
                "status": (r.status or "").lower(),
                "start": r.start_time,
                "ms": ms,
                "tokens": int(getattr(r, "total_tokens", None) or 0),
            })
        return out, None
    except Exception as e:
        log.exception("langsmith list_runs failed")
        return [], f"Could not load LangSmith traces: {e}"


def _public_share_url(run_id: str) -> str:
    """Create (or reuse) a public LangSmith share link for this run."""
    client = _langsmith_client()
    existing = client.read_run_shared_link(run_id)
    if existing:
        return existing
    return client.share_run(run_id)


@router.get("", response_class=HTMLResponse)
def index(request: Request, q: str = "", saved: str = "", user=Depends(auth.require_admin)):
    sql = "SELECT * FROM products"
    args: list = []
    if q:
        sql += " WHERE title LIKE ? OR skills LIKE ?"
        args += [f"%{q}%", f"%{q}%"]
    sql += " ORDER BY updated_at DESC, id DESC LIMIT 100"
    return render(
        request, "admin.html",
        products=rows_to_products(db.q(sql, tuple(args))),
        sync=vectors.sync_status(),
        categories=config.CATEGORIES,
        q=q, saved=saved,
        total=db.q1("SELECT COUNT(*) n FROM products")["n"],
    )


@router.get("/traces", response_class=HTMLResponse)
def traces(request: Request, user=Depends(auth.require_admin)):
    rows, note = _langsmith_runs()
    return render(
        request, "admin_traces.html",
        runs=rows, note=note, project=config.LANGSMITH_PROJECT,
    )


@router.get("/runs", response_class=HTMLResponse)
def runs_redirect(user=Depends(auth.require_admin)):
    return RedirectResponse("/admin/traces", status_code=303)


@router.get("/traces/{run_id}/share")
def share_trace(run_id: str, user=Depends(auth.require_admin)):
    """Mint a public LangSmith share URL, then send the admin there."""
    if not config.LANGSMITH_API_KEY:
        return HTMLResponse("LangSmith is not configured.", status_code=503)
    try:
        return RedirectResponse(_public_share_url(run_id), status_code=303)
    except Exception:
        log.exception("langsmith share_run failed for %s", run_id)
        return HTMLResponse("Could not create a public share link.", status_code=502)


@router.get("/new", response_class=HTMLResponse)
def new_form(request: Request, user=Depends(auth.require_admin)):
    return render(request, "admin_edit.html", p=None, categories=config.CATEGORIES)


@router.get("/{product_id}/edit", response_class=HTMLResponse)
def edit_form(request: Request, product_id: int, user=Depends(auth.require_admin)):
    row = db.q1("SELECT * FROM products WHERE id = ?", (product_id,))
    if not row:
        return HTMLResponse("<h1>Not found</h1>", status_code=404)
    return render(request, "admin_edit.html", p=dict(row), categories=config.CATEGORIES)


@router.post("/save")
def save(
    request: Request,
    product_id: str = Form(""),
    title: str = Form(...),
    description: str = Form(...),
    category: str = Form(...),
    level: str = Form(""),
    price: float = Form(0),
    provider: str = Form(""),
    skills: str = Form(""),
    rating: str = Form(""),
    is_active: str = Form(""),
    user=Depends(auth.require_admin),
):
    pid = int(product_id) if product_id else (
        (db.q1("SELECT COALESCE(MAX(id), 0) m FROM products")["m"]) + 1
    )
    existing = db.q1("SELECT prereq_ids FROM products WHERE id = ?", (pid,))
    product = {
        "id": pid,
        "title": title.strip(),
        "description": description.strip(),
        "category": category,
        "level": level or None,
        "price": price,
        "provider": provider.strip() or None,
        "skills": skills.strip() or None,
        "rating": float(rating) if rating else None,
        "prereq_ids": json.loads(existing["prereq_ids"]) if existing else [],
        "is_active": 1 if is_active else 0,
    }
    embedded = vectors.upsert_product(product)
    note = "embedded" if embedded else "saved (text unchanged, no re-embed)"
    return RedirectResponse(f"/admin?saved={note}", status_code=303)


@router.post("/{product_id}/delete")
def delete(product_id: int, user=Depends(auth.require_admin)):
    vectors.delete_product(product_id)
    return RedirectResponse("/admin?saved=deleted+from+all+three+stores", status_code=303)
