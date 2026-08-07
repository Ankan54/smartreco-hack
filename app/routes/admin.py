"""Admin catalog management. Requirement 2: a product added here must land in
BOTH the main database and the vector database, and stay in sync as it changes.

Every route below writes through vectors.upsert_product() / delete_product(),
which is the single path that touches all three stores. There is deliberately
no second way to write a product.
"""

import json

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app import auth, config, db, vectors
from app.routes.pages import render, rows_to_products

router = APIRouter(prefix="/admin")


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
