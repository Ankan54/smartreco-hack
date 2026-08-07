import json
import sqlite3

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app import auth, config, db, dossier as dossier_mod

router = APIRouter()
templates = Jinja2Templates(directory=config.ROOT / "app" / "templates")


def render(request: Request, name: str, **ctx) -> HTMLResponse:
    ctx.setdefault("user", auth.current_user(request))
    ctx.setdefault("asset_v", config.ASSET_V)
    return templates.TemplateResponse(request, name, ctx)


def rows_to_products(rows) -> list[dict]:
    out = []
    for r in rows:
        d = dict(r)
        d["skills_list"] = [s.strip() for s in (d.get("skills") or "").split(",") if s.strip()]
        out.append(d)
    return out


@router.get("/", response_class=HTMLResponse)
def catalog(request: Request, q: str = "", category: str = ""):
    sql = "SELECT * FROM products WHERE is_active = 1"
    args: list = []
    if category:
        sql += " AND category = ?"
        args.append(category)
    if q:
        # Plain LIKE here -- this is the browse filter, not the recommender's
        # retrieval path. Semantic + BM25 search lives in vectors.py (phase 4).
        sql += " AND (title LIKE ? OR skills LIKE ?)"
        args += [f"%{q}%", f"%{q}%"]
    sql += " ORDER BY rating DESC NULLS LAST, id LIMIT 60"

    counts = {r["category"]: r["n"] for r in db.q(
        "SELECT category, COUNT(*) n FROM products WHERE is_active=1 GROUP BY category")}
    return render(
        request, "catalog.html",
        products=rows_to_products(db.q(sql, tuple(args))),
        q=q, category=category,
        categories=[c for c in config.CATEGORIES if c in counts],
        counts=counts,
    )


@router.get("/p/{product_id}", response_class=HTMLResponse)
def product(request: Request, product_id: int):
    row = db.q1("SELECT * FROM products WHERE id = ? AND is_active = 1", (product_id,))
    if not row:
        return HTMLResponse("<h1>Not found</h1>", status_code=404)
    p = rows_to_products([row])[0]
    prereqs = []
    ids = json.loads(p["prereq_ids"] or "[]")
    if ids:
        ph = ",".join("?" * len(ids))
        prereqs = rows_to_products(
            db.q(f"SELECT * FROM products WHERE id IN ({ph}) AND is_active=1", tuple(ids)))
    same = rows_to_products(db.q(
        "SELECT * FROM products WHERE category=? AND id!=? AND is_active=1 "
        "ORDER BY rating DESC NULLS LAST LIMIT 4", (p["category"], product_id)))
    return render(request, "product.html", p=p, prereqs=prereqs, same=same)


# --- auth ------------------------------------------------------------------

@router.get("/login", response_class=HTMLResponse)
def login_form(request: Request, error: str = "", mode: str = "signin"):
    return render(request, "login.html", error=error, mode=mode)


@router.post("/login")
def login(request: Request, email: str = Form(...), password: str = Form(...)):
    user = auth.authenticate(email, password)
    if not user:
        return RedirectResponse("/login?error=Wrong+email+or+password", status_code=303)
    request.session["uid"] = user["id"]
    return RedirectResponse("/me", status_code=303)


@router.post("/signup")
def signup(request: Request, email: str = Form(...), password: str = Form(...)):
    if len(password) < 8:
        return RedirectResponse(
            "/login?mode=signup&error=Password+needs+8+characters+or+more", status_code=303)
    try:
        uid = auth.create_user(email, password)
    except sqlite3.IntegrityError:
        return RedirectResponse(
            "/login?mode=signup&error=That+email+is+already+registered", status_code=303)
    request.session["uid"] = uid
    return RedirectResponse("/welcome", status_code=303)


@router.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/", status_code=303)


@router.get("/me", response_class=HTMLResponse)
def dashboard(request: Request):
    user = auth.current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if not user["onboarded"]:
        return RedirectResponse("/welcome", status_code=303)

    row = db.q1("SELECT * FROM recommendations WHERE user_id = ? AND is_current = 1 "
                "ORDER BY id DESC LIMIT 1", (user["id"],))
    reco = None
    if row:
        reco = dict(row)
        items = []
        for i in json.loads(reco["items_json"]):
            p = db.q1("SELECT * FROM products WHERE id = ?", (i["product_id"],))
            if p:
                items.append({**i, "product": dict(p)})
        # NOT "items": Jinja resolves reco.items to dict.items() -- the method,
        # not the key -- and the template dies with a cryptic TypeError.
        reco["picks"] = items
    d = dossier_mod.current(user["id"])
    hist = dossier_mod.history(user["id"], limit=2)
    changed = dossier_mod.diff(d, hist[1]) if d and len(hist) > 1 else {
        "added": [], "removed": [], "changed": []}
    return render(request, "dashboard.html", reco=reco, dossier=d, changed=changed)


@router.get("/welcome", response_class=HTMLResponse)
def welcome(request: Request):
    user = auth.current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    counts = {r["category"]: r["n"] for r in db.q(
        "SELECT category, COUNT(*) n FROM products WHERE is_active=1 GROUP BY category")}
    return render(request, "welcome.html",
                  categories=[c for c in config.CATEGORIES if counts.get(c)])


@router.post("/welcome")
def welcome_submit(request: Request, picks: list[str] = Form(default=[])):
    user = auth.require_user(request)
    with db.tx() as c:
        c.execute("UPDATE users SET onboarded = 1 WHERE id = ?", (user["id"],))
    # Phase 5 seeds intent_vec from the centroid of each picked category's product
    # vectors. Storing the picks as search-ish events keeps them in the audit trail.
    if picks:
        import uuid
        with db.tx() as c:
            c.executemany(
                "INSERT OR IGNORE INTO events(id,user_id,session_id,type,query,ts) "
                "VALUES (?,?,?,'search',?,datetime('now'))",
                [(str(uuid.uuid4()), user["id"], "onboarding", p) for p in picks],
            )
    return RedirectResponse("/me", status_code=303)


@router.post("/enroll/{product_id}")
def enroll(request: Request, product_id: int):
    user = auth.current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    import uuid
    with db.tx() as c:
        c.execute(
            "INSERT OR IGNORE INTO events(id,user_id,session_id,type,product_id,ts) "
            "VALUES (?,?,?,'enroll',?,datetime('now'))",
            (str(uuid.uuid4()), user["id"], request.session.get("sid", "web"), product_id),
        )
    return RedirectResponse(f"/p/{product_id}", status_code=303)
