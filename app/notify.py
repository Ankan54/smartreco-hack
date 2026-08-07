"""Proactive delivery by email.

Email rather than Telegram because the addressing problem is already solved: we
collect an address at signup. Telegram would need every user to find the bot,
message it, and us to capture a chat_id -- and the shortcut (one shared chat id)
is not per-user delivery at all, it is a demo prop.

Plain SMTP on purpose, driven entirely by env vars. Brevo, Resend, Mailjet,
Amazon SES and Gmail all speak SMTP, so the provider is a config change rather
than a code change -- the same rule this repo applies to the LLM client.

Nothing here needs a dependency: smtplib and email.message are stdlib. And with
SMTP_HOST unset every function is a no-op, so a clean clone with only a Mesh key
still boots and still passes its checks.
"""

import html
import json
import logging
import smtplib
import ssl
from email.message import EmailMessage

from app import config, db

log = logging.getLogger(__name__)


def enabled() -> bool:
    return bool(config.SMTP_HOST and config.SMTP_FROM)


def send(to: str, subject: str, html_body: str, text_body: str) -> bool:
    """Returns True if the server accepted it. Never raises: a failed digest
    must not take down the scheduler thread."""
    if not enabled():
        log.debug("smtp not configured, skipping send to %s", to)
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = config.SMTP_FROM
    msg["To"] = to
    # Plain text first, HTML as the alternative: clients pick, and a text part
    # keeps us out of spam filters that distrust HTML-only mail.
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")

    try:
        if config.SMTP_PORT == 465:
            with smtplib.SMTP_SSL(config.SMTP_HOST, 465,
                                  context=ssl.create_default_context(), timeout=30) as s:
                if config.SMTP_USER:
                    s.login(config.SMTP_USER, config.SMTP_PASSWORD)
                s.send_message(msg)
        else:
            with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=30) as s:
                s.starttls(context=ssl.create_default_context())
                if config.SMTP_USER:
                    s.login(config.SMTP_USER, config.SMTP_PASSWORD)
                s.send_message(msg)
        return True
    except Exception:
        log.exception("smtp send to %s failed", to)
        return False


# --- the digest ------------------------------------------------------------

def _render(reco: dict, items: list[dict], base_url: str) -> tuple[str, str]:
    """(html, text). The narrative is the agent's verbatim -- a digest that
    rewrites it in a different voice is a second product with a second bug
    surface."""
    esc = html.escape
    rows = []
    plain = [reco["headline"], "", reco["narrative"], ""]

    for i, item in enumerate(items, start=1):
        p = item["product"]
        price = f"&#8377;{p['price']:,.0f}" if p["price"] else "Free"
        reason = esc(item.get("reason") or "")
        rows.append(f"""
        <tr><td style="padding:14px 0;border-bottom:1px solid #C2D4E0">
          <div style="font:500 13px/1 ui-monospace,monospace;color:#7D5A2A">{i:02d}</div>
          <a href="{base_url}/p/{p['id']}"
             style="font:500 16px/1.35 -apple-system,Segoe UI,sans-serif;color:#0E2436;
                    text-decoration:none">{esc(p['title'])}</a>
          <div style="font:400 14px/1.5 -apple-system,Segoe UI,sans-serif;color:#4C6274;
                      margin-top:4px">{reason}</div>
          <div style="font:400 12px/1 ui-monospace,monospace;color:#4C6274;
                      margin-top:6px">{price}</div>
        </td></tr>""")
        plain.append(f"{i}. {p['title']}"
                     + (f" - {item.get('reason')}" if item.get("reason") else ""))
        plain.append(f"   {base_url}/p/{p['id']}")

    body = f"""<!doctype html>
<html><body style="margin:0;background:#F1F6FA;padding:28px 16px">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
   <tr><td align="center">
    <table role="presentation" width="560" cellpadding="0" cellspacing="0"
           style="max-width:560px;background:#fff;border:1px solid #C2D4E0;
                  border-left:3px solid #C4005A;padding:26px 28px">
      <tr><td>
        <div style="font:500 11px/1 ui-monospace,monospace;letter-spacing:.14em;
                    text-transform:uppercase;color:#7D5A2A">Reckon</div>
        <h1 style="font:500 21px/1.25 -apple-system,Segoe UI,sans-serif;color:#0E2436;
                   margin:14px 0 12px">{esc(reco['headline'])}</h1>
        <p style="font:400 16px/1.65 Georgia,serif;color:#0E2436;margin:0 0 8px">
          {esc(reco['narrative'])}</p>
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
          {''.join(rows)}
        </table>
        <p style="font:400 12px/1.5 ui-monospace,monospace;color:#4C6274;margin:20px 0 0">
          Sent because you turned on the daily digest.
          <a href="{base_url}/me" style="color:#0F5C77">Turn it off</a>.
        </p>
      </td></tr>
    </table>
   </td></tr>
  </table>
</body></html>"""

    plain += ["", f"Sent because you turned on the daily digest. Turn it off: {base_url}/me"]
    return body, "\n".join(plain)


def deliver_digest(user_id: int, base_url: str | None = None) -> bool:
    """Send this user's current recommendation. Returns True if delivered.

    Only ever sends to a user who opted in: signup addresses are unverified, so
    mailing everyone who registered would be sending unsolicited email to
    strangers.
    """
    base_url = (base_url or config.PUBLIC_BASE_URL).rstrip("/")
    user = db.q1("SELECT email, digest_opt_in FROM users WHERE id = ?", (user_id,))
    if not user or not user["digest_opt_in"]:
        return False

    row = db.q1("SELECT * FROM recommendations WHERE user_id = ? AND is_current = 1 "
                "ORDER BY id DESC LIMIT 1", (user_id,))
    if not row:
        return False
    if row["delivered_at"]:
        log.debug("reco %s already delivered", row["id"])
        return False

    items = []
    for i in json.loads(row["items_json"]):
        p = db.q1("SELECT id,title,price FROM products WHERE id = ?", (i["product_id"],))
        if p:
            items.append({**i, "product": dict(p)})
    if not items:
        return False

    body, text = _render(dict(row), items, base_url)
    ok = send(user["email"], f"Reckon — {row['headline']}", body, text)
    if ok:
        with db.tx() as c:
            c.execute("UPDATE recommendations SET delivered_at = datetime('now') "
                      "WHERE id = ?", (row["id"],))
        log.info("digest emailed to user %s", user_id)
    return ok
