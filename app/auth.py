"""Email/password auth on stdlib hashlib. pbkdf2_hmac is fine here and beats a
passlib/bcrypt version fight for a 5-day build."""

import hashlib
import hmac
import os
import sqlite3

from fastapi import Depends, HTTPException, Request, status

from app import db

ITERATIONS = 240_000


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return f"pbkdf2${ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iters, salt_hex, hash_hex = stored.split("$")
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt_hex), int(iters)
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(dk.hex(), hash_hex)


def create_user(email: str, password: str, role: str = "user") -> int:
    with db.tx() as c:
        cur = c.execute(
            "INSERT INTO users(email, password_hash, role) VALUES (?,?,?)",
            (email.strip().lower(), hash_password(password), role),
        )
        return cur.lastrowid


def authenticate(email: str, password: str) -> sqlite3.Row | None:
    row = db.q1("SELECT * FROM users WHERE email = ?", (email.strip().lower(),))
    if row and verify_password(password, row["password_hash"]):
        return row
    return None


def current_user(request: Request) -> sqlite3.Row | None:
    uid = request.session.get("uid")
    if not uid:
        return None
    return db.q1("SELECT * FROM users WHERE id = ?", (uid,))


def require_user(request: Request) -> sqlite3.Row:
    user = current_user(request)
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Sign in to continue")
    return user


def require_admin(user=Depends(require_user)) -> sqlite3.Row:
    if user["role"] != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admins only")
    return user
