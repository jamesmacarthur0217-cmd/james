"""Manage login accounts for other devices (this PC itself never needs to log in).

    python tools/users.py list
    python tools/users.py add <name> [--role admin|viewer]     asks for the password
    python tools/users.py password <name>                      asks for the new password
    python tools/users.py role <name> admin|viewer
    python tools/users.py remove <name>

admin: everything. viewer: live view, search and questions only. Changing a password or role logs that user
out everywhere. Works while the app is running.
"""
from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from labeler import accounts  # noqa: E402
from labeler.config import load_config  # noqa: E402
from labeler.db import Database  # noqa: E402


def ask_password() -> str:
    first = getpass.getpass("Password: ")
    accounts.check_new_password(first)
    if getpass.getpass("Same password again: ") != first:
        raise ValueError("the two passwords are different")
    return first


def run(db: Database, args: argparse.Namespace, ask=ask_password) -> str:
    """Do one command; returns the message to print. Raises ValueError for a mistake by the user."""
    if args.command == "list":
        users = db.list_users()
        return "\n".join(f"{u['username']:20s} {u['role']:7s} since {u['created_at']}" for u in users) or "no accounts"
    if args.command == "add":
        if db.get_user(args.name):
            raise ValueError(f"user {args.name!r} already exists")
        db.add_user(args.name, accounts.hash_password(ask()), args.role)
        return f"added {args.name} ({args.role})"
    if args.command == "password":
        if not db.get_user(args.name):
            raise ValueError(f"no user {args.name!r}")
        db.update_user(args.name, password_hash=accounts.hash_password(ask()))
        return f"password changed for {args.name}; logged out everywhere"
    if args.command == "role":
        if not db.update_user(args.name, role=args.role):
            raise ValueError(f"no user {args.name!r}")
        return f"{args.name} is now {args.role}; logged out everywhere"
    if args.command == "remove":
        if not db.delete_user(args.name):
            raise ValueError(f"no user {args.name!r}")
        return f"removed {args.name}"
    raise ValueError(f"unknown command {args.command}")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Manage login accounts.")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    add = sub.add_parser("add")
    add.add_argument("name")
    add.add_argument("--role", choices=accounts.ROLES, default="viewer")
    sub.add_parser("password").add_argument("name")
    role = sub.add_parser("role")
    role.add_argument("name")
    role.add_argument("role", choices=accounts.ROLES)
    sub.add_parser("remove").add_argument("name")
    return ap


def main() -> None:
    args = parser().parse_args()
    db = Database(load_config()["database"]["url"])
    try:
        print(run(db, args))
    except ValueError as e:
        raise SystemExit(f"error: {e}")


if __name__ == "__main__":
    main()
