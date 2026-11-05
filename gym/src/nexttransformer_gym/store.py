"""Durable identities, transactions, idempotency and exact integer accounts."""

import contextlib
import json
import sqlite3
import time
from pathlib import Path
from .common import canonical, new_id, require


class Store:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        with self.transaction() as db:
            db.executescript("""
              PRAGMA journal_mode=WAL;
              CREATE TABLE IF NOT EXISTS objects(id TEXT, owner TEXT, kind TEXT, value TEXT, PRIMARY KEY(owner,kind,id));
              CREATE INDEX IF NOT EXISTS ownership ON objects(owner,kind,id);
              CREATE TABLE IF NOT EXISTS retries(owner TEXT,key TEXT,request TEXT,response TEXT,PRIMARY KEY(owner,key));
              CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT,owner TEXT,value TEXT);
            """)

    @contextlib.contextmanager
    def transaction(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def put(self, db, owner, kind, value):
        value = dict(value)
        value.setdefault("id", new_id())
        db.execute(
            "INSERT INTO objects VALUES(?,?,?,?)",
            (value["id"], owner, kind, canonical(value)),
        )
        self.event(db, owner, kind + ".created", value["id"])
        return value

    def update(self, db, owner, kind, value):
        count = db.execute(
            "UPDATE objects SET value=? WHERE id=? AND owner=? AND kind=?",
            (canonical(value), value["id"], owner, kind),
        ).rowcount
        require(count == 1, "not_found", "Object is not accessible.")
        self.event(db, owner, kind + ".updated", value["id"])
        return value

    def get(self, db, owner, kind, oid):
        row = db.execute(
            "SELECT value FROM objects WHERE id=? AND owner=? AND kind=?",
            (oid, owner, kind),
        ).fetchone()
        require(row is not None, "not_found", "Object is not accessible.")
        return json.loads(row[0])

    def all(self, db, owner, kind):
        return [
            json.loads(row[0])
            for row in db.execute(
                "SELECT value FROM objects WHERE owner=? AND kind=? ORDER BY id",
                (owner, kind),
            )
        ]

    def event(self, db, owner, kind, oid):
        db.execute(
            "INSERT INTO events(owner,value) VALUES(?,?)",
            (owner, canonical({"kind": kind, "object_id": oid, "time": time.time()})),
        )

    def budget(self, db, owner):
        account = self.get(db, owner, "account", owner)
        spent = sum(int(r["amount"]) for r in self.all(db, owner, "charge"))
        reserved = sum(int(r["remaining"]) for r in self.all(db, owner, "reservation"))
        limit = int(account["limit"])
        require(
            limit - spent - reserved >= 0,
            "integrity_violation",
            "Account invariant violated.",
        )
        return {
            "limit": str(limit),
            "spent": str(spent),
            "reserved": str(reserved),
            "available": str(limit - spent - reserved),
            "accounting_version": "gym.flops.v1",
        }

    def reserve(self, db, owner, job_id, amount):
        require(
            type(amount) is int and amount > 0,
            "incomplete_coverage",
            "A positive execution bound is required.",
        )
        require(
            amount <= int(self.budget(db, owner)["available"]),
            "insufficient_budget",
            "Reservation exceeds available FLOPs.",
        )
        return self.put(
            db,
            owner,
            "reservation",
            {"id": job_id, "remaining": str(amount), "bound": str(amount)},
        )

    def charge(self, db, owner, job_id, charge_id, amount):
        require(
            type(amount) is int and amount >= 0,
            "invalid_request",
            "Invalid trusted charge.",
        )
        charges = {r["id"]: r for r in self.all(db, owner, "charge")}
        if charge_id in charges:
            require(
                charges[charge_id]["job_id"] == job_id
                and charges[charge_id]["amount"] == str(amount),
                "conflict",
                "Charge ID was reused with different content.",
            )
            return
        reservation = self.get(db, owner, "reservation", job_id)
        require(
            amount <= int(reservation["remaining"]),
            "insufficient_budget",
            "Work exceeds reserved bound.",
        )
        reservation["remaining"] = str(int(reservation["remaining"]) - amount)
        self.update(db, owner, "reservation", reservation)
        self.put(
            db,
            owner,
            "charge",
            {"id": charge_id, "job_id": job_id, "amount": str(amount)},
        )

    def release(self, db, owner, job_id):
        reservation = self.get(db, owner, "reservation", job_id)
        reservation["remaining"] = "0"
        self.update(db, owner, "reservation", reservation)
