"""End-to-end tests for the ten-minute slot-lock (到访名额锁定) feature."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

import psycopg

from app import config as app_config
from app.config import settings

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def iso(dt):
    return dt.isoformat().replace("+00:00", "Z")


def at(hour, minute=0, day_offset=1):
    day = (datetime.now(UTC) + timedelta(days=day_offset)).date()
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=UTC)


def open_point(client, tz="UTC", capacity=1):
    r = client.post("/points", json={"name": "锁定测试点", "timezone": tz,
                                     "capacity": capacity})
    assert r.status_code == 201, r.text
    pid = r.json()["id"]
    tomorrow = (datetime.now(UTC) + timedelta(days=1)).date()
    r = client.post(
        f"/points/{pid}/rules",
        json={
            "expected_version": 0,
            "rules": {
                "weekly": [
                    {"weekday": tomorrow.isoweekday(),
                     "start_time": "08:00", "end_time": "20:00"}
                ],
                "closed_dates": [],
                "makeup": [],
            },
        },
    )
    assert r.status_code == 201, r.text
    return pid


def hold(client, pid, key, start, end):
    return client.post(
        f"/points/{pid}/locks",
        json={"idempotency_key": key, "start": iso(start), "end": iso(end)},
    )


def confirm(client, pid, lid, token):
    return client.post(
        f"/points/{pid}/locks/{lid}/confirm",
        json={"confirm_token": token},
    )


def reserve(client, pid, key, start, end):
    return client.post(
        f"/points/{pid}/reservations",
        json={"idempotency_key": key, "start": iso(start), "end": iso(end)},
    )


def waitlist(client, pid, key, start, end):
    return client.post(
        f"/points/{pid}/waitlist",
        json={"idempotency_key": key, "start": iso(start), "end": iso(end)},
    )


def wstatus(client, pid, wid, token):
    return client.get(
        f"/points/{pid}/waitlist/{wid}", params={"query_token": token}
    )


def cancel(client, pid, rid, token):
    return client.post(
        f"/points/{pid}/reservations/{rid}/cancel",
        json={"cancel_token": token},
    )


def _raw_conn():
    """Plain (tuple-row) connection bypassing the app's dict-row pool."""
    return psycopg.connect(app_config.settings.database_url)


def make_expired_lock(pid, key, start, end):
    """Insert a lock whose TTL has already elapsed (bypasses the API clock)."""
    now = datetime.now(UTC)
    with _raw_conn() as conn:
        with conn.transaction():
            conn.execute(
                "INSERT INTO slot_locks "
                "(point_id, idempotency_key, start_utc, end_utc, rules_version, "
                " confirm_token, status, expires_at) "
                "VALUES (%s, %s, %s, %s, 1, 'tok', 'held', %s)",
                (pid, key, start, end, now - timedelta(seconds=1)),
            )


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------


def test_create_lock_returns_number_confirm_token_expiry_and_held_status(client):
    pid = open_point(client)
    start, end = at(9), at(10)
    before = datetime.now(UTC)
    r = hold(client, pid, "h1", start, end)
    after = datetime.now(UTC)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["id"] > 0
    assert body["point_id"] == pid
    assert body["status"] == "held"
    assert body["confirm_token"]
    assert body["reservation_id"] is None
    assert body["confirmed_at"] is None
    assert body["rules_version"] == 1
    assert body["start"] == iso(start)
    assert body["end"] == iso(end)
    expires = datetime.fromisoformat(body["expires_at"].replace("Z", "+00:00"))
    assert before + timedelta(seconds=settings.lock_ttl_seconds) <= expires
    assert expires <= after + timedelta(seconds=settings.lock_ttl_seconds)
    assert expires - before <= timedelta(minutes=10, seconds=1)


def test_lock_ttl_is_ten_minutes(client):
    assert settings.lock_ttl_seconds == 600


def test_same_key_same_content_replays_original_lock(client):
    pid = open_point(client)
    first = hold(client, pid, "dup", at(9), at(10))
    assert first.status_code == 201
    replay = hold(client, pid, "dup", at(9), at(10))
    assert replay.status_code == 200, replay.text
    assert replay.json() == first.json()


def test_same_key_different_content_conflicts(client):
    pid = open_point(client)
    hold(client, pid, "k", at(9), at(10)).raise_for_status()
    r = hold(client, pid, "k", at(11), at(11, 30))
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "idempotency_conflict"


def test_lock_window_uses_reservation_policy(client):
    pid = open_point(client)
    # End before start.
    r = hold(client, pid, "bad1", at(10), at(9))
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_period"
    # Too long.
    r = hold(client, pid, "bad2", at(8), at(10, 1))
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_period"
    # Outside open hours (point open 08:00-20:00).
    r = hold(client, pid, "bad3", at(19, 30), at(20, 30))
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "outside_open_hours"


def test_lock_on_point_without_published_rules(client):
    r = client.post("/points", json={"name": "无规则点", "timezone": "UTC"})
    pid = r.json()["id"]
    r = hold(client, pid, "h", at(9), at(10))
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "no_published_version"


# ---------------------------------------------------------------------------
# A held lock occupies one slot everywhere
# ---------------------------------------------------------------------------


def test_held_lock_blocks_overlapping_reservation_but_allows_touching(client):
    pid = open_point(client)
    h = hold(client, pid, "h", at(9), at(10)).json()
    assert h["status"] == "held"

    blocked = reserve(client, pid, "r", at(9, 30), at(10, 30))
    assert blocked.status_code == 409
    err = blocked.json()["error"]
    assert err["code"] == "reservation_overlap"
    assert err["details"]["conflicting_lock_ids"] == [h["id"]]
    assert err["details"]["capacity"] == 1
    assert err["details"]["peak_concurrency"] == 1

    # Touching end-to-start is free under [start, end).
    touching = reserve(client, pid, "r2", at(10), at(10, 30))
    assert touching.status_code == 201, touching.text


def test_capacity_two_lock_and_reservation_coexist(client):
    pid = open_point(client, capacity=2)
    hold(client, pid, "h", at(9), at(10)).raise_for_status()
    r = reserve(client, pid, "r", at(9), at(10))
    assert r.status_code == 201, r.text
    third = reserve(client, pid, "r3", at(9, 15), at(9, 45))
    assert third.status_code == 409
    assert third.json()["error"]["details"]["peak_concurrency"] == 2


def test_two_overlapping_holds_rejected_at_capacity(client):
    pid = open_point(client)
    hold(client, pid, "h1", at(9), at(10)).raise_for_status()
    r = hold(client, pid, "h2", at(9), at(9, 30))
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "reservation_overlap"


def test_held_lock_counts_for_waitlist_admission(client):
    pid = open_point(client)
    hold(client, pid, "h", at(9), at(10)).raise_for_status()
    # At capacity because of the lock: waitlist is accepted.
    w = waitlist(client, pid, "w", at(9), at(10))
    assert w.status_code == 201, w.text
    assert w.json()["status"] == "waiting"


def test_expired_lock_does_not_block_reservation(client):
    pid = open_point(client)
    # An expired hold occupies nothing: the same window books directly.
    make_expired_lock(pid, "old-lock", at(9), at(10))
    r = reserve(client, pid, "r", at(9), at(10))
    assert r.status_code == 201, r.text


def test_expired_lock_allows_direct_booking_without_waitlist(client):
    pid = open_point(client)
    make_expired_lock(pid, "old-lock", at(9), at(10))
    r = hold(client, pid, "new-lock", at(9), at(10))
    assert r.status_code == 201, r.text
    assert r.json()["status"] == "held"
    # The expired row is marked expired as a side effect of the write.
    with _raw_conn() as conn:
        st = conn.execute(
            "SELECT status FROM slot_locks WHERE idempotency_key = 'old-lock'"
        ).fetchone()[0]
    assert st == "expired"


# ---------------------------------------------------------------------------
# Confirmation
# ---------------------------------------------------------------------------


def test_confirm_converts_lock_into_reservation(client):
    pid = open_point(client)
    h = hold(client, pid, "h", at(9), at(10)).json()
    r = confirm(client, pid, h["id"], h["confirm_token"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["lock"]["status"] == "confirmed"
    assert body["lock"]["reservation_id"] == body["reservation"]["id"]
    assert body["lock"]["confirmed_at"] is not None
    resv = body["reservation"]
    assert resv["id"] > 0
    assert resv["point_id"] == pid
    assert resv["status"] == "active"
    assert resv["revision"] == 1
    assert resv["start"] == iso(at(9))
    assert resv["end"] == iso(at(10))
    assert resv["cancel_token"]
    assert resv["rules_version"] == h["rules_version"]
    # The new reservation really occupies the slot.
    blocked = reserve(client, pid, "other", at(9, 30), at(10, 30))
    assert blocked.status_code == 409


def test_repeat_confirm_returns_same_reservation(client):
    pid = open_point(client)
    h = hold(client, pid, "h", at(9), at(10)).json()
    first = confirm(client, pid, h["id"], h["confirm_token"]).json()
    second = confirm(client, pid, h["id"], h["confirm_token"])
    assert second.status_code == 200
    body = second.json()
    assert body["reservation"]["id"] == first["reservation"]["id"]
    assert body["reservation"]["cancel_token"] == \
        first["reservation"]["cancel_token"]
    assert body["lock"]["reservation_id"] == first["reservation"]["id"]


def test_confirm_after_expiry_refused(client):
    pid = open_point(client)
    start, end = at(9), at(10)
    with _raw_conn() as conn:
        with conn.transaction():
            row = conn.execute(
                "INSERT INTO slot_locks "
                "(point_id, idempotency_key, start_utc, end_utc, rules_version, "
                " confirm_token, status, expires_at) "
                "VALUES (%s, %s, %s, %s, 1, 'secret', 'held', %s) RETURNING id",
                (pid, "h", start, end, datetime.now(UTC) - timedelta(seconds=1)),
            ).fetchone()
    lid = row[0]
    r = confirm(client, pid, lid, "secret")
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "lock_expired"


def test_confirm_with_wrong_token_is_403(client):
    pid = open_point(client)
    h = hold(client, pid, "h", at(9), at(10)).json()
    r = confirm(client, pid, h["id"], "nope")
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "invalid_lock_token"
    r = client.post(
        f"/points/{pid}/locks/{h['id']}/confirm", json={"confirm_token": None}
    )
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "invalid_lock_token"


def test_confirm_unknown_lock_is_404(client):
    pid = open_point(client)
    r = confirm(client, pid, 999999, "x")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "lock_not_found"


def test_confirmed_reservation_is_ordinary_and_cancellable(client):
    pid = open_point(client)
    h = hold(client, pid, "h", at(9), at(10)).json()
    confirmed = confirm(client, pid, h["id"], h["confirm_token"]).json()
    token = confirmed["reservation"]["cancel_token"]
    rid = confirmed["reservation"]["id"]
    r = cancel(client, pid, rid, token)
    assert r.status_code == 200
    assert r.json()["status"] == "cancelled"
    # Slot released.
    again = reserve(client, pid, "r2", at(9), at(10))
    assert again.status_code == 201


def test_expired_confirm_does_not_consume_slot(client):
    pid = open_point(client)
    start, end = at(9), at(10)
    with _raw_conn() as conn:
        with conn.transaction():
            row = conn.execute(
                "INSERT INTO slot_locks "
                "(point_id, idempotency_key, start_utc, end_utc, rules_version, "
                " confirm_token, status, expires_at) "
                "VALUES (%s, %s, %s, %s, 1, 'secret', 'held', %s) RETURNING id",
                (pid, "h", start, end, datetime.now(UTC) - timedelta(seconds=1)),
            ).fetchone()
    # The failed confirm sweeps the expired hold: the window is free.
    denied = confirm(client, pid, row[0], "secret")
    assert denied.status_code == 409
    r = reserve(client, pid, "r", at(9), at(10))
    assert r.status_code == 201, r.text


# ---------------------------------------------------------------------------
# Expiry sweep promotes the waitlist in existing order
# ---------------------------------------------------------------------------


def test_capacity_touching_write_cleans_expired_lock_and_promotes_waitlist(
    client,
):
    pid = open_point(client)
    # A reservation holds the only slot and a waitlist entry queues behind it.
    holder = reserve(client, pid, "holder", at(9), at(10)).json()
    w = waitlist(client, pid, "w", at(9), at(10)).json()
    assert w["status"] == "waiting"
    # The holder cancels — the waitlist entry is promoted into the slot.
    cancel(client, pid, holder["id"], holder["cancel_token"]).raise_for_status()
    promoted = wstatus(client, pid, w["id"], w["query_token"]).json()
    assert promoted["status"] == "promoted"
    rid = promoted["reservation"]["id"]

    # Now that reservation holds the slot; create an expired lock for the same
    # window (occupies nothing) and queue another entry behind the promotion.
    w2 = waitlist(client, pid, "w2", at(9), at(10)).json()
    assert w2["status"] == "waiting"
    # Expired lock on a different window does not free the 09:00 slot by
    # itself.  To release, cancel the promoted reservation; the write sweeps
    # any expired locks first and promotes w2.
    make_expired_lock(pid, "dead", at(15), at(16))
    cancel(client, pid, rid, promoted["reservation"]["cancel_token"]).raise_for_status()
    body = wstatus(client, pid, w2["id"], w2["query_token"]).json()
    assert body["status"] == "promoted"


def test_expired_lock_freed_slot_promotes_waitlist_on_next_write(client):
    pid = open_point(client)
    # Active lock holds the slot; queue behind it.
    h = hold(client, pid, "h", at(9), at(10)).json()
    w = waitlist(client, pid, "w", at(9), at(10)).json()
    assert w["status"] == "waiting"
    # Force the lock to expire in the database.
    with _raw_conn() as conn:
        with conn.transaction():
            conn.execute(
                "UPDATE slot_locks SET expires_at = now() - interval '1 second' "
                "WHERE id = %s",
                (h["id"],),
            )
    # Any capacity-touching write sweeps it and promotes the queue.
    other = reserve(client, pid, "touch", at(11), at(11, 30))
    assert other.status_code == 201, other.text
    body = wstatus(client, pid, w["id"], w["query_token"]).json()
    assert body["status"] == "promoted"
    assert body["reservation"]["start"] == iso(at(9))
    # The promoted reservation now blocks a direct booking.
    blocked = reserve(client, pid, "x", at(9), at(10))
    assert blocked.status_code == 409


def test_promotion_happens_before_new_write_claims_freed_slot(client):
    """The waitlist wins over a later direct request when a lock expires."""
    pid = open_point(client)
    h = hold(client, pid, "h", at(9), at(10)).json()
    waitlist(client, pid, "w", at(9), at(10)).raise_for_status()
    with _raw_conn() as conn:
        with conn.transaction():
            conn.execute(
                "UPDATE slot_locks SET expires_at = now() - interval '1 second' "
                "WHERE id = %s",
                (h["id"],),
            )
    # The direct request for the freed window arrives together with the
    # sweep; application order means the waitlist entry gets the slot.
    r = reserve(client, pid, "late", at(9), at(10))
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "reservation_overlap"


def test_expired_hold_does_not_promote_when_window_not_open_under_new_rules(
    client,
):
    pid = open_point(client)
    h = hold(client, pid, "h", at(9), at(10)).json()
    w = waitlist(client, pid, "w", at(9), at(10)).json()
    # Let the lock expire first, then publish v2 removing the slot's morning
    # coverage: an expired hold occupies nothing, so the publish is allowed;
    # the queued entry is however not promotable under the new calendar.
    with _raw_conn() as conn:
        with conn.transaction():
            conn.execute(
                "UPDATE slot_locks SET expires_at = now() - interval '1 second' "
                "WHERE id = %s",
                (h["id"],),
            )
    tomorrow = (datetime.now(UTC) + timedelta(days=1)).date()
    client.post(
        f"/points/{pid}/rules",
        json={
            "expected_version": 1,
            "rules": {
                "weekly": [
                    {"weekday": tomorrow.isoweekday(),
                     "start_time": "13:00", "end_time": "20:00"}
                ],
                "closed_dates": [],
                "makeup": [],
            },
        },
    ).raise_for_status()
    # Unrelated booking in the still-open afternoon triggers the sweep.
    reserve(client, pid, "other", at(14), at(15)).raise_for_status()
    body = wstatus(client, pid, w["id"], w["query_token"]).json()
    assert body["status"] == "waiting"


# ---------------------------------------------------------------------------
# Calendar / timezone / capacity changes must not invalidate held locks
# ---------------------------------------------------------------------------


def test_rule_change_uncovering_held_lock_is_refused(client):
    pid = open_point(client)
    h = hold(client, pid, "h", at(9), at(9, 30)).json()
    tomorrow = (datetime.now(UTC) + timedelta(days=1)).date()
    r = client.post(
        f"/points/{pid}/rules",
        json={
            "expected_version": 1,
            "rules": {
                "weekly": [
                    {"weekday": tomorrow.isoweekday(),
                     "start_time": "13:00", "end_time": "20:00"}
                ],
                "closed_dates": [],
                "makeup": [],
            },
        },
    )
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "reservation_change_conflict"
    assert f"lock-{h['id']}" in err["details"]["conflicting_reservation_ids"]


def test_rule_change_uncovering_only_expired_lock_is_allowed(client):
    pid = open_point(client)
    make_expired_lock(pid, "dead", at(9), at(9, 30))
    tomorrow = (datetime.now(UTC) + timedelta(days=1)).date()
    r = client.post(
        f"/points/{pid}/rules",
        json={
            "expected_version": 1,
            "rules": {
                "weekly": [
                    {"weekday": tomorrow.isoweekday(),
                     "start_time": "13:00", "end_time": "20:00"}
                ],
                "closed_dates": [],
                "makeup": [],
            },
        },
    )
    assert r.status_code == 201, r.text
    with _raw_conn() as conn:
        st = conn.execute(
            "SELECT status FROM slot_locks WHERE idempotency_key = 'dead'"
        ).fetchone()[0]
    assert st == "expired"


def test_rule_change_still_covering_held_lock_succeeds_and_lock_confirms(client):
    pid = open_point(client)
    h = hold(client, pid, "h", at(9), at(9, 30)).json()
    tomorrow = (datetime.now(UTC) + timedelta(days=1)).date()
    r = client.post(
        f"/points/{pid}/rules",
        json={
            "expected_version": 1,
            "rules": {
                "weekly": [
                    {"weekday": tomorrow.isoweekday(),
                     "start_time": "08:00", "end_time": "22:00"}
                ],
                "closed_dates": [],
                "makeup": [],
            },
        },
    )
    assert r.status_code == 201, r.text
    # The lock survives and confirms against the newer rules.
    confirmed = confirm(client, pid, h["id"], h["confirm_token"])
    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json()["reservation"]["rules_version"] == 2


def test_capacity_decrease_below_held_lock_peak_is_refused(client):
    pid = open_point(client, capacity=2)
    hold(client, pid, "h", at(9), at(10)).raise_for_status()
    reserve(client, pid, "r", at(9), at(10)).raise_for_status()
    # Peak is 2; lowering to 1 must be refused while the lock is valid.
    r = client.patch(f"/points/{pid}", json={"capacity": 1})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "capacity_conflict"
    # Equal to peak is allowed.
    r = client.patch(f"/points/{pid}", json={"capacity": 2})
    assert r.status_code == 200, r.text


def test_capacity_increase_promotes_waitlist_behind_locks(client):
    pid = open_point(client, capacity=1)
    hold(client, pid, "h", at(9), at(10)).raise_for_status()
    w = waitlist(client, pid, "w", at(9), at(10)).json()
    r = client.patch(f"/points/{pid}", json={"capacity": 2})
    assert r.status_code == 200
    body = wstatus(client, pid, w["id"], w["query_token"]).json()
    assert body["status"] == "promoted"


def test_timezone_change_not_blocked_by_expired_lock(client):
    # Same wall-clock rules in an equivalent-offset timezone keeps coverage;
    # an expired hold must not force the change to be refused.
    pid = open_point(client, tz="UTC")
    make_expired_lock(pid, "dead", at(9), at(9, 30))
    r = client.patch(f"/points/{pid}", json={"timezone": "Etc/GMT"})
    assert r.status_code == 200, r.text
    assert r.json()["timezone"] == "Etc/GMT"
    with _raw_conn() as conn:
        st = conn.execute(
            "SELECT status FROM slot_locks WHERE idempotency_key = 'dead'"
        ).fetchone()[0]
    assert st == "expired"


# ---------------------------------------------------------------------------
# Planning ignores expired locks but counts valid ones
# ---------------------------------------------------------------------------


def test_planner_counts_valid_lock_and_ignores_expired_lock(client):
    pid = open_point(client)
    other = open_point(client)
    # A valid lock holds the only slot tomorrow 09:00-10:00 at point 1.
    hold(client, pid, "h", at(9), at(10)).raise_for_status()
    body = {
        "stops": [
            {"point_id": pid, "duration_minutes": 60},
            {"point_id": other, "duration_minutes": 30},
        ],
        "min_gap_minutes": 0,
        "search_from": iso(at(9)),
        "search_to": iso(at(20) - timedelta(minutes=1)),
    }
    r = client.post("/trips/plan", json=body)
    assert r.status_code == 200, r.text
    first_start = r.json()["stops"][0]["start"]
    # The 09:00 slot is taken by the lock: the planner must skip to 10:00.
    assert first_start == iso(at(10))

    # Expire the lock; the planner ignores it and returns 09:00 again.
    with _raw_conn() as conn:
        with conn.transaction():
            conn.execute(
                "UPDATE slot_locks SET expires_at = now() - interval '1 second' "
                "WHERE idempotency_key = 'h'"
            )
    r = client.post("/trips/plan", json=body)
    assert r.status_code == 200
    assert r.json()["stops"][0]["start"] == iso(at(9))


# ---------------------------------------------------------------------------
# Trips count held locks
# ---------------------------------------------------------------------------


def test_trip_stop_capacity_counts_held_lock(client):
    pid = open_point(client)
    other = open_point(client)
    hold(client, pid, "h", at(9), at(10)).raise_for_status()
    r = client.post(
        "/trips",
        json={
            "idempotency_key": "trip1",
            "stops": [
                {"point_id": pid, "start": iso(at(9)), "end": iso(at(10))},
                {"point_id": other, "start": iso(at(10)), "end": iso(at(10, 30))},
            ],
        },
    )
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "reservation_overlap"
    assert err["details"]["stop_index"] == 0
    assert err["details"]["point_id"] == pid


# ---------------------------------------------------------------------------
# Concurrency at the expiry boundary
# ---------------------------------------------------------------------------


def test_concurrent_confirm_and_expiry_promotion_do_not_double_occupy(client):
    pid = open_point(client)
    h = hold(client, pid, "h", at(9), at(10)).json()
    waitlist(client, pid, "w", at(9), at(10)).raise_for_status()
    # Park the lock exactly at its deadline: a confirmation racing with the
    # sweep must either confirm it or expire it and promote the waitlist —
    # never both.
    with _raw_conn() as conn:
        with conn.transaction():
            conn.execute(
                "UPDATE slot_locks SET expires_at = %s WHERE id = %s",
                (datetime.now(UTC), h["id"]),
            )

    results = []

    def do_confirm():
        results.append(confirm(client, pid, h["id"], h["confirm_token"]))

    def do_touch():
        # A capacity-touching write runs the expiry sweep + promotion.
        results.append(reserve(client, pid, "touch", at(11), at(11, 30)))

    barrier = threading.Barrier(2)

    def run(fn):
        barrier.wait()
        fn()

    threads = [threading.Thread(target=run, args=(do_confirm,)),
               threading.Thread(target=run, args=(do_touch,))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly one reservation must occupy 09:00-10:00 afterwards.
    occupied = reserve(client, pid, "probe", at(9, 30), at(10, 30))
    assert occupied.status_code == 409
    with _raw_conn() as conn:
        n_active = conn.execute(
            "SELECT COUNT(*) FROM reservations WHERE point_id = %s "
            "AND status = 'active' AND start_utc < %s AND end_utc > %s",
            (pid, at(10), at(9, 30)),
        ).fetchone()[0]
    assert n_active == 1
    # The lock is either confirmed (one reservation) or expired + waitlist
    # promoted (one reservation); never two.
    statuses = sorted(resp.status_code for resp in results)
    assert statuses in ([200, 201], [201, 409])


def test_concurrent_confirms_return_same_reservation(client):
    pid = open_point(client)
    h = hold(client, pid, "h", at(9), at(10)).json()
    out = []

    def do():
        out.append(confirm(client, pid, h["id"], h["confirm_token"]).json())

    barrier = threading.Barrier(2)
    threads = [
        threading.Thread(target=lambda: (barrier.wait(), do())) for _ in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(out) == 2
    assert out[0]["reservation"]["id"] == out[1]["reservation"]["id"]
    with _raw_conn() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM reservations WHERE point_id = %s", (pid,)
        ).fetchone()[0]
    assert n == 1
