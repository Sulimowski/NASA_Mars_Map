"""EVA plans reduced to what the 3D page draws.

The planner's document (`eva_plans.json`, see EVA_PLANNER.md) is large; the page needs
the route of each strategy, its totals and a few notes. One function, used both when a
page is built from a file (`scripts/view3d.py`) and when the page asks for a new plan
(`scripts/serve.py`), so both show the same thing.
"""

from __future__ import annotations


def _thin(points: list, limit: int = 600) -> list:
    keep = max(1, len(points) // limit)
    return points[::keep] + ([points[-1]] if points and (len(points) - 1) % keep else [])


def plans_view(doc: dict, radiation_measured: bool | None = None) -> dict | None:
    """`eva_plans.json` document -> {sol, ltst, ..., items: [one per strategy]}, or None."""
    cond = doc.get("conditions", {})
    items = []
    for key, p in doc.get("strategies", {}).items():
        phys, radio = p.get("physiology") or {}, p.get("radio") or {}
        notes = []                       # composed here in English from the structured fields
        if p.get("skipped_targets"):
            notes.append("Left out: " + ", ".join(t["name"] for t in p["skipped_targets"]) + ".")
        if radio.get("shadow_m"):
            notes.append(f"No radio contact with the base on {radio['shadow_m']:.0f} m of the route; "
                         f"{len(radio.get('relay_points', []))} relay drop points proposed.")
        if phys.get("status") and phys["status"] != "OK":
            notes.append(f"Physiology check {phys['status']}: {phys.get('message', '')}".strip())
        notes += [str(w) for w in phys.get("warnings", []) if w]
        notes += [str(w) for w in p.get("warnings", []) if str(w).startswith("Rerouted")]
        dd = p.get("dust_devils") or {}
        items.append({
            "id": key, "title": p.get("title", key), "feasible": bool(p.get("feasible")),
            "physiology": phys.get("status"),
            "points": _thin([[w["x_m"], w["y_m"]] for w in p.get("waypoints", [])]),
            "summary": p.get("summary", {}), "usage": p.get("usage_pct", {}),
            "targets": p.get("visited_targets", []), "warnings": notes,
            # when the traveller gets to each stop, minutes after leaving the base
            "stops": [{"name": st.get("name"), "kind": st.get("kind"),
                       "x": st.get("x_m"), "y": st.get("y_m"),
                       "arrive": st.get("arrive_min"), "depart": st.get("depart_min"),
                       "dwell": st.get("dwell_min")} for st in p.get("stops", [])],
            # expected dust devil encounters, the chance of one, and the walk at other start hours
            "devils": {k: dd[k] for k in ("expected", "probability", "best_start_h", "by_start", "start_h")
                       if k in dd} or None,
        })
    if not items:
        return None
    rate, temp = cond.get("dose_rate_usv_h") or {}, cond.get("ground_temp_c") or {}
    out = {"sol": cond.get("sol"), "ltst": cond.get("ltst_h"),
           "particleEvent": bool(cond.get("particle_event")),
           "doseRate": rate.get("median"), "groundTemp": temp.get("median"),
           "vehicle": doc.get("vehicle", "astronaut"),
           "startH": cond.get("start_h"), "startReason": cond.get("start_reason", ""),
           "items": items}
    if radiation_measured is not None:
        # Without a radiation series the planner falls back to a typical Gale dose.
        out["radiationMeasured"] = radiation_measured
    return out
