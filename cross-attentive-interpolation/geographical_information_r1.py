#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone Live Map Tracker Test App (v3)
Run with: streamlit run tracker_test.py

Notes:
- A Streamlit server may not expose the visitor's real client IP. Proxy headers
  are deployment-dependent; only trust them when your hosting proxy controls them.
- The simulated visitor is synthetic and uses fixed coordinates. It does not
  depend on a geolocation API and is clearly labelled as a test record.
- JSON-file storage is suitable for a standalone test, not durable multi-instance
  production analytics.
"""

from __future__ import annotations

import ipaddress
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VISITORS_LOG_FILE = os.path.join(SCRIPT_DIR, "visitors_log_test.json")
ONLINE_THRESHOLD_S = 300
GEO_API_TIMEOUT = 5
GEO_API_URL = "https://ipapi.co/{ip}/json/"

# Fixed, synthetic test point (Warsaw, Poland). Never presented as a real visitor.
SIMULATED_KEY = "__SIMULATED_MAP_TEST__"
SIMULATED_VISITOR = {
    "ip": "SIMULATED",
    "country": "Poland",
    "country_code": "PL",
    "city": "Warsaw (simulated)",
    "region": "Mazovia",
    "latitude": 52.2297,
    "longitude": 21.0122,
    "timezone": "Europe/Warsaw",
    "is_simulated": True,
}

UNKNOWN_GEO = {
    "country": "Unknown",
    "country_code": "",
    "city": "Unknown",
    "region": "Unknown",
    "latitude": None,
    "longitude": None,
    "timezone": None,
}


st.set_page_config(page_title="Live Map Tracker Test", layout="wide", page_icon="🌍")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def iso_now() -> str:
    return now_utc().isoformat()


def format_time_ago(value: Any) -> str:
    dt = parse_dt(value)
    if dt is None:
        return "unknown"
    seconds = max(0, (now_utc() - dt).total_seconds())
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)} h ago"
    return f"{int(seconds // 86400)} d ago"


def get_country_flag(country_code: str) -> str:
    cc = str(country_code or "").upper()
    if len(cc) != 2 or not cc.isalpha():
        return "🏳️"
    try:
        return chr(ord(cc[0]) + 127397) + chr(ord(cc[1]) + 127397)
    except (ValueError, TypeError):
        return "🏳️"


def valid_ip(value: Any) -> Optional[str]:
    """Return a normalized IP address, excluding malformed header values."""
    if not value:
        return None
    candidate = str(value).strip()
    # Header values sometimes contain a port or brackets for IPv6.
    if candidate.startswith("[") and "]" in candidate:
        candidate = candidate[1:candidate.index("]")]
    else:
        # Remove an IPv4 :port only; don't split raw IPv6 addresses.
        try:
            ipaddress.ip_address(candidate)
        except ValueError:
            if candidate.count(":") == 1:
                host, port = candidate.rsplit(":", 1)
                if port.isdigit():
                    candidate = host
    try:
        return str(ipaddress.ip_address(candidate))
    except ValueError:
        return None


def is_public_ip(ip: str) -> bool:
    try:
        parsed = ipaddress.ip_address(ip)
        return parsed.is_global
    except ValueError:
        return False


def get_client_ip() -> tuple[Optional[str], str]:
    """
    Best effort. Prefer proxy headers only when supplied by a trusted deployment.
    A public egress-IP lookup is a fallback, but may return the server/proxy IP
    rather than the individual browser's address.
    """
    try:
        headers = st.context.headers
    except Exception:
        headers = {}

    # These headers are only trustworthy if the hosting/reverse proxy sets them
    # and prevents visitors from spoofing them.
    for header_name in ("CF-Connecting-IP", "X-Real-IP"):
        candidate = valid_ip(headers.get(header_name))
        if candidate:
            return candidate, f"request header: {header_name}"

    xff = headers.get("X-Forwarded-For")
    if xff:
        # Conventionally the leftmost value is the original client, but this
        # is safe only if the proxy chain is configured and sanitized correctly.
        candidates = [valid_ip(part) for part in str(xff).split(",")]
        candidates = [candidate for candidate in candidates if candidate]
        if candidates:
            return candidates[0], "request header: X-Forwarded-For"

    try:
        response = requests.get(
            "https://api.ipify.org?format=json",
            timeout=GEO_API_TIMEOUT,
        )
        response.raise_for_status()
        candidate = valid_ip(response.json().get("ip"))
        if candidate:
            return candidate, "ipify egress-IP fallback (may be server/proxy IP)"
    except (requests.RequestException, ValueError, TypeError) as exc:
        st.session_state["ip_detection_error"] = str(exc)

    return None, "unavailable"


@st.cache_data(ttl=3600, show_spinner=False)
def lookup_geo_cached(ip: str) -> tuple[dict, str]:
    """Geolocate a public IP; return both the data and diagnostic message."""
    if not is_public_ip(ip):
        return dict(UNKNOWN_GEO), "IP is not globally routable; no public geolocation lookup performed."

    try:
        response = requests.get(GEO_API_URL.format(ip=ip), timeout=GEO_API_TIMEOUT)
        if response.status_code != 200:
            return dict(UNKNOWN_GEO), f"Geolocation API returned HTTP {response.status_code}."
        data = response.json()
        if data.get("error"):
            reason = data.get("reason") or data.get("message") or "API reported an error"
            return dict(UNKNOWN_GEO), f"Geolocation API error: {reason}"

        geo = {
            "country": data.get("country_name") or "Unknown",
            "country_code": data.get("country_code") or "",
            "city": data.get("city") or "Unknown",
            "region": data.get("region") or "Unknown",
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "timezone": data.get("timezone"),
        }
        if geo["latitude"] is None or geo["longitude"] is None:
            return geo, "Lookup responded, but did not provide coordinates."
        return geo, "Geolocation lookup succeeded."
    except (requests.RequestException, ValueError, TypeError) as exc:
        return dict(UNKNOWN_GEO), f"Geolocation request failed: {exc}"


def get_geo_info(ip: Optional[str]) -> tuple[dict, str]:
    if not ip:
        return dict(UNKNOWN_GEO), "No IP address could be detected."
    try:
        parsed = ipaddress.ip_address(ip)
        if not parsed.is_global:
            return dict(UNKNOWN_GEO), "Detected address is private, reserved, or otherwise non-global."
    except ValueError:
        return dict(UNKNOWN_GEO), "Detected IP address is invalid."
    return lookup_geo_cached(ip)


def load_visitors_log() -> dict:
    if not os.path.exists(VISITORS_LOG_FILE):
        return {}
    try:
        with open(VISITORS_LOG_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        st.warning(f"Could not read visitor log: {exc}")
        return {}


def save_visitors_log(log: dict) -> None:
    """Write atomically to reduce the chance of a partially written JSON file."""
    temp_path = VISITORS_LOG_FILE + ".tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(log, handle, indent=2, ensure_ascii=False)
        os.replace(temp_path, VISITORS_LOG_FILE)
    except OSError as exc:
        st.warning(f"Could not save visitor log: {exc}")
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass


def visitor_record(key: str, ip: str, geo: dict, existing: dict, simulated: bool = False) -> dict:
    now = iso_now()
    return {
        "ip": ip,
        "last_seen": now,
        "first_seen": existing.get("first_seen", now),
        # Count once per browser session rather than on every Streamlit rerun.
        "visits": existing.get("visits", 0) + (0 if existing.get("counted_this_session") else 1),
        "country": geo.get("country", "Unknown"),
        "country_code": geo.get("country_code", ""),
        "city": geo.get("city", "Unknown"),
        "region": geo.get("region", "Unknown"),
        "latitude": geo.get("latitude"),
        "longitude": geo.get("longitude"),
        "timezone": geo.get("timezone"),
        "is_simulated": simulated,
        "counted_this_session": True,
    }


def track_visitor() -> tuple[Optional[dict], int, list[dict], dict]:
    if "vt_ip" not in st.session_state:
        ip, source = get_client_ip()
        st.session_state.vt_ip = ip
        st.session_state.vt_ip_source = source
        geo, diagnostic = get_geo_info(ip)
        st.session_state.vt_geo = geo
        st.session_state.vt_geo_diagnostic = diagnostic

    ip = st.session_state.get("vt_ip")
    geo = st.session_state.get("vt_geo") or dict(UNKNOWN_GEO)
    log = load_visitors_log()
    now = now_utc()

    if st.session_state.get("vt_simulated", False):
        key = SIMULATED_KEY
        geo = dict(SIMULATED_VISITOR)
        existing = log.get(key, {})
        log[key] = visitor_record(key, "SIMULATED", geo, existing, simulated=True)
        current = log[key]
    elif ip:
        key = ip
        existing = log.get(key, {})
        # Preserve an earlier successful geolocation if a later lookup fails.
        if geo.get("latitude") is None and existing.get("latitude") is not None:
            geo = {**geo, **{k: existing.get(k) for k in (
                "country", "country_code", "city", "region", "latitude", "longitude", "timezone"
            )}}
        log[key] = visitor_record(key, ip, geo, existing)
        current = log[key]
    else:
        current = None

    if ip or st.session_state.get("vt_simulated", False):
        save_visitors_log(log)

    online = []
    for key, info in log.items():
        last = parse_dt(info.get("last_seen"))
        if last and (now - last).total_seconds() <= ONLINE_THRESHOLD_S:
            online.append(info)

    return current, len(online), online, log


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.title("🌍 Live Map Tracker — Standalone Test")
st.caption("Test IP detection, geolocation, visitor logging and map rendering independently.")

with st.expander("🔧 Debug Info", expanded=True):
    st.write(f"**Log file:** `{VISITORS_LOG_FILE}`")
    st.write(f"**Streamlit version:** {st.__version__}")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Re-detect IP", use_container_width=True):
            for key in ("vt_ip", "vt_geo", "vt_ip_source", "vt_geo_diagnostic", "vt_simulated"):
                st.session_state.pop(key, None)
            st.rerun()
    with col2:
        if st.button("🧪 Simulate Map Point", use_container_width=True):
            st.session_state.vt_simulated = True
            st.session_state.vt_ip = None
            st.session_state.vt_ip_source = "synthetic test point (not a real visitor)"
            st.session_state.vt_geo = dict(SIMULATED_VISITOR)
            st.session_state.vt_geo_diagnostic = "Fixed test coordinates; external API not used."
            st.rerun()
    with col3:
        if st.button("🗑️ Clear Visitor Log", use_container_width=True):
            try:
                if os.path.exists(VISITORS_LOG_FILE):
                    os.remove(VISITORS_LOG_FILE)
            except OSError as exc:
                st.error(f"Could not delete log: {exc}")
            for key in ("vt_ip", "vt_geo", "vt_ip_source", "vt_geo_diagnostic", "vt_simulated"):
                st.session_state.pop(key, None)
            st.rerun()

    ip = st.session_state.get("vt_ip")
    if "vt_ip" not in st.session_state:
        ip, source = get_client_ip()
        st.session_state.vt_ip = ip
        st.session_state.vt_ip_source = source
        geo, diagnostic = get_geo_info(ip)
        st.session_state.vt_geo = geo
        st.session_state.vt_geo_diagnostic = diagnostic

    st.write(f"**Detected IP:** `{st.session_state.get('vt_ip') or 'Unavailable'}`")
    st.write(f"**Detection source:** {st.session_state.get('vt_ip_source', 'unknown')}")
    st.write(f"**Geo info:** `{st.session_state.get('vt_geo', UNKNOWN_GEO)}`")
    st.write(f"**Geolocation diagnostic:** {st.session_state.get('vt_geo_diagnostic', 'Not run')}")

    if st.session_state.get("vt_simulated", False):
        st.info("Simulation mode is active. The map point is synthetic and is labelled as simulated.")
    elif ip and not is_public_ip(ip):
        st.warning(
            "A non-public IP was detected in the request headers. This is likely an internal "
            "proxy address, not the visitor's public IP. It cannot be geolocated. Your hosting "
            "proxy must expose the original client IP for real visitor mapping."
        )
    elif not ip:
        st.warning(
            "The app could not determine a client IP. The egress-IP fallback may be blocked, "
            "and Streamlit deployments do not always expose the original browser IP."
        )

try:
    visitor_info, online_count, online_visitors, log = track_visitor()
except Exception as exc:
    st.error(f"Tracker error: {exc}")
    st.stop()

now = now_utc()
c1, c2, c3 = st.columns(3)
c1.metric("🟢 Online Now", online_count)
c2.metric("📊 Logged Visitor Keys", len(log))
try:
    log_size = os.path.getsize(VISITORS_LOG_FILE) if os.path.exists(VISITORS_LOG_FILE) else 0
except OSError:
    log_size = 0
c3.metric("📝 Log Size", f"{log_size:,} B")

if visitor_info:
    flag = get_country_flag(visitor_info.get("country_code", ""))
    label = "SIMULATED TEST POINT" if visitor_info.get("is_simulated") else f"{flag} {visitor_info.get('city', 'Unknown')}"
    with st.expander(f"📍 Current record: {label}", expanded=True):
        if visitor_info.get("is_simulated"):
            st.warning("Synthetic test record — not a real visitor or detected location.")
        st.write(f"**IP / key:** `{visitor_info.get('ip', 'Unknown')}`")
        st.write(f"**Country:** {flag} {visitor_info.get('country', 'Unknown')}")
        st.write(f"**City / Region:** {visitor_info.get('city', '?')} / {visitor_info.get('region', '?')}")
        if visitor_info.get("timezone"):
            st.write(f"**Timezone:** {visitor_info['timezone']}")
        st.write(f"**Visits / sessions counted:** {visitor_info.get('visits', 1)}")
        st.write(f"**First seen:** {format_time_ago(visitor_info.get('first_seen'))}")

# Country counts exclude synthetic records and unknown/local addresses.
country_counts: dict[tuple[str, str], int] = {}
for info in log.values():
    if info.get("is_simulated"):
        continue
    country = info.get("country", "Unknown")
    code = info.get("country_code", "")
    if country not in ("Unknown", "Local Network", ""):
        key = (country, code)
        country_counts[key] = country_counts.get(key, 0) + 1

if country_counts:
    st.markdown("#### 🌐 Top Visitor Countries")
    sorted_countries = sorted(country_counts.items(), key=lambda item: -item[1])
    cols = st.columns(min(5, len(sorted_countries)))
    for i, ((country, code), count) in enumerate(sorted_countries[:5]):
        with cols[i]:
            st.metric(f"{get_country_flag(code)} {country[:18]}", f"{count} visitor key(s)")

# Map all records with valid coordinates, including the clearly labelled test point.
map_data = []
for key, info in log.items():
    lat, lon = info.get("latitude"), info.get("longitude")
    if lat is None or lon is None:
        continue
    try:
        lat, lon = float(lat), float(lon)
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue
        last_seen = parse_dt(info.get("last_seen"))
        first_seen = parse_dt(info.get("first_seen"))
        if not last_seen:
            continue
        age = (now - last_seen).total_seconds()
        if info.get("is_simulated"):
            color, status, size = "#8B5CF6", "🧪 Simulated test point", 16
        elif age <= ONLINE_THRESHOLD_S:
            color, status, size = "#10B981", "🟢 Recently active", 12
        elif age <= 86400:
            color, status, size = "#F59E0B", "🟠 Seen today", 10
        else:
            color, status, size = "#6B7280", "⚪ Offline", 8
        map_data.append({
            "key": key,
            "lat": lat,
            "lon": lon,
            "city": info.get("city", "Unknown"),
            "country": info.get("country", "Unknown"),
            "visits": info.get("visits", 1),
            "last_seen": last_seen,
            "first_seen": first_seen or last_seen,
            "color": color,
            "status": status,
            "size": size,
            "flag": get_country_flag(info.get("country_code", "")),
            "simulated": bool(info.get("is_simulated")),
        })
    except (ValueError, TypeError):
        continue

# Always render the map so map functionality can be tested even when IP
# geolocation is unavailable. If there are no real mapped records, show a
# separate synthetic diagnostic point (never save it as a visitor).
showing_demo_only = not bool(map_data)
if showing_demo_only:
    map_data.append({
        "key": "__MAP_DIAGNOSTIC_ONLY__",
        "lat": 52.2297,
        "lon": 21.0122,
        "city": "Warsaw (map test)",
        "country": "Poland",
        "visits": 0,
        "last_seen": now,
        "first_seen": now,
        "color": "#8B5CF6",
        "status": "🧪 Map rendering test — not a visitor",
        "size": 16,
        "flag": "🧪",
        "simulated": True,
    })

with st.expander("🗺️ Visitor Map", expanded=True):
        df = pd.DataFrame(map_data)
        fig = go.Figure()
        fig.add_trace(go.Scattergeo(
            lon=df["lon"],
            lat=df["lat"],
            text=df["city"] + ", " + df["country"],
            customdata=df[["visits", "status", "flag", "simulated"]],
            mode="markers",
            marker=dict(
                size=df["size"],
                color=df["color"],
                opacity=0.88,
                line=dict(width=1, color="rgba(0,0,0,0.45)"),
            ),
            hovertemplate=(
                "<b>%{customdata[2]} %{text}</b><br>"
                "Visits/sessions: %{customdata[0]}<br>"
                "Status: %{customdata[1]}<br>"
                "Simulated: %{customdata[3]}<extra></extra>"
            ),
        ))
        fig.update_layout(
            title="Visitor locations (approximate IP geolocation)",
            geo=dict(
                projection_type="natural earth",
                showland=True,
                landcolor="rgb(243,244,246)",
                showocean=True,
                oceancolor="rgb(225,242,255)",
                showcountries=True,
                countrycolor="rgb(200,200,200)",
                fitbounds="locations",
            ),
            margin=dict(l=0, r=0, t=45, b=0),
            height=520,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "IP-based locations are approximate and may identify a network gateway rather than a person. "
            "The purple marker is synthetic when simulation mode is enabled."
        )
if showing_demo_only:
    st.info(
        "No real visitor records currently have valid coordinates. The purple Warsaw marker "
        "is a map-rendering diagnostic only; it is not saved to the visitor log and does not "
        "represent your location. To map real visitors, your deployment must expose a valid "
        "public client IP and the geolocation lookup must return coordinates."
    )

if log:
    with st.expander("📅 Recent records (last 10)", expanded=False):
        recent = sorted(
            log.items(),
            key=lambda item: item[1].get("last_seen", ""),
            reverse=True,
        )[:10]
        for key, info in recent:
            flag = "🧪" if info.get("is_simulated") else get_country_flag(info.get("country_code", ""))
            city = info.get("city", "Unknown")
            country = info.get("country", "Unknown")
            st.write(
                f"{flag} {city}, {country} • {format_time_ago(info.get('last_seen'))} "
                f"• key `{key}`"
            )

st.caption(
    "Standalone test app. Records are stored in visitors_log_test.json. "
    "Use visitor analytics transparently and follow applicable privacy requirements."
)
