#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone Live Map Tracker Test App (v2 - Fixed Local IP Handling)
====================================================================
Minimal Streamlit app to test the geographic visitor tracker independently.
Run with: streamlit run tracker_test.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import json
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple

# =============================================
# CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VISITORS_LOG_FILE = os.path.join(SCRIPT_DIR, "visitors_log_test.json")
ONLINE_THRESHOLD_S = 300   # 5 minutes
GEO_API_TIMEOUT = 4        # seconds

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="Live Map Tracker Test",
    layout="wide",
    page_icon="🌍"
)

# =============================================
# CORE FUNCTIONS
# =============================================
def get_country_flag(country_code: str) -> str:
    if not country_code or len(str(country_code)) != 2:
        return "🏳️"
    try:
        cc = str(country_code).upper()
        return chr(ord(cc[0]) + 127397) + chr(ord(cc[1]) + 127397)
    except Exception:
        return "🏳️"

def format_time_ago(dt: datetime) -> str:
    now = datetime.now()
    seconds = (now - dt).total_seconds()
    if seconds < 60: return "just now"
    if seconds < 3600: return f"{int(seconds // 60)} min ago"
    if seconds < 86400: return f"{int(seconds // 3600)} h ago"
    return f"{int(seconds // 86400)} d ago"

def get_client_ip() -> Optional[str]:
    """Best-effort retrieval of the visitor's public IP."""
    # Method 1: Streamlit headers (works when behind a proxy / deployed)
    try:
        headers = st.context.headers  # Streamlit ≥ 1.37
        xff = headers.get("X-Forwarded-For")
        if xff:
            # X-Forwarded-For can be a comma-separated list: "client, proxy1, proxy2"
            # We want the first PUBLIC IP in the list.
            ips = [ip.strip() for ip in xff.split(",")]
            private_prefixes = ("10.", "192.168.", "172.16.", "172.17.", "172.18.", "172.19.", 
                                "172.20.", "172.21.", "172.22.", "172.23.", "172.24.", "172.25.", 
                                "172.26.", "172.27.", "172.28.", "172.29.", "172.30.", "172.31.", 
                                "127.", "169.254.", "::1")
            for ip in ips:
                if not ip.startswith(private_prefixes):
                    return ip
            # If all are private, return the first one (it will be flagged as Local Network later)
            return ips[0] if ips else None
        
        xri = headers.get("X-Real-IP")
        if xri:
            xri = xri.strip()
            if not xri.startswith(("10.", "192.168.", "172.", "127.")):
                return xri
    except Exception as e:
        st.caption(f"ℹ️ Header method unavailable: {e}")

    # Method 2: External API fallback (works on localhost)
    try:
        r = requests.get("https://api.ipify.org?format=json", timeout=GEO_API_TIMEOUT)
        if r.status_code == 200:
            return r.json().get("ip")
    except Exception as e:
        st.warning(f"⚠️ IP API request failed: {e}")

    return None

def get_geo_info(ip: Optional[str]) -> Dict[str, Any]:
    """Resolve IP -> {country, country_code, city, region, lat, lon, tz}."""
    empty = {
        'country': 'Unknown', 'country_code': '', 'city': 'Unknown',
        'region': 'Unknown', 'latitude': None, 'longitude': None, 'timezone': None
    }
    if not ip:
        return empty

    # Local network detection
    if ip in ("127.0.0.1", "::1", "localhost") or ip.startswith(("192.168.", "10.", "172.", "169.254.")):
        return {
            **empty,
            'country': 'Local Network', 'country_code': 'LO',
            'city': 'Local', 'region': 'Local'
        }

    # Rate-limit cooldown
    cooldown_until = st.session_state.get('geo_api_cooldown_until')
    if cooldown_until and datetime.now() < cooldown_until:
        return empty

    try:
        r = requests.get(f"https://ipapi.co/{ip}/json/", timeout=GEO_API_TIMEOUT)
        remaining = int(r.headers.get('X-Rl', 999))
        ttl = int(r.headers.get('X-Ttl', 60))

        if remaining <= 1:
            st.session_state['geo_api_cooldown_until'] = datetime.now() + timedelta(seconds=ttl)
            st.toast(f"Geo API rate limit reached. Cooldown for {ttl} seconds.", icon="⏳")
            return empty

        if r.status_code == 200:
            d = r.json()
            return {
                'country': d.get('country_name', 'Unknown'),
                'country_code': d.get('country_code', ''),
                'city': d.get('city', 'Unknown'),
                'region': d.get('region', 'Unknown'),
                'latitude': d.get('latitude'),
                'longitude': d.get('longitude'),
                'timezone': d.get('timezone'),
            }
    except Exception as e:
        st.warning(f"⚠️ Geo lookup failed: {e}")

    return empty

def load_visitors_log() -> Dict[str, Any]:
    try:
        if os.path.exists(VISITORS_LOG_FILE):
            with open(VISITORS_LOG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"Could not load log: {e}")
    return {}

def save_visitors_log(log: Dict[str, Any]) -> None:
    try:
        with open(VISITORS_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.warning(f"Could not save log: {e}")

def track_visitor() -> Tuple[Optional[Dict], int, list]:
    """Register / refresh the current visitor."""
    if 'vt_ip' not in st.session_state or st.session_state.vt_ip is None:
        st.session_state.vt_ip = get_client_ip()
        st.session_state.vt_geo = get_geo_info(st.session_state.vt_ip)

    ip = st.session_state.vt_ip
    geo = st.session_state.vt_geo or {
        'country': 'Unknown', 'country_code': '', 'city': 'Unknown',
        'region': 'Unknown', 'latitude': None, 'longitude': None, 'timezone': None
    }
    now = datetime.now()

    log = load_visitors_log()
    existing = log.get(ip, {}) if ip else {}

    if ip:
        log[ip] = {
            'ip': ip,
            'last_seen': now.isoformat(),
            'first_seen': existing.get('first_seen', now.isoformat()),
            'visits': existing.get('visits', 0) + 1,
            'country': geo['country'],
            'country_code': geo['country_code'],
            'city': geo['city'],
            'region': geo['region'],
            'latitude': geo['latitude'],
            'longitude': geo['longitude'],
            'timezone': geo['timezone'],
        }
        save_visitors_log(log)

    online = []
    for vip, info in log.items():
        try:
            last = datetime.fromisoformat(info['last_seen'])
            if (now - last).total_seconds() <= ONLINE_THRESHOLD_S:
                online.append(info)
        except Exception:
            continue

    return log.get(ip), len(online), online

# =============================================
# UI
# =============================================
st.title("🌍 Live Map Tracker — Standalone Test")
st.caption("Minimal app to verify IP detection + animated visitor map works correctly.")

# --- Debug panel ---
with st.expander("🔧 Debug Info", expanded=True):
    st.write(f"**Log file:** `{VISITORS_LOG_FILE}`")
    st.write(f"**Streamlit version:** {st.__version__}")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("🔄 Re-detect IP"):
            st.session_state.pop('vt_ip', None)
            st.session_state.pop('vt_geo', None)
            st.rerun()
    with col_b:
        if st.button("🗑️ Clear Visitor Log"):
            if os.path.exists(VISITORS_LOG_FILE):
                os.remove(VISITORS_LOG_FILE)
            st.session_state.pop('vt_ip', None)
            st.session_state.pop('vt_geo', None)
            st.rerun()
    with col_c:
        if st.button("🧪 Simulate Public IP"):
            # Force a known public IP (Google DNS) to test the map rendering
            st.session_state.vt_ip = "8.8.8.8"
            st.session_state.vt_geo = get_geo_info("8.8.8.8")
            st.rerun()

    ip = st.session_state.get('vt_ip')
    geo = st.session_state.get('vt_geo')
    st.write(f"**Detected IP:** `{ip}`")
    st.write(f"**Geo info:** `{geo}`")
    
    if ip and ip.startswith(("10.", "192.168.", "172.")):
        st.warning("⚠️ **Local/Private IP Detected:** Your network or hosting platform is masking your real public IP with a local address (e.g., `10.x.x.x`). Local IPs cannot be geolocated. Click **Simulate Public IP** above to verify the map works, or deploy this app to a public URL to get real visitor IPs.")

# --- Track visitor ---
try:
    visitor_info, online_count, online_visitors = track_visitor()
except Exception as e:
    st.error(f"Tracker error: {e}")
    st.stop()

log = load_visitors_log()
now = datetime.now()

# --- Summary ---
c1, c2, c3 = st.columns(3)
c1.metric("🟢 Online Now", online_count)
c2.metric("📊 Total Unique Visitors", len(log))
c3.metric("📝 Log Size", f"{os.path.getsize(VISITORS_LOG_FILE) if os.path.exists(VISITORS_LOG_FILE) else 0} B")

# --- Current visitor card ---
if visitor_info:
    flag = get_country_flag(visitor_info.get('country_code', ''))
    with st.expander(f"📍 You: {flag} {visitor_info.get('city', 'Unknown')}", expanded=True):
        st.write(f"**IP:** `{visitor_info['ip']}`")
        st.write(f"**Country:** {flag} {visitor_info.get('country', 'Unknown')}")
        st.write(f"**City / Region:** {visitor_info.get('city', '?')} / {visitor_info.get('region', '?')}")
        if visitor_info.get('timezone'):
            st.write(f"**Timezone:** {visitor_info['timezone']}")
        st.write(f"**Visits:** {visitor_info.get('visits', 1)}")
        try:
            first = datetime.fromisoformat(visitor_info['first_seen'])
            st.write(f"**First seen:** {format_time_ago(first)}")
        except Exception:
            pass

# --- Top countries ---
if log:
    country_counts = {}
    for vip, info in log.items():
        cc = info.get('country_code', 'LO')
        country = info.get('country', 'Unknown')
        if country == 'Local Network':
            continue
        key = (country, cc)
        country_counts[key] = country_counts.get(key, 0) + 1

    if country_counts:
        sorted_countries = sorted(country_counts.items(), key=lambda x: -x[1])
        st.markdown("#### 🏳️ Top Visitor Countries")
        cols = st.columns(min(5, len(sorted_countries)))
        for i, ((country, cc), count) in enumerate(sorted_countries[:5]):
            with cols[i]:
                flag = get_country_flag(cc)
                short_name = country[:15] + '...' if len(country) > 15 else country
                st.metric(f"{flag} {short_name}", f"{count} visitors")

# --- Animated map ---
map_data = []
for vip, info in log.items():
    lat = info.get('latitude')
    lon = info.get('longitude')
    if lat and lon:
        try:
            last_seen_dt = datetime.fromisoformat(info.get('last_seen', '1970-01-01'))
            first_seen_dt = datetime.fromisoformat(info.get('first_seen', '1970-01-01'))

            seconds_since_last = (now - last_seen_dt).total_seconds()
            if seconds_since_last <= ONLINE_THRESHOLD_S:
                status_color = '#10B981'
                status_text = "🟢 Online"
                base_size = 12
            elif seconds_since_last <= 86400:
                status_color = '#F59E0B'
                status_text = "🟠 Seen today"
                base_size = 10
            else:
                status_color = '#6B7280'
                status_text = "⚪ Offline"
                base_size = 8

            visits = info.get('visits', 1)
            cc = info.get('country_code', '')
            flag = get_country_flag(cc)

            map_data.append({
                'lat': float(lat),
                'lon': float(lon),
                'city': info.get('city', 'Unknown'),
                'country': info.get('country', 'Unknown'),
                'visits': visits,
                'first_seen': first_seen_dt,
                'last_seen': last_seen_dt,
                'color': status_color,
                'status': status_text,
                'size': base_size + (visits * 2),
                'flag': flag
            })
        except (ValueError, TypeError) as e:
            st.warning(f"Skipping malformed entry: {e}")
            continue

if map_data:
    with st.expander("🗺️ Live Tracker Map (Animated)", expanded=True):
        df_map = pd.DataFrame(map_data)

        dates = pd.date_range(end=now.date(), periods=14, freq='D').to_pydatetime()
        frames = []
        for d in dates:
            d_py = d if isinstance(d, datetime) else pd.Timestamp(d).to_pydatetime()
            visible = df_map[df_map['first_seen'] <= d_py]
            if not visible.empty:
                frame_data = go.Scattergeo(
                    lon=visible['lon'],
                    lat=visible['lat'],
                    customdata=visible[['city', 'country', 'visits', 'status', 'flag']],
                    mode='markers',
                    marker=dict(
                        size=visible['size'],
                        color=visible['color'],
                        line=dict(width=1, color='rgba(0,0,0,0.5)'),
                        opacity=0.85
                    ),
                    hovertemplate="<b>%{customdata[4]} %{customdata[0]}, %{customdata[1]}</b><br>"
                                  "Visits: %{customdata[2]}<br>Status: %{customdata[3]}<extra></extra>"
                )
                frames.append(go.Frame(data=frame_data, name=d_py.strftime('%Y-%m-%d')))
            else:
                frames.append(go.Frame(data=go.Scattergeo(), name=d_py.strftime('%Y-%m-%d')))

        latest_dt = dates[-1] if isinstance(dates[-1], datetime) else pd.Timestamp(dates[-1]).to_pydatetime()
        visible_init = df_map[df_map['first_seen'] <= latest_dt]

        fig_map = go.Figure(
            data=go.Scattergeo(
                lon=visible_init['lon'],
                lat=visible_init['lat'],
                customdata=visible_init[['city', 'country', 'visits', 'status', 'flag']],
                mode='markers',
                marker=dict(
                    size=visible_init['size'],
                    color=visible_init['color'],
                    line=dict(width=1, color='rgba(0,0,0,0.5)'),
                    opacity=0.85
                ),
                hovertemplate="<b>%{customdata[4]} %{customdata[0]}, %{customdata[1]}</b><br>"
                              "Visits: %{customdata[2]}<br>Status: %{customdata[3]}<extra></extra>"
            ),
            frames=frames
        )

        fig_map.update_layout(
            title_text='Global Visitor Tracker (Animated)',
            geo=dict(
                projection_type='natural earth',
                showland=True,
                landcolor='rgb(243, 244, 246)',
                showocean=True,
                oceancolor='rgb(225, 242, 255)',
                showcountries=True,
                countrycolor='rgb(200, 200, 200)'
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            height=500,
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=0.95, x=0.01, xanchor="left", yanchor="top",
                buttons=[
                    dict(label="▶️ Play", method="animate",
                         args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)]),
                    dict(label="⏸️ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                ]
            )],
            sliders=[dict(
                active=len(dates) - 1,
                currentvalue=dict(prefix="Date: "),
                pad=dict(t=10),
                steps=[
                    dict(
                        label=(d.strftime('%m-%d') if isinstance(d, datetime)
                               else pd.Timestamp(d).strftime('%m-%d')),
                        method="animate",
                        args=[[(d.strftime('%Y-%m-%d') if isinstance(d, datetime)
                                else pd.Timestamp(d).strftime('%Y-%m-%d'))]]
                    )
                    for d in dates
                ]
            )]
        )
        st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info(
        "⏳ Waiting for visitor geographic data. The map will appear once public IPs are logged.\n\n"
        "**Why this happens:** Your current IP (`10.12.32.180`) is a private/local network address. "
        "These cannot be mapped to a geographic location.\n\n"
        "**How to test:** Click the **🧪 Simulate Public IP** button in the Debug Info panel above "
        "to force a test location and verify the map renders correctly."
    )

# --- Recent visitors ---
if log:
    with st.expander("📅 Recent Visitors (last 10)", expanded=False):
        sorted_visitors = sorted(
            log.items(),
            key=lambda kv: kv[1].get('last_seen', ''),
            reverse=True
        )[:10]
        for vip, info in sorted_visitors:
            try:
                last_dt = datetime.fromisoformat(info['last_seen'])
                when = format_time_ago(last_dt)
            except Exception:
                when = '?'
            flag = get_country_flag(info.get('country_code', ''))
            city = info.get('city', '?')
            country = info.get('country', '?')
            st.write(f"{flag} {city}, {country} • {when} • IP `{vip}`")

st.caption("ℹ️ Test app — IP logged for analytics only. Data stored in `visitors_log_test.json`.")
