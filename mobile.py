import os, re, time, json, hashlib, base64, io, hmac, asyncio, logging, sys, colorsys
from urllib.parse import urlparse, quote
from typing import Optional, Literal, Dict, Any, List, Tuple

from fastapi import FastAPI, HTTPException, Query, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()  
import httpx
from httpx import AsyncClient, Limits, Timeout
from contextlib import asynccontextmanager
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from colorthief import ColorThief
try:
    from openai import OpenAI  # SDK v1+
except Exception:
    OpenAI = None
try:
    import cairosvg  # type: ignore
except Exception:
    cairosvg = None

async def build_mobile_app_html(details: Dict[str, Any], *, sales_cta: bool = True) -> Tuple[str, Dict[str, Any]]:
    name = details.get("name") or "Restaurant"
    # Resolve helpers from app.py at runtime to avoid circular imports
    _app = sys.modules.get("app")
    if _app is None:
        import app as _app  # fallback import
    _safe = getattr(_app, "safe")
    _cuisine_from_types = getattr(_app, "cuisine_from_types")
    _get_restaurant_context = getattr(_app, "get_restaurant_context")
    _best_logo_with_color = getattr(_app, "best_logo_with_color")
    _select_theme_colors = getattr(_app, "select_theme_colors")
    _get_mobile_bg_data_uri = getattr(_app, "get_mobile_bg_data_uri")
    _select_hero_images = getattr(_app, "select_hero_images")
    _resolve_menu_images = getattr(_app, "resolve_menu_images")
    _try_enrich_menu_from_site = getattr(_app, "try_enrich_menu_from_site")
    _GOOGLE_API_KEY = getattr(_app, "GOOGLE_API_KEY", "")

    cuisine = _cuisine_from_types(details)
    context = _get_restaurant_context(details)
    logo, logo_color, logo_reason = await _best_logo_with_color(details)
    pal = _select_theme_colors(cuisine, context, logo_color)
    # Build high-quality logo tag (prefer 512px s2 favicon if used)
    logo_tag = ""
    try:
        if logo:
            l = str(logo)
            if "google.com/s2/favicons" in l and "sz=" in l:
                hi = re.sub(r"(?i)sz=\d+", "sz=512", l)
                lo = re.sub(r"(?i)sz=\d+", "sz=256", l)
                logo_tag = f"<img src='{_safe(hi)}' srcset='{_safe(lo)} 1x, {_safe(hi)} 2x' alt='logo'/>"
            else:
                logo_tag = f"<img src='{_safe(l)}' alt='logo'/>"
    except Exception:
        logo_tag = f"<img src='{_safe(logo or '')}' alt='logo'/>"
    # High-quality hero images from Google/Unsplash (fallback to SVG bg)
    hero_imgs: List[str] = await _select_hero_images(details, cuisine)
    # Try to enrich real menu from website, then resolve with images
    try:
        await _try_enrich_menu_from_site(details, cuisine)
    except Exception:
        pass
    try:
        await _resolve_menu_images(details, cuisine)
    except Exception:
        pass
    menu_items: List[Dict[str, Any]] = list((details.get("_resolved_menu") or [])[:6])
    lat = details.get("lat"); lng = details.get("lng")
    map_img = ""
    if lat is not None and lng is not None:
        try:
            if _GOOGLE_API_KEY:
                map_img = (
                    f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lng}&zoom=15&size=600x320&scale=2&maptype=roadmap&markers=color:red%7C{lat},{lng}&key={_GOOGLE_API_KEY}"
                )
            else:
                map_img = (
                    f"https://staticmap.openstreetmap.de/staticmap.php?center={lat},{lng}&zoom=15&size=600x320&markers={lat},{lng},red-pushpin"
                )
        except Exception:
            map_img = ""

    # --- Helpers to recolor/tint SVG data URIs to match brand palette ---
    def _parse_hex(c: str) -> Tuple[int, int, int]:
        c = c.strip()
        if c.startswith("#"):
            c = c[1:]
        if len(c) == 3:
            c = "".join(ch*2 for ch in c)
        if len(c) != 6:
            raise ValueError("bad hex")
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)

    def _rgb_to_hex(r: int, g: int, b: int) -> str:
        return "#{:02x}{:02x}{:02x}".format(max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))

    def _hex_to_hsl(c: str) -> Tuple[float, float, float]:
        r, g, b = _parse_hex(c)
        h, l, s = colorsys.rgb_to_hls(r/255.0, g/255.0, b/255.0)
        # return HSL-like tuple (0..360, 0..1, 0..1)
        return h*360.0, s, l

    def _hsl_to_hex(h: float, s: float, l: float) -> str:
        r, g, b = colorsys.hls_to_rgb((h % 360.0)/360.0, max(0.0, min(1.0, l)), max(0.0, min(1.0, s)))
        return _rgb_to_hex(int(round(r*255)), int(round(g*255)), int(round(b*255)))

    def _tint_from_lightness(base_hex: str, lightness: float, sat_scale: float = 1.0) -> str:
        """Return a color derived from base_hex with given lightness in [0,1]."""
        try:
            h, s, _ = _hex_to_hsl(base_hex)
        except Exception:
            # default fallback: brand red-like
            h, s = 0.0, 0.7
        s = max(0.0, min(1.0, s * sat_scale))
        return _hsl_to_hex(h, s, max(0.0, min(1.0, lightness)))

    def _parse_color_any(sval: str) -> Optional[Tuple[int, int, int]]:
        sval = sval.strip()
        try:
            if sval.startswith("#"):
                return _parse_hex(sval)
            if sval.startswith("rgb"):
                # rgb(255, 255, 255) or rgb(255 255 255 / a)
                nums = re.findall(r"[0-9]{1,3}", sval)
                if len(nums) >= 3:
                    return int(nums[0]), int(nums[1]), int(nums[2])
        except Exception:
            return None
        return None

    def _relative_lightness_from_rgb(rgb: Tuple[int, int, int]) -> float:
        # perceptual lightness approximation (0..1)
        r, g, b = [x/255.0 for x in rgb]
        return 0.2126*r + 0.7152*g + 0.0722*b

    def _tint_svg_data_uri(data_uri: str, base_hex: str, accent_hex: Optional[str] = None) -> str:
        """Decode an SVG data URI, adjust fills to brand hue while preserving light/dark structure, then re-encode."""
        try:
            # Extract raw SVG
            raw = ""
            if data_uri.startswith("data:image/svg+xml;base64,"):
                raw = base64.b64decode(data_uri.split(",", 1)[1]).decode("utf-8", "ignore")
            elif data_uri.startswith("data:image/svg+xml;utf8,") or data_uri.startswith("data:image/svg+xml;utf-8,"):
                raw = data_uri.split(",", 1)[1]
                raw = bytes(raw, "utf-8").decode("utf-8", "ignore")
                raw = raw.replace("%0A", "").replace("%09", "")
                try:
                    raw = re.sub(r"%([0-9A-Fa-f]{2})", lambda m: bytes.fromhex(m.group(1)).decode('latin-1'), raw)
                except Exception:
                    pass
            else:
                # plain SVG markup
                raw = data_uri

            import xml.etree.ElementTree as ET
            # Parse and recolor
            tree = ET.ElementTree(ET.fromstring(raw))
            root = tree.getroot()
            ns = {"svg": "http://www.w3.org/2000/svg"}
            # iterate over all elements with 'fill'
            for el in root.iter():
                f = el.get("fill")
                if not f or f.lower() in ("none", "transparent"):
                    continue
                rgb = _parse_color_any(f)
                if rgb is None:
                    # unknown color, force medium-light brand tone
                    el.set("fill", _tint_from_lightness(base_hex, 0.82))
                    continue
                L = _relative_lightness_from_rgb(rgb)
                # Map extremes to keep contrast
                if L >= 0.85:
                    new_col = _tint_from_lightness(base_hex, 0.90)
                elif L >= 0.65:
                    new_col = _tint_from_lightness(base_hex, 0.75)
                elif L >= 0.45:
                    new_col = _tint_from_lightness(base_hex, 0.55)
                elif L >= 0.25:
                    new_col = _tint_from_lightness(base_hex, 0.38)
                else:
                    # very dark strokes – use brand dark/accent if provided
                    new_col = (accent_hex or base_hex)
                el.set("fill", new_col)

            out_svg = ET.tostring(root, encoding="utf-8").decode("utf-8")
            # Re-encode to data URI (base64 for safety)
            encoded = base64.b64encode(out_svg.encode("utf-8")).decode("ascii")
            return "data:image/svg+xml;base64," + encoded
        except Exception:
            # Fallback: return original
            return data_uri

    mobile_bg_original = _get_mobile_bg_data_uri()
    # Tint the SVG using the primary brand; use primary_dark as accent for darker areas
    mobile_bg = _tint_svg_data_uri(mobile_bg_original, pal.get("primary", "#cc0000"), pal.get("primary_dark", pal.get("primary", "#cc0000")))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
  <title>{_safe(name)} • Mobile</title>
  <style>
    :root{{ --brand: {pal['primary']}; --brand-dark: {pal['primary_dark']}; --ink:#1b1b1d; --muted:#6b7280; --bg:#ffffff;}}
    *{{box-sizing:border-box;-webkit-tap-highlight-color:transparent}}
    html,body{{margin:0;padding:0;background:#fff;color:var(--ink);font:16px/1.35 -apple-system,BlinkMacSystemFont,'SF Pro Text','Inter',system-ui,Segoe UI,Roboto,Helvetica,Arial;}}
    .device{{position:relative;width:390px;height:844px;margin:20px auto;background:linear-gradient(180deg, color-mix(in srgb, var(--brand) 12%, #fff) 0%, #fff 48%);border-radius:46px;overflow:hidden;box-shadow:0 20px 60px rgba(0,0,0,.5),0 0 0 6px #111; }}
    .device .bg{{position:absolute;inset:0;background-image:url("__MOBILE_BG__");background-size:cover;background-position:center;opacity:.22;filter:saturate(1.05) brightness(1.02);z-index:0}}

    /* Dynamic Island */
    .dynamic-island{{position:absolute;top:8px;left:50%;transform:translateX(-50%);width:126px;height:37px;background:#000;border-radius:19px;z-index:1200;box-shadow:0 2px 10px rgba(0,0,0,.25)}}

    /* iOS-like status bar (Figma spec) */
    .statusbar{{position:absolute;top:8px;left:12px;right:12px;height:22px;display:flex;justify-content:space-between;align-items:center;padding:0 16px;background:transparent;border-radius:0;box-shadow:none;z-index:1000}}
    .sb-left{{display:flex;justify-content:center;align-items:center;width:40.333px;height:13px;line-height:13px;font-weight:600;font-size:14px;letter-spacing:.2px;color:#111;flex-shrink:0;font-feature-settings:"tnum";position:relative;top:1px}}
    .sb-right{{display:flex;align-items:center;gap:12px;flex-shrink:0}}
    .sb-cell, .sb-wifi, .sb-batt{{display:block;flex-shrink:0}}
    .sb-cell{{width:17px;height:11.333px}}
    .sb-wifi{{width:16.681px;height:12px}}
    .sb-batt{{width:25px;height:12px}}

    /* Global typography smoothing + premium feel */
    html, body{{-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale;text-rendering:optimizeLegibility}}

    /* Background gradient layer */
    body::after{{
      content:""; position:fixed; inset:0; z-index:-2;
      background:
        radial-gradient(140% 70% at 50% -10%, color-mix(in srgb, var(--brand) 12%, transparent), transparent 60%),
        linear-gradient(180deg, #ffffff 0%, rgba(255,255,255,.96) 40%, rgba(255,255,255,.92) 70%, rgba(255,255,255,.96) 100%);
      pointer-events:none;
    }}

    /* High-res hero image stack */
    .hero-stack{{position:absolute;inset:0;z-index:0;pointer-events:none;overflow:hidden;border-bottom-left-radius:28px;border-bottom-right-radius:28px;box-shadow:0 30px 60px rgba(0,0,0,.18)}}
    .hero-stack img{{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;image-rendering:auto;opacity:0;transition:opacity 900ms ease-in-out, transform 16s ease-in-out;transform:scale(1.06)}}
    .hero-stack img.active{{opacity:1;transform:scale(1.02)}}

    /* Top bar */
    .topbar{{position:absolute;top:40px;left:0;right:0;padding:12px 16px 8px;background:transparent;z-index:1300;height:56px}}
    .topbar-inner{{position:relative;height:100%}}
    .topbar .hamb{{justify-self:start}}
    .brand-chip{{position:absolute;left:50%;top:10px;transform:translateX(-50%);background:#fff;border:1px solid #00000012;padding:6px;border-radius:12px;box-shadow:0 6px 20px rgba(0,0,0,.08);margin:0;z-index:1002}}
    .brand-chip img{{height:28px;width:28px;display:block;object-fit:contain;image-rendering:auto;transform:translateZ(0);border-radius:6px}}
    .points{{position:absolute;right:16px;top:10px;display:flex;gap:6px;align-items:center;background:#fff;border-radius:22px;padding:8px 12px;box-shadow:0 6px 20px rgba(0,0,0,.08);font-weight:600;z-index:1101}}
    .hamb{{position:absolute;left:16px;top:10px;width:36px;height:36px;border-radius:12px;background:#fff;display:flex;align-items:center;justify-content:center;box-shadow:0 6px 20px rgba(0,0,0,.08);z-index:1101}}
    .hamb span{{width:18px;height:2px;background:#111;border-radius:2px;display:block;position:relative}}
    .hamb span::before,.hamb span::after{{content:"";position:absolute;left:0;right:0;height:2px;background:#111;border-radius:2px}}
    .hamb span::before{{top:-6px}}.hamb span::after{{top:6px}}

    /* Primary heading near the lower portion (no hero circle at all) */
    h1{{margin:0 0 8px 0;font-size:26px;line-height:1.15;font-weight:800;letter-spacing:-.01em}}
    .subtitle{{color:var(--muted);font-size:15px}}

    /* Fixed CTA card at bottom */
    .cta{{position:absolute;left:50%;transform:translateX(-50%);bottom:calc(84px + env(safe-area-inset-bottom) + 8px);width:clamp(320px, 90%, 560px);background:#f4f7f6;border-radius:20px;padding:18px 16px 16px;box-shadow:0 14px 40px rgba(0,0,0,.18);border:1px solid #00000010;backdrop-filter: blur(8px);z-index:1000}}
    .cta-inner{{background:#fff;border-radius:16px;padding:14px 14px 16px;border:1px solid #00000010;box-shadow:0 8px 28px rgba(0,0,0,.06) inset}}
    .cta .close{{position:absolute;right:24px;top:20px;width:28px;height:28px;border-radius:14px;background:#f2f3f5;display:flex;align-items:center;justify-content:center;box-shadow:0 4px 12px rgba(0,0,0,.08);cursor:pointer}}
    .btn{{appearance:none;-webkit-appearance:none;border:0;border-radius:16px;padding:16px 18px;font-weight:700;display:block;width:100%;text-align:center;cursor:pointer}}
    .btn-primary{{background:var(--brand);color:#fff;box-shadow:0 10px 24px rgba(219,39,32,.35)}}
    .btn-ghost{{background:#f7f7f8;color:var(--ink);border:1px solid #e5e7eb}}
    .btn + .btn{{margin-top:12px}}
    .btn-row .btn{{margin-top:0 !important}}
    .tabs3{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}}
    .tabs3 .btn{{margin-top:0 !important;width:100%}}
    .progress-rail{{position:relative;height:12px;background:#e5e7eb;border-radius:9999px;overflow:hidden;width:100%}}
    .progress-fill{{position:absolute;left:0;top:0;bottom:0;background:color-mix(in srgb, var(--brand) 35%, transparent);border-radius:inherit;min-width:2px}}

    /* Tab bar */
    .tabbar{{position:absolute;left:0;right:0;bottom:0;height:84px;padding-bottom:env(safe-area-inset-bottom);background:#fff;border-top:1px solid #E7E8EB;display:flex;justify-content:space-around;align-items:center;z-index:1000}}
    .t-seg{{display:flex;flex-direction:column;align-items:center;gap:6px;color:#9aa0a6;font-size:12px;cursor:pointer}}
    .t-seg.active{{color:var(--brand)}}
    .t-ico{{width:28px;height:28px}}

    /* Hard reset: hide any legacy hero/circle nodes if present */
    [class*="hero"], [class*="Hero"], .hero, .hero *, .hero-circle, .hero-img, .media-circle, .circle, .banner, .cover, .card-hero, .media, .mock-hero, .promo-hero {{ display:none !important; }}
    /* Prevent any circular masks anywhere */
    img, picture, video, canvas {{ border-radius:0 !important; clip-path:none !important; -webkit-clip-path:none !important; -webkit-mask:none !important; mask:none !important; }}

    /* CTA greeting heading and mini-sub styles */
    .cta h2{{margin:0 0 6px 0;font-size:20px;line-height:1.2;font-weight:800;letter-spacing:-.01em}}
    .cta .mini-sub{{color:var(--muted);font-size:14px;margin-bottom:12px}}

    /* ---- Compatibility adjustments per spec ---- */
    /* 1) main-content height fix: only subtract status bar height */
    .main-content{{ 
      height: calc(100% - 54px); 
      position: relative; 
      overflow: hidden;
    }}

    /* 1) content panel pinned just above tab bar (bottom: 100px already correct) */
    .content-panel{{ 
      position: absolute; 
      bottom: 100px; 
      left: 0; 
      right: 0; 
    }}

    /* Pages for tab navigation */
    .pages{{position:absolute; inset:120px 0 120px 0; overflow:hidden; z-index:2}}
    .page{{position:absolute; inset:0; padding:18px 16px; overflow:auto; -webkit-overflow-scrolling:touch; opacity:0; transform:translateX(20px); transition:opacity 260ms ease, transform 260ms ease; pointer-events:none}}
    .page.active{{opacity:1; transform:none; pointer-events:auto}}

    /* Generic list rows (no navigation away) */
    .list{{background:#fff;border-radius:16px;box-shadow:0 10px 24px rgba(0,0,0,.06);overflow:hidden}}
    .list-row{{display:flex;justify-content:space-between;align-items:center;padding:14px 12px;border-bottom:1px solid #0000000D;cursor:pointer}}
    .list-row:last-child{{border-bottom:0}}

    /* Drawer */
    .overlay{{position:absolute; inset:0; background:rgba(17,17,17,.28); opacity:0; visibility:hidden; transition:opacity 200ms ease, visibility 200ms ease; z-index:1300; pointer-events:none}}
    .drawer{{position:absolute; top:0; bottom:0; left:0; width:78%; max-width:360px; background:#fff; box-shadow:0 20px 60px rgba(0,0,0,.22); transform:translateX(-100%); transition:transform 240ms ease; z-index:1400; padding:22px 18px 16px; border-top-right-radius:18px; border-bottom-right-radius:18px}}
    body.drawer-open .overlay{{opacity:1; visibility:visible; pointer-events:auto}}
    body.drawer-open .drawer{{transform:translateX(0)}}
    .drawer h3{{margin:10px 0 16px 0; font-weight:800; font-size:20px}}
    .drawer a{{display:flex; align-items:center; gap:12px; padding:12px 10px; border-radius:12px; color:#111; text-decoration:none}}
    .drawer a:hover{{background:#f6f7f9}}
    .drawer .signout{{position:absolute;left:18px;right:18px;bottom:16px}}

    /* Segmented control (Delivery | Pickup | Catering) */
    .segctrl{{display:inline-flex;background:#fff;border:1px solid #00000022;border-radius:14px;overflow:hidden;box-shadow:0 6px 16px rgba(0,0,0,.06)}}
    .segbtn{{padding:10px 14px;font-weight:700;color:var(--brand);cursor:pointer;user-select:none}}
    .segbtn.active{{background:var(--brand);color:#fff}}

    /* Horizontal scrollers */
    .hstrip{{
      overflow-x:auto; overflow-y:hidden; white-space:nowrap; scrollbar-width:thin;
      -webkit-overflow-scrolling:touch; overscroll-behavior-x: contain; overscroll-behavior-inline: contain;
      touch-action: pan-x; cursor: grab; user-select:none;
    }}

    /* Segmented control (Delivery | Pickup | Catering) */
    .segctrl{{display:inline-flex;background:#fff;border:1px solid #00000022;border-radius:14px;overflow:hidden;box-shadow:0 6px 16px rgba(0,0,0,.06)}}
    .segbtn{{padding:10px 14px;font-weight:700;color:var(--brand);cursor:pointer;user-select:none}}
    .segbtn.active{{background:var(--brand);color:#fff}}

    /* 2) Rewards badge forced to right if a legacy class is present */
    .points-badge{{ 
      position: absolute; 
      top: 20px; 
      right: 24px; 
      z-index:1002;
    }}

    /* 3) Menu button forced to left if a legacy class is present */
    .nav-button{{ 
      position: absolute; 
      top: 20px; 
      left: 24px; 
      z-index:1002;
    }}

    /* 4) Ensure logo container isn’t pushed off-screen on some embeds */
    .logo-container{{ 
      margin: 20px 0 20px; 
      position: relative; 
      z-index: 10; 
    }}

  </style>
</head>
<body>
  <div class="device">
    <div class="bg"></div>
    <div class="statusbar">
      <div class="sb-left">10:01</div>
      <div class="sb-right">
        <div class="sb-cell">
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="12" viewBox="0 0 18 12" fill="none">
            <rect x="0.333313" y="7.33334" width="3" height="4" rx="1" fill="black"/>
            <rect x="5" y="5.33334" width="3" height="6" rx="1" fill="black"/>
            <rect x="9.66663" y="2.66666" width="3" height="8.66667" rx="1" fill="#C1C1C5"/>
            <rect x="14.3333" width="3" height="11.3333" rx="1" fill="#C1C1C5"/>
          </svg>
        </div>
        <div class="sb-wifi">
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="13" viewBox="0 0 18 13" fill="none">
            <path d="M0.775791 4.28592C0.6384 4.16018 0.628597 3.94648 0.758313 3.81288C1.79053 2.74947 3.01994 1.89549 4.37904 1.29925C5.84113 0.657829 7.42111 0.32885 9.01772 0.33339C10.6143 0.337929 12.1924 0.675888 13.6508 1.32562C15.0065 1.92957 16.231 2.79053 17.2572 3.85979C17.3861 3.99413 17.3751 4.20776 17.237 4.33273L15.7418 5.68553C15.6037 5.81046 15.3909 5.7993 15.2607 5.6661C14.4868 4.87409 13.5692 4.235 12.5558 3.78354C11.4397 3.2863 10.232 3.02766 9.01006 3.02418C7.78817 3.02071 6.57902 3.27248 5.46005 3.76338C4.44413 4.20907 3.52289 4.84293 2.74447 5.63053C2.61353 5.76298 2.40068 5.77293 2.2633 5.64718L0.775791 4.28592Z" fill="black"/>
            <path d="M3.87044 7.40251C3.72922 7.28112 3.71256 7.06759 3.84053 6.9323C4.46111 6.27625 5.20217 5.74417 6.02436 5.36574C6.9488 4.94028 7.95406 4.71904 8.9717 4.71711C9.98935 4.71519 10.9955 4.93263 11.9215 5.35462C12.7451 5.72993 13.4882 6.25919 14.1112 6.91288C14.2397 7.0477 14.2238 7.26129 14.0831 7.38319L12.8007 8.4939C12.6599 8.61584 12.4477 8.59959 12.3161 8.4679C11.9183 8.07013 11.4519 7.74613 10.9383 7.51214C10.3223 7.23144 9.65311 7.08681 8.97619 7.0881C8.29927 7.08938 7.63057 7.23654 7.01566 7.51956C6.503 7.75551 6.03778 8.08122 5.64155 8.48052C5.51038 8.6127 5.29831 8.62977 5.15705 8.50837L3.87044 7.40251Z" fill="black"/>
            <path d="M11.3696 10.0885C11.5012 9.95687 11.5022 9.74203 11.359 9.62282C11.0723 9.38398 10.7506 9.18934 10.4046 9.04602C9.95455 8.85961 9.4722 8.76367 8.9851 8.76367C8.49799 8.76367 8.01564 8.85961 7.5656 9.04602C7.21958 9.18934 6.89791 9.38398 6.61115 9.62282C6.46803 9.74203 6.46894 9.95687 6.60063 10.0885L8.74665 12.2346C8.87833 12.3663 9.09186 12.3663 9.22354 12.2346L11.3696 10.0885Z" fill="black"/>
          </svg>
        </div>
        <div class="sb-batt">
          <svg xmlns="http://www.w3.org/2000/svg" width="26" height="13" viewBox="0 0 26 13" fill="none">
            <rect x="1.16663" y="1.16666" width="21.6667" height="11" rx="2.83333" stroke="#919194"/>
            <rect x="2.66663" y="2.66666" width="11.3333" height="8" rx="1.66667" fill="black"/>
            <path d="M24.3333 8.66666C24.5084 8.66666 24.6818 8.61492 24.8436 8.51442C25.0053 8.41391 25.1523 8.26659 25.2761 8.08087C25.4 7.89515 25.4982 7.67467 25.5652 7.43202C25.6322 7.18937 25.6667 6.9293 25.6667 6.66666C25.6667 6.40401 25.6322 6.14394 25.5652 5.90129C25.4982 5.65864 25.4 5.43816 25.2761 5.25244C25.1523 5.06673 25.0053 4.91941 24.8436 4.8189C24.6818 4.71839 24.5084 4.66666 24.3333 4.66666L24.3333 6.66666L24.3333 8.66666Z" fill="#79797B"/>
          </svg>
        </div>
      </div>
    </div>
    <div class="dynamic-island" aria-hidden="true"></div>
    <div class="topbar">
      <div class="topbar-inner">
        <div class="hamb" aria-label="Menu"><span></span></div>
        <div class="brand-chip">{logo_tag}</div>
        <div class="points"><span class="badge">★</span><span>269</span></div>
      </div>
    </div>
    <div class="hero-stack" aria-hidden="true">
      {''.join([f"<img src='{_safe(u)}' alt='hero {i+1}' class='{'active' if i==0 else ''}' loading='{'eager' if i==0 else 'lazy'}' decoding='async'/>" for i,u in enumerate(hero_imgs[:4])])}
      {'' if hero_imgs else f"<img src='{_safe(mobile_bg)}' alt='bg' class='active'/>"}
    </div>

    

    <div class="pages">
      <section class="page page-menu active" data-page="menu">
        <div class="segwrap" style="display:flex;justify-content:center;margin:0 0 10px 0">
          <div class="segctrl" role="tablist" aria-label="Order method">
            <div class="segbtn active" data-mode="delivery" role="tab" aria-selected="true">Delivery</div>
            <div class="segbtn" data-mode="pickup" role="tab" aria-selected="false">Pickup</div>
            <div class="segbtn" data-mode="catering" role="tab" aria-selected="false">Catering</div>
          </div>
        </div>
        <div class="card" style="background:#fff;border-radius:16px;padding:14px;box-shadow:0 10px 24px rgba(0,0,0,.08)">
          <h3 style="margin:0 0 6px 0;font-weight:800;letter-spacing:-.01em">Featured</h3>
          <p style="margin:0;color:#666">Popular picks handpicked for you.</p>
          <div class="hstrip" style="display:flex;gap:12px;overflow-x:auto;overflow-y:hidden;padding:8px 2px 2px 2px">
            {"".join([
                f"<div style='flex:0 0 200px;display:flex;flex-direction:column'>"
                f"<div style='border-radius:14px;overflow:hidden;height:110px;background:#f3f4f6'>"
                f"<img src='{_safe(it.get('img') or '')}' alt='{_safe(it.get('name') or '')}' "
                f"style='display:block;width:100%;height:100%;object-fit:cover' loading='lazy' decoding='async' "
                f"onerror=\"this.style.display='none'\"></div>"
                f"<div style='margin-top:8px;font-weight:600;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden'>{_safe(it.get('name') or '')}</div>"
                f"<div style='font-size:12px;color:#6b7280;display:-webkit-box;-webkit-line-clamp:1;-webkit-box-orient:vertical;overflow:hidden'>{_safe(it.get('desc') or '')}</div>"
                f"</div>"
                for it in menu_items[:3]
              ])}
          </div>
        </div>
        <div class="card" style="background:#fff;border-radius:16px;padding:14px;box-shadow:0 10px 24px rgba(0,0,0,.08);margin-top:12px">
          <h3 style="margin:0 0 6px 0;font-weight:800;letter-spacing:-.01em">Menu</h3>
          <div style="display:flex;flex-direction:column;gap:10px">
            {"".join([
              f"<div style='display:flex;gap:12px;align-items:flex-start'>"
              f"<div style='width:64px;height:64px;border-radius:12px;overflow:hidden;background:#f3f4f6;flex-shrink:0'>"
              f"<img src='{_safe(it.get('img') or '')}' alt='{_safe(it.get('name') or '')}' style='width:100%;height:100%;object-fit:cover' loading='lazy' decoding='async' onerror=\"this.style.display=\\'none\\'\">"
              f"</div>"
              f"<div style='flex:1'>"
              f"<div style='display:flex;justify-content:space-between;gap:8px'>"
              f"<div style='font-weight:700'>{_safe(it.get('name') or '')}</div>"
              f"<div style='font-weight:700;white-space:nowrap'>{_safe(it.get('price') or '')}</div>"
              f"</div>"
              f"<div style='color:#6b7280;font-size:13px;margin-top:2px'>{_safe(it.get('desc') or '')}</div>"
              f"</div>"
              f"</div>"
              for it in menu_items
            ])}
          </div>
        </div>
        {f"<div class='card' style='margin-top:12px;background:#fff;border-radius:16px;padding:14px;box-shadow:0 10px 24px rgba(0,0,0,.08)'><h3 style='margin:0 0 8px 0;font-weight:800'>Find us</h3><div style='position:relative'><img src='{_safe(map_img)}' alt='map' style='width:100%;height:190px;object-fit:cover;border-radius:14px;box-shadow:0 10px 30px rgba(0,0,0,.12), inset 0 0 0 2px color-mix(in srgb, var(--brand) 18%, transparent)' loading='lazy' decoding='async'/><a href='{_safe(details.get('map_url') or details.get('website') or '#')}' target='_blank' rel='noopener' style='position:absolute;right:10px;bottom:10px;background:var(--brand);color:#fff;border-radius:9999px;padding:8px 12px;font-weight:700;text-decoration:none;box-shadow:0 8px 18px rgba(0,0,0,.18)'>Open in Maps</a></div><div style='margin-top:8px;color:#6b7280'>{_safe(details.get('address') or '')}</div></div>" if map_img else ''}
      </section>
      <section class="page page-rewards" data-page="rewards">
        <div class="card" style="background:#fff;border-radius:16px;padding:16px;box-shadow:0 10px 24px rgba(0,0,0,.08)">
          <div style="display:flex;align-items:center;justify-content:space-between">
            <h3 style="margin:0;font-weight:800">Deals & Rewards</h3>
            <div style="font-weight:700;color:var(--brand)">0</div>
          </div>
          <div class="tabs3" style="margin:12px 0 16px 0">
            <button class="btn btn-ghost" style="white-space:nowrap">My Points</button>
            <button class="btn btn-primary" style="white-space:nowrap">My Status</button>
            <button class="btn btn-ghost" style="white-space:nowrap">Deals</button>
          </div>
          <div style="text-align:center">
            <div style="font-weight:800;font-size:20px;margin-bottom:2px">Bronze</div>
            <div style="color:#6b7280">Tier</div>
            <div style="margin:12px 0 6px 0">
              <div class="progress-rail"><div class="progress-fill" style="width:28%"></div></div>
              <div style="display:flex;justify-content:space-between;font-size:12px;color:#6b7280;margin-top:4px">
                <span>0</span><span>2800</span>
              </div>
            </div>
            <div style="margin:8px 0 10px 0;font-size:13px;color:#6b7280">Tier Resets on : 12/31/2025</div>
            <div style="background:#FEF3C7;border:1px solid #FDE68A;color:#92400E;padding:10px 12px;border-radius:12px;display:inline-block">Reach the milestone to secure tier.</div>
            <div style="margin-top:12px"><button class="btn btn-ghost" style="font-weight:700">View Benefits</button></div>
            <div style="margin-top:6px;color:#6b7280">Lifetime points: <strong>0</strong></div>
          </div>
        </div>
      </section>
      <section class="page page-cart" data-page="cart">
        <div class="card" style="background:#fff;border-radius:16px;padding:16px;box-shadow:0 10px 24px rgba(0,0,0,.08)">
          <div style="display:flex;align-items:center;justify-content:space-between"><h3 style="margin:0;font-weight:800">Review order (1)</h3><div style="font-weight:700">5★</div></div>
          <!-- Store panel: brand-tinted light background with real address -->
          <div style="margin-top:10px;background:color-mix(in srgb, var(--brand) 14%, #ffffff);color:#111;border-radius:14px;padding:12px 12px 10px;border:1px solid #00000010">
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
              <div>
                <div style="opacity:.8">Store</div>
                <div style="margin-top:4px;background:#fff;color:#111;border-radius:12px;padding:10px 12px;display:flex;align-items:center;gap:8px;border:1px solid #00000010">
                  <span style="display:inline-block;width:10px;height:10px;border-radius:9999px;background:var(--brand)"></span>
                  {_safe(details.get('address') or details.get('name') or 'Store address')}
                </div>
              </div>
              <div>
                <div style="opacity:.8">Pickup method</div>
                <div style="margin-top:4px;background:#fff;color:#111;border-radius:12px;padding:10px 12px;border:1px solid #00000010">In store</div>
              </div>
              <div>
                <div style="opacity:.8">Pickup time</div>
                <div style="margin-top:4px;background:#fff;color:#111;border-radius:12px;padding:10px 12px;border:1px solid #00000010">4–7 mins</div>
              </div>
            </div>
          </div>
          <!-- Order item: use first real menu item when available -->
          <div style="display:flex;align-items:flex-start;gap:12px;margin:14px 0 8px 0">
            <div style="width:56px;height:56px;border-radius:12px;background:#f3f4f6;overflow:hidden">
              { (f"<img src='{_safe((menu_items[0] or {}).get('img') or '')}' alt='item' style='width:100%;height:100%;object-fit:cover' onerror=\"this.style.display='none'\">" if (menu_items and (menu_items[0] or {}).get('img')) else "") }
            </div>
            <div style="flex:1">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <strong>{_safe((menu_items[0] or {}).get('name') or 'Selected item')}</strong>
                <strong>{_safe((menu_items[0] or {}).get('price') or '$0.00')}</strong>
              </div>
              <div style="color:#6b7280;font-size:13px">{_safe((menu_items[0] or {}).get('desc') or '')}</div>
            </div>
          </div>
          <div style="margin:10px 0 6px 0;color:#6b7280;font-weight:600">YOU MAY ALSO LIKE</div>
          <div style="display:flex;gap:10px;flex-wrap:wrap;padding-bottom:4px;overflow:visible">
            {"".join([f"<button class='btn btn-ghost' style='min-width:160px'>+ {_safe(it.get('name') or '')}</button>" for it in (menu_items[1:4] if menu_items else [])])}
          </div>
          <div style="margin-top:12px;border-top:1px dashed #e5e7eb;padding-top:10px">
            <div style="display:flex;justify-content:space-between;margin:2px 0"><span>Subtotal</span><span>{_safe((menu_items[0] or {}).get('price') or '$0.00')}</span></div>
            <div style="display:flex;justify-content:space-between;margin:2px 0"><span>Tax</span><span>—</span></div>
            <div style="display:flex;justify-content:space-between;margin:6px 0;font-weight:800"><span>Total</span><span>{_safe((menu_items[0] or {}).get('price') or '$0.00')}</span></div>
            <div style="margin-top:8px"><button class="btn btn-primary" style="width:100%">Checkout</button></div>
          </div>
        </div>
      </section>
      <section class="page page-profile" data-page="profile">
        <div class="card" style="background:#fff;border-radius:16px;padding:16px;box-shadow:0 10px 24px rgba(0,0,0,.08)">
          <h3 style="margin:0 0 12px 0;font-weight:800">Account</h3>
          <div style="font-weight:700">Donald Trump</div>
          <div style="color:#6b7280">badboydt@gmail.com</div>
          <div style="height:12px"></div>
          <div style="display:flex;flex-direction:column;gap:8px">
            <a href="#" class="btn btn-ghost" style="text-align:left;text-decoration:none;font-weight:400">Profile</a>
            <a href="#" class="btn btn-ghost" style="text-align:left;text-decoration:none;font-weight:400">Personal info</a>
            <a href="#" class="btn btn-ghost" style="text-align:left;text-decoration:none;font-weight:400">Cards & payment</a>
            <a href="#" class="btn btn-ghost" style="text-align:left;text-decoration:none;font-weight:400">Transaction history</a>
            <a href="#" class="btn btn-ghost" style="text-align:left;text-decoration:none;font-weight:400">Privacy and data</a>
          </div>
          
          <div style="margin:10px 0 6px 0;font-weight:800">Help & policies</div>
          <div style="display:flex;flex-direction:column;gap:8px">
            <a href="#" class="btn btn-ghost" style="text-align:left;text-decoration:none;font-weight:400">Help</a>
            <a href="#" class="btn btn-ghost" style="text-align:left;text-decoration:none;font-weight:400">Application Terms</a>
            <a href="#" class="btn btn-ghost" style="text-align:left;text-decoration:none;font-weight:400">Privacy Notice</a>
            <a href="#" class="btn btn-ghost" style="text-align:left;text-decoration:none;font-weight:400">Delete account</a>
            <a href="#" class="btn btn-ghost" style="text-align:left;text-decoration:none;font-weight:400">Do Not Share My Personal Information</a>
            <a href="#" class="btn btn-primary" style="text-align:center">Sign out</a>
          </div>
          <div style="height:12px"></div>
          <div style="margin-top:10px;text-align:center;color:#6b7280;font-size:.5em">v1.001 – Made with <span style="color:{pal['primary']}">♥</span> by <strong>Restronaut.ai</strong></div>
        </div>
      </section>
    </div>

    <nav class="tabbar" role="navigation" aria-label="Tab bar">
      <div class="t-seg active" data-tab="menu">
        <svg class="t-ico" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M4 7h16v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V7Z" stroke="currentColor" stroke-width="1.5"/><path d="M8 7V5a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" stroke="currentColor" stroke-width="1.5"/><path d="M10 12h4M10 16h4" stroke="currentColor" stroke-width="1.5"/></svg>
        <div>Menu</div>
      </div>
      <div class="t-seg" data-tab="rewards">
        <svg class="t-ico" viewBox="0 0 24 24" fill="none"><path d="M6 12l4 4 8-8" stroke="currentColor" stroke-width="1.5" fill="none"/><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.5" fill="none"/></svg>
        <div>Rewards</div>
      </div>
      <div class="t-seg" data-tab="cart">
        <svg class="t-ico" viewBox="0 0 24 24" fill="none"><path d="M3 6h18l-2 12H5L3 6Z" stroke="currentColor" stroke-width="1.5"/><circle cx="9" cy="20" r="1.5" fill="currentColor"/><circle cx="15" cy="20" r="1.5" fill="currentColor"/></svg>
        <div>Cart</div>
      </div>
      <div class="t-seg" data-tab="profile">
        <svg class="t-ico" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="8" r="4" stroke="currentColor" stroke-width="1.5"/><path d="M4 20a8 8 0 0 1 16 0" stroke="currentColor" stroke-width="1.5"/></svg>
        <div>Profile</div>
      </div>
    </nav>
    <div class="overlay"></div>
    <aside class="drawer" role="dialog" aria-modal="true" aria-label="Menu">
      <h3>Explore</h3>
      <a href="#" data-nav="menu">Menu</a>
      <a href="#" data-nav="rewards">Rewards</a>
      <a href="#" data-nav="cart">Cart</a>
      <a href="#" data-nav="profile">Profile</a>
      <div class="signout">
        <button class="btn btn-primary btn-signout" style="width:100%;font-weight:700">Sign Out</button>
      </div>
    </aside>
    </div>
  </div>
    <script>
      // Haptic-like feedback and simple SPA router
      (function(){{ 
        const buttons = document.querySelectorAll('.btn');
        buttons.forEach(b=>{{ 
          b.addEventListener('click', () => {{ 
            b.style.transform='scale(0.98)';
            setTimeout(() => {{ b.style.transform='scale(1)'; }}, 140);
          }});
        }});
        const tabs = Array.from(document.querySelectorAll('.t-seg'));
        const pages = Array.from(document.querySelectorAll('.page'));
        function activate(page){{
          tabs.forEach(x=>x.classList.toggle('active', x.getAttribute('data-tab')===page));
          pages.forEach(p=>p.classList.toggle('active', p.getAttribute('data-page')===page));
        }}
        tabs.forEach(t=>{{ t.addEventListener('click', () => {{ activate(t.getAttribute('data-tab')); }}); }});
        document.querySelectorAll('[data-nav]')?.forEach(el=>{{
          el.addEventListener('click', (e)=>{{ e.preventDefault(); activate(el.getAttribute('data-nav')); }});
        }});
        // Segmented control toggle (Delivery | Pickup | Catering)
        (function(){{
          const root = document.querySelector('.segctrl');
          if(!root) return;
          const btns = Array.from(root.querySelectorAll('.segbtn'));
          btns.forEach(b=>{{
            b.addEventListener('click', ()=>{{
              btns.forEach(x=>{{ x.classList.remove('active'); x.setAttribute('aria-selected','false'); }});
              b.classList.add('active'); b.setAttribute('aria-selected','true');
              document.body.dataset.fulfillment = b.getAttribute('data-mode') || '';
            }});
          }});
        }})();
        // Ensure mouse-wheel scroll works inside the active page (desktop browser preview)
        const device = document.querySelector('.device');
        device?.addEventListener('wheel', (e)=>{{
          // If a horizontal strip handled the event, or the target is inside one, skip vertical scroll
          if (e.defaultPrevented) return;
          const t = e.target;
          if (t && t.closest && t.closest('.hstrip')) return;
          const active = document.querySelector('.page.active');
          if(active){{
            active.scrollTop += e.deltaY;
            e.preventDefault();
          }}
        }}, {{passive:false}});
        // Horizontal wheel-to-scroll for featured strips
        Array.from(document.querySelectorAll('.hstrip')).forEach(strip=>{{
          strip.addEventListener('wheel', (e)=>{{
            if(Math.abs(e.deltaY) > Math.abs(e.deltaX)){{
              strip.scrollLeft += e.deltaY;
              e.preventDefault();
            }}
          }}, {{passive:false}});
          // Pointer drag-to-scroll (prevents accidental page actions)
          let isDown=false, startX=0, startLeft=0, moved=false, pid=null;
          strip.addEventListener('pointerdown', (e)=>{{
            isDown=true; moved=false; pid=e.pointerId; startX=e.clientX; startLeft=strip.scrollLeft;
            strip.setPointerCapture?.(pid); strip.style.cursor='grabbing'; e.preventDefault();
          }});
          strip.addEventListener('pointermove', (e)=>{{
            if(!isDown) return; const dx=e.clientX-startX; if(Math.abs(dx)>3) moved=true; strip.scrollLeft=startLeft-dx; e.preventDefault();
          }});
          const endHandler=(e)=>{{ if(!isDown) return; isDown=false; strip.releasePointerCapture?.(pid); strip.style.cursor='grab'; if(moved) {{ e.preventDefault(); e.stopPropagation(); }} }};
          strip.addEventListener('pointerup', endHandler); strip.addEventListener('pointercancel', endHandler); strip.addEventListener('pointerleave', endHandler);
          // If the user dragged, cancel the click so we don't trigger other UI
          strip.addEventListener('click', (e)=>{{ if(moved) {{ e.preventDefault(); e.stopPropagation(); }} }});
        }});
        // Keep all in-device anchors from navigating the outer page
        document.querySelectorAll('.device a[href="#"]').forEach(a=>{{ a.addEventListener('click', (e)=> e.preventDefault()); }});
        // Logo auto-scaler for arbitrary formats/aspect ratios
        (function(){{
          const logo = document.querySelector('.brand-chip img');
          if(!logo) return;
          function fit(){{
            try{{
              const w = logo.naturalWidth||0, h = logo.naturalHeight||0;
              if(!w||!h) return;
              if(h/w > 1.2) logo.style.maxHeight = '70px';
              if(w/h > 3.5) logo.style.maxWidth = '200px';
              if(Math.max(w,h) > 1000) logo.style.imageRendering = 'auto';
            }}catch(e){{}}
          }}
          if(logo.complete) fit(); else logo.addEventListener('load', fit);
        }})();
        // Hero crossfade
        (function(){{
          const imgs = Array.from(document.querySelectorAll('.hero-stack img'));
          if(imgs.length<=1) return;
          let i=0; setInterval(()=>{{ imgs[i].classList.remove('active'); i=(i+1)%imgs.length; imgs[i].classList.add('active'); }}, 5000);
        }})();
        // Drawer toggle
        const hamb = document.querySelector('.hamb');
        const overlay = document.querySelector('.overlay');
        hamb?.addEventListener('click', ()=>{{ document.body.classList.add('drawer-open'); }});
        overlay?.addEventListener('click', ()=>{{ document.body.classList.remove('drawer-open'); }});
        // Sign out button (close drawer for now)
        document.querySelector('.btn-signout')?.addEventListener('click', ()=>{{
          document.body.classList.remove('drawer-open');
          // Hook up your real sign out flow here if needed
        }});
        // CTA dismiss + bottom sheet
        const cta = document.querySelector('.cta');
        cta?.querySelector('.close')?.addEventListener('click', ()=>{{ cta.style.opacity='0'; cta.style.transform='translateX(-50%) translateY(6px)'; setTimeout(()=>cta.remove(), 180); }});
        const sheet = document.getElementById('actionSheet');
        document.querySelectorAll('[data-sheet-open]')?.forEach(el=>{{ el.addEventListener('click', ()=>{{ sheet?.classList.add('open'); }}); }});
        sheet?.querySelector('.close')?.addEventListener('click', ()=>{{ sheet?.classList.remove('open'); }});
        // Subtle parallax for main content
        window.addEventListener('scroll', () => {{ 
          const scrolled = window.pageYOffset || document.documentElement.scrollTop || 0;
          const el = document.querySelector('.main-content');
          if(el) el.style.transform = `translateY(${{scrolled * 0.06}}px)`;
        }});
      }})();
    </script>
</body>
</html>
"""
    # Replace the SVG background placeholder with the actual data URI
    html = html.replace("__MOBILE_BG__", _safe(mobile_bg))

    meta = {"name": name, "brand": pal["primary"], "logo": logo}
    return html, meta
