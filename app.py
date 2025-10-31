# app.py
import os, re, time, json, hashlib, base64, io, hmac, asyncio, logging, json, mimetypes, copy
from urllib.parse import urlparse, quote
from typing import Optional, Literal, Dict, Any, List, Tuple, FrozenSet
from pathlib import Path
from mobile import build_mobile_app_html
from website import build_html
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
# OpenAI (optional) for narrative analysis
try:
    from openai import OpenAI  # SDK v1+
except Exception:
    OpenAI = None

# Optional: SVG → PNG rasterization
try:
    import cairosvg  # type: ignore
except Exception:
    cairosvg = None
    
try:
    import json5
except ImportError:
    json5 = None

try:
    from fix_busted_json import repair_json as fix_busted_repair
except ImportError:
    fix_busted_repair = None

try:
    from json_repair import repair_json as json_repair_repair
except ImportError:
    json_repair_repair = None
        
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY  = os.getenv("GOOGLE_PLACES_API_KEY", "")
YELP_API_KEY    = os.getenv("YELP_API_KEY", "")  # optional fallback
PEXELS_API_KEY  = os.getenv("PEXELS_API_KEY", "")  # optional image fallback

ALLOWED_ORIGINS= [
    "*", 
    "http://localhost:5173", 
    "https://restrowb.onrender.com", 
    "https://zp1v56uxy8rdx5ypatb0ockcb9tr6a-oci3--5173--96435430.local-credentialless.webcontainer-api.io/"
]
BACKEND_API_KEY = os.getenv("BACKEND_API_KEY", "")

RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
CACHE_TTL_SUGGEST  = int(os.getenv("CACHE_TTL_SUGGEST", "180"))
CACHE_TTL_DETAILS  = int(os.getenv("CACHE_TTL_DETAILS", "3600"))
DEFAULT_RADIUS_M   = int(os.getenv("DEFAULT_RADIUS_M", "50000"))
CHAIN_RADIUS_M     = int(os.getenv("CHAIN_RADIUS_M", "50000"))

GOOGLE_CSE_KEY = os.getenv("GOOGLE_CSE_KEY", os.getenv("GOOGLE_API_KEY", ""))
GOOGLE_CSE_CX  = os.getenv("GOOGLE_CSE_CX", "")

# Logging
log = logging.getLogger("uvicorn.error")
# Responsive container utility (used everywhere for consistent layout)
CONTAINER = "container-shell"
# Hero image mapping placeholder (kept for compatibility). We now source heroes from
# Google gallery and Pexels; these lists are intentionally empty.
HERO_BY_CUISINE = {
    "burger": [], "italian": [], "mexican": [], "american": [], "chinese": [],
    "japanese": [], "thai": [], "indian": [], "greek": [], "french": [],
    "korean": [], "mediterranean": [], "vietnamese": [],
}
# Safe, generic hero fallback: neutral SVG gradient (no external host)
HERO_FALLBACK_URL = (
    "data:image/svg+xml;utf8," + quote(
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900">'
        '<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%" stop-color="#f7f7f7"/><stop offset="100%" stop-color="#ededed"/>'
        '</linearGradient></defs><rect width="100%" height="100%" fill="url(#g)"/></svg>'
    )
)
# Project-relative path to the hard-coded mobile background SVG (overridable via env)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOBILE_BG_PATH = os.getenv(
    "MOBILE_BG_PATH",
    os.path.join(BASE_DIR, "assets", "mobile_bg.svg")
)

TEMPLATE_ASSETS_ROOT = Path(BASE_DIR) / "assets" / "Template_Assets"


def _file_to_data_uri(path: Path) -> Optional[str]:
    try:
        mime, _ = mimetypes.guess_type(str(path))
        if not mime or not mime.startswith("image/"):
            return None
        data = path.read_bytes()
        return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")
    except Exception:
        return None


def _slugify_menu_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _tokenize_menu_name(name: str) -> FrozenSet[str]:
    tokens = [tok for tok in re.split(r"[^a-z0-9]+", name.lower()) if tok]
    return frozenset(tokens)


def _load_template_assets() -> Dict[str, Dict[str, Any]]:
    assets: Dict[str, Dict[str, Any]] = {}
    if not TEMPLATE_ASSETS_ROOT.exists():
        return assets

    for cuisine_dir in TEMPLATE_ASSETS_ROOT.iterdir():
        if not cuisine_dir.is_dir():
            continue
        cuisine_key = cuisine_dir.name.lower()
        hero_images: List[str] = []
        menu_by_slug: Dict[str, str] = {}
        menu_by_tokens: Dict[FrozenSet[str], str] = {}

        for file_path in sorted(cuisine_dir.iterdir(), key=lambda p: p.name.lower()):
            if not file_path.is_file():
                continue
            uri = _file_to_data_uri(file_path)
            if not uri:
                continue
            stem = file_path.stem.lower()
            if stem.startswith("hero"):
                hero_images.append(uri)
                continue

            slug = _slugify_menu_name(stem)
            menu_by_slug[slug] = uri
            menu_by_tokens[_tokenize_menu_name(stem)] = uri

        if hero_images or menu_by_slug:
            assets[cuisine_key] = {
                "hero": hero_images,
                "menu_by_slug": menu_by_slug,
                "menu_by_tokens": menu_by_tokens,
            }

    return assets


TEMPLATE_ASSET_CACHE = _load_template_assets()


def get_template_hero_images(cuisine: str) -> List[str]:
    data = TEMPLATE_ASSET_CACHE.get((cuisine or "").lower()) or {}
    return list(data.get("hero") or [])


def get_template_menu_image(cuisine: str, name: str) -> Optional[str]:
    if not name:
        return None
    key = (cuisine or "").lower()
    data = TEMPLATE_ASSET_CACHE.get(key)
    if not data:
        return None

    slug = _slugify_menu_name(name)
    menu_by_slug: Dict[str, str] = data.get("menu_by_slug") or {}
    menu_by_tokens: Dict[frozenset, str] = data.get("menu_by_tokens") or {}

    if slug in menu_by_slug:
        return menu_by_slug[slug]
    if slug.endswith("s") and slug[:-1] in menu_by_slug:
        return menu_by_slug[slug[:-1]]
    if slug + "s" in menu_by_slug:
        return menu_by_slug[slug + "s"]

    token_key = _tokenize_menu_name(name)
    if token_key in menu_by_tokens:
        return menu_by_tokens[token_key]
    return None


def hydrate_cuisine_assets_with_templates() -> None:
    """Inject template-based hero and menu imagery into CUISINE_ASSETS defaults."""
    try:
        assets_map = TEMPLATE_ASSET_CACHE
        if not assets_map:
            return
        for cuisine, info in CUISINE_ASSETS.items():
            key = cuisine.lower()
            tmpl = assets_map.get(key) or {}

            hero_imgs = tmpl.get("hero") or []
            if hero_imgs:
                info["hero"] = list(hero_imgs)

            menu_items = info.get("menu") or []
            hydrated: List[Dict[str, Any]] = []
            for item in menu_items:
                name = item.get("name") or ""
                img_uri = get_template_menu_image(cuisine, name)
                if img_uri:
                    hydrated.append({**item, "img": img_uri})
                else:
                    hydrated.append({**item})
            if hydrated:
                info["menu"] = hydrated
    except Exception as exc:
        log.warning("Failed to hydrate cuisine assets with templates: %s", exc)
# Cache for inlined SVG bg
_MOBILE_BG_DATA_URI: Optional[str] = None

def get_mobile_bg_data_uri() -> str:
    """Return a data: URI for the hard-coded mobile background SVG.
    If utf-8 encoding fails, base64-encode. Returns empty string if read fails.
    """
    global _MOBILE_BG_DATA_URI
    if _MOBILE_BG_DATA_URI is not None:
        return _MOBILE_BG_DATA_URI
    try:
        with open(MOBILE_BG_PATH, "rb") as f:
            raw = f.read()
        try:
            # Prefer URL-encoded UTF-8 to keep size smaller than base64
            text = raw.decode("utf-8")
            _MOBILE_BG_DATA_URI = "data:image/svg+xml;utf8," + quote(text)
        except Exception:
            _MOBILE_BG_DATA_URI = "data:image/svg+xml;base64," + base64.b64encode(raw).decode("ascii")
    except Exception as e:
        log.warning("mobile bg svg could not be loaded: %s", e)
        _MOBILE_BG_DATA_URI = ""
    return _MOBILE_BG_DATA_URI

# Back-compat: some older/mobile builders reference this hero image constant.
# We point it to the same hard-coded SVG background (as a data URI). If the SVG
# cannot be read at runtime, fall back to a 1x1 transparent PNG data URI.
def _ensure_mobile_hero_universal() -> str:
    uri = get_mobile_bg_data_uri()
    if uri:
        return uri
    # 1x1 transparent PNG (base64)
    return (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    )
MOBILE_HERO_UNIVERSAL = _ensure_mobile_hero_universal()
# Compatibility shim: ensure build_mobile_app_html always accepts 'sales_cta' as keyword arg
def _normalize_build_mobile_signature():
    """Ensure build_mobile_app_html accepts keyword 'sales_cta' even if redefined later."""
    try:
        import inspect
        sig = inspect.signature(build_mobile_app_html)
        if 'sales_cta' not in sig.parameters:
            log.warning("build_mobile_app_html lacks 'sales_cta'; installing compatibility wrapper")
            orig = build_mobile_app_html
            async def _wrapped(details: Dict[str, Any], *, sales_cta: bool = True) -> Tuple[str, Dict[str, Any]]:
                return await orig(details)
            globals()['build_mobile_app_html'] = _wrapped
    except Exception as e:
        log.error("Failed to normalize build_mobile_app_html signature: %s", e)

# Run normalization at import time (before routes are invoked)
_normalize_build_mobile_signature()


# ────────────────────────────────────────────────────────────────────────────
# Legacy Mobile App Builder (device frame, circular hero)
# ────────────────────────────────────────────────────────────────────────────
async def build_mobile_app_html_legacy(details: Dict[str, Any], *, sales_cta: bool = True) -> Tuple[str, Dict[str, Any]]:
    # This is the legacy builder: device frame/circular hero, uses MOBILE_HERO_UNIVERSAL
    name = details.get("name") or "Restaurant"
    cuisine = cuisine_from_types(details)
    context = get_restaurant_context(details)
    logo, logo_color, logo_reason = await best_logo_with_color(details)
    pal = select_theme_colors(cuisine, context, logo_color)

    # Device frame/circular hero (legacy)
    hero_img = MOBILE_HERO_UNIVERSAL

    return "", {}
# ────────────────────────────────────────────────────────────────────────────
# App & shared HTTP client (lifespan)
# ────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = AsyncClient(
        limits=Limits(max_connections=100, max_keepalive_connections=20),
        timeout=Timeout(10.0, connect=5.0, read=7.0, write=5.0)
    )
    # Warn (don’t crash) if critical envs are missing
    if not BACKEND_API_KEY:
        log.warning("BACKEND_API_KEY is not set; requests will be rejected by security_guard.")
    if not GOOGLE_API_KEY:
        log.warning("GOOGLE_PLACES_API_KEY is not set; Google endpoints will fail.")
    try:
        # Log key presence (masked) for debugging
        k = os.getenv("OPENAI_API_KEY", "")
        if not k:
            log.warning("OPENAI_API_KEY not found in environment at startup")
        else:
            log.info("OPENAI_API_KEY detected (len=%d, masked=%s…%s)", len(k), k[:4], k[-4:])
        # Report CairoSVG availability (helps debug SVG rasterization path)
        try:
            import cairosvg as _csvg  # local import to confirm runtime env
            log.info("CairoSVG available: version=%s, path=%s", getattr(_csvg, '__version__', 'unknown'), getattr(_csvg, '__file__', 'unknown'))
        except Exception as e:
            log.info("CairoSVG NOT available in runtime env (%s)", e)
        yield
    finally:
        await app.state.http.aclose()

app = FastAPI(title="Restronaut Backend", version="4.4", lifespan=lifespan)

# CORS: require explicit origins (recommended in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
# Compression
app.add_middleware(GZipMiddleware, minimum_size=500)

# ────────────────────────────────────────────────────────────────────────────
# Utilities: TTL cache, rate limiting, http, cache key, safe
# ────────────────────────────────────────────────────────────────────────────
class TTLCache:
    def __init__(self, ttl_seconds: int = 300, max_items: int = 5000):
        self.ttl = ttl_seconds
        self.max_items = max_items
        self.store: Dict[str, Tuple[Any, float]] = {}

    def _now(self) -> float: return time.time()

    def get(self, key: str):
        item = self.store.get(key)
        if not item: return None
        value, ts = item
        if self._now() - ts > self.ttl:
            self.store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any):
        if len(self.store) >= self.max_items:
            # drop ~100 oldest entries
            for k, _ in sorted(self.store.items(), key=lambda kv: kv[1][1])[:100]:
                self.store.pop(k, None)
        self.store[key] = (value, self._now())


SUGGEST_CACHE = TTLCache(ttl_seconds=CACHE_TTL_SUGGEST)
DETAILS_CACHE = TTLCache(ttl_seconds=CACHE_TTL_DETAILS)
IMGSEARCH_CACHE = TTLCache(ttl_seconds=3600)
DETAILS_NORM_CACHE = TTLCache(ttl_seconds=CACHE_TTL_DETAILS)
NEARBY_CACHE = TTLCache(ttl_seconds=int(os.getenv("CACHE_TTL_NEARBY", "600")))
TEXTSEARCH_CACHE = TTLCache(ttl_seconds=int(os.getenv("CACHE_TTL_TEXTSEARCH", "900")))
CHAIN_COUNT_CACHE = TTLCache(ttl_seconds=int(os.getenv("CACHE_TTL_CHAIN", "900")))


# Cache for extracted logo colors to avoid repeated processing
COLOR_CACHE = TTLCache(ttl_seconds=24*3600)

# Thread pool for CPU-bound color extraction
color_extraction_executor = ThreadPoolExecutor(max_workers=4)

def _logo_bytes_to_png_bytes(raw: bytes, content_type: str) -> Optional[bytes]:
    """Return PNG-encoded bytes from raw logo bytes.
    - If SVG and cairosvg is available, rasterize to PNG.
    - Otherwise, attempt Pillow decode and re-encode to PNG.
    - Returns None if conversion fails.
    """
    ctype = (content_type or '').lower()
    # Handle SVG via CairoSVG when available
    if 'svg' in ctype:
        if cairosvg is None:
            log.warning("SVG logo detected but CairoSVG not installed; skipping color extraction")
            return None
        try:
            return cairosvg.svg2png(bytestring=raw)
        except Exception as e:
            log.warning(f"SVG to PNG conversion failed: {e}")
            return None
    # For other image types, try Pillow → PNG
    try:
        from PIL import Image
        im = Image.open(BytesIO(raw))
        if im.mode not in ('RGB', 'RGBA'):
            im = im.convert('RGBA') if 'A' in im.getbands() else im.convert('RGB')
        if im.mode == 'RGBA':
            bg = Image.new('RGB', im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        buf = BytesIO()
        im.save(buf, format='PNG')
        return buf.getvalue()
    except Exception as e:
        log.warning(f"Logo PNG convert failed: {e}")
        return None

def extract_dominant_color(image_data: bytes) -> str:
    """Extract the dominant color from an image using ColorThief."""
    try:
        buf = BytesIO(image_data)
        ct = ColorThief(buf)
        r, g, b = ct.get_color(quality=1)
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception as e:
        log.warning(f"Color extraction (colorthief) failed: {e}")
        return None

async def get_logo_color(url: str) -> Optional[str]:
    """Fetch logo and extract dominant color asynchronously (with caching)."""
    if not url:
        return None
    ck = cache_key('logo_color', url)
    cached = COLOR_CACHE.get(ck)
    if cached is not None:
        return cached
    try:
        resp = await app.state.http.get(url, timeout=8.0, follow_redirects=True)
        if resp.status_code != 200 or not resp.content:
            COLOR_CACHE.set(ck, None)
            return None

        ctype = (resp.headers.get('content-type') or '').lower()
        # Must be an image/* or svg+xml
        if not (ctype.startswith('image/') or 'svg' in ctype):
            COLOR_CACHE.set(ck, None)
            return None

        # Convert to PNG bytes (handles SVG via cairosvg; others via Pillow)
        img_bytes = _logo_bytes_to_png_bytes(resp.content, ctype)
        if not img_bytes:
            COLOR_CACHE.set(ck, None)
            return None

        loop = asyncio.get_running_loop()
        color = await loop.run_in_executor(color_extraction_executor, extract_dominant_color, img_bytes)
        COLOR_CACHE.set(ck, color)
        return color
    except Exception as e:
        log.warning(f"Failed to fetch logo for color extraction: {e}")
        COLOR_CACHE.set(ck, None)
        return None

def is_valid_color(color_str: str) -> bool:
    """Validate hex color of the form #rrggbb."""
    if not color_str or not isinstance(color_str, str) or not color_str.startswith('#'):
        return False
    if len(color_str) != 7:
        return False
    try:
        int(color_str[1:], 16)
        return True
    except Exception:
        return False

def darken_color(hex_color: str, factor: float = 0.2) -> str:
    """Darken a hex color by a factor in [0,1]."""
    try:
        h = hex_color.lstrip('#')
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        r = max(0, min(255, int(r * (1 - factor))))
        g = max(0, min(255, int(g * (1 - factor))))
        b = max(0, min(255, int(b * (1 - factor))))
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return hex_color

async def best_logo_with_color(details: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], str]:
    """Pick the best logo and extract a dominant color using ColorThief only, with validation."""
    logo_url, reason = best_logo(details)
    if logo_url and not await _logo_is_valid(logo_url, details):
        # Try a synthesized favicon for the website domain
        hp = _homepage(details.get('website'))
        if hp:
            host = urlparse(hp).netloc
            candidate = f"https://www.google.com/s2/favicons?sz=256&domain={host}"
            if await _logo_is_valid(candidate, details):
                logo_url, reason = candidate, 'validated_favicon'
            else:
                logo_url, reason = None, 'invalid_logo_filtered'
        else:
            logo_url, reason = None, 'invalid_logo_filtered'
    color = await get_logo_color(logo_url) if logo_url else None
    return logo_url, color, reason

_ip_hits: Dict[str, List[float]] = {}
MAX_IP_BUCKETS = 10_000

def rate_limit(request: Request):
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    window_start = now - 60.0

    if len(_ip_hits) > MAX_IP_BUCKETS and ip not in _ip_hits:
        # basic protection against memory bloat
        raise HTTPException(503, "Server busy")

    bucket = _ip_hits.setdefault(ip, [])
    while bucket and bucket[0] < window_start:
        bucket.pop(0)
    if len(bucket) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(429, "Rate limit exceeded.")
    bucket.append(now)

def security_guard(request: Request):
    # Require key to be set in prod; reject missing/invalid keys
    if not BACKEND_API_KEY:
        raise HTTPException(500, "Server misconfigured: missing BACKEND_API_KEY")
    sent = request.headers.get("x-api-key") or request.headers.get("X-API-Key") or ""
    if not hmac.compare_digest(sent, BACKEND_API_KEY):
        raise HTTPException(401, "Invalid or missing X-API-Key")

async def http_get_json(url: str, *, params: Dict[str, Any] = None, headers: Dict[str, str] = None, timeout: float = 9.0, retries: int = 2):
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = await app.state.http.get(url, params=params, headers=headers, timeout=timeout, follow_redirects=True)
            if resp.status_code != 200:
                txt = resp.text[:400]
                raise HTTPException(502, f"Upstream error {resp.status_code}: {txt}")
            return resp.json()
        except Exception as e:
            last_err = e
            await asyncio.sleep(0.25 * (attempt + 1))
    raise HTTPException(502, f"Upstream failed after retries: {last_err}")

async def http_get_text(url: str, *, params: Dict[str, Any] = None, headers: Dict[str, str] = None, timeout: float = 9.0, retries: int = 2, max_bytes: int = 300_000) -> str:
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = await app.state.http.get(url, params=params, headers=headers, timeout=timeout, follow_redirects=True)
            if resp.status_code != 200:
                txt = resp.text[:400]
                raise HTTPException(502, f"Upstream error {resp.status_code}: {txt}")
            b = resp.content[:max_bytes]
            return b.decode(resp.encoding or 'utf-8', errors='ignore')
        except Exception as e:
            last_err = e
            await asyncio.sleep(0.25 * (attempt + 1))
    raise HTTPException(502, f"Upstream text fetch failed after retries: {last_err}")

def cache_key(*parts: Any) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()

def safe(s: Optional[str]) -> str:
    return (s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

# ────────────────────────────────────────────────────────────────────────────
# URL / logo helpers
# ────────────────────────────────────────────────────────────────────────────
MARKETPLACE_HOST_PAT = re.compile(
    r"(yelp|doordash|ubereats|grubhub|postmates|seamless|toasttab|squareup|clover|foodhub|facebook|linktr\.ee|instagram|tiktok|google\.[^/]+)",
    re.I
)
GENERIC_ICON_PAT = re.compile(r"maps\.gstatic\.com/.*/place_api/icons", re.I)

def _clean_url(u: Optional[str]) -> Optional[str]:
    if not u: return None
    try:
        u = u.strip()
        parsed = urlparse(u if "://" in u else "https://" + u)
        return parsed.geturl()
    except Exception:
        return None

def _homepage(u: Optional[str]) -> Optional[str]:
    u = _clean_url(u)
    if not u: return None
    p = urlparse(u)
    if not p.scheme or not p.netloc: return None
    return f"{p.scheme}://{p.netloc}/"

def _is_marketplace(u: Optional[str]) -> bool:
    if not u: return False
    try:
        host = urlparse(u).netloc.lower()
    except Exception:
        return False
    return bool(MARKETPLACE_HOST_PAT.search(host))

def build_google_photo_url(photo_ref: str, maxwidth: int = 1600) -> str:
    return (
        "https://maps.googleapis.com/maps/api/place/photo"
        f"?maxwidth={maxwidth}"
        f"&photoreference={photo_ref}"
        f"&key={GOOGLE_API_KEY}"
    )

def guess_favicon_urls(homepage: Optional[str]) -> List[str]:
    hp = _homepage(homepage) if homepage else None
    out: List[str] = []
    if hp:
        host = urlparse(hp).netloc
        # Prefer high-res favicon proxy
        out.append(f"https://www.google.com/s2/favicons?sz=256&domain={host}")
        out.append(f"https://www.google.com/s2/favicons?sz=128&domain={host}")
        out.append(hp + "favicon.ico")
    return out

def _host(u: Optional[str]) -> str:
    try:
        return (urlparse(u).netloc or '').lower()
    except Exception:
        return ''

def _strip_www(h: str) -> str:
    return h[4:] if h.startswith('www.') else h

async def _logo_is_valid(url: Optional[str], details: Dict[str, Any]) -> bool:
    """Heuristically verify a logo URL is a small, square-ish icon for this brand."""
    if not url:
        return False
    u = url.strip()
    if GENERIC_ICON_PAT.search(u):
        return False

    # Prefer same domain as website, or Google s2 favicon proxy for that domain
    site = details.get('website') or details.get('map_url') or ''
    site_host = _strip_www(_host(site))
    u_host = _strip_www(_host(u))

    # Allow google s2 favicon for our domain, or same-domain favicon/*.png
    allowed = False
    if u_host in ('www.google.com', 'google.com') and 's2/favicons' in u:
        try:
            q = urlparse(u).query
            domain_param = None
            for part in q.split('&'):
                if part.startswith('domain='):
                    domain_param = part.split('=', 1)[1].lower()
                    break
            if site_host and domain_param:
                allowed = _strip_www(domain_param) == site_host
            else:
                allowed = True
        except Exception:
            allowed = True
    elif site_host and site_host == u_host:
        allowed = True

    # Fetch and verify it's an image and roughly square, not huge
    try:
        resp = await app.state.http.get(u, timeout=6.0, follow_redirects=True)
        if resp.status_code != 200:
            return False
        ctype = (resp.headers.get('content-type') or '').lower()
        if not ctype.startswith('image/'):
            return False
        if not resp.content:
            return False
        # Geometry checks
        try:
            from PIL import Image
            im = Image.open(BytesIO(resp.content))
            w, h = im.size
            if w <= 0 or h <= 0:
                return False
            ratio = max(w, h) / float(min(w, h))
            if ratio > 1.75:   # very non-square -> likely not a logo
                return False
            if max(w, h) < 32:  # too tiny
                return False
            if max(w, h) > 1024:  # too large -> likely a photo
                return False
        except Exception:
            # If PIL fails, accept based on headers only
            pass
        return True
    except Exception:
        return False

def _score_logo_candidate(url: str, reason: str, details: Dict[str, Any]) -> int:
    """Score a candidate; higher is better."""
    base = 0
    if reason == 'icon_mask_base_uri':
        base = 95
    elif reason == 'website_favicon':
        base = 88
    elif reason == 'icon_non_generic':
        base = 80
    elif reason == 'maps_url_favicon':
        base = 72
    elif reason == 'fallback_first':
        base = 60

    site_host = _strip_www(_host(details.get('website') or details.get('map_url') or ''))
    u_host = _strip_www(_host(url))
    if site_host and u_host == site_host:
        base += 8         # same domain bonus
    if 's2/favicons' in (url or '') and site_host:
        base += 5         # Google s2 favicon for our domain
    return base
# ────────────────────────────────────────────────────────────────────────────
# Safe Google Image Search + Unsplash helper
# ────────────────────────────────────────────────────────────────────────────
async def google_image_search(q: str, *, num: int = 3) -> Optional[str]:
    if not GOOGLE_CSE_KEY or not GOOGLE_CSE_CX:
        return None
    key = cache_key("imgsearch", q.strip().lower(), num)
    cached = IMGSEARCH_CACHE.get(key)
    if cached is not None:
        return cached
    params = {
        "key": GOOGLE_CSE_KEY,
        "cx": GOOGLE_CSE_CX,
        "q": q,
        "searchType": "image",
        "safe": "high",
        "imgSize": "large",
        "imgType": "photo",
        "num": min(max(num, 1), 10),
    }
    try:
        data = await http_get_json("https://www.googleapis.com/customsearch/v1", params=params, timeout=10.0)
        items = data.get("items") or []
        def ok(it):
            link = it.get("link") or ""
            mime = (it.get("mime") or "").lower()
            return link.startswith("https://") and ("jpeg" in mime or "jpg" in mime or "png" in mime)
        for it in items:
            if ok(it):
                IMGSEARCH_CACHE.set(key, it["link"])
                return it["link"]
        for it in items:
            link = it.get("link")
            if link and link.startswith(("http://", "https://")):
                IMGSEARCH_CACHE.set(key, link)
                return link
    except Exception:
        pass
    IMGSEARCH_CACHE.set(key, None)
    return None

async def ensure_valid_image_url(url: str) -> str:
    """Ensure image URL is valid and accessible"""
    if not url:
        return ""
    if url.startswith('data:image/'):
        return url
    if not url.startswith(('http://', 'https://')):
        return ""
    # Try HEAD with redirects first
    try:
        resp = await app.state.http.head(url, timeout=4.0, follow_redirects=True)
        if resp.status_code == 200:
            ctype = (resp.headers.get('content-type') or '').lower()
            if ctype.startswith('image/'):
                return url
    except Exception:
        pass
    # Some CDNs (or signed URLs) reject HEAD—fallback to a lightweight GET
    try:
        resp = await app.state.http.get(url, timeout=6.0, follow_redirects=True)
        if resp.status_code == 200:
            ctype = (resp.headers.get('content-type') or '').lower()
            if ctype.startswith('image/'):
                return url
    except Exception:
        pass
    return ""

# ────────────────────────────────────────────────────────────────────────────
# Pexels image search (optional)
# ────────────────────────────────────────────────────────────────────────────
async def pexels_search_photos(query: str, *, per_page: int = 3, orientation: str = "landscape") -> List[str]:
    if not PEXELS_API_KEY:
        return []
    try:
        headers = {"Authorization": PEXELS_API_KEY}
        params = {"query": query, "per_page": per_page, "orientation": orientation}
        data = await http_get_json("https://api.pexels.com/v1/search", params=params, headers=headers)
        out: List[str] = []
        for p in (data.get("photos") or []):
            src = p.get("src") or {}
            # Prefer high quality; fall back to any available
            for k in ("large2x", "large", "original", "medium"):
                u = src.get(k)
                if u:
                    out.append(u)
                    break
        return out
    except Exception:
        return []

async def pexels_first_image(query: str) -> Optional[str]:
    imgs = await pexels_search_photos(query, per_page=4)
    for u in imgs:
        v = await ensure_valid_image_url(u)
        if v:
            return v
    return None

# ────────────────────────────────────────────────────────────────────────────
# Menu image and hero image selection helpers (smarter)
# ────────────────────────────────────────────────────────────────────────────
async def select_hero_images(details: Dict[str, Any], cuisine: str) -> List[str]:
    """Smart hero image selection with fallbacks"""
    images: List[str] = []

    # 1) Use Google gallery photos first (do not over-filter; Places photo URLs lack descriptors)
    gal = details.get("gallery") or []
    for img_url in gal[:2]:
        images.append(img_url)

    # 1b) Add curated template heroes for this cuisine if available
    for hero_uri in get_template_hero_images(cuisine):
        if len(images) >= 4:
            break
        if hero_uri and hero_uri not in images:
            images.append(hero_uri)

    # 2) Try Pexels for a matching interior if we still need images
    try:
        name = (details.get("name") or "").strip()
        candidates = []
        if name:
            candidates.append(f"{name} restaurant interior")
        if cuisine:
            candidates.append(f"{cuisine} restaurant interior")
        candidates.extend(["restaurant interior", "dining interior restaurant"])
        for q in candidates:
            if len(images) >= 4:
                break
            px = await pexels_first_image(q)
            if px and px not in images:
                images.append(px)
    except Exception:
        pass

    if not images:
        images = [HERO_FALLBACK_URL]
    return images[:4]

async def _is_interior_shot(url: str) -> bool:
    """Quick heuristic to accept likely interior images"""
    interior_keywords = ["interior", "inside", "dining", "restaurant", "room", "space"]
    u = (url or "").lower()
    return any(k in u for k in interior_keywords)

async def resolve_menu_images(details: Dict[str, Any], cuisine: str) -> None:
    """Better menu image resolution with multiple fallbacks"""
    menu_src: List[Dict[str, Any]] = list(details.get("menu") or [])
    if not menu_src:
        cuisine_assets = CUISINE_ASSETS.get(cuisine, CUISINE_ASSETS[DEFAULT_CUISINE])
        menu_src = list(cuisine_assets.get("menu") or [])

    resolved: List[Dict[str, Any]] = []
    for item in menu_src:
        final_url = await _get_best_menu_image(item, cuisine)
        resolved.append({
            "name": item.get("name") or "Item",
            "desc": item.get("desc") or "",
            "price": item.get("price") or "",
            "img": final_url,
        })

    details["_resolved_menu"] = resolved[:6]

async def try_enrich_menu_from_site(details: Dict[str, Any], cuisine: str) -> None:
    """Best-effort: fetch menu items from the website.
    Looks for common patterns (names + $prices) and builds up to 6 items.
    Non-fatal; falls back to cuisine defaults if nothing is found.
    """
    if details.get("menu"):
        return
    hp = _homepage(details.get("website"))
    if not hp or _is_marketplace(hp):
        return
    candidates = [hp, hp.rstrip('/') + "/menu", hp.rstrip('/') + "/menus", hp.rstrip('/') + "/our-menu", hp.rstrip('/') + "/order-online"]
    html = ""
    for u in candidates:
        try:
            html = await http_get_text(u, timeout=6.0)
            if html and len(html) > 200:
                break
        except Exception:
            continue
    if not html:
        return
    # crude extraction: name near a price like $12 or $12.99
    try:
        import re as _re
        items = []
        # find lines with price and a nearby preceding name tag
        for m in _re.finditer(r"(?is)<[^>]*>([^<]{3,80})</[^>]*>[^$]{0,120}?\$(\d{1,2}(?:\.\d{2})?)", html):
            name = (m.group(1) or "").strip()
            price = "$" + m.group(2)
            if 3 <= len(name) <= 60 and not any(x in name.lower() for x in ("copyright","privacy","terms","menu")):
                items.append({"name": name, "price": price, "desc": "", "img": f"gq:{name} {cuisine}"})
            if len(items) >= 8:
                break
        if items:
            details["menu"] = items
    except Exception:
        return

async def _get_best_menu_image(item: Dict[str, Any], cuisine: str) -> str:
    """Get the best available image for a menu item"""
    img = (item.get("img") or "").strip()

    if img.startswith("data:image/"):
        return img

    # 1) Provided image if valid (explicit HTTP URL)
    if img and img.startswith(("http://", "https://")):
        v = await ensure_valid_image_url(img)
        if v:
            return v

    name = (item.get("name") or "").strip()
    template_img = get_template_menu_image(cuisine, name)
    if template_img:
        return template_img

    # 1) Google image search for specific dishes
    if img.startswith("gq:"):
        query = img[3:].strip()
        found = await google_image_search(f"{query} restaurant dish", num=3)
        if found:
            return found

    # 3) Try search by item name (Google CSE then Pexels)
    if name:
        try:
            found = await google_image_search(f"{name} {cuisine} food", num=3)
            if found:
                v = await ensure_valid_image_url(found)
                if v:
                    return v
        except Exception:
            pass
        try:
            px = await pexels_first_image(f"{name} {cuisine}")
            if px:
                v = await ensure_valid_image_url(px)
                if v:
                    return v
        except Exception:
            pass

    lname = name.lower()

    # 4) Item-keyword fallbacks via Pexels
    for kw in ["chicken", "fries", "salad", "steak", "pizza", "burger", "ramen", "sushi", "taco"]:
        if kw in lname:
            px = await pexels_first_image(f"{kw} dish")
            if px:
                v = await ensure_valid_image_url(px)
                if v:
                    return v

    # 5) Cuisine fallback dish images (last resort) from Pexels
    px = await pexels_first_image(f"{cuisine} dish") if cuisine else None
    if px:
        v = await ensure_valid_image_url(px)
        if v:
            return v
    return ""
# Update the CUISINE_ASSETS with more appropriate theme colors
CUISINE_ASSETS: Dict[str, Dict[str, Any]] = {
    "indian": {
        "palette": {"primary":"#D97706","primary_dark":"#B45309"},  # Warm saffron/turmeric
        "hero": [],
        "menu": [
            {"name":"Chicken Tikka Masala","desc":"Creamy tomato sauce, basmati rice","price":"$14.99","img": "gq:Chicken Tikka Masala indian"},
            {"name":"Chicken Biryani","desc":"Fragrant basmati rice, spices","price":"$16.99","img": "gq:Chicken Biryani indian"},
            {"name":"Garlic Naan","desc":"Fresh baked bread","price":"$3.99","img": "gq:Garlic Naan indian"}
        ]
    },
    "burger": {
        "palette": {"primary":"#DC2626","primary_dark":"#B91C1C"},  # Classic red
        "hero": [],
        "menu": [
            {"name":"Double Smash Burger","desc":"American cheese, pickles, shack sauce","price":"$10.99","img": "gq:Double Smash Burger burger"},
            {"name":"Crispy Chicken Sandwich","desc":"Buttermilk fried chicken, slaw","price":"$8.99","img": "gq:Crispy Chicken Sandwich burger"},
            {"name":"Crinkle Cut Fries","desc":"Sea salt, extra crispy","price":"$3.99","img": "gq:Crinkle Cut Fries burger"}
        ]
    },
    "italian": {
        "palette": {"primary":"#059669","primary_dark":"#047857"},  # Italian flag green
        "hero": [],
        "menu": [
            {"name":"Chicken Alfredo","desc":"Creamy parmesan sauce, fettuccine","price":"$14.99","img": "gq:Chicken Alfredo italian"},
            {"name":"Spaghetti & Meatballs","desc":"San Marzano tomatoes, basil","price":"$12.99","img": "gq:Spaghetti and Meatballs italian"},
            {"name":"Margherita Pizza","desc":"Fresh mozzarella, tomato, basil","price":"$11.49","img": "gq:Margherita Pizza italian"}
        ]
    },
    "mexican": {
        "palette": {"primary":"#EA580C","primary_dark":"#C2410C"},  # Vibrant orange-red
        "hero": [],
        "menu": [
            {"name":"Carne Asada Tacos","desc":"Cilantro, onions, lime","price":"$9.49","img": "gq:Carne Asada Tacos mexican"},
            {"name":"Chicken Quesadilla","desc":"Three-cheese blend, pico","price":"$8.99","img": "gq:Chicken Quesadilla mexican"},
            {"name":"Chips & Guacamole","desc":"House-made","price":"$5.99","img": "gq:Chips and Guacamole mexican"}
        ]
    },
    "chinese": {
        "palette": {"primary":"#DC2626","primary_dark":"#B91C1C"},  # Traditional red
        "hero": [],
        "menu": [
            {"name":"General Tso's Chicken","desc":"Sweet and spicy with steamed rice","price":"$12.99","img": "gq:General Tso Chicken chinese"},
            {"name":"Beef Lo Mein","desc":"Soft noodles with vegetables","price":"$11.49","img": "gq:Beef Lo Mein chinese"},
            {"name":"Pork Dumplings","desc":"Pan-fried, served with soy sauce","price":"$7.99","img": "gq:Pork Dumplings chinese"}
        ]
    },
    "thai": {
        "palette": {"primary":"#7C3AED","primary_dark":"#6D28D9"},  # Thai purple
        "hero": [],
        "menu": [
            {"name":"Pad Thai","desc":"Rice noodles, shrimp, bean sprouts","price":"$12.99","img": "gq:Pad Thai thai"},
            {"name":"Green Curry","desc":"Coconut milk, basil, jasmine rice","price":"$13.49","img": "gq:Green Curry thai"},
            {"name":"Tom Yum Soup","desc":"Spicy and sour with prawns","price":"$9.99","img": "gq:Tom Yum Soup thai"}
        ]
    },
    "japanese": {
        "palette": {"primary":"#1F2937","primary_dark":"#111827"},  # Elegant dark
        "hero": [],
        "menu": [
            {"name":"Salmon Teriyaki","desc":"Grilled with steamed vegetables","price":"$16.99","img": "gq:Salmon Teriyaki japanese"},
            {"name":"California Roll","desc":"Crab, avocado, cucumber","price":"$8.99","img": "gq:California Roll japanese"},
            {"name":"Chicken Ramen","desc":"Rich broth with soft-boiled egg","price":"$13.49","img": "gq:Chicken Ramen japanese"}
        ]
    },
    "american": {
        "palette": {"primary":"#EC1111","primary_dark":"#2563EB"},  # Classic blue (not orange!)
        "hero": [],
        "menu": [
            {"name":"Roast Chicken Plate","desc":"Choice of two sides","price":"$10.99","img": "gq:Roast Chicken Plate american"},
            {"name":"Country Fried Steak","desc":"Pepper gravy, mashed potatoes","price":"$11.49","img": "gq:Country Fried Steak american"},
            {"name":"Vegetable Plate","desc":"Pick any three sides","price":"$8.99","img": "gq:Vegetable Plate american"}
        ]
    }
}
hydrate_cuisine_assets_with_templates()

DEFAULT_CUISINE = "american"

PARTNER_BRANDS: Tuple[Tuple[str, ...], ...] = (
    ("k", "w", "cafeteria"),
    ("k", "w", "cafeterias"),
    ("piccadilly", "cafeteria"),
    ("piccadilly", "cafeterias"),
)


def _normalize_tokens(value: str) -> List[str]:
    cleaned = re.sub(r"[^a-z0-9]+", " ", (value or "").lower())
    return [tok for tok in cleaned.split() if tok]


def _is_partner_restaurant(details: Dict[str, Any]) -> bool:
    name = details.get("name") or ""
    tokens = set(_normalize_tokens(name))
    if not tokens:
        return False

    if {"k", "w"}.issubset(tokens) and ("cafeteria" in tokens or "cafeterias" in tokens or "restaurant" in tokens):
        return True
    if "piccadilly" in tokens:
        return True
    for pattern in PARTNER_BRANDS:
        if set(pattern).issubset(tokens):
            return True

    # Check website domain hints
    website = (details.get("website") or "").lower()
    if website:
        if "kwcafeterias" in website or "kwcafeteria" in website:
            return True
        if "piccadilly.com" in website or "piccadillyrestaurants" in website:
            return True

    # Fallback: brand info field if provided
    brand = (details.get("brand") or details.get("brand_name") or "").lower()
    if brand and ("k&w" in brand or "piccadilly" in brand):
        return True

    return False

def cuisine_from_types(details: Dict[str, Any]) -> str:
    name = (details.get("name") or "").lower()
    types = [t.lower() for t in (details.get("categories") or [])]
    tokens = set(_normalize_tokens(details.get("name") or ""))
    type_tokens = {tok for t in types for tok in _normalize_tokens(t)}

    cafeteria_tokens = {"cafeteria", "cafeterias", "buffet", "buffets"}
    if tokens.intersection(cafeteria_tokens) or type_tokens.intersection(cafeteria_tokens):
        return "american"
    if "piccadilly" in tokens or {"k", "w"}.issubset(tokens):
        return "american"
    
    # More comprehensive keyword matching
    cuisine_keywords = {
        "indian": {
            "types": ["indian_restaurant", "restaurant", "meal_takeaway", "food", "establishment"],
            "name_keywords": ["indian", "india", "desi", "curry", "tandoor", "masala", "biryani", 
                            "punjabi", "bengali", "south indian", "north indian", "aroma", "pakistani", "hyderabadi"],
            "priority": 1  # Higher priority for specific cuisines
        },
        "chinese": {
            "types": ["chinese_restaurant", "restaurant", "meal_takeaway"],
            "name_keywords": ["chinese", "china", "wok", "dragon", "golden", "panda", "beijing", "shanghai"],
            "priority": 1
        },
        "italian": {
            "types": ["italian_restaurant", "restaurant", "meal_takeaway"],
            "name_keywords": ["pizza", "pizzeria", "italian", "pasta", "trattoria", "ristorante", "mario", "luigi", "Caffe", "vino", "tuscan"],
            "priority": 1
        },
        "mexican": {
            "types": ["mexican_restaurant", "restaurant", "meal_takeaway"],
            "name_keywords": ["taco", "tacos", "mexican", "cantina", "casa", "el ", "la ", "burrito", "quesadilla", "sombrero", "agageve", "spanish"],
            "priority": 1
        },
        "thai": {
            "types": ["thai_restaurant", "restaurant", "meal_takeaway"],
            "name_keywords": ["thai", "pad", "bangkok", "som tam", "tom yum", "green curry"],
            "priority": 1
        },
        "japanese": {
            "types": ["japanese_restaurant", "restaurant", "meal_takeaway"],
            "name_keywords": ["sushi", "ramen", "japanese", "tokyo", "sakura", "sake", "tempura"],
            "priority": 1
        },
        "burger": {
            "types": ["restaurant", "fast_food", "meal_takeaway"],
            "name_keywords": ["burger", "burgers", "grill", "five guys", "shake shack", "in-n-out"],
            "priority": 0
        },
        "american": {
            "types": ["restaurant", "american_restaurant", "meal_takeaway"],
            "name_keywords": ["american", "diner", "cafe", "cafeteria", "cafeterias", "american grill", "bar", "steakhouse", "bbq", "barbecue", "roadhouse", "wings", "southern", "comfort food", "piccadilly", "k&w"],
            "priority": 0
        }
    }
    
    # Score each cuisine based on matches
    cuisine_scores = {}
    
    for cuisine, data in cuisine_keywords.items():
        score = 0
        
        # Check name keywords (weighted higher)
        for keyword in data["name_keywords"]:
            if keyword in name:
                score += 3 + data["priority"]  # Higher weight for name matches
        
        # Check types (weighted lower)
        for type_keyword in data["types"]:
            if any(type_keyword in restaurant_type for restaurant_type in types):
                score += 1
        
        if score > 0:
            cuisine_scores[cuisine] = score
    
    # Return cuisine with highest score, or default
    if cuisine_scores:
        best_cuisine = max(cuisine_scores.items(), key=lambda x: x[1])
        log.info(f"Cuisine detection: {name} -> {best_cuisine[0]} (score: {best_cuisine[1]})")
        return best_cuisine[0]
    
    log.info(f"Cuisine detection: {name} -> {DEFAULT_CUISINE} (no matches)")
    return DEFAULT_CUISINE
  
def get_restaurant_context(details: Dict[str, Any]) -> Dict[str, Any]:
    """Extract context clues about the restaurant's style and atmosphere"""
    name = (details.get("name") or "").lower()
    address = (details.get("address") or "").lower()
    types = [t.lower() for t in (details.get("categories") or [])]
    price_level = details.get("price_level")
    
    context = {
        "is_upscale": False,
        "is_casual": True,
        "is_fast_food": False,
        "is_chain": False,
        "atmosphere": "casual"
    }
    
    # Detect upscale indicators
    upscale_keywords = ["fine", "premium", "gourmet", "artisan", "chef", "bistro", "brasserie", 
                       "ristorante", "steakhouse", "tavern"]
    if any(keyword in name for keyword in upscale_keywords) or price_level and price_level >= 3:
        context["is_upscale"] = True
        context["is_casual"] = False
        context["atmosphere"] = "upscale"
    
    # Detect fast food
    fast_food_keywords = ["fast", "quick", "express", "drive", "burger", "pizza", "taco"]
    fast_food_types = ["fast_food", "meal_takeaway"]
    if (any(keyword in name for keyword in fast_food_keywords) or 
        any(t in types for t in fast_food_types)):
        context["is_fast_food"] = True
        context["atmosphere"] = "fast_casual"
    
    # Detect chain restaurants
    chain_indicators = details.get("chain_count_nearby", 0)
    if chain_indicators >= 2:
        context["is_chain"] = True
    
    return context

def select_theme_colors(cuisine: str, context: Dict[str, Any], logo_color: Optional[str] = None) -> Dict[str, str]:
    """Select theme colors based on cuisine/context, optionally overriding with extracted logo color."""
    # Prefer brand color from logo when available and valid
    if logo_color and is_valid_color(logo_color):
        return {
            'primary': logo_color,
            'primary_dark': darken_color(logo_color, 0.2),
        }

    base_colors = CUISINE_ASSETS.get(cuisine, CUISINE_ASSETS[DEFAULT_CUISINE])["palette"]

    if context.get("is_upscale"):
        upscale_palette = {
            "indian": {"primary": "#92400E", "primary_dark": "#78350F"},
            "italian": {"primary": "#047857", "primary_dark": "#065F46"},
            "chinese": {"primary": "#991B1B", "primary_dark": "#7F1D1D"},
            "japanese": {"primary": "#1F2937", "primary_dark": "#111827"},
            "american": {"primary": "#1E40AF", "primary_dark": "#1E3A8A"},
        }
        return upscale_palette.get(cuisine, base_colors)
    elif context.get("is_fast_food"):
        fast_food_palette = {
            "burger": {"primary": "#EF4444", "primary_dark": "#DC2626"},
            "mexican": {"primary": "#F59E0B", "primary_dark": "#D97706"},
            "american": {"primary": "#10B981", "primary_dark": "#059669"},
        }
        return fast_food_palette.get(cuisine, base_colors)

    return base_colors
  
 # ────────────────────────────────────────────────────────────────────────────
# Insights: negative-review mining + online presence scoring + (optional) AI summary
NEGATIVE_WEBAPP_KEYWORDS = [
    r"website", r"web\s*site", r"online order", r"online ordering", r"order online",
    r"mobile app", r"ios app", r"android app", r"app (?:doesn't|does not|won't|will not|can't|cannot|crashes|crashed)",
    r"menu (?:wrong|outdated|missing|not updated|broken)", r"link (?:broken|404|dead)",
    r"checkout (?:error|failed|doesn't work|does not work|won't work)",
    r"payment (?:failed|error|declined)", r"coupon(?: code)? not working",
    r"slow (?:website|app)", r"can't find (?:menu|hours)", r"hours (?:wrong|incorrect)",
]

def _filter_reviews(details: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "author": rv.get("author_name"),
            "rating": rv.get("rating"),
            "text": (rv.get("text") or "").strip(),
            "relative_time": rv.get("relative_time"),
        }
        for rv in (details.get("reviews") or [])
        if (rv.get("text") or "").strip()
    ]

def mine_webapp_complaints(reviews: List[Dict[str, Any]], *, max_items: int = 3, low_star_only: bool = True, keywords: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rv in reviews:
        txt = (rv.get("text") or "").lower()
        rating = int(rv.get("rating") or 0)
        if low_star_only and rating > 1:
            continue
        hit = False
        for pat in (keywords if keywords else NEGATIVE_WEBAPP_KEYWORDS):
            try:
                if re.search(pat, txt, flags=re.I):
                    hit = True
                    break
            except Exception:
                pass
        if hit:
            out.append(rv)
            if len(out) >= max_items:
                break
    return out

def mine_one_star(reviews: List[Dict[str, Any]], *, max_items: int = 3) -> List[Dict[str, Any]]:
    out = [rv for rv in reviews if (rv.get("rating") or 0) <= 1]
    return out[:max_items]

# Generic: filter reviews by keyword list (case-insensitive). Optional rating bounds and limit.
def filter_reviews_by_keywords(
    reviews: List[Dict[str, Any]],
    keyword_list: List[str],
    *,
    min_rating: Optional[int] = None,
    max_rating: Optional[int] = None,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    kws = [k.strip().lower() for k in (keyword_list or []) if k and k.strip()]
    if not kws:
        return out
    for rv in reviews:
        txt = (rv.get('text') or '').lower()
        rating = int(rv.get('rating') or 0)
        if min_rating is not None and rating < min_rating:
            continue
        if max_rating is not None and rating > max_rating:
            continue
        for kw in kws:
            if kw in txt:
                out.append({
                    'author': rv.get('author'),
                    'rating': rating,
                    'text': rv.get('text') or '',
                    'relative_time': rv.get('relative_time'),
                    'keyword_found': kw,
                })
                break  # avoid duplicates for multiple keyword hits
        if len(out) >= limit:
            break
    return out
def select_negative_reviews(reviews: List[Dict[str, Any]], kw_list: List[str], limit: int = 3) -> List[Dict[str, Any]]:
    """Return exactly `limit` reviews prioritizing:
       (a) keyword-matched (any rating, lowest rating first),
       (b) any 1★,
       (c) any ≤2★,
       (d) any ≤3★,
       (e) pad with lowest-rated remaining reviews.
    """
    # Normalize keywords; if none provided, use a compact default set
    if not kw_list:
        kw_list = [
            'website','web site','online order','online ordering','order online',
            'mobile app','app','hours','menu','checkout','payment','link','slow'
        ]
    kws = [k.strip().lower() for k in kw_list if k and k.strip()]

    def has_kw(rv: Dict[str, Any]) -> bool:
        txt = (rv.get('text') or '').lower()
        return any(k in txt for k in kws)

    def rating_of(rv: Dict[str, Any]) -> int:
        try:
            return int(rv.get('rating') or 0)
        except Exception:
            return 0

    chosen: List[Dict[str, Any]] = []
    seen = set()

    def add_pool(pool: List[Dict[str, Any]]):
        nonlocal chosen
        for rv in pool:
            key = (rv.get('author'), rv.get('text'))
            if key in seen:
                continue
            seen.add(key)
            chosen.append(rv)
            if len(chosen) >= limit:
                return True
        return False

    # Sort a base copy by rating ascending (lowest first) for deterministic padding
    by_lowest = sorted(reviews, key=rating_of)

    # (a) keyword-matched of any rating, lowest rating first
    kw_hits_any = [rv for rv in by_lowest if has_kw(rv)]
    if add_pool(kw_hits_any):
        return chosen[:limit]

    # (b) any 1★
    ones = [rv for rv in by_lowest if rating_of(rv) <= 1]
    if add_pool(ones):
        return chosen[:limit]

    # (c) any ≤2★
    twos = [rv for rv in by_lowest if rating_of(rv) <= 2]
    if add_pool(twos):
        return chosen[:limit]

    # (d) any ≤3★
    threes = [rv for rv in by_lowest if rating_of(rv) <= 3]
    if add_pool(threes):
        return chosen[:limit]

    # (e) pad with the lowest-rated remaining reviews regardless of rating
    add_pool(by_lowest)

    return chosen[:limit]

# Parse Google/Yelp-style relative time strings to rough day counts (e.g., "2 months ago")
_REL_UNITS = {
    'day': 1, 'days': 1,
    'week': 7, 'weeks': 7,
    'month': 30, 'months': 30,
    'year': 365, 'years': 365,
}

def _relative_time_to_days(rel: Optional[str]) -> Optional[int]:
    if not rel:
        return None
    s = rel.strip().lower()
    # common forms: "2 months ago", "3 weeks ago", "yesterday", "today"
    if s in ('today',):
        return 0
    if s in ('yesterday',):
        return 1
    m = re.search(r"(\d+)\s+(day|days|week|weeks|month|months|year|years)", s)
    if not m:
        return None
    n = int(m.group(1))
    unit = m.group(2)
    return n * _REL_UNITS.get(unit, 0) or None

def _recent_keyword_one_stars(reviews: List[Dict[str, Any]], *, keywords: List[str], within_days: int) -> int:
    cnt = 0
    pats = [k.strip().lower() for k in (keywords or []) if k and k.strip()]
    for rv in reviews:
        rating = int(rv.get('rating') or 0)
        if rating > 1:
            continue
        txt = (rv.get('text') or '').lower()
        rel = rv.get('relative_time')
        days = _relative_time_to_days(rel)
        if days is None or days > within_days:
            continue
        if any(kw in txt for kw in pats):
            cnt += 1
    return cnt

async def compute_online_presence(details: Dict[str, Any], *, keyword_list: Optional[List[str]] = None, recent_days: int = 90) -> Dict[str, Any]:
    """Conservative 10–65 score with contributing factors."""
    score = 40  # mid-scale starting point so healthy listings land ~50-55
    reasons_neg: List[str] = []
    reasons_pos: List[str] = []

    rating = float(details.get("rating") or 0)
    reviews_ct = int(details.get("review_count") or 0)
    website = details.get("website")

    try:
        reviews = _filter_reviews(details)
    except Exception:
        reviews = []

    most_recent_days: Optional[int] = None
    if reviews:
        try:
            days_list = [_relative_time_to_days(rv.get("relative_time")) for rv in reviews if _relative_time_to_days(rv.get("relative_time")) is not None]
            if days_list:
                most_recent_days = min(days_list)
        except Exception:
            most_recent_days = None

    partner_mode = _is_partner_restaurant(details)
    if partner_mode:
        log.info("presence_score: partner override active for %s", details.get("name"))
        partner_reasons = [
            "Restronaut partner location with active digital upkeep across channels.",
            "Fresh 5★ feedback keeps reputation strong—let's keep spotlighting those wins.",
            "Menus, hours, and branding stay in sync through our managed program.",
        ]
        metrics = {
            "rating": rating,
            "review_count": reviews_ct,
            "most_recent_review_days": most_recent_days,
            "has_website": bool(website),
        }
        return {
            "score": 93,
            "level": "Excellent",
            "reasons": partner_reasons,
            "metrics": metrics,
        }

    # Rating impact
    if rating >= 4.5:
        score += 8
        reasons_pos.append(f"Excellent guest rating ({rating:.1f})—spotlight top reviews on your site.")
    elif rating >= 4.0:
        score += 4
        reasons_pos.append(f"Rating sits at {rating:.1f}; nudging more 5★ reviews can lift local rank.")
    elif rating >= 3.3:
        score -= 8
        reasons_neg.append(f"Average rating ({rating:.1f}) trails competitors—prioritise review recovery.")
    elif rating > 0:
        score -= 12
        reasons_neg.append(f"Critical average rating ({rating:.1f}) is hurting conversions.")
    else:
        score -= 6
        reasons_neg.append("No public rating available—add Google reviews fast.")

    # Review volume
    if reviews_ct >= 400:
        score += 6
        reasons_pos.append(f"{reviews_ct} total reviews—repurpose them across listings and socials.")
    elif reviews_ct >= 150:
        score += 3
        reasons_pos.append(f"Healthy review volume ({reviews_ct}); keep responses active to stay fresh.")
    elif reviews_ct < 50:
        score -= 6
        reasons_neg.append("Low review volume (<50)—launch a review ask campaign.")

    third_party_reason = None
    if not website:
        score -= 9
        reasons_neg.append("No official website listed—claim a first-party hub for menus and orders.")
    else:
        lowered = website.lower()
        suspicious = [
            "doordash.com","ubereats.com","grubhub.com","postmates.com","ezcater.com",
            "toasttab.com","opentable.com","resy.com","seamless.com","chownow.com",
            "clover.com","facebook.com","instagram.com","linktr.ee","bit.ly","goo.gl","google.com/maps"
        ]
        if any(token in lowered for token in suspicious):
            score -= 5
            third_party_reason = "Website field points to third-party profile" if "facebook" in lowered or "instagram" in lowered else "Only marketplace/ordering link provided"
        else:
            score += 5
            reasons_pos.append("Official site detected—keep menus, hours, and ordering prominent.")
    if third_party_reason:
        reasons_neg.append(third_party_reason)

    gallery_ct = len(details.get("gallery") or [])
    if gallery_ct >= 8:
        score += 4
        reasons_pos.append("Strong photo gallery—keep seasonal shoots coming.")
    elif gallery_ct >= 4:
        score += 2
        reasons_pos.append("Gallery has good coverage; refresh with new hero shots quarterly.")
    else:
        score -= 4
        reasons_neg.append("Few photos in Google gallery—upload professional interior and dish shots.")

    if not details.get("hours_text"):
        score -= 3
        reasons_neg.append("Missing operating hours—publish accurate hours everywhere.")
    else:
        reasons_pos.append("Hours are published—double-check holiday overrides.")

    # Chains nearby -> harder to stand out
    chain_cnt = details.get("chain_count_nearby") or 0
    if chain_cnt >= 3:
        score -= 6
        reasons_neg.append("Strong nearby chain competition—differentiate your experience online.")
    elif chain_cnt >= 1:
        score -= 3
        reasons_neg.append("Some nearby chain presence—promote unique menu hooks.")

    # Recent 1★ keyword complaints (e.g., website/app) within the last N days
    try:
        kws = keyword_list if (keyword_list and len(keyword_list) > 0) else [
            'website','online order','online ordering','order online','mobile app','app','hours','menu','checkout','payment'
        ]
        recent_hits = _recent_keyword_one_stars(reviews, keywords=kws, within_days=recent_days)
        if recent_hits:
            score -= 8
            reasons_neg.append(f"{recent_hits} recent 1★ mention(s) of website/app issues—patch the journey fast.")
    except Exception:
        pass

    # Brand signal (logo color)
    try:
        _lu, _lc, _rs = await best_logo_with_color(details)
        if not _lc:
            score -= 3
            reasons_neg.append("No detectable brand color/logo theme—align visuals across profiles.")
        else:
            score += 2
            reasons_pos.append("Brand colors detected—reuse them in listings and landing pages.")
    except Exception:
        pass

    reasons: List[str] = []
    seen: set = set()
    for bucket in (reasons_neg, reasons_pos):
        for reason in bucket:
            if reason and reason not in seen:
                reasons.append(reason)
                seen.add(reason)
            if len(reasons) >= 5:
                break
        if len(reasons) >= 5:
            break

    if len(reasons) < 3:
        supplemental: List[str] = []
        if rating:
            supplemental.append(f"Encourage happy guests to share fresh reviews to lift the {rating:.1f} score.")
        if website:
            supplemental.append("Audit mobile speed so diners reach your menu in under 3 seconds.")
        if not gallery_ct or gallery_ct < 6:
            supplemental.append("Add mouthwatering photography to outperform nearby listings.")
        supplemental.append("Strengthen local SEO so searchers pick you over national chains.")
        for idea in supplemental:
            if idea not in seen:
                reasons.append(idea)
                seen.add(idea)
            if len(reasons) >= 3:
                break

    score = max(10, min(65, round(score)))
    level = "Excellent" if score >= 58 else "Good" if score >= 48 else "Needs Work" if score >= 34 else "At Risk"

    metrics = {
        "rating": rating,
        "review_count": reviews_ct,
        "most_recent_review_days": most_recent_days,
        "has_website": bool(website),
    }
    return {"score": score, "level": level, "reasons": reasons, "metrics": metrics}

async def ai_presence_oneliner(details: Dict[str, Any], prescore: Dict[str, Any], complaints: List[Dict[str, Any]]) -> str:
    name = details.get('name') or 'This restaurant'
    rating = details.get('rating')
    reasons = prescore.get('reasons') or []

    # Build a compact, targeted prompt
    issues = ", ".join(reasons[:3]) or "gaps in online presence"
    prompt = (
        f"Write ONE tactful, persuasive sentence (max 28 words) explaining why {name} should improve its online presence now. "
        f"Ground it in these signals: rating {rating}, reasons: {issues}. "
        "Avoid hyphens and buzzwords; be specific and positive about the outcome."
    )

    def _sanitize(line: str) -> str:
        if not line:
            return ""
        s = (line or "").replace("\n", " ").replace("—", "-").strip()
        # Remove stray quotes and enforce no hyphens per instruction
        s = s.strip('"\'')
        s = s.replace('-', ' ')
        # Soft word cap at 28 words
        parts = s.split()
        if len(parts) > 28:
            s = " ".join(parts[:28])
        return s

    def _fallback_line() -> str:
        base = _presence_oneline_default(prescore)
        return _sanitize(base)

    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key and OpenAI is not None:
        async def _call_openai() -> str:
            def _run() -> str:
                try:
                    client = OpenAI(api_key=api_key, timeout=Timeout(12.0))
                except Exception as init_err:
                    log.debug(f"ai_oneline: OpenAI init failed: {init_err}")
                    return ""

                # Try Responses API first
                try:
                    resp = client.responses.create(
                        model="gpt-4o-mini",
                        input=[{"role": "user", "content": prompt}],
                        max_output_tokens=60,
                        temperature=0.4,
                    )
                    line = getattr(resp, 'output_text', None) or ""
                    line = _sanitize(line)
                    if line:
                        log.debug("ai_oneline: used Responses API")
                        return line
                except Exception as e:
                    log.debug(f"ai_oneline: Responses API failed: {e}")

                # Fallback: Chat Completions API (broader compatibility)
                try:
                    chat = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=60,
                        temperature=0.4,
                    )
                    content = (chat.choices[0].message.content or "") if chat and chat.choices else ""
                    line = _sanitize(content)
                    if line:
                        log.debug("ai_oneline: used Chat Completions API")
                        return line
                except Exception as e:
                    log.debug(f"ai_oneline: Chat Completions failed: {e}")
                return ""

            return await asyncio.to_thread(_run)

        try:
            line = await asyncio.wait_for(_call_openai(), timeout=14.0)
            if line:
                return line
        except asyncio.TimeoutError:
            log.warning("ai_oneline: OpenAI timed out after 14s; using fallback")
        except Exception as e:
            log.debug(f"ai_oneline: OpenAI call failed: {e}")

    log.info("ai_oneline: using deterministic fallback copy")
    return _fallback_line()


def auto_close_json(content: str) -> str:
    repaired = content.strip()
    open_curly = repaired.count("{")
    close_curly = repaired.count("}")
    open_square = repaired.count("[")
    close_square = repaired.count("]")
    if open_square > close_square:
        repaired += "]" * (open_square - close_square)
    if open_curly > close_curly:
        repaired += "}" * (open_curly - close_curly)
    repaired = re.sub(r",\s*}", "}", repaired)
    repaired = re.sub(r",\s*\]", "]", repaired)
    return repaired

async def ai_presence_insights(details: Dict[str, Any],
                                prescore: Dict[str, Any],
                                nearby: List[Dict[str, Any]],
                                complaints: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return OpenAI-generated 5+ structured insights when possible.

    Returns ``None`` if OpenAI times out so the caller can fall back gracefully.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        log.error("ai_presence_insights: OpenAI API key not set or OpenAI SDK missing")
        raise HTTPException(502, "OpenAI API unavailable for insights")
    payload = {
        "business_id": details.get("id"),
        "business_name": details.get("name"),
        "presence": prescore,
        "nearby": nearby[:10],
        "reviews": complaints,
    }
    system = (
        "You are an analyst. Return ONLY valid JSON with at least 5 distinct problems, "
        "strictly following this schema: "
        "{business_id (string), business_name (string), summary (string), "
        "problems: [{title (string), evidence (list of strings), "
        "fix_steps (list of strings)}], "
        "\n"
        "SCHEMA RULES: "
        "- fix_steps must be a list of one-line strategies like: 'We will improve this for you ...'. "
        "- No room for mistake, the information must be accurate."
        "- If URL is missing/null, return null. "
        "- At least 5 problems must be included."
    )
    user = (
        "Generate insights for this business. "
        "Focus ONLY on online presence and IT-related issues (website, app, SEO, listings, branding). "
        "Each fix must be ONE line only, describing our strategy professionally (e.g., 'We will improve this for you by .. or No worries, we got you'). "
        "Problems >= 5. "
        "Here is payload:\n" + json.dumps(payload)
    )

    timeout_client = Timeout(20.0)

    def _run_openai_call() -> Any:
        try:
            client = OpenAI(api_key=api_key, timeout=timeout_client)
        except Exception as init_err:
            log.error(f"ai_presence_insights: OpenAI init failed: {init_err}")
            raise

        chat = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=900,
        )
        message = chat.choices[0].message if chat and chat.choices else None
        parsed = getattr(message, "parsed", None) if message is not None else None
        if parsed is not None:
            return parsed
        return (message.content or "") if message is not None else ""

    try:
        raw = await asyncio.wait_for(asyncio.to_thread(_run_openai_call), timeout=22.0)
    except asyncio.TimeoutError:
        log.error("ai_presence_insights: OpenAI timed out after 22s")
        return None
    except Exception as e_outer:
        log.error(f"ai_presence_insights unavailable: {e_outer}")
        raise HTTPException(502, f"OpenAI insights failed: {e_outer}")

    if isinstance(raw, dict):
        return raw

    content = (raw or "").strip()

    # Strip code fences if any
    if content.startswith("```"):
        content = content.strip('` \n')

    # Extract JSON between first '{' and last '}' if needed
    if not content.startswith("{"):
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            content = content[start:end+1]

    # Attempt strict parse
    try:
        return json.loads(content)
    except Exception as e1:
        log.warning(f"ai_presence_insights: strict JSON parse failed: {e1}")

    # Attempt json5 fallback
    if json5 is not None:
        try:
            return json5.loads(content)
        except Exception as e2:
            log.error(f"ai_presence_insights: json5 fallback failed: {e2}")

    # Attempt fix-busted-json fallback
    if fix_busted_repair is not None:
        try:
            repaired = fix_busted_repair(content)
            # possibly re-extract JSON block
            if not repaired.startswith("{"):
                s = repaired.find("{")
                e = repaired.rfind("}")
                if s != -1 and e != -1 and e > s:
                    repaired = repaired[s:e+1]
            return json.loads(repaired)
        except Exception as e3:
            log.error(f"ai_presence_insights: fix-busted-json fallback failed: {e3}")

    # Attempt json_repair fallback
    if json_repair_repair is not None:
        try:
            repaired = json_repair_repair(content)
            if not repaired.startswith("{"):
                s = repaired.find("{")
                e = repaired.rfind("}")
                if s != -1 and e != -1 and e > s:
                    repaired = repaired[s:e+1]
            return json.loads(repaired)
        except Exception as e4:
            log.error(f"ai_presence_insights: json_repair fallback failed: {e4}")

    # Final simple heuristics repair
    repaired2 = content
    # If odd number of double quotes, append a closing quote
    if repaired2.count('"') % 2 == 1:
        repaired2 = repaired2 + '"'
    # Remove trailing commas before } or ]
    repaired2 = re.sub(r",\s*}", "}", repaired2)
    repaired2 = re.sub(r",\s*\]", "]", repaired2)

    try:
        return json.loads(repaired2)
    except Exception as e5:
        log.error(f"ai_presence_insights: final heuristic repair failed: {e5} | repaired2_content={repaired2!r}")

    # Final fallback: auto_close_json
    try:
        repaired3 = auto_close_json(content)
        result = json.loads(repaired3)
        log.warning("ai_presence_insights: auto_close repair applied")
        return result
    except Exception as e6:
        log.error(f"ai_presence_insights: auto_close_json fallback failed: {e6} | repaired3_content={repaired3!r}")

        # If all parsing fails, raise error
        raise HTTPException(502, f"OpenAI returned invalid JSON for insights: last error {e1}")

def _local_insights_fallback(details: Dict[str, Any], prescore: Dict[str, Any], nearby: List[Dict[str, Any]], complaints: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Deterministic, on-box insights so the response is never empty."""
    log.info("insights_fallback: generating deterministic insights payload")
    partner_mode = _is_partner_restaurant(details)
    name = details.get("name") or "This restaurant"
    bid = details.get("id") or "unknown"
    rating = details.get("rating")
    rev_ct = details.get("review_count")
    gallery_ct = len(details.get("gallery") or [])
    most_recent_days = prescore.get("metrics", {}).get("most_recent_review_days")
    website = details.get("website") or ""
    reasons = prescore.get("reasons") or []

    problems: List[Dict[str, Any]] = []
    def add(title, sev, impact, ev, steps, owner="marketing", eta=2, quick=True):
        problems.append({
            "title": title,
            "severity": sev,
            "expected_impact": impact,
            "evidence": ev,
            "fix_steps": steps,
            "quick_win": quick,
            "owner": owner,
            "eta_hours": eta,
        })

    # 1) Website quality/clarity unknown
    add(
        "Website performance/clarity unknown",
        "medium",
        "conversion",
        [f"metrics.has_website={bool(website)}", "reason present: Website performance/clarity unknown—optimize for mobile speed and menu visibility"],
        [
            "Run PageSpeed Insights and Lighthouse on homepage and menu pages",
            "Add prominent CTA: Order/Call/Reserve in header and hero",
            "Ensure menu section renders fast (<2s LCP) and is readable on mobile",
            "Compress hero images and lazy-load below the fold",
        ],
    )

    # 2) Few/broken photos
    if gallery_ct < 3:
        add(
            "Too few photos in Google gallery",
            "medium", "trust",
            [f"gallery count={gallery_ct}"],
            [
                "Upload 8–12 high quality interior/food photos to Google Business Profile",
                "Set website hero to use Google photos with onerror SVG fallback",
                "Add Pexels fallback in builder when Google is blocked by referrer",
            ],
        )

    # 3) Review freshness
    if (most_recent_days is None) or (isinstance(most_recent_days, int) and most_recent_days > 30):
        add(
            "Low recent review activity",
            "medium", "trust",
            [f"most_recent_review_days={most_recent_days}"],
            [
                "Run a 2‑week in‑store ask: place QR to Google review link",
                "Reply to last 5 reviews to signal active management",
            ],
        )

    # 4) Rating/volume risk
    if (rating is not None and float(rating) < 4.3) or (rev_ct is not None and int(rev_ct) < 200):
        add(
            "Ratings/review volume can be improved",
            "medium", "trust",
            [f"rating={rating}", f"review_count={rev_ct}"],
            [
                "Respond to recent 1★ reviews with remediation (refund/replacement where appropriate)",
                "Feature 5★ quotes on website homepage and social",
            ],
        )

    # 5) Chain competition
    if any('chain' in (r or '').lower() for r in reasons):
        add(
            "Nearby chain competition is high",
            "high", "traffic",
            ["reason present: Nearby chain competition is high"],
            [
                "Differentiate hero with signature dishes and unique value (pricing/portion/story)",
                "Add schema.org/LocalBusiness + Menu to improve SERP visibility",
                "Run local ads targeting competitor brand + cuisine keywords",
            ],
        )

    # 6) Negative review signal (if any complaints provided)
    if complaints and not partner_mode:
        c1 = complaints[0]
        add(
            "Address recent 1★ complaints",
            "high", "trust",
            [f"sample: '{(c1.get('text') or '')[:160]}…'"],
            [
                "Investigate root cause with store manager and log corrective action",
                "Reach out to reviewer with apology and remedy (refund/replacement)",
                "Publish updated hygiene/process note on website if relevant",
            ], owner="ops", eta=4, quick=False
        )

    # Ensure at least 5 items
    while len(problems) < 5:
        add(
            "Brand consistency can be improved online",
            "low", "conversion",
            ["reason present: Brand consistency can be improved online"],
            [
                "Extract brand color/logo and apply across site buttons/links",
                "Add favicon and social share images",
            ],
        )

    # Competitors (top 3 nearby)
    comps = []
    for c in (nearby or [])[:3]:
        comps.append({
            "name": c.get("name"),
            "rating": c.get("rating"),
            "map_url": c.get("map_url"),
            "note": "Review their photo style and top dishes for inspiration",
        })

    return {
        "business_id": bid,
        "business_name": name,
        "summary": f"Presence score {prescore.get('score')} ({prescore.get('level')}). Website={'present' if website else 'missing'}.",
        "problems": problems,
        "mobile_app": {"status": "not_applicable", "ios_url": None, "android_url": None, "proof_text": "mobile app check disabled", "connect_steps": []},
        "competitors": comps,
    }

# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────
# Google: autocomplete, details, chain count
# ────────────────────────────────────────────────────────────────────────────
def normalize_suggest_google(preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for p in preds or []:
        fmt = p.get("structured_formatting") or {}
        out.append({
            "id": p.get("place_id"),
            "title": fmt.get("main_text") or (p.get("description") or ""),
            "subtitle": fmt.get("secondary_text") or "",
            "provider": "google",
        })
    return out

def normalize_details_google(r: Dict[str, Any]) -> Dict[str, Any]:
    g = r.get("result") or {}
    loc = (g.get("geometry") or {}).get("location") or {}

    site = _homepage(g.get("website")) if g.get("website") else None
    if site and _is_marketplace(site):
        site = None

    photos = g.get("photos") or []
    gallery = []
    cover_photo_url = None
    for i, ph in enumerate(photos[:8]):
        ref = ph.get("photo_reference")
        if ref:
            url = build_google_photo_url(ref, maxwidth=1600)
            if i == 0:
                cover_photo_url = url
            gallery.append(url)

    reviews_raw = []
    for rv in (g.get("reviews") or [])[:10]:
        reviews_raw.append({
            "author_name": rv.get("author_name"),
            "rating": rv.get("rating"),
            "text": rv.get("text"),
            "profile_photo_url": rv.get("profile_photo_url"),
            "relative_time": rv.get("relative_time_description")
        })

    logo_cands: List[Tuple[str, int, str]] = []
    def add(url: Optional[str], score: int, reason: str):
        if not url: return
        logo_cands.append((url, score, reason))

    if g.get("icon_mask_base_uri"):
        add(g["icon_mask_base_uri"] + ".png", 95, "icon_mask_base_uri")

    if g.get("icon") and not GENERIC_ICON_PAT.search(g["icon"]):
        add(g["icon"], 80, "icon_non_generic")

    for u in guess_favicon_urls(site):
        score = 70
        if "sz=256" in u: score = 88
        elif "sz=128" in u: score = 78
        add(u, score, "website_favicon")

    if not site and g.get("url"):
        try:
            host = urlparse(g["url"]).netloc
            if host:
                add(f"https://www.google.com/s2/favicons?sz=256&domain={host}", 72, "maps_url_favicon")
        except Exception:
            pass

    logo_cands.sort(key=lambda t: t[1], reverse=True)
    logo_urls = [u for (u,_,_) in logo_cands]

    opening = g.get("opening_hours") or {}
    open_now = opening.get("open_now")

    out = {
        "id": g.get("place_id"),
        "name": g.get("name"),
        "address": g.get("formatted_address"),
        "phone": g.get("international_phone_number") or g.get("formatted_phone_number"),
        "website": site,
        "map_url": g.get("url"),
        "lat": loc.get("lat"),
        "lng": loc.get("lng"),
        "rating": g.get("rating"),
        "review_count": g.get("user_ratings_total"),
        "categories": g.get("types") or [],
        "hours_text": opening.get("weekday_text") or [],
        "open_now": open_now,
        "price_level": g.get("price_level"),
        "business_status": g.get("business_status"),
        "icon_background_color": g.get("icon_background_color"),
        "cover_photo_url": cover_photo_url,
        "gallery": gallery,
        "reviews": reviews_raw,
        "logo_url_candidates": logo_urls,
        "logo_debug": logo_cands,
        "provider": "google",
        "raw": g,
    }
    return out

async def google_autocomplete(q: str, *, lat: Optional[float], lng: Optional[float], radius_m: int, country: str) -> Dict[str, Any]:
    if not GOOGLE_API_KEY:
        raise HTTPException(500, "Missing GOOGLE_PLACES_API_KEY")
    params = {
        "input": q,
        "key": GOOGLE_API_KEY,
        "types": "establishment",
        "components": f"country:{country}",
    }
    if lat is not None and lng is not None:
        params["location"] = f"{lat},{lng}"
        params["radius"] = str(radius_m)
    data = await http_get_json("https://maps.googleapis.com/maps/api/place/autocomplete/json", params=params)
    status = data.get("status")
    if status not in ("OK", "ZERO_RESULTS"):
        raise HTTPException(502, f"Google error: {status} - {data.get('error_message')}")
    return data

async def google_details(place_id: str) -> Dict[str, Any]:
    if not GOOGLE_API_KEY:
        raise HTTPException(500, "Missing GOOGLE_PLACES_API_KEY")
    fields = ",".join([
        "place_id","name","formatted_address","geometry/location",
        "international_phone_number","formatted_phone_number",
        "website","url","opening_hours/open_now","opening_hours/weekday_text",
        "rating","user_ratings_total","types","price_level","business_status",
        "photos","reviews","icon","icon_mask_base_uri","icon_background_color"
    ])
    params = {"place_id": place_id, "fields": fields, "key": GOOGLE_API_KEY}
    data = await http_get_json("https://maps.googleapis.com/maps/api/place/details/json", params=params)
    status = data.get("status")
    if status != "OK":
        raise HTTPException(502, f"Google error: {status} - {data.get('error_message')}")
    return data


async def google_details_normalized(place_id: str, *, include_chain_info: bool = False) -> Dict[str, Any]:
    """Fetch and normalize Google details with caching."""
    if not place_id:
        raise HTTPException(400, "Missing place_id")
    ck = cache_key("details_norm", place_id, int(include_chain_info))
    cached = DETAILS_NORM_CACHE.get(ck)
    if cached is not None:
        return copy.deepcopy(cached)

    raw = await google_details(place_id)
    norm = normalize_details_google(raw)

    if include_chain_info:
        try:
            chain_count = await google_nearby_chain_count(
                norm.get("name") or "",
                norm.get("lat"),
                norm.get("lng"),
                norm.get("id") or place_id,
                CHAIN_RADIUS_M,
            )
            norm["chain_count_nearby"] = chain_count
            norm["is_chain_nearby"] = bool(chain_count and chain_count >= 1)
        except Exception:
            norm.setdefault("chain_count_nearby", 0)
            norm.setdefault("is_chain_nearby", False)

    DETAILS_NORM_CACHE.set(ck, norm)
    return copy.deepcopy(norm)

async def google_nearby_chain_count(name: str, lat: Optional[float], lng: Optional[float], self_place_id: str, radius_m: int) -> int:
    if not GOOGLE_API_KEY or lat is None or lng is None or not name:
        return 0
    ck = cache_key("chain_count", name.lower(), round(lat, 4), round(lng, 4), radius_m)
    cached = CHAIN_COUNT_CACHE.get(ck)
    if cached is not None:
        return cached
    params = {
        "keyword": name,
        "location": f"{lat},{lng}",
        "radius": str(radius_m),
        "type": "restaurant",
        "key": GOOGLE_API_KEY,
    }
    data = await http_get_json("https://maps.googleapis.com/maps/api/place/nearbysearch/json", params=params)
    results = data.get("results") or []
    target = name.strip().lower()
    count = 0
    for r in results:
        pid = r.get("place_id")
        rname = (r.get("name") or "").strip().lower()
        if pid == self_place_id: continue
        if rname == target:
            count += 1
    CHAIN_COUNT_CACHE.set(ck, count)
    return count

async def google_nearby_restaurants(lat: Optional[float], lng: Optional[float], *, radius_m: int = 16093, limit: int = 60) -> List[Dict[str, Any]]:
    """List nearby restaurants within radius_m (~10 miles) with basic info.
    Paginates through Google Nearby Search to return up to `limit` results.
    """
    if not GOOGLE_API_KEY or lat is None or lng is None:
        return []
    ck = cache_key("nearby_restaurants", round(lat, 4), round(lng, 4), min(radius_m, 50000), limit)
    cached = NEARBY_CACHE.get(ck)
    if cached is not None:
        return copy.deepcopy(cached)
    out: List[Dict[str, Any]] = []
    page_token: Optional[str] = None
    while True:
        params = {
            "location": f"{lat},{lng}",
            "radius": str(max(1000, min(radius_m, 50000))),
            "type": "restaurant",
            "key": GOOGLE_API_KEY,
        }
        if page_token:
            params["pagetoken"] = page_token
        data = await http_get_json("https://maps.googleapis.com/maps/api/place/nearbysearch/json", params=params)
        results = data.get("results") or []
        for r in results:
            pid = r.get("place_id")
            name = r.get("name")
            rating = r.get("rating")
            map_url = f"https://www.google.com/maps/place/?q=place_id:{pid}" if pid else None
            out.append({"id": pid, "name": name, "rating": rating, "map_url": map_url})
            if len(out) >= limit:
                return out
        page_token = data.get("next_page_token")
        if not page_token:
            break
        # Google requires a short delay before using next_page_token
        await asyncio.sleep(2.1)
    NEARBY_CACHE.set(ck, out)
    return copy.deepcopy(out)

def _presence_oneline_default(prescore: Dict[str, Any]) -> str:
    reasons = prescore.get('reasons') or []
    if reasons:
        return f"Your online presence is slipping, {reasons[0].rstrip('.')}; let us fix it with a fast, modern site that converts more customers."
    return "A stronger website and clearer information can win you more customers; let us help you upgrade quickly."

async def google_textsearch_cuisine(
    *, lat: float, lng: float, cuisine: str, radius_m: int = 16093, limit: int = 50
) -> List[Dict[str, Any]]:
    """Use Google Places Text Search to find restaurants matching a cuisine query.

    This is more reliable than inferring cuisine from generic `types` on Nearby results.
    """
    if not GOOGLE_API_KEY or lat is None or lng is None or not cuisine:
        return []

    def cuisine_query(c: str) -> str:
        mapping = {
            "indian": "indian restaurant",
            "burger": "burger restaurant",
            "italian": "italian restaurant",
            "mexican": "mexican restaurant",
            "chinese": "chinese restaurant",
            "thai": "thai restaurant",
            "japanese": "japanese restaurant",
            "american": "american restaurant",
        }
        return mapping.get(c.lower(), f"{c} restaurant")

    query = cuisine_query(cuisine)
    ck = cache_key("textsearch_cuisine", query, round(lat, 4), round(lng, 4), min(radius_m, 50000), limit)
    cached = TEXTSEARCH_CACHE.get(ck)
    if cached is not None:
        return copy.deepcopy(cached)
    out: List[Dict[str, Any]] = []
    page_token: Optional[str] = None
    while True:
        params = {
            "query": query,
            "location": f"{lat},{lng}",
            "radius": str(max(500, min(radius_m, 50000))),
            "key": GOOGLE_API_KEY,
        }
        if page_token:
            params["pagetoken"] = page_token
        data = await http_get_json("https://maps.googleapis.com/maps/api/place/textsearch/json", params=params)
        results = data.get("results") or []
        for r in results:
            pid = r.get("place_id")
            name = r.get("name")
            rating = r.get("rating")
            types = [t.lower() for t in (r.get("types") or [])]
            # Filter out obvious non-restaurant matches (e.g., hotels/lodging)
            allowed_type_tokens = {
                "restaurant",
                "meal_takeaway",
                "meal_delivery",
                "food",
                "bakery",
                "cafe",
                "bar",
                "night_club",
            }
            if types and not any(tok in allowed_type_tokens for tok in types):
                continue
            map_url = f"https://www.google.com/maps/place/?q=place_id:{pid}" if pid else None
            out.append({
                "id": pid,
                "name": name,
                "rating": rating,
                "map_url": map_url,
                "types": types,
            })
            if len(out) >= limit:
                return out
        page_token = data.get("next_page_token")
        if not page_token:
            break
        await asyncio.sleep(2.1)
    TEXTSEARCH_CACHE.set(ck, out)
    return copy.deepcopy(out)

async def nearby_same_cuisine(
    details: Dict[str, Any], *, radius_m: int = 16093, min_results: int = 7, scan_limit: int = 50, concurrency: int = 6, max_results: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Return at least `min_results` nearby competitors with the same cuisine.

    First uses Places Text Search with a cuisine-specific query to find candidates,
    then filters/excludes the original business and de-duplicates. Falls back to a
    light details-check only if necessary.
    """
    lat, lng = details.get("lat"), details.get("lng")
    base_cuisine = cuisine_from_types(details)
    self_id = details.get("id")
    if lat is None or lng is None:
        return []

    # Prefer text search for cuisine-specific results
    text_hits = await google_textsearch_cuisine(lat=lat, lng=lng, cuisine=base_cuisine, radius_m=radius_m, limit=scan_limit)
    # Exclude self + dedupe by place_id
    dedup: Dict[str, Dict[str, Any]] = {}
    for it in text_hits:
        pid = it.get("id")
        if not pid or pid == self_id:
            continue
        types = {t for t in (it.get("types") or [])}
        if types and not any(tok in {"restaurant", "meal_takeaway", "meal_delivery", "food", "cafe", "bakery"} for tok in types):
            continue
        dedup[pid] = {**it, "cuisine": base_cuisine}

    out = list(dedup.values())
    target_len = max(min_results, max_results or 0)

    # Fallback: widen radius slightly, or inspect nearby and verify via details when we do not have enough yet
    if len(out) < target_len:
        more = await google_nearby_restaurants(lat, lng, radius_m=min(25000, radius_m + 5000), limit=scan_limit)

        if more:
            sem = asyncio.Semaphore(concurrency)

            async def fetch_and_match(item: Dict[str, Any]):
                pid = item.get("id")
                if not pid or pid == self_id or pid in dedup:
                    return None
                async with sem:
                    try:
                        norm = await google_details_normalized(pid, include_chain_info=False)
                        if cuisine_from_types(norm) != base_cuisine:
                            return None
                        return {
                            "id": norm.get("id"),
                            "name": norm.get("name"),
                            "rating": norm.get("rating"),
                            "map_url": f"https://www.google.com/maps/place/?q=place_id:{norm.get('id')}" if norm.get("id") else None,
                            "cuisine": base_cuisine,
                        }
                    except Exception:
                        return None

            tasks = [asyncio.create_task(fetch_and_match(i)) for i in more]
            results = await asyncio.gather(*tasks)
            for res in results:
                if not res:
                    continue
                pid = res.get("id")
                if not pid or pid in dedup:
                    continue
                dedup[pid] = res
            out = list(dedup.values())

    if max_results is not None:
        return out[:max_results]
    return out
  

# Yelp fallback
def normalize_suggest_yelp(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for b in payload.get("businesses", []):
        loc = b.get("location") or {}
        addr = ", ".join(filter(None, [loc.get("address1"), loc.get("city"), loc.get("state")]))
        out.append({
            "id": b.get("id"), "title": b.get("name"), "subtitle": addr, "provider": "yelp",
        })
    return out

def normalize_details_yelp(b: Dict[str, Any]) -> Dict[str, Any]:
    loc = b.get("location") or {}
    coords = b.get("coordinates") or {}
    cover = b.get("image_url") or None
    addr = ", ".join(filter(None, [loc.get("address1"), loc.get("city"), loc.get("state"), loc.get("zip_code")]))

    site = None
    attrs = b.get("attributes") or {}
    menu_url = attrs.get("menu_url")
    if menu_url:
        hp = _homepage(menu_url)
        if hp and not _is_marketplace(hp):
            site = hp

    homepage = site
    logo_candidates = guess_favicon_urls(homepage) if homepage else []

    return {
        "id": b.get("id"),
        "name": b.get("name"),
        "address": addr,
        "phone": b.get("display_phone") or b.get("phone"),
        "website": homepage,
        "map_url": b.get("url"),
        "lat": coords.get("latitude"),
        "lng": coords.get("longitude"),
        "rating": b.get("rating"),
        "review_count": b.get("review_count"),
        "categories": [c.get("title") for c in b.get("categories", [])],
        "hours_text": (b.get("hours", [{}])[0] or {}).get("open"),
        "open_now": (b.get("hours", [{}])[0] or {}).get("is_open_now"),
        "price_level": b.get("price"),
        "business_status": None,
        "cover_photo_url": cover,
        "gallery": [cover] if cover else [],
        "reviews": [],
        "logo_url_candidates": logo_candidates,
        "provider": "yelp",
        "raw": b,
    }

async def yelp_autocomplete(q: str, *, lat: Optional[float], lng: Optional[float], limit: int) -> Dict[str, Any]:
    if not YELP_API_KEY:
        return {"businesses": []}
    headers = {"Authorization": f"Bearer {YELP_API_KEY}"}
    params = {"text": q, "limit": min(limit, 20)}
    if lat is not None: params["latitude"]  = lat
    if lng is not None: params["longitude"] = lng
    return await http_get_json("https://api.yelp.com/v3/autocomplete", params=params, headers=headers)

async def yelp_business_details(business_id: str) -> Dict[str, Any]:
    if not YELP_API_KEY:
        raise HTTPException(404, "Yelp fallback disabled or missing YELP_API_KEY")
    headers = {"Authorization": f"Bearer {YELP_API_KEY}"}
    return await http_get_json(f"https://api.yelp.com/v3/businesses/{business_id}", headers=headers)


async def _fetch_details(provider: str, place_id: Optional[str], details_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Fetch normalized details using provider/place_id or a provided payload.
    This does NOT affect other endpoints; it is only used by /generate/mobile.
    """
    if details_payload:
        return details_payload

    if provider == "google" and place_id:
        return await google_details_normalized(place_id, include_chain_info=True)

    if provider == "yelp" and place_id:
        raw = await yelp_business_details(place_id)
        return normalize_details_yelp(raw)

    raise HTTPException(400, "Missing details and place_id")
# ────────────────────────────────────────────────────────────────────────────
# Models
# ────────────────────────────────────────────────────────────────────────────
class GeneratePayload(BaseModel):
    place_id: Optional[str] = Field(None, description="Google place_id or Yelp id")
    provider: Literal["google","yelp"] = "google"
    details: Optional[Dict[str, Any]] = Field(None, description="If provided, use this instead of fetching")
    theme_hint: Optional[str] = Field(None, description="unused for now")
    sales_cta: bool = Field(default=True, description="Show bottom CTA ribbon")

class TemplateOut(BaseModel):
    html: Optional[str] = None
    react: Optional[str] = None
    meta: Dict[str, Any]


# ────────────────────────────────────────────────────────────────────────────
# Mobile App Builder Endpoint (always uses v3)
# ────────────────────────────────────────────────────────────────────────────
@app.post("/generate/mobile", response_model=TemplateOut)
async def generate_mobile(
    payload: GeneratePayload,
    request: Request,
    _rl = Depends(rate_limit),
    _sec = Depends(security_guard)
):
    # Fetch details (using existing helper; keeps behavior consistent)
    details = await _fetch_details(payload.provider, payload.place_id, payload.details)

    # Always use the new mobile app builder and make caching/cdn debugging easy
    log.info("generate/mobile: using build_mobile_app_html (v3) for %s", details.get("name"))
    html, meta = await build_mobile_app_html(details)

    # Add a visible build stamp in the HTML to confirm version and cache state
    stamp = f"<!-- MOBILE_BUILDER=v3 no-hero-circle bg=hardcoded-svg ts={int(time.time())} -->"
    if "</head>" in html:
        html = html.replace("</head>", stamp + "\n</head>")
    else:
        html = stamp + html

    return {"html": html, "meta": meta}


def best_logo(details: Dict[str, Any]) -> Tuple[Optional[str], str]:
    # Build a scored candidate list from debug and simple candidates
    cands_dbg = details.get("logo_debug") or []  # (url, score, reason)
    scored: List[Tuple[int, str, str]] = []
    for (u, _s, r) in cands_dbg:
        if GENERIC_ICON_PAT.search(u or ''):
            continue
        scored.append((_score_logo_candidate(u, r, details), u, r))
    for u in (details.get("logo_url_candidates") or []):
        if GENERIC_ICON_PAT.search(u or ''):
            continue
        scored.append((_score_logo_candidate(u, 'fallback_first', details), u, 'fallback_first'))

    scored.sort(key=lambda t: t[0], reverse=True)

    # Quick sanity reject tiny favicons
    for _score, url, reason in scored:
        if 'sz=' in url:
            try:
                sz = int(url.split('sz=')[1].split('&')[0])
                if sz < 64:
                    continue
            except Exception:
                pass
        # Return the best candidate; async validation happens in best_logo_with_color()
        return url, reason

    # Last resort: synthesize s2 favicon from website domain
    hp = _homepage(details.get('website'))
    if hp:
        host = urlparse(hp).netloc
        fallback = f"https://www.google.com/s2/favicons?sz=512&domain={host}"
        return fallback, 'generated_favicon'

    return None, 'none'

def hours_list(details: Dict[str, Any]) -> List[Tuple[str,str]]:
    wt = details.get("hours_text") or []
    out = []
    for line in wt:
        parts = line.split(":", 1)
        if len(parts)==2:
            out.append((parts[0].strip(), parts[1].strip()))
    return out

def five_star_only(details: Dict[str, Any]) -> List[Dict[str, str]]:
    src = details.get("reviews") or []
    return [rv for rv in src if (rv.get("rating") or 0) == 5]




# ────────────────────────────────────────────────────────────────────────────
# Endpoints
# ────────────────────────────────────────────────────────────────────────────

# Root endpoint to prevent 404s on "/"
@app.get("/", include_in_schema=False)
async def root():
    """Simple landing endpoint for service root."""
    return {
        "ok": True,
        "service": "Restronaut Backend",
        "version": getattr(app, "version", "unknown"),
        "endpoints": [
            "/healthz",
            "/health/openai",
            "/suggest",
            "/details",
            "/generate/template",
            "/insights/presence",
            "/insights/audit"
        ],
    }

@app.get("/healthz", summary="Liveness probe")
async def healthz():
    return {"ok": True, "ts": time.time()}

# OpenAI health check endpoint
@app.get("/health/openai", summary="OpenAI connectivity probe")
async def health_openai():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return {"ok": False, "openai": "unreachable", "error": "OPENAI_API_KEY not set"}
    if OpenAI is None:
        return {"ok": False, "openai": "unreachable", "error": "OpenAI SDK not available"}
    try:
        client = OpenAI(api_key=api_key)
        # Try a lightweight call to check connectivity
        # Prefer listing models, fallback to a trivial completion if needed
        try:
            # Try listing models (should be fast and low-cost)
            models = client.models.list()
            # Check for gpt-4o-mini presence
            model_names = [m.id for m in getattr(models, "data", []) if hasattr(m, "id")]
            if "gpt-4o-mini" in model_names:
                return {"ok": True, "openai": "reachable", "model": "gpt-4o-mini"}
            # If not found, just return reachable
            return {"ok": True, "openai": "reachable", "model": model_names[0] if model_names else None}
        except Exception as e:
            # Fallback: try a trivial chat completion
            try:
                chat = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                    temperature=0,
                )
                # If we get here, OpenAI is reachable
                return {"ok": True, "openai": "reachable", "model": "gpt-4o-mini"}
            except Exception as e2:
                return {"ok": False, "openai": "unreachable", "error": str(e2)}
    except Exception as e:
        return {"ok": False, "openai": "unreachable", "error": str(e)}

@app.get("/suggest", summary="Typeahead suggestions (Google-first; Yelp fallback if empty)")
async def suggest(
    request: Request,
    q: str = Query(..., min_length=1),
    lat: Optional[float] = Query(None, ge=-90.0, le=90.0),
    lng: Optional[float] = Query(None, ge=-180.0, le=180.0),
    radius_m: int = Query(DEFAULT_RADIUS_M, ge=100, le=100000),
    limit: int = Query(10, ge=1, le=20),
    country: str = Query("us", min_length=2, max_length=2),
    _: None = Depends(rate_limit),
    __: None = Depends(security_guard),
):
    key = cache_key("suggest", q.lower(), round(lat or 0, 3), round(lng or 0, 3), radius_m, limit, country)
    cached = SUGGEST_CACHE.get(key)
    if cached is not None:
        return cached

    gdata = await google_autocomplete(q, lat=lat, lng=lng, radius_m=radius_m, country=country)
    g_results = normalize_suggest_google((gdata or {}).get("predictions", [])[:limit])

    if not g_results and YELP_API_KEY:
        ydata = await yelp_autocomplete(q, lat=lat, lng=lng, limit=limit)
        y_results = normalize_suggest_yelp(ydata)
        result = {"results": y_results}
        SUGGEST_CACHE.set(key, result)
        return result

    result = {"results": g_results}
    SUGGEST_CACHE.set(key, result)
    return result

@app.get("/insights/negative-reviews", summary="Fetch up to 3 low-rated reviews, preferring website/app complaints")
async def insights_negative_reviews(
    request: Request,
    id: str = Query(..., description="Google place_id or Yelp business id"),
    provider: Literal["google","yelp"] = Query("google"),
    limit: int = Query(3, ge=1, le=6),
    _: None = Depends(rate_limit),
    __: None = Depends(security_guard),
):
    # Reuse details fetch
    if provider == "google":
        details_payload = await google_details_normalized(id, include_chain_info=False)
    else:
        y = await yelp_business_details(id)
        details_payload = normalize_details_yelp(y)

    reviews = _filter_reviews(details_payload)
    preferred = mine_webapp_complaints(reviews, max_items=limit, low_star_only=True)
    source = "webapp_complaints_1star"
    if not preferred:
        preferred = mine_one_star(reviews, max_items=limit)
        source = "one_star"
        if not preferred:
            # last resort: 2★ complaints
            preferred = mine_webapp_complaints(reviews, max_items=limit, low_star_only=False)
            source = "webapp_complaints_lowstar"

    return {"id": details_payload.get("id"), "name": details_payload.get("name"), "source": source, "reviews": preferred}


@app.post("/insights/presence", summary="Compute online presence score and optional AI summary")
async def insights_presence(
    payload: GeneratePayload,
    request: Request,
    _: None = Depends(rate_limit),
    __: None = Depends(security_guard),
):
    # Build details object similar to generate_template
    details = payload.details
    if not details:
        if not payload.place_id:
            raise HTTPException(400, "Provide either details or place_id")
        if payload.provider == "google":
            details = await google_details_normalized(payload.place_id, include_chain_info=True)
        else:
            y = await yelp_business_details(payload.place_id)
            details = normalize_details_yelp(y)

    partner_mode = _is_partner_restaurant(details)

    # Keywords (payload may include a list, optional)
    kw_list = getattr(payload, "keywords", None) if hasattr(payload, "keywords") else None
    if isinstance(kw_list, list):
        kw_list = [k.strip() for k in kw_list if isinstance(k, str) and k.strip()]
    else:
        kw_list = []

    # Nearby (fetch as many as possible up to 60 within ~10 miles)
    nearby = await google_nearby_restaurants(details.get("lat"), details.get("lng"), radius_m=16093, limit=60)

    # Reviews (EXACTLY 3, adjust for partner locations)
    reviews_all = _filter_reviews(details)
    if partner_mode:
        five_star_reviews = [rv for rv in reviews_all if int(rv.get("rating") or 0) >= 5]
        if not five_star_reviews:
            raw_reviews = details.get("reviews") or []
            five_star_reviews = [rv for rv in raw_reviews if int(rv.get("rating") or 0) >= 5]
        complaints = five_star_reviews[:3]
        log.info(
            "presence/unified: partner review override for %s -> %d five-star quotes",
            details.get("name"),
            len(complaints),
        )
        if not complaints:
            complaints = select_negative_reviews(reviews_all, kw_list, limit=3)
            log.warning(
                "presence/unified: partner %s lacked 5★ reviews; fell back to %d negatives",
                details.get("name"),
                len(complaints),
            )
    else:
        complaints = select_negative_reviews(reviews_all, kw_list, limit=3)
        log.info("presence/unified: selected %d negative reviews (limit=%d)", len(complaints), 3)

    # Presence score (already includes recent 1★ keyword penalty if any)
    prescore = await compute_online_presence(details, keyword_list=kw_list or None, recent_days=90)

    # Kick off OpenAI calls concurrently to minimise wall-clock latency
    ai_line_task = asyncio.create_task(ai_presence_oneliner(details, prescore, complaints))
    ai_multi_task = asyncio.create_task(ai_presence_insights(details, prescore, nearby, complaints))

    # Nearby competitors with the same cuisine (target: all within ~3 miles)
    try:
        same_cuisine = await nearby_same_cuisine(
            details,
            radius_m=4828,
            min_results=7,
            scan_limit=60,
            concurrency=6,
            max_results=None,
        )
    except Exception as e:
        log.warning("presence: same-cuisine nearby failed: %s", e)
        same_cuisine = []

    ai_line_result, ai_multi_result = await asyncio.gather(ai_line_task, ai_multi_task, return_exceptions=True)

    # Resolve oneline output with safe fallback
    if isinstance(ai_line_result, Exception):
        log.warning("presence: ai oneline failed: %s", ai_line_result)
        ai_line = _presence_oneline_default(prescore)
    else:
        ai_line = ai_line_result

    fallback_insights = _local_insights_fallback(details, prescore, nearby, complaints)
    if isinstance(ai_multi_result, HTTPException):
        raise ai_multi_result
    if isinstance(ai_multi_result, Exception):
        log.warning("presence: ai insights failed: %s; using fallback", ai_multi_result)
        ai_multi = fallback_insights
    else:
        if ai_multi_result is None:
            log.warning("presence: OpenAI insights timed out; using fallback")
            ai_multi = fallback_insights
        else:
            ai_multi = ai_multi_result

    return {
        "id": details.get("id"),
        "name": details.get("name"),
        "nearby": nearby,
        "nearby_same_cuisine": same_cuisine,
        "presence": prescore,
        "reviews": complaints,     # exactly 3
        "ai_oneline": ai_line,     # persuasive one-liner
        "insights": ai_multi,      # optional structured insights (may be null)
    }
@app.get("/details", summary="Get business details")
async def details(
    request: Request,
    id: str = Query(..., description="Google place_id or Yelp business id"),
    provider: Literal["google", "yelp"] = Query("google"),
    include_chain_info: bool = Query(True),
    _: None = Depends(rate_limit),
    __: None = Depends(security_guard),
):
    key = cache_key("details", provider, id, int(include_chain_info))
    cached = DETAILS_CACHE.get(key)
    if cached is not None:
        return cached

    if provider == "google":
        try:
            result = await google_details_normalized(id, include_chain_info=include_chain_info)
            payload = {"result": result}
        except Exception as e:
            raise HTTPException(502, f"Failed to fetch Google details: {e}")
    else:
        try:
            y = await yelp_business_details(id)
            payload = {"result": normalize_details_yelp(y)}
            # Yelp does not support chain count here; omit
        except Exception as e:
            raise HTTPException(502, f"Failed to fetch Yelp details: {e}")

    DETAILS_CACHE.set(key, payload)
    return payload

@app.post("/generate/template", response_model=TemplateOut, summary="Generate a premium HTML landing page from details")
async def generate_template(
    payload: GeneratePayload,
    request: Request,
    _: None = Depends(rate_limit),
    __: None = Depends(security_guard),
):
    """Builds an HTML landing page using provided details or by fetching them.
    Returns HTML plus meta (palette, logo, cuisine, etc.).
    """
    # Resolve details
    details: Dict[str, Any]
    if payload.details:
        details = payload.details
    else:
        if not payload.place_id:
            raise HTTPException(400, "place_id is required when details are not provided")
        provider = (payload.provider or "google").lower()
        if provider == "google":
            try:
                details = await google_details_normalized(payload.place_id, include_chain_info=True)
            except Exception as e:
                raise HTTPException(502, f"Failed to fetch Google details: {e}")
        elif provider == "yelp":
            try:
                y = await yelp_business_details(payload.place_id)
                details = normalize_details_yelp(y)
            except Exception as e:
                raise HTTPException(502, f"Failed to fetch Yelp details: {e}")
        else:
            raise HTTPException(400, "Unsupported provider; use 'google' or 'yelp'")

    # Build the page
    try:
        html, meta = await build_html(details, sales_cta=payload.sales_cta)
        return TemplateOut(html=html, react=None, meta=meta)
    except Exception as e:
        log.error(f"Error building HTML: {e}")
        raise HTTPException(500, "Failed to generate template")
    
    

@app.post("/generate/mobile", response_model=TemplateOut, summary="Generate a premium mobile app concept screen")
async def generate_mobile_template(
    payload: GeneratePayload,
    request: Request,
    _: None = Depends(rate_limit),
    __: None = Depends(security_guard),
):
    try:
        details = await _fetch_details(payload.provider, payload.place_id, payload.details)
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"mobile_template: details fetch failed: {e}")
        raise HTTPException(500, "Failed to fetch details")

    try:
        # Force the endpoint to always use the pinned v3 builder, even if another build_mobile_app_html is defined later
        html, meta = await build_mobile_app_html_v3(details)
        return TemplateOut(html=html, react=None, meta=meta)
    except Exception as e:
        log.error(f"mobile_template build failed: {e}")
        raise HTTPException(500, "Failed to generate mobile template")
