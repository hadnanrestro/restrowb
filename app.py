# app.py
import os, re, time, json, hashlib, base64, io, hmac, asyncio, logging
from urllib.parse import urlparse, quote
from typing import Optional, Literal, Dict, Any, List, Tuple

from fastapi import FastAPI, HTTPException, Query, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()  # Load .env early so OPENAI_API_KEY and others are available
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
    
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


# ────────────────────────────────────────────────────────────────────────────
# Env & Config
# ────────────────────────────────────────────────────────────────────────────

GOOGLE_API_KEY  = os.getenv("GOOGLE_PLACES_API_KEY", "")
YELP_API_KEY    = os.getenv("YELP_API_KEY", "")  # optional fallback

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
CONTAINER = "mx-auto max-w-[1400px] xl:max-w-[1600px] px-4 sm:px-6 lg:px-8"
# Core Clean Restaurant Ambiance Images (Interior/Exterior)
U1 = "https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?q=80&w=1600&auto=format&fit=crop"  # Bright modern interior
U2 = "https://images.unsplash.com/photo-1555396273-367ea4eb4db5?q=80&w=1600&auto=format&fit=crop"    # Elegant dining
U3 = "https://images.unsplash.com/photo-1590846406792-0adc7f938f1d?q=80&w=1600&auto=format&fit=crop" # Clean minimalist
U4 = "https://images.unsplash.com/photo-1571003123894-1f0594d2b5d9?q=80&w=1600&auto=format&fit=crop" # Modern atmosphere

## American/Burger - Clean Modern
U5 = "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?q=80&w=1600&auto=format&fit=crop"    # Modern diner
U6 = "https://images.unsplash.com/photo-1533777857889-4be7c70b33f7?q=80&w=1600&auto=format&fit=crop" # Minimal burger joint

## Italian - Elegant Clean
U7 = "https://images.unsplash.com/photo-1414235077428-338989a2e8c0?q=80&w=1600&auto=format&fit=crop" # Fine dining
U8 = "https://images.unsplash.com/photo-1540189549336-e6e99c3679fe?q=80&w=1600&auto=format&fit=crop" # Italian trattoria

## Mexican - Bright Clean
U9 = "https://images.unsplash.com/photo-1585032226651-759b368d7246?q=80&w=1600&auto=format&fit=crop" # Modern cantina
U10 = "https://images.unsplash.com/photo-1578662996442-48f60103fc96?q=80&w=1600&auto=format&fit=crop" # Clean Mexican

## Asian - Clean Modern
U11 = "https://images.unsplash.com/photo-1537047902294-62a40c20a6ae?q=80&w=1600&auto=format&fit=crop" # Asian restaurant
U12 = "https://images.unsplash.com/photo-1514933651103-005eec06c04b?q=80&w=1600&auto=format&fit=crop" # Japanese minimal

# Updated cuisine mapping
HERO_BY_CUISINE = {
    "burger": [U5, U6, U1, U2],
    "italian": [U7, U8, U1, U2],
    "mexican": [U9, U10, U1, U2],
    "american": [U5, U6, U1, U2],
    "chinese": [U11, U12, U1, U2],
    "japanese": [U12, U11, U1, U2],
    "thai": [U11, U12, U1, U2],
    "indian": [U11, U12, U1, U2],  # Using Asian as base
    "greek": [U7, U8, U1, U2],     # Using Italian as base
    "french": [U7, U8, U1, U2],    # Using Italian as base
    "korean": [U11, U12, U1, U2],
    "mediterranean": [U7, U8, U1, U2],
    "vietnamese": [U11, U12, U1, U2],
}
# Safe, generic hero fallback if one of the hero URLs fails
HERO_FALLBACK_URL = "https://source.unsplash.com/1600x900/?restaurant,interior"
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
    if not url or not url.startswith(('http://', 'https://')):
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
# Menu image and hero image selection helpers (smarter)
# ────────────────────────────────────────────────────────────────────────────
async def select_hero_images(details: Dict[str, Any], cuisine: str) -> List[str]:
    """Smart hero image selection with fallbacks"""
    images: List[str] = []

    # 1) Use Google gallery interior shots first
    for img_url in (details.get("gallery") or []):
        if await _is_interior_shot(img_url):
            images.append(img_url)
            if len(images) >= 2:
                break

    # 2) Fill with curated Unsplash per cuisine
    cuisine_heroes = HERO_BY_CUISINE.get(cuisine, HERO_BY_CUISINE.get(DEFAULT_CUISINE, []))
    for hero_url in cuisine_heroes:
        if len(images) >= 4:
            break
        if hero_url not in images:
            images.append(hero_url)

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

async def _get_best_menu_image(item: Dict[str, Any], cuisine: str) -> str:
    """Get the best available image for a menu item"""
    img = (item.get("img") or "").strip()

    # 1) Google image search for specific dishes
    if img.startswith("gq:"):
        query = img[3:].strip()
        found = await google_image_search(f"{query} restaurant dish", num=3)
        if found:
            return found

    # 2) Provided image if valid
    if img and img.startswith(("http://", "https://")):
        v = await ensure_valid_image_url(img)
        if v:
            return v

    name = (item.get("name") or "").lower()

    # 3) Item-keyword fallbacks (more specific than cuisine-level)
    keyword_fallbacks = [
        ("chicken", "https://images.unsplash.com/photo-1606755962773-d324e2dabd17?q=80&w=1200&auto=format&fit=crop"),  # crispy chicken sandwich
        ("fries",   "https://images.unsplash.com/photo-1576107232684-1279f390859f?q=80&w=1200&auto=format&fit=crop"),  # fries
        ("salad",   "https://images.unsplash.com/photo-1540420773420-3366772f4999?q=80&w=1200&auto=format&fit=crop"),  # salad
        ("steak",   "https://images.unsplash.com/photo-1544025162-d76694265947?q=80&w=1200&auto=format&fit=crop"),  # steak
        ("pizza",   "https://images.unsplash.com/photo-1548365328-9f547fb0953f?q=80&w=1200&auto=format&fit=crop"),  # pizza
        ("burger",  "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?q=80&w=1200&auto=format&fit=crop"),  # burger
        ("ramen",   "https://images.unsplash.com/photo-1569718212165-3a8278d5f624?q=80&w=1200&auto=format&fit=crop"),  # ramen
        ("sushi",   "https://images.unsplash.com/photo-1553621042-f6e147245754?q=80&w=1200&auto=format&fit=crop"),  # sushi
        ("taco",    "https://images.unsplash.com/photo-1565299585323-38174c267b34?q=80&w=1200&auto=format&fit=crop"),  # tacos
    ]
    for kw, url_fallback in keyword_fallbacks:
        if kw in name:
            v = await ensure_valid_image_url(url_fallback)
            if v:
                return v

    # 4) Cuisine fallback dish images (last resort)
    cuisine_dishes = {
        "burger": "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?q=80&w=1200&auto=format&fit=crop",
        "italian": "https://images.unsplash.com/photo-1604382354936-07c5d9983bd3?q=80&w=1200&auto=format&fit=crop",
        "mexican": "https://images.unsplash.com/photo-1565299585323-38174c267b34?q=80&w=1200&auto=format&fit=crop",
        "chinese": "https://images.unsplash.com/photo-1525755662778-989d0524087e?q=80&w=1200&auto=format&fit=crop",
        "thai": "https://images.unsplash.com/photo-1559847844-5315695dadae?q=80&w=1200&auto=format&fit=crop",
        "japanese": "https://images.unsplash.com/photo-1579871494447-9811cf80d66c?q=80&w=1200&auto=format&fit=crop",
        "american": "https://images.unsplash.com/photo-1555992336-03a23c7b7887?q=80&w=1200&auto=format&fit=crop",
    }
    return cuisine_dishes.get(cuisine, cuisine_dishes["american"])
# Update the CUISINE_ASSETS with more appropriate theme colors
CUISINE_ASSETS: Dict[str, Dict[str, Any]] = {
    "indian": {
        "palette": {"primary":"#D97706","primary_dark":"#B45309"},  # Warm saffron/turmeric
        "hero": HERO_BY_CUISINE["indian"],
        "menu": [
            {"name":"Chicken Tikka Masala","desc":"Creamy tomato sauce, basmati rice","price":"$14.99","img": "https://images.unsplash.com/photo-1585937421612-70a008356fbe?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Lamb Biryani","desc":"Fragrant basmati rice, spices","price":"$16.99","img": "https://images.unsplash.com/photo-1563379091339-03246963d96c?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Garlic Naan","desc":"Fresh baked bread","price":"$3.99","img": "https://images.unsplash.com/photo-1601050690597-df0568f70950?q=80&w=1200&auto=format&fit=crop"}
        ]
    },
    "burger": {
        "palette": {"primary":"#DC2626","primary_dark":"#B91C1C"},  # Classic red
        "hero": HERO_BY_CUISINE["burger"],
        "menu": [
            {"name":"Double Smash Burger","desc":"American cheese, pickles, shack sauce","price":"$10.99","img": "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Crispy Chicken Sandwich","desc":"Buttermilk fried chicken, slaw","price":"$8.99","img": "https://images.unsplash.com/photo-1700768400970-428c50bffc11?q=80&w=764&auto=format&fit=crop"},
            {"name":"Crinkle Cut Fries","desc":"Sea salt, extra crispy","price":"$3.99","img": "https://images.unsplash.com/photo-1576107232684-1279f390859f?q=80&w=1200&auto=format&fit=crop"}
        ]
    },
    "italian": {
        "palette": {"primary":"#059669","primary_dark":"#047857"},  # Italian flag green
        "hero": HERO_BY_CUISINE["italian"],
        "menu": [
            {"name":"Chicken Alfredo","desc":"Creamy parmesan sauce, fettuccine","price":"$14.99","img": "https://images.unsplash.com/photo-1621996346565-e3dbc353d2e5?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Spaghetti & Meatballs","desc":"San Marzano tomatoes, basil","price":"$12.99","img": "https://images.unsplash.com/photo-1551183053-bf91a1d81141?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Margherita Pizza","desc":"Fresh mozzarella, tomato, basil","price":"$11.49","img": "https://images.unsplash.com/photo-1604382354936-07c5d9983bd3?q=80&w=1200&auto=format&fit=crop"}
        ]
    },
    "mexican": {
        "palette": {"primary":"#EA580C","primary_dark":"#C2410C"},  # Vibrant orange-red
        "hero": HERO_BY_CUISINE["mexican"],
        "menu": [
            {"name":"Carne Asada Tacos","desc":"Cilantro, onions, lime","price":"$9.49","img": "https://images.unsplash.com/photo-1565299585323-38174c267b34?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Chicken Quesadilla","desc":"Three-cheese blend, pico","price":"$8.99","img": "https://images.unsplash.com/photo-1618040996337-56904b7850b9?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Chips & Guacamole","desc":"House-made","price":"$5.99","img": "https://images.unsplash.com/photo-1541544741938-0af808871cc0?q=80&w=1200&auto=format&fit=crop"}
        ]
    },
    "chinese": {
        "palette": {"primary":"#DC2626","primary_dark":"#B91C1C"},  # Traditional red
        "hero": HERO_BY_CUISINE["chinese"],
        "menu": [
            {"name":"General Tso's Chicken","desc":"Sweet and spicy with steamed rice","price":"$12.99","img": "https://images.unsplash.com/photo-1525755662778-989d0524087e?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Beef Lo Mein","desc":"Soft noodles with vegetables","price":"$11.49","img": "https://images.unsplash.com/photo-1582878826629-29b7ad1cdc43?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Pork Dumplings","desc":"Pan-fried, served with soy sauce","price":"$7.99","img": "https://images.unsplash.com/photo-1563379091339-03246963d96c?q=80&w=1200&auto=format&fit=crop"}
        ]
    },
    "thai": {
        "palette": {"primary":"#7C3AED","primary_dark":"#6D28D9"},  # Thai purple
        "hero": HERO_BY_CUISINE["thai"],
        "menu": [
            {"name":"Pad Thai","desc":"Rice noodles, shrimp, bean sprouts","price":"$12.99","img": "https://images.unsplash.com/photo-1559847844-5315695dadae?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Green Curry","desc":"Coconut milk, basil, jasmine rice","price":"$13.49","img": "https://images.unsplash.com/photo-1604908176997-125f25cc6f3d?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Tom Yum Soup","desc":"Spicy and sour with prawns","price":"$9.99","img": "https://images.unsplash.com/photo-1596040033229-a9821ebd058d?q=80&w=1200&auto=format&fit=crop"}
        ]
    },
    "japanese": {
        "palette": {"primary":"#1F2937","primary_dark":"#111827"},  # Elegant dark
        "hero": HERO_BY_CUISINE["japanese"],
        "menu": [
            {"name":"Salmon Teriyaki","desc":"Grilled with steamed vegetables","price":"$16.99","img": "https://images.unsplash.com/photo-1579871494447-9811cf80d66c?q=80&w=1200&auto=format&fit=crop"},
            {"name":"California Roll","desc":"Crab, avocado, cucumber","price":"$8.99","img": "https://images.unsplash.com/photo-1553621042-f6e147245754?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Chicken Ramen","desc":"Rich broth with soft-boiled egg","price":"$13.49","img": "https://images.unsplash.com/photo-1569718212165-3a8278d5f624?q=80&w=1200&auto=format&fit=crop"}
        ]
    },
    "american": {
        "palette": {"primary":"#EC1111","primary_dark":"#2563EB"},  # Classic blue (not orange!)
        "hero": HERO_BY_CUISINE["american"],
        "menu": [
            {"name":"Roast Chicken Plate","desc":"Choice of two sides","price":"$10.99","img": "https://images.unsplash.com/photo-1598515214211-89d3c73ae83b?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Country Fried Steak","desc":"Pepper gravy, mashed potatoes","price":"$11.49","img": "https://images.unsplash.com/photo-1562967916-eb82221dfb92?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Vegetable Plate","desc":"Pick any three sides","price":"$8.99","img": "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?q=80&w=1200&auto=format&fit=crop"}
        ]
    }
}

DEFAULT_CUISINE = "american"

def cuisine_from_types(details: Dict[str, Any]) -> str:
    name = (details.get("name") or "").lower()
    types = [t.lower() for t in (details.get("categories") or [])]
    
    # More comprehensive keyword matching
    cuisine_keywords = {
        "indian": {
            "types": ["indian_restaurant", "restaurant", "meal_takeaway", "food", "establishment"],
            "name_keywords": ["indian", "india", "desi", "curry", "tandoor", "masala", "biryani", 
                            "punjabi", "bengali", "south indian", "north indian", "spice", "aroma"],
            "priority": 1  # Higher priority for specific cuisines
        },
        "chinese": {
            "types": ["chinese_restaurant", "restaurant", "meal_takeaway"],
            "name_keywords": ["chinese", "china", "wok", "dragon", "golden", "panda", "beijing", "shanghai"],
            "priority": 1
        },
        "italian": {
            "types": ["italian_restaurant", "restaurant", "meal_takeaway"],
            "name_keywords": ["pizza", "pizzeria", "italian", "pasta", "trattoria", "ristorante", "mario", "luigi"],
            "priority": 1
        },
        "mexican": {
            "types": ["mexican_restaurant", "restaurant", "meal_takeaway"],
            "name_keywords": ["taco", "tacos", "mexican", "cantina", "casa", "el ", "la ", "burrito", "quesadilla"],
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
            "name_keywords": ["american", "diner", "cafe", "grill", "bar", "steakhouse"],
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
    """Heuristic 0–100 score with contributing factors."""
    score = 100
    reasons: List[str] = []

    rating = float(details.get("rating") or 0)
    reviews_ct = int(details.get("review_count") or 0)
    if rating < 3.5:
        score -= 15
        reasons.append(f"Low average rating ({rating:.1f})")
    if reviews_ct < 100:
        score -= 10
        reasons.append("Low review volume (<100)")

    if not details.get("website"):
        score -= 20
        reasons.append("No official website listed")

    if len(details.get("gallery") or []) < 3:
        score -= 8
        reasons.append("Few photos in Google gallery")

    if not details.get("hours_text"):
        score -= 6
        reasons.append("Missing operating hours")

    # Chains nearby -> harder to stand out
    if (details.get("chain_count_nearby") or 0) >= 2:
        score -= 6
        reasons.append("Nearby chain competition is high")

    # Recent 1★ keyword complaints (e.g., website/app) within the last N days
    try:
        reviews = _filter_reviews(details)
        kws = keyword_list if (keyword_list and len(keyword_list) > 0) else [
            'website','online order','online ordering','order online','mobile app','app','hours','menu','checkout','payment'
        ]
        recent_hits = _recent_keyword_one_stars(reviews, keywords=kws, within_days=recent_days)
        if recent_hits:
            score -= 12
            reasons.append(f"{recent_hits} recent 1★ mention(s) of website/app issues")
    except Exception:
        pass

    # Infer whether they have a mobile app (heuristic)
    has_mobile_app = False
    try:
        urls_to_check = [details.get('website') or '', details.get('map_url') or '']
        def _looks_like_app_link(u: str) -> bool:
            s = (u or '').lower()
            return ('apps.apple.com' in s) or ('play.google.com' in s) or ('/app' in s) or (s.split('://')[-1].startswith('app.'))
        if any(_looks_like_app_link(u) for u in urls_to_check):
            has_mobile_app = True
        else:
            for rv in reviews:
                t = (rv.get('text') or '').lower()
                if ('mobile app' in t) or re.search(r'\bthe app\b', t) or re.search(r'\bapp\b', t):
                    has_mobile_app = True
                    break
    except Exception:
        pass

    # Explicit reason if mobile app not found
    if not has_mobile_app:
        # avoid duplicate phrasing if already added by earlier logic
        if not any("mobile app" in r.lower() for r in reasons):
            reasons.append("Mobile app availability not found")

    # Brand signal (logo color)
    try:
        _lu, _lc, _rs = await best_logo_with_color(details)
        if not _lc:
            score -= 4
            reasons.append("No detectable brand color/logo theme")
    except Exception:
        pass

    # Ensure we surface at least 4 reasons
    website = details.get("website")
    most_recent_days = None
    try:
        # Try to find the most recent review days for context
        if reviews:
            days_list = [_relative_time_to_days(rv.get("relative_time")) for rv in reviews if _relative_time_to_days(rv.get("relative_time")) is not None]
            if days_list:
                most_recent_days = min(days_list)
    except Exception:
        pass
    chain_cnt = details.get("chain_count_nearby") or 0
    if len(reasons) < 4:
        if (most_recent_days is None) or (most_recent_days is not None and most_recent_days > 30):
            reasons.append("Low recent review activity")
        if website:
            reasons.append("Website performance/clarity unknown—optimize for mobile speed and menu visibility")
        if chain_cnt >= 1 and all('chain' not in r for r in reasons):
            reasons.append("Some nearby chain presence")
        if len(reasons) < 4:
            reasons.append("Brand consistency can be improved online")

    score = max(0, min(100, score))
    score = max(0, min(100, score - 15))
    score = max(0, min(100, score - 20))
    level = "Excellent" if score >= 85 else "Good" if score >= 70 else "Needs Work" if score >= 50 else "At Risk"
    rating = float(details.get("rating") or 0)
    reviews_ct = int(details.get("review_count") or 0)
    metrics = {
        "rating": rating,
        "review_count": reviews_ct,
        "most_recent_review_days": most_recent_days,
        "has_website": bool(website),
        "has_mobile_app": has_mobile_app,
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

    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key and OpenAI is not None:
        try:
            client = OpenAI(api_key=api_key)
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
        except Exception as e:
            log.debug(f"ai_oneline: OpenAI initialization failed: {e}")

    # Fallback deterministic line
    if reasons:
        return _sanitize(f"Your online presence is slipping, {reasons[0].rstrip('.')}; let us fix it with a fast, modern site that converts more customers.")
    return "A stronger website and clearer information can win you more customers; let us help you upgrade quickly."

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

async def google_nearby_chain_count(name: str, lat: Optional[float], lng: Optional[float], self_place_id: str, radius_m: int) -> int:
    if not GOOGLE_API_KEY or lat is None or lng is None or not name:
        return 0
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
    return count

async def google_nearby_restaurants(lat: Optional[float], lng: Optional[float], *, radius_m: int = 16093, limit: int = 60) -> List[Dict[str, Any]]:
    """List nearby restaurants within radius_m (~10 miles) with basic info.
    Paginates through Google Nearby Search to return up to `limit` results.
    """
    if not GOOGLE_API_KEY or lat is None or lng is None:
        return []
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
        fallback = f"https://www.google.com/s2/favicons?sz=256&domain={host}"
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

# HTML builder
async def build_html(details: Dict[str, Any], *, sales_cta: bool) -> Tuple[str, Dict[str, Any]]:
    name = details.get("name") or "Restaurant"
    address = details.get("address") or ""
    website = details.get("website") or ""
    phone = details.get("phone") or ""
    rating = details.get("rating")
    review_count = details.get("review_count")
    map_url = details.get("map_url") or "#"

    # Use improved cuisine detection
    cuisine = cuisine_from_types(details)

    # Get restaurant context
    context = get_restaurant_context(details)

    # Select appropriate theme colors
    # Get logo and brand color
    logo, logo_color, logo_reason = await best_logo_with_color(details)
    if not logo_color:
        log.info("ColorThief could not extract a brand color for %s; falling back to cuisine palette", details.get("name"))
    pal = select_theme_colors(cuisine, context, logo_color)

    # Get assets for the detected cuisine
    assets = CUISINE_ASSETS.get(cuisine, CUISINE_ASSETS[DEFAULT_CUISINE])

    # Log the detection results
    log.info("BUILD PAGE: name=%s cuisine=%s context=%s colors=%s logo_color=%s",
             name, cuisine, context["atmosphere"], pal["primary"], logo_color)
    
    # Smart hero image selection (prefers interior shots from Google, then curated Unsplash)
    hero_imgs: List[str] = await select_hero_images(details, cuisine)

    await resolve_menu_images(details, cuisine)
    raw_menu_items: List[Dict[str, str]] = list((details.get("_resolved_menu") or [])[:3])
    menu_items: List[Dict[str, str]] = []
    for it in raw_menu_items:
        # Ensure menu images have proper URLs
        img_url = it.get("img", "")
        if img_url and not img_url.startswith(('http://', 'https://')):
            # If it's still a photo- ID, convert to full Unsplash URL
            if img_url.startswith("photo-"):
                img_url = f"https://images.unsplash.com/{img_url}?q=80&w=1600&auto=format&fit=crop"
            else:
                img_url = ""
        
        menu_items.append({
            "name": it["name"],
            "desc": it["desc"],
            "price": it["price"],
            "img": img_url,
        })

    # Rest of the function remains the same...
    revs = five_star_only(details)
    show_reviews = len(revs) >= 2
    revs = revs[:4]
    gallery_raw = details.get("gallery") or []
    
    # Use only Google-provided gallery images; keep unique and limit to 4
    gallery: List[str] = []
    for u in gallery_raw:
        if u and u not in gallery:
            gallery.append(u)
    gallery = gallery[:4]
    
    hrs = hours_list(details)
    fallback_foto = assets.get("fallback") or HERO_FALLBACK_URL

    # Continue with HTML generation using the dynamic palette...
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{safe(name)}</title>
<link rel="icon" href="{safe(logo or '')}"/>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;700&display=swap" rel="stylesheet"/>
<script src="https://cdn.tailwindcss.com"></script>
<script>
tailwind.config = {{
  theme: {{
    extend: {{
      colors: {{
        brand: "{pal['primary']}",
        brandd: "{pal['primary_dark']}",
        pearl: "#F7F3EE",
        ink: "#1B1B1B"
      }},
      fontFamily: {{
        display: ['Playfair Display','serif'],
        body: ['Inter','system-ui','-apple-system','Segoe UI','Roboto','sans-serif']
      }},
      boxShadow: {{
        soft: "0 10px 30px rgba(0,0,0,.12)",
        card: "0 12px 40px rgba(0,0,0,.12)"
      }}
    }}
  }}
}}
</script>
<style>
  html,body{{background:linear-gradient(#FBF8F3,#F5F2ED) fixed; color:#1B1B1B;}}
  .glass{{background:rgba(255,255,255,.55); backdrop-filter: blur(12px); border:1px solid rgba(0,0,0,.06);}}
  .fade-wrap{{position:relative;height:clamp(320px,60vh,820px);overflow:hidden;border-radius:1.5rem}}
  @media (min-width:1024px){{.fade-wrap{{height:clamp(420px,62vh,880px)}}}}
  @media (min-width:1536px){{.fade-wrap{{height:clamp(520px,64vh,920px)}}}}
  .fade-wrap img{{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;object-position:center;opacity:0;transition:opacity 900ms ease-in-out;filter:saturate(1.05) contrast(1.06) brightness(0.98);transform-origin:center;animation:hero-zoom 18s ease-in-out infinite alternate}}
  .fade-wrap img.active{{opacity:1}}
  @keyframes hero-zoom{{from{{transform:scale(1.02)}}to{{transform:scale(1.08)}}}}
  .dot{{width:8px;height:8px;border-radius:9999px;background:#0003}}
  @media (min-width:1536px){{.dot{{width:10px;height:10px}}}}
  .dot.active{{background:{pal['primary']};}}
  .card{{background:#FFF;border-radius:1.25rem;box-shadow:var(--tw-shadow, 0 10px 30px rgba(0,0,0,.10));}}
</style>
<meta http-equiv="Content-Security-Policy"
  content="default-src 'self'; img-src 'self' data: https:; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com; connect-src 'self' https:">
<meta name="referrer" content="no-referrer">
</head>
<body class="font-body">

  <header class="sticky top-0 z-50">
    <nav class="{CONTAINER} py-3 flex items-center justify-between bg-white/80 backdrop-blur border-b border-black/5">
      <a class="flex items-center gap-3" href="#top" aria-label="{safe(name)}">
        {"<img src='"+safe(logo)+"' class='h-7 w-7 rounded object-contain' alt='logo'/>" if logo else ""}
        <span class="font-semibold tracking-wide">{safe(name)}</span>
      </a>
      <ul class="hidden md:flex items-center gap-6 text-sm">
        <li><a href="#menu" class="hover:text-brand">Menu</a></li>
        <li><a href="#about" class="hover:text-brand">About</a></li>
        <li><a href="#gallery" class="hover:text-brand">Gallery</a></li>
        <li><a href="#reviews" class="hover:text-brand">Reviews</a></li>
        <li><a href="#contact" class="hover:text-brand">Contact</a></li>
      </ul>
      <div class="flex items-center gap-2">
        <a href="{safe(website or map_url)}" target="_blank" rel="noopener" class="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-brand text-white font-semibold hover:bg-brandd transition">Order Online</a>
      </div>
    </nav>
  </header>

  <section id="top" class="pt-6">
    <div class="{CONTAINER}">
      <div class="fade-wrap shadow-soft" id="heroWrap">
        {"".join([f"<img class='hero-img {'active' if i==0 else ''}' src='{safe(u)}' alt='hero image {i+1}' {'loading=\"eager\"' if i==0 else 'loading=\"lazy\"'} decoding='async' />" for i,u in enumerate(hero_imgs)])}
      </div>
      <div class="relative -mt-10 md:-mt-12">
        <div class="glass rounded-3xl p-6 md:p-8 shadow-soft">
          <div class="md:flex md:items-end md:justify-between gap-8">
            <div>
              <h1 class="font-display text-4xl md:text-5xl tracking-tight">{safe(name)}</h1>
              <p class="mt-3 opacity-80">{safe(address)}</p>
              <div class="mt-4 flex flex-wrap gap-2">
                {"".join([
                  f"<span class='inline-flex items-center rounded-full px-3 py-1 text-sm bg-brand/10 text-brand'>★ {rating:.1f}/5</span>" if rating else "",
                  f"<span class='inline-flex items-center rounded-full px-3 py-1 text-sm bg-black/5'>{int(review_count)}+ reviews</span>" if review_count else "",
                  f"<span class='inline-flex items-center rounded-full px-3 py-1 text-sm bg-black/5'>$ · Affordable</span>" if (details.get('price_level') is not None) else ""
                ])}
              </div>
            </div>
            <div class="mt-6 md:mt-0 shrink-0 flex gap-3">
              <a href="#menu" class="px-5 py-3 rounded-xl bg-brand text-white font-semibold hover:bg-brandd transition">See Menu</a>
              <a href="{safe(map_url)}" target="_blank" rel="noopener" class="px-5 py-3 rounded-xl border border-black/10 hover:bg-black/5 transition">Get Directions</a>
            </div>
          </div>
        </div>
        <div class="mt-3 flex items-center justify-center gap-2" id="heroDots">
          {"".join([f"<div class='dot {'active' if i==0 else ''}' data-idx='{i}'></div>" for i in range(len(hero_imgs))])}
        </div>
      </div>
    </div>
  </section>

  <section id="menu" class="mt-12 md:mt-16">
    <div class="{CONTAINER}">
      <div class="flex items-end justify-between gap-4">
        <h2 class="font-display text-3xl md:text-4xl">Menu Highlights</h2>
        <a href="{safe(website or map_url)}" target="_blank" rel="noopener" class="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-brand text-white font-semibold hover:bg-brandd transition shadow-soft">Order Online</a>
      </div>
      <div class="mt-6 grid md:grid-cols-2 lg:grid-cols-3 gap-7">
        {"".join([f"""
        <article class="card overflow-hidden">
          <div class="aspect-[4/3] w-full overflow-hidden">
            <img class="w-full h-full object-cover"
                 src="{safe(item['img'])}"
                 alt="{safe(item['name'])}"
                 loading="lazy"
                 onerror="this.style.display='none'; this.nextElementSibling.style.display='flex'"/>
            <div style="display:none" class="w-full h-full bg-gray-100 flex items-center justify-center text-gray-400">
              <span>Image not available</span>
            </div>
          </div>
          <div class="p-5 md:p-6 flex items-start justify-between gap-3">
            <div>
              <h3 class="font-semibold text-lg">{safe(item['name'])}</h3>
              <p class="mt-1 opacity-75 text-sm">{safe(item['desc'])}</p>
            </div>
            <span class="inline-flex items-center rounded-full px-3 py-1 text-sm bg-brand/15 text-brand">{safe(item['price'])}</span>
          </div>
        </article>
        """ for item in menu_items])}
      </div>
      <p class="mt-3 text-sm opacity-70">*Pricing and availability may vary by location.</p>
    </div>
  </section>

  <section id="about" class="mt-12 md:mt-16">
    <div class="{CONTAINER} grid md:grid-cols-2 gap-7">
      <article class="card p-6 md:p-8">
        <h2 class="font-display text-2xl md:text-3xl">About Us</h2>
        <p class="mt-3 text-[17px] leading-7 opacity-90">
          {safe(details.get("summary") or "Comforting classics made from scratch daily, served with warm hospitality.")}
        </p>
        <div class="mt-5 flex flex-wrap gap-2">
          <span class="inline-flex items-center rounded-full px-3 py-1 text-sm bg-brand/10 text-brand">Family Friendly</span>
          <span class="inline-flex items-center rounded-full px-3 py-1 text-sm bg-black/5">Catering Available</span>
        </div>
      </article>

      <aside class="card p-6 md:p-8">
        <h2 class="font-display text-2xl md:text-3xl">Operating Hours</h2>
        <ul class="mt-4 divide-y divide-black/5">
          {"".join([f"<li class='flex justify-between py-2'><span>{safe(d)}</span><span class='font-medium'>{safe(h)}</span></li>" for d,h in (hrs or [])])}
        </ul>
        <div class="mt-4 text-sm opacity-75">Hours may vary on holidays.</div>
      </aside>
    </div>
  </section>

  <section id="gallery" class="mt-12 md:mt-16">
    <div class="{CONTAINER}">
      <h2 class="font-display text-3xl md:text-4xl">Gallery</h2>
      <div class="mt-6 grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {"".join([
          f"<a href='{safe(u)}' target='_blank' rel='noopener' class='block rounded-2xl overflow-hidden card'><img src='{safe(u)}' data-fallbacks='{safe(fallback_foto)}' onerror='__imgSwap(this)' class='w-full h-48 object-cover' loading='lazy' decoding='async'/></a>"
          for u in (gallery[:4] or [])
        ])}
      </div>
    </div>
  </section>

  {""
  if not show_reviews else
  f'''
  <section id="reviews" class="mt-12 md:mt-16">
    <div class="{CONTAINER}">
      <div class="flex items-end justify-between gap-4">
        <h2 class="font-display text-3xl md:text-4xl">What Guests Say</h2>
        {("<div class='opacity-80'>Google rating <span class='font-semibold'>"+str(rating)+"</span> · "+str(int(review_count))+"+ reviews</div>") if rating and review_count else ""}
      </div>
      <div class="mt-6 grid md:grid-cols-2 gap-7">
        { "".join([f"""
        <article class="card p-6">
          <div class="flex items-center gap-3">
            <img src="{safe(rv.get('profile_photo_url') or '')}" class="h-12 w-12 rounded-full object-cover" alt="avatar"/>
            <div>
              <div class="font-semibold">{safe(rv.get('author_name') or 'Guest')}</div>
              <div class="text-sm text-brand">★★★★★</div>
            </div>
          </div>
          <p class="mt-4 opacity-90">{safe(rv.get('text') or '')}</p>
          <div class="mt-3 text-sm opacity-60">{safe(rv.get('relative_time') or '')}</div>
        </article>""" for rv in revs]) }
      </div>
      <div class="mt-3 text-xs opacity-60">Verified 5★ Google reviews.</div>
    </div>
  </section>
  '''}

  <section id="contact" class="mt-12 md:mt-16 mb-16">
    <div class="{CONTAINER}">
      <div class="card p-8 grid md:grid-cols-3 gap-6">
        <div>
          <h3 class="font-display text-2xl">Visit Us</h3>
          <p class="mt-2 opacity-85">{safe(address)}</p>
          {"<p class='mt-2'><a class='underline' href='tel:"+safe(phone)+"'>"+safe(phone)+"</a></p>" if phone else ""}
        </div>
        <div>
          <h3 class="font-display text-2xl">Online</h3>
          {"<p class='mt-2'><a class='underline' href='"+safe(website)+"' target='_blank' rel='noopener'>"+safe(website)+"</a></p>" if website else "<p class='mt-2 opacity-75'>Website not provided</p>"}
          <p class="mt-2"><a class="underline" href="{safe(map_url)}" target="_blank" rel="noopener">Google Maps</a></p>
        </div>
        <div class="flex items-end md:items-center md:justify-end">
          <a href="{safe(website or map_url)}" target="_blank" rel="noopener" class="inline-flex items-center gap-2 px-5 py-3 rounded-xl bg-brand text-white font-semibold hover:bg-brandd transition">Book / Order</a>
        </div>
      </div>
      <div class="mt-6 text-center opacity-70">
        <span>© {time.strftime("%Y")} {safe(name)} • Created by <strong>Restronaut.ai</strong></span>
      </div>
    </div>
  </section>

<script>
  function __imgSwap(img) {{
    try {{
      var list = (img.getAttribute('data-fallbacks') || '').split('|').filter(Boolean);
      img.__idx = img.__idx || 0;
      if (img.__idx < list.length) {{
        img.src = list[img.__idx++];
      }}
    }} catch (e) {{}}
  }}

  (function() {{
    var wrap = document.getElementById('heroWrap');
    if (!wrap) return;

    var imgs = Array.prototype.slice.call(wrap.querySelectorAll('img'));
    var dotsWrap = document.getElementById('heroDots');
    var dots = dotsWrap ? Array.prototype.slice.call(dotsWrap.querySelectorAll('.dot')) : [];
    var idx = 0;

    function show(i) {{
      imgs.forEach(function(im, k) {{ im.classList.toggle('active', k === i); }});
      dots.forEach(function(d, k) {{ d.classList.toggle('active', k === i); }});
      idx = i;
    }}

    dots.forEach(function(d) {{
      d.addEventListener('click', function() {{
        var n = parseInt(d.getAttribute('data-idx') || '0', 10);
        show(n);
      }});
    }});

    setInterval(function() {{
      show((idx + 1) % imgs.length);
    }}, 5000);
  }})();
</script>

</body>
</html>
"""
    meta = {
        "palette": pal,
        "logo_url": logo,
        "cuisine": cuisine,
        "name": name,
        "address": address,
        "website": website,
        "map_url": map_url,
        "logo_color": logo_color,
        "logo_reason": logo_reason,
    }
    return html, meta


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
        data = await google_details(id)
        details_payload = normalize_details_google(data)
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
            g = await google_details(payload.place_id)
            details = normalize_details_google(g)
            try:
                chain_count = await google_nearby_chain_count(
                    details.get("name") or "",
                    details.get("lat"), details.get("lng"), details.get("id"), CHAIN_RADIUS_M
                )
            except Exception:
                chain_count = 0
            details["chain_count_nearby"] = chain_count
            details["is_chain_nearby"] = bool(chain_count and chain_count >= 1)
        else:
            y = await yelp_business_details(payload.place_id)
            details = normalize_details_yelp(y)

    # Keywords (payload may include a list, optional)
    kw_list = getattr(payload, "keywords", None) if hasattr(payload, "keywords") else None
    if isinstance(kw_list, list):
        kw_list = [k.strip() for k in kw_list if isinstance(k, str) and k.strip()]
    else:
        kw_list = []

    # Nearby (fetch as many as possible up to 60 within ~10 miles)
    nearby = await google_nearby_restaurants(details.get("lat"), details.get("lng"), radius_m=16093, limit=60)

    # Reviews (EXACTLY 3, with keyword preference then fallbacks)
    reviews_all = _filter_reviews(details)
    complaints = select_negative_reviews(reviews_all, kw_list, limit=3)
    log.info("presence/unified: selected %d negative reviews (limit=%d)", len(complaints), 3)

    # Presence score (already includes recent 1★ keyword penalty if any)
    prescore = await compute_online_presence(details, keyword_list=kw_list or None, recent_days=90)

    # One-line AI pitch
    ai_line = await ai_presence_oneliner(details, prescore, complaints)

    return {
        "id": details.get("id"),
        "name": details.get("name"),
        "nearby": nearby,
        "presence": prescore,
        "reviews": complaints,     # exactly 3
        "ai_oneline": ai_line,     # persuasive one-liner
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
            g = await google_details(id)
            payload = {"result": normalize_details_google(g)}
            if include_chain_info:
                r = payload["result"]
                try:
                    chain_count = await google_nearby_chain_count(
                        r.get("name") or "",
                        r.get("lat"),
                        r.get("lng"),
                        r.get("id") or id,
                        CHAIN_RADIUS_M,
                    )
                except Exception:
                    chain_count = 0
                r["chain_count_nearby"] = chain_count
                r["is_chain_nearby"] = bool(chain_count and chain_count >= 1)
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
                g = await google_details(payload.place_id)
                details = normalize_details_google(g)
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