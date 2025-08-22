# app.py
import os, re, time, json, hashlib, base64, io, hmac, ipaddress, asyncio, logging, socket
from urllib.parse import urlparse, quote
from typing import Optional, Literal, Dict, Any, List, Tuple

from fastapi import FastAPI, HTTPException, Query, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import httpx
from httpx import AsyncClient, Limits, Timeout
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse

# ────────────────────────────────────────────────────────────────────────────
# Env & Config
# ────────────────────────────────────────────────────────────────────────────
load_dotenv()

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
## Core Clean Restaurant Ambiance Images (Interior/Exterior) 
U1 = "https://images.unsplash.com/photo-1555396273-367ea4eb4db5?q=80&w=1600&auto=format&fit=crop" # Modern clean restaurant interior
U2 = "https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?q=80&w=1600&auto=format&fit=crop" # Bright restaurant interior 
U3 = "https://images.unsplash.com/photo-1590846406792-0adc7f938f1d?q=80&w=1600&auto=format&fit=crop" # Clean dining space
U4 = "https://images.unsplash.com/photo-1551218808-94e220e084d2?q=80&w=1600&auto=format&fit=crop" # Minimalist restaurant
U5 = "https://images.unsplash.com/photo-1466978913421-dad2ebd01d17?q=80&w=1600&auto=format&fit=crop" # Clean kitchen/dining view
#U6 = "https://images.unsplash.com/photo-1571997478779-2adcbbe9ab2f?q=80&w=1600&auto=format&fit=crop" # Clean restaurant exterior
U7 = "https://images.unsplash.com/photo-1522708323590-d24dbb6b0267?q=80&w=1600&auto=format&fit=crop" # Clean bar/dining area
U8 = "https://images.unsplash.com/photo-1559339352-11d035aa65de?q=80&w=1600&auto=format&fit=crop" # Bright restaurant atmosphere

## American/Burger Cuisine - Clean Modern Ambiance
U9 = "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?q=80&w=1600&auto=format&fit=crop" # Clean modern diner
U10 = "https://images.unsplash.com/photo-1533777857889-4be7c70b33f7?q=80&w=1600&auto=format&fit=crop" # Minimal burger restaurant
#U11 = "https://images.unsplash.com/photo-1559329007-40df8dd8534d?q=80&w=1600&auto=format&fit=crop" # Bright modern casual dining interior
U12 = "https://images.unsplash.com/photo-1424847651672-bf20a4b0982b?q=80&w=1600&auto=format&fit=crop" # Clean modern restaurant interior

## Italian Cuisine - Clean Elegant Ambiance  
U13 = "https://images.unsplash.com/photo-1414235077428-338989a2e8c0?q=80&w=1600&auto=format&fit=crop" # Clean fine dining setup
U14 = "https://images.unsplash.com/photo-1540189549336-e6e99c3679fe?q=80&w=1600&auto=format&fit=crop" # Minimalist Italian style
U15 = "https://images.unsplash.com/photo-1578985545062-69928b1d9587?q=80&w=1600&auto=format&fit=crop" # Modern clean restaurant
U16 = "https://images.unsplash.com/photo-1555992336-03a23c7b7887?q=80&w=1600&auto=format&fit=crop" # Clean trattoria style

## Mexican Cuisine - Bright Clean Ambiance
U17 = "https://images.unsplash.com/photo-1578662996442-48f60103fc96?q=80&w=1600&auto=format&fit=crop" # Clean Mexican restaurant
U18 = "https://images.unsplash.com/photo-1585032226651-759b368d7246?q=80&w=1600&auto=format&fit=crop" # Bright modern cantina
U19 = "https://images.unsplash.com/photo-1511690743698-d9d85f2fbf38?q=80&w=1600&auto=format&fit=crop" # Clean colorful setting
U20 = "https://images.unsplash.com/photo-1559627021-1b0a96c1e0cf?q=80&w=1600&auto=format&fit=crop" # Modern Mexican interior

## Asian Cuisine - Clean Modern Ambiance (Chinese/Japanese/Thai/Korean/Vietnamese)
U21 = "https://images.unsplash.com/photo-1537047902294-62a40c20a6ae?q=80&w=1600&auto=format&fit=crop" # Clean Asian restaurant
U22 = "https://images.unsplash.com/photo-1514933651103-005eec06c04b?q=80&w=1600&auto=format&fit=crop" # Minimal Japanese style
U23 = "https://images.unsplash.com/photo-1555742535-4b3dff5c7a66?q=80&w=1600&auto=format&fit=crop" # Clean Asian dining
U24 = "https://images.unsplash.com/photo-1571003123894-1f0594d2b5d9?q=80&w=1600&auto=format&fit=crop" # Modern Asian interior

## BBQ/Grill - Clean Modern Barbecue Ambiance  
U25 = "https://images.unsplash.com/photo-1550966871-3ed3cdb5ed0c?q=80&w=1600&auto=format&fit=crop" # Clean BBQ restaurant
U26 = "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?q=80&w=1600&auto=format&fit=crop" # Modern grill house
U27 = "https://images.unsplash.com/photo-1564069114553-7215e1ff1890?q=80&w=1600&auto=format&fit=crop" # Clean smokehouse style
U28 = "https://images.unsplash.com/photo-1590846406792-0adc7f938f1d?q=80&w=1600&auto=format&fit=crop" # Minimal BBQ interior

## Mediterranean/Greek/Indian - Clean Elegant Ambiance
U29 = "https://images.unsplash.com/photo-1578985545062-69928b1d9587?q=80&w=1600&auto=format&fit=crop" # Clean Mediterranean
U30 = "https://images.unsplash.com/photo-1571003123894-1f0594d2b5d9?q=80&w=1600&auto=format&fit=crop" # Modern ethnic restaurant
U31 = "https://images.unsplash.com/photo-1555742535-4b3dff5c7a66?q=80&w=1600&auto=format&fit=crop" # Clean traditional style
U32 = "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?q=80&w=1600&auto=format&fit=crop" # Bright ethnic dining

## French Cuisine - Elegant Clean Ambiance
U33 = "https://images.unsplash.com/photo-1414235077428-338989a2e8c0?q=80&w=1600&auto=format&fit=crop" # Clean French bistro
U34 = "https://images.unsplash.com/photo-1555396273-367ea4eb4db5?q=80&w=1600&auto=format&fit=crop" # Elegant clean dining
U35 = "https://images.unsplash.com/photo-1590846406792-0adc7f938f1d?q=80&w=1600&auto=format&fit=crop" # Modern French style
U36 = "https://images.unsplash.com/photo-1578985545062-69928b1d9587?q=80&w=1600&auto=format&fit=crop" # Clean fine dining

# Cuisine-specific hero image sets (4 clean images each, perfect for white theme)
HERO_BY_CUISINE = {
    "burger": [U9, U10, U25, U12],          # Clean modern diner/burger vibes
    "italian": [U13, U14, U15, U16],        # Clean Italian elegant dining  
    "mexican": [U17, U18, U19, U20],        # Bright clean Mexican settings
    "bbq": [U25, U26, U27, U28],            # Clean modern BBQ atmosphere
    "american": [U1, U9, U25, U2],          # Classic clean American dining
    "chinese": [U21, U23, U24, U5],         # Clean Asian restaurant ambiance
    "japanese": [U22, U21, U23, U24],       # Minimal Japanese/Asian style
    "thai": [U21, U24, U23, U4],            # Clean Thai/Asian restaurant 
    "indian": [U30, U31, U32, U7],          # Clean Indian restaurant decor
    "greek": [U29, U31, U32, U2],           # Clean Greek/Mediterranean
    "french": [U33, U34, U35, U36],         # Elegant clean French dining
    "korean": [U24, U21, U23, U22],         # Clean modern Korean/Asian
    "mediterranean": [U29, U31, U32, U35],  # Clean Mediterranean vibes
    "vietnamese": [U21, U22, U24, U23],     # Clean Vietnamese/Asian fusion
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

# ────────────────────────────────────────────────────────────────────────────
# Safe Google Image Search + Unsplash helper
# ────────────────────────────────────────────────────────────────────────────
def unsplash(id_or_url: str, w: int = 1600) -> str:
    """
    Returns a usable image URL.
    Supports:
      • 'gq:<query>'              -> to be resolved via Google Image Search before HTML build
      • Full http(s) URL          -> returned as-is
      • 'photo-…' fragment        -> Unsplash CDN (legacy support)
    """
    u = (id_or_url or "").strip()
    if not u:
        return ""
    if u.startswith("http://") or u.startswith("https://"):
        return u
    if u.startswith("photo-"):
        return f"https://images.unsplash.com/{u}?q=80&w={w}&auto=format&fit=crop"
    # 'gq:' handled earlier (pre-resolved); pass token through unchanged for now
    return u

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
    
    # Quick HEAD request to verify image exists
    try:
        resp = await app.state.http.head(url, timeout=3.0)
        if resp.status_code == 200:
            content_type = resp.headers.get('content-type', '').lower()
            if content_type.startswith('image/'):
                return url
    except Exception:
        pass
    
    return ""
  
  
# ────────────────────────────────────────────────────────────────────────────
# Cuisine → curated assets (hero + menu)
# ────────────────────────────────────────────────────────────────────────────
CUISINE_ASSETS: Dict[str, Dict[str, Any]] = {
    "burger": {
        "palette": {"primary":"#FF7C4D","primary_dark":"#D9653D"},
        "hero": HERO_BY_CUISINE["burger"],
        "menu": [
            {"name":"Double Smash Burger","desc":"American cheese, pickles, shack sauce","price":"$10.99","img": "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Crispy Chicken Sandwich","desc":"Buttermilk fried chicken, slaw","price":"$8.99","img": "https://images.unsplash.com/photo-1606755962773-d324e2dabd17?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Crinkle Cut Fries","desc":"Sea salt, extra crispy","price":"$3.99","img": "https://images.unsplash.com/photo-1576107232684-1279f390859f?q=80&w=1200&auto=format&fit=crop"}
        ],
        "fallback": ""
    },
    "italian": {
        "palette": {"primary":"#CF3A2B","primary_dark":"#A82E22"},
        "hero": HERO_BY_CUISINE["italian"],
        "menu": [
            {"name":"Chicken Alfredo","desc":"Creamy parmesan sauce, fettuccine","price":"$14.99","img": "https://images.unsplash.com/photo-1621996346565-e3dbc353d2e5?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Spaghetti & Meatballs","desc":"San Marzano tomatoes, basil","price":"$12.99","img": "https://images.unsplash.com/photo-1551183053-bf91a1d81141?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Margherita Pizza","desc":"Fresh mozzarella, tomato, basil","price":"$11.49","img": "https://images.unsplash.com/photo-1604382354936-07c5d9983bd3?q=80&w=1200&auto=format&fit=crop"}
        ],
        "fallback": ""
    },
    "mexican": {
        "palette": {"primary":"#EA7A28","primary_dark":"#BE621F"},
        "hero": HERO_BY_CUISINE["mexican"],
        "menu": [
            {"name":"Carne Asada Tacos","desc":"Cilantro, onions, lime","price":"$9.49","img": "https://images.unsplash.com/photo-1565299585323-38174c267b34?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Chicken Quesadilla","desc":"Three-cheese blend, pico","price":"$8.99","img": "https://images.unsplash.com/photo-1618040996337-56904b7850b9?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Chips & Guacamole","desc":"House-made","price":"$5.99","img": "https://images.unsplash.com/photo-1541544741938-0af808871cc0?q=80&w=1200&auto=format&fit=crop"}
        ],
        "fallback": ""
    },
    "bbq": {
        "palette": {"primary":"#9B4F2A","primary_dark":"#7B3F22"},
        "hero": HERO_BY_CUISINE["bbq"],
        "menu": [
            {"name":"Smoked Brisket Plate","desc":"Pickles, onions, white bread","price":"$15.99","img": "https://images.unsplash.com/photo-1544025162-d76694265947?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Pork Ribs (Half Rack)","desc":"House rub, sticky glaze","price":"$16.49","img": "https://images.unsplash.com/photo-1558030006-450675393462?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Mac & Cheese","desc":"Three-cheese blend","price":"$5.99","img": "https://images.unsplash.com/photo-1571197119282-7c4e04cc3f2e?q=80&w=1200&auto=format&fit=crop"}
        ],
        "fallback": ""
    },
    "american": {
        "palette": {"primary":"#F28C3A","primary_dark":"#C6732F"},
        "hero": HERO_BY_CUISINE["american"],
        "menu": [
            {"name":"Roast Chicken Plate","desc":"Choice of two sides","price":"$10.99","img": "https://images.unsplash.com/photo-1598515214211-89d3c73ae83b?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Country Fried Steak","desc":"Pepper gravy, mashed potatoes","price":"$11.49","img": "https://images.unsplash.com/photo-1562967916-eb82221dfb92?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Vegetable Plate","desc":"Pick any three sides","price":"$8.99","img": "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?q=80&w=1200&auto=format&fit=crop"}
        ],
        "fallback": ""
    },
    "chinese": {
        "palette": {"primary":"#DC2626","primary_dark":"#B91C1C"},
        "hero": HERO_BY_CUISINE["chinese"],
        "menu": [
            {"name":"General Tso's Chicken","desc":"Sweet and spicy with steamed rice","price":"$12.99","img": "https://images.unsplash.com/photo-1525755662778-989d0524087e?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Beef Lo Mein","desc":"Soft noodles with vegetables","price":"$11.49","img": "https://images.unsplash.com/photo-1582878826629-29b7ad1cdc43?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Pork Dumplings","desc":"Pan-fried, served with soy sauce","price":"$7.99","img": "https://images.unsplash.com/photo-1563379091339-03246963d96c?q=80&w=1200&auto=format&fit=crop"}
        ],
        "fallback": ""
    },
    "japanese": {
        "palette": {"primary":"#7C3AED","primary_dark":"#6D28D9"},
        "hero": HERO_BY_CUISINE["japanese"],
        "menu": [
            {"name":"Salmon Teriyaki","desc":"Grilled with steamed vegetables","price":"$16.99","img": "https://images.unsplash.com/photo-1579871494447-9811cf80d66c?q=80&w=1200&auto=format&fit=crop"},
            {"name":"California Roll","desc":"Crab, avocado, cucumber","price":"$8.99","img": "https://images.unsplash.com/photo-1553621042-f6e147245754?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Chicken Ramen","desc":"Rich broth with soft-boiled egg","price":"$13.49","img": "https://images.unsplash.com/photo-1569718212165-3a8278d5f624?q=80&w=1200&auto=format&fit=crop"}
        ],
        "fallback": ""
    },
    "thai": {
        "palette": {"primary":"#F59E0B","primary_dark":"#D97706"},
        "hero": HERO_BY_CUISINE["thai"],
        "menu": [
            {"name":"Pad Thai","desc":"Rice noodles, shrimp, bean sprouts","price":"$12.99","img": "https://images.unsplash.com/photo-1559847844-5315695dadae?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Green Curry","desc":"Coconut milk, basil, jasmine rice","price":"$13.49","img": "https://images.unsplash.com/photo-1604908176997-125f25cc6f3d?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Tom Yum Soup","desc":"Spicy and sour with prawns","price":"$9.99","img": "https://images.unsplash.com/photo-1596040033229-a9821ebd058d?q=80&w=1200&auto=format&fit=crop"}
        ],
        "fallback": ""
    },
    "indian": {
        "palette": {"primary":"#EF4444","primary_dark":"#DC2626"},
        "hero": HERO_BY_CUISINE["indian"],
        "menu": [
            {"name":"Chicken Tikka Masala","desc":"Creamy tomato sauce, basmati rice","price":"$14.99","img": "https://images.unsplash.com/photo-1585937421612-70a008356fbe?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Lamb Biryani","desc":"Fragrant basmati rice, spices","price":"$16.99","img": "https://images.unsplash.com/photo-1563379091339-03246963d96c?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Garlic Naan","desc":"Fresh baked bread","price":"$3.99","img": "https://images.unsplash.com/photo-1601050690597-df0568f70950?q=80&w=1200&auto=format&fit=crop"}
        ],
        "fallback": ""
    },
    "greek": {
        "palette": {"primary":"#3B82F6","primary_dark":"#2563EB"},
        "hero": HERO_BY_CUISINE["greek"],
        "menu": [
            {"name":"Gyro Platter","desc":"Lamb and beef, tzatziki, pita","price":"$13.99","img": "https://images.unsplash.com/photo-1529692236671-f1f6cf9683ba?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Greek Salad","desc":"Feta, olives, tomatoes, cucumbers","price":"$9.99","img": "https://images.unsplash.com/photo-1540420773420-3366772f4999?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Spanakopita","desc":"Spinach and feta phyllo pastry","price":"$7.99","img": "https://images.unsplash.com/photo-1551218808-94e220e084d2?q=80&w=1200&auto=format&fit=crop"}
        ],
        "fallback": ""
    },
    "french": {
        "palette": {"primary":"#6366F1","primary_dark":"#4F46E5"},
        "hero": HERO_BY_CUISINE["french"],
        "menu": [
            {"name":"Coq au Vin","desc":"Braised chicken in red wine","price":"$18.99","img": "https://images.unsplash.com/photo-1598515214211-89d3c73ae83b?q=80&w=1200&auto=format&fit=crop"},
            {"name":"French Onion Soup","desc":"Gruyere cheese, baguette croutons","price":"$8.99","img": "https://images.unsplash.com/photo-1547592166-23ac45744acd?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Croque Monsieur","desc":"Ham and cheese sandwich, bechamel","price":"$11.99","img": "https://images.unsplash.com/photo-1509440159596-0249088772ff?q=80&w=1200&auto=format&fit=crop"}
        ],
        "fallback": ""
    },
    "korean": {
        "palette": {"primary":"#EF4444","primary_dark":"#DC2626"},
        "hero": HERO_BY_CUISINE["korean"],
        "menu": [
            {"name":"Korean BBQ Bowl","desc":"Bulgogi beef, kimchi, steamed rice","price":"$14.99","img": "https://images.unsplash.com/photo-1498654896293-37aacf113fd9?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Bibimbap","desc":"Mixed vegetables, egg, gochujang","price":"$12.99","img": "https://images.unsplash.com/photo-1582878826629-29b7ad1cdc43?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Korean Fried Chicken","desc":"Crispy wings, sweet chili glaze","price":"$11.99","img": "https://images.unsplash.com/photo-1606755962773-d324e2dabd17?q=80&w=1200&auto=format&fit=crop"}
        ],
        "fallback": ""
    },
    "mediterranean": {
        "palette": {"primary":"#10B981","primary_dark":"#059669"},
        "hero": HERO_BY_CUISINE["mediterranean"],
        "menu": [
            {"name":"Mediterranean Bowl","desc":"Hummus, tabbouleh, grilled chicken","price":"$13.99","img": "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Grilled Sea Bass","desc":"Lemon, herbs, roasted vegetables","price":"$19.99","img": "https://images.unsplash.com/photo-1579871494447-9811cf80d66c?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Falafel Wrap","desc":"Tahini sauce, fresh vegetables","price":"$9.99","img": "https://images.unsplash.com/photo-1529692236671-f1f6cf9683ba?q=80&w=1200&auto=format&fit=crop"}
        ],
        "fallback": ""
    },
    "vietnamese": {
        "palette": {"primary":"#16A34A","primary_dark":"#15803D"},
        "hero": HERO_BY_CUISINE["vietnamese"],
        "menu": [
            {"name":"Pho Bo","desc":"Beef noodle soup, fresh herbs","price":"$11.99","img": "https://images.unsplash.com/photo-1569718212165-3a8278d5f624?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Banh Mi","desc":"Vietnamese sandwich, pickled vegetables","price":"$8.99","img": "https://images.unsplash.com/photo-1509440159596-0249088772ff?q=80&w=1200&auto=format&fit=crop"},
            {"name":"Spring Rolls","desc":"Fresh vegetables, peanut dipping sauce","price":"$6.99","img": "https://images.unsplash.com/photo-1563379091339-03246963d96c?q=80&w=1200&auto=format&fit=crop"}
        ],
        "fallback": ""
    }
}
DEFAULT_CUISINE = "american"

def cuisine_from_types(details: Dict[str, Any]) -> str:
    name = (details.get("name") or "").lower()
    types = [t.lower() for t in (details.get("categories") or [])]
    
    # More comprehensive cuisine detection
    cuisine_keywords = {
        "burger": ["hamburger_restaurant", "fast_food", "burger", "american_restaurant", "sandwich_shop"],
        "italian": ["italian_restaurant", "pizzeria", "pizza_place", "pasta_restaurant"],
        "mexican": ["mexican_restaurant", "taco_restaurant", "tex_mex_restaurant", "burrito_restaurant", "latin_american_restaurant"],
        "bbq": ["barbecue_restaurant", "smokehouse", "grill"],
        "chinese": ["chinese_restaurant", "asian_restaurant"],
        "japanese": ["japanese_restaurant", "sushi_restaurant", "ramen_restaurant"],
        "indian": ["indian_restaurant", "curry_restaurant"],
        "thai": ["thai_restaurant"],
        "mediterranean": ["mediterranean_restaurant", "greek_restaurant", "middle_eastern_restaurant"]
    }
    
    # Check categories first
    for cuisine, keywords in cuisine_keywords.items():
        if any(keyword in types for keyword in keywords):
            return cuisine
    
    # Check name for keywords
    name_keywords = {
        "burger": ["burger", "burgers", "grill", "bbq", "bar-b-que"],
        "italian": ["pizza", "pizzeria", "italian", "pasta", "trattoria", "ristorante"],
        "mexican": ["taco", "tacos", "mexican", "cantina", "casa", "el ", "la "],
        "bbq": ["bbq", "bar-b-que", "barbecue", "smokehouse", "pit"],
        "chinese": ["chinese", "china", "wok", "dragon", "golden", "panda"],
        "japanese": ["sushi", "ramen", "japanese", "tokyo", "sakura"],
        "indian": ["indian", "curry", "tandoor", "masala"],
        "thai": ["thai", "pad", "bangkok"],
        "mediterranean": ["mediterranean", "greek", "gyro", "kebab", "falafel"]
    }
    
    for cuisine, keywords in name_keywords.items():
        if any(keyword in name for keyword in keywords):
            return cuisine
    
    return DEFAULT_CUISINE

async def resolve_menu_images_with_google(details: Dict[str, Any]) -> None:
    cuisine = cuisine_from_types(details)
    assets = CUISINE_ASSETS.get(cuisine, CUISINE_ASSETS[DEFAULT_CUISINE])
    details["_resolved_menu"] = [
        {"name": it["name"], "desc": it["desc"], "price": it["price"], "img": it["img"]}
        for it in assets["menu"][:3]
    ]

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

class PdfOut(BaseModel):
    html: str
    pdf_base64: Optional[str] = None

# ────────────────────────────────────────────────────────────────────────────
# View helpers (logo pick, hours, reviews 5★ only)
# ────────────────────────────────────────────────────────────────────────────
def best_logo(details: Dict[str, Any]) -> Tuple[Optional[str], str]:
    for (u, score, reason) in details.get("logo_debug") or []:
        if "sz=" in u:
            try:
                sz = int(u.split("sz=")[1].split("&")[0])
                if sz < 64:
                    continue
            except Exception:
                pass
        if GENERIC_ICON_PAT.search(u):
            continue
        return u, reason
    cands = details.get("logo_url_candidates") or []
    return (cands[0], "fallback_first") if cands else (None, "none")

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

def _menu_primary_and_fallbacks(item: Dict[str, Any], cuisine_fallback: str) -> Tuple[str, str]:
    primary = None
    fallbacks: List[str] = []
    imgs = item.get("imgs") or []
    if imgs:
        primary = imgs[0]
        fallbacks = imgs[1:]
    else:
        primary = item.get("img") or cuisine_fallback
    if cuisine_fallback and (not fallbacks or fallbacks[-1] != cuisine_fallback):
        fallbacks.append(cuisine_fallback)
    return primary, "|".join(fallbacks)

# ────────────────────────────────────────────────────────────────────────────
# HTML builder
# ────────────────────────────────────────────────────────────────────────────
def build_html(details: Dict[str, Any], *, sales_cta: bool) -> Tuple[str, Dict[str, Any]]:
    name = details.get("name") or "Restaurant"
    address = details.get("address") or ""
    website = details.get("website") or ""
    phone = details.get("phone") or ""
    rating = details.get("rating")
    review_count = details.get("review_count")
    map_url = details.get("map_url") or "#"

    cuisine = cuisine_from_types(details)
    assets = CUISINE_ASSETS.get(cuisine, CUISINE_ASSETS[DEFAULT_CUISINE])
    pal = assets["palette"]
    
    # Hero images are direct URLs now
    hero_imgs: List[str] = list((assets.get("hero") or HERO_BY_CUISINE.get(cuisine) or HERO_BY_CUISINE["american"])[:4])

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

    logo, logo_reason = best_logo(details)
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

    log.info("BUILD PAGE: name=%s cuisine=%s hero=%d menu=%d gallery=%d",
             name, cuisine, len(hero_imgs), len(menu_items), len(gallery))

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
        "map_url": map_url
    }
    return html, meta

# ────────────────────────────────────────────────────────────────────────────
# Endpoints
# ────────────────────────────────────────────────────────────────────────────
@app.get("/healthz", summary="Liveness probe")
async def healthz():
    return {"ok": True, "ts": time.time()}

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
                chain_count = await google_nearby_chain_count(r.get("name") or "", r.get("lat"), r.get("lng"), r.get("id"), CHAIN_RADIUS_M)
                r["chain_count_nearby"] = chain_count
                r["is_chain_nearby"] = bool(chain_count and chain_count >= 1)
            DETAILS_CACHE.set(key, payload)
            return payload
        except HTTPException:
            if not YELP_API_KEY:
                raise

    if provider == "yelp":
        if not YELP_API_KEY:
            raise HTTPException(404, "Yelp fallback disabled or missing YELP_API_KEY")
        y = await yelp_business_details(id)
        result = normalize_details_yelp(y)
        if include_chain_info and result.get("lat") is not None and result.get("lng") is not None and result.get("name"):
            try:
                fake_self_id = "yelp-" + (result.get("id") or "")
                chain_count = await google_nearby_chain_count(result["name"], result["lat"], result["lng"], fake_self_id, CHAIN_RADIUS_M)
                result["chain_count_nearby"] = chain_count
                result["is_chain_nearby"] = bool(chain_count and chain_count >= 1)
            except Exception:
                pass
        payload = {"result": result}
        DETAILS_CACHE.set(key, payload)
        return payload

    raise HTTPException(400, "Invalid provider")

@app.post("/generate/template", response_model=TemplateOut, summary="Generate a premium HTML landing page from details")
async def generate_template(
    payload: GeneratePayload,
    request: Request,
    _: None = Depends(rate_limit),
    __: None = Depends(security_guard),
):
    """
    Builds the premium HTML landing page.
    Also resolves 'gq:' menu images via Google CSE first.
    """
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
                    details.get("lat"),
                    details.get("lng"),
                    details.get("id"),
                    CHAIN_RADIUS_M
                )
            except Exception:
                chain_count = 0
            details["chain_count_nearby"] = chain_count
            details["is_chain_nearby"] = bool(chain_count and chain_count >= 1)
        else:
            y = await yelp_business_details(payload.place_id)
            details = normalize_details_yelp(y)

    try:
        await resolve_menu_images_with_google(details)
    except Exception as e:
        log.warning("resolve_menu_images_with_google failed: %r", e)

    html, meta = build_html(details, sales_cta=payload.sales_cta)
    return TemplateOut(html=html, react=None, meta=meta)

# ────────────────────────────────────────────────────────────────────────────
# PDF generation
# ────────────────────────────────────────────────────────────────────────────
def html_to_pdf_b64(html: str) -> Optional[str]:
    try:
        from weasyprint import HTML
        pdf_bytes = HTML(string=html).write_pdf()
        return base64.b64encode(pdf_bytes).decode("ascii")
    except Exception as e:
        log.warning("PDF generation skipped: %r", e)
        return None

@app.post("/generate/pdf", response_model=PdfOut, summary="Generate a PDF brand pack from the HTML")
async def generate_pdf(
    payload: GeneratePayload,
    request: Request,
    _: None = Depends(rate_limit),
    __: None = Depends(security_guard),
):
    tpl: TemplateOut = await generate_template(payload, request)  # type: ignore
    pdf_b64 = html_to_pdf_b64(tpl.html or "")
    return PdfOut(html=tpl.html or "", pdf_base64=pdf_b64)

# ────────────────────────────────────────────────────────────────────────────
# Hardened image proxy (SSRF-safe + size/type checks)
# ────────────────────────────────────────────────────────────────────────────
ALLOWED_IMG_HOSTS = {
    "images.unsplash.com",
    "maps.googleapis.com",
    "lh3.googleusercontent.com",
    "i.imgur.com", "imgur.com",
    "assets.simpleviewinc.com",
    # add any others you rely on
}
MAX_IMAGE_BYTES = 6_000_000  # ~6 MB

def _is_private_host(u: str) -> bool:
    try:
        p = urlparse(u)
        if p.scheme not in {"http", "https"}:
            return True
        host = p.hostname or ""
        infos = socket.getaddrinfo(host, None)
        for _family, _t, _p, _cn, sockaddr in infos:
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return True
        return False
    except Exception:
        return True

@app.get("/imgproxy", summary="Fetch whitelisted images with SSRF protection")
async def imgproxy(u: str, _: None = Depends(rate_limit)):
    u = _clean_url(u)
    if not u:
        raise HTTPException(400, "Bad URL")

    if _is_private_host(u):
        raise HTTPException(403, "Blocked host")

    p = urlparse(u)
    host = (p.hostname or "").lower()
    if host not in ALLOWED_IMG_HOSTS:
        raise HTTPException(403, "Host not allowed")

    # HEAD first (size/type)
    try:
        head = await app.state.http.head(u, follow_redirects=True, timeout=5.0)
    except Exception as e:
        raise HTTPException(404, f"Fetch failed: {e}")

    ctype = (head.headers.get("content-type") or "").lower()
    if not ctype.startswith("image/"):
        raise HTTPException(415, "Not an image")

    clen = int(head.headers.get("content-length") or "0")
    if clen and clen > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image too large")

    # GET
    r = await app.state.http.get(u, follow_redirects=True, timeout=7.0)
    if r.status_code == 200 and (r.headers.get("content-type") or "").lower().startswith("image/"):
        content = r.content if len(r.content) <= MAX_IMAGE_BYTES else r.content[:MAX_IMAGE_BYTES]
        return StreamingResponse(io.BytesIO(content), media_type=r.headers["content-type"])

    raise HTTPException(404, "Image not available")