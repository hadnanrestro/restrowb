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

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
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

# ────────────────────────────────────────────────────────────────────────────
# Cuisine → curated assets (hero + menu)
# ────────────────────────────────────────────────────────────────────────────
CUISINE_ASSETS: Dict[str, Dict[str, Any]] = {
    "burger": {
        "palette": {"primary":"#FF7C4D","primary_dark":"#D9653D"},
        "hero": [
            unsplash("photo-1551183053-bf91a1d81141"),
            unsplash("photo-1551782450-17144c3a380e"),
            unsplash("photo-1504754524776-8f4f37790ca0"),
            unsplash("photo-1550547660-d9450f859349"),
        ],
        "menu": [
            {"name":"Double Smash Burger","desc":"American cheese, pickles, shack sauce","price":"$10.99","img": "gq:double smash burger"},
            {"name":"Crispy Chicken Sandwich","desc":"Buttermilk fried chicken, slaw","price":"$8.99","img": "gq:crispy chicken sandwich"},
            {"name":"Crinkle Cut Fries","desc":"Sea salt, extra crispy","price":"$3.99","img": "gq:crinkle cut fries"},
        ],
        "fallback": "",
    },
    "italian": {
        "palette": {"primary":"#CF3A2B","primary_dark":"#A82E22"},
        "hero": [
            unsplash("photo-1525755662778-989d0524087e"),
            unsplash("photo-1513104890138-7c749659a591"),
            unsplash("photo-1523986371872-9d3ba2e2f5fd"),
            unsplash("photo-1540189549336-e6e99c3679fe"),
        ],
        "menu": [
            {"name":"Chicken Alfredo","desc":"Creamy parmesan sauce, fettuccine","price":"$14.99","img": "gq:chicken alfredo pasta"},
            {"name":"Spaghetti & Meatballs","desc":"San Marzano tomatoes, basil","price":"$12.99","img": "gq:spaghetti and meatballs"},
            {"name":"Margherita Pizza","desc":"Fresh mozzarella, tomato, basil","price":"$11.49","img": "gq:margherita pizza"},
        ],
        "fallback": "",
    },
    "mexican": {
        "palette": {"primary":"#EA7A28","primary_dark":"#BE621F"},
        "hero": [
            unsplash("photo-1543352634-8732c721c8e7"),
            unsplash("photo-1552332386-f8dd00dc2f85"),
            unsplash("photo-1590080876475-34c4f3f2d7b7"),
            unsplash("photo-1617195737498-7f5d8dfe8f3f"),
        ],
        "menu": [
            {"name":"Carne Asada Tacos","desc":"Cilantro, onions, lime","price":"$9.49","img": "gq:carne asada tacos"},
            {"name":"Chicken Quesadilla","desc":"Three-cheese blend, pico","price":"$8.99","img": "gq:chicken quesadilla"},
            {"name":"Chips & Guacamole","desc":"House-made","price":"$5.99","img": "gq:chips and guacamole"},
        ],
        "fallback": "",
    },
    "bbq": {
        "palette": {"primary":"#9B4F2A","primary_dark":"#7B3F22"},
        "hero": [
            unsplash("photo-1558030006-450675c69ddf"),
            unsplash("photo-1544025162-d76694265947"),
            unsplash("photo-1514517220033-e6d36f4a0a76"),
            unsplash("photo-1523987355523-c7b5b92adce3"),
        ],
        "menu": [
            {"name":"Smoked Brisket Plate","desc":"Pickles, onions, white bread","price":"$15.99","img": "gq:texas smoked brisket plate"},
            {"name":"Pork Ribs (Half Rack)","desc":"House rub, sticky glaze","price":"$16.49","img": "gq:pork ribs barbecue"},
            {"name":"Mac & Cheese","desc":"Three-cheese blend","price":"$5.99","img": "gq:mac and cheese bowl"},
        ],
        "fallback": "",
    },
    "american": {
        "palette": {"primary":"#F28C3A","primary_dark":"#C6732F"},
        "hero": [
            unsplash("photo-1551183053-bf91a1d81141"),
            unsplash("photo-1553621042-f6e147245754"),
            unsplash("photo-1504754524776-8f4f37790ca0"),
            unsplash("photo-1550547660-d9450f859349"),
        ],
        "menu": [
            {"name":"Roast Chicken Plate","desc":"Choice of two sides","price":"$10.99","img": "gq:roast chicken plate"},
            {"name":"Country Fried Steak","desc":"Pepper gravy, mashed potatoes","price":"$11.49","img": "gq:country fried steak with gravy"},
            {"name":"Vegetable Plate","desc":"Pick any three sides","price":"$8.99","img": "gq:vegetable plate southern sides"},
        ],
        "fallback": "",
    },
}
DEFAULT_CUISINE = "american"

def cuisine_from_types(details: Dict[str, Any]) -> str:
    name = (details.get("name") or "").lower()
    types = [t.lower() for t in (details.get("categories") or [])]
    if any(t in types for t in ["hamburger_restaurant","fast_food","burger","american_restaurant"]): return "burger"
    if any(t in types for t in ["italian_restaurant","pizzeria"]): return "italian"
    if any(t in types for t in ["mexican_restaurant","taco_restaurant"]): return "mexican"
    if any(t in types for t in ["barbecue_restaurant"]): return "bbq"
    if "pizza" in name: return "italian"
    if "burger" in name: return "burger"
    if "bbq" in name or "bar-b-que" in name: return "bbq"
    if "taco" in name: return "mexican"
    return DEFAULT_CUISINE

async def resolve_menu_images_with_google(details: Dict[str, Any]) -> None:
    cuisine = cuisine_from_types(details)
    assets = CUISINE_ASSETS.get(cuisine, CUISINE_ASSETS[DEFAULT_CUISINE])
    resolved_menu = []
    for item in assets["menu"]:
        img = item.get("img") or ""
        if img.startswith("gq:"):
            query = img.split(":", 1)[1].strip()
            q = f"{query} {details.get('name') or ''} {cuisine} restaurant"
            found = await google_image_search(q) or await google_image_search(query)
            resolved_img = found or ""
        else:
            resolved_img = unsplash(img)
        resolved_menu.append({
            "name": item["name"],
            "desc": item["desc"],
            "price": item["price"],
            "img": resolved_img,
        })
    details["_resolved_menu"] = resolved_menu

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

# ────────────────────────────────────────────────────────────────────────────
# Yelp fallback
# ────────────────────────────────────────────────────────────────────────────
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
    hero_imgs: List[str] = list(assets["hero"])[:4]

    raw_menu_items: List[Dict[str, str]] = list((details.get("_resolved_menu") or [])[:3])
    menu_items: List[Dict[str, str]] = []
    for it in raw_menu_items:
        menu_items.append({
            "name": it["name"],
            "desc": it["desc"],
            "price": it["price"],
            "img": it["img"],       # resolved URL
            "fallbacks": "",
        })

    logo, logo_reason = best_logo(details)
    revs = five_star_only(details)
    show_reviews = len(revs) >= 2
    revs = revs[:4]
    gallery = details.get("gallery") or []
    hrs = hours_list(details)

    log.info("BUILD PAGE: name=%s cuisine=%s hero=%d menu=%d gallery=%d",
             name, cuisine, len(hero_imgs), len(menu_items), len(gallery))

    # Note: fixed URL encoding for /imgproxy image param.
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
  .fade-wrap{{position:relative;height:56vh;max-height:620px;overflow:hidden;border-radius:1.5rem}}
  .fade-wrap img{{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;opacity:0;transition:opacity 900ms ease-in-out}}
  .fade-wrap img.active{{opacity:1}}
  .dot{{width:8px;height:8px;border-radius:9999px;background:#0003}}
  .dot.active{{background:{pal['primary']};}}
  .card{{background:#FFF;border-radius:1.25rem;box-shadow:var(--tw-shadow, 0 10px 30px rgba(0,0,0,.10));}}
</style>
<meta http-equiv="Content-Security-Policy"
  content="default-src 'self'; img-src 'self' data: https:; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com; connect-src 'self'">
<meta name="referrer" content="no-referrer">
</head>
<body class="font-body">

  <header class="sticky top-0 z-50">
    <nav class="mx-auto max-w-[1200px] px-4 md:px-6 lg:px-8 py-3 flex items-center justify-between bg-white/80 backdrop-blur border-b border-black/5">
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
    <div class="mx-auto max-w-[1200px] px-4 md:px-6 lg:px-8">
      <div class="fade-wrap shadow-soft" id="heroWrap">
        {"".join([f"<img src='{safe(u)}' data-fallbacks='{safe(assets['fallback'])}' alt='hero image {i+1}' {'class=\"active\"' if i==0 else ''} onerror='__imgSwap(this)' />" for i,u in enumerate(hero_imgs)])}
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
    <div class="mx-auto max-w-[1200px] px-4 md:px-6 lg:px-8">
      <div class="flex items-end justify-between gap-4">
        <h2 class="font-display text-3xl md:text-4xl">Menu Highlights</h2>
        <a href="{safe(website or map_url)}" target="_blank" rel="noopener" class="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-brand text-white font-semibold hover:bg-brandd transition shadow-soft">Order Online</a>
      </div>
      <div class="mt-6 grid md:grid-cols-2 lg:grid-cols-3 gap-7">
        {"".join([f"""
        <article class="card overflow-hidden">
          <div class="aspect-[4/3] w-full overflow-hidden">
            <img class="w-full h-full object-cover"
                 src="/imgproxy?u={quote(item['img'] or '', safe='')}"
                 alt="{safe(item['name'])}"
                 loading="lazy"/>
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
    <div class="mx-auto max-w-[1200px] px-4 md:px-6 lg:px-8 grid md:grid-cols-2 gap-7">
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
    <div class="mx-auto max-w-[1200px] px-4 md:px-6 lg:px-8">
      <h2 class="font-display text-3xl md:text-4xl">Gallery</h2>
      <div class="mt-6 grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {"".join([f"<a href='{safe(u)}' target='_blank' rel='noopener' class='block rounded-2xl overflow-hidden card'><img src='{safe(u)}' data-fallbacks='{safe(assets['fallback'])}' onerror='__imgSwap(this)' class='w-full h-48 object-cover'/></a>" for u in gallery[:4]])}
      </div>
    </div>
  </section>

  {""
  if not show_reviews else
  f'''
  <section id="reviews" class="mt-12 md:mt-16">
    <div class="mx-auto max-w-[1200px] px-4 md:px-6 lg:px-8">
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
    <div class="mx-auto max-w-[1200px] px-4 md:px-6 lg:px-8">
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
async def imgproxy(u: str, _: None = Depends(rate_limit), __: None = Depends(security_guard)):
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