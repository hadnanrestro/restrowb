# app.py
import os
import re
import time
import hashlib
from urllib.parse import urlparse
from typing import Optional, Literal, Dict, Any, List, Tuple

from fastapi import FastAPI, HTTPException, Query, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import httpx

# ────────────────────────────────────────────────────────────────────────────
# Env & Config
# ────────────────────────────────────────────────────────────────────────────
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")
YELP_API_KEY   = os.getenv("YELP_API_KEY", "")  # optional fallback for suggest/details gaps

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
BACKEND_API_KEY = os.getenv("BACKEND_API_KEY")  # optional shared secret

RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
CACHE_TTL_SUGGEST  = int(os.getenv("CACHE_TTL_SUGGEST", "180"))
CACHE_TTL_DETAILS  = int(os.getenv("CACHE_TTL_DETAILS", "3600"))
DEFAULT_RADIUS_M   = int(os.getenv("DEFAULT_RADIUS_M", "50000"))  # suggest bias
CHAIN_RADIUS_M     = int(os.getenv("CHAIN_RADIUS_M", "50000"))    # chain detection radius

app = FastAPI(title="Restaurant Search (Google-first with Yelp fallback)", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────────────────────────────────────────────────────────
# Utilities: TTL cache, rate limit, security, HTTP
# ────────────────────────────────────────────────────────────────────────────
class TTLCache:
    def __init__(self, ttl_seconds: int = 300, max_items: int = 5000):
        self.ttl = ttl_seconds
        self.max_items = max_items
        self.store: Dict[str, Tuple[Any, float]] = {}

    def _now(self) -> float:
        return time.time()

    def get(self, key: str):
        item = self.store.get(key)
        if not item:
            return None
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

_ip_hits: Dict[str, List[float]] = {}

def rate_limit(request: Request):
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    window_start = now - 60.0
    bucket = _ip_hits.setdefault(ip, [])
    while bucket and bucket[0] < window_start:
        bucket.pop(0)
    if len(bucket) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(429, "Rate limit exceeded. Try again shortly.")
    bucket.append(now)

def security_guard(request: Request):
    if BACKEND_API_KEY:
        sent = request.headers.get("x-api-key") or request.headers.get("X-API-Key")
        if sent != BACKEND_API_KEY:
            raise HTTPException(401, "Invalid or missing X-API-Key")

async def http_get_json(url: str, *, params: Dict[str, Any] = None, headers: Dict[str, str] = None, timeout: float = 8.0):
    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        resp = await client.get(url, params=params)
        if resp.status_code != 200:
            txt = resp.text[:400]
            raise HTTPException(502, f"Upstream error {resp.status_code}: {txt}")
        return resp.json()

def cache_key(*parts: Any) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()

# ────────────────────────────────────────────────────────────────────────────
# URL + branding helpers
# ────────────────────────────────────────────────────────────────────────────
MARKETPLACE_HOST_PAT = re.compile(
    r"(yelp|doordash|ubereats|grubhub|postmates|seamless|toasttab|opentable|squareup|clover|foodhub|facebook|linktr\.ee|instagram|tiktok|google\.[^/]+)",
    re.I
)

def _clean_url(u: Optional[str]) -> Optional[str]:
    if not u:
        return None
    try:
        u = u.strip()
        parsed = urlparse(u if "://" in u else "https://" + u)
        return parsed.geturl()
    except Exception:
        return None

def _homepage(u: Optional[str]) -> Optional[str]:
    """Return canonical homepage (scheme://host/), or None if not parseable."""
    u = _clean_url(u)
    if not u:
        return None
    p = urlparse(u)
    if not p.scheme or not p.netloc:
        return None
    return f"{p.scheme}://{p.netloc}/"

def _is_marketplace(u: Optional[str]) -> bool:
    if not u:
        return False
    try:
        host = urlparse(u).netloc.lower()
    except Exception:
        return False
    return bool(MARKETPLACE_HOST_PAT.search(host))

def build_google_photo_url(photo_ref: str, maxwidth: int = 1000) -> str:
    return (
        "https://maps.googleapis.com/maps/api/place/photo"
        f"?maxwidth={maxwidth}"
        f"&photoreference={photo_ref}"
        f"&key={GOOGLE_API_KEY}"
    )

def guess_favicon_urls(homepage: Optional[str]) -> List[str]:
    """Return candidates for favicon: site favicon.ico and Google's s2 proxy."""
    hp = _homepage(homepage) if homepage else None
    out = []
    if hp:
        out.append(hp + "favicon.ico")
        # s2 favicons proxy (tiny but reliable)
        host = urlparse(hp).netloc
        out.append(f"https://www.google.com/s2/favicons?domain={host}")
        out.append(f"https://www.google.com/s2/favicons?sz=128&domain={host}")
    return out

# ────────────────────────────────────────────────────────────────────────────
# Google: Autocomplete, Details, Nearby/Chain count
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

    # Only keep a real website; do NOT treat maps/google/social as website
    site = _homepage(g.get("website")) if g.get("website") else None
    if site and _is_marketplace(site):
        site = None

    # cover photo (first place photo)
    cover_photo_url = None
    photos = g.get("photos") or []
    if photos:
        pref = photos[0].get("photo_reference")
        if pref:
            cover_photo_url = build_google_photo_url(pref, maxwidth=1600)

    # hours
    opening = g.get("opening_hours") or {}
    open_now = opening.get("open_now")

    return {
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
        "categories": g.get("types"),
        "hours_text": opening.get("weekday_text"),
        "open_now": open_now,
        "price_level": g.get("price_level"),
        "business_status": g.get("business_status"),
        "cover_photo_url": cover_photo_url,
        "logo_url_candidates": guess_favicon_urls(site),
        "provider": "google",
        "raw": g,  # keep raw for debugging
    }

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
        "photos"
    ])
    params = {"place_id": place_id, "fields": fields, "key": GOOGLE_API_KEY}
    data = await http_get_json("https://maps.googleapis.com/maps/api/place/details/json", params=params)
    status = data.get("status")
    if status != "OK":
        raise HTTPException(502, f"Google error: {status} - {data.get('error_message')}")
    return data

async def google_nearby_chain_count(name: str, lat: Optional[float], lng: Optional[float], self_place_id: str, radius_m: int) -> int:
    """Heuristic: count other places with same (case-insensitive) name within radius."""
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
        if pid == self_place_id:
            continue
        # basic string match; you can improve with token cleanup
        if rname == target:
            count += 1
    return count

# ────────────────────────────────────────────────────────────────────────────
# Yelp fallback (optional)
# ────────────────────────────────────────────────────────────────────────────
def normalize_suggest_yelp(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for b in payload.get("businesses", []):
        loc = b.get("location") or {}
        addr = ", ".join(filter(None, [loc.get("address1"), loc.get("city"), loc.get("state")]))
        out.append({
            "id": b.get("id"),
            "title": b.get("name"),
            "subtitle": addr,
            "provider": "yelp",
        })
    return out

def normalize_details_yelp(b: Dict[str, Any]) -> Dict[str, Any]:
    loc = b.get("location") or {}
    coords = b.get("coordinates") or {}
    # Yelp usually has a hero image (not logo) — can serve as cover fallback
    cover = b.get("image_url") or None
    addr = ", ".join(filter(None, [loc.get("address1"), loc.get("city"), loc.get("state"), loc.get("zip_code")]))

    # Yelp rarely exposes official site; we avoid treating yelp.com as "website"
    site = None
    # Sometimes attributes.menu_url points to first-party site — keep as homepage if so
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
        "website": homepage,                   # may be None
        "map_url": b.get("url"),               # Yelp page
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
# Endpoints
# ────────────────────────────────────────────────────────────────────────────
@app.get("/suggest", summary="Typeahead suggestions (Google-first; Yelp fallback if empty)")
async def suggest(
    request: Request,
    q: str = Query(..., min_length=1),
    lat: Optional[float] = Query(None, description="Optional latitude for local bias"),
    lng: Optional[float] = Query(None, description="Optional longitude for local bias"),
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

    # Google first
    gdata = await google_autocomplete(q, lat=lat, lng=lng, radius_m=radius_m, country=country)
    g_results = normalize_suggest_google((gdata or {}).get("predictions", [])[:limit])

    # Yelp fallback if Google empty and Yelp enabled
    if not g_results and YELP_API_KEY:
        ydata = await yelp_autocomplete(q, lat=lat, lng=lng, limit=limit)
        y_results = normalize_suggest_yelp(ydata)
        result = {"results": y_results}
        SUGGEST_CACHE.set(key, result)
        return result

    result = {"results": g_results}
    SUGGEST_CACHE.set(key, result)
    return result

@app.get("/details", summary="Get business details (Google; Yelp fallback if id is Yelp or Google fails)")
async def details(
    request: Request,
    id: str = Query(..., description="Google place_id or Yelp business id"),
    provider: Literal["google", "yelp"] = Query("google"),
    include_chain_info: bool = Query(True),
    _: None = Depends(rate_limit),
    __: None = Depends(security_guard),
):
    """
    - If provider=google (default), fetch Google Place Details, enrich with cover photo, favicon candidates, and chain count.
    - If Google fails OR provider=yelp, use Yelp details (when YELP_API_KEY is set).
    """
    key = cache_key("details", provider, id, int(include_chain_info))
    cached = DETAILS_CACHE.get(key)
    if cached is not None:
        return cached

    if provider == "google":
        try:
            g = await google_details(id)
            payload = {"result": normalize_details_google(g)}
            # Chain detection
            if include_chain_info:
                r = payload["result"]
                chain_count = await google_nearby_chain_count(r.get("name") or "", r.get("lat"), r.get("lng"), r.get("id"), CHAIN_RADIUS_M)
                r["chain_count_nearby"] = chain_count
                r["is_chain_nearby"] = bool(chain_count and chain_count >= 1)
            DETAILS_CACHE.set(key, payload)
            return payload
        except HTTPException as e:
            # fall through to Yelp only if we explicitly allow Yelp fallback
            if not YELP_API_KEY:
                raise

    # Yelp fallback (explicit or due to Google failure)
    if provider == "yelp":
        if not YELP_API_KEY:
            raise HTTPException(404, "Yelp fallback disabled or missing YELP_API_KEY")
        y = await yelp_business_details(id)
        result = normalize_details_yelp(y)
        # Best-effort chain detection via Google using Yelp lat/lng + Yelp name
        if include_chain_info and result.get("lat") is not None and result.get("lng") is not None and result.get("name"):
            try:
                # Query Google Nearby using Yelp name
                fake_self_id = "yelp-" + (result.get("id") or "")
                chain_count = await google_nearby_chain_count(result["name"], result["lat"], result["lng"], fake_self_id, CHAIN_RADIUS_M)
                result["chain_count_nearby"] = chain_count
                result["is_chain_nearby"] = bool(chain_count and chain_count >= 1)
            except Exception:
                pass
        payload = {"result": result}
        DETAILS_CACHE.set(key, payload)
        return payload

    # If provider not recognized
    raise HTTPException(400, "Invalid provider")