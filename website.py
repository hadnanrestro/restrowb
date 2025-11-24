import sys
import time
import base64
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

log = logging.getLogger("uvicorn.error")

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"


def _asset_data_uri(filename: str) -> str:
    path = ASSETS_DIR / filename
    try:
        data = path.read_bytes()
        mime = "image/png" if path.suffix.lower() != ".svg" else "image/svg+xml"
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    except Exception as exc:
        log.warning("website assets: failed to load %s (%s)", filename, exc)
        return ""


def _font_data_uri(path: Path) -> str:
    try:
        data = path.read_bytes()
        return "data:font/ttf;base64," + base64.b64encode(data).decode("ascii")
    except Exception as exc:
        log.warning("website font: failed to load %s (%s)", path, exc)
        return ""


INTER_FONT_PATH = BASE_DIR / "Inter" / "Inter-VariableFont_opsz,wght.ttf"
INTER_FONT_DATA = _font_data_uri(INTER_FONT_PATH)
WEBSITE_BOTTOM_SVG = _asset_data_uri("websitebottom.png")




SERVICE_SHOWCASE = [
    {"title": "Catering", "img": _asset_data_uri("Catering.png")},
    {"title": "Group Dining", "img": _asset_data_uri("group dining.png")},
    {"title": "Food Service", "img": _asset_data_uri("food service.png")},
]


def _city_from_details(address: str, details: Dict[str, Any]) -> str:
    maybe_city = (details.get("city") or details.get("locality") or "").strip()
    if maybe_city:
        return maybe_city
    parts = [p.strip() for p in (address or "").split(",")]
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    if parts and parts[0]:
        return parts[0]
    return "Town"

# HTML builder
async def build_html(details: Dict[str, Any], *, sales_cta: bool) -> Tuple[str, Dict[str, Any]]:
    # Resolve helpers from app.py at runtime to avoid circular imports
    _app = sys.modules.get("app")
    if _app is None:
        import app as _app  # fallback import if not present in sys.modules

    safe = getattr(_app, "safe")
    cuisine_from_types = getattr(_app, "cuisine_from_types")
    get_restaurant_context = getattr(_app, "get_restaurant_context")
    best_logo_with_color = getattr(_app, "best_logo_with_color")
    select_theme_colors = getattr(_app, "select_theme_colors")
    get_template_hero_images = getattr(_app, "get_template_hero_images")
    resolve_menu_images = getattr(_app, "resolve_menu_images")
    try_enrich_menu_from_site = getattr(_app, "try_enrich_menu_from_site")
    five_star_only = getattr(_app, "five_star_only")
    hours_list = getattr(_app, "hours_list")
    CONTAINER = getattr(_app, "CONTAINER")
    HERO_FALLBACK_URL = getattr(_app, "HERO_FALLBACK_URL")
    CUISINE_ASSETS = getattr(_app, "CUISINE_ASSETS")
    DEFAULT_CUISINE = getattr(_app, "DEFAULT_CUISINE")
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
    
    # Website hero images should always use curated template assets.
    hero_imgs: List[str] = get_template_hero_images(cuisine)
    if not hero_imgs:
        hero_imgs = [HERO_FALLBACK_URL]

    # Build menu the same way as mobile: first enrich from site, then resolve images
    try:
        await try_enrich_menu_from_site(details, cuisine)
    except Exception:
        pass
    await resolve_menu_images(details, cuisine)
    raw_menu_items: List[Dict[str, str]] = list((details.get("_resolved_menu") or [])[:3])
    menu_items: List[Dict[str, str]] = [
        {
            "name": it.get("name", ""),
            "desc": it.get("desc", ""),
            "price": it.get("price", ""),
            "img": it.get("img", ""),  # already a fully qualified URL from resolver
        }
        for it in raw_menu_items
    ]

    hrs = hours_list(details)

    def _nice_join(values: List[str]) -> str:
        cleaned = [v for v in values if v]
        if not cleaned:
            return ""
        if len(cleaned) == 1:
            return cleaned[0]
        if len(cleaned) == 2:
            return f"{cleaned[0]} and {cleaned[1]}"
        return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"

    signature_copy = _nice_join([it["name"] for it in menu_items][:3])
    if signature_copy:
        signature_copy = f"Guests rave about {signature_copy}."
    else:
        signature_copy = "Guests love our chef-crafted specials and seasonal tastings."

    if context.get("is_upscale"):
        atmosphere_copy = "Soft lighting and polished service make it ideal for celebrations and date nights."
    elif context.get("is_fast_food"):
        atmosphere_copy = "Fast-casual energy with quick counter service and comfy seating for laid-back hangs."
    else:
        atmosphere_copy = "Relaxed dining room that feels warm, welcoming, and perfect for gatherings."

    today_name = time.strftime("%A")
    today_hours = None
    for day, hours in hrs or []:
        if day.lower().startswith(today_name[:3].lower()):
            today_hours = hours
            break

    if details.get("open_now"):
        service_tip = "Open now — swing by or order ahead."
    elif today_hours:
        service_tip = f"{today_name}'s hours: {today_hours}."
    else:
        service_tip = "Order ahead for pickup or plan a relaxed dine-in visit."

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
    fallback_foto = assets.get("fallback") or HERO_FALLBACK_URL

    cuisine_display = (cuisine or DEFAULT_CUISINE).replace("_", " ").title()
    if "food" not in cuisine_display.lower():
        cuisine_phrase = f"{cuisine_display} Food"
    else:
        cuisine_phrase = cuisine_display
    city_display = _city_from_details(address, details).title()
    hero_title = f"Best {cuisine_phrase} in {city_display}"
    hero_subline = "Get $10 off on your first online order."
    nearest_location = address or "Find a Restronaut location near you"
    search_placeholder = "Search food items and offers"
    service_cards_html = "".join([
        f"""
        <article class="service-card" style="background-image:url('{safe(card['img']) or HERO_FALLBACK_URL}');">
          <div class="service-card__scrim"></div>
          <div class="service-card__content">
            <h3>{safe(card['title'])}</h3>
            <p>Your brand settings will appear consistently across all of your stores in Restronaut.</p>
          </div>
        </article>
        """
        for card in SERVICE_SHOWCASE
    ])
    inter_face_css = ""
    if INTER_FONT_DATA:
        inter_face_css = (
            "@font-face{font-family:'InterLocal';src:url("
            f"{INTER_FONT_DATA}"
            ") format('truetype');font-weight:100 900;font-style:normal;font-display:swap;}"
        )

    # Continue with HTML generation using the dynamic palette...
    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"UTF-8\"/>
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\"/>
<title>{safe(name)}</title>
<link rel=\"icon\" href=\"{safe(logo or '')}\"/>
<link href=\"https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600&display=swap\" rel=\"stylesheet\"/>
<script src=\"https://cdn.tailwindcss.com\"></script>
<script>
tailwind.config = {{
  theme: {{
    extend: {{
      colors: {{
        brand: \"{pal['primary']}\",
        brandd: \"{pal['primary_dark']}\"
      }},
      fontFamily: {{
        display: ['InterLocal','Inter','system-ui','-apple-system','Segoe UI','sans-serif'],
        body: ['InterLocal','Inter','system-ui','-apple-system','Segoe UI','sans-serif']
      }}
    }}
  }}
}}
</script>
<style>
  {inter_face_css}
  html,body{{background:#F5F2ED;color:#111;margin:0;min-height:100%;}}
  header, .address-bar, nav {{
    padding-left: 0 !important;
    padding-right: 0 !important;
  }}
  body{{font-family:'InterLocal','Inter','system-ui',sans-serif;}}`
  .container-shell{{width:clamp(320px,min(100%,1480px),1480px);margin-inline:auto;padding-inline:clamp(1.25rem,4vw,3.5rem);}}
  .address-bar{{background:linear-gradient(120deg,#fff6ee,#f5eee7);color:#0f172a;font-size:.9rem;border-bottom:1px solid rgba(15,23,42,.06);box-shadow:0 12px 28px rgba(0,0,0,.07);}}
  .address-info{{display:flex;align-items:center;gap:.55rem;background:#fff;padding:.45rem 1.2rem;border-radius:999px;box-shadow:0 12px 28px rgba(0,0,0,.08);font-weight:600;}}
  .address-info svg{{width:16px;height:16px;color:{pal['primary']};}}
  .location-btn{{background:#fff;border:1px solid rgba(15,23,42,.12);border-radius:999px;padding:.45rem 1rem;font-weight:600;box-shadow:0 8px 18px rgba(0,0,0,.08);cursor:pointer;}}
  .location-btn:hover{{border-color:{pal['primary']};color:{pal['primary']};}}
  .nav-shell{{display:flex;align-items:center;gap:1.5rem;padding:1.25rem 0;flex-wrap:wrap;}}
  .nav-brand{{display:flex;align-items:center;gap:.75rem;font-weight:700;font-size:1.05rem;text-decoration:none;color:#0f172a;}}
  .nav-brand img{{height:40px;width:40px;border-radius:12px;object-fit:cover;border:1px solid rgba(0,0,0,.08);background:#fff;padding:6px;}}
  .nav-links{{display:flex;align-items:center;gap:1.5rem;font-weight:600;}}
  .nav-links a{{color:#1f1f1f;text-decoration:none;position:relative;padding-bottom:.2rem;}}
  .nav-links a:hover::after{{content:\"\";position:absolute;left:0;right:0;bottom:0;height:2px;background:{pal['primary']};}}
  .search-shell{{flex:1;display:flex;align-items:center;gap:.5rem;padding:.55rem 1rem;border-radius:999px;background:#fff;border:1px solid rgba(0,0,0,.06);box-shadow:0 12px 24px rgba(0,0,0,.08);min-width:220px;}}
  .search-shell input{{flex:1;border:0;background:transparent;font-size:.95rem;outline:none;}}
  .search-shell svg{{width:18px;height:18px;color:#94a3b8;}}
  .nav-cta{{display:flex;align-items:center;gap:.65rem;flex-wrap:wrap;}}
  .cart-btn{{width:42px;height:42px;border-radius:999px;border:1px solid rgba(0,0,0,.12);display:flex;align-items:center;justify-content:center;background:#fff;box-shadow:0 6px 14px rgba(0,0,0,.08);}}
  .cart-btn svg{{width:18px;height:18px;}}
  .btn-base{{display:inline-flex;align-items:center;gap:.6rem;padding:.6rem 1.2rem;border-radius:999px;font-weight:600;transition:transform .18s ease,box-shadow .18s ease,opacity .12s;border:0;text-decoration:none;cursor:pointer;}}
  .btn-primary{{background:linear-gradient(135deg, {pal['primary']} 0%, {pal['primary_dark']} 100%);color:#fff;box-shadow:0 10px 26px rgba(0,0,0,.18);}}
  .btn-primary:hover{{transform:translateY(-2px);}}
  .btn-secondary{{background:#fff;color:#111;border:1px solid rgba(0,0,0,.12);}}
  .btn-pill{{border-radius:999px;}}
  .fade-wrap{{position:relative;height:clamp(360px,70vh,760px);overflow:hidden;border-radius:30px;box-shadow:0 40px 80px rgba(0,0,0,.2);}}
  .fade-wrap img{{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;object-position:center;opacity:0;transition:opacity 900ms ease-in-out;}}
  .fade-wrap img.active{{opacity:1;}}
  .hero-copy{{position:absolute;inset:0;padding:clamp(1.8rem,4vw,4rem);display:flex;flex-direction:column;justify-content:flex-end;background:linear-gradient(120deg,rgba(16,7,3,.78),rgba(5,5,5,.2) 55%,rgba(0,0,0,0));color:#fff;border-radius:30px;}}
  .hero-kicker{{text-transform:uppercase;font-size:.75rem;letter-spacing:.25em;opacity:.78;margin-bottom:.6rem;font-weight:600;}}
  .hero-headline{{font-family:'InterLocal','Inter','system-ui',sans-serif;font-size:clamp(2.4rem,5vw,4rem);line-height:1.1;margin:0;}}
  .hero-subline{{font-size:1.2rem;margin-top:1rem;font-weight:600;}}
  .hero-actions{{display:flex;flex-wrap:wrap;gap:.9rem;margin-top:1.5rem;}}
  .hero-meta{{margin-top:1.2rem;font-size:.95rem;opacity:.9;display:flex;flex-wrap:wrap;gap:1rem;align-items:center;}}
  .hero-dots{{position:absolute;bottom:18px;right:32px;display:flex;gap:.4rem;}}
  .hero-dots .dot{{width:10px;height:10px;border-radius:999px;background:rgba(255,255,255,.45);}}
  .hero-dots .dot.active{{background:#fff;box-shadow:0 0 0 3px rgba(255,255,255,.25);}}
  .section-badge{{text-transform:uppercase;font-size:.75rem;letter-spacing:.3em;color:#6b7280;font-weight:600;}}
  .service-card{{position:relative;border-radius:18px;overflow:hidden;min-height:230px;display:flex;align-items:flex-end;padding:1.5rem;background-size:cover;background-position:center;box-shadow:0 18px 40px rgba(0,0,0,.25);color:#fff;}}
  .service-card__scrim{{position:absolute;inset:0;background:linear-gradient(180deg,rgba(0,0,0,0.05) 0%,rgba(0,0,0,0.9) 100%);}}
  .service-card__content{{position:relative;z-index:1;}}
  .service-card__content h3{{margin:0;font-size:1.25rem;font-weight:600;}}
  .service-card__content p{{margin:.35rem 0 0;font-size:.95rem;line-height:1.5;color:rgba(255,255,255,.85);}}
  .services-grid{{margin-top:2rem;display:grid;gap:1.5rem;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));}}
  

  .card{{background:#FFF;border-radius:1.25rem;box-shadow:var(--tw-shadow, 0 10px 30px rgba(0,0,0,.10));}}
  @media (max-width: 1024px) {{
    .nav-shell{{flex-direction:column;align-items:flex-start;}}
    .search-shell{{width:100%;}}
    .nav-cta{{width:100%;justify-content:flex-start;}}
  }}
  @media (max-width: 768px) {{
    .hero-copy{{position:relative;background:linear-gradient(120deg,rgba(16,7,3,.85),rgba(16,7,3,.65));border-radius:0 0 30px 30px;}}
    .hero-dots{{position:static;margin-top:1.2rem;}}
    .phone-graphic{{order:-1;}}
  }}
  main, section, footer, .container-shell {{
    padding-left: 20px !important;
    padding-right: 20px !important;
  }}
</style>
<meta http-equiv=\"Content-Security-Policy\"
  content=\"default-src 'self'; img-src 'self' data: https:; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com; connect-src 'self' https:;\">
<meta name=\"referrer\" content=\"no-referrer\">
</head>
<body class=\"font-body\">

  <header class="relative z-50">
    <div class="address-bar">
      <div class="{CONTAINER} flex flex-wrap items-center justify-between gap-3 py-2">
        <div class="address-info">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2C8.134 2 5 5.134 5 9c0 4.25 4.875 10.75 7 12.75C14.125 19.75 19 13.25 19 9c0-3.866-3.134-7-7-7Zm0 9.5A2.5 2.5 0 1 1 12 6.5a2.5 2.5 0 0 1 0 5Z" fill="#0f172a"/></svg>
          <span>{safe(nearest_location)}</span>
        </div>
        <button class="location-btn" type="button">Pick Another Location</button>
      </div>
    </div>
    <nav class="{CONTAINER} nav-shell">
      <a class="nav-brand" href="#top" aria-label="{safe(name)}">
        {"<img src='"+safe(logo)+"' alt='logo'/>" if logo else ""}
        <span>{safe(name)}</span>
      </a>
      <div class="nav-links">
        <a href="#rewards">Rewards</a>
        <a href="#contact">Locations</a>
        <a href="#menu">About</a>
      </div>
      <div class="search-shell">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><line x1="16.65" y1="16.65" x2="21" y2="21"/></svg>
        <input type="text" aria-label="Search" placeholder="{search_placeholder}"/>
      </div>
      <div class="nav-cta">
        <button class="cart-btn" aria-label="Cart">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="9" cy="21" r="1"/><circle cx="20" cy="21" r="1"/><path d="M5 6h2l1.68 8.39a2 2 0 0 0 1.99 1.61H19a2 2 0 0 0 1.98-1.75L22 7H7"/></svg>
        </button>
        <a href="{safe(website or map_url)}" target="_blank" rel="noopener" class="btn-base btn-primary btn-pill">Order Now</a>
        <button class="btn-base btn-secondary btn-pill" type="button">Log in</button>
      </div>
    </nav>
  </header>

  <section id=\"top\" class=\"pt-6\">
    <div class=\"{CONTAINER} relative\">
      <div class=\"fade-wrap\" id=\"heroWrap\">
        {"".join([f"<img class='hero-img {'active' if i==0 else ''}' src='{safe(u)}' alt='hero image {i+1}' {'loading=\\\"eager\\\"' if i==0 else 'loading=\\\"lazy\\\"'} decoding='async' onerror=\"this.src='{safe(HERO_FALLBACK_URL)}'\"/>" for i,u in enumerate(hero_imgs if hero_imgs else [HERO_FALLBACK_URL])])}
        <div class=\"hero-copy\">
          <p class=\"hero-kicker\">{safe(name)}</p>
          <h1 class=\"hero-headline\">{safe(hero_title)}</h1>
          <p class=\"hero-subline\">{hero_subline}</p>
          <div class=\"hero-actions\">
            <a href=\"{safe(website or map_url)}\" target=\"_blank\" rel=\"noopener\" class=\"btn-base btn-primary btn-pill\">Order online</a>
            <a href=\"#menu\" class=\"btn-base btn-ghost btn-pill\">View menu</a>
          </div>
          <div class=\"hero-meta\">
            {f"<span>★ {rating:.1f} ({int(review_count)}+ reviews)</span>" if rating and review_count else ""}
            {f"<span>{safe(service_tip)}</span>" if service_tip else ""}
          </div>
        </div>
        <div class=\"hero-dots\" id=\"heroDots\">
          {"".join([f"<div class='dot {'active' if i==0 else ''}' data-idx='{i}'></div>" for i in range(len(hero_imgs))])}
        </div>
      </div>
    </div>
  </section>

  <section id=\"menu\" class=\"mt-12 md:mt-16\">
    <div class=\"{CONTAINER}\">
      <div class=\"flex items-end justify-between gap-4 flex-wrap\">
        <div>
          <span class=\"section-badge\">Featured items</span>
          <h2 class=\"font-display text-3xl md:text-4xl mt-2\">Menu Highlights</h2>
          <p class=\"mt-2 text-[17px] leading-7 opacity-80\">{signature_copy}</p>
        </div>
        <a href=\"{safe(website or map_url)}\" target=\"_blank\" rel=\"noopener\" class=\"btn-base btn-primary btn-pill\">Order Online</a>
      </div>
      <div class=\"mt-6 grid md:grid-cols-2 lg:grid-cols-3 gap-7\">
        {"".join([f"""
        <article class=\"card overflow-hidden\">
          <div class=\"aspect-[4/3] w-full overflow-hidden\">
            <img class=\"w-full h-full object-cover\"
                 src=\"{safe(item['img'])}\"
                 alt=\"{safe(item['name'])}\"
                 loading=\"lazy\"
                 onerror=\"this.style.display='none'; this.nextElementSibling.style.display='flex'\"/>
            <div style=\"display:none\" class=\"w-full h-full bg-gray-100 flex items-center justify-center text-gray-400\">
              <span>Image not available</span>
            </div>
          </div>
          <div class=\"p-5 md:p-6 flex items-start justify-between gap-3\">
            <div>
              <h3 class=\"font-semibold text-lg\">{safe(item['name'])}</h3>
              <p class=\"mt-1 opacity-75 text-sm\">{safe(item['desc'])}</p>
            </div>
            <span class=\"inline-flex items-center rounded-full px-3 py-1 text-sm bg-brand/15 text-brand\">{safe(item['price'])}</span>
          </div>
        </article>
        """ for item in menu_items])}
      </div>
      <p class=\"mt-3 text-sm opacity-70\">*Pricing and availability may vary by location.</p>
    </div>
  </section>

  <section id=\"services\" class=\"mt-12 md:mt-20\">
    <div class=\"{CONTAINER}\">
      <div class=\"flex flex-wrap items-end justify-between gap-6\">
        <div>
          <span class=\"section-badge\">Beyond the dining room</span>
          <h2 class=\"font-display text-3xl md:text-4xl mt-2\">Bring {safe(name)} to every occasion</h2>
          <p class=\"mt-2 text-[17px] leading-7 opacity-80\">Your brand settings will appear consistently across all of your stores in Restronaut.</p>
        </div>
      </div>
      <div class=\"services-grid\">
        {service_cards_html}
      </div>
    </div>
  </section>
  <section id=\"gallery\" class=\"mt-12 md:mt-16\">
    <div class=\"{CONTAINER}\">
      <span class=\"section-badge\">Inside look</span>
      <h2 class=\"font-display text-3xl md:text-4xl mt-2\">Gallery</h2>
      <div class=\"mt-6 grid sm:grid-cols-2 lg:grid-cols-4 gap-4\">
        {"".join([
          f"<a href='{safe(u)}' target='_blank' rel='noopener' class='block rounded-2xl overflow-hidden card'><img src='{safe(u)}' data-fallbacks='{safe(fallback_foto)}' onerror='__imgSwap(this)' class='w-full h-48 object-cover' loading='lazy' decoding='async'/></a>"
          for u in (gallery[:4] or [])
        ])}
      </div>
    </div>
  </section>
  <section id="website-bottom" class="mt-8" style="padding-left:0 !important; padding-right:0 !important; margin-top:20px;">
    <div style="width:100%; display:flex; justify-content:center; margin:0;">
      <img src="{WEBSITE_BOTTOM_SVG}" alt="Website Bottom Graphic" style="width:100%; height:auto; display:block; margin:0; padding:0;"/>
    </div>
  </section>
  <section id="contact" class="mt-12 md:mt-16 mb-16" style="padding-left:0; padding-right:0; border-top-left-radius:0; border-top-right-radius:0">
    <div class=\"{CONTAINER}\">
      <div class=\"card p-8 grid md:grid-cols-3 gap-6\">
        <div>
          <h3 class=\"font-display text-2xl\">Visit Us</h3>
          <p class=\"mt-2 opacity-85\">{safe(address)}</p>
          {"<p class='mt-2'><a class='underline' href='tel:"+safe(phone)+"'>"+safe(phone)+"</a></p>" if phone else ""}
        </div>
        <div>
          <h3 class=\"font-display text-2xl\">Online</h3>
          {"<p class='mt-2'><a class='underline' href='"+safe(website)+"' target='_blank' rel='noopener'>"+safe(website)+"</a></p>" if website else "<p class='mt-2 opacity-75'>Website not provided</p>"}
          <p class=\"mt-2\"><a class=\"underline\" href=\"{safe(map_url)}\" target=\"_blank\" rel=\"noopener\">Google Maps</a></p>
        </div>
        <div class=\"flex items-end md:items-center md:justify-end\">
          <a href=\"{safe(website or map_url)}\" target=\"_blank\" rel=\"noopener\" class=\"btn-base btn-primary btn-pill\">Book / Order</a>
        </div>
      </div>
      <div class=\"mt-6 text-center opacity-70\">
        <span>© {time.strftime("%Y")} {safe(name)} • Made by <strong>Restronaut.ai</strong></span>
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
