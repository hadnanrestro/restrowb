import sys
import time
import logging
from typing import Dict, Any, List, Tuple

log = logging.getLogger("uvicorn.error")

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
    select_hero_images = getattr(_app, "select_hero_images")
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
    
    # Smart hero image selection (prefers interior shots from Google, then curated Unsplash)
    hero_imgs: List[str] = await select_hero_images(details, cuisine)

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
        {"".join([f"<img class='hero-img {'active' if i==0 else ''}' src='{safe(u)}' alt='hero image {i+1}' {'loading=\"eager\"' if i==0 else 'loading=\"lazy\"'} decoding='async' onerror=\"this.src='{safe(HERO_FALLBACK_URL)}'\"/>" for i,u in enumerate(hero_imgs if hero_imgs else [HERO_FALLBACK_URL])])}
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
