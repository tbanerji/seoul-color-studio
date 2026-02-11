import streamlit as st
from PIL import Image
import numpy as np
import io, zipfile

# ----------------------------
# PAGE SETUP — Barbie/Glossier Seoul Studio
# ----------------------------
st.set_page_config(page_title="Seoul Color Studio", layout="wide")

st.markdown("""
<style>
  :root{
    --bg: #FFF1F7;
    --card: #FFFFFF;
    --ink: #2B2B2B;
    --muted: #6B6B6B;
    --accent: #FF4DA6;
    --accent2: #FFD1E8;
    --border: rgba(255, 77, 166, 0.18);
    --shadow: 0 14px 35px rgba(17, 12, 25, 0.08);
    --radius: 20px;
  }

  [data-testid="stAppViewContainer"]{ background: var(--bg); }
  .block-container{ max-width: 1180px; padding-top: 1.2rem; padding-bottom: 3rem; }

  /* Typography */
  h1,h2,h3{ color: var(--ink); letter-spacing: -0.03em; }
  .label{
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 12px;
    color: var(--muted);
    font-weight: 850;
  }
  .sub{
    color: var(--muted);
    font-size: 0.95rem;
    line-height: 1.35;
  }

  /* Cards */
  .hero{
    background: linear-gradient(180deg, rgba(255,255,255,0.88), rgba(255,255,255,0.78));
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 16px 18px;
    margin-bottom: 14px;
    position: relative;
    overflow: hidden;
  }
  .hero:after{
    content:"";
    position:absolute;
    right:-120px;
    top:-120px;
    width: 320px;
    height: 320px;
    background: radial-gradient(circle, rgba(255,77,166,0.20), transparent 60%);
    transform: rotate(12deg);
  }
  .card{
    background: rgba(255,255,255,0.88);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 14px;
  }
  .chip{
    display:inline-block;
    padding: 7px 10px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: rgba(255,209,232,0.35);
    color: var(--ink);
    font-weight: 800;
    font-size: 12px;
    margin-right: 8px;
  }
  .badge{
    display:inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    background: var(--accent);
    color: white;
    font-weight: 900;
    font-size: 12px;
  }
  .divider{
    height: 1px;
    background: rgba(43,43,43,0.10);
    margin: 14px 0;
  }

  /* Buttons */
  div.stButton > button{
    width:100%;
    height: 44px;
    border-radius: 14px;
    border: 1px solid var(--border);
    background: white;
    font-weight: 900;
    color: var(--ink);
  }
  div.stButton > button:hover{
    border-color: rgba(255,77,166,0.45);
    background: rgba(255,209,232,0.25);
  }
  .primary div.stButton > button{
    background: var(--accent);
    color: white;
    border: 1px solid rgba(255,77,166,0.55);
  }
  .primary div.stButton > button:hover{
    background: #ff2f93;
  }

  /* Sidebar */
  section[data-testid="stSidebar"]{
    background: rgba(255,255,255,0.78);
    border-right: 1px solid var(--border);
  }

  /* Make radios tighter */
  [data-testid="stRadio"] label{ font-weight: 800; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# STATE
# ----------------------------
if "top_hex" not in st.session_state: st.session_state.top_hex = "#0B0D12"
if "bot_hex" not in st.session_state: st.session_state.bot_hex = "#0B0D12"
if "masks" not in st.session_state: st.session_state.masks = None
if "last_file" not in st.session_state: st.session_state.last_file = None
if "lookbook" not in st.session_state: st.session_state.lookbook = []

# ----------------------------
# MODEL LOADING (SegFormer clothes)
# ----------------------------
@st.cache_resource
def load_engine():
    import cv2
    import torch
    import torch.nn as nn
    from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model.eval()
    return processor, model, cv2, torch, nn

def get_mask_soft(image_pil):
    """
    Returns SOFT float alpha mask (0..1) for clothes. Feathered + tightened to reduce halos.
    """
    processor, model, cv2, torch, nn = load_engine()

    inputs = processor(images=image_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits.cpu()
    upsampled = nn.functional.interpolate(
        logits, size=image_pil.size[::-1], mode="bilinear", align_corners=False
    )
    pred = upsampled.argmax(dim=1)[0].numpy()

    clothing_indices = [4, 5, 6, 7, 9, 10, 12]
    hard = np.isin(pred, clothing_indices).astype(np.float32)

    hard_u8 = (hard * 255).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    hard_u8 = cv2.morphologyEx(hard_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    hard = hard_u8.astype(np.float32) / 255.0

    blur_k = max(7, int(min(image_pil.size) * 0.010) // 2 * 2 + 1)
    soft = cv2.GaussianBlur(hard, (blur_k, blur_k), 0)

    soft = np.clip((soft - 0.08) / (1.0 - 0.08), 0.0, 1.0)
    return soft

# ----------------------------
# COLOR ENGINE (soft alpha blend – no white line)
# ----------------------------
def hex_to_rgb(hx):
    hx = hx.lstrip("#")
    return np.array([int(hx[0:2],16), int(hx[2:4],16), int(hx[4:6],16)], dtype=np.float32)

def apply_solid_color(image_pil, mask_soft, split_ratio, top_hex, bot_hex, intensity=1.0, texture_boost=0.14):
    import cv2

    img = np.array(image_pil.convert("RGB")).astype(np.float32)
    h, w = img.shape[:2]
    split_y = int(h * split_ratio)

    m = np.clip(mask_soft, 0.0, 1.0).astype(np.float32)
    top_m = np.zeros((h, w), dtype=np.float32); top_m[:split_y, :] = m[:split_y, :]
    bot_m = np.zeros((h, w), dtype=np.float32); bot_m[split_y:, :] = m[split_y:, :]

    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    g = cv2.normalize(gray, None, alpha=0.18, beta=0.98, norm_type=cv2.NORM_MINMAX)
    g = np.clip((g - 0.5) * (1.0 + texture_boost) + 0.5, 0.0, 1.0)

    def make_layer(hex_code):
        rgb = hex_to_rgb(hex_code)
        layer = np.zeros_like(img)
        layer[:,:,0] = g * rgb[0]
        layer[:,:,1] = g * rgb[1]
        layer[:,:,2] = g * rgb[2]
        layer = np.clip(layer * intensity, 0, 255)
        return layer

    out = img.copy()
    for layer, alpha in [(make_layer(top_hex), top_m), (make_layer(bot_hex), bot_m)]:
        a3 = alpha[..., None]
        out = out * (1.0 - a3) + layer * a3

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

# ----------------------------
# DATA — Seoul-style palettes (mood names)
# ----------------------------
SEASONS = {
    "Winter Girl ❄️ (Cool / Deep)": {
        "tag": "crisp + high contrast",
        "swatches": [("#0B0D12","Ink"),("#0E2A47","Midnight"),("#2F2C7A","Violet"),("#7B002C","Berry"),("#0E6E5C","Teal"),("#E9F2FF","Ice")]
    },
    "Spring Bloom 🌸 (Warm / Clear)": {
        "tag": "bright + clear",
        "swatches": [("#FFF06A","Lemon"),("#FF8A3D","Tangerine"),("#2ED8B6","Mint"),("#FF4FA6","Jelly Pink"),("#C7B6FF","Lilac"),("#FFF7FB","Milk")]
    },
    "Autumn Latte 🍂 (Warm / Rich)": {
        "tag": "rich + earthy",
        "swatches": [("#3A1F14","Espresso"),("#9C4A1A","Cinnamon"),("#556B2F","Olive"),("#C79A2B","Ochre"),("#B4433C","Brick"),("#FFF1E6","Cream")]
    },
    "Summer Soft ☁️ (Cool / Muted)": {
        "tag": "soft + airy",
        "swatches": [("#E8E6FF","Periwinkle"),("#BFD7FF","Sky"),("#D8A7B1","Dusty Rose"),("#A9D6C8","Sage Mint"),("#B8B8C8","Mauve Gray"),("#FFFFFF","White")]
    },
}

WEAR_AVOID = {
    "Winter Girl ❄️ (Cool / Deep)": {
        "wear": ["#0B0D12","#0E2A47","#7B002C","#0E6E5C","#E9F2FF"],
        "avoid": ["#F4D1B8","#C7B08B","#E2C56A","#B58C6D","#D9B2A9"],
        "copy": "Deep cool tones = instant glow. High contrast makes your skin look clearer + sharper."
    },
    "Spring Bloom 🌸 (Warm / Clear)": {
        "wear": ["#FFF06A","#FF8A3D","#2ED8B6","#FF4FA6","#FFF7FB"],
        "avoid": ["#0B0D12","#2B2B2B","#5B5B5B","#2F2C7A","#6A5D4D"],
        "copy": "Clear warm tones brighten your face. Too-dark shades can feel heavy."
    },
    "Autumn Latte 🍂 (Warm / Rich)": {
        "wear": ["#3A1F14","#9C4A1A","#556B2F","#C79A2B","#FFF1E6"],
        "avoid": ["#E9F2FF","#FFFFFF","#BFD7FF","#2F2C7A","#FF4FA6"],
        "copy": "Rich warmth adds depth. Icy tones can wash you out."
    },
    "Summer Soft ☁️ (Cool / Muted)": {
        "wear": ["#E8E6FF","#BFD7FF","#D8A7B1","#A9D6C8","#FFFFFF"],
        "avoid": ["#0B0D12","#7B002C","#FF4FA6","#C79A2B","#3A1F14"],
        "copy": "Soft contrast looks effortless. Super-saturated shades can overpower you."
    },
}

def swatch_row(hex_list):
    html = "<div style='display:flex; gap:10px; flex-wrap:wrap;'>"
    for hx in hex_list:
        html += f"<div title='{hx}' style='width:54px;height:36px;border-radius:14px;border:1px solid rgba(43,43,43,0.10);background:{hx};'></div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def make_zip_from_lookbook(items):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, it in enumerate(items, start=1):
            img_bytes = io.BytesIO()
            it["img"].save(img_bytes, format="PNG")
            zf.writestr(f"look_{i:02d}.png", img_bytes.getvalue())
    return buf.getvalue()

# ----------------------------
# CLICKABLE TILE SWATCHES
# ----------------------------
def render_clickable_swatch_tiles(swatches, pick_target: str):
    """
    swatches: list of (hex, name)
    pick_target: "Top" or "Bottom"
    """
    selected = st.session_state.top_hex if pick_target == "Top" else st.session_state.bot_hex

    st.markdown(f"<div class='label'>Tap a color for your {pick_target.lower()}</div>", unsafe_allow_html=True)

    # 3 columns × 2 rows (works nicely for 6 swatches)
    cols = st.columns(3, gap="medium")
    for i, (hx, nm) in enumerate(swatches):
        with cols[i % 3]:
            is_sel = (hx.lower() == (selected or "").lower())
            border = "2px solid rgba(255,77,166,0.85)" if is_sel else "1px solid rgba(255,77,166,0.18)"
            ring = "box-shadow: 0 0 0 6px rgba(255,77,166,0.12);" if is_sel else ""
            badge = "<span class='badge'>✓</span>" if is_sel else ""

            st.markdown(
                f"""
                <div style="border:{border}; {ring} border-radius:18px; padding:12px; background:rgba(255,255,255,0.9);">
                  <div style="display:flex; align-items:center; justify-content:space-between; gap:10px;">
                    <div style="font-weight:900; font-size:14px; color:#2B2B2B;">{nm}</div>
                    {badge}
                  </div>
                  <div style="margin-top:10px; height:44px; border-radius:14px; border:1px solid rgba(43,43,43,0.10); background:{hx};"></div>
                  <div class="sub" style="margin-top:8px; font-size:12px;">{hx}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Button acts like "tap" action
            if st.button("Tap", key=f"tap_{pick_target}_{i}"):
                if pick_target == "Top":
                    st.session_state.top_hex = hx
                else:
                    st.session_state.bot_hex = hx
                st.rerun()

# ----------------------------
# HERO
# ----------------------------
st.markdown("""
<div class="hero">
  <div class="label">Seoul Color Studio</div>
  <div style="display:flex; align-items:flex-end; justify-content:space-between; gap:14px;">
    <div>
      <h1 style="margin:0;">K-Style Color Try-On</h1>
      <div class="sub">Minimal, girly, Gen-Z. Test tops + bottoms like you’re shopping in Hongdae.</div>
      <div style="margin-top:10px;">
        <span class="chip">1 Upload</span>
        <span class="chip">2 Pick Season</span>
        <span class="chip">3 Tap Colors</span>
        <span class="chip">4 Save & Export</span>
      </div>
    </div>
    <div style="text-align:right;">
      <div class="label">Vibe</div>
      <div style="font-weight:950; font-size:18px;">Glossier × Barbie</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# SIDEBAR
# ----------------------------
with st.sidebar:
    st.markdown("### Studio")
    uploaded = st.file_uploader("Upload your photo", type=["jpg","jpeg","png"])
    st.caption("Best in daylight. Avoid yellow indoor lighting.")

    st.markdown("---")
    st.markdown("**Refine fit (optional)**")
    split_ratio = st.slider("Waistline", 0.35, 0.80, 0.53, 0.01)
    intensity = st.slider("Color strength", 0.70, 1.35, 1.05, 0.05)
    texture_boost = st.slider("Fabric texture", 0.00, 0.35, 0.14, 0.01)
    rerun = st.button("Re-run outfit detection")

# ----------------------------
# MAIN
# ----------------------------
if not uploaded:
    st.markdown("<div class='card'><b>Start here</b><div class='sub' style='margin-top:6px;'>Upload a photo where your outfit is visible (torso in frame). Then pick a season board and tap colors to try them on you.</div></div>", unsafe_allow_html=True)
    st.stop()

image = Image.open(uploaded).convert("RGB")

new_file = (st.session_state.last_file != uploaded.name)
if new_file or rerun or st.session_state.masks is None:
    with st.spinner("✨ Reading your outfit…"):
        st.session_state.masks = get_mask_soft(image)
        st.session_state.last_file = uploaded.name

mask_soft = st.session_state.masks

# ----------------------------
# RESULTS LAYOUT
# ----------------------------
colL, colR = st.columns([1.05, 1.65], gap="large")

with colL:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='label'>Step 2</div>", unsafe_allow_html=True)
    st.markdown("## Pick your season board")

    season_names = list(SEASONS.keys())
    picked = st.radio("Season boards", season_names, index=0, label_visibility="collapsed")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='label'>Season tiles</div>", unsafe_allow_html=True)

    for name in season_names:
        data = SEASONS[name]
        selected = (name == picked)
        opacity = "1.0" if selected else "0.55"
        border = "rgba(255,77,166,0.45)" if selected else "rgba(255,77,166,0.18)"
        badge = "<span class='badge'>BEST MATCH</span>" if selected else ""

        st.markdown(f"""
        <div style="border:1px solid {border}; border-radius:18px; padding:12px; background: rgba(255,255,255,{opacity}); margin-bottom:10px;">
          <div style="display:flex; align-items:center; justify-content:space-between;">
            <div style="font-weight:950; font-size:16px;">{name}</div>
            {badge}
          </div>
          <div class="sub" style="margin-top:4px;">{data['tag']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='label'>Step 3</div>", unsafe_allow_html=True)
    st.markdown("## Tap colors (like a studio mirror)")

    swatches = SEASONS[picked]["swatches"]

    pick_target = st.radio("Applying to…", ["Top", "Bottom"], horizontal=True, index=0)
    render_clickable_swatch_tiles(swatches, pick_target)

    with st.expander("Custom colors"):
        st.session_state.top_hex = st.color_picker("Custom top", st.session_state.top_hex)
        st.session_state.bot_hex = st.color_picker("Custom bottom", st.session_state.bot_hex)

    st.markdown("</div>", unsafe_allow_html=True)

with colR:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='label'>Your Seoul Color Match</div>", unsafe_allow_html=True)
    st.markdown(f"## {picked.split('(')[0].strip()}")
    st.markdown(f"<div class='sub'>{WEAR_AVOID[picked]['copy']}</div>", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    final_img = apply_solid_color(
        image_pil=image,
        mask_soft=mask_soft,
        split_ratio=split_ratio,
        top_hex=st.session_state.top_hex,
        bot_hex=st.session_state.bot_hex,
        intensity=intensity,
        texture_boost=texture_boost
    )

    view = st.radio("View", ["Before / After", "After only"], horizontal=True, index=0)
    if view == "Before / After":
        a, b = st.columns(2, gap="medium")
        with a:
            st.markdown("<div class='label'>Before</div>", unsafe_allow_html=True)
            st.image(image, use_container_width=True)
        with b:
            st.markdown("<div class='label'>After</div>", unsafe_allow_html=True)
            st.image(final_img, use_container_width=True)
    else:
        st.image(final_img, use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    wcol, acol = st.columns(2, gap="medium")
    with wcol:
        st.markdown("<div class='label'>Wear this</div>", unsafe_allow_html=True)
        st.markdown("**Your glow colors**")
        swatch_row(WEAR_AVOID[picked]["wear"])
    with acol:
        st.markdown("<div class='label'>Avoid this</div>", unsafe_allow_html=True)
        st.markdown("**These can dull you**")
        swatch_row(WEAR_AVOID[picked]["avoid"])

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown("<div class='label'>On you</div>", unsafe_allow_html=True)
    st.markdown("### Mini try-on set")

    preview_hexes = [hx for hx,_ in swatches][:4] if len(swatches) >= 4 else [hx for hx,_ in swatches]
    grid = st.columns(4, gap="small")
    for i, hx in enumerate(preview_hexes):
        im = apply_solid_color(
            image_pil=image,
            mask_soft=mask_soft,
            split_ratio=split_ratio,
            top_hex=hx,
            bot_hex=st.session_state.bot_hex,
            intensity=intensity,
            texture_boost=texture_boost
        )
        with grid[i]:
            st.image(im, use_container_width=True)
            st.caption(hx)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    cta1, cta2 = st.columns([1, 1], gap="medium")

    with cta1:
        st.markdown("<div class='primary'>", unsafe_allow_html=True)
        if st.button("💖 Save this look"):
            title = f"{picked.split('(')[0].strip()} • top {st.session_state.top_hex} • bot {st.session_state.bot_hex}"
            st.session_state.lookbook.append({"title": title, "img": final_img})
            st.success("Saved to Lookbook.")
        st.markdown("</div>", unsafe_allow_html=True)

    with cta2:
        out = io.BytesIO()
        final_img.save(out, format="PNG")
        st.download_button(
            "📥 Download result (PNG)",
            data=out.getvalue(),
            file_name="seoul_color_tryon.png",
            mime="image/png",
            use_container_width=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# LOOKBOOK
# ----------------------------
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='label'>Lookbook</div>", unsafe_allow_html=True)
st.markdown("## Saved looks")

if not st.session_state.lookbook:
    st.markdown("<div class='sub'>Save looks to compare outfits like a real Seoul studio recap.</div>", unsafe_allow_html=True)
else:
    cols = st.columns(3, gap="medium")
    for i, it in enumerate(st.session_state.lookbook):
        with cols[i % 3]:
            st.image(it["img"], use_container_width=True)
            st.caption(it["title"])
            if st.button("Remove", key=f"rm_{i}"):
                st.session_state.lookbook.pop(i)
                st.rerun()

    zip_bytes = make_zip_from_lookbook(st.session_state.lookbook)
    st.download_button(
        "👜 Export Lookbook (ZIP)",
        data=zip_bytes,
        file_name="seoul_color_lookbook.zip",
        mime="application/zip",
        use_container_width=True
    )

st.markdown("</div>", unsafe_allow_html=True)
