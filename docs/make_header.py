"""Generate the README header SVG. Self-hosted on purpose.

capsule-render and friends time out through GitHub's camo proxy often enough
to leave a broken image on the front page, so the header is generated here and
committed. Regenerate with:  python docs/make_header.py

Validates as XML before writing - a bare '&' in SVG text is invalid XML and
GitHub renders it as "Invalid image source" with no other warning.
"""
import pathlib
import xml.dom.minidom

W, H = 1200, 340
BG = "#0A0A14"
AMBER, CORAL, PINK = "#FFB020", "#FF6B4A", "#FF4E88"
LIME, BLUE, DIM = "#A3E635", "#5B9BFF", "#8B8BA7"

BARS = [("majority", 0.483, DIM), ("fine-tune", 0.624, DIM),
        ("probe", 0.670, DIM), ("ours", 0.697, None), ("Jev", 0.727, CORAL)]


def bars_svg(x0, y0, w, h, gap=14):
    """Horizontal accuracy bars, drawn to scale against a 0.80 axis."""
    out, bh = [], (h - gap * (len(BARS) - 1)) / len(BARS)
    for i, (name, v, colour) in enumerate(BARS):
        y = y0 + i * (bh + gap)
        bw = w * v / 0.80
        fill = "url(#grad)" if colour is None else colour
        op = "1" if colour is None else ("0.9" if colour == CORAL else "0.35")
        out.append(f'<rect x="{x0}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
                   f'rx="3" fill="{fill}" opacity="{op}"/>')
        out.append(f'<text x="{x0 + bw + 10:.1f}" y="{y + bh / 2 + 4:.1f}" '
                   f'font-family="ui-monospace,SFMono-Regular,Menlo,monospace" '
                   f'font-size="13" fill="{"#FFFFFF" if colour is None else DIM}" '
                   f'font-weight="{"700" if colour is None else "400"}">'
                   f'{v:.3f}  {name}</text>')
    return "\n    ".join(out)


SVG = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"
     viewBox="0 0 {W} {H}" role="img"
     aria-label="Open Jev: a typed decision engine you can train for free.
     0.697 accuracy against TypeSafe Jev's 0.727, 2.5x better calibrated,
     4x faster, zero cost.">
  <defs>
    <linearGradient id="grad" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="{AMBER}"/>
      <stop offset="55%" stop-color="{CORAL}"/>
      <stop offset="100%" stop-color="{PINK}"/>
    </linearGradient>
    <linearGradient id="fade" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="{CORAL}" stop-opacity="0.22"/>
      <stop offset="100%" stop-color="{BG}" stop-opacity="0"/>
    </linearGradient>
    <pattern id="grid" width="28" height="28" patternUnits="userSpaceOnUse">
      <path d="M28 0 L0 0 0 28" fill="none" stroke="#FFFFFF" stroke-opacity="0.04"
            stroke-width="1"/>
    </pattern>
  </defs>

  <rect width="{W}" height="{H}" fill="{BG}"/>
  <rect width="{W}" height="{H}" fill="url(#grid)"/>
  <circle cx="{W - 160}" cy="70" r="240" fill="url(#fade)"/>

  <text x="64" y="86" font-family="ui-monospace,SFMono-Regular,Menlo,monospace"
        font-size="13" letter-spacing="3.5" fill="{LIME}">SYSTEM ONE MODEL
        &#183; OPEN REPRODUCTION</text>

  <text x="60" y="164" font-family="Georgia,'Times New Roman',serif"
        font-size="68" font-weight="700" fill="url(#grad)">Open Jev</text>

  <text x="64" y="204" font-family="-apple-system,Segoe UI,Helvetica,sans-serif"
        font-size="19" fill="#E8E8F0">A typed decision engine you can train
        for free.</text>
  <text x="64" y="232" font-family="-apple-system,Segoe UI,Helvetica,sans-serif"
        font-size="19" fill="{DIM}">State + typed questions &#8594; one pass
        &#8594; calibrated answers.</text>

  <g font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="12">
    <rect x="64" y="262" width="150" height="28" rx="14" fill="none"
          stroke="{AMBER}" stroke-opacity="0.55"/>
    <text x="139" y="281" fill="{AMBER}" text-anchor="middle">150M PARAMS</text>

    <rect x="226" y="262" width="164" height="28" rx="14" fill="none"
          stroke="{LIME}" stroke-opacity="0.55"/>
    <text x="308" y="281" fill="{LIME}" text-anchor="middle">2.5&#215; CALIBRATION</text>

    <rect x="402" y="262" width="132" height="28" rx="14" fill="none"
          stroke="{BLUE}" stroke-opacity="0.55"/>
    <text x="468" y="281" fill="{BLUE}" text-anchor="middle">4&#215; FASTER</text>

    <rect x="546" y="262" width="108" height="28" rx="14" fill="none"
          stroke="{PINK}" stroke-opacity="0.55"/>
    <text x="600" y="281" fill="{PINK}" text-anchor="middle">$0 / CALL</text>
  </g>

  <text x="820" y="74" font-family="ui-monospace,SFMono-Regular,Menlo,monospace"
        font-size="11" letter-spacing="2" fill="{DIM}">ACCURACY &#183;
        typed-decisions, 400 cases</text>
  <g>
    {bars_svg(820, 92, 205, 196)}
  </g>
</svg>
"""

if __name__ == "__main__":
    out = pathlib.Path(__file__).parent / "header.svg"
    xml.dom.minidom.parseString(SVG)          # fails loudly on a bare '&'
    out.write_text(SVG, encoding="utf-8")
    print(f"wrote {out} ({len(SVG)} bytes, valid XML)")
