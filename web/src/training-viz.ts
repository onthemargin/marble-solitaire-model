// Real greedy-eval marble counts for each deployed gen, measured by
// running each ONNX model from the initial board to terminal state.
const GENS = [
  { name: 'Clueless',     marbles: 12, xPct: 0.04 },
  { name: 'Beginner',     marbles: 8,  xPct: 0.18 },
  { name: 'Intermediate', marbles: 8,  xPct: 0.36 },
  { name: 'Advanced',     marbles: 5,  xPct: 0.66 },
  { name: 'Master',       marbles: 2,  xPct: 0.96 },
];

const OPTIMUM = 2;
const Y_MAX = 13; // chart range: 2 (top) → 13 (bottom)
const Y_MIN = 2;

// Chart geometry (SVG viewBox 360×200)
const W = 360, H = 200;
const PL = 14, PR = 16, PT = 18, PB = 36;
const cw = W - PL - PR;
const ch = H - PT - PB;

const xPctToSvg = (pct: number) => PL + pct * cw;
const yMarbleToSvg = (m: number) => PT + ((m - Y_MIN) / (Y_MAX - Y_MIN)) * ch;

const sleep = (ms: number) => new Promise<void>(r => setTimeout(r, ms));

// Smooth path that never overshoots between anchors — horizontal-tangent
// cubic Beziers, so plateaus stay flat and steep segments stay steep.
function buildCurvePath(): { d: string } {
  const pts = [
    { x: PL, y: yMarbleToSvg(GENS[0].marbles) },
    ...GENS.map(g => ({ x: xPctToSvg(g.xPct), y: yMarbleToSvg(g.marbles) })),
  ];
  let d = `M ${pts[0].x} ${pts[0].y}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const mx = (p1.x + p2.x) / 2;
    d += ` C ${mx} ${p1.y}, ${mx} ${p2.y}, ${p2.x} ${p2.y}`;
  }
  return { d };
}

function svgEl(tag: string, attrs: Record<string, string | number> = {}): SVGElement {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const k in attrs) el.setAttribute(k, String(attrs[k]));
  return el;
}

function buildChart(svg: SVGSVGElement): { path: SVGPathElement; dot: SVGCircleElement; gens: SVGCircleElement[]; labels: HTMLElement[] } {
  svg.innerHTML = '';
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);

  // Optimum line (dashed) at marbles=2
  const optY = yMarbleToSvg(OPTIMUM);
  svg.appendChild(svgEl('line', {
    x1: PL, x2: W - PR, y1: optY, y2: optY,
    class: 'tv-opt-line',
  }));
  const optLabel = svgEl('text', {
    x: W - PR, y: optY - 6, class: 'tv-opt-label', 'text-anchor': 'end',
  });
  optLabel.textContent = 'Proven minimum: 2';
  svg.appendChild(optLabel);

  // Y-axis label
  const yLabel = svgEl('text', {
    x: PL, y: PT - 6, class: 'tv-axis-label',
  });
  yLabel.textContent = 'marbles left ↓';
  svg.appendChild(yLabel);

  // Curve path
  const { d } = buildCurvePath();
  const path = svgEl('path', { d, class: 'tv-curve' }) as SVGPathElement;
  svg.appendChild(path);

  // Gen markers + x-axis labels
  const gens: SVGCircleElement[] = [];
  const xAxisY = H - PB + 12;
  for (const g of GENS) {
    const cx = xPctToSvg(g.xPct);
    const cy = yMarbleToSvg(g.marbles);
    const marker = svgEl('circle', {
      cx, cy, r: 4, class: 'tv-gen-marker',
    }) as SVGCircleElement;
    svg.appendChild(marker);
    gens.push(marker);

    const tick = svgEl('line', {
      x1: cx, x2: cx, y1: H - PB, y2: H - PB + 4, class: 'tv-tick',
    });
    svg.appendChild(tick);

    const lbl = svgEl('text', {
      x: cx, y: xAxisY, class: 'tv-gen-label', 'text-anchor': 'middle',
    });
    lbl.textContent = g.name;
    svg.appendChild(lbl);
  }

  // Animated dot
  const dot = svgEl('circle', {
    cx: xPctToSvg(0), cy: yMarbleToSvg(GENS[0].marbles), r: 5, class: 'tv-cursor',
  }) as SVGCircleElement;
  svg.appendChild(dot);

  return { path, dot, gens, labels: [] };
}

export function initTrainingViz(root: HTMLDetailsElement) {
  const svg = root.querySelector('.tv-chart') as SVGSVGElement;
  const marbleReadout = root.querySelector('.tv-readout-marbles') as HTMLElement | null;
  const genReadout = root.querySelector('.tv-readout-gen') as HTMLElement | null;

  let token = 0;

  function nearestGen(xCoord: number): { name: string; marbles: number } {
    let best = GENS[0];
    let bestDist = Math.abs(xPctToSvg(best.xPct) - xCoord);
    for (const g of GENS) {
      const d = Math.abs(xPctToSvg(g.xPct) - xCoord);
      if (d < bestDist) { best = g; bestDist = d; }
    }
    return best;
  }

  async function loop(myToken: number) {
    const { path, dot, gens } = buildChart(svg);
    const totalLen = path.getTotalLength();
    const durationMs = 6000;

    while (myToken === token) {
      // Reset
      gens.forEach(m => m.classList.remove('hit'));
      dot.classList.remove('settled');
      if (marbleReadout) marbleReadout.textContent = String(GENS[0].marbles);
      if (genReadout) genReadout.textContent = GENS[0].name;
      const startTime = performance.now();
      const hitFlags = new Array(GENS.length).fill(false);

      while (myToken === token) {
        const elapsed = performance.now() - startTime;
        const t = Math.min(1, elapsed / durationMs);
        const pt = path.getPointAtLength(t * totalLen);
        dot.setAttribute('cx', String(pt.x));
        dot.setAttribute('cy', String(pt.y));

        // Marble count readout: snap to curve y-value
        const marblesNow = Math.max(OPTIMUM, Y_MIN + (pt.y - PT) / ch * (Y_MAX - Y_MIN));
        if (marbleReadout) marbleReadout.textContent = marblesNow.toFixed(1);

        // Gen marker hit detection
        for (let i = 0; i < GENS.length; i++) {
          if (!hitFlags[i]) {
            const gx = xPctToSvg(GENS[i].xPct);
            if (pt.x >= gx - 1) {
              hitFlags[i] = true;
              gens[i].classList.add('hit');
              if (genReadout) genReadout.textContent = GENS[i].name;
            }
          }
        }

        if (t >= 1) break;
        await sleep(33);
      }
      if (myToken !== token) return;

      // Hold at the plateau
      dot.classList.add('settled');
      if (marbleReadout) marbleReadout.textContent = '2.0';
      if (genReadout) genReadout.textContent = 'Master · optimum';
      await sleep(2500);
      if (myToken !== token) return;
      await sleep(500);
    }
  }

  function start() {
    const t = ++token;
    loop(t);
  }
  function stop() {
    ++token;
  }

  root.addEventListener('toggle', () => {
    if (root.open) start();
    else stop();
  });
}
