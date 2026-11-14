/**
 * lightLayout.ts
 *
 * Ceiling layouts with more than four LEDs, and the four-colour tone-reuse pattern
 * the backend uses (VLCL_AI/integrated_vlcl/multicell.py: assign_reuse_colours).
 * On a rectangular grid the pattern is 2x2: LEDs of the same colour (same tone set)
 * are two grid steps apart.
 */
import { LEDParams, RoomParams } from "./types";

export const REUSE_COLOURS_HEX = [0xfde68a, 0x67e8f9, 0xa78bfa, 0x6ee7b7]; // tone set 1..4
export const REUSE_COLOURS_CSS = ["#fde68a", "#67e8f9", "#a78bfa", "#6ee7b7"];

/** colour 1..4 per LED id. Grid → 2x2 pattern; otherwise greedy "farthest same colour". */
export function reuseColours(leds: LEDParams[]): Record<number, number> {
  const r2 = (v: number) => Math.round(v * 100) / 100;
  const xs = [...new Set(leds.map((l) => r2(l.position[0])))].sort((a, b) => a - b);
  const ys = [...new Set(leds.map((l) => r2(l.position[1])))].sort((a, b) => a - b);
  const out: Record<number, number> = {};
  if (xs.length * ys.length === leds.length) {
    leds.forEach((l) => {
      const ix = xs.indexOf(r2(l.position[0]));
      const iy = ys.indexOf(r2(l.position[1]));
      out[l.id] = 1 + (ix % 2) + 2 * (iy % 2);
    });
    return out;
  }
  const cx = leds.reduce((s, l) => s + l.position[0], 0) / leds.length;
  const cy = leds.reduce((s, l) => s + l.position[1], 0) / leds.length;
  const order = [...leds].sort(
    (a, b) => Math.hypot(a.position[0] - cx, a.position[1] - cy) - Math.hypot(b.position[0] - cx, b.position[1] - cy)
  );
  for (const l of order) {
    let best = 1, bestD = -1;
    for (let c = 1; c <= 4; c++) {
      const members = Object.entries(out).filter(([, v]) => v === c).map(([k]) => leds.find((m) => m.id === Number(k))!);
      const d = members.length ? Math.min(...members.map((m) => Math.hypot(m.position[0] - l.position[0], m.position[1] - l.position[1]))) : Infinity;
      if (d > bestD + 1e-9) { best = c; bestD = d; }
    }
    out[l.id] = best;
  }
  return out;
}

/** n x n LEDs at `spacing`, centred in a room that grows to fit (keeps the template's other settings). */
export function gridLayout(n: number, spacing: number, room: RoomParams, template: LEDParams):
  { leds: LEDParams[]; room: RoomParams } {
  const side = Math.max(room.width, room.length, n * spacing + 1.0);
  const newRoom = { ...room, width: side, length: side };
  const leds: LEDParams[] = [];
  let id = 1;
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const x = side / 2 + (i - (n - 1) / 2) * spacing;
      const y = side / 2 + (j - (n - 1) / 2) * spacing;
      leds.push({ ...template, id, position: [Math.round(x * 100) / 100, Math.round(y * 100) / 100, template.position[2]] });
      id++;
    }
  }
  return { leds, room: newRoom };
}
