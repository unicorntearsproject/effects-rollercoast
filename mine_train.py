"""Mine Train — the runaway cart, underground.

Same real 3-D track curve and on-rail camera as the rest of the pack, enclosed.
Instead of a sky there is rock: timber hoops ride the track at intervals, drawn
as screen-space rings whose radius falls off with depth, so the tunnel closes
around the rider and opens out again on the straights.  Torches burn at
intervals along the wall and light the rails as they pass.

The cart is genuinely running away — the profile is all short, sharp drops and
tight recoveries rather than one big hill, and the speed integrates off the
gradient, so it scrabbles uphill and bolts on every pitch down.

Audio reactivity:
  - bass    -> speed, cart shake, torch flare
  - mid     -> palette drift, rock tint
  - treble  -> rail sparks, dust motes
  - beat    -> spark burst off the rails, flash

Randomised per activation: palette hue, position on the circuit, track seed,
rock variant, and ``self.time``.

Config keys (under ``[effects.MineTrain]``):
    speed, zoom, reactivity, glitter, torch_gain
"""
from __future__ import annotations

import moderngl
import numpy as np

from unicornviz.effects.base import AudioData, BaseEffect

__version__ = '0.1.0'

_VERT = """
#version 330
in vec2 in_vert;
out vec2 v_uv;
void main() {
    v_uv = in_vert * 0.5 + 0.5;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""

_FRAG = """
#version 330
uniform float iTime;
uniform vec2  iResolution;
uniform float iBass;
uniform float iMid;
uniform float iTreble;
uniform float iBeat;
uniform float iPunch;
uniform float iHue;
uniform float iZoom;
uniform float iGlitter;
uniform float iTravel;     // CPU-integrated distance along the ride
uniform float iSeed;       // per-activation track shape
uniform float iSky;        // per-activation sky variant
uniform float iTorchGain;  // tweakable: how hot the wall torches burn
uniform float iGrade;      // current track gradient (drives the horizon)
in  vec2 v_uv;
out vec4 fragColor;

#define PI  3.14159265
#define TAU 6.28318530
#define STEPS 56
#define MIDWAY vec3(0.35, 0.85, 1.00)

float hash21(vec2 p) {
    p = mod(p, 512.0);
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
vec3 palette(float t) { return 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.33, 0.67))); }

// ── the ride profile ────────────────────────────────────────────────────────
// One circuit is CIRCUIT metres: a chain lift, a crest, the drop, then a run
// of camelbacks that decay back to station height.
const float CIRCUIT = 620.0;

float rideHeight(float s) {
    float u = mod(s, CIRCUIT) / CIRCUIT;
    // Runaway profile: a short haul up, then a staircase of sharp drops with
    // scrambling recoveries -- no single big hill to coast off.
    float haul = smoothstep(0.0, 0.16, u) * 26.0;
    float stair = 0.0;
    for (int i = 0; i < 4; i++) {
        float c = 0.20 + float(i) * 0.19;
        stair -= smoothstep(c, c + 0.07, u) * (7.0 - float(i) * 1.1);
        stair += smoothstep(c + 0.07, c + 0.15, u) * (4.4 - float(i) * 0.7);
    }
    return 9.0 + haul + stair + sin(u * TAU * 9.0 + iSeed * 3.0) * 1.6;
}

// Lateral wander: a couple of bounded sines, seeded per activation.
float rideLateral(float s) {
    float a = sin(s * 0.0210 + iSeed * 6.3) * 7.5;
    float b = sin(s * 0.0104 + iSeed * 2.1 + 1.3) * 9.0;
    return a + b;
}

vec3 trackPoint(float s) { return vec3(rideLateral(s), rideHeight(s), s); }

// ── camera ──────────────────────────────────────────────────────────────────
struct Cam { vec3 pos; vec3 right; vec3 up; vec3 fwd; };

Cam rideCam(float s, float shake) {
    Cam c;
    vec3 here = trackPoint(s);
    vec3 tangent = normalize(trackPoint(s + 1.0) - trackPoint(s - 1.0));
    // Look where the track is going, not just where it points right now.
    c.fwd = normalize(mix(tangent, normalize(trackPoint(s + 20.0) - here), 0.55));
    // Bank into the turn: roll from how fast the lateral offset is changing.
    float dl = (rideLateral(s + 6.0) - rideLateral(s - 6.0)) / 12.0;
    float roll = clamp(dl * 1.8, -1.0, 1.0) * 0.42;
    vec3 worldUp = vec3(sin(roll), cos(roll), 0.0);
    c.right = normalize(cross(c.fwd, worldUp));
    c.up = cross(c.right, c.fwd);
    // Seat is above the rails; shake rides the bass and the beat.
    c.pos = here + c.up * 1.9
          + c.right * shake * 0.25 + c.up * shake * 0.18;
    return c;
}

bool project(Cam c, vec3 P, out vec2 sp, out float depth) {
    vec3 q = P - c.pos;
    float z = dot(q, c.fwd);
    if (z < 0.35) return false;
    sp = vec2(dot(q, c.right), dot(q, c.up)) / z;
    depth = z;
    return true;
}

float segDist(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h);
}

// ── sky + ground ────────────────────────────────────────────────────────────
vec3 skyColour(vec2 p, float tb) {
    float up = clamp(p.y * 0.9 + 0.35 - iGrade * 0.35, 0.0, 1.0);
    // Rock, not sky: dark stone that closes in toward the edges of frame.
    vec3 rock = iSky < 0.34 ? vec3(0.055, 0.040, 0.032)
              : iSky < 0.67 ? vec3(0.040, 0.045, 0.055)
                            : vec3(0.050, 0.036, 0.050);
    rock = mix(rock, palette(iHue + 0.55) * 0.10, 0.35 + iMid * 0.15);
    // Strata, so the walls have grain rather than being flat fill.
    float grain = hash21(floor(vec2(p.x * 46.0, p.y * 30.0)));
    rock *= 0.72 + grain * 0.55;
    rock += palette(iHue + 0.2) * 0.05
          * smoothstep(0.0, 1.0, sin(p.y * 26.0 + p.x * 6.0) * 0.5 + 0.5);
    // The bore darkens away from its centre, and that centre rides the track
    // gradient (`up` carries iGrade): pitched down you see more floor, pitched
    // up more roof, which is what sells the cart tipping over each drop.
    vec2 bore = p - vec2(0.0, (up - 0.5) * 0.55);
    rock *= mix(1.0, 0.22, smoothstep(0.15, 1.0, length(bore)));
    rock += palette(iHue + 0.35) * 0.03 * smoothstep(0.9, 0.25, length(bore));
    return rock;
}

void main() {
    float A = iResolution.x / max(iResolution.y, 1.0);
    float tb = mod(iTime, 300.0);
    float shake = (iBass * 0.5 + iPunch * 0.8);
    vec2 p = (v_uv - 0.5) * vec2(A, 1.0) / max(iZoom, 0.05);
    p += vec2(sin(tb * 47.0), sin(tb * 39.0 + 1.1)) * shake * 0.006;

    Cam cam = rideCam(iTravel, shake);
    vec3 col = skyColour(p, tb);

    // Real ground plane: intersect the view ray with y=0 so the park has
    // actual perspective under the ride instead of a flat wash of haze.
    vec3 rd = normalize(cam.fwd + p.x * cam.right + p.y * cam.up);
    if (rd.y < -0.0015 && cam.pos.y > 0.2) {
        float td = -cam.pos.y / rd.y;
        if (td > 0.0 && td < 900.0) {
            vec3 g = cam.pos + rd * td;
            vec2 cell = abs(fract(g.xz * 0.06) - 0.5);
            float line = smoothstep(0.045, 0.0, min(cell.x, cell.y));
            vec3 gnd = mix(vec3(0.030, 0.035, 0.055), palette(iHue + 0.55), 0.25);
            gnd += mix(MIDWAY, palette(iHue + 0.2), 0.35) * line * (0.35 + iBass * 0.5);
            // Midway lights scattered across the park.
            float lamp = hash21(floor(g.xz * 0.09));
            gnd += mix(vec3(1.0, 0.80, 0.45), palette(iHue + 0.1), 0.3)
                 * step(0.972, lamp)
                 * smoothstep(0.42, 0.0, length(fract(g.xz * 0.09) - 0.5))
                 * (0.8 + iTreble * 1.2);
            col = mix(col, gnd, exp(-td * 0.0045));
        }
    }

    // ── march the track far -> near so crests occlude what is beyond ────────
    vec3 railCol = mix(vec3(0.85, 0.90, 1.00), palette(iHue + 0.15), 0.45);
    vec3 tieCol  = mix(vec3(0.35, 0.22, 0.14), palette(iHue + 0.6), 0.30);
    const float NEAR = 2.0, FAR = 300.0, GAUGE = 1.5;

    for (int i = STEPS - 1; i >= 0; i--) {
        float f0 = float(i) / float(STEPS);
        float f1 = float(i + 1) / float(STEPS);
        float s0 = iTravel + NEAR + f0 * f0 * FAR;
        float s1 = iTravel + NEAR + f1 * f1 * FAR;
        vec3 c0 = trackPoint(s0), c1 = trackPoint(s1);

        // Bank the rail pair with the local roll so the gauge tilts in turns.
        float dl = (rideLateral(s0 + 6.0) - rideLateral(s0 - 6.0)) / 12.0;
        float roll = clamp(dl * 1.8, -1.0, 1.0) * 0.42;
        vec3 lat = vec3(cos(roll), sin(roll), 0.0);
        vec3 nrm = vec3(-sin(roll), cos(roll), 0.0);

        float fade = 1.0 - f0;
        for (int r = -1; r <= 1; r += 2) {
            vec2 a, b; float da, db;
            if (!project(cam, c0 + lat * (GAUGE * float(r)), a, da)) continue;
            if (!project(cam, c1 + lat * (GAUGE * float(r)), b, db)) continue;
            float w = clamp(0.16 / da, 0.0015, 0.05);
            float d = segDist(p, a, b);
            float m = smoothstep(w, w * 0.35, d);
            vec3 c = railCol * (0.45 + fade * 0.9 + iBass * 0.35);
            col = mix(col, c, m);
            // Hot specular running along the rail head.
            col += railCol * smoothstep(w * 0.5, 0.0, d) * (0.35 + iTreble * 1.1) * fade;
        }
        // Sleepers every few metres.
        if (mod(floor(s0 * 0.5), 1.0) < 0.5) {
            vec2 a, b; float da, db;
            if (project(cam, c0 - lat * GAUGE * 1.25, a, da)
                && project(cam, c0 + lat * GAUGE * 1.25, b, db)) {
                float w = clamp(0.10 / da, 0.001, 0.03);
                col = mix(col, tieCol * (0.35 + fade * 0.7),
                          smoothstep(w, w * 0.4, segDist(p, a, b)));
            }
        }
        // Timber hoops every few metres: a screen-space ring around the
        // projected centreline, radius falling off with depth, which is what
        // closes the bore around the rider.
        if (mod(floor(s0 * 0.1), 1.0) < 0.5) {
            vec2 hc; float hd;
            if (project(cam, c0, hc, hd)) {
                float rr = 7.5 / hd;
                float ring = smoothstep(0.055 / hd, 0.0, abs(length(p - hc) - rr));
                vec3 timber = mix(vec3(0.30, 0.19, 0.11), palette(iHue + 0.5), 0.25);
                col = mix(col, timber * (0.35 + fade * 0.8), ring * 0.9);
            }
        }
        // Wall torches, alternating sides, lighting the rock as they pass.
        if (mod(floor(s0 * 0.05), 1.0) < 0.5) {
            float side = (mod(floor(s0 * 0.05), 2.0) < 1.0) ? -1.0 : 1.0;
            vec2 tc; float td2;
            if (project(cam, c0 + lat * (5.6 * side) + nrm * 2.6, tc, td2)) {
                float flick = 0.72 + 0.28 * hash21(floor(vec2(tb * 26.0, s0)));
                vec3 flame = mix(vec3(1.0, 0.55, 0.16), palette(iHue + 0.1), 0.25);
                // Tight core plus a small halo, and both fall off with depth.
                // Scaling the *radius* by depth instead made every distant
                // torch a screen-wide wash, and fifteen of them stacked the
                // whole bore to white.
                float dd = length(p - tc);
                col += flame * (exp(-dd * dd * 900.0) * 1.5 + exp(-dd * 4.5) * 0.16)
                     * flick * (0.55 + iBass * 0.75) * iTorchGain * fade;
            }
        }
        // Support columns dropping to the ground.
        if (mod(floor(s0 * 0.125), 1.0) < 0.5 && c0.y > 3.0) {
            vec2 a, b; float da, db;
            if (project(cam, c0 - nrm * 1.0, a, da)
                && project(cam, vec3(c0.x, 0.0, c0.z), b, db)) {
                float w = clamp(0.09 / da, 0.001, 0.03);
                col = mix(col, mix(vec3(0.18, 0.20, 0.26), palette(iHue + 0.4), 0.3)
                               * (0.3 + fade * 0.6),
                          smoothstep(w, w * 0.4, segDist(p, a, b)) * 0.85);
            }
        }
    }

    // Sparks thrown off the rails, plus dust hanging in the torchlight.
    for (int i = 0; i < 10; i++) {
        float fi = float(i);
        float ph = fract(iTravel * 0.05 + fi * 0.161);
        float sa = hash21(vec2(fi, floor(iTravel * 0.05 + fi * 0.161))) * TAU;
        vec2 sp2 = vec2(cos(sa), sin(sa)) * (0.05 + ph * 0.55) + vec2(0.0, -0.12);
        col += mix(vec3(1.0, 0.78, 0.30), palette(iHue + fi * 0.07), 0.3)
             * exp(-dot(p - sp2, p - sp2) * 2600.0) * (1.0 - ph)
             * (0.4 + iPunch * 2.2 + iTreble * 0.9) * iGlitter;
    }
    float dust = hash21(floor(vec2(p.x * 70.0, p.y * 70.0 + iTravel * 0.4)));
    col += vec3(0.9, 0.80, 0.62) * step(0.9955, dust)
         * (0.25 + iTreble * 0.7) * iGlitter;

    col += palette(iHue + 0.5) * iBeat * 0.12;
    col *= smoothstep(1.5, 0.22, length(v_uv - 0.5));
    col = col / (1.0 + col * 0.55);
    col = pow(clamp(col, 0.0, 1.0), vec3(0.85));
    fragColor = vec4(col, 1.0);
}
"""


class MineTrain(BaseEffect):
    """First-person runaway mine cart through a torch-lit bore."""

    NAME = 'Mine Train'
    AUTHOR = 'unicorn-viz'
    TAGS = ['rollercoast', 'ride', 'neon', 'intense', 'energetic', 'hard']
    PING_PONG_FRIENDS = [
        'First Drop', 'Corkscrew', 'Night Coaster', 'Coaster Cam',
        'Warp Drive', 'Canyon Run', 'Audio Spectrum',
    ]

    _CIRCUIT = 620.0

    def _init(self) -> None:
        self.parameters = {
            'speed': float(self.config.get('speed', 1.0)),
            'zoom': float(self.config.get('zoom', 1.0)),
            'reactivity': float(self.config.get('reactivity', 1.3)),
            'glitter': float(self.config.get('glitter', 1.0)),
            'torch_gain': float(self.config.get('torch_gain', 1.0)),
        }
        self._prog = self._make_program(_VERT, _FRAG)
        self._vao, self._vbo = self._fullscreen_quad(self._prog)

        self._bass = self._mid = self._treble = 0.0
        self._beat = 0.0
        self._punch = 0.0
        self._hue = float(self.rng.uniform(0.0, 1.0))
        self._seed = float(self.rng.uniform(0.0, 1.0))
        self._sky = float(self.rng.uniform(0.0, 1.0))
        self._travel = float(self.rng.uniform(0.0, self._CIRCUIT))
        self._grade = 0.0

    def _height(self, s: float) -> float:
        """Mirror of the shader's rideHeight, so the CPU can read the gradient."""
        u = (s % self._CIRCUIT) / self._CIRCUIT
        if u < 0.34:
            return _smoothstep(0.0, 0.34, u) * 62.0
        if u < 0.40:
            return 62.0 - _smoothstep(0.34, 0.40, u) * 3.0
        if u < 0.52:
            d = _smoothstep(0.40, 0.52, u)
            return 59.0 - d * d * 55.0
        d = (u - 0.52) / 0.48
        return 4.0 + (1.0 - d) * 22.0 * (0.5 + 0.5 * float(np.cos(d * 14.0)))

    def update(self, dt: float, audio: AudioData) -> None:
        """Advance the train along the ride and track the audio envelopes."""
        super().update(dt, audio)
        react = max(0.05, float(self.parameters['reactivity']))
        self._bass = float(np.clip(audio.bass_n * react, 0.0, 2.0))
        self._mid = float(np.clip(audio.mid_n * react, 0.0, 2.0))
        self._treble = float(np.clip(audio.treble_n * react, 0.0, 2.0))
        if audio.beat > 0.5:
            self._beat = 1.0
            self._punch = 1.0
        self._beat = max(0.0, self._beat - dt * 5.0)
        self._punch = max(0.0, self._punch - dt * 2.2)

        # Gradient of the ride at the train, used both to speed it up going
        # downhill and to tilt the horizon in the shader.
        ahead = 4.0
        grade = (self._height(self._travel + ahead) - self._height(self._travel - ahead)) / (2 * ahead)
        self._grade = float(np.clip(grade, -1.5, 1.5))

        speed = max(0.05, float(self.parameters['speed']))
        # Downhill accelerates, uphill drags; audio rides on top of that rather
        # than replacing it, so the ride keeps its own physical rhythm.
        base = 26.0 - self._grade * 30.0
        self._travel = (self._travel
                        + dt * speed * max(6.0, base) * (0.75 + self._bass * 0.7)
                        ) % (self._CIRCUIT * 64.0)

    def render(self) -> None:
        """Upload ride state and draw the view from the front car."""
        p = self._prog
        p['iTime'].value = float(self.time)
        p['iResolution'].value = (float(self.width), float(self.height))
        p['iBass'].value = self._bass
        p['iMid'].value = self._mid
        p['iTreble'].value = self._treble
        p['iBeat'].value = self._beat
        p['iPunch'].value = self._punch
        p['iHue'].value = self._hue
        p['iZoom'].value = max(0.05, float(self.parameters['zoom']))
        p['iGlitter'].value = max(0.0, float(self.parameters['glitter']))
        p['iTravel'].value = float(self._travel)
        p['iSeed'].value = self._seed
        p['iSky'].value = self._sky
        p['iTorchGain'].value = max(0.0, float(self.parameters['torch_gain']))
        p['iGrade'].value = float(self._grade)
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        """Release GL resources."""
        self._vao.release()
        self._vbo.release()
        self._prog.release()


def _smoothstep(a: float, b: float, x: float) -> float:
    t = float(np.clip((x - a) / max(b - a, 1e-6), 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)
