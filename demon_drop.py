"""Demon Drop — the Cedar Point freefall tower, ridden facing out.

A freefall tower is not a hill, and the difference is structural: the track has
to go *vertical*.  A plain parametric loop cannot do that, because a track
whose tangent is exactly the world up-vector has no lateral direction and the
ride frame collapses to NaN.  So the plan shape here is **phase-warped**: the
car's progress around the oval very nearly stops for the whole tower section,
drifting just enough (about six metres of run against forty of climb) to keep
the frame well conditioned while reading as a vertical hoist.  Everything else
about the circuit is the pack's shared closed-loop model, so the car laps
forever inside a bounded footprint.

The cycle is the real one: roll out of the station, hoist up the mast, **hang**
at the top -- the car crawls here, which is the whole point of the ride -- then
freefall, a curving pull-out onto the brake track, and the return run home.
Speed away from the hoist and the hang comes from energy conservation, so the
plunge accelerates because it is falling, not because a curve says so.

The mast is a real lattice: four legs from grade to crown, tied at every bay
with horizontals and X-bracing, standing in place while the car runs the rest
of the circuit around it.

Audio reactivity:
  - bass    -> speed, camera shake, mast glow
  - mid     -> palette drift, sky banding
  - treble  -> beacon sparkle, air rush
  - beat    -> flash, strobe on the crown beacons

Randomised per activation: palette hue, position in the cycle, plan seed, sky
variant, and ``self.time``.

Config keys (under ``[effects.DemonDrop]``):
    speed, zoom, reactivity, glitter, hang_tension
"""
from __future__ import annotations

import math

import moderngl
import numpy as np

from unicornviz.effects.base import AudioData, BaseEffect

__version__ = '0.1.0'

CIRCUIT = 620.0
TOWER_TOP = 42.0
GRAVITY = 9.81

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
uniform float iTravel;       // position in the cycle, always within [0, CIRCUIT)
uniform float iSeed;         // per-activation plan shape
uniform float iSky;          // per-activation sky variant
uniform float iGrade;        // current track gradient, tilts the horizon
uniform float iHang;         // 1 while the car hangs at the crown, else 0
uniform float iHangTension;  // tweakable: how hard the hang is sold
in  vec2 v_uv;
out vec4 fragColor;

#define PI  3.14159265
#define TAU 6.28318530
#define STEPS 44
#define CIRCUIT 620.0
#define LAP_R 120.0

float hash21(vec2 p) {
    p = mod(p, 512.0);
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
vec3 palette(float t) { return 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.33, 0.67))); }

// ── the cycle ───────────────────────────────────────────────────────────────
// Plan progress around the oval. Almost frozen through the tower so the car
// climbs, hangs and falls in one place; the small residual drift is what keeps
// the ride frame from degenerating on a truly vertical tangent.
float planPhase(float u) {
    if (u < 0.16) return u / 0.16 * 0.26;             // roll out of the station
    if (u < 0.62) return 0.26 + (u - 0.16) / 0.46 * 0.012;   // hoist, hang, fall
    if (u < 0.72) return 0.272 + (u - 0.62) / 0.10 * 0.068;  // curving pull-out
    return 0.34 + (u - 0.72) / 0.28 * 0.66;                  // brake run home
}

float rideHeight(float s) {
    float u = fract(s / CIRCUIT);
    if (u < 0.10) return 2.5;                                       // station
    if (u < 0.16) return 2.5 + smoothstep(0.10, 0.16, u) * 1.5;     // approach
    if (u < 0.44) return 4.0 + smoothstep(0.16, 0.44, u) * 38.0;    // the hoist
    if (u < 0.52) return 42.0;                                      // the hang
    if (u < 0.62) {                                                 // freefall
        float d = smoothstep(0.52, 0.62, u);
        return 42.0 - d * d * 30.0;
    }
    if (u < 0.72) return 12.0 - smoothstep(0.62, 0.72, u) * 9.0;    // pull-out
    return mix(3.0, 2.5, smoothstep(0.72, 1.0, u));                 // brake run
}

vec3 trackPoint(float s) {
    float u = fract(s / CIRCUIT);
    float th = planPhase(u) * TAU;
    float r = LAP_R * (1.0 + 0.17 * sin(th * 2.0 + iSeed * 6.3)
                           + 0.07 * sin(th * 3.0 + 2.4));
    return vec3(r * sin(th), rideHeight(s), r * cos(th));
}

float trackRoll(float s) {
    vec3 a = trackPoint(s - 9.0), b = trackPoint(s), c = trackPoint(s + 9.0);
    vec2 d1 = normalize(b.xz - a.xz + vec2(1e-5)), d2 = normalize(c.xz - b.xz + vec2(1e-5));
    float turn = d1.x * d2.y - d1.y * d2.x;
    return clamp(turn * 9.0, -1.0, 1.0) * 0.5;
}

struct Cam { vec3 pos; vec3 right; vec3 up; vec3 fwd; };

Cam rideCam(float s, float shake) {
    Cam c;
    vec3 here = trackPoint(s);
    vec3 tangent = normalize(trackPoint(s + 1.5) - trackPoint(s - 1.5));
    c.fwd = normalize(mix(tangent, normalize(trackPoint(s + 20.0) - here), 0.45));
    // A freefall rider faces *out*, not up: on the hoist and through the hang
    // the car is tipped forward over the park. Without this the camera stares
    // at empty sky for the whole climb and the hang frame is nearly blank.
    if (c.fwd.y > 0.10) {
        vec3 outward = normalize(vec3(c.fwd.x, 0.0, c.fwd.z) + vec3(1e-4, 0.0, 1e-4));
        c.fwd = normalize(mix(c.fwd, outward, smoothstep(0.10, 0.75, c.fwd.y)));
    }
    // Over the hang the car tips forward, so the rider spends those seconds
    // looking straight down the shaft they are about to fall down.
    c.fwd = normalize(c.fwd - vec3(0.0, clamp(iHang, 0.0, 1.0) * 0.55, 0.0));
    float roll = trackRoll(s);
    vec3 worldUp = vec3(sin(roll), cos(roll), 0.0);
    c.right = normalize(cross(c.fwd, worldUp));
    c.up = cross(c.right, c.fwd);
    c.pos = here + c.up * 1.8 + c.right * shake * 0.22 + c.up * shake * 0.16;
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
void strut(inout vec3 col, Cam cam, vec2 p, vec3 A, vec3 B,
           float radius, vec3 c, float glow) {
    vec2 a, b; float da, db;
    if (!project(cam, A, a, da)) return;
    if (!project(cam, B, b, db)) return;
    float w = clamp(radius / min(da, db), 0.0007, 0.09);
    float d = segDist(p, a, b);
    col = mix(col, c, smoothstep(w, w * 0.45, d));
    if (glow > 0.0) col += c * smoothstep(w * 0.55, 0.0, d) * glow;
}
void frameAt(float s, out vec3 lat, out vec3 nrm) {
    float roll = trackRoll(s);
    vec3 tangent = normalize(trackPoint(s + 1.5) - trackPoint(s - 1.5));
    vec3 worldUp = vec3(sin(roll), cos(roll), 0.0);
    lat = normalize(cross(tangent, worldUp));
    nrm = cross(lat, tangent);
}

vec3 skyColour(vec2 p, float tb) {
    float up = clamp(p.y * 0.9 + 0.35 - iGrade * 0.35, 0.0, 1.0);
    vec3 lo, hi;
    if (iSky < 0.34)      { lo = vec3(0.98, 0.55, 0.30); hi = vec3(0.06, 0.09, 0.30); }
    else if (iSky < 0.67) { lo = vec3(0.70, 0.85, 0.95); hi = vec3(0.05, 0.22, 0.52); }
    else                  { lo = vec3(0.85, 0.25, 0.45); hi = vec3(0.04, 0.04, 0.18); }
    vec3 col = mix(lo, hi, pow(up, 0.8));
    col = mix(col, palette(iHue + up * 0.3), 0.20 + iMid * 0.12);
    float band = sin(p.y * 18.0 - tb * 0.2 + sin(p.x * 2.4) * 1.1);
    col += vec3(1.0, 0.9, 0.8) * smoothstep(0.92, 1.0, band)
         * (0.08 + iMid * 0.18) * (1.0 - up);
    return col;
}

// The mast: four legs, bay ties and X-bracing, standing where the tower is.
void drawMast(inout vec3 col, Cam cam, vec2 p, vec3 legCol) {
    vec3 anchor = trackPoint(CIRCUIT * 0.39);   // centred on the vertical run
    vec2 c = anchor.xz;
    float half_ = 5.2;
    vec2 o[4];
    o[0] = vec2(-half_, -half_); o[1] = vec2(half_, -half_);
    o[2] = vec2(half_, half_);   o[3] = vec2(-half_, half_);
    for (int i = 0; i < 4; i++) {
        vec3 a = vec3(c.x + o[i].x, 0.0, c.y + o[i].y);
        vec3 b = vec3(c.x + o[i].x, 47.0, c.y + o[i].y);
        strut(col, cam, p, a, b, 0.30, legCol, 0.0);
    }
    for (int bay = 0; bay < 8; bay++) {
        float y0 = float(bay) * 6.0, y1 = y0 + 6.0;
        for (int i = 0; i < 4; i++) {
            int j = int(mod(float(i) + 1.0, 4.0));
            vec3 a0 = vec3(c.x + o[i].x, y1, c.y + o[i].y);
            vec3 b0 = vec3(c.x + o[j].x, y1, c.y + o[j].y);
            vec3 a1 = vec3(c.x + o[i].x, y0, c.y + o[i].y);
            vec3 b1 = vec3(c.x + o[j].x, y0, c.y + o[j].y);
            strut(col, cam, p, a0, b0, 0.16, legCol * 0.92, 0.0);
            strut(col, cam, p, a1, b0, 0.12, legCol * 0.80, 0.0);
        }
    }
    // Crown beacons, strobing on the beat the way aircraft warning lights do.
    for (int i = 0; i < 4; i++) {
        vec3 b = vec3(c.x + o[i].x, 47.6, c.y + o[i].y);
        vec2 sp; float dz;
        if (project(cam, b, sp, dz)) {
            float d = length(p - sp);
            col += mix(vec3(1.0, 0.30, 0.22), palette(iHue + 0.1), 0.3)
                 * (exp(-d * d * 1400.0) * 1.4 + exp(-d * 5.0) * 0.10)
                 * (0.5 + iBeat * 1.6 + iTreble * 0.5);
        }
    }
}

void main() {
    float A = iResolution.x / max(iResolution.y, 1.0);
    float tb = mod(iTime, 300.0);
    float shake = iBass * 0.5 + iPunch * 0.9;
    vec2 p = (v_uv - 0.5) * vec2(A, 1.0) / max(iZoom, 0.05);
    p += vec2(sin(tb * 51.0), sin(tb * 43.0 + 1.3)) * shake * 0.005;

    Cam cam = rideCam(iTravel, shake);
    vec3 col = skyColour(p, tb);

    vec3 rd = normalize(cam.fwd + p.x * cam.right + p.y * cam.up);
    if (rd.y < -0.0015 && cam.pos.y > 0.2) {
        float td = -cam.pos.y / rd.y;
        if (td > 0.0 && td < 900.0) {
            vec3 g = cam.pos + rd * td;
            vec2 cell = abs(fract(g.xz * 0.05) - 0.5);
            float line = smoothstep(0.04, 0.0, min(cell.x, cell.y));
            vec3 gnd = mix(vec3(0.035, 0.040, 0.055), palette(iHue + 0.55), 0.22);
            gnd += mix(vec3(0.40, 0.85, 1.0), palette(iHue + 0.2), 0.35)
                 * line * (0.30 + iBass * 0.5);
            float lamp = hash21(floor(g.xz * 0.09));
            gnd += mix(vec3(1.0, 0.82, 0.45), palette(iHue + 0.1), 0.3)
                 * step(0.968, lamp)
                 * smoothstep(0.42, 0.0, length(fract(g.xz * 0.09) - 0.5))
                 * (0.8 + iTreble * 1.1);
            col = mix(col, gnd, exp(-td * 0.0042));
        }
    }

    vec3 legCol  = mix(vec3(0.24, 0.25, 0.30), palette(iHue + 0.42), 0.32);
    vec3 railCol = mix(vec3(0.88, 0.92, 1.00), palette(iHue + 0.15), 0.42);
    vec3 tieCol  = mix(vec3(0.34, 0.30, 0.26), palette(iHue + 0.62), 0.26);
    drawMast(col, cam, p, legCol);

    const float NEAR = 2.0, FAR = 380.0, GAUGE = 1.5, SPINE = 1.9;
    for (int i = STEPS - 1; i >= 0; i--) {
        float f0 = float(i) / float(STEPS);
        float f1 = float(i + 1) / float(STEPS);
        float s0 = iTravel + NEAR + f0 * f0 * FAR;
        float s1 = iTravel + NEAR + f1 * f1 * FAR;
        vec3 c0 = trackPoint(s0), c1 = trackPoint(s1);
        vec3 lat0, nrm0, lat1, nrm1;
        frameAt(s0, lat0, nrm0);
        frameAt(s1, lat1, nrm1);
        float fade = 1.0 - f0 * 0.72;
        float glow = (0.28 + iTreble * 0.85) * fade;

        strut(col, cam, p, c0 - nrm0 * SPINE - lat0 * 0.45,
                          c1 - nrm1 * SPINE - lat1 * 0.45, 0.18, legCol * fade, 0.0);
        strut(col, cam, p, c0 - nrm0 * SPINE + lat0 * 0.45,
                          c1 - nrm1 * SPINE + lat1 * 0.45, 0.18, legCol * fade, 0.0);
        if (f0 < 0.40) {
            float zig = mod(floor(s0 * 0.5), 2.0) < 1.0 ? -1.0 : 1.0;
            strut(col, cam, p, c0 - nrm0 * SPINE + lat0 * (0.45 * zig),
                              c1 + lat1 * (GAUGE * zig), 0.10, legCol * 0.9 * fade, 0.0);
        }
        if (mod(floor(s0 * 0.6), 1.0) < 0.5) {
            strut(col, cam, p, c0 - lat0 * GAUGE * 1.2, c0 + lat0 * GAUGE * 1.2,
                  0.12, tieCol * (0.4 + fade * 0.7), 0.0);
        }
        for (int r = -1; r <= 1; r += 2) {
            strut(col, cam, p, c0 + lat0 * (GAUGE * float(r)),
                              c1 + lat1 * (GAUGE * float(r)),
                  0.16, railCol * (0.45 + fade * 0.85 + iBass * 0.3), glow);
        }
        // Brake-run bents, skipped through the tower where the mast carries it.
        float u0 = fract(s0 / CIRCUIT);
        if (mod(floor(s0 * 0.08), 1.0) < 0.5 && c0.y > 3.5
            && (u0 < 0.14 || u0 > 0.66)) {
            vec3 baseL = vec3((c0 - lat0 * 2.4).x, 0.0, (c0 - lat0 * 2.4).z);
            vec3 baseR = vec3((c0 + lat0 * 2.4).x, 0.0, (c0 + lat0 * 2.4).z);
            vec3 topL = c0 - nrm0 * SPINE - lat0 * 0.8;
            vec3 topR = c0 - nrm0 * SPINE + lat0 * 0.8;
            vec3 midL = mix(topL, baseL, 0.5), midR = mix(topR, baseR, 0.5);
            strut(col, cam, p, topL, baseL, 0.15, legCol * fade, 0.0);
            strut(col, cam, p, topR, baseR, 0.15, legCol * fade, 0.0);
            strut(col, cam, p, topL, midR, 0.09, legCol * 0.85 * fade, 0.0);
            strut(col, cam, p, topR, midL, 0.09, legCol * 0.85 * fade, 0.0);
            strut(col, cam, p, midL, midR, 0.09, legCol * 0.85 * fade, 0.0);
        }
    }

    // The hang: the whole frame holds its breath, then the floor drops out.
    float hang = clamp(iHang, 0.0, 1.0) * clamp(iHangTension, 0.0, 3.0);
    col *= 1.0 - hang * 0.30;
    col += palette(iHue + 0.05) * hang * 0.06
         * (0.5 + 0.5 * sin(tb * 9.0));

    // Air rush past the car once it is falling.
    float rad = length(p);
    float lane = floor(atan(p.y, p.x) / TAU * 84.0);
    float lh = hash21(vec2(lane, 3.0));
    float along = fract(rad * 1.7 - fract(lh + iTravel * 0.03) * 2.0);
    float across = abs(fract(atan(p.y, p.x) / TAU * 84.0) - 0.5) * 2.0;
    col += vec3(0.88, 0.94, 1.0)
         * smoothstep(0.24, 0.0, across)
         * smoothstep(0.55, 0.94, along) * smoothstep(1.0, 0.92, along)
         * step(0.5, lh) * smoothstep(0.18, 0.8, rad)
         * (0.25 + iTreble * 0.9 + iBass * 0.5) * iGlitter * (1.0 - hang * 0.8);

    col += palette(iHue + 0.5) * iBeat * 0.12;
    col *= smoothstep(1.5, 0.22, length(v_uv - 0.5));
    col = col / (1.0 + col * 0.55);
    col = pow(clamp(col, 0.0, 1.0), vec3(0.85));
    fragColor = vec4(col, 1.0);
}
"""


def _smoothstep(a: float, b: float, x: float) -> float:
    t = min(1.0, max(0.0, (x - a) / max(b - a, 1e-6)))
    return t * t * (3.0 - 2.0 * t)


def ride_height(s: float) -> float:
    """CPU mirror of the shader's rideHeight, for the physics integrator."""
    u = (s / CIRCUIT) % 1.0
    if u < 0.10:
        return 2.5
    if u < 0.16:
        return 2.5 + _smoothstep(0.10, 0.16, u) * 1.5
    if u < 0.44:
        return 4.0 + _smoothstep(0.16, 0.44, u) * 38.0
    if u < 0.52:
        return 42.0
    if u < 0.62:
        d = _smoothstep(0.52, 0.62, u)
        return 42.0 - d * d * 30.0
    if u < 0.72:
        return 12.0 - _smoothstep(0.62, 0.72, u) * 9.0
    return 3.0 + (2.5 - 3.0) * _smoothstep(0.72, 1.0, u)


class DemonDrop(BaseEffect):
    """First-person freefall tower: hoist, hang, plunge, brake, repeat."""

    NAME = 'Demon Drop'
    AUTHOR = 'unicorn-viz'
    TAGS = ['rollercoast', 'ride', 'neon', 'intense', 'energetic', 'hard']
    PING_PONG_FRIENDS = [
        'First Drop', 'Corkscrew', 'Night Coaster', 'Mine Train',
        'Coaster Cam', 'Log Flume', 'Audio Spectrum',
    ]

    _HOIST_SPEED = 6.0       # m/s, cable speed up the mast
    _HANG_SPEED = 1.1        # m/s, the crawl at the crown
    _MIN_SPEED = 5.0
    _BRAKE_SPEED = 13.0      # m/s ceiling once the brake fins have it

    def _init(self) -> None:
        self.parameters = {
            'speed': float(self.config.get('speed', 1.0)),
            'zoom': float(self.config.get('zoom', 1.0)),
            'reactivity': float(self.config.get('reactivity', 1.3)),
            'glitter': float(self.config.get('glitter', 1.0)),
            'hang_tension': float(self.config.get('hang_tension', 1.0)),
        }
        self._prog = self._make_program(_VERT, _FRAG)
        self._vao, self._vbo = self._fullscreen_quad(self._prog)

        self._bass = self._mid = self._treble = 0.0
        self._beat = 0.0
        self._punch = 0.0
        self._hue = float(self.rng.uniform(0.0, 1.0))
        self._seed = float(self.rng.uniform(0.0, 1.0))
        self._sky = float(self.rng.uniform(0.0, 1.0))
        self._travel = float(self.rng.uniform(0.0, CIRCUIT))
        self._grade = 0.0
        self._hang = 0.0

    def update(self, dt: float, audio: AudioData) -> None:
        """Run the cycle: cable on the hoist, gravity everywhere else."""
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

        u = (self._travel / CIRCUIT) % 1.0
        h = ride_height(self._travel)
        ahead = 5.0
        self._grade = float(np.clip(
            (ride_height(self._travel + ahead) - ride_height(self._travel - ahead))
            / (2 * ahead), -1.5, 1.5))

        if 0.16 <= u < 0.44:
            v = self._HOIST_SPEED
            self._hang = 0.0
        elif 0.44 <= u < 0.52:
            v = self._HANG_SPEED
            self._hang = 1.0
        else:
            v = math.sqrt(max(0.0, 2.0 * GRAVITY * max(0.0, TOWER_TOP - h)))
            v = max(self._MIN_SPEED, v)
            if u >= 0.72 or u < 0.10:
                v = min(v, self._BRAKE_SPEED)
            self._hang = 0.0

        speed = max(0.05, float(self.parameters['speed']))
        self._travel = (self._travel
                        + dt * speed * v * (0.8 + self._bass * 0.5)) % CIRCUIT

    def render(self) -> None:
        """Upload cycle state and draw the view from the falling car."""
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
        p['iGrade'].value = float(self._grade)
        p['iHang'].value = float(self._hang)
        p['iHangTension'].value = max(0.0, float(self.parameters['hang_tension']))
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        """Release GL resources."""
        self._vao.release()
        self._vbo.release()
        self._prog.release()
