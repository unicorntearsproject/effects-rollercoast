"""Log Flume — a water circuit, ridden from the front seat of the log.

The other rides in this pack are built from lines: rails, chords, bracing.  A
flume is built from *surfaces*, so this one adds a filled-quad primitive on top
of the shared closed-circuit model.  Each marched section of the circuit paints
a real trough -- two timber side walls and the water between them -- as screen
-space quads with a proper triangle-coverage test, which is what makes the
channel read as something you are floating *inside* rather than a line you are
following.  The trough stands on timber trestle bents with X-bracing and sill
beams, the way a real flume's runs are carried.

The water moves.  Its shimmer is driven along the channel by the log's own
travel, so the surface streams past at the speed the ride is actually doing,
and the splashdown throws a burst of spray that decays over the run-out.

Two conveyors, a baby drop and the big splashdown.  Away from the conveyors the
log runs on energy conservation like the coasters, but with a water-drag
ceiling on the run-out -- a flume that kept its splashdown speed all the way
home would look wrong, and a real one does not.

Audio reactivity:
  - bass    -> speed, hull rock, water body glow
  - mid     -> palette drift, sky and timber tint
  - treble  -> surface sparkle, spray density
  - beat    -> splash burst, flash

Randomised per activation: palette hue, position on the circuit, plan seed, sky
variant, and ``self.time``.

Config keys (under ``[effects.LogFlume]``):
    speed, zoom, reactivity, glitter, splash_gain
"""
from __future__ import annotations

import math

import moderngl
import numpy as np

from unicornviz.effects.base import AudioData, BaseEffect

__version__ = '0.1.0'

CIRCUIT = 780.0
H_TOP = 32.0
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
uniform float iTravel;      // position on the lap, always within [0, CIRCUIT)
uniform float iSeed;        // per-activation plan shape
uniform float iSky;         // per-activation sky variant
uniform float iGrade;       // current track gradient, tilts the horizon
uniform float iSplash;      // 1 at splashdown, decaying over the run-out
uniform float iSplashGain;  // tweakable: how much water gets thrown
in  vec2 v_uv;
out vec4 fragColor;

#define PI  3.14159265
#define TAU 6.28318530
#define STEPS 40
#define CIRCUIT 780.0
#define LAP_R 150.0

float hash21(vec2 p) {
    p = mod(p, 512.0);
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
vec3 palette(float t) { return 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.33, 0.67))); }

// ── the circuit ─────────────────────────────────────────────────────────────
float rideHeight(float s) {
    float u = fract(s / CIRCUIT);
    if (u < 0.08) return 2.0;                                        // station
    if (u < 0.22) return 2.0 + smoothstep(0.08, 0.22, u) * 10.0;     // conveyor 1
    if (u < 0.34) return 12.0 - smoothstep(0.22, 0.34, u) * 5.0;     // baby drop
    if (u < 0.40) return 7.0;                                        // meander
    if (u < 0.58) return 7.0 + smoothstep(0.40, 0.58, u) * 25.0;     // conveyor 2
    if (u < 0.62) return 32.0;                                       // the crest
    if (u < 0.74) {                                                  // splashdown
        float d = smoothstep(0.62, 0.74, u);
        return 32.0 - d * d * 30.0;
    }
    return 2.0;                                                      // run-out
}

vec3 trackPoint(float s) {
    float th = fract(s / CIRCUIT) * TAU;
    float r = LAP_R * (1.0 + 0.22 * sin(th * 2.0 + iSeed * 6.3)
                           + 0.11 * sin(th * 3.0 + 1.4));
    return vec3(r * sin(th), rideHeight(s), r * cos(th));
}

float trackRoll(float s) {
    vec3 a = trackPoint(s - 9.0), b = trackPoint(s), c = trackPoint(s + 9.0);
    vec2 d1 = normalize(b.xz - a.xz), d2 = normalize(c.xz - b.xz);
    float turn = d1.x * d2.y - d1.y * d2.x;
    // A flume's water does not bank; only the channel leans a little.
    return clamp(turn * 8.0, -1.0, 1.0) * 0.20;
}

struct Cam { vec3 pos; vec3 right; vec3 up; vec3 fwd; };

Cam rideCam(float s, float rock) {
    Cam c;
    vec3 here = trackPoint(s);
    vec3 tangent = normalize(trackPoint(s + 1.5) - trackPoint(s - 1.5));
    c.fwd = normalize(mix(tangent, normalize(trackPoint(s + 24.0) - here), 0.5));
    float roll = trackRoll(s) + sin(iTime * 1.7) * 0.03 * (0.4 + rock);
    vec3 worldUp = vec3(sin(roll), cos(roll), 0.0);
    c.right = normalize(cross(c.fwd, worldUp));
    c.up = cross(c.right, c.fwd);
    c.pos = here + c.up * 1.35 + c.up * rock * 0.12;
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

// Coverage of a screen-space triangle: 1 inside, 0 outside. The three edge
// cross-products agree in sign only for points within the triangle, which is
// true whichever way round the triangle was wound.
float triCover(vec2 p, vec2 a, vec2 b, vec2 c) {
    float d1 = (p.x - b.x) * (a.y - b.y) - (a.x - b.x) * (p.y - b.y);
    float d2 = (p.x - c.x) * (b.y - c.y) - (b.x - c.x) * (p.y - c.y);
    float d3 = (p.x - a.x) * (c.y - a.y) - (c.x - a.x) * (p.y - a.y);
    bool neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
    bool pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);
    return (neg && pos) ? 0.0 : 1.0;
}

// A flat quad in world space, filled as two triangles. Returns the coverage so
// the caller can shade what it just painted.
float quadFill(inout vec3 col, Cam cam, vec2 p, vec3 A, vec3 B, vec3 C, vec3 D,
               vec3 c, float alpha) {
    vec2 a, b, cc, d; float za, zb, zc, zd;
    if (!project(cam, A, a, za)) return 0.0;
    if (!project(cam, B, b, zb)) return 0.0;
    if (!project(cam, C, cc, zc)) return 0.0;
    if (!project(cam, D, d, zd)) return 0.0;
    float cov = max(triCover(p, a, b, cc), triCover(p, a, cc, d));
    col = mix(col, c, cov * alpha);
    return cov;
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
    if (iSky < 0.34)      { lo = vec3(0.98, 0.72, 0.40); hi = vec3(0.05, 0.20, 0.42); }
    else if (iSky < 0.67) { lo = vec3(0.55, 0.88, 0.85); hi = vec3(0.04, 0.16, 0.46); }
    else                  { lo = vec3(0.75, 0.35, 0.62); hi = vec3(0.03, 0.06, 0.24); }
    vec3 col = mix(lo, hi, pow(up, 0.8));
    col = mix(col, palette(iHue + up * 0.3), 0.20 + iMid * 0.12);
    float band = sin(p.y * 16.0 - tb * 0.18 + sin(p.x * 1.8) * 1.4);
    col += vec3(1.0, 0.95, 0.85) * smoothstep(0.90, 1.0, band)
         * (0.08 + iMid * 0.16) * (1.0 - up);
    col += mix(vec3(1.0, 0.88, 0.55), palette(iHue + 0.12), 0.30)
         * exp(-length(p - vec2(-0.34, 0.16 - iGrade * 0.5)) * 5.5)
         * (0.8 + iBass * 0.6);
    return col;
}

void main() {
    float A = iResolution.x / max(iResolution.y, 1.0);
    float tb = mod(iTime, 300.0);
    float rock = iBass * 0.5 + iPunch * 0.7;
    vec2 p = (v_uv - 0.5) * vec2(A, 1.0) / max(iZoom, 0.05);
    p += vec2(sin(tb * 21.0), sin(tb * 17.0 + 0.7)) * rock * 0.004;

    Cam cam = rideCam(iTravel, rock);
    vec3 col = skyColour(p, tb);

    vec3 rd = normalize(cam.fwd + p.x * cam.right + p.y * cam.up);
    if (rd.y < -0.0015 && cam.pos.y > 0.2) {
        float td = -cam.pos.y / rd.y;
        if (td > 0.0 && td < 900.0) {
            vec3 g = cam.pos + rd * td;
            // The catch pool the whole ride stands in.
            vec3 gnd = mix(vec3(0.020, 0.070, 0.095), palette(iHue + 0.52), 0.20);
            float rip = sin(g.x * 0.35 + tb * 1.1) * sin(g.z * 0.31 - tb * 0.9);
            gnd += mix(vec3(0.30, 0.85, 1.0), palette(iHue + 0.25), 0.35)
                 * smoothstep(0.55, 1.0, rip) * (0.20 + iBass * 0.45);
            gnd += vec3(0.8, 0.95, 1.0)
                 * step(0.9945, hash21(floor(g.xz * 2.0 + tb * 0.6)))
                 * (0.4 + iTreble * 1.0);
            col = mix(col, gnd, exp(-td * 0.004));
        }
    }

    vec3 timber = mix(vec3(0.36, 0.24, 0.15), palette(iHue + 0.58), 0.30);
    vec3 capCol = mix(vec3(0.62, 0.46, 0.28), palette(iHue + 0.20), 0.35);
    vec3 waterC = mix(vec3(0.05, 0.33, 0.46), palette(iHue + 0.48), 0.28);

    const float NEAR = 1.8, FAR = 400.0, HALFW = 2.0, WALL = 1.5, DECK = 1.0;
    for (int i = STEPS - 1; i >= 0; i--) {
        float f0 = float(i) / float(STEPS);
        float f1 = float(i + 1) / float(STEPS);
        float s0 = iTravel + NEAR + f0 * f0 * FAR;
        float s1 = iTravel + NEAR + f1 * f1 * FAR;
        vec3 c0 = trackPoint(s0), c1 = trackPoint(s1);
        vec3 lat0, nrm0, lat1, nrm1;
        frameAt(s0, lat0, nrm0);
        frameAt(s1, lat1, nrm1);
        float fade = 1.0 - f0 * 0.70;

        // Outer faces of the two side walls, then the water between them.
        for (int w = -1; w <= 1; w += 2) {
            float sw = float(w);
            vec3 lo0 = c0 + lat0 * (HALFW * sw) - nrm0 * DECK;
            vec3 lo1 = c1 + lat1 * (HALFW * sw) - nrm1 * DECK;
            vec3 hi0 = c0 + lat0 * (HALFW * sw) + nrm0 * WALL;
            vec3 hi1 = c1 + lat1 * (HALFW * sw) + nrm1 * WALL;
            vec3 face = timber * (0.35 + fade * 0.85)
                      * (w < 0 ? 0.82 : 1.0);   // one side catches the light
            quadFill(col, cam, p, lo0, lo1, hi1, hi0, face, 1.0);
            // Cap rail along the top of the wall.
            strut(col, cam, p, hi0, hi1, 0.13, capCol * (0.4 + fade * 0.9), 0.0);
            // Plank lines, near sections only.
            if (f0 < 0.30 && mod(floor(s0 * 1.1), 1.0) < 0.5) {
                strut(col, cam, p, lo0, hi0, 0.05, timber * 0.55 * fade, 0.0);
            }
        }

        vec3 wl0 = c0 - lat0 * HALFW, wr0 = c0 + lat0 * HALFW;
        vec3 wl1 = c1 - lat1 * HALFW, wr1 = c1 + lat1 * HALFW;
        float cov = quadFill(col, cam, p, wl0, wr0, wr1, wl1,
                             waterC * (0.45 + fade * 0.9), 1.0);
        if (cov > 0.0) {
            // Surface streaming past at the speed the log is actually doing.
            float flow = sin(s0 * 0.9 - iTravel * 0.55 + p.x * 5.0);
            col += mix(vec3(0.45, 0.90, 1.0), palette(iHue + 0.3), 0.30)
                 * smoothstep(0.55, 1.0, flow) * cov * fade
                 * (0.14 + iBass * 0.35);
            col += vec3(0.9, 0.98, 1.0) * cov
                 * step(0.994, hash21(floor(vec2(p.x * 210.0,
                                                 p.y * 210.0 + iTravel * 3.0))))
                 * (0.35 + iTreble * 1.1) * iGlitter;
        }

        // Timber trestle bents: paired legs, sill beam and X-bracing.
        if (mod(floor(s0 * 0.09), 1.0) < 0.5 && c0.y > 3.0) {
            vec3 footL = vec3((c0 - lat0 * 2.7).x, 0.0, (c0 - lat0 * 2.7).z);
            vec3 footR = vec3((c0 + lat0 * 2.7).x, 0.0, (c0 + lat0 * 2.7).z);
            vec3 topL = c0 - lat0 * 2.2 - nrm0 * DECK;
            vec3 topR = c0 + lat0 * 2.2 - nrm0 * DECK;
            vec3 midL = mix(topL, footL, 0.55), midR = mix(topR, footR, 0.55);
            vec3 legC = timber * (0.30 + fade * 0.8);
            strut(col, cam, p, topL, footL, 0.16, legC, 0.0);
            strut(col, cam, p, topR, footR, 0.16, legC, 0.0);
            strut(col, cam, p, topL, topR, 0.13, legC * 1.1, 0.0);
            strut(col, cam, p, midL, midR, 0.10, legC * 0.9, 0.0);
            strut(col, cam, p, topL, midR, 0.09, legC * 0.85, 0.0);
            strut(col, cam, p, topR, midL, 0.09, legC * 0.85, 0.0);
        }
    }

    // The log's own hull filling the bottom of frame, and its bow wave. The
    // hull edge is a shallow parabola, so the nose reads as rounded rather than
    // as a wedge across the channel.
    float hullEdge = (p.y + 0.35) + p.x * p.x * 0.26;
    float hull = smoothstep(0.012, -0.012, hullEdge);
    vec3 hullCol = timber * 0.48 + palette(iHue + 0.1) * 0.06
                 * smoothstep(0.0, -0.30, hullEdge);
    col = mix(col, hullCol, hull);
    col += mix(vec3(0.75, 0.97, 1.0), palette(iHue + 0.3), 0.20)
         * smoothstep(0.026, 0.0,
                      abs(hullEdge - 0.022 - sin(p.x * 26.0 + tb * 7.0) * 0.008))
         * (0.45 + iBass * 0.9);

    // Splashdown: a wall of water thrown up over the camera, then droplets.
    float splash = clamp(iSplash, 0.0, 1.0) * clamp(iSplashGain, 0.0, 3.0);
    if (splash > 0.001) {
        float sheet = smoothstep(-0.55, 0.35, -p.y + splash * 1.15
                                 + sin(p.x * 9.0 + tb * 12.0) * 0.06);
        col = mix(col, vec3(0.86, 0.96, 1.0), sheet * splash * 0.75);
        for (int i = 0; i < 14; i++) {
            float fi = float(i);
            float ph = fract(tb * 0.8 + fi * 0.137);
            vec2 dp = vec2((hash21(vec2(fi, 1.0)) - 0.5) * 1.5,
                           -0.45 + ph * 1.1 - ph * ph * 0.55);
            col += vec3(0.85, 0.96, 1.0)
                 * exp(-dot(p - dp, p - dp) * 1100.0)
                 * (1.0 - ph) * splash * (0.8 + iTreble * 0.8);
        }
    }

    col += palette(iHue + 0.45) * iBeat * 0.10;
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
    if u < 0.08:
        return 2.0
    if u < 0.22:
        return 2.0 + _smoothstep(0.08, 0.22, u) * 10.0
    if u < 0.34:
        return 12.0 - _smoothstep(0.22, 0.34, u) * 5.0
    if u < 0.40:
        return 7.0
    if u < 0.58:
        return 7.0 + _smoothstep(0.40, 0.58, u) * 25.0
    if u < 0.62:
        return 32.0
    if u < 0.74:
        d = _smoothstep(0.62, 0.74, u)
        return 32.0 - d * d * 30.0
    return 2.0


def on_conveyor(s: float) -> bool:
    """True while a conveyor belt is pushing the log uphill."""
    u = (s / CIRCUIT) % 1.0
    return 0.08 <= u < 0.22 or 0.40 <= u < 0.58


class LogFlume(BaseEffect):
    """First-person water flume: conveyors, meander and a splashdown."""

    NAME = 'Log Flume'
    AUTHOR = 'unicorn-viz'
    TAGS = ['rollercoast', 'ride', 'water', 'neon', 'energetic', 'chill']
    PING_PONG_FRIENDS = [
        'First Drop', 'Corkscrew', 'Night Coaster', 'Mine Train',
        'Coaster Cam', 'Demon Drop', 'Audio Spectrum',
    ]

    _CONVEYOR_SPEED = 5.0    # m/s, belt speed
    _MIN_SPEED = 4.0
    _DRAG_SPEED = 9.0        # m/s ceiling on the run-out; water eats the rest

    def _init(self) -> None:
        self.parameters = {
            'speed': float(self.config.get('speed', 1.0)),
            'zoom': float(self.config.get('zoom', 1.0)),
            'reactivity': float(self.config.get('reactivity', 1.3)),
            'glitter': float(self.config.get('glitter', 1.0)),
            'splash_gain': float(self.config.get('splash_gain', 1.0)),
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
        self._splash = 0.0

    def update(self, dt: float, audio: AudioData) -> None:
        """Belt on the conveyors, gravity elsewhere, drag on the run-out."""
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

        if on_conveyor(self._travel):
            v = self._CONVEYOR_SPEED
        else:
            v = math.sqrt(max(0.0, 2.0 * GRAVITY * max(0.0, H_TOP - h)))
            v = max(self._MIN_SPEED, v)
            if u >= 0.76 or u < 0.08:
                v = min(v, self._DRAG_SPEED)

        # The hull hits the pool at the foot of the splashdown; the sheet of
        # water it throws hangs over the camera for a moment afterwards.
        if 0.72 <= u < 0.76:
            self._splash = 1.0
        self._splash = max(0.0, self._splash - dt * 0.9)

        speed = max(0.05, float(self.parameters['speed']))
        self._travel = (self._travel
                        + dt * speed * v * (0.8 + self._bass * 0.5)) % CIRCUIT

    def render(self) -> None:
        """Upload ride state and draw the view from the log's front seat."""
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
        p['iSplash'].value = float(self._splash)
        p['iSplashGain'].value = max(0.0, float(self.parameters['splash_gain']))
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        """Release GL resources."""
        self._vao.release()
        self._vbo.release()
        self._prog.release()
