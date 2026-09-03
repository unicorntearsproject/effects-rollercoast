"""First Drop — the lift hill, the crest and the plunge, ridden first-person.

The ride is a **closed circuit**: the track is a parametric loop that returns
to its own station, so the train laps it forever. That matters for more than
tidiness. The 1.x track ran `z = s` — an endless straight line — so world
coordinates grew without bound until they were teleported back every 64 laps,
and the height profile did not match across the lap seam, which spiked the
gradient feeding the speed integrator once per lap. Closing the loop keeps
every world coordinate inside a bounded footprint forever and makes the profile
continuous at the seam, so the ride is as stable on lap 900 as on lap one.

Speed is real. Rather than a hand-tuned function of the gradient, the train
conserves energy: away from the chain lift its speed comes from how far it has
fallen below the crest, ``v = sqrt(2 g (H_top - h))``, so it crawls over
summits and screams through valleys because that is what the geometry demands.
The lift itself runs at constant chain speed, as a real lift does.

The structure is a real one. Each track section carries two running rails and a
box spine beneath them, tied together by zig-zag truss lacing; the whole thing
stands on support bents — paired legs with cross-bracing and footers — spaced
along the circuit. Because the circuit is closed and compact, you see the rest
of the ride across the infield, which is most of what makes a coaster read as a
coaster rather than as a corridor.

Audio reactivity:
  - bass    -> speed surge, camera shake, rail glow
  - mid     -> palette drift, sky banding, structure tint
  - treble  -> rail sparkle, speed-line density
  - beat    -> flash, boost kick, sky burst

Randomised per activation: palette hue, position on the circuit, track seed
(the loop's plan shape), sky variant, and ``self.time``.

Config keys (under ``[effects.FirstDrop]``):
    speed, zoom, reactivity, glitter, drop_force
"""
from __future__ import annotations

import math

import moderngl
import numpy as np

from unicornviz.effects.base import AudioData, BaseEffect

__version__ = '0.2.0'

# One lap, in metres. Shared by the shader and the CPU physics.
CIRCUIT = 900.0
# Crest of the ride; the energy model measures every fall from just above it.
H_TOP = 68.0
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
uniform float iTravel;     // position on the lap, always within [0, CIRCUIT)
uniform float iSeed;       // per-activation plan shape of the loop
uniform float iSky;        // per-activation sky variant
uniform float iDropForce;  // tweakable: how violent the plunge is
uniform float iGrade;      // current track gradient, tilts the horizon
in  vec2 v_uv;
out vec4 fragColor;

#define PI  3.14159265
#define TAU 6.28318530
#define STEPS 46
#define CIRCUIT 900.0
#define LAP_R 168.0

float hash21(vec2 p) {
    p = mod(p, 512.0);
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
vec3 palette(float t) { return 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.33, 0.67))); }

// ── the circuit ─────────────────────────────────────────────────────────────
// Height as a function of lap fraction. Every branch meets its neighbour, and
// the last value equals the first, so there is no step at the lap seam -- the
// 1.x profile jumped 4m there and spiked the speed integrator once per lap.
float rideHeight(float s) {
    float u = fract(s / CIRCUIT);
    if (u < 0.06) return 3.0;                                        // station
    if (u < 0.34) return 3.0 + smoothstep(0.06, 0.34, u) * 62.0;     // chain lift
    if (u < 0.40) return 65.0 - smoothstep(0.34, 0.40, u) * 3.0;     // crest
    if (u < 0.52) {                                                  // the drop
        float d = smoothstep(0.40, 0.52, u);
        return 62.0 - d * d * 54.0;
    }
    if (u < 0.90) {                                                  // camelbacks
        float d = (u - 0.52) / 0.38;
        return 8.0 + (1.0 - d) * 18.0 * (0.5 - 0.5 * cos(d * 13.0));
    }
    return mix(8.0, 3.0, smoothstep(0.90, 1.0, u));                  // brake run
}

// The loop's plan shape: a closed curve, so every coordinate stays bounded no
// matter how many laps have been run.
vec3 trackPoint(float s) {
    float th = fract(s / CIRCUIT) * TAU;
    float r = LAP_R * (1.0 + 0.20 * sin(th * 2.0 + iSeed * 6.3)
                           + 0.09 * sin(th * 3.0 + 1.9));
    return vec3(r * sin(th), rideHeight(s), r * cos(th));
}

// Bank into the turn from the actual plan curvature, not from a sine.
float trackRoll(float s) {
    vec3 a = trackPoint(s - 9.0), b = trackPoint(s), c = trackPoint(s + 9.0);
    vec2 d1 = normalize(b.xz - a.xz), d2 = normalize(c.xz - b.xz);
    float turn = d1.x * d2.y - d1.y * d2.x;
    return clamp(turn * 11.0, -1.0, 1.0) * 0.62;
}

struct Cam { vec3 pos; vec3 right; vec3 up; vec3 fwd; };

Cam rideCam(float s, float shake) {
    Cam c;
    vec3 here = trackPoint(s);
    vec3 tangent = normalize(trackPoint(s + 1.5) - trackPoint(s - 1.5));
    c.fwd = normalize(mix(tangent, normalize(trackPoint(s + 22.0) - here), 0.5));
    float roll = trackRoll(s);
    vec3 worldUp = vec3(sin(roll), cos(roll), 0.0);
    c.right = normalize(cross(c.fwd, worldUp));
    c.up = cross(c.right, c.fwd);
    c.pos = here + c.up * 1.9 + c.right * shake * 0.25 + c.up * shake * 0.18;
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

// One structural member, drawn with depth-correct thickness.
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

// Local frame at s: lateral and normal, both banked with the track.
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
    if (iSky < 0.34)      { lo = vec3(1.00, 0.42, 0.22); hi = vec3(0.10, 0.06, 0.34); }
    else if (iSky < 0.67) { lo = vec3(0.95, 0.75, 0.25); hi = vec3(0.05, 0.30, 0.55); }
    else                  { lo = vec3(0.60, 0.10, 0.55); hi = vec3(0.03, 0.05, 0.20); }
    vec3 col = mix(lo, hi, pow(up, 0.75));
    col = mix(col, palette(iHue + up * 0.3), 0.22 + iMid * 0.10);
    float band = sin(p.y * 22.0 - tb * 0.25 + sin(p.x * 2.0) * 1.2);
    col += vec3(1.0, 0.85, 0.75) * smoothstep(0.90, 1.0, band)
         * (0.10 + iMid * 0.20) * (1.0 - up);
    vec2 sunP = vec2(0.30, 0.10 - iGrade * 0.5);
    col += mix(vec3(1.0, 0.75, 0.35), palette(iHue + 0.1), 0.35)
         * exp(-length(p - sunP) * 5.0) * (0.9 + iBass * 0.7);
    return col;
}

void main() {
    float A = iResolution.x / max(iResolution.y, 1.0);
    float tb = mod(iTime, 300.0);
    float shake = (iBass * 0.5 + iPunch * 0.8) * iDropForce;
    vec2 p = (v_uv - 0.5) * vec2(A, 1.0) / max(iZoom, 0.05);
    p += vec2(sin(tb * 47.0), sin(tb * 39.0 + 1.1)) * shake * 0.006;

    Cam cam = rideCam(iTravel, shake);
    vec3 col = skyColour(p, tb);

    vec3 rd = normalize(cam.fwd + p.x * cam.right + p.y * cam.up);
    if (rd.y < -0.0015 && cam.pos.y > 0.2) {
        float td = -cam.pos.y / rd.y;
        if (td > 0.0 && td < 900.0) {
            vec3 g = cam.pos + rd * td;
            vec2 cell = abs(fract(g.xz * 0.06) - 0.5);
            float line = smoothstep(0.045, 0.0, min(cell.x, cell.y));
            vec3 gnd = mix(vec3(0.030, 0.035, 0.055), palette(iHue + 0.55), 0.25);
            gnd += mix(vec3(0.35, 0.85, 1.0), palette(iHue + 0.2), 0.35)
                 * line * (0.35 + iBass * 0.5);
            float lamp = hash21(floor(g.xz * 0.10));
            gnd += mix(vec3(1.0, 0.80, 0.45), palette(iHue + 0.1), 0.3)
                 * step(0.965, lamp)
                 * smoothstep(0.42, 0.0, length(fract(g.xz * 0.10) - 0.5))
                 * (0.8 + iTreble * 1.2);
            col = mix(col, gnd, exp(-td * 0.0045));
        }
    }

    // ── the structure, marched far to near so crests occlude what is beyond ──
    vec3 railCol  = mix(vec3(0.85, 0.90, 1.00), palette(iHue + 0.15), 0.45);
    vec3 spineCol = mix(vec3(0.16, 0.18, 0.24), palette(iHue + 0.45), 0.30);
    vec3 tieCol   = mix(vec3(0.40, 0.26, 0.16), palette(iHue + 0.60), 0.28);
    const float NEAR = 2.0, FAR = 430.0, GAUGE = 1.6, SPINE = 2.1;

    for (int i = STEPS - 1; i >= 0; i--) {
        float f0 = float(i) / float(STEPS);
        float f1 = float(i + 1) / float(STEPS);
        float s0 = iTravel + NEAR + f0 * f0 * FAR;
        float s1 = iTravel + NEAR + f1 * f1 * FAR;
        vec3 c0 = trackPoint(s0), c1 = trackPoint(s1);
        vec3 lat0, nrm0, lat1, nrm1;
        frameAt(s0, lat0, nrm0);
        frameAt(s1, lat1, nrm1);
        float fade = 1.0 - f0 * 0.75;
        float glow = (0.30 + iTreble * 0.9) * fade;

        // Box spine under the rails: two chords, so the track has depth.
        strut(col, cam, p, c0 - nrm0 * SPINE - lat0 * 0.5,
                          c1 - nrm1 * SPINE - lat1 * 0.5, 0.20, spineCol * fade, 0.0);
        strut(col, cam, p, c0 - nrm0 * SPINE + lat0 * 0.5,
                          c1 - nrm1 * SPINE + lat1 * 0.5, 0.20, spineCol * fade, 0.0);

        // Zig-zag truss lacing tying the spine to the rails. Near steps only --
        // at distance it collapses into noise and costs the same to draw.
        if (f0 < 0.42) {
            float zig = mod(floor(s0 * 0.5), 2.0) < 1.0 ? -1.0 : 1.0;
            strut(col, cam, p, c0 - nrm0 * SPINE + lat0 * (0.5 * zig),
                              c1 + lat1 * (GAUGE * zig), 0.11, spineCol * 0.95 * fade, 0.0);
        }

        // Cross ties.
        if (mod(floor(s0 * 0.55), 1.0) < 0.5) {
            strut(col, cam, p, c0 - lat0 * GAUGE * 1.25, c0 + lat0 * GAUGE * 1.25,
                  0.13, tieCol * (0.4 + fade * 0.7), 0.0);
        }

        // The running rails, drawn last so they sit on top of the structure.
        for (int r = -1; r <= 1; r += 2) {
            strut(col, cam, p, c0 + lat0 * (GAUGE * float(r)),
                              c1 + lat1 * (GAUGE * float(r)),
                  0.17, railCol * (0.45 + fade * 0.85 + iBass * 0.35), glow);
        }

        // Support bents: paired legs, a cross brace and footers.
        if (mod(floor(s0 * 0.0714), 1.0) < 0.5 && c0.y > 4.0) {
            vec3 baseL = vec3((c0 - lat0 * 2.6).x, 0.0, (c0 - lat0 * 2.6).z);
            vec3 baseR = vec3((c0 + lat0 * 2.6).x, 0.0, (c0 + lat0 * 2.6).z);
            vec3 topL = c0 - nrm0 * SPINE - lat0 * 0.9;
            vec3 topR = c0 - nrm0 * SPINE + lat0 * 0.9;
            vec3 legCol = mix(vec3(0.20, 0.22, 0.28), palette(iHue + 0.4), 0.30) * fade;
            strut(col, cam, p, topL, baseL, 0.17, legCol, 0.0);
            strut(col, cam, p, topR, baseR, 0.17, legCol, 0.0);
            // Cross bracing, and a horizontal tie halfway down.
            vec3 midL = mix(topL, baseL, 0.5), midR = mix(topR, baseR, 0.5);
            strut(col, cam, p, topL, midR, 0.10, legCol * 0.9, 0.0);
            strut(col, cam, p, topR, midL, 0.10, legCol * 0.9, 0.0);
            strut(col, cam, p, midL, midR, 0.10, legCol * 0.9, 0.0);
            strut(col, cam, p, midL, baseR, 0.09, legCol * 0.85, 0.0);
            strut(col, cam, p, midR, baseL, 0.09, legCol * 0.85, 0.0);
        }
    }

    // Speed lines radiating from the vanishing point.
    float ang = atan(p.y, p.x);
    float rad = length(p);
    float lane = floor(ang / TAU * 96.0);
    float lh = hash21(vec2(lane, 7.0));
    float ph = fract(lh + iTravel * 0.02);
    float along = fract(rad * 1.6 - ph * 2.0);
    float across = abs(fract(ang / TAU * 96.0) - 0.5) * 2.0;
    float streak = smoothstep(0.22, 0.0, across)
                 * smoothstep(0.55, 0.92, along) * smoothstep(1.0, 0.93, along);
    col += vec3(0.85, 0.93, 1.0) * streak * step(0.55, lh)
         * smoothstep(0.18, 0.75, rad) * (0.30 + iTreble * 0.9 + iBass * 0.5)
         * iGlitter;

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
    if u < 0.06:
        return 3.0
    if u < 0.34:
        return 3.0 + _smoothstep(0.06, 0.34, u) * 62.0
    if u < 0.40:
        return 65.0 - _smoothstep(0.34, 0.40, u) * 3.0
    if u < 0.52:
        d = _smoothstep(0.40, 0.52, u)
        return 62.0 - d * d * 54.0
    if u < 0.90:
        d = (u - 0.52) / 0.38
        return 8.0 + (1.0 - d) * 18.0 * (0.5 - 0.5 * math.cos(d * 13.0))
    return 8.0 + (3.0 - 8.0) * _smoothstep(0.90, 1.0, u)


def on_lift(s: float) -> bool:
    """True while the chain has hold of the train."""
    return ((s / CIRCUIT) % 1.0) < 0.34


class FirstDrop(BaseEffect):
    """First-person lift hill, crest and plunge on a closed 3-D circuit."""

    NAME = 'First Drop'
    AUTHOR = 'unicorn-viz'
    TAGS = ['rollercoast', 'ride', 'neon', 'intense', 'energetic', 'hard']
    PING_PONG_FRIENDS = [
        'Corkscrew', 'Night Coaster', 'Mine Train', 'Coaster Cam',
        'Demon Drop', 'Log Flume', 'Audio Spectrum',
    ]

    _LIFT_SPEED = 9.0        # m/s, chain speed
    _MIN_SPEED = 5.0         # m/s, so a crest never stalls the ride

    def _init(self) -> None:
        self.parameters = {
            'speed': float(self.config.get('speed', 1.0)),
            'zoom': float(self.config.get('zoom', 1.0)),
            'reactivity': float(self.config.get('reactivity', 1.3)),
            'glitter': float(self.config.get('glitter', 1.0)),
            'drop_force': float(self.config.get('drop_force', 1.0)),
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

    def update(self, dt: float, audio: AudioData) -> None:
        """Advance the train by conserving energy, and track the audio."""
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

        h = ride_height(self._travel)
        ahead = 5.0
        self._grade = float(np.clip(
            (ride_height(self._travel + ahead) - ride_height(self._travel - ahead))
            / (2 * ahead), -1.5, 1.5))

        # Energy conservation: everything the train has, it got by falling.
        # On the lift the chain sets the pace instead.
        if on_lift(self._travel):
            v = self._LIFT_SPEED
        else:
            v = math.sqrt(max(0.0, 2.0 * GRAVITY * max(0.0, H_TOP - h)))
            v = max(self._MIN_SPEED, v)

        speed = max(0.05, float(self.parameters['speed']))
        # Travel stays inside one lap forever, so no coordinate ever grows.
        self._travel = (self._travel
                        + dt * speed * v * (0.8 + self._bass * 0.5)) % CIRCUIT

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
        p['iDropForce'].value = max(0.0, float(self.parameters['drop_force']))
        p['iGrade'].value = float(self._grade)
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        """Release GL resources."""
        self._vao.release()
        self._vbo.release()
        self._prog.release()
