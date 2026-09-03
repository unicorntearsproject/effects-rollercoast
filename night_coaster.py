"""Night Coaster — the same circuit after dark, lit only by the park.

Nothing here is lit by a sun.  The rails are emissive, the midway below is a
field of lamps, and the beat throws fireworks that briefly light the structure
from above.

Because the circuit is closed and compact you can see the rest of the ride
across the infield, which at night means a lit ribbon of track hanging in the
dark — the thing that actually reads as a coaster from inside the park.

**The track is sampled on the CPU, once per frame.** The geometry does not vary
per pixel, but deriving it inside the fragment loop meant about 560
``trackPoint()`` evaluations per pixel per frame — 1.2 billion at 1080p — which
is what pinned this pack at 8 fps. ``_build_march()`` samples the circuit into
a small uniform table and hands the shader the camera basis too, so the
fragment program has no track model in it at all: it projects the table and
rasterizes.

Audio reactivity:
  - bass    -> rail neon gain, speed, lamp swell
  - mid     -> palette drift, sky nebula
  - treble  -> rail sparkle, lamp twinkle
  - beat    -> firework bursts, flash, boost

Randomised per activation: palette hue, position on the circuit, track seed,
sky variant, and ``self.time``.

Config keys (under ``[effects.NightCoaster]``):
    speed, zoom, reactivity, glitter, neon_gain
"""
from __future__ import annotations

import math

import moderngl
import numpy as np

from unicornviz.effects.base import AudioData, BaseEffect

__version__ = '0.3.0'

# One lap, in metres. Shared by the CPU physics and the track sampler.
CIRCUIT = 900.0
# Crest of the ride; the energy model measures every fall from just above it.
H_TOP = 68.0
GRAVITY = 9.81
LAP_R = 168.0
TAU = 2.0 * math.pi

# March: how many track sections are sampled ahead of the train, and how far.
STEPS = 46
NEAR = 2.0
FAR = 430.0

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
uniform float iSky;        // per-activation sky variant
uniform float iGrade;      // current track gradient, tilts the horizon
uniform float iNeonGain;   // tweakable: how hot the park's emitters run
// The march table, sampled on the CPU once per frame. xyz is the world point
// on the centreline; iP.w is s, the distance along the circuit, which the
// structure uses to place ties and bents.
uniform vec4  iP[47];
uniform vec4  iL[47];      // xyz = lateral unit vector at that point
uniform vec4  iN[47];      // xyz = normal ("up" out of the track bed)
uniform vec3  iCamPos;
uniform vec3  iCamRight;
uniform vec3  iCamUp;
uniform vec3  iCamFwd;
in  vec2 v_uv;
out vec4 fragColor;

#define PI  3.14159265
#define TAU 6.28318530
#define STEPS 46

float hash21(vec2 p) {
    p = mod(p, 512.0);
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
vec3 palette(float t) { return 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.33, 0.67))); }

// World point -> screen. Fragment-independent, so it is called once per node
// per step rather than once per structural member.
bool project(vec3 P, out vec2 sp, out float depth) {
    vec3 q = P - iCamPos;
    float z = dot(q, iCamFwd);
    if (z < 0.35) return false;
    float iz = 1.0 / z;
    sp = vec2(dot(q, iCamRight), dot(q, iCamUp)) * iz;
    depth = z;
    return true;
}
float segDist(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h);
}

// One structural member, already projected. The screen-space bounding-box
// reject in front is what makes a 46-step march affordable: for any given
// pixel almost every member is nowhere near it.
void seg2(inout vec3 col, vec2 p, vec2 a, vec2 b, float w, vec3 c, float glow) {
    if (p.x < min(a.x, b.x) - w || p.x > max(a.x, b.x) + w ||
        p.y < min(a.y, b.y) - w || p.y > max(a.y, b.y) + w) return;
    float d = segDist(p, a, b);
    col = mix(col, c, smoothstep(w, w * 0.45, d));
    if (glow > 0.0) col += c * smoothstep(w * 0.55, 0.0, d) * glow;
}

// A member given in world space, for nodes outside the shared per-step set.
void strut(inout vec3 col, vec2 p, vec3 A, vec3 B, float radius, vec3 c, float glow) {
    vec2 a, b; float da, db;
    if (!project(A, a, da)) return;
    if (!project(B, b, db)) return;
    seg2(col, p, a, b, clamp(radius / min(da, db), 0.0007, 0.09), c, glow);
}

vec3 skyColour(vec2 p, float tb) {
    float up = clamp(p.y * 0.9 + 0.35 - iGrade * 0.35, 0.0, 1.0);
    vec3 lo, hi;
    if (iSky < 0.34)      { lo = vec3(0.030, 0.020, 0.070); hi = vec3(0.004, 0.006, 0.024); }
    else if (iSky < 0.67) { lo = vec3(0.020, 0.045, 0.080); hi = vec3(0.003, 0.008, 0.020); }
    else                  { lo = vec3(0.055, 0.014, 0.060); hi = vec3(0.006, 0.004, 0.018); }
    vec3 col = mix(lo, hi, pow(up, 0.6));
    col += palette(iHue + up * 0.4) * 0.035
         * (0.5 + 0.5 * sin(p.x * 2.2 + p.y * 3.1 + tb * 0.05)) * (0.6 + iMid * 0.8);
    // Stars: a soft round point inside its cell, not a filled cell.
    vec2 scell = p * 90.0;
    vec2 sg = floor(scell);
    float sh = hash21(sg);
    if (sh > 0.9955) {
        vec2 sd = fract(scell) - vec2(hash21(sg + 5.3), hash21(sg + 9.1));
        col += vec3(0.85, 0.90, 1.0) * exp(-dot(sd, sd) * 26.0)
             * (0.4 + 0.6 * sin(tb * 2.0 + sh * 40.0)) * (0.6 + iTreble * 0.8) * up;
    }
    col += vec3(0.80, 0.86, 1.0) * exp(-length(p - vec2(-0.34, 0.28)) * 16.0) * 1.2;
    return col;
}

void main() {
    float A = iResolution.x / max(iResolution.y, 1.0);
    float tb = mod(iTime, 300.0);
    float shake = iBass * 0.5 + iPunch * 0.8;
    vec2 p = (v_uv - 0.5) * vec2(A, 1.0) / max(iZoom, 0.05);
    p += vec2(sin(tb * 47.0), sin(tb * 39.0 + 1.1)) * shake * 0.006;

    vec3 col = skyColour(p, tb);

    vec3 rd = normalize(iCamFwd + p.x * iCamRight + p.y * iCamUp);
    if (rd.y < -0.0015 && iCamPos.y > 0.2) {
        float td = -iCamPos.y / rd.y;
        if (td > 0.0 && td < 900.0) {
            vec3 g = iCamPos + rd * td;
            vec2 cell = abs(fract(g.xz * 0.06) - 0.5);
            float line = smoothstep(0.045, 0.0, min(cell.x, cell.y));
            vec3 gnd = mix(vec3(0.030, 0.035, 0.055), palette(iHue + 0.55), 0.25);
            gnd += mix(vec3(0.35, 0.85, 1.0), palette(iHue + 0.2), 0.35)
                 * line * (0.35 + iBass * 0.5);
            float lamp = hash21(floor(g.xz * 0.17));
            gnd += mix(vec3(1.0, 0.80, 0.45), palette(iHue + 0.1), 0.3)
                 * step(0.920, lamp)
                 * smoothstep(0.42, 0.0, length(fract(g.xz * 0.17) - 0.5))
                 * (0.8 + iTreble * 1.2);
            col = mix(col, gnd, exp(-td * 0.0045));
        }
    }

    // ── the structure, marched far to near so crests occlude what is beyond ──
    vec3 railCol  = mix(vec3(0.85, 0.90, 1.00), palette(iHue + 0.15), 0.45);
    vec3 spineCol = mix(vec3(0.16, 0.18, 0.24), palette(iHue + 0.45), 0.30);
    vec3 tieCol   = mix(vec3(0.40, 0.26, 0.16), palette(iHue + 0.60), 0.28);
    const float GAUGE = 1.6, SPINE = 2.1;

    for (int i = STEPS - 1; i >= 0; i--) {
        vec3 c0 = iP[i].xyz,     c1 = iP[i + 1].xyz;
        vec3 l0 = iL[i].xyz,     l1 = iL[i + 1].xyz;
        vec3 n0 = iN[i].xyz,     n1 = iN[i + 1].xyz;
        float s0 = iP[i].w, s1 = iP[i + 1].w;

        // Ties and bents sit at fixed spacings along the track, and a section
        // carries one when its span crosses that spacing. Testing the section's
        // own s against a square wave instead -- what 0.2.0 did -- ties the
        // structure's density to the sampling: near sections are two metres
        // apart, so three or four in a row would each stack a bent in almost
        // the same place, and far sections would each grow one 30 m long.
        bool hasTie  = floor(s1 / 1.8) > floor(s0 / 1.8);
        bool hasBent = floor(s1 / 14.0) > floor(s0 / 14.0) && c0.y > 4.0;

        // Reject the whole section if this pixel is nowhere near it. A
        // section's members all live within about 3.6 m of its centreline, so
        // that radius over the near depth bounds them on screen.
        vec2 a, b; float da, db;
        bool va = project(c0, a, da), vb = project(c1, b, db);
        bool inSection = true;
        if (va && vb) {
            float rad = 3.6 / max(min(da, db), 0.5);
            inSection = !(p.x < min(a.x, b.x) - rad || p.x > max(a.x, b.x) + rad ||
                          p.y < min(a.y, b.y) - rad || p.y > max(a.y, b.y) + rad);
        } else if (!va && !vb) {
            inSection = false;
        }
        // Reject the bent against the section's own projected point before
        // projecting its four corners. A long far section always spans a bent
        // spacing, so without this every far section pays four projections per
        // pixel whether or not the bent is anywhere near it -- which was most
        // of the frame. The bent hangs below the track within about 3 m of it.
        if (hasBent && va) {
            float bw = 4.0 / max(da, 0.5);
            hasBent = !(p.x < a.x - bw || p.x > a.x + bw || p.y > a.y + bw);
        }
        if (!inSection && !hasBent) continue;

        float f0 = float(i) / float(STEPS);
        float fade = 1.0 - f0 * 0.75;
        float glow = (0.30 + iTreble * 0.9) * fade;

        // Box spine under the rails: two chords, so the track has depth.
        if (inSection) {
        strut(col, p, c0 - n0 * SPINE - l0 * 0.5,
                      c1 - n1 * SPINE - l1 * 0.5, 0.20, spineCol * fade, 0.0);
        strut(col, p, c0 - n0 * SPINE + l0 * 0.5,
                      c1 - n1 * SPINE + l1 * 0.5, 0.20, spineCol * fade, 0.0);

        // Zig-zag truss lacing tying the spine to the rails. Near steps only --
        // at distance it collapses into noise and costs the same to draw.
        if (f0 < 0.42) {
            float zig = mod(floor(s0 * 0.5), 2.0) < 1.0 ? -1.0 : 1.0;
            strut(col, p, c0 - n0 * SPINE + l0 * (0.5 * zig),
                          c1 + l1 * (GAUGE * zig), 0.11, spineCol * 0.95 * fade, 0.0);
        }

        // Cross ties.
        if (hasTie) {
            strut(col, p, c0 - l0 * GAUGE * 1.25, c0 + l0 * GAUGE * 1.25,
                  0.13, tieCol * (0.4 + fade * 0.7), 0.0);
        }

        // The running rails, drawn last so they sit on top of the structure.
        for (int r = -1; r <= 1; r += 2) {
            strut(col, p, c0 + l0 * (GAUGE * float(r)), c1 + l1 * (GAUGE * float(r)),
                  0.17, railCol * (0.35 + fade * 0.7 + iBass * 0.55)
                                * (0.7 + iNeonGain), glow * (1.0 + iNeonGain));
        }
        }

        // Support bents: paired legs, a cross brace and footers. A bent reaches
        // all the way to grade, far outside its section's screen box, so it
        // carries its own -- four corners projected, then one box test gates
        // the whole assembly. Letting it bypass the cull instead cost 68% of
        // the frame.
        if (hasBent) {
            vec3 baseL = vec3((c0 - l0 * 2.6).x, 0.0, (c0 - l0 * 2.6).z);
            vec3 baseR = vec3((c0 + l0 * 2.6).x, 0.0, (c0 + l0 * 2.6).z);
            vec3 topL = c0 - n0 * SPINE - l0 * 0.9;
            vec3 topR = c0 - n0 * SPINE + l0 * 0.9;
            vec2 bl, br, tl, tr; float z0, z1, z2, z3;
            bool ok = project(baseL, bl, z0) && project(baseR, br, z1)
                   && project(topL, tl, z2) && project(topR, tr, z3);
            bool draw = true;
            if (ok) {
                float bw = 0.3 / max(min(min(z0, z1), min(z2, z3)), 0.5);
                vec2 lo = min(min(bl, br), min(tl, tr)) - bw;
                vec2 hi = max(max(bl, br), max(tl, tr)) + bw;
                draw = !(p.x < lo.x || p.x > hi.x || p.y < lo.y || p.y > hi.y);
            }
            if (draw && ok) {
                // Project the six distinct nodes once and draw in screen space.
                // Seven world-space members would re-project shared nodes 14
                // times over, and projection is the bent's dominant cost.
                vec2 ml, mr; float z4, z5;
                project(mix(topL, baseL, 0.5), ml, z4);
                project(mix(topR, baseR, 0.5), mr, z5);
                vec3 legCol = mix(vec3(0.20, 0.22, 0.28), palette(iHue + 0.4), 0.30) * fade;
                float wL = clamp(0.17 / min(z2, z0), 0.0007, 0.09);
                float wR = clamp(0.17 / min(z3, z1), 0.0007, 0.09);
                float wX = clamp(0.10 / min(min(z2, z3), min(z4, z5)), 0.0007, 0.09);
                float wF = clamp(0.09 / min(min(z4, z5), min(z0, z1)), 0.0007, 0.09);
                seg2(col, p, tl, bl, wL, legCol, 0.0);
                seg2(col, p, tr, br, wR, legCol, 0.0);
                // Cross bracing, and a horizontal tie halfway down.
                seg2(col, p, tl, mr, wX, legCol * 0.9, 0.0);
                seg2(col, p, tr, ml, wX, legCol * 0.9, 0.0);
                seg2(col, p, ml, mr, wX, legCol * 0.9, 0.0);
                seg2(col, p, ml, br, wF, legCol * 0.85, 0.0);
                seg2(col, p, mr, bl, wF, legCol * 0.85, 0.0);
            }
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

    // Fireworks over the park, relit on the beat.
    for (int i = 0; i < 3; i++) {
        float fi = float(i);
        float ph = fract(iTravel * 0.0035 + fi * 0.37);
        vec2 fp = vec2((hash21(vec2(fi, floor(iTravel * 0.0035 + fi * 0.37))) - 0.5) * 1.5,
                       0.22 + hash21(vec2(fi + 3.0, 2.0)) * 0.30);
        float r = ph * 0.42;
        float d = length(p - fp);
        vec3 fc = palette(iHue + fi * 0.28 + 0.15);
        col += fc * smoothstep(0.035, 0.0, abs(d - r)) * (1.0 - ph) * (0.7 + iPunch * 1.6);
        float spokes = pow(0.5 + 0.5 * sin(atan(p.y - fp.y, p.x - fp.x) * 16.0 + fi), 6.0);
        col += fc * smoothstep(r * 1.05, 0.0, d) * spokes * (1.0 - ph) * 0.45;
    }

    col += palette(iHue + 0.5) * iBeat * 0.12;
    col *= smoothstep(1.5, 0.22, length(v_uv - 0.5));
    col = col / (1.0 + col * 0.55);
    col = pow(clamp(col, 0.0, 1.0), vec3(0.85));
    fragColor = vec4(col, 1.0);
}
"""


def _sstep(a: float, b: float, x: np.ndarray) -> np.ndarray:
    """Vectorised GLSL smoothstep."""
    t = np.clip((x - a) / max(b - a, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def ride_height(s: np.ndarray) -> np.ndarray:
    """Height above grade at distance ``s`` along the circuit.

    Every branch meets its neighbour and the last value equals the first, so
    there is no step at the lap seam -- the 1.x profile jumped 4 m there and
    spiked the speed integrator once per lap.
    """
    u = (s / CIRCUIT) % 1.0
    d_drop = _sstep(0.40, 0.52, u)
    d_cam = (u - 0.52) / 0.38
    return np.select(
        [u < 0.06, u < 0.34, u < 0.40, u < 0.52, u < 0.90],
        [np.full_like(u, 3.0),                              # station
         3.0 + _sstep(0.06, 0.34, u) * 62.0,                # chain lift
         65.0 - _sstep(0.34, 0.40, u) * 3.0,                # crest
         62.0 - d_drop * d_drop * 54.0,                     # the drop
         8.0 + (1.0 - d_cam) * 18.0 * (0.5 - 0.5 * np.cos(d_cam * 13.0))],
        8.0 + (3.0 - 8.0) * _sstep(0.90, 1.0, u))           # brake run


def track_point(s: np.ndarray, seed: float) -> np.ndarray:
    """The circuit's centreline: a closed curve, so coordinates stay bounded."""
    th = ((s / CIRCUIT) % 1.0) * TAU
    r = LAP_R * (1.0 + 0.20 * np.sin(th * 2.0 + seed * 6.3)
                     + 0.09 * np.sin(th * 3.0 + 1.9))
    return np.stack([r * np.sin(th), ride_height(s), r * np.cos(th)], axis=-1)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-6)


def track_roll(s: np.ndarray, seed: float) -> np.ndarray:
    """Bank angle, taken from the actual plan curvature rather than a sine."""
    a = track_point(s - 9.0, seed)
    b = track_point(s, seed)
    c = track_point(s + 9.0, seed)
    d1 = _unit(b[..., ::2] - a[..., ::2])
    d2 = _unit(c[..., ::2] - b[..., ::2])
    turn = d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]
    return np.clip(turn * 11.0, -1.0, 1.0) * 0.62


def track_frame(s: np.ndarray, seed: float) -> tuple[np.ndarray, np.ndarray]:
    """Lateral and normal unit vectors at ``s``, banked with the track."""
    roll = track_roll(s, seed)
    tan = _unit(track_point(s + 1.5, seed) - track_point(s - 1.5, seed))
    up = np.stack([np.sin(roll), np.cos(roll), np.zeros_like(roll)], axis=-1)
    lat = _unit(np.cross(tan, up))
    return lat, np.cross(lat, tan)


def on_lift(s: float) -> bool:
    """True while the chain has hold of the train."""
    return ((s / CIRCUIT) % 1.0) < 0.34


class NightCoaster(BaseEffect):
    """First-person night lap lit entirely by the park's own emitters."""

    NAME = 'Night Coaster'
    AUTHOR = 'unicorn-viz'
    TAGS = ['rollercoast', 'ride', 'neon', 'intense', 'energetic', 'hard']
    PING_PONG_FRIENDS = [
        'First Drop', 'Corkscrew', 'Mine Train', 'Coaster Cam',
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
            'neon_gain': float(self.config.get('neon_gain', 1.0)),
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
        self._shake = 0.0

        # March table, filled in place each frame and uploaded as uniforms.
        f = np.arange(STEPS + 1, dtype=np.float64) / STEPS
        self._march_f2 = f * f
        self._tbl_p = np.zeros((STEPS + 1, 4), dtype='f4')
        self._tbl_l = np.zeros((STEPS + 1, 4), dtype='f4')
        self._tbl_n = np.zeros((STEPS + 1, 4), dtype='f4')
        self._probe = np.zeros(3, dtype=np.float64)
        self._build_march()

    def _build_march(self) -> None:
        """Sample the circuit ahead of the train and place the camera on it.

        Everything the shader needs about the track is produced here, once per
        frame: the fragment program has no track model in it at all. That is
        both the performance fix (the geometry was being re-derived per pixel)
        and the correctness one -- there is a single source of truth for the
        circuit rather than a GLSL copy and a Python copy free to drift.
        """
        s = self._travel + NEAR + self._march_f2 * FAR
        pts = track_point(s, self._seed)
        lat, nrm = track_frame(s, self._seed)
        self._tbl_p[:, :3] = pts
        self._tbl_p[:, 3] = s
        self._tbl_l[:, :3] = lat
        self._tbl_n[:, :3] = nrm

        # The camera rides the curve: position from it, orientation from its
        # tangent, roll from its lateral curvature.
        t = np.array([self._travel], dtype=np.float64)
        here = track_point(t, self._seed)[0]
        tangent = _unit(track_point(t + 1.5, self._seed)
                        - track_point(t - 1.5, self._seed))[0]
        ahead = _unit(track_point(t + 22.0, self._seed)[0] - here)
        fwd = _unit(tangent + (ahead - tangent) * 0.5)
        roll = float(track_roll(t, self._seed)[0])
        world_up = np.array([math.sin(roll), math.cos(roll), 0.0])
        right = _unit(np.cross(fwd, world_up))
        up = np.cross(right, fwd)
        shake = self._shake
        pos = here + up * 1.9 + right * shake * 0.25 + up * shake * 0.18

        p = self._prog
        p['iP'].write(self._tbl_p.tobytes())
        p['iL'].write(self._tbl_l.tobytes())
        p['iN'].write(self._tbl_n.tobytes())
        p['iCamPos'].value = tuple(float(x) for x in pos)
        p['iCamRight'].value = tuple(float(x) for x in right)
        p['iCamUp'].value = tuple(float(x) for x in up)
        p['iCamFwd'].value = tuple(float(x) for x in fwd)

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
        self._shake = ((self._bass * 0.5 + self._punch * 0.8)
                       )

        ahead = 5.0
        self._probe[:] = (self._travel - ahead, self._travel, self._travel + ahead)
        heights = ride_height(self._probe)
        h = float(heights[1])
        self._grade = float(np.clip((heights[2] - heights[0]) / (2 * ahead), -1.5, 1.5))

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
        p['iSky'].value = self._sky
        p['iGrade'].value = float(self._grade)
        p['iNeonGain'].value = max(0.0, float(self.parameters['neon_gain']))
        self._build_march()
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        """Release GL resources."""
        self._vao.release()
        self._vbo.release()
        self._prog.release()
