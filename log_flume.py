"""Log Flume — a water circuit, ridden from the front seat of the log.

The other rides in this pack are built from lines: rails, chords, bracing.  A
flume is built from *surfaces*, so this one adds a filled-quad primitive on top
of the shared closed-circuit model.  Each marched section paints a real trough
— two timber side walls and the water between them — as screen-space quads
with a proper triangle-coverage test, which is what makes the channel read as
something you are floating *inside* rather than a line you are following.  The
trough is carried on timber trestle bents with X-bracing and sill beams, the
way a real flume's runs are.

The water moves.  Its shimmer is driven along the channel by the log's own
travel, so the surface streams past at the speed the ride is actually doing,
and the splashdown throws a sheet of spray over the camera that decays across
the run-out.

Two conveyors, a baby drop and the big splashdown.  Away from the conveyors the
log runs on energy conservation like the coasters, but with a water-drag
ceiling on the run-out — a flume that kept its splashdown speed all the way
home would look wrong, and a real one does not.

**The track is sampled on the CPU, once per frame.** The geometry does not vary
per pixel, but deriving it inside the fragment loop meant about 560
``trackPoint()`` evaluations per pixel per frame — 1.2 billion at 1080p — which
is what pinned this pack at 8 fps. ``_build_march()`` samples the circuit into
a small uniform table and hands the shader the camera basis too, so the
fragment program has no track model in it at all: it projects the table and
rasterizes.

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

__version__ = '0.5.0'

# One lap, in metres. Shared by the CPU physics and the track sampler.
CIRCUIT = 780.0
# Crest of the ride; the energy model measures every fall from just above it.
H_TOP = 32.0            # the splashdown crest
# ── speed model ─────────────────────────────────────────────────────────────
# A coaster's true energy curve -- v = sqrt(2 g (H_top - h)) -- is physically
# right and visually wrong here. It parks the ride on a chain lift for a third
# of the lap and lets it creep over every summit: measured on the 0.3.0
# profile, 43% of each lap ran under 12 m/s, with a 4.7x spread between the
# crest and the bottom of the drop. That is a long time to spend crawling in
# something whose job is to carry the energy of the music.
#
# The *shape* of the energy curve is what sells a drop, so it is kept -- speed
# still comes from how far the train has fallen -- but it is remapped onto a
# band that never creeps, and the result is passed through a first-order lag so
# no seam between profile branches, and no handover between the chain and
# gravity, can step the speed.
CRUISE = 26.0            # m/s, the pace the ride holds when nothing is happening
_SPEED_LO = 0.78         # multiple of CRUISE at a crest, and on the lift
_SPEED_HI = 1.52         # multiple of CRUISE at the lowest point on the circuit
_LIFT_MULT = 0.88        # multiple of CRUISE while the chain has hold
_SPEED_TAU = 0.30        # s, lag constant; smooths every profile seam
LAP_R = 150.0
TAU = 2.0 * math.pi

# March: how many track sections are sampled ahead of the train, and how far.
STEPS = 40
NEAR = 2.0
FAR = 400.0

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
uniform float iCruise;     // 1 on the flat and climbing, 0 in a plunge
uniform float iRing;       // beat shockwave, 0 at the beat -> 1 spent
// Embers, placed on the CPU: xy = screen position, z = remaining life.
// Deriving them per fragment cost seven hashes and the live registers to
// hold them, and this shader sits right on an occupancy cliff -- with the
// per-pixel version the march loop itself ran twice as slow.
uniform vec4  iEmber[7];
// The march table, sampled on the CPU once per frame. xyz is the world point
// on the centreline; iP.w is s, the distance along the circuit, which the
// structure uses to place ties and bents.
uniform vec4  iP[41];
uniform vec4  iL[41];      // xyz = lateral unit vector at that point
uniform vec4  iN[41];      // xyz = normal ("up" out of the track bed)
uniform vec3  iCamPos;
uniform vec3  iCamRight;
uniform vec3  iCamUp;
uniform vec3  iCamFwd;
uniform float iSplash;      // 1 at splashdown, decaying over the run-out
uniform float iSplashGain;  // tweakable: how much water gets thrown
in  vec2 v_uv;
out vec4 fragColor;

#define PI  3.14159265
#define TAU 6.28318530
#define STEPS 40

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

// Coverage of a screen-space triangle: 1 inside, 0 outside. The three edge
// cross-products agree in sign only for points within the triangle, whichever
// way round it was wound.
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
float quadFill(inout vec3 col, vec2 p, vec3 A, vec3 B, vec3 C, vec3 D,
               vec3 c, float alpha) {
    vec2 a, b, cc, d; float za, zb, zc, zd;
    if (!project(A, a, za) || !project(B, b, zb) ||
        !project(C, cc, zc) || !project(D, d, zd)) return 0.0;
    if (p.x < min(min(a.x, b.x), min(cc.x, d.x)) ||
        p.x > max(max(a.x, b.x), max(cc.x, d.x)) ||
        p.y < min(min(a.y, b.y), min(cc.y, d.y)) ||
        p.y > max(max(a.y, b.y), max(cc.y, d.y))) return 0.0;
    float cov = max(triCover(p, a, b, cc), triCover(p, a, cc, d));
    col = mix(col, c, cov * alpha);
    return cov;
}

vec3 skyColour(vec2 p, float tb) {
    float up = clamp(p.y * 0.9 + 0.35 - iGrade * 0.35, 0.0, 1.0);
    vec3 lo, hi;
    if (iSky < 0.34)      { lo = vec3(0.98, 0.72, 0.40); hi = vec3(0.05, 0.20, 0.42); }
    else if (iSky < 0.67) { lo = vec3(0.55, 0.88, 0.85); hi = vec3(0.04, 0.16, 0.46); }
    else                  { lo = vec3(0.75, 0.35, 0.62); hi = vec3(0.03, 0.06, 0.24); }
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
    float shake = iBass * 0.5 + iPunch * 0.8;
    // A short punch of field of view on the beat: the cheapest thing that
    // makes a first-person ride feel like it is being hit by the music.
    vec2 p = (v_uv - 0.5) * vec2(A, 1.0)
           / max(iZoom * (1.0 - iPunch * 0.055), 0.05);
    p += vec2(sin(tb * 47.0), sin(tb * 39.0 + 1.1)) * shake * 0.006;

    vec3 col = skyColour(p, tb);

    vec3 rd = normalize(iCamFwd + p.x * iCamRight + p.y * iCamUp);
    if (rd.y < -0.0015 && iCamPos.y > 0.2) {
        float td = -iCamPos.y / rd.y;
        if (td > 0.0 && td < 900.0) {
            vec3 g = iCamPos + rd * td;
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

    // ── the structure, marched far to near so crests occlude what is beyond ──
    vec3 timber = mix(vec3(0.36, 0.24, 0.15), palette(iHue + 0.58), 0.30);
    vec3 capCol = mix(vec3(0.62, 0.46, 0.28), palette(iHue + 0.20), 0.35);
    vec3 waterC = mix(vec3(0.05, 0.33, 0.46), palette(iHue + 0.48), 0.28);
    vec3 flowC  = mix(vec3(0.45, 0.90, 1.0), palette(iHue + 0.3), 0.30);
    const float HALFW = 2.0, WALL = 1.5, DECK = 1.0;

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

        if (inSection) {
        // Outer faces of the two side walls, then the water between them.
        for (int w = -1; w <= 1; w += 2) {
            float sw = float(w);
            vec3 lo0 = c0 + l0 * (HALFW * sw) - n0 * DECK;
            vec3 lo1 = c1 + l1 * (HALFW * sw) - n1 * DECK;
            vec3 hi0 = c0 + l0 * (HALFW * sw) + n0 * WALL;
            vec3 hi1 = c1 + l1 * (HALFW * sw) + n1 * WALL;
            quadFill(col, p, lo0, lo1, hi1, hi0,
                     timber * (0.35 + fade * 0.85) * (w < 0 ? 0.82 : 1.0), 1.0);
            strut(col, p, hi0, hi1, 0.13, capCol * (0.4 + fade * 0.9), 0.0);
            if (f0 < 0.30 && hasTie) {
                strut(col, p, lo0, hi0, 0.05, timber * 0.55 * fade, 0.0);
            }
        }

        float cov = quadFill(col, p, c0 - l0 * HALFW, c0 + l0 * HALFW,
                             c1 + l1 * HALFW, c1 - l1 * HALFW,
                             waterC * (0.45 + fade * 0.9), 1.0);
        if (cov > 0.0) {
            // Surface streaming past at the speed the log is actually doing.
            float flow = sin(s0 * 0.9 - iTravel * 0.55 + p.x * 5.0);
            col += flowC * smoothstep(0.55, 1.0, flow) * cov * fade
                 * (0.14 + iBass * 0.35);
            col += vec3(0.9, 0.98, 1.0) * cov
                 * step(0.994, hash21(floor(vec2(p.x * 210.0,
                                                 p.y * 210.0 + iTravel * 3.0))))
                 * (0.35 + iTreble * 1.1) * iGlitter;
        }
        }

        // Support bents: paired legs, a cross brace and footers. A bent reaches
        // all the way to grade, far outside its section's screen box, so it
        // carries its own -- four corners projected, then one box test gates
        // the whole assembly. Letting it bypass the cull instead cost 68% of
        // the frame.
        if (hasBent) {
            vec3 baseL = vec3((c0 - l0 * 2.7).x, 0.0, (c0 - l0 * 2.7).z);
            vec3 baseR = vec3((c0 + l0 * 2.7).x, 0.0, (c0 + l0 * 2.7).z);
            vec3 topL = c0 - n0 * DECK - l0 * 2.2;
            vec3 topR = c0 - n0 * DECK + l0 * 2.2;
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
                vec3 legCol = timber * (0.30 + fade * 0.8);
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

    // The log's own hull filling the bottom of frame, and its bow wave. The
    // hull edge is a shallow parabola, so the nose reads as rounded rather
    // than as a wedge across the channel.
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
            vec2 dv = p - dp;
            float q = dot(dv, dv);
            if (q < 0.006) col += vec3(0.85, 0.96, 1.0) * exp(-q * 1100.0)
                                * (1.0 - ph) * splash * (0.8 + iTreble * 0.8);
        }
    }

    col += palette(iHue + 0.45) * iBeat * 0.10;
    // ── atmosphere ──────────────────────────────────────────────────────
    // Everything here is scaled by iCruise, so it swells on the lift, over a
    // crest and along the flats -- exactly where the track geometry stops
    // carrying the frame on its own -- and gets out of the way in a plunge,
    // where the speed is doing the work.
    float air = iCruise;

    // Light sweeping across the backdrop, driven by the low end.
    float sweep = sin(p.x * 3.1 - p.y * 1.5 + tb * 1.05)
                * sin(p.y * 2.3 + tb * 0.61);
    col += palette(iHue + 0.35) * smoothstep(0.45, 1.0, sweep)
         * air * (0.09 + iBass * 0.42);

    // A shockwave thrown out of the vanishing point on every beat. iRing is
    // integrated on the CPU, so it expands at a fixed rate rather than being
    // tied to whatever the frame rate happens to be.
    float rr = length(p - vec2(0.0, 0.03));
    float ringR = iRing * 1.35;
    col += palette(iHue + 0.62)
         * smoothstep(0.075, 0.0, abs(rr - ringR)) * (1.0 - iRing)
         * (0.30 + air * 1.05) * (0.5 + iBass * 0.8);

    // Embers riding the airflow, thickest when the ride is not plunging.
    vec3 emberA = palette(iHue + 0.10), emberB = palette(iHue + 0.42);
    float emberG = air * (0.45 + iTreble * 0.9) * iGlitter;
    for (int i = 0; i < 7; i++) {
        vec2 ev = p - iEmber[i].xy;
        float eq = dot(ev, ev);
        if (eq < 0.004) col += mix(emberA, emberB, float(i) * 0.1667)
                             * exp(-eq * 1500.0) * iEmber[i].z * emberG;
    }

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
    d_fall = _sstep(0.62, 0.74, u)
    return np.select(
        [u < 0.08, u < 0.22, u < 0.34, u < 0.40, u < 0.58, u < 0.62, u < 0.74],
        [np.full_like(u, 2.0),                              # station
         2.0 + _sstep(0.08, 0.22, u) * 10.0,                # first conveyor
         12.0 - _sstep(0.22, 0.34, u) * 5.0,                # baby drop
         np.full_like(u, 7.0),                              # meander
         7.0 + _sstep(0.40, 0.58, u) * 25.0,                # big conveyor
         np.full_like(u, 32.0),                             # the crest
         32.0 - d_fall * d_fall * 30.0],                    # splashdown
        np.full_like(u, 2.0))                               # run-out


# The lowest point on the circuit, which anchors the speed band.
H_LOW = float(ride_height(np.linspace(0.0, CIRCUIT, 512)).min())

def track_point(s: np.ndarray, seed: float) -> np.ndarray:
    """The circuit's centreline: a closed curve, so coordinates stay bounded."""
    th = ((s / CIRCUIT) % 1.0) * TAU
    r = LAP_R * (1.0 + 0.22 * np.sin(th * 2.0 + seed * 6.3)
                     + 0.11 * np.sin(th * 3.0 + 1.4))
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
    # A flume's water does not bank; only the channel leans a little.
    return np.clip(turn * 8.0, -1.0, 1.0) * 0.20


def track_frame(s: np.ndarray, seed: float) -> tuple[np.ndarray, np.ndarray]:
    """Lateral and normal unit vectors at ``s``, banked with the track."""
    roll = track_roll(s, seed)
    tan = _unit(track_point(s + 1.5, seed) - track_point(s - 1.5, seed))
    up = np.stack([np.sin(roll), np.cos(roll), np.zeros_like(roll)], axis=-1)
    lat = _unit(np.cross(tan, up))
    return lat, np.cross(lat, tan)


def lift_weight(s: float) -> float:
    """How much of a conveyor has hold of the log, as a smooth 0..1 window.

    Two belts, each faded in and out. A hard boolean here is what produced the
    one real jerk left in the speed trace: crossing onto a belt stepped the
    target by 16 m/s in a single frame.
    """
    u = (s / CIRCUIT) % 1.0
    first = _sstep(0.06, 0.11, u) * (1.0 - _sstep(0.19, 0.25, u))
    second = _sstep(0.38, 0.43, u) * (1.0 - _sstep(0.55, 0.61, u))
    return float(min(1.0, first + second))


class LogFlume(BaseEffect):
    """First-person water flume: conveyors, meander and a splashdown."""

    NAME = 'Log Flume'
    AUTHOR = 'unicorn-viz'
    TAGS = ['rollercoast', 'ride', 'water', 'neon', 'energetic', 'chill']
    PING_PONG_FRIENDS = [
        'First Drop', 'Corkscrew', 'Night Coaster', 'Mine Train',
        'Coaster Cam', 'Audio Spectrum',
    ]


    # Widest the view may go. `zoom` divides the screen coordinate, so it is a
    # field of view control rather than a dolly: at 1.0 the horizontal FOV is
    # about 83 degrees, and every step down widens it. Below about 0.8 the
    # rails -- which are right beside and below a rider -- project out to the
    # frame corners, and the shot stops reading as riding the track and starts
    # reading as standing behind it. Measured across the pack: 0.85 is the last
    # value that still reads correctly, and the beat's field-of-view punch off
    # this floor lands at 0.80, which is also clean.
    # Four steps (0.10 each) tighter than the other rides. The flume's
    # channel is open water with the scenery set well back, so the same field
    # of view that frames a coaster's track reads as standing much too far
    # away here. The default moves with the floor -- a floor above the default
    # would clamp it anyway, and silently.
    _MIN_ZOOM = 1.25

    def _init(self) -> None:
        self.parameters = {
            'speed': float(self.config.get('speed', 1.0)),
            'zoom': float(self.config.get('zoom', 1.25)),
            'reactivity': float(self.config.get('reactivity', 1.3)),
            'glitter': float(self.config.get('glitter', 1.0)),
            'splash_gain': float(self.config.get('splash_gain', 1.0)),
        }
        # March budget: the config editor's Effects tab pins 'march_steps';
        # the shader bakes it in, so a change lands at the next activation.
        self._steps = int(max(12, min(96, self.parameters.setdefault(
            'march_steps', float(self.config.get('march_steps', STEPS))))))
        self._prog = self._make_program(
            _VERT, _FRAG.replace(f'#define STEPS {STEPS}', f'#define STEPS {self._steps}'))
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
        self._v = CRUISE
        self._cruise = 0.5
        self._ring = 1.0
        self._ember_i = np.arange(7, dtype=np.float64)
        self._tbl_ember = np.zeros((7, 4), dtype='f4')
        self._splash = 0.0

        # March table, filled in place each frame and uploaded as uniforms.
        f = np.arange(self._steps + 1, dtype=np.float64) / self._steps
        self._march_f2 = f * f
        self._tbl_p = np.zeros((self._steps + 1, 4), dtype='f4')
        self._tbl_l = np.zeros((self._steps + 1, 4), dtype='f4')
        self._tbl_n = np.zeros((self._steps + 1, 4), dtype='f4')
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
            self._ring = 0.0
        self._ring = min(1.0, self._ring + dt * 1.45)
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
        u = (self._travel / CIRCUIT) % 1.0
        # Where the ride wants to be, then a lag toward it. Reading the target
        # straight off the profile is what made hills and even slight inclines
        # jerk: the branch seams step the gradient, and the chain/gravity
        # handover stepped the speed outright.
        fall = (H_TOP - h) / max(H_TOP - H_LOW, 1e-3)
        rolling = CRUISE * (_SPEED_LO + (_SPEED_HI - _SPEED_LO)
                            * math.sqrt(min(1.0, max(0.0, fall))))
        w = lift_weight(self._travel)
        target = rolling + (CRUISE * _LIFT_MULT - rolling) * w
        self._v += (target - self._v) * (1.0 - math.exp(-dt / _SPEED_TAU))
        v = self._v

        # How much the ride is *not* plunging: 1 on the lift and over a crest,
        # 0 at full speed. The backdrop uses it to pick up exactly where the
        # geometry stops carrying the frame on its own.
        span = CRUISE * (_SPEED_HI - _SPEED_LO)
        self._cruise = float(np.clip(
            1.0 - (v - CRUISE * _SPEED_LO) / span, 0.0, 1.0))

        # The hull hits the pool at the foot of the splashdown; the sheet of
        # water it throws hangs over the camera for a moment afterwards.
        if 0.72 <= u < 0.76:
            self._splash = 1.0
        self._splash = max(0.0, self._splash - dt * 0.9)

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
        p['iZoom'].value = max(self._MIN_ZOOM,
                               float(self.parameters['zoom']))
        p['iGlitter'].value = max(0.0, float(self.parameters['glitter']))
        p['iTravel'].value = float(self._travel)
        p['iSky'].value = self._sky
        p['iGrade'].value = float(self._grade)
        p['iCruise'].value = float(self._cruise)
        p['iRing'].value = float(self._ring)

        # Ember placement is identical for every pixel, so it happens here.
        k = (self.time % 300.0) * (0.11 + self._ember_i * 0.017) + self._ember_i * 0.371
        eph = k % 1.0
        ehx = (np.sin(np.floor(k) * 12.9898 + self._ember_i * 78.233)
               * 43758.5453) % 1.0
        self._tbl_ember[:, 0] = (ehx - 0.5) * 1.7
        self._tbl_ember[:, 1] = -0.55 + eph * 1.15
        self._tbl_ember[:, 2] = 1.0 - eph
        p['iEmber'].write(self._tbl_ember.tobytes())
        p['iSplash'].value = float(self._splash)
        p['iSplashGain'].value = max(0.0, float(self.parameters['splash_gain']))
        self._build_march()
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        """Release GL resources."""
        self._vao.release()
        self._vbo.release()
        self._prog.release()
