# effects-rollercoast — Unicorn Viz effect pack

**Version 0.3.0**

Ride effects: First Drop, Corkscrew, Night Coaster, Mine Train, Coaster Cam,
Demon Drop, Log Flume.

Each effect subclasses `unicornviz.effects.base.BaseEffect` and is auto-discovered.

## How the pack works

Every effect rides the **same kind of real 3-D track curve** — a parametric
`trackPoint(s)` giving plan position, height and distance along the ride. The
camera is placed *on* that curve: position from the curve, orientation from its
tangent, roll from the lateral curvature, so banking falls out of the geometry
instead of being animated on top of it. The structure is drawn by marching the
curve from far to near and measuring each pixel's distance to the projected
segment, which is what lets a hill crest genuinely occlude the track beyond it.

**The track is a closed circuit.** `trackPoint(s)` is a loop that returns to its
own station, so every ride laps forever inside a bounded footprint and you see
the rest of the ride across the infield — which is most of what makes a coaster
read as a coaster rather than as a corridor. The height profile is continuous
across the lap seam, so nothing steps or spikes once per lap.

**Speed is energy.** Away from a lift, a train's speed is `v = sqrt(2 g (H_top -
h))` — it got everything it has by falling — so it crawls over summits and
screams through valleys because the geometry demands it, not because a curve
says so. Lifts, chains, conveyors, cable hoists and brake fins each set their
own pace where a real ride's would. Audio rides on top of that rather than
replacing it, so each ride keeps its own physical rhythm.

**The structure is a real one.** Each section carries running rails over a box
spine, tied by zig-zag truss lacing and cross ties, standing on braced support
bents with footers. Rides that need something else build it: Mine Train's timber
hoops, Demon Drop's four-legged lattice mast, Log Flume's trestle bents and
filled-quad trough.

## Effects

- **First Drop** — the chain lift, the crest, and a plunge that dumps into a run
  of airtime camelbacks before the brake run brings it home.
- **Corkscrew** — level, then a hard spiral through the vertical and out again.
  The roll is periodic across the lap, so the inversion happens at the same
  place every lap and unwinds completely afterwards.
- **Night Coaster** — the ride after dark, lit only by what the park emits:
  emissive rails, a dense midway of lamps, and fireworks on the beat.
- **Mine Train** — a runaway cart underground. Timber hoops close the bore
  around the rider, torches burn along the wall, and the profile is a staircase
  of short sharp drops rather than one big hill.
- **Coaster Cam** — the odd one out: a chase camera alongside the structure,
  watching a train whose cars each sit at their own point on the curve, so the
  consist stretches on the drops and rolls through the banking one car at a
  time.
- **Demon Drop** — the freefall tower. The plan shape is phase-warped so the
  track can go vertical without the ride frame degenerating: the car hoists up a
  lattice mast, **hangs** at the crown tipped forward over the drop, falls, and
  curves out onto the brake run.
- **Log Flume** — a water circuit built from surfaces rather than lines: two
  conveyors, a meander, and a splashdown, seen from the front seat of the log
  with the trough painted as filled quads around you.

## Tweakables

All seven take `speed`, `zoom`, `reactivity` and `glitter`, plus one of their
own: `drop_force` (First Drop), `roll_force` (Corkscrew), `neon_gain` (Night
Coaster), `torch_gain` (Mine Train), `cam_offset` (Coaster Cam), `hang_tension`
(Demon Drop), `splash_gain` (Log Flume).

## Changelog

- **0.3.0** — the pack ran at 8-11 fps at 1080p; it now runs at 50-104. The
  cause was structural: the fragment shader carried the whole track model, so
  every pixel re-derived the curve. `trackPoint()` was evaluated about 560
  times per pixel per frame — **1.2 billion times a frame** at 1080p, each with
  three sines and a branch chain. The track is now sampled once per frame on
  the CPU into a small uniform table (position, lateral, normal per section)
  along with the camera basis, and the shader has no track model in it at all:
  it projects the table and rasterizes. That also leaves one source of truth
  for the geometry instead of a GLSL copy and a Python copy free to drift.

  Three further fixes came out of measuring rather than guessing. Support bents
  bypassed the section cull, so every far section paid four projections per
  pixel whether or not the bent was near it — 68% of the frame. Structure
  density was tied to the *sampling* rather than to distance, so near sections
  stacked three or four bents in almost the same place. And in Mine Train the
  hoops and torches were drawn after the section cull, which clipped the bore's
  rings away entirely and drew the torchlight as hard rectangles — the cull box
  itself, made visible. A texture-backed table was also tried for the march and
  measured 2x *slower* than uniform arrays on this driver, so it was dropped.

  Demon Drop and Log Flume are rebuilt on the same core. Demon Drop's lattice
  mast is now built and projected on the CPU too, and the camera faces outward
  on the hoist and tips forward through the hang, so the climb looks over the
  park and the hang looks down the shaft instead of at empty sky. Log Flume
  keeps its filled-quad trough, now with a rounded hull and bow wave in the
  bottom of frame rather than a translucent wedge.

- **0.2.0** — 2.0 uplift: closed-circuit track on every ride (no more unbounded
  travel or per-lap height step), energy-conservation physics, bounded periodic
  corkscrew roll, and truss/bent structure throughout. Adds Demon Drop and
  Log Flume.
- **0.1.0** — initial release: five rollercoaster effects on a shared real 3-D
  track core.
