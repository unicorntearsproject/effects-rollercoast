# effects-rollercoast — Unicorn Viz effect pack

**Version 0.4.0**

Ride effects: First Drop, Corkscrew, Night Coaster, Mine Train, Coaster Cam,
Log Flume.

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

**Speed follows energy, on a band that never creeps.** A train's pace still
comes from how far it has fallen — that shape is what sells a drop — but the
raw energy curve `v = sqrt(2 g (H_top - h))` is physically right and visually
wrong here: it parks the ride on a chain lift for a third of the lap and lets
it creep over every summit. The curve is remapped onto a band running about
0.78x to 1.5x of a cruising pace, and the result passes through a first-order
lag, so no seam between profile branches and no handover between the chain and
gravity can step the speed. Audio rides on top of that.

**The structure is a real one.** Each section carries running rails over a box
spine, tied by zig-zag truss lacing and cross ties, standing on braced support
bents with footers. Rides that need something else build it: Mine Train's timber
hoops and Log Flume's trestle bents and filled-quad trough.

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
- **Log Flume** — a water circuit built from surfaces rather than lines: two
  conveyors, a meander, and a splashdown, seen from the front seat of the log
  with the trough painted as filled quads around you.

## Tweakables

All six take `speed`, `zoom`, `reactivity` and `glitter`, plus one of their
own: `drop_force` (First Drop), `roll_force` (Corkscrew), `neon_gain` (Night
Coaster), `torch_gain` (Mine Train), `cam_offset` (Coaster Cam), `splash_gain`
(Log Flume).

## Changelog

- **0.4.0** — the rides no longer creep, and the backdrops pick up where the
  track stops carrying the frame.

  *Pace.* The energy model was physically faithful and no fun: **43% of every
  lap ran under 12 m/s**, with a 4.7x spread between the crest and the bottom of
  the drop and laps of 59-87 seconds. Speed still comes from how far the train
  has fallen, but that curve is now remapped onto a band that never creeps and
  smoothed with a lag. Measured over two full laps per ride: **22.9-39.5 m/s
  (1.73x), 0% of the lap under 12 m/s, laps of 33-37 s**.

  *Smoothness.* The chain-to-gravity handover was a hard boolean, so entering
  the lift stepped the target 16 m/s in one frame — **53.8 m/s²** through the
  lag, which is the jerk on hills and inclines. The handover is now a smooth
  window and peak acceleration is **13.6 m/s²**, about 1.4 g, which is a figure
  a real coaster actually pulls and is meant to be felt.

  *Atmosphere.* Every ride carries an `iCruise` signal — 1 on a lift, over a
  crest and along the flats, 0 in a plunge — and the backdrop swells with it:
  light sweeping across the sky, a shockwave thrown out of the vanishing point
  on each beat, and embers riding the airflow. Speed lines now stretch and
  brighten with how fast the ride is genuinely going, and the beat lands a short
  punch of field of view.

  *Mine Train* gets the biggest immersion change: its rock was a static screen
  texture, so the walls never moved while the rails ran underneath them. The
  bore is now textured in log-polar space and scrolled along the radial axis,
  which is exactly the outward expansion perspective produces, so the wall
  really flies past — with ore veins in it that light up on the mids. *Night
  Coaster* gets a ring of light running out across the midway on the beat.

  Particle placement (embers, sparks, firework shells) moved to the CPU. These
  shaders sit on an occupancy cliff: with the per-pixel version, the march loop
  itself measured twice as slow, and removing either the fireworks or the star
  field alone changed nothing while removing both halved the frame.

  **Demon Drop is retired** at the owner's request.

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
