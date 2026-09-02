# effects-rollercoast — Unicorn Viz effect pack

**Version 0.1.0**

Rollercoaster effects: First Drop, Corkscrew, Night Coaster, Mine Train,
Coaster Cam.

Each effect subclasses `unicornviz.effects.base.BaseEffect` and is auto-discovered.

## How the pack works

Every effect rides the **same kind of real 3-D track curve** — a parametric
`trackPoint(s)` giving lateral offset, height and distance along the ride. The
camera is placed *on* that curve: position from the curve, orientation from its
tangent, roll from the lateral curvature, so banking falls out of the geometry
instead of being animated on top of it. Rails are drawn by marching the curve
from far to near and measuring each pixel's distance to the projected segment,
which is what lets a hill crest genuinely occlude the track beyond it.

Speed is integrated on the CPU from the profile's own gradient, so a train
accelerates downhill and drags going up. Audio rides on top of that rather than
replacing it, so each ride keeps its own physical rhythm.

## Effects

- **First Drop** — the chain lift, the hang at the crest, and a near-vertical
  plunge that dumps into a run of airtime camelbacks.
- **Corkscrew** — level, then a hard spiral through the vertical and out again.
  The rail pair banks by the same roll as the camera, so the structure
  corkscrews around the rider rather than the image simply spinning.
- **Night Coaster** — the ride after dark, lit only by what the park emits:
  emissive rails, a dense midway of lamps, and fireworks on the beat.
- **Mine Train** — a runaway cart underground. Timber hoops close the bore
  around the rider, torches burn along the wall, and the profile is a staircase
  of short sharp drops rather than one big hill.
- **Coaster Cam** — the odd one out: a chase camera alongside the structure,
  watching a train whose cars each sit at their own point on the curve, so the
  consist stretches on the drops and rolls through the banking one car at a
  time.

## Tweakables

All five take `speed`, `zoom`, `reactivity` and `glitter`, plus one of their
own: `drop_force` (First Drop), `roll_force` (Corkscrew), `neon_gain` (Night
Coaster), `torch_gain` (Mine Train), `cam_offset` (Coaster Cam).

## Changelog

- **0.1.0** — initial release: five rollercoaster effects on a shared real 3-D
  track core.
