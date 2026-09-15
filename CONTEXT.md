# MiniRes

MiniRes estimates sliced resin mass for pre-supported miniatures.
Its estimates are distinct from a shop's actual resin consumption and prices.

## Language

**Pre-supported miniature**:
A miniature whose input geometry already includes printing supports.

**Sliced resin mass**:
The resin mass in grams reported for a sliced object under specified slicing conditions, including resin density. This is a slicer-derived reference, not a measurement of actual shop consumption.
_Avoid_: Actual consumption, measured consumption

**Estimated sliced resin mass**:
A prediction of sliced resin mass from a miniature's geometry, within the estimator's validated scope.
_Avoid_: Price prediction, actual resin usage

**Operational allowance**:
An allowance for resin losses from failures, handling, and residue, considered separately from estimated sliced resin mass when calculating prices.

**Anonymous source group**:
A grouping of examples by their originating source without publishing the source's identity. It supports private evaluation, not prediction input.

**Unseen-source performance**:
Prediction quality on examples from a source excluded from model fitting and selection for that evaluation.

**Miniature family**:
Related parts and variants of a miniature grouped together for evaluation, so related examples do not appear on both sides of an evaluation split.

**Validated scope**:
The input categories and slicing conditions for which evaluation supports the estimator's use. Geometry checks alone cannot establish that an input belongs to this scope.

**Needs review**:
An outcome indicating that an input is invalid, outside the validated scope, or lacks required scope confirmation. Its reason distinguishes those cases from a usable estimate.

**Prediction interval**:
A range for sliced resin mass whose stated coverage is supported by held-out evaluation within a defined scope. It is distinct from an operational allowance.

**Surface-integral mesh volume**:
The signed, additive volume implied by a mesh's oriented triangle shells. It is not occupied union volume: overlapping closed shells can be counted more than once, reversed shells can subtract, and open or inconsistently wound meshes do not support a reliable solid-volume interpretation.
_Avoid_: Occupied volume, union volume

**Axis-aligned bounding-box volume**:
The product of the three extents of the axis-aligned box enclosing a mesh's vertices. A surface-integral mesh volume is not universally bounded by this value because the two measurements have different semantics.
_Avoid_: Bounding-box area

**Bounding-box occupancy factor**:
A label-derived training representation equal to sliced resin volume divided by axis-aligned bounding-box volume. It is not assumed to be at most one and is not evidence of physical occupied-union volume.
_Avoid_: Physical occupancy, occupied-volume fraction
