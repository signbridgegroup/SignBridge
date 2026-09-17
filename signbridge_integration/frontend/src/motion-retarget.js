import * as THREE from 'three';

/* =====================================================================
 * SignBridge motion retargeter - SOURCE FAITHFUL MODE + PALM PITCH CALIBRATION
 * ---------------------------------------------------------------------
 * WHAT CHANGED vs the previous version, and why:
 *
 * 1. FINGERS ARE ANIMATED AGAIN.  This is the whole ball game.
 *    The previous active path was:
 *        BakedAvatarMotion.sample -> applyContinuousFrame
 *            -> #orientArmContinuous   (arm)
 *            -> #orientHandContinuous  (hand bone ONLY)
 *    The finger loop lived in #orientHand(), which nothing called any
 *    more.  Fingers therefore sat frozen at rest for the entire clip.
 *    That regression entered with motion-retarget.before-spline.js, when
 *    #orientHandFiltered replaced #orientHand and dropped the loop.
 *    prepareFilteredTracks() now builds 15 filtered direction tracks per
 *    hand, and #orientFingersContinuous() drives them.
 *
 * 2. FOREARM TWIST BONES ARE DRIVEN.  The GLB contains LeftForeArm1 /
 *    LeftForeArm2 (and the Right pair) as SIBLINGS of the hand under
 *    the forearm - they are skinning twist helpers.  Nothing wrote to
 *    them, so all wrist roll landed on one joint and pinched the mesh.
 *    They now receive a share of the hand's twist.  They are siblings,
 *    not parents, so this cannot disturb palm orientation.
 *
 * 3. CONFIDENCE-WEIGHTED FINGER DRIVE.  stack.motion.json reports
 *    rightHandRate 0.388 - the right hand is only really seen in 45 of
 *    116 frames.  Interpolating a handshape across an 8-frame dropout
 *    produces confident-looking nonsense.  Fingers now ease toward the
 *    neutral shape while confidence is low instead of following it.
 *
 * 4. #applyFingerRelax NO LONGER OVERWRITES restQuaternions.  It kept a
 *    third definition of "rest" that disagreed with main.js's own copy.
 *    Neutral finger pose now lives in its own map.
 *
 * 5. DEAD CODE REMOVED: #orientBoneWithReference, #captureFingerReferences,
 *    fingerBases, fingerReferenceSigns, smoothQuaternionTrack,
 *    limitQuaternionSpeed, keepQuaternionHemisphere, maximumStepForBone,
 *    apply(), applySequential(), #orientArm(), #orientHand().  None were
 *    reachable from the active path; together they were roughly 40% of
 *    the file and made it very hard to see what actually ran.
 *
 * WHAT DID NOT CHANGE, because measurement says it was already correct:
 *    - AXIS_SIGN {x:+1, y:-1, z:-1}.  Verified against the clip's own
 *      poseWorld: subject-left is +x (L_shoulder.x +0.156 vs R -0.166,
 *      matching the rig's LeftArm at world x +0.150); raw +y is DOWN
 *      (shoulder.y -0.441 vs hip.y -0.011); raw -z is toward the camera
 *      (nose.z -0.273 vs hips.z +0.002), so z:-1 maps the front of the
 *      body to +z.  The "z still not independently confirmed" note in
 *      the old header is now resolved: -1 is right.
 *    - The palm-basis construction.  Both rig bases measure det = +1.000,
 *      and the rig and the landmark target are built by the identical
 *      formula, so the mirror in the rig cancels out.  No handedness bug.
 *    - The two-bone IK, One-Euro filtering, Hermite gap fill and
 *      Catmull-Rom sampling.  These are sound and stay as they were.
 * ===================================================================== */

/* ---------------------------------------------------------------------
 * TUNING - every knob in one place.  Mutate at runtime from the debug
 * panel in main.js; nothing caches these values.
 * ------------------------------------------------------------------- */
export const TUNING = {
  // Finger flexion multiplier.  The source handshape is usually
  // under-curled once retargeted, because MediaPipe's finger landmarks
  // sit on the skin surface rather than the joint centres.  1.0 = raw.
  fingerGain: 1.0,
  thumbGain: 1.0,

  // SOURCE FAITHFUL MODE:
  // direct FK from measured shoulder->elbow and elbow->wrist directions.
  faithfulMode: true,

  // 0 = raw measured arm directions. Raise only if raw playback is visibly jittery.
  faithfulSmoothingHz: 0,

  // Only trust palm orientation on frames with genuine hand detection.
  faithfulPalmRequiresDetection: true,

  // Constant palm pitch calibration in degrees.
  // Start at 0 and calibrate visually on the exact source take.
  palmPitchOffsetLeftDeg: 0,
  palmPitchOffsetRightDeg: 0,

  // Bridge short internal hand-detection gaps with SLERP.
  maxInterpolatedGapFrames: 10,
  edgeBlendFrames: 4,

  // Forearm screen-plane correction.
  // Left uses the 2D image-landmark direction in screen plane while
  // preserving depth from poseWorld. Right stays untouched.
  carryEdgeHandPose: false,
  forearmImagePlaneLeft: 0,
  forearmImagePlaneRight: 0,

  // Minimal depth-only clearance.
  // Direct FK stays intact. When the 2D wrist lies over the torso region,
  // both upper-arm and forearm directions receive only a small forward-Z
  // component. No lateral X/Y push, no IK, no bone-length scaling.
  depthOnlyClearance: true,
  depthOnlyBiasMeters: 0.030,      // maximum: 3 cm
  depthOnlyForwardSign: 1,         // +1 = toward camera in current mapped axes
  depthOnlyTorsoPadding: 0.035,    // normalized-image padding around torso box
  depthOnlyMinWeight: 0.10,

  // Hard anatomical ceilings, radians, per joint index (1 = MCP knuckle).
  maxFlex: { 1: 1.60, 2: 1.92, 3: 1.40 },
  maxThumbFlex: { 1: 1.05, 2: 1.20, 3: 1.20 },

  // How much a finger may straighten past its neutral pose.
  maxExtend: 0.40,

  // Below this per-frame confidence the fingers ease back to neutral
  // instead of following interpolated data.
  fingerConfidenceFloor: 0.35,

  // SOURCE-TRUTH finger gate.
  // In faithful mode, never drive finger bones from an interpolated hand
  // across a source frame where hand detection is missing.  This prevents
  // synthetic finger geometry from entering the torso during detection gaps.
  // Recovered frames that were validated upstream still have HandDetected=true,
  // so they remain usable.
  faithfulFingersRequireDetection: true,

  // Palm measurement authority.
  // IMPORTANT FOR THIS FIRST TEST: 0 means the hand follows the forearm
  // with its rest-local rotation, without an independent measured palm override.
  // If this restores the gradual rise, increase later to 0.25 or 0.55.
  palmAuthority: 0.0,

  // Below this confidence, ignore measured palm orientation.
  palmConfidenceFloor: 0.45,

  // Maximum allowed measured palm target rotation per source frame.
  palmMaxStepDeg: 6.0,

  // Quaternion smoothing passes for measured palm orientation.
  palmSmoothPasses: 2,

  // Share of the hand's twist handed to each forearm helper bone.
  // Roughly matches how pronation distributes along a real radius/ulna.
  twistShare1: 0.33,
  twistShare2: 0.66,
  maxForearmTwist: 1.45, // rad, ~83 deg

  // Per-side master switches.  Turn a hand off when its source data is
  // too sparse to be worth showing (see detection rates in the JSON).
  driveLeftFingers: true,
  driveRightFingers: true,

  // Blend the avatar's authored (splayed) bind-pose hand toward a flatter
  // neutral.  0 = the GLB's own shape.  Only seen when confidence is low.
  fingerRelax: 0,

  // Set true to feed identity everywhere - the avatar must hold its bind
  // pose exactly.  The fastest way to prove the basis maths is intact.
  identityTest: false,
};

const POSE = Object.freeze({
  leftShoulder: 11,
  rightShoulder: 12,
  leftElbow: 13,
  rightElbow: 14,
  leftWrist: 15,
  rightWrist: 16,
});

// MediaPipe hand landmark chains.  Bone <finger><n> is aimed from
// landmark[n-1] toward landmark[n], e.g. Index1 spans lm5 -> lm6.
const FINGERS = Object.freeze({
  Thumb: [1, 2, 3, 4],
  Index: [5, 6, 7, 8],
  Middle: [9, 10, 11, 12],
  Ring: [13, 14, 15, 16],
  Pinky: [17, 18, 19, 20],
});

const SIDES = ['Left', 'Right'];

const tempDirection = new THREE.Vector3();
const tempTargetDirection = new THREE.Vector3();
const tempBasisX = new THREE.Vector3();
const tempBasisY = new THREE.Vector3();
const tempBasisZ = new THREE.Vector3();
const tempSide = new THREE.Vector3();
const tempForward = new THREE.Vector3();
const tempNormal = new THREE.Vector3();
const tempWorldQuaternion = new THREE.Quaternion();
const tempParentQuaternion = new THREE.Quaternion();
const tempDeltaQuaternion = new THREE.Quaternion();
const tempDesiredQuaternion = new THREE.Quaternion();
const tempLocalQuaternion = new THREE.Quaternion();
const tempMatrix = new THREE.Matrix4();
const IDENTITY_QUATERNION = new THREE.Quaternion();

const ikTargetWrist = new THREE.Vector3();
const ikSolvedElbow = new THREE.Vector3();
const ikSolvedWrist = new THREE.Vector3();
const ikToTarget = new THREE.Vector3();
const ikDirToTarget = new THREE.Vector3();
const ikXAxis = new THREE.Vector3();
const ikYAxis = new THREE.Vector3();
const ikElbowOffset = new THREE.Vector3();

const scratchQuatA = new THREE.Quaternion();
const scratchQuatB = new THREE.Quaternion();
const scratchVecA = new THREE.Vector3();

function assertMotion(data) {
  if (data?.format !== 'signbridge-motion-v1') {
    throw new Error('Unsupported SignBridge motion format.');
  }
  if (!Number.isFinite(data.fps) || data.fps <= 0 || !data.frames?.length) {
    throw new Error('The SignBridge motion file contains no playable frames.');
  }
}

// Verified against this clip's own data - see the header block.
const AXIS_SIGN = Object.freeze({ x: 1, y: -1, z: -1 });

function mapPoint(raw, target) {
  target.set(AXIS_SIGN.x * raw[0], AXIS_SIGN.y * raw[1], AXIS_SIGN.z * raw[2]);
  return target;
}

function normalizedOrNull(vector) {
  const lengthSquared = vector.lengthSq();
  if (!Number.isFinite(lengthSquared) || lengthSquared < 1e-10) return null;
  return vector.multiplyScalar(1 / Math.sqrt(lengthSquared));
}

function orthonormalPalmBasis(forward, side, normal) {
  if (!normalizedOrNull(forward)) return false;
  side.addScaledVector(forward, -side.dot(forward));
  if (!normalizedOrNull(side)) return false;
  normal.crossVectors(side, forward);
  return Boolean(normalizedOrNull(normal));
}

/* ---------------------------------------------------------------------
 * Quaternion helpers used by the finger gain and the twist distribution.
 * ------------------------------------------------------------------- */

// Split q into swing * twist about `axis` (axis normalized, same space).
function decomposeTwist(q, axis, outTwist) {
  const projection = axis.x * q.x + axis.y * q.y + axis.z * q.z;
  outTwist.set(axis.x * projection, axis.y * projection, axis.z * projection, q.w);
  if (outTwist.lengthSq() < 1e-9) {
    outTwist.set(0, 0, 0, 1);
  } else {
    outTwist.normalize();
  }
  return outTwist;
}

// Signed rotation angle of q about its own axis, wrapped to [-PI, PI].
function signedAngle(q) {
  const w = THREE.MathUtils.clamp(q.w, -1, 1);
  let angle = 2 * Math.acos(w);
  if (angle > Math.PI) angle -= 2 * Math.PI;
  return angle;
}

// Scale a rotation's angle while keeping its axis, then clamp.
function scaleAndClampRotation(q, gain, minAngle, maxAngle, out) {
  const w = THREE.MathUtils.clamp(q.w, -1, 1);
  let angle = 2 * Math.acos(w);
  const sinHalf = Math.sqrt(Math.max(0, 1 - w * w));
  if (sinHalf < 1e-6 || !Number.isFinite(angle)) {
    return out.copy(IDENTITY_QUATERNION);
  }
  scratchVecA.set(q.x / sinHalf, q.y / sinHalf, q.z / sinHalf);
  if (angle > Math.PI) {
    angle -= 2 * Math.PI;
    scratchVecA.negate();
    angle = -angle;
    scratchVecA.negate();
  }
  const scaled = THREE.MathUtils.clamp(angle * gain, minAngle, maxAngle);
  return out.setFromAxisAngle(scratchVecA, scaled);
}

/* ---------------------------------------------------------------------
 * Closed-form two-bone IK.  Unchanged - it was correct.
 * ------------------------------------------------------------------- */
function solveTwoBoneIK(shoulderPos, targetPos, poleDir, upperLen, lowerLen, outElbow, outWrist) {
  ikToTarget.copy(targetPos).sub(shoulderPos);
  let dist = ikToTarget.length();
  const maxReach = (upperLen + lowerLen) * 0.97;
  const minReach = Math.abs(upperLen - lowerLen) + 1e-4;
  dist = THREE.MathUtils.clamp(dist, minReach, maxReach);

  if (ikToTarget.lengthSq() < 1e-10) ikToTarget.set(0, -1, 0);
  ikDirToTarget.copy(ikToTarget).normalize();

  const cosShoulder = THREE.MathUtils.clamp(
    (upperLen * upperLen + dist * dist - lowerLen * lowerLen) / (2 * upperLen * dist),
    -1, 1,
  );
  const shoulderAngle = Math.acos(cosShoulder);

  ikXAxis.copy(ikDirToTarget);
  ikYAxis.copy(poleDir).addScaledVector(ikXAxis, -poleDir.dot(ikXAxis));
  if (ikYAxis.lengthSq() < 1e-8) ikYAxis.set(0, 1, 0.001);
  ikYAxis.normalize();

  ikElbowOffset.copy(ikXAxis).multiplyScalar(Math.cos(shoulderAngle) * upperLen)
    .addScaledVector(ikYAxis, Math.sin(shoulderAngle) * upperLen);

  outElbow.copy(shoulderPos).add(ikElbowOffset);
  outWrist.copy(shoulderPos).addScaledVector(ikDirToTarget, dist);
}

/* ---------------------------------------------------------------------
 * Temporal filtering.  Unchanged: confidence-aware Hermite gap fill ->
 * two-pass One-Euro -> kinematic clamp.
 * ------------------------------------------------------------------- */

class OneEuroFilterScalar {
  constructor(minCutoff = 1.0, beta = 0.0, dCutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.xPrev = null;
    this.dxPrev = 0;
  }

  static alpha(cutoff, dt) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  filter(x, dt) {
    if (this.xPrev === null) {
      this.xPrev = x;
      this.dxPrev = 0;
      return x;
    }
    const dx = (x - this.xPrev) / dt;
    const aD = OneEuroFilterScalar.alpha(this.dCutoff, dt);
    const dxHat = aD * dx + (1 - aD) * this.dxPrev;
    const cutoff = this.minCutoff + this.beta * Math.abs(dxHat);
    const a = OneEuroFilterScalar.alpha(cutoff, dt);
    const xHat = a * x + (1 - a) * this.xPrev;
    this.xPrev = xHat;
    this.dxPrev = dxHat;
    return xHat;
  }
}

function oneEuroFilterVectorTrack(values, dt, minCutoff, beta) {
  const runPass = (arr) => {
    const fx = new OneEuroFilterScalar(minCutoff, beta);
    const fy = new OneEuroFilterScalar(minCutoff, beta);
    const fz = new OneEuroFilterScalar(minCutoff, beta);
    return arr.map((v) => new THREE.Vector3(
      fx.filter(v.x, dt), fy.filter(v.y, dt), fz.filter(v.z, dt),
    ));
  };
  const forward = runPass(values);
  const backward = runPass([...forward].reverse()).reverse();
  return values.map((_, i) => forward[i].clone().add(backward[i]).multiplyScalar(0.5));
}

function oneEuroFilterScalarTrack(values, dt, minCutoff, beta) {
  const runPass = (arr) => {
    const f = new OneEuroFilterScalar(minCutoff, beta);
    return arr.map((v) => f.filter(v, dt));
  };
  const forward = runPass(values);
  const backward = runPass([...forward].reverse()).reverse();
  return values.map((_, i) => (forward[i] + backward[i]) * 0.5);
}

function hermite(p0, m0, p1, m1, t, span) {
  const t2 = t * t;
  const t3 = t2 * t;
  const h00 = 2 * t3 - 3 * t2 + 1;
  const h10 = t3 - 2 * t2 + t;
  const h01 = -2 * t3 + 3 * t2;
  const h11 = t3 - t2;
  return p0.clone().multiplyScalar(h00)
    .addScaledVector(m0, h10 * span)
    .addScaledVector(p1, h01)
    .addScaledVector(m1, h11 * span);
}

function estimateTangent(values, confident, i, dt) {
  const hasPrev = i > 0 && confident[i - 1];
  const hasNext = i < values.length - 1 && confident[i + 1];
  if (hasPrev && hasNext) {
    return values[i + 1].clone().sub(values[i - 1]).multiplyScalar(1 / (2 * dt));
  }
  if (hasPrev) return values[i].clone().sub(values[i - 1]).multiplyScalar(1 / dt);
  if (hasNext) return values[i + 1].clone().sub(values[i]).multiplyScalar(1 / dt);
  return new THREE.Vector3(0, 0, 0);
}

function fillGapsConfidenceAware(rawValues, confident, dt) {
  const n = rawValues.length;
  const out = rawValues.map((v) => v.clone());
  let i = 0;
  while (i < n) {
    if (confident[i]) { i += 1; continue; }
    let j = i;
    while (j < n && !confident[j]) j += 1;
    const hasLeft = i > 0;
    const hasRight = j < n;
    if (!hasLeft && !hasRight) {
      // no confident frame anywhere - keep the raw values
    } else if (!hasLeft) {
      for (let k = i; k < j; k += 1) out[k].copy(rawValues[j]);
    } else if (!hasRight) {
      for (let k = i; k < j; k += 1) out[k].copy(rawValues[i - 1]);
    } else {
      const p0 = rawValues[i - 1];
      const p1 = rawValues[j];
      const m0 = estimateTangent(rawValues, confident, i - 1, dt);
      const m1 = estimateTangent(rawValues, confident, j, dt);
      const span = (j - (i - 1)) * dt;
      for (let k = i; k < j; k += 1) {
        const t = ((k - (i - 1)) * dt) / span;
        out[k].copy(hermite(p0, m0, p1, m1, t, span));
      }
    }
    i = j;
  }
  return out;
}

function clampKinematics(values, dt, maxSpeed, maxAccel) {
  const out = values.map((v) => v.clone());
  let prevVel = new THREE.Vector3();
  const vel = new THREE.Vector3();
  const accel = new THREE.Vector3();
  for (let i = 1; i < out.length; i += 1) {
    vel.copy(out[i]).sub(out[i - 1]).divideScalar(dt);
    const speed = vel.length();
    if (speed > maxSpeed) {
      vel.multiplyScalar(maxSpeed / speed);
      out[i].copy(out[i - 1]).addScaledVector(vel, dt);
    }
    accel.copy(vel).sub(prevVel).divideScalar(dt);
    const accelMag = accel.length();
    if (accelMag > maxAccel) {
      accel.multiplyScalar(maxAccel / accelMag);
      vel.copy(prevVel).addScaledVector(accel, dt);
      out[i].copy(out[i - 1]).addScaledVector(vel, dt);
    }
    prevVel = vel.clone();
  }
  return out;
}

function smoothStepLocal(x) {
  const c = THREE.MathUtils.clamp(x, 0, 1);
  return c * c * (3 - 2 * c);
}

function easeEnvelope(t, duration, easeIn, easeOut) {
  if (t < easeIn) return smoothStepLocal(t / easeIn);
  if (t > duration - easeOut) return smoothStepLocal((duration - t) / easeOut);
  return 1;
}

/* ---------------------------------------------------------------------
 * Continuous-time sampling (Catmull-Rom).  Unchanged.
 * ------------------------------------------------------------------- */
function catmullRomComponent(p0, p1, p2, p3, t, t2, t3) {
  return 0.5 * (
    (2 * p1)
    + (-p0 + p2) * t
    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
    + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
  );
}

function catmullRomVector(p0, p1, p2, p3, t, out) {
  const t2 = t * t;
  const t3 = t2 * t;
  out.set(
    catmullRomComponent(p0.x, p1.x, p2.x, p3.x, t, t2, t3),
    catmullRomComponent(p0.y, p1.y, p2.y, p3.y, t, t2, t3),
    catmullRomComponent(p0.z, p1.z, p2.z, p3.z, t, t2, t3),
  );
  return out;
}

const catmullRomOut = new THREE.Vector3();
function sampleTrackSmooth(track, framePosition, out = catmullRomOut) {
  const n = track.length;
  const i1 = THREE.MathUtils.clamp(Math.floor(framePosition), 0, n - 1);
  const i0 = Math.max(i1 - 1, 0);
  const i2 = Math.min(i1 + 1, n - 1);
  const i3 = Math.min(i1 + 2, n - 1);
  const t = THREE.MathUtils.clamp(framePosition - i1, 0, 1);
  return catmullRomVector(track[i0], track[i1], track[i2], track[i3], t, out);
}

function sampleScalarTrack(track, framePosition) {
  const n = track.length;
  const i1 = THREE.MathUtils.clamp(Math.floor(framePosition), 0, n - 1);
  const i2 = Math.min(i1 + 1, n - 1);
  const t = THREE.MathUtils.clamp(framePosition - i1, 0, 1);
  return THREE.MathUtils.lerp(track[i1], track[i2], t);
}

/* ---------------------------------------------------------------------
 * Faithful-mode direction sampling.
 * Spherical interpolation between the two neighbouring source directions
 * cannot overshoot and passes through every integer source frame exactly.
 * ------------------------------------------------------------------- */
function sampleDirectionSlerp(track, framePosition, out) {
  const n = track.length;
  const i0 = THREE.MathUtils.clamp(Math.floor(framePosition), 0, n - 1);
  const i1 = Math.min(i0 + 1, n - 1);
  const t = THREE.MathUtils.clamp(framePosition - i0, 0, 1);

  if (t <= 1e-6 || i0 === i1) return out.copy(track[i0]);

  const a = track[i0];
  const b = track[i1];
  const dot = THREE.MathUtils.clamp(a.dot(b), -1, 1);

  if (dot > 0.99995) {
    return out.copy(a).lerp(b, t).normalize();
  }

  const theta = Math.acos(dot);
  const sinTheta = Math.sin(theta);

  return out
    .copy(a)
    .multiplyScalar(Math.sin((1 - t) * theta) / sinTheta)
    .addScaledVector(b, Math.sin(t * theta) / sinTheta)
    .normalize();
}

/* ---------------------------------------------------------------------
 * Palm quaternion helpers.
 * ------------------------------------------------------------------- */

function alignQuaternionHemisphere(track) {
  for (let i = 1; i < track.length; i += 1) {
    if (track[i - 1].dot(track[i]) < 0) {
      track[i].set(-track[i].x, -track[i].y, -track[i].z, -track[i].w);
    }
  }
  return track;
}

function clampQuaternionStep(track, maxStepRad) {
  for (let i = 1; i < track.length; i += 1) {
    const angle = track[i - 1].angleTo(track[i]);
    if (angle > maxStepRad) {
      track[i].copy(track[i - 1]).slerp(track[i], maxStepRad / angle).normalize();
    }
  }
  return track;
}

function smoothPalmQuaternionTrack(track, passes) {
  const mid = new THREE.Quaternion();

  for (let p = 0; p < passes; p += 1) {
    const source = track.map((q) => q.clone());

    for (let i = 1; i < track.length - 1; i += 1) {
      mid.copy(source[i - 1]).slerp(source[i + 1], 0.5);
      track[i].copy(source[i]).slerp(mid, 0.3).normalize();
    }

    alignQuaternionHemisphere(track);
  }

  return track;
}

function sampleQuaternionTrack(track, framePosition, out) {
  const n = track.length;
  const i0 = THREE.MathUtils.clamp(Math.floor(framePosition), 0, n - 1);
  const i1 = Math.min(i0 + 1, n - 1);
  const t = THREE.MathUtils.clamp(framePosition - i0, 0, 1);
  return out.copy(track[i0]).slerp(track[i1], t).normalize();
}

const tempPalmQuaternion = new THREE.Quaternion();
const tempRestLocalQuaternion = new THREE.Quaternion();

/* ===================================================================== */

export class SignMotionClip {
  constructor(data) {
    assertMotion(data);
    this.name = data.name || 'Sign';
    this.fps = Number(data.fps);
    this.frames = data.frames;
    this.frameCount = data.frames.length;
    this.duration = (this.frameCount - 1) / this.fps;
    this.detection = data.detection || {};
    this.faceBlendshapeNames = data.faceBlendshapeNames || [];
    // NOTE: fingerMode is deliberately IGNORED now.  The old code treated
    // fingerMode === 'open' as "freeze the fingers", which is what made
    // the handshape constant for the whole clip.  Fingers are always
    // driven; confidence weighting handles the frames without real data.
    this.fingerMode = data.fingerMode || 'tracked';
    this.sourceWidth = Number(data.source?.width) || 0;
    this.sourceHeight = Number(data.source?.height) || 0;
    this.aspectYOverX = (this.sourceWidth > 0 && this.sourceHeight > 0)
      ? this.sourceHeight / this.sourceWidth
      : 1;
    console.info(
      `[Clip] ${this.name}: ${this.frameCount} frames @ ${this.fps}fps, `
      + `source ${this.sourceWidth}x${this.sourceHeight}, `
      + `aspectYOverX ${this.aspectYOverX.toFixed(4)}`
      + (this.aspectYOverX > 1 ? ' (portrait)' : ' (landscape)'),
    );
  }
}

export class AvatarMotionRetargeter {
  constructor(avatar) {
    this.avatar = avatar;
    this.bones = new Map();
    this.restQuaternions = new Map();      // true GLB bind pose, never mutated
    this.neutralFingerQuaternions = new Map(); // relaxed finger pose (display only)
    this.axes = new Map();
    this.palmBases = new Map();
    this.palmNormalsLocal = new Map();
    this.morphBindings = new Map();
    this.requiredBones = [];
    this.twistBones = new Map();           // side -> [bone, bone]
    this.forearmAxis = new Map();          // side -> local long axis of the forearm

    this.armLengths = new Map();
    this.naturalRestQuaternions = new Map();
    this.scale = 1;
    this.filtered = null;

    this.debugEnabled = false;
    this.debugMarkers = null;
    this.debugScene = null;

    this.#registerArm('Left');
    this.#registerArm('Right');
    this.#registerTwistBones('Left');
    this.#registerTwistBones('Right');
    this.#registerMorphTargets();

    const missing = this.requiredBones.filter((name) => !this.bones.has(name));
    if (missing.length) {
      throw new Error(`Missing avatar motion bones: ${missing.join(', ')}`);
    }

    this.avatar.updateMatrixWorld(true);
    this.#capturePalmBasis('Left');
    this.#capturePalmBasis('Right');
    this.#captureNeutralFingerPose('Left');
    this.#captureNeutralFingerPose('Right');
    this.#captureArmLengths('Left');
    this.#captureArmLengths('Right');
    this.#captureNaturalRestPose('Left');
    this.#captureNaturalRestPose('Right');
    this.reset();
  }

  /* ---------------- registration ---------------- */

  #registerBone(name, childName = null, required = true) {
    if (required) this.requiredBones.push(name);
    const bone = this.avatar.getObjectByName(name);
    if (!bone) return null;

    this.bones.set(name, bone);
    this.restQuaternions.set(name, bone.quaternion.clone());

    if (childName) {
      const child = this.avatar.getObjectByName(childName);
      if (child) this.axes.set(name, child.position.clone().normalize());
    }
    return bone;
  }

  #registerArm(side) {
    this.#registerBone(`${side}Arm`, `${side}ForeArm`);
    this.#registerBone(`${side}ForeArm`, `${side}Hand`);
    this.#registerBone(`${side}Hand`);

    for (const finger of Object.keys(FINGERS)) {
      for (let segment = 1; segment <= 3; segment += 1) {
        this.#registerBone(
          `${side}Hand${finger}${segment}`,
          `${side}Hand${finger}${segment + 1}`,
        );
      }
    }
  }

  // Optional twist helpers.  This avatar has LeftForeArm1 / LeftForeArm2
  // (and the Right pair) as siblings of the hand.  Absent on other rigs,
  // so registration is non-fatal.
  #registerTwistBones(side) {
    const found = [];
    for (const suffix of ['1', '2']) {
      const bone = this.#registerBone(`${side}ForeArm${suffix}`, null, false);
      if (bone) found.push(bone);
    }
    this.twistBones.set(side, found);

    const hand = this.avatar.getObjectByName(`${side}Hand`);
    const axis = hand ? hand.position.clone().normalize() : new THREE.Vector3(0, 1, 0);
    this.forearmAxis.set(side, axis);
  }

  #registerMorphTargets() {
    this.avatar.traverse((object) => {
      if (!object.morphTargetDictionary || !object.morphTargetInfluences) return;
      for (const [name, index] of Object.entries(object.morphTargetDictionary)) {
        if (!this.morphBindings.has(name)) this.morphBindings.set(name, []);
        this.morphBindings.get(name).push({ object, index });
      }
    });
  }

  #capturePalmBasis(side) {
    const hand = this.bones.get(`${side}Hand`);
    const index = this.bones.get(`${side}HandIndex1`);
    const middle = this.bones.get(`${side}HandMiddle1`);
    const pinky = this.bones.get(`${side}HandPinky1`);

    tempForward.copy(middle.position);
    tempSide.copy(index.position).sub(pinky.position);
    if (!orthonormalPalmBasis(tempForward, tempSide, tempNormal)) {
      throw new Error(`Could not calculate ${side} hand basis.`);
    }

    tempMatrix.makeBasis(tempSide, tempForward, tempNormal);
    this.palmBases.set(side, new THREE.Quaternion().setFromRotationMatrix(tempMatrix));
    this.palmNormalsLocal.set(side, tempNormal.clone());
    hand.updateMatrixWorld(true);
  }

  // The GLB's authored hand is noticeably splayed - measured rest angles
  // on the proximal bones: Thumb1 69.2deg, Index1 17.0, Middle1 27.0,
  // Ring1 32.1, Pinky1 41.2.  That descending splay is exactly why the
  // fingers used to look "differently wrong" from each other while frozen.
  // This stores a flatter neutral WITHOUT touching restQuaternions, so the
  // true bind pose stays available for the retarget maths.
  #captureNeutralFingerPose(side) {
    for (const finger of Object.keys(FINGERS)) {
      for (let segment = 1; segment <= 3; segment += 1) {
        const name = `${side}Hand${finger}${segment}`;
        const rest = this.restQuaternions.get(name);
        if (!rest) continue;
        this.neutralFingerQuaternions.set(
          name,
          rest.clone().slerp(IDENTITY_QUATERNION, THREE.MathUtils.clamp(TUNING.fingerRelax, 0, 1)),
        );
      }
    }
  }

  #captureArmLengths(side) {
    const shoulder = this.bones.get(`${side}Arm`);
    const elbow = this.bones.get(`${side}ForeArm`);
    const wrist = this.bones.get(`${side}Hand`);
    const shoulderPos = shoulder.getWorldPosition(new THREE.Vector3());
    const elbowPos = elbow.getWorldPosition(new THREE.Vector3());
    const wristPos = wrist.getWorldPosition(new THREE.Vector3());
    const upper = shoulderPos.distanceTo(elbowPos);
    const lower = elbowPos.distanceTo(wristPos);
    this.armLengths.set(side, {
      upper, lower, total: upper + lower, shoulderRest: shoulderPos.clone(),
    });
  }

  #captureNaturalRestPose(side) {
    const sign = side === 'Left' ? 1 : -1;
    const armBone = this.bones.get(`${side}Arm`);
    const foreArmBone = this.bones.get(`${side}ForeArm`);

    this.#orientBone(`${side}Arm`, new THREE.Vector3(sign * 0.15, -1, 0.05));
    this.avatar.updateMatrixWorld(true);
    this.#orientBone(`${side}ForeArm`, new THREE.Vector3(sign * 0.05, -1, 0.15));
    this.avatar.updateMatrixWorld(true);

    this.naturalRestQuaternions.set(`${side}Arm`, armBone.quaternion.clone());
    this.naturalRestQuaternions.set(`${side}ForeArm`, foreArmBone.quaternion.clone());

    armBone.quaternion.copy(this.restQuaternions.get(`${side}Arm`));
    foreArmBone.quaternion.copy(this.restQuaternions.get(`${side}ForeArm`));
    this.avatar.updateMatrixWorld(true);
  }

  /* ---------------- calibration ---------------- */

  // One shared scale for both sides.  On the current stack.motion.json the
  // measured real arm totals are 0.4164 m (left) and 0.4328 m (right) -
  // only 3.8% apart, so the shared factor costs almost nothing and still
  // guards against one noisy side skewing the other.
  calibrateScale(frames) {
    const totals = { Left: [], Right: [] };
    for (const frame of frames) {
      if (!frame.poseDetected || !frame.poseWorld) continue;
      for (const side of SIDES) {
        const lower = side.toLowerCase();
        const shoulder = frame.poseWorld[POSE[`${lower}Shoulder`]];
        const elbow = frame.poseWorld[POSE[`${lower}Elbow`]];
        const wrist = frame.poseWorld[POSE[`${lower}Wrist`]];
        const upper = Math.hypot(
          elbow[0] - shoulder[0], elbow[1] - shoulder[1], elbow[2] - shoulder[2],
        );
        const fore = Math.hypot(
          wrist[0] - elbow[0], wrist[1] - elbow[1], wrist[2] - elbow[2],
        );
        totals[side].push(upper + fore);
      }
    }

    const median = (values) => {
      const sorted = [...values].sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    };

    const realLeft = totals.Left.length ? median(totals.Left) : null;
    const realRight = totals.Right.length ? median(totals.Right) : null;
    const valid = [realLeft, realRight].filter((v) => v && v > 1e-4);
    if (!valid.length) {
      console.warn('[Retargeter] Could not calibrate scale; using 1.');
      return;
    }
    const realArmLen = valid.reduce((a, b) => a + b, 0) / valid.length;
    const avatarArmLen = (this.armLengths.get('Left').total + this.armLengths.get('Right').total) / 2;
    this.scale = avatarArmLen / realArmLen;
    console.info(
      `[Retargeter] scale=${this.scale.toFixed(4)} `
      + `(avatar ${avatarArmLen.toFixed(4)} / real ${realArmLen.toFixed(4)}; `
      + `L=${realLeft?.toFixed(4)} R=${realRight?.toFixed(4)})`,
    );
  }

  /* ---------------- track preparation ---------------- */

  prepareFilteredTracks(frames, fps) {
    const dt = 1 / fps;
    this.filtered = {};

    for (const side of SIDES) {
      const lower = side.toLowerCase();
      const handKey = `${lower}HandWorld`;
      const sIdx = POSE[`${lower}Shoulder`];
      const eIdx = POSE[`${lower}Elbow`];
      const wIdx = POSE[`${lower}Wrist`];

      const poseConfident = frames.map((f) => Boolean(f.poseDetected && f.poseWorld));
      const handConfident = frames.map(
        (f) => Boolean(f[`${lower}HandDetected`] && f[handKey]),
      );

      const detectedCount = handConfident.filter(Boolean).length;
      const detectionRate = detectedCount / Math.max(frames.length, 1);

      const diff = (a, b) => new THREE.Vector3(
        AXIS_SIGN.x * (a[0] - b[0]),
        AXIS_SIGN.y * (a[1] - b[1]),
        AXIS_SIGN.z * (a[2] - b[2]),
      );

      const rawWristOffset = frames.map(
        (f) => diff(f.poseWorld[wIdx], f.poseWorld[sIdx]),
      );
      const rawPole = frames.map(
        (f) => diff(f.poseWorld[eIdx], f.poseWorld[sIdx]),
      );

      const rawPalmForward = frames.map((f) => {
        const h = f[handKey];
        return h ? diff(h[9], h[0]) : new THREE.Vector3(0, 1, 0.001);
      });

      const rawPalmSide = frames.map((f) => {
        const h = f[handKey];
        return h ? diff(h[5], h[17]) : new THREE.Vector3(1, 0, 0);
      });

      // Finger direction tracks: 15 per hand.
      const fingerTracks = new Map();

      for (const [finger, landmarks] of Object.entries(FINGERS)) {
        for (let segment = 0; segment < 3; segment += 1) {
          const boneName = `${side}Hand${finger}${segment + 1}`;

          const raw = frames.map((f) => {
            const h = f[handKey];
            if (!h) return new THREE.Vector3(0, 1, 0.001);

            return diff(
              h[landmarks[segment + 1]],
              h[landmarks[segment]],
            );
          });

          const filled = fillGapsConfidenceAware(raw, handConfident, dt);
          const smoothed = oneEuroFilterVectorTrack(filled, dt, 1.4, 0.25)
            .map(
              (v) => normalizedOrNull(v) || new THREE.Vector3(0, 1, 0.001),
            );

          fingerTracks.set(boneName, smoothed);
        }
      }

      // Smoothed hand-detection confidence.
      const rawConfidence = handConfident.map((c) => (c ? 1 : 0));
      const confidenceTrack = oneEuroFilterScalarTrack(
        rawConfidence,
        dt,
        1.2,
        0.0,
      ).map((v) => THREE.MathUtils.clamp(v, 0, 1));

      const filledWristOffset = fillGapsConfidenceAware(
        rawWristOffset,
        poseConfident,
        dt,
      );
      const filledPole = fillGapsConfidenceAware(
        rawPole,
        poseConfident,
        dt,
      );
      const filledPalmForward = fillGapsConfidenceAware(
        rawPalmForward,
        handConfident,
        dt,
      );
      const filledPalmSide = fillGapsConfidenceAware(
        rawPalmSide,
        handConfident,
        dt,
      );

      let wristOffsetTrack = oneEuroFilterVectorTrack(
        filledWristOffset,
        dt,
        1.0,
        0.3,
      ).map((v) => v.multiplyScalar(this.scale));

      wristOffsetTrack = clampKinematics(
        wristOffsetTrack,
        dt,
        2.5,
        25,
      );

      const poleTrack = oneEuroFilterVectorTrack(
        filledPole,
        dt,
        1.0,
        0.2,
      ).map(
        (v) => normalizedOrNull(v) || new THREE.Vector3(0, -1, 0.001),
      );

      const palmForwardTrack = oneEuroFilterVectorTrack(
        filledPalmForward,
        dt,
        1.0,
        0.2,
      ).map(
        (v) => normalizedOrNull(v) || new THREE.Vector3(0, 1, 0.001),
      );

      const palmSideTrack = oneEuroFilterVectorTrack(
        filledPalmSide,
        dt,
        1.0,
        0.2,
      ).map(
        (v) => normalizedOrNull(v) || new THREE.Vector3(1, 0, 0),
      );

      // Build palm orientation once per source frame as a quaternion track.
      // This lets us hemisphere-align, angular-speed clamp, and smooth
      // orientation directly in rotation space.
      const palmQuatTrack = [];
      const basisForward = new THREE.Vector3();
      const basisSide = new THREE.Vector3();
      const basisNormal = new THREE.Vector3();
      const basisMatrix = new THREE.Matrix4();

      for (let i = 0; i < frames.length; i += 1) {
        basisForward.copy(palmForwardTrack[i]);
        basisSide.copy(palmSideTrack[i]);

        const q = new THREE.Quaternion();

        if (orthonormalPalmBasis(
          basisForward,
          basisSide,
          basisNormal,
        )) {
          basisMatrix.makeBasis(
            basisSide,
            basisForward,
            basisNormal,
          );
          q.setFromRotationMatrix(basisMatrix);
        } else if (palmQuatTrack.length) {
          q.copy(palmQuatTrack[palmQuatTrack.length - 1]);
        }

        palmQuatTrack.push(q);
      }

      alignQuaternionHemisphere(palmQuatTrack);
      clampQuaternionStep(
        palmQuatTrack,
        THREE.MathUtils.degToRad(TUNING.palmMaxStepDeg),
      );
      smoothPalmQuaternionTrack(
        palmQuatTrack,
        Math.max(0, TUNING.palmSmoothPasses | 0),
      );

      this.filtered[side] = {
        wristOffset: wristOffsetTrack,
        pole: poleTrack,
        palmForward: palmForwardTrack,
        palmSide: palmSideTrack,
        palmQuat: palmQuatTrack,
        fingers: fingerTracks,
        confidence: confidenceTrack,
        detectionRate,
      };

      let maxStep = 0;
      let totalStep = 0;

      for (let i = 1; i < palmQuatTrack.length; i += 1) {
        const angle = palmQuatTrack[i - 1].angleTo(palmQuatTrack[i]);
        maxStep = Math.max(maxStep, angle);
        totalStep += angle;
      }

      const firstReal = handConfident.indexOf(true);
      const lastReal = handConfident.lastIndexOf(true);

      console.info(
        `[Retargeter] ${side}: hand detected ${(detectionRate * 100).toFixed(0)}% `
        + `(frames ${firstReal}..${lastReal}; `
        + `${Math.max(firstReal, 0)} frozen at head, `
        + `${frames.length - 1 - lastReal} at tail) | palm rotation: `
        + `mean ${THREE.MathUtils.radToDeg(
          totalStep / Math.max(palmQuatTrack.length - 1, 1),
        ).toFixed(2)}°/frame, max `
        + `${THREE.MathUtils.radToDeg(maxStep).toFixed(1)}°/frame`,
      );

      if (detectionRate < 0.6) {
        console.warn(
          `[Retargeter] ${side}: detection too sparse for trustworthy palm orientation. `
          + 'Lower TUNING.palmAuthority or disable this hand.',
        );
      }
    }

    // Build raw/direct arm-direction tracks for SOURCE FAITHFUL MODE.
    this.prepareDirectionTracks(frames, fps);
  }

  prepareDirectionTracks(frames, fps) {
    const dt = 1 / fps;
    this.direct = {};

    for (const side of SIDES) {
      const lower = side.toLowerCase();
      const sIdx = POSE[`${lower}Shoulder`];
      const eIdx = POSE[`${lower}Elbow`];
      const wIdx = POSE[`${lower}Wrist`];

      const diff = (a, b) => new THREE.Vector3(
        AXIS_SIGN.x * (a[0] - b[0]),
        AXIS_SIGN.y * (a[1] - b[1]),
        AXIS_SIGN.z * (a[2] - b[2]),
      );

      let upper = frames.map((f) => diff(f.poseWorld[eIdx], f.poseWorld[sIdx]));
      let fore = frames.map((f) => diff(f.poseWorld[wIdx], f.poseWorld[eIdx]));

      const hz = Number(TUNING.faithfulSmoothingHz) || 0;
      if (hz > 0) {
        upper = oneEuroFilterVectorTrack(upper, dt, hz, 0.2);
        fore = oneEuroFilterVectorTrack(fore, dt, hz, 0.2);
      }

      upper = upper.map(
        (v) => normalizedOrNull(v) || new THREE.Vector3(0, -1, 0),
      );
      fore = fore.map(
        (v) => normalizedOrNull(v) || new THREE.Vector3(0, -1, 0),
      );

      // Screen-plane correction, forearm only.
      // Match the X/Y forearm direction to the directly observed 2D pose
      // landmarks while keeping the world-landmark Z/depth component.
      const blend = side === 'Left'
        ? (TUNING.forearmImagePlaneLeft ?? 0)
        : (TUNING.forearmImagePlaneRight ?? 0);

      if (blend > 0.001 && frames[0]?.pose) {
        const aspect = this.clipInfo?.aspectYOverX || 1;
        const corrected = new THREE.Vector3();
        let sum = 0;
        let worst = 0;

        fore = fore.map((worldDir, i) => {
          const p = frames[i].pose;
          if (!p) return worldDir;

          let ix = p[wIdx][0] - p[eIdx][0];
          let iy = -(p[wIdx][1] - p[eIdx][1]) * aspect;

          const ilen = Math.hypot(ix, iy);
          if (ilen < 1e-9) return worldDir;

          ix /= ilen;
          iy /= ilen;

          const xy = Math.hypot(worldDir.x, worldDir.y);
          corrected.set(ix * xy, iy * xy, worldDir.z);

          if (!normalizedOrNull(corrected)) return worldDir;

          const moved = THREE.MathUtils.radToDeg(
            worldDir.angleTo(corrected),
          );
          sum += moved;
          worst = Math.max(worst, moved);

          return blend >= 0.999
            ? corrected.clone()
            : worldDir.clone().lerp(corrected, blend).normalize();
        });

        console.info(
          `[Faithful] ${side} forearm screen-plane correction at ${blend.toFixed(2)}: `
          + `3D direction moved mean ${(sum / frames.length).toFixed(1)}°, max ${worst.toFixed(1)}°`,
        );
      }


      // Minimal depth-only torso clearance.
      //
      // We use only the source IMAGE to decide WHEN the hand is over the torso;
      // we never use image-space to decide depth magnitude.  When active, the
      // correction adds at most ~3 cm of forward displacement distributed over
      // the two arm segments, then renormalizes their directions.
      //
      // This preserves the measured X/Y sign placement far better than a
      // lateral collision push and is intentionally much smaller than the
      // failed full depth reconstruction.
      if (TUNING.depthOnlyClearance && frames[0]?.pose) {
        const armLengths = this.armLengths.get(side);
        const Lu = armLengths?.upper;
        const Lf = armLengths?.lower;

        if (Lu > 1e-6 && Lf > 1e-6) {
          const FWD = Number(TUNING.depthOnlyForwardSign) >= 0 ? 1 : -1;
          const maxBias = Math.max(0, Number(TUNING.depthOnlyBiasMeters) || 0);
          const pad = Math.max(0, Number(TUNING.depthOnlyTorsoPadding) || 0);
          const minWeight = THREE.MathUtils.clamp(
            Number(TUNING.depthOnlyMinWeight) || 0,
            0,
            1,
          );

          let corrected = 0;
          let maxAppliedCm = 0;

          for (let i = 0; i < frames.length; i += 1) {
            const p = frames[i].pose;
            if (!p) continue;

            const ls = p[POSE.leftShoulder];
            const rs = p[POSE.rightShoulder];
            const lh = p[23];
            const rh = p[24];
            const wrist2 = p[wIdx];

            if (!ls || !rs || !lh || !rh || !wrist2) continue;

            const minX = Math.min(ls[0], rs[0], lh[0], rh[0]) - pad;
            const maxX = Math.max(ls[0], rs[0], lh[0], rh[0]) + pad;
            const minY = Math.min(ls[1], rs[1], lh[1], rh[1]) - pad;
            const maxY = Math.max(ls[1], rs[1], lh[1], rh[1]) + pad;

            const x = wrist2[0];
            const y = wrist2[1];

            if (x < minX || x > maxX || y < minY || y > maxY) continue;

            // Weight is strongest toward the torso-box centre and fades near
            // its edges. This prevents a hard pop entering/leaving the region.
            const cx = 0.5 * (minX + maxX);
            const cy = 0.5 * (minY + maxY);
            const hx = Math.max(0.5 * (maxX - minX), 1e-6);
            const hy = Math.max(0.5 * (maxY - minY), 1e-6);

            const nx = Math.abs(x - cx) / hx;
            const ny = Math.abs(y - cy) / hy;
            const edge = Math.max(nx, ny);
            const rawWeight = THREE.MathUtils.clamp(1 - edge, 0, 1);
            const weight = rawWeight * rawWeight * (3 - 2 * rawWeight);

            if (weight < minWeight) continue;

            const bias = maxBias * weight;

            const correctedUpper = upper[i].clone();
            const correctedFore = fore[i].clone();

            correctedUpper.z += FWD * (bias * 0.42 / Lu);
            correctedFore.z += FWD * (bias * 0.58 / Lf);

            if (!normalizedOrNull(correctedUpper)
                || !normalizedOrNull(correctedFore)) {
              continue;
            }

            upper[i] = correctedUpper;
            fore[i] = correctedFore;

            corrected += 1;
            maxAppliedCm = Math.max(maxAppliedCm, bias * 100);
          }

          console.info(
            `[Depth Minimal] ${side}: corrected ${corrected}/${frames.length} frames; `
            + `max forward bias ${maxAppliedCm.toFixed(1)} cm`,
          );
        }
      }

      this.direct[side] = {
        upper,
        fore,
        handDetected: frames.map(
          (f) => Boolean(f[`${lower}HandDetected`] && f[`${lower}HandWorld`]),
        ),
      };
    }

    this.#carryEdgeHandPose(frames);

    console.info(
      `[Faithful] direction tracks built, smoothing ${TUNING.faithfulSmoothingHz || 0} Hz`,
    );
  }


  #findReliableAnchor(side, frames) {
    const lower = side.toLowerCase();
    const det = this.direct[side].handDetected;
    const firstDet = det.indexOf(true);
    if (firstDet < 0) return -1;

    const fps = this.clipInfo?.fps || 30;
    const K = Math.max(3, Math.round(0.15 * fps));
    const maxDelay = Math.round(0.25 * fps);
    const STEP_MAX_DEG = 12;
    const DISAGREE_FACTOR = 1.5;

    const width = this.clipInfo?.width || 0;
    const height = this.clipInfo?.height || 0;
    const key2D = `${lower}Hand`;
    const key3D = `${lower}HandWorld`;
    const wristIdx = POSE[`${lower}Wrist`];

    const has2D = Boolean(
      width
      && height
      && frames[firstDet]?.[key2D]?.length
      && frames[firstDet]?.pose?.length,
    );

    if (!has2D) {
      console.warn(
        `[Anchor] ${side}: no 2D hand landmarks or no clip dimensions - condition 3 `
        + 'disabled. Falling back to the stability test alone, which on its own does '
        + 'NOT reliably detect a convergence ramp.',
      );
    }

    const fwd = new THREE.Vector3();
    const sid = new THREE.Vector3();
    const nrm = new THREE.Vector3();
    const mat = new THREE.Matrix4();

    const mapped = (h, a, b, out) => out.set(
      AXIS_SIGN.x * (h[a][0] - h[b][0]),
      AXIS_SIGN.y * (h[a][1] - h[b][1]),
      AXIS_SIGN.z * (h[a][2] - h[b][2]),
    );

    const rawPalm = (i, out) => {
      const h = frames[i]?.[key3D];
      if (!h) return null;

      mapped(h, 9, 0, fwd);
      mapped(h, 5, 17, sid);

      if (!orthonormalPalmBasis(fwd, sid, nrm)) return null;

      mat.makeBasis(sid, fwd, nrm);
      return out.setFromRotationMatrix(mat);
    };

    const disagreement = (i) => {
      if (!has2D) return null;

      const h = frames[i]?.[key2D];
      const p = frames[i]?.pose;
      if (!h || !p || !p[wristIdx]) return null;

      let minX = Infinity;
      let maxX = -Infinity;
      let minY = Infinity;
      let maxY = -Infinity;

      for (const lm of h) {
        if (lm[0] < minX) minX = lm[0];
        if (lm[0] > maxX) maxX = lm[0];
        if (lm[1] < minY) minY = lm[1];
        if (lm[1] > maxY) maxY = lm[1];
      }

      const sizePx = Math.hypot(
        (maxX - minX) * width,
        (maxY - minY) * height,
      );
      if (!(sizePx > 1e-6)) return null;

      const dPx = Math.hypot(
        (h[0][0] - p[wristIdx][0]) * width,
        (h[0][1] - p[wristIdx][1]) * height,
      );

      return dPx / sizePx;
    };

    const median = (values) => {
      const s = [...values].sort((a, b) => a - b);
      const m = s.length >> 1;
      return s.length % 2
        ? s[m]
        : (s[m - 1] + s[m]) / 2;
    };

    const searchEnd = Math.min(
      firstDet + maxDelay,
      frames.length - 1,
    );

    const qa = new THREE.Quaternion();
    const qb = new THREE.Quaternion();
    const stepDeg = new Map();

    for (
      let i = firstDet + 1;
      i <= Math.min(searchEnd + K, frames.length - 1);
      i += 1
    ) {
      if (!det[i] || !det[i - 1]) continue;
      if (!rawPalm(i - 1, qa) || !rawPalm(i, qb)) continue;

      stepDeg.set(
        i,
        THREE.MathUtils.radToDeg(
          qa.angleTo(qb),
        ),
      );
    }

    const report = [];
    let chosen = -1;

    for (let a = firstDet; a <= searchEnd; a += 1) {
      const run = [];
      for (let k = 0; k < K; k += 1) {
        run.push(a + k);
      }

      const runDetected = run.every(
        (i) => i < frames.length && det[i],
      );

      const steps = run
        .slice(1)
        .map((i) => stepDeg.get(i))
        .filter((v) => v !== undefined);

      const worstStep = steps.length
        ? Math.max(...steps)
        : Infinity;

      const stable = (
        runDetected
        && steps.length === run.length - 1
        && worstStep <= STEP_MAX_DEG
      );

      const runDis = runDetected
        ? run
          .map(disagreement)
          .filter(
            (v) => v !== null && Number.isFinite(v),
          )
        : [];

      const dHere = disagreement(a);
      const dMedian = runDis.length
        ? median(runDis)
        : null;

      const ratio = (
        dHere !== null
        && dMedian
      )
        ? dHere / dMedian
        : null;

      const converged = (
        !has2D
        || ratio === null
        || ratio <= DISAGREE_FACTOR
      );

      const pass = (
        runDetected
        && stable
        && converged
      );

      report.push({
        frame: a,
        detected: runDetected,
        worstStepDeg: Number.isFinite(worstStep)
          ? +worstStep.toFixed(2)
          : null,
        stable,
        disagree: dHere === null
          ? null
          : +dHere.toFixed(3),
        runMedian: dMedian === null
          ? null
          : +dMedian.toFixed(3),
        ratio: ratio === null
          ? null
          : +ratio.toFixed(2),
        converged,
        pass,
      });

      if (pass && chosen < 0) {
        chosen = a;
      }
    }

    console.info(
      `[Anchor] ${side}: firstDetected=${firstDet}  K=${K}  `
      + `maxDelay=${maxDelay} frames  stepMax=${STEP_MAX_DEG}deg  `
      + `disagreeFactor=${DISAGREE_FACTOR}`
      + (
        has2D
          ? ''
          : '  (condition 3 DISABLED - no 2D landmarks)'
      ),
    );

    console.table(report);

    if (chosen < 0) {
      console.warn(
        `[Anchor] ${side}: no frame passed within ${maxDelay} frames - `
        + `falling back to firstDetected=${firstDet}`,
      );
      return firstDet;
    }

    console.info(
      `[Anchor] ${side}: reliable anchor = frame ${chosen} `
      + `(delay +${chosen - firstDet} frames from first detection)`,
    );

    return chosen;
  }

  #carryEdgeHandPose(frames) {
    if (!TUNING.carryEdgeHandPose) return;

    const rot = new THREE.Quaternion();
    for (const side of SIDES) {
      const det = this.direct[side].handDetected;
      const fore = this.direct[side].fore;
      const tracks = this.filtered[side];
      if (!tracks || !det.some(Boolean)) continue;

      const first = this.#findReliableAnchor(side, frames);
      const last = det.lastIndexOf(true);

      if (first < 0) continue;
      const vectorTracks = [tracks.palmForward, tracks.palmSide];
      if (tracks.fingers) {
        for (const track of tracks.fingers.values()) vectorTracks.push(track);
      }

      const carry = (anchor, from, to) => {
        for (let i = from; i <= to; i += 1) {
          if (i < 0 || i >= frames.length) continue;
          rot.setFromUnitVectors(fore[anchor], fore[i]);
          for (const track of vectorTracks) {
            if (!track?.[anchor] || !track?.[i]) continue;
            track[i].copy(track[anchor]).applyQuaternion(rot).normalize();
          }
        }
      };

      if (last >= 0 && last < frames.length - 1) carry(last, last + 1, frames.length - 1);
      if (first > 0) carry(first, 0, first - 1);

      for (let i = 0; i < frames.length; i += 1) {
        if (!det[i] && (i > last || i < first)) det[i] = true;
      }

      if (tracks.confidence) {
        for (let i = 0; i < frames.length; i += 1) {
          if (i > last || i < first) tracks.confidence[i] = 1;
        }
      }

      this._gapIndex = {};

      console.info(
        `[Faithful] ${side} edge hand pose carried: leading 0-${Math.max(first - 1, -1)}, `
        + `trailing ${last + 1}-${frames.length - 1} (anchors ${first} and ${last})`,
      );
    }
  }

  // Direct FK: no IK, no global scale, no pole reconstruction, no relaxed-arm blend.
  #orientArmDirectFK(side, framePosition) {
    const tracks = this.direct?.[side];
    if (!tracks) return;

    const armBone = this.bones.get(`${side}Arm`);
    const foreArmBone = this.bones.get(`${side}ForeArm`);
    if (!armBone || !foreArmBone) return;

    const armRest = this.restQuaternions.get(`${side}Arm`);
    const foreRest = this.restQuaternions.get(`${side}ForeArm`);
    if (!armRest || !foreRest) return;

    armBone.quaternion.copy(armRest);
    foreArmBone.quaternion.copy(foreRest);
    armBone.updateWorldMatrix(true, true);

    sampleDirectionSlerp(
      tracks.upper,
      framePosition,
      tempTargetDirection,
    );
    this.#orientBone(`${side}Arm`, tempTargetDirection);

    sampleDirectionSlerp(
      tracks.fore,
      framePosition,
      tempTargetDirection,
    );
    this.#orientBone(`${side}ForeArm`, tempTargetDirection);

    if (this.debugEnabled) {
      const elbow = foreArmBone.getWorldPosition(new THREE.Vector3());
      const hand = this.bones.get(`${side}Hand`);
      if (hand) {
        const wrist = hand.getWorldPosition(new THREE.Vector3());
        this.#updateDebugMarkers(side, elbow, wrist);
      }
    }
  }

  // Faithful palm policy:
  // use palm data only when this exact source frame has a genuine hand detection.
  // On an undetected frame, keep the bind-pose local hand rotation.
  #orientHandFaithful(side, framePosition) {
    const handBone = this.bones.get(`${side}Hand`);
    const localPalmBasis = this.palmBases.get(side);
    const handRest = this.restQuaternions.get(`${side}Hand`);
    const detected = this.direct?.[side]?.handDetected;
    const tracks = this.filtered?.[side];

    if (!handBone || !localPalmBasis || !handRest || !detected || !tracks) return;

    const n = detected.length;

    // Cache nearest detected frames for each side.
    if (!this._gapIndex) this._gapIndex = {};

    if (!this._gapIndex[side]) {
      const prevValid = new Int32Array(n);
      const nextValid = new Int32Array(n);

      let last = -1;
      for (let i = 0; i < n; i += 1) {
        if (detected[i]) last = i;
        prevValid[i] = last;
      }

      last = -1;
      for (let i = n - 1; i >= 0; i -= 1) {
        if (detected[i]) last = i;
        nextValid[i] = last;
      }

      this._gapIndex[side] = { prevValid, nextValid };
    }

    const { prevValid, nextValid } = this._gapIndex[side];

    const offsetDeg = side === 'Left'
      ? (TUNING.palmPitchOffsetLeftDeg || 0)
      : (TUNING.palmPitchOffsetRightDeg || 0);

    const palmWorldAt = (index, out) => {
      const i = THREE.MathUtils.clamp(index, 0, n - 1);

      tempForward.copy(tracks.palmForward[i]);
      tempSide.copy(tracks.palmSide[i]);

      if (!orthonormalPalmBasis(tempForward, tempSide, tempNormal)) return null;

      if (Math.abs(offsetDeg) > 1e-4) {
        scratchQuatA.setFromAxisAngle(
          tempSide,
          THREE.MathUtils.degToRad(offsetDeg),
        );

        tempForward.applyQuaternion(scratchQuatA).normalize();
        tempNormal.crossVectors(tempSide, tempForward).normalize();
      }

      tempMatrix.makeBasis(tempSide, tempForward, tempNormal);
      return out.setFromRotationMatrix(tempMatrix);
    };

    const i0 = THREE.MathUtils.clamp(
      Math.floor(framePosition),
      0,
      n - 1,
    );

    const frac = THREE.MathUtils.clamp(framePosition - i0, 0, 1);
    const palmWorld = new THREE.Quaternion();
    let restBlend = 0;

    if (detected[i0]) {
      tempForward.copy(
        sampleTrackSmooth(tracks.palmForward, framePosition),
      );

      tempSide.copy(
        sampleTrackSmooth(tracks.palmSide, framePosition),
      );

      if (!orthonormalPalmBasis(tempForward, tempSide, tempNormal)) return;

      if (Math.abs(offsetDeg) > 1e-4) {
        scratchQuatA.setFromAxisAngle(
          tempSide,
          THREE.MathUtils.degToRad(offsetDeg),
        );

        tempForward.applyQuaternion(scratchQuatA).normalize();
        tempNormal.crossVectors(tempSide, tempForward).normalize();
      }

      tempMatrix.makeBasis(tempSide, tempForward, tempNormal);
      palmWorld.setFromRotationMatrix(tempMatrix);
    } else {
      const a = prevValid[i0];
      const b = nextValid[i0];
      const maxGap = TUNING.maxInterpolatedGapFrames ?? 10;
      const edge = Math.max(0, TUNING.edgeBlendFrames ?? 4);

      if (a >= 0 && b >= 0 && (b - a - 1) <= maxGap) {
        // Short internal gap: bridge measured endpoints with SLERP.
        const qa = palmWorldAt(a, new THREE.Quaternion());
        const qb = palmWorldAt(b, new THREE.Quaternion());

        if (!qa || !qb) return;

        if (qa.dot(qb) < 0) {
          qb.set(-qb.x, -qb.y, -qb.z, -qb.w);
        }

        const t = THREE.MathUtils.clamp(
          ((i0 + frac) - a) / (b - a),
          0,
          1,
        );

        palmWorld.copy(qa).slerp(qb, t).normalize();
      } else if (a >= 0 && b < 0) {
        // Trailing edge: ease from last measured orientation to rest.
        if (!palmWorldAt(a, palmWorld)) return;

        restBlend = edge > 0
          ? smoothStepLocal(((i0 + frac) - a) / edge)
          : 1;
      } else if (a < 0 && b >= 0) {
        // Leading edge: ease from rest into first measured orientation.
        if (!palmWorldAt(b, palmWorld)) return;

        restBlend = edge > 0
          ? smoothStepLocal((b - (i0 + frac)) / edge)
          : 1;
      } else {
        handBone.quaternion.copy(handRest);
        handBone.updateWorldMatrix(false, true);
        return;
      }

      restBlend = THREE.MathUtils.clamp(restBlend, 0, 1);
    }

    tempDesiredQuaternion
      .copy(palmWorld)
      .multiply(scratchQuatA.copy(localPalmBasis).invert());

    handBone.parent
      .getWorldQuaternion(tempParentQuaternion)
      .invert();

    tempLocalQuaternion
      .copy(tempParentQuaternion)
      .multiply(tempDesiredQuaternion)
      .normalize();

    if (restBlend > 0.001) {
      tempLocalQuaternion
        .slerp(scratchQuatB.copy(handRest), restBlend)
        .normalize();
    }

    handBone.quaternion.copy(tempLocalQuaternion);
    handBone.updateWorldMatrix(false, true);
  }

  /* ---------------- per-frame application ---------------- */

  applyContinuousFrame(
    framePosition,
    faceFirst,
    faceSecond,
    faceAlpha,
    easeWeight,
    faceBlendshapeNames = [],
  ) {
    if (TUNING.identityTest) {
      this.resetToBindPose();
      return;
    }

    if (TUNING.faithfulMode) {
      // Deliberately ignore easeWeight. Faithful mode preserves the
      // source timeline instead of blending the beginning/end toward
      // an invented relaxed pose.
      for (const side of SIDES) {
        this.#orientArmDirectFK(side, framePosition);
        this.#orientHandFaithful(side, framePosition);
        this.#orientFingersContinuous(side, framePosition, 1);
        this.#distributeForearmTwist(side, 1);
      }

      this.#applyFace(
        faceFirst,
        faceSecond,
        faceAlpha,
        faceBlendshapeNames,
      );
      this.avatar.updateMatrixWorld(true);
      return;
    }

    for (const side of SIDES) {
      this.#orientArmContinuous(side, framePosition, easeWeight);
      this.#orientHandContinuous(side, framePosition, easeWeight);
      this.#orientFingersContinuous(side, framePosition, easeWeight);
      this.#distributeForearmTwist(side, easeWeight);
    }

    this.#applyFace(
      faceFirst,
      faceSecond,
      faceAlpha,
      faceBlendshapeNames,
    );
    this.avatar.updateMatrixWorld(true);
  }

  #orientArmContinuous(side, framePosition, easeWeight) {
    const lengths = this.armLengths.get(side);
    const shoulderPos = lengths.shoulderRest;

    const wristOffset = sampleTrackSmooth(this.filtered[side].wristOffset, framePosition).clone();
    const pole = sampleTrackSmooth(this.filtered[side].pole, framePosition).clone();
    normalizedOrNull(pole);

    ikTargetWrist.copy(shoulderPos).add(wristOffset);
    solveTwoBoneIK(shoulderPos, ikTargetWrist, pole, lengths.upper, lengths.lower, ikSolvedElbow, ikSolvedWrist);

    const armBone = this.bones.get(`${side}Arm`);
    const foreArmBone = this.bones.get(`${side}ForeArm`);
    const armRest = this.naturalRestQuaternions.get(`${side}Arm`);
    const foreArmRest = this.naturalRestQuaternions.get(`${side}ForeArm`);

    armBone.quaternion.copy(armRest);
    foreArmBone.quaternion.copy(foreArmRest);
    this.avatar.updateMatrixWorld(true);

    this.#orientBone(`${side}Arm`, tempDirection.copy(ikSolvedElbow).sub(shoulderPos));
    const armSolved = scratchQuatA.copy(armBone.quaternion);
    armBone.quaternion.copy(armRest).slerp(armSolved, easeWeight);
    armBone.updateWorldMatrix(false, true);

    this.#orientBone(`${side}ForeArm`, tempDirection.copy(ikSolvedWrist).sub(ikSolvedElbow));
    const foreArmSolved = scratchQuatB.copy(foreArmBone.quaternion);
    foreArmBone.quaternion.copy(foreArmRest).slerp(foreArmSolved, easeWeight);
    foreArmBone.updateWorldMatrix(false, true);

    this.#updateDebugMarkers(side, ikSolvedElbow, ikSolvedWrist);
  }

  #orientHandContinuous(side, framePosition, easeWeight) {
    const handBone = this.bones.get(`${side}Hand`);
    const localPalmBasis = this.palmBases.get(side);
    const handRest = this.restQuaternions.get(`${side}Hand`);

    if (!handBone || !localPalmBasis || !handRest) return;

    tempForward.copy(
      sampleTrackSmooth(
        this.filtered[side].palmForward,
        framePosition,
      ),
    );

    tempSide.copy(
      sampleTrackSmooth(
        this.filtered[side].palmSide,
        framePosition,
      ),
    );

    if (!orthonormalPalmBasis(tempForward, tempSide, tempNormal)) return;

    const offsetDeg = side === 'Left'
      ? (TUNING.palmPitchOffsetLeftDeg || 0)
      : (TUNING.palmPitchOffsetRightDeg || 0);

    if (Math.abs(offsetDeg) > 1e-4) {
      scratchQuatA.setFromAxisAngle(
        tempSide,
        THREE.MathUtils.degToRad(offsetDeg),
      );

      tempForward.applyQuaternion(scratchQuatA).normalize();
      tempNormal.crossVectors(tempSide, tempForward).normalize();
    }

    tempMatrix.makeBasis(
      tempSide,
      tempForward,
      tempNormal,
    );

    tempDesiredQuaternion
      .setFromRotationMatrix(tempMatrix)
      .multiply(
        scratchQuatA.copy(localPalmBasis).invert(),
      );

    handBone.parent
      .getWorldQuaternion(tempParentQuaternion)
      .invert();

    tempLocalQuaternion
      .copy(tempParentQuaternion)
      .multiply(tempDesiredQuaternion)
      .normalize();

    const confidence = sampleScalarTrack(
      this.filtered[side].confidence,
      framePosition,
    );

    const floor = THREE.MathUtils.clamp(
      TUNING.palmConfidenceFloor ?? 0.45,
      0,
      0.95,
    );

    const confidenceWeight = confidence <= floor
      ? 0
      : smoothStepLocal((confidence - floor) / (1 - floor));

    const authority = THREE.MathUtils.clamp(
      TUNING.palmAuthority ?? 1,
      0,
      1,
    );

    const weight = easeWeight * authority * confidenceWeight;

    handBone.quaternion
      .copy(handRest)
      .slerp(tempLocalQuaternion, weight)
      .normalize();

    handBone.updateWorldMatrix(false, true);
  }

  // THE FIX.  Drives all 15 finger bones per hand from the filtered
  // direction tracks, with a flexion gain and a confidence weight.
  #orientFingersContinuous(side, framePosition, easeWeight) {
    if (side === 'Left' && !TUNING.driveLeftFingers) return;
    if (side === 'Right' && !TUNING.driveRightFingers) return;

    const tracks = this.filtered[side]?.fingers;
    if (!tracks) return;

    // IMPORTANT: filtered confidence is intentionally smooth, which means it
    // can stay high for a few frames after MediaPipe has actually lost the hand.
    // That was allowing gap-filled finger tracks to keep driving the avatar
    // during missing detections.  For faithful mode we therefore consult the
    // ORIGINAL per-frame detection mask before allowing any finger drive.
    //
    // Conservative rule:
    //   - exact integer frame: that frame must be detected;
    //   - between two frames: BOTH neighbours must be detected.
    //
    // This prevents interpolation across a missing source frame.  Frames
    // recovered/validated upstream remain active because their JSON has
    // HandDetected=true.
    let sourceDetectionWeight = 1;

    if (TUNING.faithfulMode && TUNING.faithfulFingersRequireDetection) {
      const detected = this.direct?.[side]?.handDetected;

      if (detected?.length) {
        const i0 = THREE.MathUtils.clamp(
          Math.floor(framePosition),
          0,
          detected.length - 1,
        );
        const i1 = Math.min(i0 + 1, detected.length - 1);
        const frac = THREE.MathUtils.clamp(framePosition - i0, 0, 1);

        if (frac <= 1e-6 || i0 === i1) {
          sourceDetectionWeight = detected[i0] ? 1 : 0;
        } else {
          sourceDetectionWeight = (detected[i0] && detected[i1]) ? 1 : 0;
        }
      } else {
        sourceDetectionWeight = 0;
      }
    }

    const confidence = sampleScalarTrack(
      this.filtered[side].confidence,
      framePosition,
    );
    const floor = THREE.MathUtils.clamp(
      TUNING.fingerConfidenceFloor,
      0,
      1,
    );
    const confidenceWeight = confidence <= floor
      ? 0
      : smoothStepLocal((confidence - floor) / (1 - floor));

    const drive = confidenceWeight * sourceDetectionWeight * easeWeight;

    for (const [finger, landmarks] of Object.entries(FINGERS)) {
      const isThumb = finger === 'Thumb';
      const gain = isThumb ? TUNING.thumbGain : TUNING.fingerGain;
      const limits = isThumb ? TUNING.maxThumbFlex : TUNING.maxFlex;

      for (let segment = 1; segment <= 3; segment += 1) {
        const name = `${side}Hand${finger}${segment}`;
        const bone = this.bones.get(name);
        const rest = this.restQuaternions.get(name);
        const neutral = this.neutralFingerQuaternions.get(name) || rest;
        if (!bone || !rest) continue;

        if (drive <= 0.001) {
          bone.quaternion.copy(neutral);
          bone.updateWorldMatrix(false, true);
          continue;
        }

        const track = tracks.get(name);
        if (!track) continue;

        // 1. aim the bone down the measured finger segment
        bone.quaternion.copy(rest);
        bone.updateWorldMatrix(false, true);
        this.#orientBone(name, sampleTrackSmooth(track, framePosition, tempTargetDirection));

        // 2. scale the resulting bend, then clamp to an anatomical range
        scratchQuatA.copy(rest).invert().multiply(bone.quaternion);
        scaleAndClampRotation(
          scratchQuatA, gain, -TUNING.maxExtend, limits[segment] ?? 1.6, scratchQuatB,
        );
        bone.quaternion.copy(rest).multiply(scratchQuatB);

        // 3. fade toward the neutral shape where the data is not real
        bone.quaternion.copy(neutral).slerp(scratchQuatA.copy(bone.quaternion), drive).normalize();
        bone.updateWorldMatrix(false, true);
      }
      void landmarks;
    }
  }

  // The GLB carries LeftForeArm1 / LeftForeArm2 (and the Right pair) as
  // SIBLINGS of the hand under the forearm - skinning twist helpers that
  // nothing was writing to, so every degree of wrist roll pinched the mesh
  // at a single joint.  Give them a share of the hand's twist.  Because
  // they are siblings, this changes skinning only and cannot move the hand.
  #distributeForearmTwist(side, easeWeight) {
    const helpers = this.twistBones.get(side);
    if (!helpers?.length) return;

    const handBone = this.bones.get(`${side}Hand`);
    const handRest = this.restQuaternions.get(`${side}Hand`);
    const axis = this.forearmAxis.get(side);
    if (!handBone || !handRest || !axis) return;

    scratchQuatA.copy(handRest).invert().multiply(handBone.quaternion);
    decomposeTwist(scratchQuatA, axis, scratchQuatB);
    let twist = signedAngle(scratchQuatB);
    twist = THREE.MathUtils.clamp(twist, -TUNING.maxForearmTwist, TUNING.maxForearmTwist) * easeWeight;

    const shares = [TUNING.twistShare1, TUNING.twistShare2];
    helpers.forEach((bone, i) => {
      const rest = this.restQuaternions.get(bone.name);
      if (!rest) return;
      scratchQuatA.setFromAxisAngle(axis, twist * (shares[i] ?? 0.5));
      bone.quaternion.copy(rest).multiply(scratchQuatA);
      bone.updateWorldMatrix(false, true);
    });
  }

  #applyFace(first, second, alpha, names) {
    if (!first?.face?.length || !names.length) return;
    names.forEach((name, index) => {
      const value = THREE.MathUtils.lerp(first.face[index], second.face[index], alpha);
      for (const { object, index: morphIndex } of this.morphBindings.get(name) || []) {
        object.morphTargetInfluences[morphIndex] = value;
      }
    });
  }

  measureFidelity(clip) {
    const frames = clip.frames;
    const out = {};

    for (const side of SIDES) {
      const lower = side.toLowerCase();
      const sIdx = POSE[`${lower}Shoulder`];
      const eIdx = POSE[`${lower}Elbow`];
      const wIdx = POSE[`${lower}Wrist`];

      const sourceDirection = (a, b) => new THREE.Vector3(
        AXIS_SIGN.x * (a[0] - b[0]),
        AXIS_SIGN.y * (a[1] - b[1]),
        AXIS_SIGN.z * (a[2] - b[2]),
      ).normalize();

      const armBone = this.bones.get(`${side}Arm`);
      const foreBone = this.bones.get(`${side}ForeArm`);
      const handBone = this.bones.get(`${side}Hand`);

      const shoulderPos = new THREE.Vector3();
      const elbowPos = new THREE.Vector3();
      const wristPos = new THREE.Vector3();

      let upperSum = 0;
      let upperMax = 0;
      let foreSum = 0;
      let foreMax = 0;
      let flexSum = 0;
      let flexMax = 0;

      for (let i = 0; i < frames.length; i += 1) {
        this.applyContinuousFrame(
          i,
          frames[i],
          frames[i],
          0,
          1,
          [],
        );

        armBone.getWorldPosition(shoulderPos);
        foreBone.getWorldPosition(elbowPos);
        handBone.getWorldPosition(wristPos);

        const avatarUpper = elbowPos.clone().sub(shoulderPos).normalize();
        const avatarFore = wristPos.clone().sub(elbowPos).normalize();

        const sourceUpper = sourceDirection(
          frames[i].poseWorld[eIdx],
          frames[i].poseWorld[sIdx],
        );
        const sourceFore = sourceDirection(
          frames[i].poseWorld[wIdx],
          frames[i].poseWorld[eIdx],
        );

        const upperError = THREE.MathUtils.radToDeg(
          avatarUpper.angleTo(sourceUpper),
        );
        const foreError = THREE.MathUtils.radToDeg(
          avatarFore.angleTo(sourceFore),
        );

        const avatarFlex = 180 - THREE.MathUtils.radToDeg(
          avatarUpper.angleTo(avatarFore),
        );
        const sourceFlex = 180 - THREE.MathUtils.radToDeg(
          sourceUpper.angleTo(sourceFore),
        );
        const flexError = Math.abs(avatarFlex - sourceFlex);

        upperSum += upperError;
        upperMax = Math.max(upperMax, upperError);
        foreSum += foreError;
        foreMax = Math.max(foreMax, foreError);
        flexSum += flexError;
        flexMax = Math.max(flexMax, flexError);
      }

      const n = frames.length;
      out[side] = {
        upperArmDirMean: +(upperSum / n).toFixed(2),
        upperArmDirMax: +upperMax.toFixed(1),
        foreArmDirMean: +(foreSum / n).toFixed(2),
        foreArmDirMax: +foreMax.toFixed(1),
        elbowFlexMean: +(flexSum / n).toFixed(2),
        elbowFlexMax: +flexMax.toFixed(1),
      };
    }

    console.table(out);
    console.info(
      `[Fidelity] mode = ${TUNING.faithfulMode ? 'FAITHFUL (direct FK)' : 'IK'} | `
      + 'target for faithful mode: upper/fore direction error < 0.5 deg',
    );

    return out;
  }

  /* ---------------- pose helpers ---------------- */

  // Minimal rotation that swings the bone's own long axis onto
  // `targetDirection`, converted into parent-local space.  The caller is
  // responsible for having set the parent first.
  #orientBone(name, targetDirection) {
    const bone = this.bones.get(name);
    const axis = this.axes.get(name);
    tempTargetDirection.copy(targetDirection);
    if (!bone || !axis || !normalizedOrNull(tempTargetDirection)) return;

    bone.updateWorldMatrix(true, false);
    bone.getWorldQuaternion(tempWorldQuaternion);
    tempDirection.copy(axis).applyQuaternion(tempWorldQuaternion).normalize();
    tempDeltaQuaternion.setFromUnitVectors(tempDirection, tempTargetDirection);
    tempDesiredQuaternion.copy(tempDeltaQuaternion).multiply(tempWorldQuaternion);
    bone.parent.getWorldQuaternion(tempParentQuaternion).invert();
    tempLocalQuaternion.copy(tempParentQuaternion).multiply(tempDesiredQuaternion);
    bone.quaternion.copy(tempLocalQuaternion).normalize();
    bone.updateWorldMatrix(false, true);
  }

  // Idle pose: relaxed arms, flattened-neutral fingers.
  reset() {
    for (const [name, bone] of this.bones) {
      const natural = this.naturalRestQuaternions.get(name);
      const neutral = this.neutralFingerQuaternions.get(name);
      bone.quaternion.copy(natural || neutral || this.restQuaternions.get(name));
    }
    this.avatar.updateMatrixWorld(true);
  }

  // Exact GLB bind pose.  Used by the identity test - see TUNING.
  resetToBindPose() {
    for (const [name, bone] of this.bones) {
      bone.quaternion.copy(this.restQuaternions.get(name));
    }
    this.avatar.updateMatrixWorld(true);
  }

  // Re-derive the neutral finger pose after TUNING.fingerRelax changes.
  refreshNeutralFingerPose() {
    this.#captureNeutralFingerPose('Left');
    this.#captureNeutralFingerPose('Right');
  }

  /* ---------------- debug ---------------- */

  enableDebug(scene) {
    this.debugScene = scene;
    this.debugEnabled = true;
    if (!this.debugMarkers) {
      const makeMarker = (color, size) => new THREE.Mesh(
        new THREE.SphereGeometry(size, 12, 12),
        new THREE.MeshBasicMaterial({ color, depthTest: false }),
      );
      this.debugMarkers = {};
      for (const side of SIDES) {
        this.debugMarkers[side] = {
          targetElbow: makeMarker(0xffaa00, 0.018),
          targetWrist: makeMarker(0xff2222, 0.022),
          actualElbow: makeMarker(0x00ccff, 0.014),
          actualWrist: makeMarker(0x33ff66, 0.018),
        };
      }
    }
    for (const side of SIDES) {
      for (const marker of Object.values(this.debugMarkers[side])) scene.add(marker);
    }
  }

  disableDebug() {
    this.debugEnabled = false;
    if (this.debugMarkers && this.debugScene) {
      for (const side of SIDES) {
        for (const marker of Object.values(this.debugMarkers[side])) {
          this.debugScene.remove(marker);
        }
      }
    }
  }

  #updateDebugMarkers(side, elbow, wrist) {
    if (!this.debugEnabled || !this.debugMarkers) return;
    const markers = this.debugMarkers[side];
    markers.targetElbow.position.copy(elbow);
    markers.targetWrist.position.copy(wrist);
    markers.actualElbow.position.copy(
      this.bones.get(`${side}ForeArm`).getWorldPosition(new THREE.Vector3()),
    );
    markers.actualWrist.position.copy(
      this.bones.get(`${side}Hand`).getWorldPosition(new THREE.Vector3()),
    );
  }
}

export class BakedAvatarMotion {
  constructor(clip, retargeter, options = {}) {
    this.clip = clip;
    this.retargeter = retargeter;
    this.avatar = retargeter.avatar;
    this.duration = clip.duration;
    this.easeInSeconds = options.easeInSeconds ?? 0.4;
    this.easeOutSeconds = options.easeOutSeconds ?? 0.4;
    this.#compile();
  }

  #compile() {
    this.retargeter.clipInfo = {
      aspectYOverX: this.clip.aspectYOverX,
      width: this.clip.sourceWidth,
      height: this.clip.sourceHeight,
      fps: this.clip.fps,
    };
    this.retargeter.calibrateScale(this.clip.frames);
    this.retargeter.prepareFilteredTracks(this.clip.frames, this.clip.fps);
    this.retargeter.reset();
  }

  sample(timeSeconds) {
    const clamped = THREE.MathUtils.clamp(timeSeconds, 0, this.duration);
    const framePosition = THREE.MathUtils.clamp(
      clamped * this.clip.fps, 0, this.clip.frameCount - 1,
    );
    const firstIndex = Math.floor(framePosition);
    const secondIndex = Math.min(firstIndex + 1, this.clip.frameCount - 1);
    const alpha = framePosition - firstIndex;
    const easeWeight = easeEnvelope(clamped, this.duration, this.easeInSeconds, this.easeOutSeconds);

    this.retargeter.applyContinuousFrame(
      framePosition,
      this.clip.frames[firstIndex],
      this.clip.frames[secondIndex],
      alpha,
      easeWeight,
      this.clip.faceBlendshapeNames,
    );

    return { ended: timeSeconds >= this.duration, framePosition };
  }

  // Scrub to an exact frame with no ease envelope - for frame-by-frame
  // comparison against the source video.
  seekFrame(frameIndex) {
    const framePosition = THREE.MathUtils.clamp(frameIndex, 0, this.clip.frameCount - 1);
    const firstIndex = Math.floor(framePosition);
    const secondIndex = Math.min(firstIndex + 1, this.clip.frameCount - 1);
    this.retargeter.applyContinuousFrame(
      framePosition,
      this.clip.frames[firstIndex],
      this.clip.frames[secondIndex],
      framePosition - firstIndex,
      1,
      this.clip.faceBlendshapeNames,
    );
    return { framePosition };
  }
}