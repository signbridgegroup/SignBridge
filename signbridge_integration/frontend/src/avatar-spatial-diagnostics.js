import * as THREE from 'three';

/**
 * SignBridge spatial diagnostics
 * ------------------------------
 * Diagnostic only. This file does NOT modify retargeting.
 *
 * Measures, frame by frame:
 * - avatar wrist positions normalized by avatar shoulder width
 * - source wrist positions normalized by source shoulder width
 * - source-vs-avatar wrist location error
 * - left-right hand distance
 * - palm-center position
 * - fingertip positions where tip bones exist
 * - a torso penetration RISK proxy based on a spine capsule
 *
 * IMPORTANT:
 * The torso capsule is only a geometric proxy, NOT proof of mesh collision.
 * Negative proxy distance means "inside the diagnostic torso capsule".
 */

const AXIS_SIGN = Object.freeze({ x: 1, y: -1, z: -1 });

const POSE = Object.freeze({
  leftShoulder: 11,
  rightShoulder: 12,
  leftWrist: 15,
  rightWrist: 16,
});

const SIDES = ['Left', 'Right'];
const FINGERS = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'];

const vA = new THREE.Vector3();
const vB = new THREE.Vector3();
const vC = new THREE.Vector3();

function finiteVec3(v) {
  return Number.isFinite(v.x) && Number.isFinite(v.y) && Number.isFinite(v.z);
}

function rawPoint(raw) {
  return new THREE.Vector3(
    AXIS_SIGN.x * raw[0],
    AXIS_SIGN.y * raw[1],
    AXIS_SIGN.z * raw[2],
  );
}

function median(values) {
  const a = values.filter(Number.isFinite).sort((x, y) => x - y);
  if (!a.length) return NaN;
  const m = Math.floor(a.length / 2);
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}

function percentile(values, p) {
  const a = values.filter(Number.isFinite).sort((x, y) => x - y);
  if (!a.length) return NaN;
  const x = (a.length - 1) * p;
  const lo = Math.floor(x);
  const hi = Math.ceil(x);
  if (lo === hi) return a[lo];
  return THREE.MathUtils.lerp(a[lo], a[hi], x - lo);
}

function pointSegmentDistance(point, a, b) {
  vA.copy(b).sub(a);
  const len2 = vA.lengthSq();
  if (len2 < 1e-12) return point.distanceTo(a);

  const t = THREE.MathUtils.clamp(
    vB.copy(point).sub(a).dot(vA) / len2,
    0,
    1,
  );

  vC.copy(a).addScaledVector(vA, t);
  return point.distanceTo(vC);
}

function firstExisting(root, names) {
  for (const name of names) {
    const obj = root.getObjectByName(name);
    if (obj) return obj;
  }
  return null;
}

function worldPosition(object) {
  return object?.getWorldPosition(new THREE.Vector3()) ?? null;
}

function palmCenter(retargeter, side) {
  const names = [
    `${side}Hand`,
    `${side}HandIndex1`,
    `${side}HandMiddle1`,
    `${side}HandPinky1`,
  ];

  const pts = names
    .map((name) => worldPosition(retargeter.bones.get(name)))
    .filter(Boolean);

  if (!pts.length) return null;

  const out = new THREE.Vector3();
  for (const p of pts) out.add(p);
  return out.multiplyScalar(1 / pts.length);
}

function fingertipPosition(retargeter, side, finger) {
  const root = retargeter.avatar;

  // MetaPerson rigs commonly include the terminal *4 bone although
  // the active retargeter drives only segments 1..3.
  const tip4 = root.getObjectByName(`${side}Hand${finger}4`);
  if (tip4) return worldPosition(tip4);

  const distal = retargeter.bones.get(`${side}Hand${finger}3`);
  if (!distal) return null;

  // Fallback: distal bone position. Labelled as fallback in the CSV.
  return worldPosition(distal);
}

function torsoDefinition(retargeter) {
  const root = retargeter.avatar;

  const leftShoulder = retargeter.bones.get('LeftArm');
  const rightShoulder = retargeter.bones.get('RightArm');

  const hips = firstExisting(root, [
    'Hips',
    'hips',
    'Pelvis',
    'pelvis',
  ]);

  const spineLower = firstExisting(root, [
    'Spine',
    'Spine1',
    'spine',
    'spine_01',
  ]);

  const neck = firstExisting(root, [
    'Neck',
    'neck',
    'Neck1',
  ]);

  if (!leftShoulder || !rightShoulder) {
    throw new Error('Could not locate LeftArm / RightArm shoulder bones.');
  }

  const l = worldPosition(leftShoulder);
  const r = worldPosition(rightShoulder);
  const shoulderMid = l.clone().add(r).multiplyScalar(0.5);
  const shoulderWidth = l.distanceTo(r);

  let lower = worldPosition(hips) || worldPosition(spineLower);
  if (!lower) {
    // Conservative fallback derived from the avatar's own shoulder width.
    lower = shoulderMid.clone().add(new THREE.Vector3(0, -shoulderWidth * 1.25, 0));
  }

  let upper = worldPosition(neck);
  if (!upper) {
    upper = shoulderMid.clone().lerp(lower, 0.08);
  }

  // Proxy only: capsule radius derived from the avatar's own shoulder span.
  const radius = shoulderWidth * 0.34;

  return {
    leftShoulder: l,
    rightShoulder: r,
    shoulderMid,
    shoulderWidth,
    torsoLower: lower,
    torsoUpper: upper,
    radius,
    foundHips: hips?.name ?? null,
    foundSpine: spineLower?.name ?? null,
    foundNeck: neck?.name ?? null,
  };
}

function normalizedAvatarPoint(point, torso) {
  return point.clone()
    .sub(torso.shoulderMid)
    .multiplyScalar(1 / Math.max(torso.shoulderWidth, 1e-9));
}

function sourceFrameBasis(frame) {
  if (!frame?.poseWorld) return null;

  const ls = rawPoint(frame.poseWorld[POSE.leftShoulder]);
  const rs = rawPoint(frame.poseWorld[POSE.rightShoulder]);
  const shoulderMid = ls.clone().add(rs).multiplyScalar(0.5);
  const shoulderWidth = ls.distanceTo(rs);

  if (!(shoulderWidth > 1e-9)) return null;

  return { ls, rs, shoulderMid, shoulderWidth };
}

function normalizedSourceWrist(frame, side) {
  const basis = sourceFrameBasis(frame);
  if (!basis) return null;

  const idx = side === 'Left' ? POSE.leftWrist : POSE.rightWrist;
  const wrist = rawPoint(frame.poseWorld[idx]);

  return wrist
    .sub(basis.shoulderMid)
    .multiplyScalar(1 / basis.shoulderWidth);
}

function sourceInterHandDistance(frame) {
  const basis = sourceFrameBasis(frame);
  if (!basis) return NaN;

  const lw = rawPoint(frame.poseWorld[POSE.leftWrist]);
  const rw = rawPoint(frame.poseWorld[POSE.rightWrist]);
  return lw.distanceTo(rw) / basis.shoulderWidth;
}

function applyExactFrame(retargeter, clip, frameIndex) {
  const i = THREE.MathUtils.clamp(
    Math.round(frameIndex),
    0,
    clip.frameCount - 1,
  );

  const frame = clip.frames[i];

  retargeter.applyContinuousFrame(
    i,
    frame,
    frame,
    0,
    1,
    clip.faceBlendshapeNames,
  );

  retargeter.avatar.updateMatrixWorld(true);
}

function pointRisk(point, torso) {
  if (!point) {
    return { signed: NaN, inside: false };
  }

  const distanceToAxis = pointSegmentDistance(
    point,
    torso.torsoLower,
    torso.torsoUpper,
  );

  const signed = (
    distanceToAxis - torso.radius
  ) / Math.max(torso.shoulderWidth, 1e-9);

  return {
    signed,
    inside: signed < 0,
  };
}

function fmt(value, digits = 5) {
  return Number.isFinite(value) ? Number(value.toFixed(digits)) : '';
}

function csvEscape(value) {
  const s = String(value ?? '');
  if (/[",\n]/.test(s)) return `"${s.replaceAll('"', '""')}"`;
  return s;
}

function downloadCsv(rows, filename) {
  if (!rows.length) return;

  const headers = Object.keys(rows[0]);
  const csv = [
    headers.join(','),
    ...rows.map(
      (row) => headers.map((h) => csvEscape(row[h])).join(','),
    ),
  ].join('\n');

  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export function inspectRigForSpatialDiagnostics() {
  const retargeter = window.motionRetargeter;
  if (!retargeter?.avatar) {
    throw new Error('window.motionRetargeter is not ready yet.');
  }

  const bones = [];
  const meshes = [];

  retargeter.avatar.traverse((obj) => {
    if (obj.isBone) bones.push(obj.name);
    if (obj.isSkinnedMesh || obj.isMesh) {
      meshes.push({
        name: obj.name,
        type: obj.type,
        triangles: obj.geometry?.index
          ? Math.floor(obj.geometry.index.count / 3)
          : Math.floor((obj.geometry?.attributes?.position?.count ?? 0) / 3),
      });
    }
  });

  console.info('[SpatialDiag] Bone names:');
  console.table(bones.map((name) => ({ name })));

  console.info('[SpatialDiag] Meshes:');
  console.table(meshes);

  return { bones, meshes };
}

export function runSpatialDiagnostics({
  download = true,
  filename = 'queue_avatar_spatial_diagnostics.csv',
} = {}) {
  const retargeter = window.motionRetargeter;
  const clip = window.stackMotionClip;

  if (!retargeter?.avatar) {
    throw new Error('Avatar/retargeter is not ready. Wait until the avatar page finishes loading.');
  }

  if (!clip?.frames?.length) {
    throw new Error('No motion is loaded in window.stackMotionClip.');
  }

  const originalPose = new Map();
  for (const [name, bone] of retargeter.bones) {
    originalPose.set(name, bone.quaternion.clone());
  }

  const rows = [];

  try {
    // Put avatar in exact current clip frame 0 first.
    applyExactFrame(retargeter, clip, 0);
    const torso0 = torsoDefinition(retargeter);

    console.info('[SpatialDiag] Torso proxy:', {
      shoulderWidth: torso0.shoulderWidth,
      radius: torso0.radius,
      hipsBone: torso0.foundHips,
      spineBone: torso0.foundSpine,
      neckBone: torso0.foundNeck,
      note: 'Negative proxy distance = inside diagnostic torso capsule; NOT proof of mesh collision.',
    });

    for (let i = 0; i < clip.frameCount; i += 1) {
      applyExactFrame(retargeter, clip, i);

      const torso = torsoDefinition(retargeter);
      const frame = clip.frames[i];

      const avatarLW = worldPosition(retargeter.bones.get('LeftHand'));
      const avatarRW = worldPosition(retargeter.bones.get('RightHand'));

      const avatarLP = palmCenter(retargeter, 'Left');
      const avatarRP = palmCenter(retargeter, 'Right');

      const normLW = normalizedAvatarPoint(avatarLW, torso);
      const normRW = normalizedAvatarPoint(avatarRW, torso);
      const normLP = avatarLP ? normalizedAvatarPoint(avatarLP, torso) : null;
      const normRP = avatarRP ? normalizedAvatarPoint(avatarRP, torso) : null;

      const srcLW = normalizedSourceWrist(frame, 'Left');
      const srcRW = normalizedSourceWrist(frame, 'Right');

      const leftWristError = srcLW ? normLW.distanceTo(srcLW) : NaN;
      const rightWristError = srcRW ? normRW.distanceTo(srcRW) : NaN;

      const leftPoints = {
        wrist: avatarLW,
        palm: avatarLP,
      };
      const rightPoints = {
        wrist: avatarRW,
        palm: avatarRP,
      };

      for (const finger of FINGERS) {
        leftPoints[`${finger.toLowerCase()}Tip`] =
          fingertipPosition(retargeter, 'Left', finger);
        rightPoints[`${finger.toLowerCase()}Tip`] =
          fingertipPosition(retargeter, 'Right', finger);
      }

      const leftRisks = Object.entries(leftPoints).map(([name, p]) => ({
        name,
        ...pointRisk(p, torso),
      }));

      const rightRisks = Object.entries(rightPoints).map(([name, p]) => ({
        name,
        ...pointRisk(p, torso),
      }));

      const leftWorst = leftRisks.reduce(
        (a, b) => (!Number.isFinite(a.signed) || b.signed < a.signed ? b : a),
        { name: '', signed: NaN, inside: false },
      );

      const rightWorst = rightRisks.reduce(
        (a, b) => (!Number.isFinite(a.signed) || b.signed < a.signed ? b : a),
        { name: '', signed: NaN, inside: false },
      );

      const avatarInterHand =
        avatarLW.distanceTo(avatarRW) / Math.max(torso.shoulderWidth, 1e-9);

      rows.push({
        frame: i,
        time_s: fmt(i / clip.fps, 4),

        src_left_wrist_x_sw: fmt(srcLW?.x),
        src_left_wrist_y_sw: fmt(srcLW?.y),
        src_left_wrist_z_sw: fmt(srcLW?.z),
        avatar_left_wrist_x_sw: fmt(normLW.x),
        avatar_left_wrist_y_sw: fmt(normLW.y),
        avatar_left_wrist_z_sw: fmt(normLW.z),
        left_wrist_location_error_sw: fmt(leftWristError),

        src_right_wrist_x_sw: fmt(srcRW?.x),
        src_right_wrist_y_sw: fmt(srcRW?.y),
        src_right_wrist_z_sw: fmt(srcRW?.z),
        avatar_right_wrist_x_sw: fmt(normRW.x),
        avatar_right_wrist_y_sw: fmt(normRW.y),
        avatar_right_wrist_z_sw: fmt(normRW.z),
        right_wrist_location_error_sw: fmt(rightWristError),

        avatar_left_palm_x_sw: fmt(normLP?.x),
        avatar_left_palm_y_sw: fmt(normLP?.y),
        avatar_left_palm_z_sw: fmt(normLP?.z),
        avatar_right_palm_x_sw: fmt(normRP?.x),
        avatar_right_palm_y_sw: fmt(normRP?.y),
        avatar_right_palm_z_sw: fmt(normRP?.z),

        src_inter_hand_distance_sw: fmt(sourceInterHandDistance(frame)),
        avatar_inter_hand_distance_sw: fmt(avatarInterHand),

        left_min_torso_proxy_signed_sw: fmt(leftWorst.signed),
        left_min_torso_proxy_point: leftWorst.name,
        left_inside_torso_proxy: leftWorst.inside ? 1 : 0,

        right_min_torso_proxy_signed_sw: fmt(rightWorst.signed),
        right_min_torso_proxy_point: rightWorst.name,
        right_inside_torso_proxy: rightWorst.inside ? 1 : 0,

        left_hand_detected: frame.leftHandDetected ? 1 : 0,
        right_hand_detected: frame.rightHandDetected ? 1 : 0,
        left_hand_source: frame.leftHandSource ?? '',
        right_hand_source: frame.rightHandSource ?? '',
      });
    }
  } finally {
    // Restore the exact pose that existed before the diagnostic.
    for (const [name, q] of originalPose) {
      const bone = retargeter.bones.get(name);
      if (bone) bone.quaternion.copy(q);
    }
    retargeter.avatar.updateMatrixWorld(true);
  }

  const leftErrors = rows.map((r) => Number(r.left_wrist_location_error_sw));
  const rightErrors = rows.map((r) => Number(r.right_wrist_location_error_sw));

  const leftRiskFrames = rows
    .filter((r) => r.left_inside_torso_proxy)
    .map((r) => r.frame);

  const rightRiskFrames = rows
    .filter((r) => r.right_inside_torso_proxy)
    .map((r) => r.frame);

  const summary = {
    clip: clip.name,
    frames: clip.frameCount,
    fps: clip.fps,

    leftWristLocationErrorMedianSW: fmt(median(leftErrors), 4),
    leftWristLocationErrorP95SW: fmt(percentile(leftErrors, 0.95), 4),
    leftWristLocationErrorMaxSW: fmt(Math.max(...leftErrors.filter(Number.isFinite)), 4),

    rightWristLocationErrorMedianSW: fmt(median(rightErrors), 4),
    rightWristLocationErrorP95SW: fmt(percentile(rightErrors, 0.95), 4),
    rightWristLocationErrorMaxSW: fmt(Math.max(...rightErrors.filter(Number.isFinite)), 4),

    leftTorsoProxyRiskFrameCount: leftRiskFrames.length,
    leftTorsoProxyRiskFrames: leftRiskFrames.join(' '),

    rightTorsoProxyRiskFrameCount: rightRiskFrames.length,
    rightTorsoProxyRiskFrames: rightRiskFrames.join(' '),

    note: 'Torso proxy is a diagnostic capsule, not exact avatar-mesh collision.',
  };

  console.info('[SpatialDiag] Summary');
  console.table([summary]);

  const ranked = [...rows]
    .sort((a, b) => (
      Math.min(
        Number(a.left_min_torso_proxy_signed_sw),
        Number(a.right_min_torso_proxy_signed_sw),
      )
      - Math.min(
        Number(b.left_min_torso_proxy_signed_sw),
        Number(b.right_min_torso_proxy_signed_sw),
      )
    ))
    .slice(0, 15);

  console.info('[SpatialDiag] 15 highest torso-penetration-risk frames');
  console.table(ranked.map((r) => ({
    frame: r.frame,
    time_s: r.time_s,
    leftProxy: r.left_min_torso_proxy_signed_sw,
    leftPoint: r.left_min_torso_proxy_point,
    rightProxy: r.right_min_torso_proxy_signed_sw,
    rightPoint: r.right_min_torso_proxy_point,
    leftWristError: r.left_wrist_location_error_sw,
    rightWristError: r.right_wrist_location_error_sw,
    sourceHandsApart: r.src_inter_hand_distance_sw,
    avatarHandsApart: r.avatar_inter_hand_distance_sw,
  })));

  if (download) {
    downloadCsv(rows, filename);
    console.info(`[SpatialDiag] CSV download requested: ${filename}`);
  }

  window.__signbridgeSpatialRows = rows;
  window.__signbridgeSpatialSummary = summary;

  return { summary, rows };
}
