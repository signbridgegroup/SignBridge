/**
 * Focused, framework-free tests for CameraRecorder's pure orientation/state
 * logic (no DOM, no getUserMedia, no MediaRecorder - those need a real
 * browser and are out of scope here; see docs/CAMERA_ORIENTATION.md for the
 * manual browser checklist). Run with plain Node, no test runner installed:
 *
 *   node src/camera-recorder.test.mjs
 *
 * camera-recorder.js has zero imports and does not touch the DOM except
 * inside start(), so importing and exercising shouldMirrorPreview and
 * _describeError works under plain Node with no jsdom/browser needed.
 */
import assert from 'node:assert/strict';
import { CameraRecorder } from './camera-recorder.js';

let passed = 0;
function test(name, fn) {
  try {
    fn();
    passed += 1;
    console.log(`[OK] ${name}`);
  } catch (error) {
    console.error(`[FAIL] ${name}`);
    console.error(error);
    process.exitCode = 1;
  }
}

test('defaults to user (front) facing mode', () => {
  const recorder = new CameraRecorder();
  assert.equal(recorder.facingMode, 'user');
});

test('front camera should be mirrored', () => {
  const recorder = new CameraRecorder();
  recorder.facingMode = 'user';
  assert.equal(recorder.shouldMirrorPreview, true);
});

test('rear/environment camera should not be mirrored', () => {
  const recorder = new CameraRecorder();
  recorder.facingMode = 'environment';
  assert.equal(recorder.shouldMirrorPreview, false);
});

test('an unrecognized facing mode still mirrors by default (safer default: selfie-style)', () => {
  const recorder = new CameraRecorder();
  recorder.facingMode = 'left'; // e.g. a multi-camera device constraint value
  assert.equal(recorder.shouldMirrorPreview, true);
});

test('permission-denied error maps to a clear Arabic message', () => {
  const recorder = new CameraRecorder();
  const described = recorder._describeError({ name: 'NotAllowedError' });
  assert.match(described.message, /رفض/);
  assert.equal(described.cameraErrorName, 'NotAllowedError');
});

test('no-camera-found error maps to a distinct Arabic message', () => {
  const recorder = new CameraRecorder();
  const described = recorder._describeError({ name: 'NotFoundError' });
  assert.match(described.message, /لم يتم العثور/);
});

test('camera-in-use error maps to a distinct Arabic message', () => {
  const recorder = new CameraRecorder();
  const described = recorder._describeError({ name: 'NotReadableError' });
  assert.match(described.message, /قيد الاستخدام/);
});

test('each mapped error state has a message distinct from the others', () => {
  const recorder = new CameraRecorder();
  const names = ['NotAllowedError', 'NotFoundError', 'NotReadableError', 'OverconstrainedError', 'SecurityError'];
  const messages = new Set(names.map((name) => recorder._describeError({ name }).message));
  assert.equal(messages.size, names.length, 'every error state must produce a distinguishable message');
});

test('unknown error names fall back to a generic Arabic message without throwing', () => {
  const recorder = new CameraRecorder();
  const described = recorder._describeError({ name: 'SomeBrandNewBrowserError' });
  assert.equal(typeof described.message, 'string');
  assert.ok(described.message.length > 0);
  assert.equal(described.cameraErrorName, 'SomeBrandNewBrowserError');
});

test('the original browser error is preserved as .cause for debugging', () => {
  const recorder = new CameraRecorder();
  const original = { name: 'NotAllowedError' };
  const described = recorder._describeError(original);
  assert.equal(described.cause, original);
});

test('closeCamera stops every track and clears internal stream/recorder references', () => {
  const recorder = new CameraRecorder();
  let stopped = 0;
  recorder.stream = { getTracks: () => [{ stop: () => { stopped += 1; } }, { stop: () => { stopped += 1; } }] };
  recorder.mediaRecorder = {};
  recorder.closeCamera();
  assert.equal(stopped, 2, 'every track must be stopped');
  assert.equal(recorder.stream, null);
  assert.equal(recorder.mediaRecorder, null);
});

test('closeCamera is a no-op (does not throw) when no stream was ever started', () => {
  const recorder = new CameraRecorder();
  assert.doesNotThrow(() => recorder.closeCamera());
});

console.log(`\n${passed} test(s) passed.`);
if (process.exitCode) {
  console.error('Some tests FAILED.');
}
