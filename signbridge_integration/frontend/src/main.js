import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader.js';
import './style.css';
import { askQuestion, fetchHealth, motionUrl, recognizeVideo } from './api.js';
import { CameraRecorder } from './camera-recorder.js';
import { MotionLibraryBrowser } from './motion-library.js';
import { MotionQueuePlayer } from './motion-queue.js';
import { t, getLanguage, setLanguage, applyTranslations, onLanguageChange } from './i18n.js';
import { getTheme, setTheme, onThemeChange } from './theme.js';
import { getColorVisionMode, setColorVisionMode, onColorVisionChange } from './accessibility.js';

const AVATAR_URL = '/quickmagic/SignBridge_Boy_Clean.glb';
const QUICKMAGIC_MOTION_BASE = '/quickmagic/motions';

// QuickMagic FBX files are used as animation sources only. Their meshes,
// materials, textures, and cameras are never added to the scene, so the
// visible avatar always remains SignBridge_Boy_Clean.glb.
const QUICKMAGIC_MOTION_FILES = {
  before: 'Before_01_1_BoyFBX.fbx',
  'binary relationship': 'Binary relationship_03_1_BoyFBX.fbx',
  call: 'Call_03_1_BoyFBX.fbx',
  char: 'Char_03_1_BoyFBX.fbx',
  check: 'Check_01_1_BoyFBX.fbx',
  code: 'Code_01_1_BoyFBX.fbx',
  column: 'column_01_1_BoyFBX.fbx',
  'complex composite': 'Complex- composite_01_1_BoyFBX.fbx',
  database: 'database_01_1_BoyFBX.fbx',
  declare: 'Declare_03_1_BoyFBX.fbx',
  depend: 'Depend_03_1_BoyFBX.fbx',
  derived: 'Derived_01_1_BoyFBX.fbx',
  description: 'Description_03_1_BoyFBX.fbx',
  diamond: 'diamond_01_1_BoyFBX.fbx',
  disjoint: 'disjoint_01_1_BoyFBX.fbx',
  domain: 'domain_01_1_BoyFBX.fbx',
  'dot operator': 'dot operator_03_1_BoyFBX.fbx',
};

function normalizeQuickMagicToken(token) {
  return String(token ?? '')
    .trim()
    .toLowerCase()
    .replace(/[_/\\-]+/g, ' ')
    .replace(/\s+/g, ' ');
}

function hasQuickMagicMotion(token) {
  const normalized = normalizeQuickMagicToken(token);
  return normalized === 'stack' || Boolean(QUICKMAGIC_MOTION_FILES[normalized]);
}

const QUICKMAGIC_SKIN_COLOR = 0xd9b79e;
const QUICKMAGIC_HEAD_TINT = 0xdfccc3;

/* ---------------- DOM references ---------------- */
const stage = document.querySelector('#avatar-stage');
const loadingLayer = document.querySelector('#loading-layer');
const loadingMessage = document.querySelector('#loading-message');
const avatarErrorCard = document.querySelector('#avatar-error-card');
const avatarErrorMessage = document.querySelector('#avatar-error-message');
const stageLabel = document.querySelector('#stage-label');
const viewButtons = [...document.querySelectorAll('[data-view]')];
const backendDot = document.querySelector('#backend-status-dot');
const backendText = document.querySelector('#backend-status-text');

// Step flow: #flow-entry presents one dominant choice ("ask" / "camera" /
// "upload" / "library"); choosing one hides the entry and shows #qa-flow
// with only that one panel visible, matching "one dominant action per
// screen" — no simultaneous tabs, no separate scroll-down tools section.
const flowEntry = document.querySelector('#flow-entry');
const qaFlow = document.querySelector('#qa-flow');
const flowBackButton = document.querySelector('#flow-back');

// Keep Back independent from qa-flow/grid/overflow so it can never be clipped
// or hidden behind a section. We control its visibility explicitly below.
if (flowBackButton) {
  document.body.appendChild(flowBackButton);
  flowBackButton.hidden = true;
}
const flowChoiceButtons = [...document.querySelectorAll('.flow-row, .library-spotlight, .flow-choice-minor, .ask-panel-alt-link')];
const panels = {
  ask: document.querySelector('#panel-ask'),
  camera: document.querySelector('#panel-camera'),
  upload: document.querySelector('#panel-upload'),
  library: document.querySelector('#panel-library'),
};

const askForm = document.querySelector('#ask-form');
const questionInput = document.querySelector('#question-input');
const uploadForm = document.querySelector('#upload-form');
const videoFileInput = document.querySelector('#video-file-input');
const uploadPreview = document.querySelector('#upload-preview');
const uploadReviewActions = document.querySelector('#upload-review-actions');
const uploadChooseAnotherButton = document.querySelector('#upload-choose-another');

const cameraPreview = document.querySelector('#camera-preview');
const cameraPlayback = document.querySelector('#camera-playback');
const cameraStartButton = document.querySelector('#camera-start-button');
const cameraRecordButton = document.querySelector('#camera-record-button');
const cameraStopButton = document.querySelector('#camera-stop-button');
const cameraRetakeButton = document.querySelector('#camera-retake-button');
const cameraAnalyzeButton = document.querySelector('#camera-analyze-button');
const cameraCloseButton = document.querySelector('#camera-close-button');
const cameraStatus = document.querySelector('#camera-status');
const recordingIndicator = document.querySelector('#recording-indicator');
const recordingTimer = document.querySelector('#recording-timer');

const processingStatus = document.querySelector('#processing-status');
const resultsPanel = document.querySelector('#results-panel');
const detectedQuestionText = document.querySelector('#detected-question-text');
const resultRecognition = document.querySelector('#result-recognition');
const recognitionFields = document.querySelector('#recognition-fields');
const resultAnswer = document.querySelector('#result-answer');
const educationalAnswerText = document.querySelector('#educational-answer-text');
const simplifiedAnswerText = document.querySelector('#simplified-answer-text');
const routedSubjectText = document.querySelector('#routed-subject-text');
const resultUnavailable = document.querySelector('#result-unavailable');
const unavailableNoteText = document.querySelector('#unavailable-note-text');
const resultSources = document.querySelector('#result-sources');
const sourcesList = document.querySelector('#sources-list');
const resultTokens = document.querySelector('#result-tokens');
const signTokensList = document.querySelector('#sign-tokens-list');
const resultMotions = document.querySelector('#result-motions');
const motionCoverageText = document.querySelector('#motion-coverage-text');
const missingMotionsList = document.querySelector('#missing-motions-list');

const nowPlayingText = document.querySelector('#now-playing-text');
const replayButton = document.querySelector('#replay-button');
const stopButton = document.querySelector('#stop-button');
const avatarCaptionText = document.querySelector('#avatar-caption-text');
const askAgainButton = document.querySelector('#ask-again-button');
const experienceStage = document.querySelector('.experience-stage');
const viewerCard = document.querySelector('.viewer-card');

// Library-only feedback note. Put it INSIDE the avatar card so the message
// is physically overlaid on the avatar instead of appearing below it.
let libraryMotionNote = document.querySelector('#library-motion-note');
if (!libraryMotionNote && viewerCard) {
  libraryMotionNote = document.createElement('p');
  libraryMotionNote.id = 'library-motion-note';
  libraryMotionNote.className = 'library-motion-note';
  libraryMotionNote.hidden = true;
  viewerCard.appendChild(libraryMotionNote);
}

/* ---------------- language + theme switchers ---------------- */
applyTranslations();

const langButtons = [...document.querySelectorAll('[data-lang]')];
function syncLangButtons() {
  const current = getLanguage();
  langButtons.forEach((button) => {
    button.classList.toggle('is-active', button.dataset.lang === current);
    button.setAttribute('aria-pressed', String(button.dataset.lang === current));
  });
  document.querySelectorAll('[data-current-language]').forEach((label) => {
    label.textContent = current === 'ar' ? t('lang.ar') : t('lang.en');
  });
}
langButtons.forEach((button) => button.addEventListener('click', () => setLanguage(button.dataset.lang)));
syncLangButtons();

const themeToggleButton = document.querySelector('#theme-toggle');
function syncThemeButton() {
  if (!themeToggleButton) return;
  const isDark = getTheme() === 'dark';
  themeToggleButton.setAttribute('aria-pressed', String(isDark));
  const label = themeToggleButton.querySelector('.theme-toggle-label');
  if (label) label.textContent = isDark ? t('theme.dark') : t('theme.light');
}
themeToggleButton?.addEventListener('click', () => setTheme(getTheme() === 'dark' ? 'light' : 'dark'));
syncThemeButton();
onThemeChange(syncThemeButton);

const colorVisionToggleButton = document.querySelector('#color-vision-toggle');
function syncColorVisionButton() {
  if (!colorVisionToggleButton) return;
  const friendly = getColorVisionMode() === 'friendly';
  colorVisionToggleButton.setAttribute('aria-pressed', String(friendly));
  const label = colorVisionToggleButton.querySelector('.accessibility-toggle-label');
  if (label) label.textContent = t(friendly ? 'accessibility.colorblind_on' : 'accessibility.colorblind_off');
}
colorVisionToggleButton?.addEventListener('click', () =>
  setColorVisionMode(getColorVisionMode() === 'friendly' ? 'default' : 'friendly'),
);
syncColorVisionButton();
onColorVisionChange(syncColorVisionButton);

const mobileMenuButton = document.querySelector('#mobile-menu-toggle');
const mobileMenu = document.querySelector('#mobile-menu');
mobileMenuButton?.addEventListener('click', () => {
  const isOpen = mobileMenu.classList.toggle('is-open');
  mobileMenuButton.setAttribute('aria-expanded', String(isOpen));
});
mobileMenu?.querySelectorAll('a, button').forEach((el) => {
  el.addEventListener('click', () => {
    mobileMenu.classList.remove('is-open');
    mobileMenuButton?.setAttribute('aria-expanded', 'false');
  });
});

const footerRights = document.querySelector('#footer-rights');
function syncFooterRights() {
  if (footerRights) footerRights.textContent = t('footer.rights', { year: new Date().getFullYear() });
}
syncFooterRights();


/* ---------------- smooth header condense on scroll ---------------- */
// Keep the condensed-header behavior, but use hysteresis + requestAnimationFrame
// so the header cannot rapidly toggle around a single scroll threshold.
// Condense only after scrolling well past the top, and expand again only
// after returning close to the top.
const siteHeader = document.querySelector('.site-header');

let headerCondensed = false;
let headerScrollTicking = false;

const HEADER_CONDENSE_AT = 72;
const HEADER_EXPAND_AT = 20;

function updateHeaderCondensedState() {
  if (!siteHeader) return;

  const y = window.scrollY || window.pageYOffset || 0;

  if (!headerCondensed && y >= HEADER_CONDENSE_AT) {
    headerCondensed = true;
    siteHeader.classList.add('is-condensed');
  } else if (headerCondensed && y <= HEADER_EXPAND_AT) {
    headerCondensed = false;
    siteHeader.classList.remove('is-condensed');
  }
}

function scheduleHeaderCondensedUpdate() {
  if (headerScrollTicking) return;

  headerScrollTicking = true;
  window.requestAnimationFrame(() => {
    updateHeaderCondensedState();
    headerScrollTicking = false;
  });
}

updateHeaderCondensedState();
window.addEventListener('scroll', scheduleHeaderCondensedUpdate, { passive: true });
window.addEventListener('resize', scheduleHeaderCondensedUpdate);

/* ---------------- smooth screen transitions ---------------- */
// Run on EVERY screen change, not only the first one.
// Timers are reset and the animation class is forcibly restarted so
// Home -> Ask/Camera/Upload/Library -> Results -> Back can all animate.
let screenTransitionOutTimer = null;
let screenTransitionInTimer = null;

function runScreenTransition(changeScreen) {
  const shell = document.querySelector('#experience');

  if (!shell || window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    changeScreen();
    return;
  }

  if (screenTransitionOutTimer) clearTimeout(screenTransitionOutTimer);
  if (screenTransitionInTimer) clearTimeout(screenTransitionInTimer);

  shell.classList.remove('is-screen-entering', 'is-screen-leaving');

  // Force the browser to register the clean state so the transition can
  // restart even if the previous navigation finished only moments ago.
  void shell.offsetWidth;

  shell.classList.add('is-screen-leaving');

  screenTransitionOutTimer = window.setTimeout(() => {
    changeScreen();

    shell.classList.remove('is-screen-leaving');

    requestAnimationFrame(() => {
      shell.classList.remove('is-screen-entering');
      void shell.offsetWidth;
      shell.classList.add('is-screen-entering');

      screenTransitionInTimer = window.setTimeout(() => {
        shell.classList.remove('is-screen-entering');
        screenTransitionInTimer = null;
      }, 280);
    });

    screenTransitionOutTimer = null;
  }, 150);
}

/* ---------------- header nav: active-section indicator (presentational only) ---------------- */
const headerNavLinks = [...document.querySelectorAll('.header-nav a[href^="#"]')];
const navObserverTargets = headerNavLinks
  .map((link) => document.querySelector(link.getAttribute('href')))
  .filter(Boolean);
if (headerNavLinks.length && navObserverTargets.length && 'IntersectionObserver' in window) {
  const setActiveNavLink = (id) => {
    headerNavLinks.forEach((link) => {
      link.classList.toggle('is-active', link.getAttribute('href') === `#${id}`);
    });
  };
  const navObserver = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (visible) setActiveNavLink(visible.target.id);
    },
    { rootMargin: '-45% 0px -45% 0px', threshold: [0, 0.25, 0.5, 0.75, 1] },
  );
  navObserverTargets.forEach((section) => navObserver.observe(section));
}

/* ---------------- gentle scroll-reveal polish ---------------- */
// Replay the reveal every time a section enters the viewport, whether the
// user is scrolling down or back up. Do not unobserve after the first reveal.
const revealTargets = [...document.querySelectorAll('.reveal-on-scroll')];
if (revealTargets.length) {
  if ('IntersectionObserver' in window && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    const revealObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          entry.target.classList.toggle('is-visible', entry.isIntersecting);
        });
      },
      { rootMargin: '0px 0px -10% 0px', threshold: 0.12 },
    );
    revealTargets.forEach((target) => revealObserver.observe(target));
  } else {
    revealTargets.forEach((target) => target.classList.add('is-visible'));
  }
}



/* ---------------- subtle text fade-in ----------------
 * Text only gets a very small upward fade so the interface still feels calm.
 * The delay is staggered inside each section, while buttons/media keep their
 * existing motion and remain immediately understandable/clickable.
 */
const textFadeSelectors = [
  '#hero .landing-copy > :not(.landing-actions):not(.university-pill)',
  '#about .about-copy-block > *',
  '#about .about-point h3',
  '#about .about-point p',
  '#how .project-section-inner > .section-kicker',
  '#how .project-section-inner > h2',
  '#how .how-step h3',
  '#how .how-step p',
  '#experience .home-copy > *',
  '#experience .flow-entry-inner > h2',
  '#experience .flow-entry-inner > .flow-entry-lead',
  '#thanks .thanks-copy > *',
  '.site-footer .footer-brand p',
  '.site-footer .footer-links > *',
];
const textFadeTargets = [...new Set(textFadeSelectors.flatMap((selector) => [...document.querySelectorAll(selector)]))];
textFadeTargets.forEach((target, index) => {
  target.classList.add('text-fade');
  target.style.setProperty('--text-fade-delay', `${Math.min((index % 6) * 55, 275)}ms`);
});

if (textFadeTargets.length) {
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if ('IntersectionObserver' in window && !reducedMotion) {
    const textFadeObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          entry.target.classList.toggle('is-text-visible', entry.isIntersecting);
        });
      },
      { rootMargin: '0px 0px -6% 0px', threshold: 0.08 },
    );
    textFadeTargets.forEach((target) => textFadeObserver.observe(target));
  } else {
    textFadeTargets.forEach((target) => target.classList.add('is-text-visible'));
  }
}

onLanguageChange(() => {
  syncLangButtons();
  syncThemeButton();
  syncColorVisionButton();
  syncFooterRights();
  // Cheap, always-deterministic UI bits refresh immediately; text that
  // depends on the last server/user interaction (recognition/answer
  // results, camera status) intentionally reflects the new language only
  // from the next action onward, documented as a known limitation.
  stageLabel.textContent = stageLabel.dataset.view === 'full' ? t('avatar.view.full') : t('avatar.view.signing');
});

/* ---------------- three.js scene (same setup as the approved viewer) ---------------- */
// Background/fog/floor colors only (visual redesign requirement: "a
// bright, neutral background that creates enough contrast" for the avatar
// stage) — camera, lights, controls, loading, and animation logic below
// are unchanged. The stage keeps a bright, brand-consistent icy-blue
// background (matching the "Clarity" palette) in both light and dark
// interface themes (the avatar itself needs strong, consistent contrast
// regardless of the surrounding page theme).
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xeaf6fc);
scene.fog = new THREE.Fog(0xeaf6fc, 6.5, 14);

const camera = new THREE.PerspectiveCamera(29, 1, 0.05, 100);
camera.position.set(0, 1.4, 3.5);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
stage.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.enablePan = false;
controls.minDistance = 1.1;
controls.maxDistance = 5.5;
controls.minPolarAngle = Math.PI * 0.22;
controls.maxPolarAngle = Math.PI * 0.72;

scene.add(new THREE.HemisphereLight(0xe4f4ff, 0x17283a, 2.15));
const keyLight = new THREE.DirectionalLight(0xfff7ef, 3.6);
keyLight.position.set(2.8, 4.5, 3.5);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(2048, 2048);
scene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0x80cbff, 1.4);
fillLight.position.set(-3.5, 2.5, 2.5);
scene.add(fillLight);

const rimLight = new THREE.DirectionalLight(0x70e2cb, 1.6);
rimLight.position.set(1.2, 3.3, -3.2);
scene.add(rimLight);

const floor = new THREE.Mesh(
  new THREE.CircleGeometry(2.4, 80),
  new THREE.MeshStandardMaterial({ color: 0xcfe8f3, roughness: 0.86, metalness: 0.04 }),
);
floor.rotation.x = -Math.PI / 2;
floor.receiveShadow = true;
scene.add(floor);

const clock = new THREE.Clock();
const loader = new GLTFLoader();
const fbxLoader = new FBXLoader();
const blinkBindings = [];

let avatar = null;
let avatarHeight = 1.7;
let nextBlinkAt = 2.2;
let blinkStartedAt = null;
let motionPlayer = null;
let quickMagicMixer = null;
let quickMagicAction = null;
let quickMagicLastToken = 'Stack';
let quickMagicQueue = [];
let quickMagicQueueIndex = 0;
let quickMagicPlaybackGeneration = 0;
let quickMagicLastSequence = [{ token: 'Stack', status: 'ready' }];
const quickMagicClipCache = new Map();

function setLoading(message) {
  loadingMessage.textContent = message;
  loadingLayer.hidden = false;
  avatarErrorCard.hidden = true;
}

function setAvatarReady() {
  loadingLayer.hidden = true;
  avatarErrorCard.hidden = true;
}

function setAvatarError(error) {
  loadingLayer.hidden = true;
  avatarErrorCard.hidden = false;
  avatarErrorMessage.textContent = t('avatar.error_message');
  console.error(error);
}

function collectBlinkTargets(mesh) {
  if (!mesh.morphTargetDictionary || !mesh.morphTargetInfluences) return;
  for (const name of ['eyeBlinkLeft', 'eyeBlinkRight']) {
    const index = mesh.morphTargetDictionary[name];
    if (Number.isInteger(index)) blinkBindings.push({ mesh, index });
  }
}

function fitAvatar(root) {
  root.updateMatrixWorld(true);
  const initialBox = new THREE.Box3().setFromObject(root);
  const center = initialBox.getCenter(new THREE.Vector3());
  root.position.x -= center.x;
  root.position.y -= initialBox.min.y;
  root.position.z -= center.z;
  root.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(root);
  avatarHeight = Math.max(box.getSize(new THREE.Vector3()).y, 1.45);
  setCameraView('signing', false);
}

function applyQuickMagicBlueFabric(object, material) {
  const signature = [
    object.name,
    material.name,
    material.map?.name,
    material.map?.image?.name,
  ].filter(Boolean).join(' ').toLowerCase();
  const isJacket = signature.includes('jacket');
  const isPants = signature.includes('pants');
  if (!isJacket && !isPants) return;

  material.onBeforeCompile = (shader) => {
    shader.fragmentShader = shader.fragmentShader.replace(
      '#include <map_fragment>',
      `#include <map_fragment>
#ifdef USE_MAP
  float sbMaxChannel = max(diffuseColor.r, max(diffuseColor.g, diffuseColor.b));
  float sbMinChannel = min(diffuseColor.r, min(diffuseColor.g, diffuseColor.b));
  float sbChroma = sbMaxChannel - sbMinChannel;
  float sbLuma = dot(diffuseColor.rgb, vec3(0.2126, 0.7152, 0.0722));
  float sbNeutral = 1.0 - smoothstep(0.060, 0.22, sbChroma);
  float sbNotWhite = 1.0 - smoothstep(0.82, 0.96, sbLuma);
  float sbNotInk = smoothstep(0.002, 0.020, sbLuma);
  float sbFabric = sbNeutral * sbNotWhite * sbNotInk;
  vec3 sbBlue = vec3(0.025, 0.39, 0.82);
  float sbTextureShade = mix(0.44, 1.14, pow(clamp(sbLuma, 0.0, 1.0), 0.32));
  diffuseColor.rgb = mix(diffuseColor.rgb, sbBlue * sbTextureShade, sbFabric * 0.985);

  float sbPrint = 1.0 - smoothstep(0.002, 0.018, sbLuma);
  diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.96, 0.985, 1.0), sbPrint);

  float sbSkinRG = diffuseColor.r - diffuseColor.g;
  float sbSkinGB = diffuseColor.g - diffuseColor.b;
  float sbSkinPatch =
    smoothstep(0.025, 0.10, sbSkinRG) *
    smoothstep(0.015, 0.09, sbSkinGB) *
    smoothstep(0.42, 0.68, sbLuma) *
    (1.0 - smoothstep(0.45, 0.75, sbSkinRG));
  diffuseColor.rgb = mix(
    diffuseColor.rgb,
    vec3(0.693872, 0.473531, 0.341914),
    sbSkinPatch
  );
#endif`,
    );
  };
  material.customProgramCacheKey = () => `signbridge-quickmagic-blue-${isJacket ? 'jacket' : 'pants'}`;
  material.needsUpdate = true;
}

function makeQuickMagicMaterial(source) {
  if (!source?.map) {
    return new THREE.MeshBasicMaterial({
      name: source?.name ?? 'SignBridge_Skin_Unified',
      color: QUICKMAGIC_SKIN_COLOR,
    });
  }

  source.map.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.MeshBasicMaterial({
    name: source.name,
    map: source.map,
    alphaMap: source.alphaMap ?? null,
    transparent: source.transparent,
    opacity: source.opacity,
    alphaTest: source.alphaTest,
    side: THREE.FrontSide,
    depthWrite: source.depthWrite,
  });
  const partName = (source.name ?? '').toLowerCase();
  material.color.set(partName.includes('head') ? QUICKMAGIC_HEAD_TINT : 0xffffff);
  material.vertexColors = false;
  return material;
}

function findQuickMagicBone(requestedName) {
  if (!avatar) return null;

  const normalizeName = (name) =>
    String(name || '')
      .toLowerCase()
      .replace(/[^a-z0-9]/g, '');

  const targetName = normalizeName(requestedName);
  let matchedBone = null;

  avatar.traverse((object) => {
    if (matchedBone || !object.isBone) return;

    const currentName = normalizeName(object.name);

    if (
      currentName === targetName ||
      currentName.endsWith(targetName) ||
      targetName.endsWith(currentName)
    ) {
      matchedBone = object;
    }
  });

  return matchedBone;
}

function aimQuickMagicBone(boneName, childName, worldDirection) {
  const bone = findQuickMagicBone(boneName);
  const child = findQuickMagicBone(childName);

  if (!bone || !child || !bone.parent) {
    console.warn('[QuickMagic idle pose] Bone not found:', {
      boneName,
      childName,
      foundBone: bone?.name,
      foundChild: child?.name,
    });
    return;
  }

  bone.parent.updateWorldMatrix(true, false);

  const parentWorldQuaternion = new THREE.Quaternion();
  bone.parent.getWorldQuaternion(parentWorldQuaternion);

  const desiredInParent = worldDirection
    .clone()
    .normalize()
    .applyQuaternion(parentWorldQuaternion.invert());

  const boneAxis = child.position.clone().normalize();

  bone.quaternion.setFromUnitVectors(boneAxis, desiredInParent);
  bone.updateWorldMatrix(true, true);
}

function setQuickMagicIdlePose() {
  if (!avatar) return;

  // A relaxed A-pose made independently from the Stack clip: shoulders are
  // slightly open and both forearms point down beside the torso.
 aimQuickMagicBone(
  'Bip001 L UpperArm',
  'Bip001 L Forearm',
  new THREE.Vector3(0.19, -1, 0.08),
);

aimQuickMagicBone(
  'Bip001 R UpperArm',
  'Bip001 R Forearm',
  new THREE.Vector3(-0.22, -1, 0.05),
);

aimQuickMagicBone(
  'Bip001 L Forearm',
  'Bip001 L Hand',
  new THREE.Vector3(-0.16, -0.90, 0.38),
);

aimQuickMagicBone(
  'Bip001 R Forearm',
  'Bip001 R Hand',
  new THREE.Vector3(0.12, -0.93, 0.30),
);
  avatar.updateMatrixWorld(true);
}

function setCameraView(view, animate = true) {
  if (!avatar) return;
  const presets = {
    full: {
      position: new THREE.Vector3(0, avatarHeight * 0.52, avatarHeight * 2.0),
      target: new THREE.Vector3(0, avatarHeight * 0.48, 0),
      labelKey: 'avatar.view.full',
    },
    signing: {
      position: new THREE.Vector3(0, avatarHeight * 0.70, avatarHeight * 1.32),
      target: new THREE.Vector3(0, avatarHeight * 0.69, 0),
      labelKey: 'avatar.view.signing',
    },
  };
  const destination = presets[view] || presets.signing;
  if (animate) {
    camera.position.lerp(destination.position, 0.82);
    controls.target.lerp(destination.target, 0.82);
  } else {
    camera.position.copy(destination.position);
    controls.target.copy(destination.target);
  }
  controls.minDistance = avatarHeight * 0.58;
  controls.maxDistance = avatarHeight * 2.7;
  controls.update();
  stageLabel.textContent = t(destination.labelKey);
  stageLabel.dataset.view = view;
  viewButtons.forEach((button) => {
    const active = button.dataset.view === view;
    button.classList.toggle('is-active', active);
    button.setAttribute('aria-pressed', String(active));
  });
}


function getQuickMagicTrackParts(trackName) {
  const lastDot = trackName.lastIndexOf('.');
  if (lastDot < 0) return { nodeName: trackName, property: '' };
  return {
    nodeName: trackName.slice(0, lastDot),
    property: trackName.slice(lastDot + 1),
  };
}

function isQuickMagicUpperBodyBone(nodeName) {
  const name = String(nodeName || '')
    .toLowerCase()
    .replace(/[^a-z0-9]/g, '');

  // Keep only the signing-relevant upper body. Lower-body rotations from
  // QuickMagic can bend the legs or move them outside the approved avatar
  // pose, so pelvis / thigh / calf / foot / toe tracks are intentionally
  // excluded. Finger and thumb tracks remain enabled.
  const upperBodyMarkers = [
    // Keep the torso locked to the approved neutral standing pose.
    // QuickMagic spine/chest rotations can contain whole-body lean, which
    // makes the avatar tip sideways even when the sign itself is correct.
    'neck',
    'head',
    'clavicle',
    'shoulder',
    'upperarm',
    'forearm',
    'hand',
    'finger',
    'thumb',
  ];

  return upperBodyMarkers.some((marker) => name.includes(marker));
}

function prepareQuickMagicClip(sourceClip, token) {
  // Upper-body-only motion retargeting:
  // - keep quaternion rotations for neck/head, shoulders, arms, hands and fingers;
  // - drop ALL position and scale tracks;
  // - drop torso lean plus ALL lower-body rotations (spine/chest/pelvis/thighs/calves/feet/toes).
  // This preserves the current SignBridge avatar's approved standing pose
  // and appearance while retaining the sign-language motion itself.
  const tracks = [];

  for (const sourceTrack of sourceClip.tracks) {
    const { nodeName, property } = getQuickMagicTrackParts(sourceTrack.name);
    if (property.toLowerCase() !== 'quaternion') continue;
    if (!isQuickMagicUpperBodyBone(nodeName)) continue;
    tracks.push(sourceTrack.clone());
  }

  const clip = new THREE.AnimationClip(
    `SignBridge_${normalizeQuickMagicToken(token).replace(/\s+/g, '_')}`,
    sourceClip.duration,
    tracks,
  );
  clip.resetDuration();
  return clip;
}

async function loadQuickMagicClip(token) {
  const normalized = normalizeQuickMagicToken(token);

  if (quickMagicClipCache.has(normalized)) {
    return quickMagicClipCache.get(normalized);
  }

  const filename = QUICKMAGIC_MOTION_FILES[normalized];
  if (!filename) return null;

  const loadPromise = (async () => {
    const source = await fbxLoader.loadAsync(
      encodeURI(`${QUICKMAGIC_MOTION_BASE}/${filename}`),
    );

    const sourceClip = source.animations?.[0];
    if (!sourceClip) {
      throw new Error(`QuickMagic FBX has no animation: ${filename}`);
    }

    return prepareQuickMagicClip(sourceClip, token);
  })();

  // Cache the promise immediately so repeated clicks do not download the
  // same FBX more than once.
  quickMagicClipCache.set(normalized, loadPromise);

  try {
    const clip = await loadPromise;
    quickMagicClipCache.set(normalized, clip);
    return clip;
  } catch (error) {
    quickMagicClipCache.delete(normalized);
    throw error;
  }
}

function startQuickMagicQueueEntry(generation) {
  if (!quickMagicMixer || generation !== quickMagicPlaybackGeneration) return;

  const entry = quickMagicQueue[quickMagicQueueIndex];
  if (!entry) {
    quickMagicAction = null;
    quickMagicMixer.update(0);
    setQuickMagicIdlePose();
    nowPlayingText.textContent = t('avatar.now_playing.complete');
    replayButton.disabled = false;
    stopButton.disabled = true;
    return;
  }

  quickMagicLastToken = entry.token;
  quickMagicAction = quickMagicMixer.clipAction(entry.clip);
  quickMagicAction.reset();
  quickMagicAction.setLoop(THREE.LoopOnce, 1);
  quickMagicAction.clampWhenFinished = true;
  quickMagicAction.paused = false;
  quickMagicAction.play();

  nowPlayingText.textContent = t('avatar.now_playing.playing', {
    token: quickMagicLastToken,
  });
  replayButton.disabled = false;
  stopButton.disabled = false;
}

async function playQuickMagicSequence(sequence) {
  if (!quickMagicMixer) return false;

  const requested = sequence.filter((item) => hasQuickMagicMotion(item.token));
  if (!requested.length) return false;

  const generation = ++quickMagicPlaybackGeneration;
  const loaded = [];

  for (const item of requested) {
    if (generation !== quickMagicPlaybackGeneration) return true;

    const normalized = normalizeQuickMagicToken(item.token);
    let clip = null;

    if (normalized === 'stack') {
      clip = quickMagicClipCache.get('stack') ?? null;
    } else {
      try {
        clip = await loadQuickMagicClip(item.token);
      } catch (error) {
        console.error(`[QuickMagic] Failed to load motion "${item.token}"`, error);
      }
    }

    if (clip) loaded.push({ token: item.token, clip });
  }

  if (generation !== quickMagicPlaybackGeneration) return true;

  if (!loaded.length) {
    const firstToken = requested[0]?.token ?? '—';
    nowPlayingText.textContent = t('avatar.now_playing.skip', { token: firstToken });
    replayButton.disabled = true;
    stopButton.disabled = true;
    return true;
  }

  if (quickMagicAction) {
    quickMagicAction.stop();
    quickMagicAction = null;
  }

  quickMagicQueue = loaded;
  quickMagicQueueIndex = 0;
  quickMagicLastSequence = requested.map((item) => ({ ...item }));
  quickMagicMixer.update(0);
  setQuickMagicIdlePose();
  startQuickMagicQueueEntry(generation);
  return true;
}

function prepareAvatar(gltf) {
  avatar = gltf.scene;
  blinkBindings.length = 0;
  avatar.traverse((object) => {
    if (object.isMesh || object.isSkinnedMesh) {
      object.castShadow = true;
      object.frustumCulled = false;
      collectBlinkTargets(object);
      const sourceMaterials = Array.isArray(object.material) ? object.material : [object.material];
      const materials = sourceMaterials.map(makeQuickMagicMaterial);
      object.material = Array.isArray(object.material) ? materials : materials[0];
      materials.filter(Boolean).forEach((material) => {
        material.side = THREE.FrontSide;
        applyQuickMagicBlueFabric(object, material);
        material.needsUpdate = true;
      });
    }
  });
  scene.add(avatar);
  fitAvatar(avatar);

  quickMagicMixer = new THREE.AnimationMixer(avatar);

  if (gltf.animations.length) {
    // The cleaned GLB already contains the approved Stack animation.
    // Register it as one more clip in the same QuickMagic motion library.
    quickMagicClipCache.set('stack', gltf.animations[0]);
  }

  quickMagicMixer.update(0);
  setQuickMagicIdlePose();

  quickMagicMixer.addEventListener('finished', (event) => {
    if (!quickMagicAction || event.action !== quickMagicAction) return;

    quickMagicAction.stop();
    quickMagicAction = null;
    quickMagicMixer.update(0);
    setQuickMagicIdlePose();

    quickMagicQueueIndex += 1;
    if (quickMagicQueueIndex < quickMagicQueue.length) {
      startQuickMagicQueueEntry(quickMagicPlaybackGeneration);
      return;
    }

    nowPlayingText.textContent = t('avatar.now_playing.complete');
    replayButton.disabled = false;
    stopButton.disabled = true;
  });
}

function updateBlink(now) {
  if (!blinkBindings.length) return;
  if (blinkStartedAt === null && now >= nextBlinkAt) blinkStartedAt = now;
  if (blinkStartedAt === null) return;
  const elapsed = now - blinkStartedAt;
  const amount = elapsed < 0.085 ? elapsed / 0.085 : 1 - (elapsed - 0.085) / 0.115;
  const value = THREE.MathUtils.clamp(amount, 0, 1);
  blinkBindings.forEach(({ mesh, index }) => { mesh.morphTargetInfluences[index] = value; });
  if (elapsed >= 0.2) {
    blinkBindings.forEach(({ mesh, index }) => { mesh.morphTargetInfluences[index] = 0; });
    blinkStartedAt = null;
    nextBlinkAt = now + 2.4 + Math.random() * 3.8;
  }
}

async function loadAvatar() {
  setLoading(t('avatar.loading'));
  try {
    const gltf = await loader.loadAsync(AVATAR_URL);
    prepareAvatar(gltf);
    // Keep the existing JSON motion player as an optional fallback for
    // motions that are not part of the QuickMagic FBX library. A fallback
    // initialization failure must never mark the already-loaded avatar as
    // failed, so isolate it from the avatar load path.
    try {
      motionPlayer = new MotionQueuePlayer(avatar);
      motionPlayer.onTokenStart = (entry, status) => {
        nowPlayingText.textContent = status === 'ready'
          ? t('avatar.now_playing.playing', { token: entry.token })
          : t('avatar.now_playing.skip', { token: entry.token });
      };
      motionPlayer.onQueueComplete = () => {
        nowPlayingText.textContent = t('avatar.now_playing.complete');
        replayButton.disabled = false;
        stopButton.disabled = true;
      };
    } catch (fallbackError) {
      motionPlayer = null;
      console.warn('[SignBridge] JSON motion fallback unavailable; QuickMagic avatar remains ready.', fallbackError);
    }
    setAvatarReady();
  } catch (error) {
    setAvatarError(error);
  }
}

function resize() {
  const width = Math.max(stage.clientWidth, 1);
  const height = Math.max(stage.clientHeight, 1);
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

viewButtons.forEach((button) => button.addEventListener('click', () => setCameraView(button.dataset.view)));
new ResizeObserver(resize).observe(stage);
window.addEventListener('resize', resize);
renderer.setAnimationLoop(() => {
  const delta = Math.min(clock.getDelta(), 0.05);
  const now = clock.elapsedTime;
  controls.update();
  updateBlink(now);
  quickMagicMixer?.update(delta);
  motionPlayer?.update(now);
  renderer.render(scene, camera);
});
resize();
loadAvatar();

/* ---------------- step flow (entry choice -> one active panel) ---------------- */
function showFlowPanel(mode) {
  Object.entries(panels).forEach(([key, panel]) => { panel.hidden = key !== mode; });
  if (mode !== 'library') setLibraryMotionNoteVisible(false);
  if (mode !== 'camera') {
    cameraRecorder.closeCamera();
    cameraPreview.hidden = true;
    cameraPlayback.hidden = true;
    setCameraUiState('idle');
  }
  if (mode === 'library') motionLibraryBrowser.ensureLoaded();
}

function enterFlow(mode) {
  resetPreviousInteractionState();

  runScreenTransition(() => {
    document.body.dataset.screen = mode;
    flowEntry.hidden = true;
    qaFlow.hidden = false;
    if (flowBackButton) flowBackButton.hidden = false;
    showFlowPanel(mode);

    requestAnimationFrame(() => {
      const headerHeight = siteHeader?.getBoundingClientRect().height ?? 0;
      const appShell = document.querySelector('#experience');
      if (!appShell) return;

      const targetTop = Math.max(
        window.scrollY + appShell.getBoundingClientRect().top - headerHeight - 8,
        0,
      );

      window.scrollTo({ top: targetTop, behavior: 'smooth' });
    });
  });
}

function exitFlow() {
  runScreenTransition(() => {
    document.body.dataset.screen = 'home';
    if (flowBackButton) flowBackButton.hidden = true;
    cameraRecorder.closeCamera();
    cameraPreview.hidden = true;
    cameraPlayback.hidden = true;
    setCameraUiState('idle');
    Object.values(panels).forEach((panel) => { panel.hidden = true; });
    resultsPanel.hidden = true;
    processingStatus.hidden = true;
    avatarCaptionText.hidden = true;
    qaFlow.hidden = true;
    flowEntry.hidden = false;
    flowEntry.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
}

flowChoiceButtons.forEach((button) => button.addEventListener('click', () => enterFlow(button.dataset.choice)));
flowBackButton?.addEventListener('click', exitFlow);
askAgainButton?.addEventListener('click', exitFlow);

// The manifest listing (~140 KB for 1,347 entries) is only fetched the
// first time the user opens this tab (MotionLibraryBrowser.ensureLoaded is
// a no-op after the first successful load); playing a specific motion only
// ever fetches that one motion's JSON, via the same MotionQueuePlayer used
// by Path A/B.
function setLibraryMotionNoteVisible(visible) {
  if (!libraryMotionNote) return;

  if (!visible) {
    libraryMotionNote.hidden = true;
    libraryMotionNote.textContent = '';
    return;
  }

  libraryMotionNote.textContent = getLanguage() === 'ar'
    ? 'هذه الحركة غير مدعومة حاليًا، ولكن قد يتم دعمها مستقبلًا.'
    : 'This motion is not currently supported, but it may be supported in the future.';
  libraryMotionNote.hidden = false;
}

function stopAvatarForUnsupportedLibraryMotion() {
  quickMagicPlaybackGeneration += 1;
  quickMagicQueue = [];
  quickMagicQueueIndex = 0;
  quickMagicLastSequence = [];

  if (quickMagicAction) {
    quickMagicAction.stop();
    quickMagicAction = null;
  }

  quickMagicMixer?.update(0);
  setQuickMagicIdlePose();
  motionPlayer?.stop();

  nowPlayingText.textContent = t('avatar.now_playing.none');
  replayButton.disabled = true;
  stopButton.disabled = true;
}

async function handleLibraryMotionSelection(entry) {
  if (!entry?.token) return;

  // The visible avatar is intentionally the fixed SignBridge QuickMagic
  // character. Only Stack + the 17 approved QuickMagic clips are presented
  // as supported avatar motions in the public library view.
  if (hasQuickMagicMotion(entry.token)) {
    setLibraryMotionNoteVisible(false);

    await playQuickMagicSequence([{
      token: entry.token,
      status: 'ready',
      motion_url: entry.motion_url ?? null,
    }]);
  } else {
    stopAvatarForUnsupportedLibraryMotion();
    setLibraryMotionNoteVisible(true);
  }

  // On narrow screens the library and avatar stack vertically, so move the
  // avatar into view after the user chooses a motion. On desktop the avatar
  // stays visible beside the library.
  if (window.matchMedia('(max-width: 850px)').matches) {
    experienceStage?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

onLanguageChange(() => {
  if (libraryMotionNote && !libraryMotionNote.hidden) {
    setLibraryMotionNoteVisible(true);
  }
});

const motionLibraryBrowser = new MotionLibraryBrowser(
  document.querySelector('#motion-library-root'),
  { onPlay: handleLibraryMotionSelection },
);

/* ---------------- backend health ---------------- */
let lastHealthOk = null;
async function refreshHealth() {
  try {
    const health = await fetchHealth();
    lastHealthOk = health;
    backendDot.classList.add('is-ready');
    backendDot.classList.remove('is-error');
    backendText.textContent = health.recognition_model_loaded ? t('backend.connected_model') : t('backend.connected');
  } catch (error) {
    lastHealthOk = null;
    backendDot.classList.add('is-error');
    backendDot.classList.remove('is-ready');
    backendText.textContent = t('backend.error');
  }
}
refreshHealth();
onLanguageChange(() => {
  if (lastHealthOk) {
    backendText.textContent = lastHealthOk.recognition_model_loaded ? t('backend.connected_model') : t('backend.connected');
  } else if (backendDot.classList.contains('is-error')) {
    backendText.textContent = t('backend.error');
  } else {
    backendText.textContent = t('backend.connecting');
  }
});

/* ---------------- results rendering ---------------- */
function hideAllResultBlocks() {
  [resultRecognition, resultAnswer, resultUnavailable, resultSources, resultTokens, resultMotions].forEach((el) => {
    el.hidden = true;
  });
  detectedQuestionText.hidden = true;
  avatarCaptionText.hidden = true;
}

function showResultsPanel() {
  runScreenTransition(() => {
    hideAllResultBlocks();
    Object.values(panels).forEach((panel) => { panel.hidden = true; });
    document.body.dataset.screen = 'results';
    resultsPanel.hidden = false;
    resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
}

function renderRecognition(recognition) {
  const statusText = document.querySelector('#recognition-status-text');
  const rejected = recognition.resolved_mode === 'rejected' || recognition.accepted === false;

  if (rejected) {
    statusText.textContent = t('results.recognition.retry_message');
  } else if (recognition.predicted_label) {
    // Confidence is only shown when it reflects one candidate's own
    // top-1 score (isolated it/karsl); a continuous multi-sign sequence's
    // averaged confidence is labeled differently so it isn't mistaken for
    // the same kind of number.
    const isIsolated = recognition.resolved_mode === 'it' || recognition.resolved_mode === 'karsl';
    const hasMeaningfulConfidence = recognition.confidence != null && recognition.confidence > 0;
    if (hasMeaningfulConfidence) {
      const percent = (recognition.confidence * 100).toFixed(0);
      statusText.textContent = isIsolated
        ? t('results.recognition.recognized_confidence', { label: recognition.predicted_label, percent })
        : t('results.recognition.recognized_confidence_avg', { label: recognition.predicted_label, percent });
    } else {
      statusText.textContent = t('results.recognition.recognized', { label: recognition.predicted_label });
    }
  } else {
    statusText.textContent = t('results.recognition.none');
  }

  recognitionFields.innerHTML = '';
  const na = t('results.recognition.na');
  const rows = [
    [t('results.recognition.requested_mode'), recognition.requested_mode],
    [t('results.recognition.resolved_mode'), recognition.resolved_mode],
    [t('results.recognition.accepted'), recognition.accepted === false ? t('results.recognition.no') : t('results.recognition.yes')],
    [t('results.recognition.reason'), recognition.routing_reason || na],
    [t('results.recognition.predicted_label'), recognition.predicted_label || na],
    [t('results.recognition.confidence'), recognition.confidence != null ? `${(recognition.confidence * 100).toFixed(1)}%` : na],
    [t('results.recognition.frame_count'), recognition.frame_count],
    [t('results.recognition.active_segment'), `${recognition.active_start_frame} - ${recognition.active_end_frame} (${recognition.active_source})`],
  ];
  for (const [label, value] of rows) {
    const dt = document.createElement('dt');
    dt.textContent = label;
    const dd = document.createElement('dd');
    dd.textContent = String(value);
    recognitionFields.append(dt, dd);
  }
  if (recognition.top_k?.length) {
    const dt = document.createElement('dt');
    dt.textContent = t('results.recognition.top_k');
    const dd = document.createElement('dd');
    dd.textContent = recognition.top_k.map((p) => `${p.label} (${(p.confidence * 100).toFixed(1)}%)`).join('، ');
    recognitionFields.append(dt, dd);
  }
  if (recognition.candidate_scores && Object.keys(recognition.candidate_scores).length) {
    const dt = document.createElement('dt');
    dt.textContent = t('results.recognition.candidate_scores');
    const dd = document.createElement('dd');
    dd.textContent = JSON.stringify(recognition.candidate_scores);
    recognitionFields.append(dt, dd);
  }
  resultRecognition.hidden = false;
}

// Path A: the motion for the sign actually recognized (dataset-exact
// resolution), shown and queued separately from Path B (the RAG/semantic
// answer's motion sequence, rendered by renderAnswer/playMotionSequence).
function renderRecognizedMotion(recognizedMotionSequence) {
  if (!recognizedMotionSequence?.length) return;
  const ready = recognizedMotionSequence.filter(
    (m) => m.status === 'ready' || hasQuickMagicMotion(m.token),
  );
  const missing = recognizedMotionSequence.filter(
    (m) => m.status !== 'ready' && !hasQuickMagicMotion(m.token),
  );
  const dt = document.createElement('dt');
  dt.textContent = t('results.recognition.motion_label');
  const dd = document.createElement('dd');
  dd.textContent = ready.length
    ? t('results.recognition.motion_available', { tokens: ready.map((m) => m.token).join('، ') })
    : t('results.recognition.motion_unavailable', { tokens: missing.map((m) => m.token).join('، ') || t('results.recognition.na') });
  recognitionFields.append(dt, dd);
}

function renderAnswer(answer, { autoplay = true } = {}) {
  if (!answer.educational_answer_available) {
    unavailableNoteText.textContent = answer.unavailable_reason || t('results.answer.unavailable_default');
    resultUnavailable.hidden = false;
    return;
  }

  educationalAnswerText.textContent = answer.educational_answer || '';
  simplifiedAnswerText.textContent = answer.simplified_text || '';
  if (answer.simplified_text) {
    avatarCaptionText.textContent = answer.simplified_text;
    avatarCaptionText.hidden = false;
  }
  routedSubjectText.textContent = answer.routed_subject?.length
    ? t('results.answer.topic', { topics: answer.routed_subject.join('، ') })
    : '';
  resultAnswer.hidden = false;

  if (answer.sources?.length) {
    sourcesList.innerHTML = '';
    for (const source of answer.sources) {
      const li = document.createElement('li');
      li.textContent = t('results.sources.item', {
        domain: source.domain || '',
        title: source.slide_title || '',
        score: Number(source.retrieval_score ?? 0).toFixed(2),
      });
      sourcesList.append(li);
    }
    resultSources.hidden = false;
  }

  if (answer.sign_tokens?.length) {
    signTokensList.innerHTML = '';
    for (const token of answer.sign_tokens) {
      const chip = document.createElement('span');
      chip.className = 'token-chip';
      chip.textContent = token;
      signTokensList.append(chip);
    }
    resultTokens.hidden = false;
  }

  const coveragePercent = Math.round((answer.physical_motion_coverage || 0) * 100);
  motionCoverageText.textContent = t('results.motions.coverage', { percent: coveragePercent });
  missingMotionsList.innerHTML = '';
  if (answer.missing_motion_tokens?.length) {
    const p = document.createElement('p');
    p.textContent = t('results.motions.missing', { tokens: answer.missing_motion_tokens.join('، ') });
    missingMotionsList.append(p);
  }
  resultMotions.hidden = false;

  if (autoplay) playMotionSequence(answer.resolved_motion_sequence || []);
}

async function playMotionSequence(sequence) {
  if (!sequence.length) return;

  // Prefer one of the prepared QuickMagic FBX clips whenever the token is
  // in the approved QuickMagic library. These clips animate the existing
  // SignBridge_Boy_Clean skeleton only; their FBX meshes are never shown.
  const handledByQuickMagic = await playQuickMagicSequence(sequence);
  if (handledByQuickMagic) return;

  if (!motionPlayer) return;
  const entries = sequence.map((item) => ({
    token: item.token,
    status: item.status,
    motion_url: item.motion_url ? motionUrl(item.motion_url) : null,
  }));
  await motionPlayer.loadSequence(entries);
  motionPlayer.play(clock.elapsedTime);
  replayButton.disabled = false;
  stopButton.disabled = false;
}

replayButton.addEventListener('click', async () => {
  if (quickMagicLastSequence.some((item) => hasQuickMagicMotion(item.token))) {
    await playQuickMagicSequence(quickMagicLastSequence);
  } else {
    motionPlayer?.replay(clock.elapsedTime);
  }
  if (simplifiedAnswerText.textContent) {
    avatarCaptionText.textContent = simplifiedAnswerText.textContent;
    avatarCaptionText.hidden = false;
  }
  stopButton.disabled = false;
});

stopButton.addEventListener('click', () => {
  quickMagicPlaybackGeneration += 1;
  quickMagicQueue = [];
  quickMagicQueueIndex = 0;

  if (quickMagicAction) {
    quickMagicAction.stop();
    quickMagicAction = null;
    quickMagicMixer?.update(0);
    setQuickMagicIdlePose();
  } else {
    motionPlayer?.stop();
  }

  nowPlayingText.textContent = t('avatar.now_playing.stopped');
  avatarCaptionText.hidden = true;
  stopButton.disabled = true;
  replayButton.disabled = false;
});

function resetPreviousInteractionState() {
  // Stop any currently playing avatar motion and return to a clean idle state.
  quickMagicPlaybackGeneration += 1;
  quickMagicQueue = [];
  quickMagicQueueIndex = 0;
  quickMagicLastSequence = [];

  if (quickMagicAction) {
    quickMagicAction.stop();
    quickMagicAction = null;
  }

  quickMagicMixer?.update(0);
  setQuickMagicIdlePose();
  motionPlayer?.stop();

  // Clear any previous library-only note.
  setLibraryMotionNoteVisible(false);

  // Clear previous result content so a new ask/upload/camera run starts clean.
  hideAllResultBlocks();
  resultsPanel.hidden = true;
  processingStatus.hidden = true;

  detectedQuestionText.textContent = '';
  recognitionFields.innerHTML = '';
  educationalAnswerText.textContent = '';
  simplifiedAnswerText.textContent = '';
  routedSubjectText.textContent = '';
  unavailableNoteText.textContent = '';
  sourcesList.innerHTML = '';
  signTokensList.innerHTML = '';
  motionCoverageText.textContent = '';
  missingMotionsList.innerHTML = '';

  avatarCaptionText.textContent = '';
  avatarCaptionText.hidden = true;

  // IMPORTANT: clear the actual user inputs too, not only the result.
  // Text question
  questionInput.value = '';
  syncQuestionCounter();
  questionError.hidden = true;
  questionInput.removeAttribute('aria-invalid');

  // Uploaded video + local preview
  videoFileInput.value = '';
  resetUploadPreview();

  // Recorded camera video + playback preview
  try {
    cameraRecorder.discardRecording();
  } catch {
    // Safe no-op if there is no active/recorded camera blob.
  }
  cameraRecordedBlob = null;
  stopRecordingTimer();

  if (cameraPlayback.src) {
    try {
      URL.revokeObjectURL(cameraPlayback.src);
    } catch {
      // Ignore non-object URLs.
    }
  }

  cameraPlayback.pause();
  cameraPlayback.removeAttribute('src');
  cameraPlayback.load();
  cameraPlayback.hidden = true;

  cameraPreview.hidden = true;
  cameraRecorder.closeCamera();
  setCameraUiState('idle');
  cameraStatus.textContent = '';

  nowPlayingText.textContent = t('avatar.now_playing.none');
  replayButton.disabled = true;
  stopButton.disabled = true;
}

function showProcessing(message) {
  processingStatus.hidden = false;
  processingStatus.textContent = message;
}
function hideProcessing() {
  processingStatus.hidden = true;
}

function currentRecognitionMode() {
  return document.querySelector('input[name="recognition-mode"]:checked')?.value || 'auto';
}

/* ---------------- ask flow ---------------- */
const questionError = document.querySelector('#question-error');
const questionCounter = document.querySelector('.input-counter');
function syncQuestionCounter() {
  if (questionCounter) questionCounter.textContent = String(questionInput.value.length) + ' / 500';
}
questionInput.addEventListener('input', () => {
  syncQuestionCounter();
  if (questionInput.value.trim()) {
    questionError.hidden = true;
    questionInput.removeAttribute('aria-invalid');
  }
});

// Enter submits the question; Shift+Enter keeps the normal newline behavior.
questionInput.addEventListener('keydown', (event) => {
  if (event.key !== 'Enter' || event.shiftKey || event.isComposing) return;

  event.preventDefault();

  if (typeof askForm.requestSubmit === 'function') {
    askForm.requestSubmit();
  } else {
    askForm.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
  }
});

syncQuestionCounter();

askForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) {
    questionError.hidden = false;
    questionInput.setAttribute('aria-invalid', 'true');
    questionInput.focus();
    return;
  }
  questionError.hidden = true;
  questionInput.removeAttribute('aria-invalid');

  resetPreviousInteractionState();
  showResultsPanel();
  detectedQuestionText.textContent = t('results.your_question', { question });
  detectedQuestionText.hidden = false;
  showProcessing(t('ask.loading'));
  try {
    const answer = await askQuestion(question);
    renderAnswer(answer);
  } catch (error) {
    resultUnavailable.hidden = false;
    unavailableNoteText.textContent = error.message || t('ask.generic_error');
  } finally {
    hideProcessing();
  }
});

/* ---------------- upload flow ---------------- */
// Review-before-send: selecting a file only previews it locally (an object
// URL, never uploaded); the existing recognizeVideo() call below still only
// fires on explicit form submit — no change to what gets sent or when.
let uploadObjectUrl = null;
function resetUploadPreview() {
  if (uploadObjectUrl) {
    URL.revokeObjectURL(uploadObjectUrl);
    uploadObjectUrl = null;
  }
  uploadPreview.hidden = true;
  uploadPreview.removeAttribute('src');
  uploadReviewActions.hidden = true;
}

videoFileInput.addEventListener('change', () => {
  const file = videoFileInput.files?.[0];
  if (uploadObjectUrl) {
    URL.revokeObjectURL(uploadObjectUrl);
    uploadObjectUrl = null;
  }
  if (!file) {
    resetUploadPreview();
    return;
  }
  uploadObjectUrl = URL.createObjectURL(file);
  uploadPreview.src = uploadObjectUrl;
  uploadPreview.hidden = false;
  uploadReviewActions.hidden = false;
});

uploadChooseAnotherButton.addEventListener('click', () => {
  videoFileInput.value = '';
  resetUploadPreview();
});

uploadForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const file = videoFileInput.files?.[0];
  if (!file) return;
  await runRecognition(file, file.name);
});

async function runRecognition(blob, filename, { onStatusChange } = {}) {
  resetPreviousInteractionState();
  showResultsPanel();
  showProcessing(t('camera.status.uploading'));
  onStatusChange?.('uploading');
  try {
    const result = await recognizeVideo(blob, filename, currentRecognitionMode());
    renderRecognition(result.recognition);
    renderRecognizedMotion(result.recognized_motion_sequence);

    // Path A (the recognized sign's own motion) and Path B (the RAG/
    // semantic answer's motion sequence) are resolved independently and
    // must never both try to drive the single avatar player at once.
    // Prefer Path A when it actually has something ready to show — it is
    // the more literal match for a recognized-video request — and fall
    // back to Path B only when Path A has nothing playable.
    const readyRecognized = (result.recognized_motion_sequence || []).filter(
      (m) => m.status === 'ready' || hasQuickMagicMotion(m.token),
    );
    if (readyRecognized.length) {
      renderAnswer(result.answer, { autoplay: false });
      await playMotionSequence(
        readyRecognized.map((m) => ({
          token: m.token,
          status: hasQuickMagicMotion(m.token) ? 'ready' : m.status,
          motion_url: m.motion_url,
        })),
      );
    } else {
      renderAnswer(result.answer, { autoplay: true });
    }
    onStatusChange?.('completed');
  } catch (error) {
    resultUnavailable.hidden = false;
    unavailableNoteText.textContent = error.message || t('camera.status.generic_error');
    onStatusChange?.('failed');
  } finally {
    hideProcessing();
  }
}

/* ---------------- camera flow (record-then-analyze) ---------------- */
const cameraRecorder = new CameraRecorder();
let cameraRecordedBlob = null;
let recordingIntervalId = null;
let recordingStartedAt = 0;

// Mirroring is a display-only CSS transform applied to the preview
// elements; it never touches the MediaStream, the MediaRecorder output, or
// the Blob eventually sent to the recognition backend (see
// camera-recorder.js's orientation contract).
function applyPreviewMirroring() {
  const mirrored = cameraRecorder.shouldMirrorPreview;
  cameraPreview.classList.toggle('is-mirrored', mirrored);
  cameraPlayback.classList.toggle('is-mirrored', mirrored);
}

// Progressive button visibility: only the button relevant to the current
// step is shown, instead of showing every camera control at once.
// 'idle' -> only Start Camera. 'ready' -> Start Recording. 'recording' ->
// Stop Recording (+ live timer). 'recorded' -> Retake + Analyze. Close
// stays available (but visually secondary) in every state but 'idle'.
function setCameraUiState(state) {
  cameraStartButton.hidden = state !== 'idle';
  cameraRecordButton.hidden = state !== 'ready';
  cameraStopButton.hidden = state !== 'recording';
  cameraRetakeButton.hidden = state !== 'recorded';
  cameraAnalyzeButton.hidden = state !== 'recorded';
  cameraCloseButton.hidden = state === 'idle';
  cameraRecordButton.disabled = false;
  cameraStopButton.disabled = false;
  cameraAnalyzeButton.disabled = false;
  cameraRetakeButton.disabled = false;
}
setCameraUiState('idle');

function startRecordingTimer() {
  recordingStartedAt = Date.now();
  recordingTimer.textContent = '00:00';
  recordingIndicator.hidden = false;
  recordingIntervalId = window.setInterval(() => {
    const elapsedSeconds = Math.floor((Date.now() - recordingStartedAt) / 1000);
    const minutes = String(Math.floor(elapsedSeconds / 60)).padStart(2, '0');
    const seconds = String(elapsedSeconds % 60).padStart(2, '0');
    recordingTimer.textContent = `${minutes}:${seconds}`;
  }, 250);
}
function stopRecordingTimer() {
  if (recordingIntervalId !== null) window.clearInterval(recordingIntervalId);
  recordingIntervalId = null;
  recordingIndicator.hidden = true;
}

cameraStartButton.addEventListener('click', async () => {
  try {
    cameraStatus.textContent = t('camera.status.requesting');
    await cameraRecorder.start(cameraPreview, { facingMode: 'user' });
    applyPreviewMirroring();
    cameraPreview.hidden = false;
    cameraPlayback.hidden = true;
    setCameraUiState('ready');
    cameraStatus.textContent = t('camera.status.ready');
  } catch (error) {
    cameraStatus.textContent = error.message || t('camera.error.generic');
  }
});

cameraRecordButton.addEventListener('click', () => {
  cameraRecorder.beginRecording();
  setCameraUiState('recording');
  startRecordingTimer();
  cameraStatus.textContent = t('camera.status.recording');
});

cameraStopButton.addEventListener('click', async () => {
  cameraRecordedBlob = await cameraRecorder.stopRecording();
  stopRecordingTimer();
  cameraPreview.hidden = true;
  cameraPlayback.hidden = false;
  cameraPlayback.src = URL.createObjectURL(cameraRecordedBlob);
  setCameraUiState('recorded');
  cameraStatus.textContent = t('camera.status.recorded');
});

cameraRetakeButton.addEventListener('click', async () => {
  cameraRecorder.discardRecording();
  cameraRecordedBlob = null;
  cameraPlayback.hidden = true;
  cameraPreview.hidden = false;
  setCameraUiState('ready');
  cameraStatus.textContent = t('camera.status.retake_ready');
});

cameraAnalyzeButton.addEventListener('click', async () => {
  if (!cameraRecordedBlob) return;
  cameraAnalyzeButton.disabled = true;
  cameraRetakeButton.disabled = true;
  const statusKeys = {
    uploading: 'camera.status.uploading',
    completed: 'camera.status.completed',
    failed: 'camera.status.failed',
  };
  await runRecognition(cameraRecordedBlob, 'camera-recording.webm', {
    onStatusChange: (state) => {
      if (statusKeys[state]) cameraStatus.textContent = t(statusKeys[state]);
    },
  });
  cameraAnalyzeButton.disabled = false;
  cameraRetakeButton.disabled = false;
});

cameraCloseButton.addEventListener('click', () => {
  cameraRecorder.closeCamera();
  stopRecordingTimer();
  cameraPreview.hidden = true;
  cameraPlayback.hidden = true;
  setCameraUiState('idle');
  cameraStatus.textContent = t('camera.status.closed');
});

window.addEventListener('beforeunload', () => cameraRecorder.closeCamera());
