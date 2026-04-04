"use client";

import { useEffect, useMemo, useRef, useState } from "react";

export interface Personality {
  metabolism: number;
  digestibility: number;
  curiosityLevel?: number;
  rewardKeywords: string[];
  avoidanceCount: number;
  turnCount: number;
  driftDetected?: boolean;
}

interface Props {
  growthTrend: "growing" | "stable" | "degrading" | "none";
  recentScore: number;
  principleHealth: number;
  policyCount: number;
  level: number;
  dataVersion: number;
  personality?: Personality;
}

type Mood = "thriving" | "content" | "uneasy" | "stressed" | "sleeping";
type Sprite = number[][];
type Palette = Record<number, string>;
type GenerationTraits = {
  metabolism: number;
  digestibility: number;
};

type CreatureTraits = {
  bodyWidth: number;
  bodyHeight: number;
  headToBodyRatio: number;
  roundness: number;
  eyeStyle: number;
  eyeSize: number;
  eyeSpacing: number;
  mouthStyle: number;
  hasTopFeature: boolean;
  topFeatureHeight: number;
  topFeatureKind: number;
  hasTail: boolean;
  tailLength: number;
  tailSide: -1 | 1;
  hasHat: boolean;
  hatStyle: number;
  hasCheeks: boolean;
  hasPattern: boolean;
  patternStyle: number;
  ornamentBias: number;
  detailRolls: number[];
};

const GRID_SIZE = 16;
const DEFAULT_PERSONALITY_SEED = 42;
const DEFAULT_TRAITS: GenerationTraits = {
  metabolism: 0.5,
  digestibility: 0.5,
};

function deriveMood(props: Props): Mood {
  const { growthTrend, recentScore, principleHealth, policyCount } = props;
  if (growthTrend === "none" && policyCount === 0) return "sleeping";
  if (growthTrend === "degrading" || recentScore < 0.3) return "stressed";
  if (recentScore < 0.5 || principleHealth < 0.5) return "uneasy";
  if (growthTrend === "growing" && recentScore > 0.7 && principleHealth > 0.8) return "thriving";
  return "content";
}

function deriveStage(props: Props): number {
  if (props.level <= 1 && props.policyCount === 0) return 0;
  if (props.policyCount === 0 && props.principleHealth < 0.3) return 1;
  if (props.policyCount === 0) return 2;
  if (props.principleHealth >= 0.8) return 4;
  return 3;
}

const MOOD_LABELS: Record<Mood, string> = {
  thriving: "ごきげん",
  content: "おだやか",
  uneasy: "そわそわ",
  stressed: "ぐったり",
  sleeping: "すやすや",
};

const STAGE_LABELS = ["たまご", "あかちゃん", "こども", "おとな", "けんじゃ"];

// Sprite path → species label mapping
const SPRITE_LABELS: Record<string, string> = {
  "/sprites/01-egg.png": "たまご",
  "/sprites/02-curious-kitten.png": "こねこ",
  "/sprites/03-curious-cat.png": "ねこ",
  "/sprites/04-curious-star-cat.png": "ほしねこ",
  "/sprites/05-curious-owl-sage.png": "ふくろう",
  "/sprites/06-aggressive-puppy.png": "こいぬ",
  "/sprites/07-aggressive-wolf.png": "おおかみ",
  "/sprites/08-aggressive-lion.png": "ライオン",
  "/sprites/09-aggressive-armored-dog.png": "よろいいぬ",
  "/sprites/10-conservative-bear-baby.png": "こぐま",
  "/sprites/11-conservative-bear-adult.png": "おおぐま",
  "/sprites/12-conservative-crystal-bear.png": "こおりぐま",
  "/sprites/13-conservative-turtle-sage.png": "かめせんにん",
  "/sprites/14-balanced-rabbit-baby.png": "こうさぎ",
  "/sprites/15-balanced-rabbit-adult.png": "うさぎ",
  "/sprites/16-balanced-fox-master.png": "きつね",
  "/sprites/17-balanced-golden-fox.png": "きんのきつね",
  "/sprites/18-drift-chameleon.png": "カメレオン",
  "/sprites/19-escalation-phoenix.png": "ほうおう",
  "/sprites/20-transcend-dragon.png": "りゅう",
};

export function mulberry32(seed: number) {
  return function () {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

export function personalitySeed(p: Personality): number {
  let h = Math.floor(p.metabolism * 1000) * 73856093;
  h ^= Math.floor(p.digestibility * 1000) * 19349663;
  h ^= p.avoidanceCount * 83492791;
  h ^= p.turnCount * 4256233;
  return h >>> 0;
}

// ── Name generation from seed ──
const NAME_PARTS_A = ["ポ","モ","チ","プ","ニ","コ","ミ","ル","ピ","フ","ム","ク","リ","ト","ヌ","キ","ノ","ヨ","マ","ハ"];
const NAME_PARTS_B = ["ん","ー","っ","り","る","ち","み","く","の","よ","ぷ","ぴ","に","も","む"];
const NAME_PARTS_C = ["たん","ちゃん","まる","ぴょん","っこ","ー","すけ","のすけ","べえ","ぽん"];

export function generateName(rng: () => number): string {
  const a = NAME_PARTS_A[Math.floor(rng() * NAME_PARTS_A.length)];
  const b = NAME_PARTS_B[Math.floor(rng() * NAME_PARTS_B.length)];
  const c = NAME_PARTS_C[Math.floor(rng() * NAME_PARTS_C.length)];
  return a + b + c;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function wrapHue(value: number): number {
  return ((value % 360) + 360) % 360;
}

function hsl(hue: number, saturation: number, lightness: number): string {
  return `hsl(${Math.round(wrapHue(hue))}, ${Math.round(clamp(saturation, 0, 100))}%, ${Math.round(clamp(lightness, 0, 100))}%)`;
}

function createGrid(): Sprite {
  return Array.from({ length: GRID_SIZE }, () => Array<number>(GRID_SIZE).fill(0));
}

function cloneSprite(sprite: Sprite): Sprite {
  return sprite.map((row) => [...row]);
}

function setPixel(sprite: Sprite, x: number, y: number, color: number): void {
  const px = Math.round(x);
  const py = Math.round(y);
  if (px < 0 || px >= GRID_SIZE || py < 0 || py >= GRID_SIZE) return;
  sprite[py][px] = color;
}

function drawBlock(sprite: Sprite, startX: number, startY: number, width: number, height: number, color: number): void {
  const x0 = Math.round(startX);
  const y0 = Math.round(startY);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      setPixel(sprite, x0 + x, y0 + y, color);
    }
  }
}

function fillSuperellipse(
  sprite: Sprite,
  centerX: number,
  centerY: number,
  radiusX: number,
  radiusY: number,
  color: number,
  power: number,
): void {
  for (let y = 0; y < GRID_SIZE; y++) {
    for (let x = 0; x < GRID_SIZE; x++) {
      const dx = Math.abs((x + 0.5 - centerX) / Math.max(radiusX, 0.001));
      const dy = Math.abs((y + 0.5 - centerY) / Math.max(radiusY, 0.001));
      if (Math.pow(dx, power) + Math.pow(dy, power) <= 1) {
        sprite[y][x] = color;
      }
    }
  }
}

function addOutline(sprite: Sprite): Sprite {
  const outlined = cloneSprite(sprite);

  for (let y = 0; y < GRID_SIZE; y++) {
    for (let x = 0; x < GRID_SIZE; x++) {
      if (sprite[y][x] !== 0) continue;

      let shouldOutline = false;
      for (let dy = -1; dy <= 1 && !shouldOutline; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || nx >= GRID_SIZE || ny < 0 || ny >= GRID_SIZE) continue;
          if (sprite[ny][nx] !== 0) {
            shouldOutline = true;
            break;
          }
        }
      }

      if (shouldOutline) {
        outlined[y][x] = 1;
      }
    }
  }

  return outlined;
}

function createCreatureTraits(rng: () => number, digestibility: number, metabolism: number): CreatureTraits {
  return {
    bodyWidth: 4 + Math.floor(rng() * 4),
    bodyHeight: 5 + Math.floor(rng() * 3),
    headToBodyRatio: 0.3 + rng() * 0.4,
    roundness: clamp(0.2 + digestibility * 0.7, 0.2, 0.9),
    eyeStyle: Math.floor(rng() * 4),
    eyeSize: 1 + Math.floor(rng() * 2),
    eyeSpacing: 2 + Math.floor(rng() * 3),
    mouthStyle: Math.floor(rng() * 3),
    hasTopFeature: rng() > 0.5,
    topFeatureHeight: 1 + Math.floor(rng() * 3),
    topFeatureKind: Math.floor(rng() * 2),
    hasTail: rng() > 0.6,
    tailLength: 1 + Math.floor(rng() * 3),
    tailSide: rng() > 0.5 ? 1 : -1,
    hasHat: rng() > 0.7,
    hatStyle: Math.floor(rng() * 3),
    hasCheeks: rng() > 0.5,
    hasPattern: rng() > 0.6,
    patternStyle: Math.floor(rng() * 3),
    ornamentBias: 0.2 + metabolism * 0.5,
    detailRolls: Array.from({ length: 12 }, () => rng()),
  };
}

function drawEye(sprite: Sprite, centerX: number, centerY: number, style: number, size: number): void {
  const eyeX = Math.round(centerX);
  const eyeY = Math.round(centerY);

  switch (style) {
    case 0:
      drawBlock(sprite, eyeX - Math.floor(size / 2), eyeY - Math.floor(size / 2), size, size, 4);
      return;
    case 1:
      drawBlock(sprite, eyeX - Math.floor(size / 2), eyeY - 1, size + 1, size, 4);
      return;
    case 2:
      drawBlock(sprite, eyeX - 1, eyeY, size + 1, 1, 4);
      return;
    default:
      setPixel(sprite, eyeX, eyeY, 4);
      setPixel(sprite, eyeX - 1, eyeY, 4);
      setPixel(sprite, eyeX + 1, eyeY, 4);
      setPixel(sprite, eyeX, eyeY - 1, 4);
      setPixel(sprite, eyeX, eyeY + 1, 4);
      if (size === 2) {
        setPixel(sprite, eyeX - 1, eyeY - 1, 4);
        setPixel(sprite, eyeX + 1, eyeY + 1, 4);
      }
    }
}

function drawMouth(sprite: Sprite, centerX: number, mouthY: number, style: number): void {
  const x = Math.round(centerX);
  const y = Math.round(mouthY);

  if (style === 0) {
    setPixel(sprite, x - 1, y, 1);
    setPixel(sprite, x, y + 1, 1);
    setPixel(sprite, x + 1, y, 1);
    return;
  }

  if (style === 1) {
    setPixel(sprite, x - 1, y, 1);
    setPixel(sprite, x, y, 1);
    setPixel(sprite, x + 1, y, 1);
    return;
  }

  setPixel(sprite, x - 1, y, 5);
  setPixel(sprite, x, y, 5);
  setPixel(sprite, x, y + 1, 5);
}

function drawTopFeature(sprite: Sprite, centerX: number, topY: number, traits: CreatureTraits): void {
  const leftX = centerX - 2;
  const rightX = centerX + 2;
  const color = traits.topFeatureKind === 0 ? 2 : 5;

  for (let step = 0; step < traits.topFeatureHeight; step++) {
    const lift = topY - step;
    if (traits.topFeatureKind === 0) {
      setPixel(sprite, leftX - (step === traits.topFeatureHeight - 1 ? 1 : 0), lift, color);
      setPixel(sprite, rightX + (step === traits.topFeatureHeight - 1 ? 1 : 0), lift, color);
    } else {
      setPixel(sprite, leftX, lift, color);
      setPixel(sprite, rightX, lift, color);
      if (step === traits.topFeatureHeight - 1) {
        setPixel(sprite, leftX, lift - 1, 8);
        setPixel(sprite, rightX, lift - 1, 8);
      }
    }
  }
}

function drawHat(sprite: Sprite, centerX: number, topY: number, traits: CreatureTraits): void {
  if (traits.hatStyle === 0) {
    drawBlock(sprite, centerX - 2, topY, 5, 1, 5);
    setPixel(sprite, centerX - 1, topY - 1, 5);
    setPixel(sprite, centerX + 1, topY - 1, 5);
    setPixel(sprite, centerX, topY - 2, 8);
    return;
  }

  if (traits.hatStyle === 1) {
    drawBlock(sprite, centerX - 2, topY, 5, 1, 5);
    setPixel(sprite, centerX - 2, topY - 1, 6);
    setPixel(sprite, centerX - 3, topY - 1, 6);
    setPixel(sprite, centerX + 2, topY - 1, 6);
    setPixel(sprite, centerX + 3, topY - 1, 6);
    return;
  }

  drawBlock(sprite, centerX, topY - 2, 1, 3, 5);
  setPixel(sprite, centerX, topY - 3, 8);
  setPixel(sprite, centerX - 1, topY, 5);
  setPixel(sprite, centerX + 1, topY, 5);
}

function drawPattern(sprite: Sprite, centerX: number, bellyY: number, traits: CreatureTraits): void {
  if (!traits.hasPattern) return;

  if (traits.patternStyle === 0) {
    const spot1X = centerX - 1 + Math.round(traits.detailRolls[2] * 2);
    const spot1Y = bellyY - 1;
    const spot2X = centerX + Math.round(traits.detailRolls[3] * 2);
    const spot2Y = bellyY;
    const spot3X = centerX - 2 + Math.round(traits.detailRolls[4] * 3);
    const spot3Y = bellyY + 1;
    setPixel(sprite, spot1X, spot1Y, 8);
    setPixel(sprite, spot2X, spot2Y, 8);
    setPixel(sprite, spot3X, spot3Y, 8);
    return;
  }

  if (traits.patternStyle === 1) {
    for (let dx = -2; dx <= 2; dx++) {
      setPixel(sprite, centerX + dx, bellyY - 1, 8);
      if (dx !== -2 && dx !== 2) {
        setPixel(sprite, centerX + dx, bellyY + 1, 8);
      }
    }
    return;
  }

  setPixel(sprite, centerX - 1, bellyY, 8);
  setPixel(sprite, centerX, bellyY, 8);
  setPixel(sprite, centerX, bellyY + 1, 8);
}

function drawWingDecoration(sprite: Sprite, centerX: number, bodyY: number, bodyRadiusX: number, traits: CreatureTraits): void {
  const wingOffset = Math.round(bodyRadiusX + 2);
  const featherY = Math.round(bodyY - 1 + traits.detailRolls[5] * 2);

  setPixel(sprite, centerX - wingOffset, featherY, 8);
  setPixel(sprite, centerX - wingOffset - 1, featherY + 1, 8);
  setPixel(sprite, centerX + wingOffset, featherY, 8);
  setPixel(sprite, centerX + wingOffset + 1, featherY + 1, 8);
}

function drawShellPattern(sprite: Sprite, centerX: number, shellY: number, traits: CreatureTraits): void {
  if (!traits.hasPattern) {
    setPixel(sprite, centerX, shellY - 2, 8);
    setPixel(sprite, centerX - 1, shellY - 1, 8);
    return;
  }

  if (traits.patternStyle === 0) {
    setPixel(sprite, centerX - 2, shellY - 1, 8);
    setPixel(sprite, centerX + 1, shellY, 8);
    setPixel(sprite, centerX - 1, shellY + 2, 8);
    return;
  }

  if (traits.patternStyle === 1) {
    for (let offset = -2; offset <= 2; offset++) {
      setPixel(sprite, centerX + offset, shellY + offset * 0.4, 8);
    }
    return;
  }

  setPixel(sprite, centerX - 1, shellY - 1, 5);
  setPixel(sprite, centerX, shellY, 5);
  setPixel(sprite, centerX + 1, shellY - 1, 5);
}

function generateEggSprite(traits: CreatureTraits): Sprite {
  const sprite = createGrid();
  const centerX = 7.5;
  const centerY = 8.5;
  const shellRadiusX = 2.8 + traits.bodyWidth * 0.48;
  const shellRadiusY = 3.4 + traits.bodyHeight * 0.42;
  const shellPower = 2.7 - traits.roundness * 0.5;

  fillSuperellipse(sprite, centerX, centerY, shellRadiusX, shellRadiusY, 2, shellPower);
  fillSuperellipse(sprite, centerX, centerY + 1.4, shellRadiusX * 0.55, shellRadiusY * 0.35, 3, 2.2);
  fillSuperellipse(sprite, centerX - shellRadiusX * 0.25, centerY - shellRadiusY * 0.45, shellRadiusX * 0.4, shellRadiusY * 0.25, 8, 2.1);

  const outlined = addOutline(sprite);
  drawShellPattern(outlined, Math.round(centerX), Math.round(centerY), traits);
  return outlined;
}

function generateEvolvedSprite(stage: number, traits: CreatureTraits): Sprite {
  const sprite = createGrid();
  const centerX = 7.5;
  const centerXi = Math.round(centerX);
  const bodyRadiusX = traits.bodyWidth * 0.55 + stage * 0.18;
  const bodyRadiusY = traits.bodyHeight * 0.52 + stage * 0.12;
  const bodyY = stage === 1 ? 8.7 : stage === 2 ? 8.4 : 8.1;
  const headRadiusX = Math.max(1.8, bodyRadiusX * (0.7 + traits.headToBodyRatio * 0.25));
  const headRadiusY = Math.max(1.6, bodyRadiusY * (0.45 + traits.headToBodyRatio * 0.3));
  const headY = bodyY - bodyRadiusY * (0.9 + traits.headToBodyRatio * 0.1);
  const bodyPower = 2 + (1 - traits.roundness) * 1.5;

  if (stage >= 3) {
    const wingRadiusX = stage === 4 ? 1.9 + traits.ornamentBias : 1.2 + traits.ornamentBias * 0.8;
    const wingRadiusY = stage === 4 ? 3.0 : 2.1;
    const wingOffset = bodyRadiusX + (stage === 4 ? 1.2 : 0.4);
    fillSuperellipse(sprite, centerX - wingOffset, bodyY - 0.2, wingRadiusX, wingRadiusY, 7, 2.4);
    fillSuperellipse(sprite, centerX + wingOffset, bodyY - 0.2, wingRadiusX, wingRadiusY, 7, 2.4);
  }

  fillSuperellipse(sprite, centerX, bodyY, bodyRadiusX, bodyRadiusY, 2, bodyPower);
  fillSuperellipse(sprite, centerX, headY, headRadiusX, headRadiusY, 2, bodyPower);
  fillSuperellipse(sprite, centerX, bodyY + bodyRadiusY * 0.4, bodyRadiusX * 0.55, bodyRadiusY * 0.42, 3, 2.2);
  fillSuperellipse(sprite, centerX - headRadiusX * 0.25, headY - headRadiusY * 0.35, headRadiusX * 0.35, headRadiusY * 0.22, 8, 2.2);

  if (stage >= 2) {
    const footY = Math.round(bodyY + bodyRadiusY) + 1;
    setPixel(sprite, centerXi - 2, footY, 5);
    setPixel(sprite, centerXi - 1, footY, 5);
    setPixel(sprite, centerXi + 1, footY, 5);
    setPixel(sprite, centerXi + 2, footY, 5);
  }

  if (traits.hasTail && stage >= 2) {
    const tailAnchorX = centerXi + Math.round((bodyRadiusX + 0.5) * traits.tailSide);
    const tailY = Math.round(bodyY + bodyRadiusY * 0.1);
    for (let step = 0; step < traits.tailLength; step++) {
      const tailX = tailAnchorX + (step + 1) * traits.tailSide;
      const bend = Math.floor((step + traits.detailRolls[0] * 2) / 2);
      setPixel(sprite, tailX, tailY - bend, stage >= 3 ? 7 : 5);
    }
  }

  const topY = Math.round(headY - headRadiusY) - 1;
  if (traits.hasTopFeature) {
    drawTopFeature(sprite, centerXi, topY, traits);
  }
  if (traits.hasHat) {
    drawHat(sprite, centerXi, topY - (traits.hasTopFeature ? traits.topFeatureHeight : 0), traits);
  }

  const outlined = addOutline(sprite);
  const eyeY = Math.round(headY - 0.1);
  const leftEyeX = Math.round(centerX - (traits.eyeSpacing / 2 + (traits.eyeSize === 2 ? 0.5 : 0)));
  const rightEyeX = Math.round(centerX + (traits.eyeSpacing / 2 + (traits.eyeSize === 2 ? 0.5 : 0)));
  const mouthY = eyeY + traits.eyeSize + 1;
  const cheekY = mouthY;
  const bellyY = Math.round(bodyY + bodyRadiusY * 0.45);

  drawPattern(outlined, centerXi, bellyY, traits);
  drawEye(outlined, leftEyeX, eyeY, traits.eyeStyle, traits.eyeSize);
  drawEye(outlined, rightEyeX, eyeY, traits.eyeStyle, traits.eyeSize);
  drawMouth(outlined, centerXi, mouthY, traits.mouthStyle);

  if (traits.hasCheeks) {
    setPixel(outlined, leftEyeX, cheekY, 6);
    setPixel(outlined, rightEyeX, cheekY, 6);
  }

  if (stage === 4) {
    drawWingDecoration(outlined, centerXi, bodyY, bodyRadiusX, traits);
  }

  return outlined;
}

export function generateSprite(stage: number, rng: () => number): number[][];
export function generateSprite(stage: number, rng: () => number, traits: GenerationTraits): number[][];
export function generateSprite(stage: number, rng: () => number, traits: GenerationTraits = DEFAULT_TRAITS): number[][] {
  const normalizedStage = Math.max(0, Math.min(4, stage));
  const digestibility = clamp(traits.digestibility, 0, 1);
  const metabolism = clamp(traits.metabolism, 0, 1);
  const creatureTraits = createCreatureTraits(rng, digestibility, metabolism);

  if (normalizedStage === 0) {
    return generateEggSprite(creatureTraits);
  }

  return generateEvolvedSprite(normalizedStage, creatureTraits);
}

function createPalette(mood: Mood, rng: () => number, metabolism: number, digestibility: number): Palette {
  const baseHue = rng() * 360;
  const accentHue = baseHue + (rng() > 0.5 ? 34 : -34);
  const wingHue = baseHue + (rng() > 0.5 ? -48 : 48);
  const saturation = 48 + clamp(metabolism, 0, 1) * 32;
  const moodLightness = {
    thriving: 60,
    content: 55,
    uneasy: 51,
    stressed: 48,
    sleeping: 58,
  }[mood] + (digestibility - 0.5) * 6;

  return {
    1: hsl(baseHue, saturation * 0.42, moodLightness - 30),
    2: hsl(baseHue, saturation, moodLightness),
    3: hsl(baseHue + 18, saturation * 0.72, moodLightness + 12),
    4: hsl(baseHue + 210, 20 + digestibility * 24, 14 + (1 - metabolism) * 6),
    5: hsl(accentHue, saturation * 0.96, moodLightness - 2),
    6: hsl(baseHue + 320, 72, moodLightness + 16),
    7: hsl(wingHue, saturation * 0.76, moodLightness - 7),
    8: hsl(baseHue + 10, saturation * 0.55, Math.min(92, moodLightness + 24)),
  };
}

function blinkSprite(sprite: Sprite): Sprite {
  return sprite.map((row) => row.map((color) => (color === 4 ? 1 : color)));
}

function sleepSprite(sprite: Sprite): Sprite {
  return sprite.map((row) => row.map((color) => (color === 4 ? 1 : color)));
}

// ── Sprite selection from personality + stage ──

function selectSprite(
  stage: number,
  metabolism: number,
  digestibility: number,
  curiosityLevel: number,
  driftDetected: boolean,
  recentScore: number,
  level: number,
): string {
  // Stage 0: egg
  if (stage === 0) return "/sprites/01-egg.png";

  // Special states override personality
  if (driftDetected) return "/sprites/18-drift-chameleon.png";
  if (recentScore < 0.3 && level > 3) return "/sprites/19-escalation-phoenix.png";

  // Final form: dragon (high level + high policy count implied by stage 4)
  if (stage >= 4 && level >= 10) return "/sprites/20-transcend-dragon.png";

  // Personality-based selection
  const isCurious = curiosityLevel >= 0.6;
  const isAggressive = metabolism >= 0.6;
  const isConservative = metabolism <= 0.4;
  // else balanced

  if (isCurious) {
    if (stage <= 1) return "/sprites/02-curious-kitten.png";
    if (stage === 2) return "/sprites/03-curious-cat.png";
    if (digestibility >= 0.6) return "/sprites/04-curious-star-cat.png";
    return "/sprites/05-curious-owl-sage.png";
  }

  if (isAggressive) {
    if (stage <= 1) return "/sprites/06-aggressive-puppy.png";
    if (stage === 2) return "/sprites/07-aggressive-wolf.png";
    if (digestibility <= 0.4) return "/sprites/09-aggressive-armored-dog.png";
    return "/sprites/08-aggressive-lion.png";
  }

  if (isConservative) {
    if (stage <= 1) return "/sprites/10-conservative-bear-baby.png";
    if (stage === 2) return "/sprites/11-conservative-bear-adult.png";
    if (digestibility >= 0.6) return "/sprites/12-conservative-crystal-bear.png";
    return "/sprites/13-conservative-turtle-sage.png";
  }

  // Balanced
  if (stage <= 1) return "/sprites/14-balanced-rabbit-baby.png";
  if (stage === 2) return "/sprites/15-balanced-rabbit-adult.png";
  if (stage === 3) return "/sprites/16-balanced-fox-master.png";
  return "/sprites/17-balanced-golden-fox.png";
}

const NAME_STORAGE_KEY = "correx-tamagotchi-name";

export default function Tamagotchi(props: Props) {
  const [jumpFrame, setJumpFrame] = useState(0);
  const [isEditingName, setIsEditingName] = useState(false);
  const [customName, setCustomName] = useState<string | null>(null);
  const [hasMounted, setHasMounted] = useState(false);
  const nameInputRef = useRef<HTMLInputElement>(null);
  const editCancelledRef = useRef(false);

  const mood = deriveMood(props);
  const stage = deriveStage(props);
  const metabolism = props.personality?.metabolism ?? DEFAULT_TRAITS.metabolism;
  const digestibility = props.personality?.digestibility ?? DEFAULT_TRAITS.digestibility;
  const curiosityLevel = props.personality?.curiosityLevel ?? 0.5;
  const driftDetected = props.personality?.driftDetected ?? false;
  const seed = props.personality ? personalitySeed(props.personality) : DEFAULT_PERSONALITY_SEED;

  const spriteSrc = useMemo(
    () => selectSprite(stage, metabolism, digestibility, curiosityLevel, driftDetected, props.recentScore, props.level),
    [stage, metabolism, digestibility, curiosityLevel, driftDetected, props.recentScore, props.level],
  );

  // Bounce animation on data change
  const [bouncing, setBouncing] = useState(false);
  useEffect(() => {
    setBouncing(true);
    const t = setTimeout(() => setBouncing(false), 600);
    return () => clearTimeout(t);
  }, [props.dataVersion]);

  const handleClick = () => {
    setBouncing(true);
    setTimeout(() => setBouncing(false), 600);
  };

  // Mood-based hue for name color
  const moodColor = {
    thriving: "hsl(140, 60%, 55%)",
    content: "hsl(200, 50%, 55%)",
    uneasy: "hsl(40, 60%, 55%)",
    stressed: "hsl(0, 60%, 55%)",
    sleeping: "hsl(240, 30%, 60%)",
  }[mood];

  const autoName = useMemo(() => {
    const nameRng = mulberry32(seed + 7777);
    return generateName(nameRng);
  }, [seed]);

  // Load custom name from localStorage on mount (avoids SSR hydration mismatch)
  useEffect(() => {
    setHasMounted(true);
    try {
      const stored = localStorage.getItem(NAME_STORAGE_KEY);
      if (stored) setCustomName(stored);
    } catch {}
  }, []);

  // Before mount, always show autoName to avoid SSR hydration mismatch
  const displayName = hasMounted ? (customName || autoName) : autoName;

  const handleNameClick = () => {
    editCancelledRef.current = false;
    setIsEditingName(true);
    setTimeout(() => nameInputRef.current?.focus(), 50);
  };

  const handleNameSubmit = (value: string) => {
    if (editCancelledRef.current) {
      editCancelledRef.current = false;
      setIsEditingName(false);
      return;
    }
    const trimmed = value.trim();
    if (trimmed && trimmed !== autoName) {
      setCustomName(trimmed);
      try { localStorage.setItem(NAME_STORAGE_KEY, trimmed); } catch {}
    } else {
      // Reset to auto-generated
      setCustomName(null);
      try { localStorage.removeItem(NAME_STORAGE_KEY); } catch {}
    }
    setIsEditingName(false);
  };

  return (
    <div className="flex flex-col items-center gap-1">
      <div
        className="relative cursor-pointer"
        onClick={handleClick}
        style={{ width: 96, height: 96 }}
      >
        {/* Shadow */}
        <div
          className="absolute bottom-1 left-1/2 -translate-x-1/2 rounded-full"
          style={{ width: 40, height: 8, background: "rgba(0,0,0,0.15)" }}
        />
        {/* Sprite */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={spriteSrc}
          alt={MOOD_LABELS[mood]}
          className={`absolute inset-0 m-auto ${bouncing ? "animate-bounce" : ""} ${mood === "sleeping" ? "opacity-60 grayscale-[30%]" : ""}`}
          style={{ imageRendering: "pixelated", width: 64, height: 64 }}
          draggable={false}
        />
      </div>
      {isEditingName ? (
        <input
          ref={nameInputRef}
          className="text-xs font-bold text-center bg-transparent border-b outline-none w-20"
          style={{ color: moodColor, borderColor: moodColor }}
          defaultValue={displayName}
          maxLength={12}
          onBlur={(e) => handleNameSubmit(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") handleNameSubmit((e.target as HTMLInputElement).value);
            if (e.key === "Escape") { editCancelledRef.current = true; (e.target as HTMLInputElement).blur(); }
          }}
        />
      ) : (
        <p
          className="text-xs font-bold cursor-pointer hover:underline"
          style={{ color: moodColor }}
          onClick={handleNameClick}
          title="クリックで名前を変更"
        >
          {displayName}
        </p>
      )}
      <div className="flex items-center gap-2">
        <p className="text-[10px]" style={{ color: moodColor }}>
          {MOOD_LABELS[mood]}
        </p>
        <p className="text-[9px]" style={{ color: "var(--text-faint)" }}>
          {SPRITE_LABELS[spriteSrc] || STAGE_LABELS[stage]}
        </p>
      </div>
    </div>
  );
}
