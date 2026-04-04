import type { MemoryData, ConversationTurn, DreamLogEntry } from "./types";

// --- Level System ---

const LEVELS = [
  { threshold: 0, name: "Sprouting", nameEn: "Sprouting" },
  { threshold: 30, name: "Learning", nameEn: "Learning" },
  { threshold: 100, name: "Growing", nameEn: "Growing" },
  { threshold: 250, name: "Establishing", nameEn: "Establishing" },
  { threshold: 500, name: "Maturing", nameEn: "Maturing" },
  { threshold: 1000, name: "Autonomous", nameEn: "Autonomous" },
] as const;

// --- 5-Axis Maturity (Übermensch Model) ---

export interface MaturityAxis {
  name: string;
  nameEn: string;
  value: number; // 0-100
  weight: number;
  description: string;
}

export interface MaturityInfo {
  score: number;
  level: number;
  levelName: string;
  levelNameEn: string;
  nextLevelAt: number;
  progressPercent: number;
  axes: MaturityAxis[];
}

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

function stddev(arr: number[]): number {
  if (arr.length < 2) return 0;
  const mean = arr.reduce((s, v) => s + v, 0) / arr.length;
  return Math.sqrt(arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length);
}

export function calculateMaturity(data: MemoryData): MaturityInfo {
  const promoted = data.rules.filter((r) => r.status === "promoted");
  const allRules = data.rules;

  // Axis 1: Rule Effectiveness (30%) — do promoted rules actually work?
  const totalSuccess = allRules.reduce((s, r) => s + (r.success_count ?? 0), 0);
  const totalFailure = allRules.reduce((s, r) => s + (r.failure_count ?? 0), 0);
  const effectiveness = totalSuccess + totalFailure > 0
    ? (totalSuccess / (totalSuccess + totalFailure)) * 100
    : promoted.length > 0 ? 50 : 0; // default 50 if no success/failure data yet

  // Axis 2: Guidance Impact (25%) — does guidance improve output?
  const guidedTurns = data.turns.filter((t) => t.guidance_applied && t.reaction_score != null);
  const unguidedTurns = data.turns.filter((t) => !t.guidance_applied && t.reaction_score != null);
  const guidedAvg = guidedTurns.length > 0
    ? guidedTurns.reduce((s, t) => s + (t.reaction_score ?? 0), 0) / guidedTurns.length
    : 0;
  const unguidedAvg = unguidedTurns.length > 0
    ? unguidedTurns.reduce((s, t) => s + (t.reaction_score ?? 0), 0) / unguidedTurns.length
    : 0;
  const impactDelta = guidedTurns.length > 0 && unguidedTurns.length > 0
    ? guidedAvg - unguidedAvg
    : data.growth.length > 0
      ? data.growth.reduce((s, g) => s + g.delta, 0) / data.growth.length
      : 0;
  const impact = clamp(impactDelta * 200, 0, 100);

  // Axis 3: Coverage (15%) — how many domains does the system understand?
  const uniqueScopes = new Set(promoted.map((r) => r.applies_to_scope).filter(Boolean));
  const uniqueContexts = new Set(
    allRules.flatMap((r) => (r.applies_when_tags ?? []).slice(0, 3))
  );
  const coverage = clamp(uniqueScopes.size * 12 + Math.min(50, uniqueContexts.size * 2), 0, 100);

  // Axis 4: Stability (20%) — are recent scores consistent?
  const recentScores = data.turns
    .slice(-30)
    .map((t) => t.reaction_score)
    .filter((s): s is number => s != null);
  const variance = stddev(recentScores);
  const stability = recentScores.length >= 3
    ? clamp((1 - variance * 3) * 100, 0, 100)
    : 0;

  // Axis 5: Learning Speed (10%) — how fast do corrections become rules?
  const turnsWithCorrections = data.turns.filter(
    (t) => (t.extracted_corrections?.length ?? 0) > 0
  ).length;
  const promotionRate = turnsWithCorrections > 0
    ? (promoted.length / turnsWithCorrections) * 100
    : 0;
  const speed = clamp(promotionRate * 2, 0, 100);

  const axes: MaturityAxis[] = [
    { name: "Effect.", nameEn: "Effectiveness", value: Math.round(effectiveness), weight: 0.30, description: "Are rules actually working?" },
    { name: "Impact", nameEn: "Impact", value: Math.round(impact), weight: 0.25, description: "Does guidance improve output?" },
    { name: "Cover.", nameEn: "Coverage", value: Math.round(coverage), weight: 0.15, description: "Breadth of supported domains" },
    { name: "Stable", nameEn: "Stability", value: Math.round(stability), weight: 0.20, description: "Consistency of judgments" },
    { name: "Speed", nameEn: "Speed", value: Math.round(speed), weight: 0.10, description: "Speed from correction to rule" },
  ];

  // Composite score
  const score = Math.round(
    axes.reduce((s, a) => s + a.value * a.weight, 0) * 10 // scale 0-1000
  );

  let levelIdx = 0;
  for (let i = LEVELS.length - 1; i >= 0; i--) {
    if (score >= LEVELS[i].threshold) {
      levelIdx = i;
      break;
    }
  }

  const currentLevel = LEVELS[levelIdx];
  const nextLevel = LEVELS[levelIdx + 1];
  const nextLevelAt = nextLevel?.threshold ?? currentLevel.threshold + 50;
  const rangeStart = currentLevel.threshold;
  const rangeSize = nextLevelAt - rangeStart;
  const progressPercent =
    rangeSize > 0
      ? Math.min(100, Math.round(((score - rangeStart) / rangeSize) * 100))
      : 100;

  return {
    score,
    level: levelIdx + 1,
    levelName: currentLevel.name,
    levelNameEn: currentLevel.nameEn,
    nextLevelAt,
    progressPercent,
    axes,
  };
}

// --- Activity Feed ---

export interface ActivityItem {
  icon: string;
  text: string;
  detail: string;
  timestamp: string;
  type: "rule" | "correction" | "guidance" | "feedback" | "dream" | "episode";
  color: string;
}

export function buildActivityFeed(data: MemoryData): ActivityItem[] {
  const items: ActivityItem[] = [];

  // Turns → corrections & guidance events
  for (const turn of data.turns) {
    if (turn.extracted_corrections?.length) {
      items.push({
        icon: "\u{1F331}",
        text: "Found a new rule seed",
        detail: turn.extracted_corrections[0],
        timestamp: turn.recorded_at,
        type: "correction",
        color: "emerald",
      });
    }
    if (turn.guidance_applied && (turn.reaction_score ?? 0) >= 0.7) {
      items.push({
        icon: "\u{1F4A1}",
        text: "Guidance was effective",
        detail: turn.task_scope,
        timestamp: turn.recorded_at,
        type: "guidance",
        color: "cyan",
      });
    }
    if ((turn.reaction_score ?? 1) < 0.5) {
      items.push({
        icon: "\u{1F527}",
        text: "Received a correction",
        detail: turn.user_feedback?.slice(0, 60) ?? "",
        timestamp: turn.recorded_at,
        type: "feedback",
        color: "amber",
      });
    }
  }

  // Promoted rules
  for (const rule of data.rules) {
    if (rule.status === "promoted") {
      items.push({
        icon: "\u2B06\uFE0F",
        text: "Rule promoted!",
        detail: rule.instruction?.slice(0, 60) ?? "",
        timestamp: rule.last_recorded_at,
        type: "rule",
        color: "emerald",
      });
    }
  }

  // Dream log
  for (const dream of data.dreamLog || []) {
    if (!dream.errors || dream.errors.length === 0) {
      items.push({
        icon: "\u{1F319}",
        text: "Memory consolidated",
        detail: `Rules ${dream.rules_before} → ${dream.rules_after}`,
        timestamp: dream.ran_at,
        type: "dream",
        color: "violet",
      });
    }
  }

  // Episodes
  for (const ep of data.episodes) {
    items.push({
      icon: "\u{1F4E6}",
      text: "Episode saved",
      detail: ep.title?.slice(0, 60) ?? "",
      timestamp: ep.timestamp,
      type: "episode",
      color: "cyan",
    });
  }

  // Sort newest first
  items.sort((a, b) => (b.timestamp ?? "").localeCompare(a.timestamp ?? ""));
  return items.slice(0, 10);
}

// --- Guidance Impact ---

export interface GuidanceImpact {
  guidedAvg: number;
  unguidedAvg: number;
  delta: number;
  guidedCount: number;
  unguidedCount: number;
}

export function calculateGuidanceImpact(
  turns: ConversationTurn[]
): GuidanceImpact {
  const guided = turns.filter(
    (t) => t.guidance_applied && t.reaction_score != null
  );
  const unguided = turns.filter(
    (t) => !t.guidance_applied && t.reaction_score != null
  );

  const avg = (arr: ConversationTurn[]) =>
    arr.length > 0
      ? arr.reduce((s, t) => s + (t.reaction_score ?? 0), 0) / arr.length
      : 0;

  const guidedAvg = avg(guided);
  const unguidedAvg = avg(unguided);

  return {
    guidedAvg,
    unguidedAvg,
    delta: guidedAvg - unguidedAvg,
    guidedCount: guided.length,
    unguidedCount: unguided.length,
  };
}

// --- Learning Speed Grade ---

export function learningGrade(data: MemoryData): string {
  if (data.turns.length === 0) return "-";
  const correctionsPerTurn =
    data.turns.reduce(
      (s, t) => s + (t.extracted_corrections?.length ?? 0),
      0
    ) / data.turns.length;
  if (correctionsPerTurn >= 3) return "S";
  if (correctionsPerTurn >= 2) return "A";
  if (correctionsPerTurn >= 1) return "B";
  if (correctionsPerTurn >= 0.5) return "C";
  return "D";
}

// --- Elapsed since first data ---

export interface ElapsedTime {
  days: number;
  hours: number;
  minutes: number;
  label: string;
  since: string; // original date string
}

function parseDate(dateStr: string): number {
  if (!dateStr) return NaN;
  const normalized = dateStr.replace(/\//g, "-").replace(" ", "T");
  return new Date(normalized).getTime();
}

export function calculateElapsed(data: MemoryData): ElapsedTime | null {
  const timestamps: number[] = [];

  for (const t of data.turns) {
    const ts = parseDate(t.recorded_at);
    if (!isNaN(ts)) timestamps.push(ts);
  }
  for (const r of data.rules) {
    const ts = parseDate(r.first_recorded_at);
    if (!isNaN(ts)) timestamps.push(ts);
  }
  for (const e of data.episodes) {
    const ts = parseDate(e.timestamp);
    if (!isNaN(ts)) timestamps.push(ts);
  }

  if (timestamps.length === 0) return null;

  const oldest = Math.min(...timestamps);
  const diffMs = Date.now() - oldest;
  if (diffMs < 0) return null;

  const totalMin = Math.floor(diffMs / 60000);
  const days = Math.floor(totalMin / 1440);
  const hours = Math.floor((totalMin % 1440) / 60);
  const minutes = totalMin % 60;

  let label: string;
  if (days > 0) {
    label = `${days}d ${hours}h`;
  } else if (hours > 0) {
    label = `${hours}h ${minutes}m`;
  } else {
    label = `${minutes}m`;
  }

  // Find original date for display
  const oldestDate = new Date(oldest);
  const since = `${oldestDate.getFullYear()}/${String(oldestDate.getMonth() + 1).padStart(2, "0")}/${String(oldestDate.getDate()).padStart(2, "0")}`;

  return { days, hours, minutes, label, since };
}

// --- Relative Time ---

export function relativeTime(dateStr: string): string {
  if (!dateStr) return "";
  const now = Date.now();
  // Handle "2026/03/31 01:33" format
  const normalized = dateStr.replace(/\//g, "-").replace(" ", "T");
  const then = new Date(normalized).getTime();
  if (isNaN(then)) return dateStr;
  const diffMin = Math.floor((now - then) / 60000);
  if (diffMin < 1) return "just now";
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffH = Math.floor(diffMin / 60);
  if (diffH < 24) return `${diffH}h ago`;
  const diffD = Math.floor(diffH / 24);
  if (diffD < 7) return `${diffD}d ago`;
  return `${Math.floor(diffD / 7)}w ago`;
}
