import { NextResponse } from "next/server";
import { execFileSync } from "child_process";
import fs from "fs";
import path from "path";

export const dynamic = "force-dynamic";

// ---------------------------------------------------------------------------
// Primary path: Python unified data script (SQLite + JSON merged)
// ---------------------------------------------------------------------------

function tryPythonBridge(): Record<string, unknown> | null {
  const projectRoot = path.resolve(process.cwd(), "..");
  const script = path.join(projectRoot, "scripts", "serve_unified_data.py");

  if (!fs.existsSync(script)) return null;

  const dbPath =
    process.env.DATABASE_PATH ||
    path.join(projectRoot, "data", "pseudo_intelligence.sqlite3");
  const memoryDir =
    process.env.MEMORY_DIR ||
    path.join(process.env.HOME || "", ".pseudo-intelligence");

  const args = [`--memory-dir`, memoryDir];
  if (fs.existsSync(dbPath)) {
    args.push("--database", dbPath);
  }

  try {
    const stdout = execFileSync("/usr/bin/env", ["python3", script, ...args], {
      timeout: 10000,
      encoding: "utf-8",
      maxBuffer: 10 * 1024 * 1024, // 10MB
    });
    return JSON.parse(stdout);
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Fallback: direct JSON file reads (no SQLite data)
// ---------------------------------------------------------------------------

function loadJson(filePath: string): unknown[] {
  try {
    if (!fs.existsSync(filePath)) return [];
    const raw = JSON.parse(fs.readFileSync(filePath, "utf-8"));
    if (Array.isArray(raw)) return raw;
    if (raw && typeof raw === "object" && "items" in raw) return raw.items;
    return [];
  } catch {
    return [];
  }
}

function loadJsonObject(filePath: string): Record<string, unknown> {
  try {
    if (!fs.existsSync(filePath)) return {};
    return JSON.parse(fs.readFileSync(filePath, "utf-8"));
  } catch {
    return {};
  }
}

function loadGrowth(growthDir: string): unknown[] {
  try {
    if (!fs.existsSync(growthDir)) return [];
    return fs
      .readdirSync(growthDir)
      .filter((f) => f.endsWith(".json"))
      .sort()
      .map((f) => {
        try {
          return JSON.parse(
            fs.readFileSync(path.join(growthDir, f), "utf-8"),
          );
        } catch {
          return null;
        }
      })
      .filter(
        (r): r is Record<string, unknown> =>
          r != null && typeof r === "object" && "case_id" in r,
      );
  } catch {
    return [];
  }
}

function loadFromDir(dir: string) {
  const episodes = loadJson(path.join(dir, "history.json"));
  const turns = loadJson(path.join(dir, "conversation_history.json"));
  const personalRules = loadJson(path.join(dir, "preference_rules.json"));
  const growth = loadGrowth(path.join(dir, "growth"));
  const transitions = loadJson(path.join(dir, "context_transitions.json"));
  const dreamLog = loadJson(path.join(dir, "dream_log.json"));
  const meanings = loadJson(path.join(dir, "meanings.json"));
  const principles = loadJson(path.join(dir, "principles.json"));

  const profilesData = loadJsonObject(path.join(dir, "profiles.json")) as {
    active?: string;
    profiles?: Record<string, unknown>;
  };
  const activeProfile = profilesData.active || "personal";
  const profilesList = profilesData.profiles || {};
  const publicRules = loadJson(
    path.join(dir, "profiles", "public_rules.json"),
  );

  let rules: unknown[];
  if (activeProfile === "public") {
    rules = publicRules;
  } else if (activeProfile === "hybrid") {
    rules = [...personalRules, ...publicRules];
  } else {
    rules = personalRules;
  }

  if (profilesList) {
    const p = profilesList as Record<string, Record<string, unknown>>;
    if (p.personal) p.personal.rules_count = personalRules.length;
    if (p.public) p.public.rules_count = publicRules.length;
    if (p.hybrid)
      p.hybrid.rules_count = personalRules.length + publicRules.length;
  }

  return {
    episodes,
    turns,
    rules,
    growth,
    transitions,
    dreamLog,
    meanings,
    principles,
    profiles: {
      active: activeProfile,
      list: profilesList,
      personalCount: personalRules.length,
      publicCount: publicRules.length,
    },
    // Empty SQLite fields for type compatibility
    experiments: [],
    benchCases: [],
    transferEvaluations: [],
    skills: [],
    adaptiveRules: [],
    adaptiveCorrections: [],
    ghosts: loadJson(path.join(dir, "ghosts.json")),
    ghostTrajectories: loadJson(path.join(dir, "ghost_trajectories.json")),
    policies: loadJson(path.join(dir, "policies.json")),
    universalLaws: loadJson(path.join(dir, "ghost_universal_laws.json")),
    positiveLaws: loadJson(path.join(dir, "ghost_positive_laws.json")),
    personality: (() => { const p = loadJson(path.join(dir, "personality.json")); return p.length > 0 ? p[p.length - 1] : null; })(),
    _source: { sqlite: null, json: dir, timestamp: new Date().toISOString() },
  };
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

export async function GET() {
  // Try Python bridge first (unified SQLite + JSON)
  const unified = tryPythonBridge();
  if (unified) {
    // Supplement with doctrine data not in Python bridge
    const home = process.env.HOME || "";
    const memDir = process.env.MEMORY_DIR || path.join(home, ".pseudo-intelligence");
    if (!unified.policies) unified.policies = loadJson(path.join(memDir, "policies.json"));
    if (!unified.universalLaws) unified.universalLaws = loadJson(path.join(memDir, "ghost_universal_laws.json"));
    if (!unified.positiveLaws) unified.positiveLaws = loadJson(path.join(memDir, "ghost_positive_laws.json"));
    if (!unified.personality) {
      const pData = loadJson(path.join(memDir, "personality.json"));
      unified.personality = pData.length > 0 ? pData[pData.length - 1] : null;
    }
    return NextResponse.json(unified);
  }

  // Fallback: JSON-only
  const memoryDir = process.env.MEMORY_DIR;
  if (memoryDir) {
    return NextResponse.json(loadFromDir(memoryDir));
  }

  const home = process.env.HOME;
  if (home) {
    const defaultDir = path.join(home, ".pseudo-intelligence");
    if (fs.existsSync(path.join(defaultDir, "history.json"))) {
      return NextResponse.json(loadFromDir(defaultDir));
    }
  }

  return NextResponse.json(loadFromDir(""));
}
