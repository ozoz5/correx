import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";

function rulesPath(): string {
  const dir = process.env.MEMORY_DIR
    || path.join(process.env.HOME || "/tmp", ".pseudo-intelligence");
  return path.join(dir, "preference_rules.json");
}

/** Atomic write: write to temp file, fsync, then rename over the target. */
function atomicWriteJson(filePath: string, data: unknown): void {
  const dir = path.dirname(filePath);
  const tmpPath = path.join(dir, `.tmp-${process.pid}-${Date.now()}.json`);
  const content = JSON.stringify(data, null, 2);
  const fd = fs.openSync(tmpPath, "w");
  try {
    fs.writeSync(fd, content, 0, "utf-8");
    fs.fsyncSync(fd);
  } finally {
    fs.closeSync(fd);
  }
  // Backup current file before replacing
  if (fs.existsSync(filePath)) {
    fs.copyFileSync(filePath, filePath + ".bak");
  }
  fs.renameSync(tmpPath, filePath);
}

export async function POST(req: NextRequest) {
  const { id } = await req.json();
  if (!id) return NextResponse.json({ error: "id required" }, { status: 400 });

  const filePath = rulesPath();
  if (!fs.existsSync(filePath)) {
    return NextResponse.json({ error: "rules file not found" }, { status: 404 });
  }

  const raw = JSON.parse(fs.readFileSync(filePath, "utf-8"));
  const rules: Record<string, unknown>[] = Array.isArray(raw) ? raw : raw.items ?? [];

  const rule = rules.find((r) => r.id === id);
  if (!rule) return NextResponse.json({ error: "rule not found" }, { status: 404 });

  rule.status = "promoted";
  rule.evidence_count = Math.max(2, Number(rule.evidence_count ?? 1));

  atomicWriteJson(filePath, rules);

  return NextResponse.json({ ok: true });
}
