import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

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
  if (fs.existsSync(filePath)) {
    fs.copyFileSync(filePath, filePath + ".bak");
  }
  fs.renameSync(tmpPath, filePath);
}

export async function POST(req: Request) {
  try {
    const { id, action, instruction } = await req.json();
    if (!id || !action) {
      return NextResponse.json({ error: "Missing id or action" }, { status: 400 });
    }

    const defaultDir = process.env.MEMORY_DIR || path.join(process.env.HOME || "/tmp", ".pseudo-intelligence");
    const rulesPath = path.join(defaultDir, "preference_rules.json");

    if (!fs.existsSync(rulesPath)) {
      return NextResponse.json({ error: "Rules file not found" }, { status: 404 });
    }

    const rules = JSON.parse(fs.readFileSync(rulesPath, "utf-8"));
    let updated = false;

    const newRules = rules.map((r: Record<string, unknown>) => {
      if (r.id === id) {
        updated = true;
        if (action === "demote") r.status = "demoted";
        if (action === "promote") {
            r.status = (r.evidence_count as number) >= 2 ? "promoted" : "candidate";
        }
        if (action === "edit" && instruction) r.instruction = instruction;
      }
      return r;
    });

    if (updated) {
      atomicWriteJson(rulesPath, newRules);
      return NextResponse.json({ success: true });
    } else {
      return NextResponse.json({ error: "Rule not found" }, { status: 404 });
    }
  } catch (error) {
    return NextResponse.json({ error: "Server error" }, { status: 500 });
  }
}
