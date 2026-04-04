import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export const dynamic = "force-dynamic";

function getProfilesPath(): string {
  const memoryDir = process.env.MEMORY_DIR || path.join(process.env.HOME || "/tmp", ".pseudo-intelligence");
  return path.join(memoryDir, "profiles.json");
}

export async function POST(request: Request) {
  try {
    const { profile } = await request.json();
    if (!profile || !["personal", "public", "hybrid"].includes(profile)) {
      return NextResponse.json({ error: "Invalid profile" }, { status: 400 });
    }

    const profilesPath = getProfilesPath();
    if (!fs.existsSync(profilesPath)) {
      return NextResponse.json({ error: "profiles.json not found" }, { status: 404 });
    }

    const data = JSON.parse(fs.readFileSync(profilesPath, "utf-8"));
    data.active = profile;

    fs.writeFileSync(profilesPath, JSON.stringify(data, null, 2), "utf-8");

    return NextResponse.json({ ok: true, active: profile });
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
