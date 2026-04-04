"use client";

import { useMemo } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useMemory } from "@/lib/useMemory";
import { useTheme } from "@/lib/useTheme";
import { calculateMaturity, calculateElapsed } from "@/lib/maturity";

const navItems = [
  { href: "/",           label: "OVERVIEW",    code: "00" },
  { href: "/doctrine",   label: "POLICIES",    code: "01" },
  { href: "/principles", label: "PRINCIPLES",  code: "02" },
  { href: "/meanings",   label: "PATTERNS",    code: "03" },
  { href: "/rules",      label: "RULES",       code: "04" },
  { href: "/growth",     label: "GROWTH",      code: "05" },
  { href: "/turns",      label: "HISTORY",     code: "06" },
  { href: "/episodes",   label: "EPISODES",    code: "07" },
  { href: "/palace",     label: "MEMORY MAP",  code: "08" },
];

function ThemeToggle() {
  const { dark, toggle } = useTheme();
  return (
    <button
      onClick={toggle}
      className="p-2 rounded-lg transition-colors"
      style={{ background: "var(--bg-subtle)", color: "var(--text-muted)" }}
      title={dark ? "Switch to light mode" : "Switch to dark mode"}
    >
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        {dark ? (
          // Sun icon
          <>
            <circle cx="12" cy="12" r="5" />
            <line x1="12" y1="1" x2="12" y2="3" />
            <line x1="12" y1="21" x2="12" y2="23" />
            <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
            <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
            <line x1="1" y1="12" x2="3" y2="12" />
            <line x1="21" y1="12" x2="23" y2="12" />
            <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
            <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
          </>
        ) : (
          // Moon icon
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
        )}
      </svg>
    </button>
  );
}

// ProfileSelector removed — not implemented in MCP server yet

function LiveIndicator() {
  const { data, lastUpdated } = useMemory();
  const maturity = useMemo(() => calculateMaturity(data), [data]);
  const elapsed = useMemo(() => calculateElapsed(data), [data]);
  const promoted = data.rules.filter((r) => r.status === "promoted").length;
  const timeStr = lastUpdated
    ? lastUpdated.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false })
    : "--:--:--";

  return (
    <div className="p-4 space-y-2" style={{ borderTop: "1px solid var(--border)" }}>
      <div className="flex items-center gap-2 mb-1">
        <span className="text-sm font-bold" style={{ color: "var(--text-primary)" }}>Lv.{maturity.level}</span>
        <span className="text-[10px] font-medium" style={{ color: "var(--color-emerald)" }}>{maturity.levelName}</span>
      </div>
      <div className="h-1 rounded-full overflow-hidden" style={{ background: "var(--bg-bar)" }}>
        <div
          className="h-full rounded-full transition-all duration-1000"
          style={{
            width: `${maturity.progressPercent}%`,
            background: `linear-gradient(to right, var(--cyan), var(--emerald))`,
          }}
        />
      </div>

      <div className="flex items-center gap-2 mt-2">
        <div className="relative">
          <div className="w-1.5 h-1.5 rounded-full" style={{ background: "var(--color-emerald)" }} />
          <div className="w-1.5 h-1.5 rounded-full absolute inset-0 animate-ping-slow" style={{ background: "var(--color-emerald)" }} />
        </div>
        <span className="text-[9px] tracking-widest" style={{ color: "var(--color-emerald)" }}>LIVE {timeStr}</span>
      </div>
      <p className="text-[8px]" style={{ color: "var(--text-label)" }}>
        {promoted} PROMOTED · {data.rules.length} TOTAL
      </p>
      {elapsed && (
        <p className="text-[8px]" style={{ color: "var(--text-faint)" }}>
          {elapsed.label} elapsed
        </p>
      )}
    </div>
  );
}

export default function MainLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="flex min-h-screen" style={{ background: "var(--bg)" }}>
      <nav className="w-52 shrink-0 flex flex-col" style={{ background: "var(--bg-sidebar)", borderRight: "1px solid var(--border)" }}>
        {/* Logo */}
        <div className="p-5" style={{ borderBottom: "1px solid var(--border)" }}>
          <div className="flex items-center justify-between mb-1">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full animate-pulse-dot" style={{ background: "var(--cyan)" }} />
              <span className="text-xs font-bold tracking-[0.25em] glow-cyan">CORREX</span>
            </div>
            <ThemeToggle />
          </div>
          <p className="text-[9px] tracking-widest mt-1" style={{ color: "var(--text-label)" }}>AI CORRECTION OS</p>
          <div className="mt-3 h-px" style={{ background: `linear-gradient(to right, var(--cyan), transparent)`, opacity: 0.3 }} />
        </div>

        {/* Nav */}
        <div className="flex-1 p-3 space-y-0.5">
          {navItems.map((item) => {
            const active = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-xs transition-all relative overflow-hidden"
                style={
                  active
                    ? { background: "var(--bg-subtle)", color: "var(--cyan)", fontWeight: 500 }
                    : { color: "var(--text-muted)" }
                }
              >
                {active && (
                  <div className="absolute left-0 top-1 bottom-1 w-0.5 rounded-full" style={{ background: "var(--cyan)" }} />
                )}
                <span className="text-[8px]" style={{ color: active ? "var(--text-label)" : "var(--text-faint)" }}>{item.code}</span>
                <span className="tracking-widest">{item.label}</span>
                {active && (
                  <div className="ml-auto w-1.5 h-1.5 rounded-full" style={{ background: "var(--cyan)" }} />
                )}
              </Link>
            );
          })}
        </div>

        <LiveIndicator />
      </nav>

      <main className="flex-1 overflow-auto" style={{ background: "var(--bg-main)" }}>{children}</main>
    </div>
  );
}
