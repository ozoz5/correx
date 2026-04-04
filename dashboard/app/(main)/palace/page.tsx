"use client";

import { useEffect, useRef, useState } from "react";
import { useMemory } from "@/lib/useMemory";
import type { PreferenceRule } from "@/lib/types";

// ─── Bigram Jaccard similarity ──────────────────────────────────────────────

function getBigrams(text: string): Set<string> {
  const s = text.toLowerCase().replace(/\s+/g, " ").trim();
  const bigrams = new Set<string>();
  for (let i = 0; i < s.length - 1; i++) {
    bigrams.add(s.slice(i, i + 2));
  }
  return bigrams;
}

function bigramJaccard(a: string, b: string): number {
  if (!a || !b) return 0;
  const ba = getBigrams(a);
  const bb = getBigrams(b);
  if (ba.size === 0 || bb.size === 0) return 0;
  let intersection = 0;
  for (const g of ba) {
    if (bb.has(g)) intersection++;
  }
  const union = ba.size + bb.size - intersection;
  return union === 0 ? 0 : intersection / union;
}

// ─── Types ──────────────────────────────────────────────────────────────────

interface GraphNode {
  id: string;
  rule: PreferenceRule;
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
}

interface GraphEdge {
  a: number;
  b: number;
  similarity: number;
}

// ─── Color helper ────────────────────────────────────────────────────────────

function nodeColor(rule: PreferenceRule): string {
  if (rule.status === "promoted") {
    const needsRevision = rule.tags?.includes("needs_revision");
    return needsRevision ? "#d97706" : "#059669";
  }
  return "#94a3b8";
}

function nodeGlow(rule: PreferenceRule): string {
  if (rule.status === "promoted") {
    const needsRevision = rule.tags?.includes("needs_revision");
    return needsRevision ? "rgba(217,119,6,0.3)" : "rgba(5,150,105,0.3)";
  }
  return "rgba(148,163,184,0.15)";
}

// ─── Edge fire animation state ───────────────────────────────────────────────

interface EdgeFire {
  edgeIndex: number;
  t: number; // 0..1 progress
  dir: 1 | -1;
}

// ─── Canvas component ────────────────────────────────────────────────────────

function PalaceCanvas({ rules }: { rules: PreferenceRule[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const nodesRef = useRef<GraphNode[]>([]);
  const edgesRef = useRef<GraphEdge[]>([]);
  const edgeFiresRef = useRef<EdgeFire[]>([]);
  const dragRef = useRef<{ nodeIndex: number; offsetX: number; offsetY: number } | null>(null);
  const hoveredRef = useRef<number | null>(null);
  const frameRef = useRef<number>(0);
  const frozenRef = useRef(false);
  const [hoverRule, setHoverRule] = useState<PreferenceRule | null>(null);
  const [hoverPos, setHoverPos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [canvasSize, setCanvasSize] = useState({ w: 800, h: 600 });

  // Measure container
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        setCanvasSize({ w: Math.floor(width), h: Math.floor(height) });
      }
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  // Build graph when rules or canvas size changes
  useEffect(() => {
    const { w, h } = canvasSize;
    const EDGE_THRESHOLD = 0.2;

    const nodes: GraphNode[] = rules.map((rule, i) => ({
      id: rule.id,
      rule,
      x: w * 0.1 + Math.random() * w * 0.8,
      y: h * 0.1 + Math.random() * h * 0.8,
      vx: 0,
      vy: 0,
      r: (rule.status === "promoted" ? 10 : 7) + (rule.confidence_score ?? 0) * 4,
    }));

    const edges: GraphEdge[] = [];
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const sim = bigramJaccard(
          nodes[i].rule.instruction || nodes[i].rule.statement || "",
          nodes[j].rule.instruction || nodes[j].rule.statement || ""
        );
        if (sim >= EDGE_THRESHOLD) {
          edges.push({ a: i, b: j, similarity: sim });
        }
      }
    }

    nodesRef.current = nodes;
    edgesRef.current = edges;
    edgeFiresRef.current = [];
    frozenRef.current = false;
  }, [rules, canvasSize]);

  // Periodically spark edge fires
  useEffect(() => {
    const interval = setInterval(() => {
      const edges = edgesRef.current;
      if (edges.length === 0 || frozenRef.current) return;
      const edge = edges[Math.floor(Math.random() * edges.length)];
      const idx = edgesRef.current.indexOf(edge);
      edgeFiresRef.current.push({
        edgeIndex: idx,
        t: 0,
        dir: Math.random() > 0.5 ? 1 : -1,
      });
    }, 1200);
    return () => clearInterval(interval);
  }, []);

  // Draw loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d") as CanvasRenderingContext2D | null;
    if (!ctx) return;
    const safeCtx = ctx;

    const { w, h } = canvasSize;
    let localFrame = 0;
    const FREEZE_FRAME = 100;

    const REPULSION = 3500;
    const SPRING_K = 0.04;
    const SPRING_LEN = 120;
    const DAMPING = 0.85;

    function tick() {
      frameRef.current = requestAnimationFrame(tick);
      const ctx = safeCtx;
      localFrame++;

      const nodes = nodesRef.current;
      const edges = edgesRef.current;

      if (!frozenRef.current) {
        // Force-directed physics
        for (let i = 0; i < nodes.length; i++) {
          let fx = 0;
          let fy = 0;

          // Repulsion between all pairs
          for (let j = 0; j < nodes.length; j++) {
            if (i === j) continue;
            const dx = nodes[i].x - nodes[j].x;
            const dy = nodes[i].y - nodes[j].y;
            const dist = Math.sqrt(dx * dx + dy * dy) || 1;
            const force = REPULSION / (dist * dist);
            fx += (dx / dist) * force;
            fy += (dy / dist) * force;
          }

          // Spring attraction along edges
          for (const edge of edges) {
            let other = -1;
            if (edge.a === i) other = edge.b;
            if (edge.b === i) other = edge.a;
            if (other === -1) continue;
            const dx = nodes[other].x - nodes[i].x;
            const dy = nodes[other].y - nodes[i].y;
            const dist = Math.sqrt(dx * dx + dy * dy) || 1;
            const stretch = dist - SPRING_LEN;
            fx += (dx / dist) * SPRING_K * stretch;
            fy += (dy / dist) * SPRING_K * stretch;
          }

          // Center gravity
          fx += (w / 2 - nodes[i].x) * 0.003;
          fy += (h / 2 - nodes[i].y) * 0.003;

          nodes[i].vx = (nodes[i].vx + fx * 0.016) * DAMPING;
          nodes[i].vy = (nodes[i].vy + fy * 0.016) * DAMPING;

          // Skip drag target
          if (dragRef.current?.nodeIndex === i) continue;

          nodes[i].x += nodes[i].vx;
          nodes[i].y += nodes[i].vy;

          // Boundary
          nodes[i].x = Math.max(nodes[i].r + 5, Math.min(w - nodes[i].r - 5, nodes[i].x));
          nodes[i].y = Math.max(nodes[i].r + 5, Math.min(h - nodes[i].r - 5, nodes[i].y));
        }

        if (localFrame >= FREEZE_FRAME) {
          frozenRef.current = true;
          nodes.forEach((n) => { n.vx = 0; n.vy = 0; });
        }
      }

      // Advance edge fires
      edgeFiresRef.current = edgeFiresRef.current.filter((ef) => {
        ef.t += 0.025;
        return ef.t <= 1;
      });

      // Draw
      ctx.clearRect(0, 0, w, h);

      // Edges
      for (let ei = 0; ei < edges.length; ei++) {
        const edge = edges[ei];
        const na = nodes[edge.a];
        const nb = nodes[edge.b];
        if (!na || !nb) continue;
        const alpha = edge.similarity * 0.3;
        ctx.beginPath();
        ctx.strokeStyle = `rgba(8,145,178,${alpha})`;
        ctx.lineWidth = 0.5 + edge.similarity;
        ctx.moveTo(na.x, na.y);
        ctx.lineTo(nb.x, nb.y);
        ctx.stroke();
      }

      // Edge fires (animated particle)
      for (const ef of edgeFiresRef.current) {
        const edge = edges[ef.edgeIndex];
        if (!edge) continue;
        const na = nodes[edge.a];
        const nb = nodes[edge.b];
        if (!na || !nb) continue;
        const t = ef.dir === 1 ? ef.t : 1 - ef.t;
        const px = na.x + (nb.x - na.x) * t;
        const py = na.y + (nb.y - na.y) * t;
        const alpha = Math.sin(ef.t * Math.PI) * 0.6;
        const grad = ctx.createRadialGradient(px, py, 0, px, py, 6);
        grad.addColorStop(0, `rgba(8,145,178,${alpha})`);
        grad.addColorStop(1, "rgba(8,145,178,0)");
        ctx.beginPath();
        ctx.arc(px, py, 6, 0, Math.PI * 2);
        ctx.fillStyle = grad;
        ctx.fill();
      }

      // Nodes
      for (let i = 0; i < nodes.length; i++) {
        const node = nodes[i];
        const color = nodeColor(node.rule);
        const glow = nodeGlow(node.rule);
        const isHovered = hoveredRef.current === i;
        const r = isHovered ? node.r * 1.4 : node.r;

        // Glow halo (reduced for light bg)
        const haloGrad = ctx.createRadialGradient(node.x, node.y, r * 0.5, node.x, node.y, r * 2.5);
        haloGrad.addColorStop(0, glow);
        haloGrad.addColorStop(1, "rgba(255,255,255,0)");
        ctx.beginPath();
        ctx.arc(node.x, node.y, r * 2.5, 0, Math.PI * 2);
        ctx.fillStyle = haloGrad;
        ctx.fill();

        // Circle
        ctx.beginPath();
        ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
        ctx.fillStyle = color;
        if (isHovered) {
          ctx.shadowBlur = 8;
          ctx.shadowColor = color;
        } else {
          ctx.shadowBlur = 0;
        }
        ctx.fill();
        ctx.shadowBlur = 0;

        // Candidate: dim ring
        if (node.rule.status !== "promoted") {
          ctx.beginPath();
          ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
          ctx.strokeStyle = "rgba(148,163,184,0.4)";
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }
    }

    tick();
    return () => cancelAnimationFrame(frameRef.current);
  }, [canvasSize]);

  // Mouse events
  function getNodeAt(cx: number, cy: number): number {
    const nodes = nodesRef.current;
    for (let i = nodes.length - 1; i >= 0; i--) {
      const n = nodes[i];
      const dx = cx - n.x;
      const dy = cy - n.y;
      if (Math.sqrt(dx * dx + dy * dy) <= n.r + 4) return i;
    }
    return -1;
  }

  function handleMouseDown(e: React.MouseEvent<HTMLCanvasElement>) {
    const rect = canvasRef.current!.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const idx = getNodeAt(cx, cy);
    if (idx !== -1) {
      const node = nodesRef.current[idx];
      dragRef.current = { nodeIndex: idx, offsetX: cx - node.x, offsetY: cy - node.y };
    }
  }

  function handleMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    const rect = canvasRef.current!.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;

    if (dragRef.current) {
      const node = nodesRef.current[dragRef.current.nodeIndex];
      node.x = cx - dragRef.current.offsetX;
      node.y = cy - dragRef.current.offsetY;
      frozenRef.current = false;
      return;
    }

    const idx = getNodeAt(cx, cy);
    hoveredRef.current = idx === -1 ? null : idx;
    if (idx !== -1) {
      setHoverRule(nodesRef.current[idx].rule);
      setHoverPos({ x: e.clientX, y: e.clientY });
    } else {
      setHoverRule(null);
    }
  }

  function handleMouseUp() {
    dragRef.current = null;
  }

  function handleMouseLeave() {
    dragRef.current = null;
    hoveredRef.current = null;
    setHoverRule(null);
  }

  return (
    <div ref={containerRef} className="relative w-full h-full">
      <canvas
        ref={canvasRef}
        width={canvasSize.w}
        height={canvasSize.h}
        className="w-full h-full cursor-crosshair"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
      />
      {hoverRule && (
        <div
          className="fixed z-50 pointer-events-none max-w-xs p-3 rounded-lg text-xs leading-relaxed"
          style={{
            left: hoverPos.x + 14,
            top: hoverPos.y - 10,
            background: "var(--bg-card)",
            border: "1px solid rgba(8,145,178,0.2)",
            boxShadow: "0 4px 16px rgba(0,0,0,0.1)",
            color: "var(--text-body)",
          }}
        >
          <p className="font-medium mb-1" style={{ color: "var(--text-primary)" }}>{hoverRule.instruction || hoverRule.statement}</p>
          <div className="flex flex-wrap gap-1.5 text-[9px]" style={{ color: "var(--text-label)" }}>
            <span>{hoverRule.applies_to_scope || "General"}</span>
            <span>{"Evidence\u00D7"}{hoverRule.evidence_count}</span>
            <span
              style={{
                color: hoverRule.status === "promoted" ? "var(--color-emerald)" : "var(--text-muted)",
              }}
            >
              {hoverRule.status === "promoted" ? "PROMOTED" : "CANDIDATE"}
            </span>
            {hoverRule.support_score != null && hoverRule.support_score > 0 && (
              <span>{"Support: "}{hoverRule.support_score.toFixed(1)}</span>
            )}
            {hoverRule.confidence_score != null && hoverRule.confidence_score > 0 && (
              <span>{"Conf: "}{(hoverRule.confidence_score * 100).toFixed(0)}%</span>
            )}
            {hoverRule.expected_gain != null && hoverRule.expected_gain > 0 && (
              <span style={{ color: "var(--color-emerald)" }}>+{hoverRule.expected_gain.toFixed(2)}</span>
            )}
            {hoverRule.tags?.includes("needs_revision") && (
              <span style={{ color: "var(--color-amber)" }}>Needs review</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Legend ──────────────────────────────────────────────────────────────────

function Legend() {
  return (
    <div className="flex items-center gap-5 text-[9px]" style={{ color: "var(--text-label)" }}>
      <div className="flex items-center gap-1.5">
        <div className="w-2.5 h-2.5 rounded-full" style={{ background: "#059669", boxShadow: "0 0 4px rgba(5,150,105,0.3)" }} />
        <span>Promoted</span>
      </div>
      <div className="flex items-center gap-1.5">
        <div className="w-2.5 h-2.5 rounded-full" style={{ background: "#d97706", boxShadow: "0 0 4px rgba(217,119,6,0.3)" }} />
        <span>Needs review</span>
      </div>
      <div className="flex items-center gap-1.5">
        <div className="w-2 h-2 rounded-full" style={{ background: "#94a3b8", border: "1px solid rgba(148,163,184,0.4)" }} />
        <span>Learning</span>
      </div>
      <div className="flex items-center gap-1.5">
        <div className="h-px w-6" style={{ background: "rgba(8,145,178,0.4)" }} />
        <span>Related</span>
      </div>
    </div>
  );
}

// ─── Page ────────────────────────────────────────────────────────────────────

export default function PalacePage() {
  const { data, loading } = useMemory();

  if (loading) {
    return (
      <div className="flex items-center gap-2 p-8 text-xs" style={{ color: "var(--text-muted)" }}>
        <div className="w-1.5 h-1.5 rounded-full bg-cyan-500 animate-ping" />
        Loading…
      </div>
    );
  }

  const rules = data.rules;

  return (
    <div className="flex flex-col h-screen p-5 gap-4">
      {/* Header */}
      <div className="shrink-0 space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div
              className="w-1.5 h-1.5 rounded-full"
              style={{ background: "#059669", boxShadow: "0 0 4px rgba(5,150,105,0.3)" }}
            />
            <h1 className="text-[10px] tracking-[0.25em] text-emerald-600">MEMORY PALACE</h1>
          </div>
          <span className="text-[9px]" style={{ color: "var(--text-muted)" }}>{rules.length} rules</span>
        </div>
        <p className="text-[10px]" style={{ color: "var(--text-muted)" }}>
          Memory palace of rules. Closer nodes are more related. Brighter means more established.
        </p>
        <Legend />
      </div>

      {/* Canvas */}
      {rules.length === 0 ? (
        <div className="flex-1 flex items-center justify-center">
          <p className="text-sm tracking-wide" style={{ color: "var(--text-muted)" }}>No rules yet</p>
        </div>
      ) : (
        <div
          className="flex-1 rounded-xl overflow-hidden"
          style={{
            background: "var(--bg-subtle)",
            border: "1px solid rgba(8,145,178,0.1)",
            boxShadow: "inset 0 0 40px rgba(0,0,0,0.03)",
          }}
        >
          <PalaceCanvas rules={rules} />
        </div>
      )}
    </div>
  );
}
