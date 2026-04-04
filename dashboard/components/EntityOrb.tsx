"use client";

import { useEffect, useRef, useCallback } from "react";

interface Props {
  level: number;
  promotedRules: number;
  candidateRules: number;
  avgConfidence: number;
  dataVersion: number; // increment to trigger burst
}

const LEVEL_COLORS: Record<number, [string, string]> = {
  1: ["#3b82f6", "#1e40af"], // blue
  2: ["#0891b2", "#0e7490"], // cyan
  3: ["#059669", "#047857"], // emerald
  4: ["#34d399", "#059669"], // emerald-bright
  5: ["#d97706", "#b45309"], // amber
  6: ["#fbbf24", "#d97706"], // gold
};

interface Particle {
  angle: number;
  radius: number;
  speed: number;
  size: number;
  alpha: number;
  promoted: boolean;
}

interface BurstParticle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  alpha: number;
}

export default function EntityOrb({
  level,
  promotedRules,
  candidateRules,
  avgConfidence,
  dataVersion,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const particlesRef = useRef<Particle[]>([]);
  const burstsRef = useRef<BurstParticle[]>([]);
  const prevVersionRef = useRef(dataVersion);

  const initParticles = useCallback(() => {
    const particles: Particle[] = [];
    for (let i = 0; i < promotedRules; i++) {
      particles.push({
        angle: (Math.PI * 2 * i) / Math.max(promotedRules, 1),
        radius: 50 + Math.random() * 15,
        speed: 0.008 + Math.random() * 0.004,
        size: 3.5,
        alpha: 0.9,
        promoted: true,
      });
    }
    for (let i = 0; i < candidateRules; i++) {
      particles.push({
        angle: (Math.PI * 2 * i) / Math.max(candidateRules, 1),
        radius: 70 + Math.random() * 25,
        speed: 0.003 + Math.random() * 0.003,
        size: 1.8,
        alpha: 0.4,
        promoted: false,
      });
    }
    particlesRef.current = particles;
  }, [promotedRules, candidateRules]);

  useEffect(() => {
    initParticles();
  }, [initParticles]);

  // Burst on data change
  useEffect(() => {
    if (dataVersion !== prevVersionRef.current) {
      prevVersionRef.current = dataVersion;
      const bursts: BurstParticle[] = [];
      for (let i = 0; i < 20; i++) {
        const angle = (Math.PI * 2 * i) / 20;
        bursts.push({
          x: 0,
          y: 0,
          vx: Math.cos(angle) * (1.5 + Math.random()),
          vy: Math.sin(angle) * (1.5 + Math.random()),
          life: 1,
          alpha: 0.8,
        });
      }
      burstsRef.current = bursts;
    }
  }, [dataVersion]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    const cx = W / 2;
    const cy = H / 2;

    const [coreColor, darkColor] = LEVEL_COLORS[level] ?? LEVEL_COLORS[1];
    const coreRadius = 18 + level * 5;

    let frame = 0;

    const draw = () => {
      frame++;
      ctx.clearRect(0, 0, W, H);

      const breathe = 1 + Math.sin(frame * 0.02) * 0.03;
      const r = coreRadius * breathe;

      // Outer glow (reduced for light bg)
      const glowAlpha = 0.06 + avgConfidence * 0.08;
      const glow = ctx.createRadialGradient(cx, cy, r, cx, cy, r * 2.5);
      glow.addColorStop(0, `${coreColor}${Math.round(glowAlpha * 255).toString(16).padStart(2, "0")}`);
      glow.addColorStop(1, "transparent");
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(cx, cy, r * 2.5, 0, Math.PI * 2);
      ctx.fill();

      // Core orb
      const grad = ctx.createRadialGradient(
        cx - r * 0.3,
        cy - r * 0.3,
        0,
        cx,
        cy,
        r
      );
      grad.addColorStop(0, coreColor);
      grad.addColorStop(0.7, darkColor);
      grad.addColorStop(1, `${darkColor}aa`);
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fill();

      // Inner shimmer
      const shimmerAlpha = 0.2 + Math.sin(frame * 0.04) * 0.1;
      ctx.fillStyle = `rgba(255,255,255,${shimmerAlpha})`;
      ctx.beginPath();
      ctx.ellipse(cx - r * 0.25, cy - r * 0.25, r * 0.4, r * 0.2, -0.5, 0, Math.PI * 2);
      ctx.fill();

      // Orbiting particles
      for (const p of particlesRef.current) {
        p.angle += p.speed;
        const px = cx + Math.cos(p.angle) * p.radius * breathe;
        const py = cy + Math.sin(p.angle) * p.radius * breathe;

        if (p.promoted) {
          // Promoted: bright with subtle glow
          const pg = ctx.createRadialGradient(px, py, 0, px, py, p.size * 2.5);
          pg.addColorStop(0, `rgba(5, 150, 105, ${p.alpha * 0.7})`);
          pg.addColorStop(1, "transparent");
          ctx.fillStyle = pg;
          ctx.beginPath();
          ctx.arc(px, py, p.size * 2.5, 0, Math.PI * 2);
          ctx.fill();
        }

        ctx.fillStyle = p.promoted
          ? `rgba(5, 150, 105, ${p.alpha})`
          : `rgba(120, 140, 160, ${p.alpha})`;
        ctx.beginPath();
        ctx.arc(px, py, p.size, 0, Math.PI * 2);
        ctx.fill();

        // Connection filament to core (reduced alpha)
        if (p.promoted) {
          ctx.strokeStyle = `rgba(5, 150, 105, 0.08)`;
          ctx.lineWidth = 0.5;
          ctx.beginPath();
          ctx.moveTo(cx, cy);
          ctx.lineTo(px, py);
          ctx.stroke();
        }
      }

      // Burst particles (adjusted for light bg)
      burstsRef.current = burstsRef.current.filter((b) => {
        b.x += b.vx;
        b.y += b.vy;
        b.life -= 0.015;
        b.alpha = b.life * 0.5;
        if (b.life <= 0) return false;

        ctx.fillStyle = `rgba(5, 150, 105, ${b.alpha})`;
        ctx.beginPath();
        ctx.arc(cx + b.x, cy + b.y, 2, 0, Math.PI * 2);
        ctx.fill();
        return true;
      });

      animRef.current = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(animRef.current);
  }, [level, avgConfidence, promotedRules, candidateRules]);

  return (
    <canvas
      ref={canvasRef}
      width={300}
      height={200}
      className="mx-auto"
      style={{ maxWidth: "100%" }}
    />
  );
}
