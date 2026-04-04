"use client";

import { useEffect, useRef } from "react";

interface Node {
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
  phase: number;
  firing: number;
}

interface Props {
  width?: number;
  height?: number;
  nodeCount?: number;
  /** Increment this to trigger a burst of synaptic firing */
  fireCount?: number;
}

export function NeuralNet({ width = 600, height = 160, nodeCount = 28, fireCount = 0 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fireCountRef = useRef(fireCount);
  const nodesRef = useRef<Node[]>([]);

  // Initialize nodes once
  useEffect(() => {
    nodesRef.current = Array.from({ length: nodeCount }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.08, // very slow drift at rest
      vy: (Math.random() - 0.5) * 0.08,
      r: Math.random() * 1.5 + 1,
      phase: Math.random() * Math.PI * 2,
      firing: 0,
    }));
  }, [nodeCount, width, height]);

  // Trigger burst when fireCount changes
  useEffect(() => {
    if (fireCount === fireCountRef.current) return;
    fireCountRef.current = fireCount;

    const nodes = nodesRef.current;
    if (!nodes.length) return;

    // Fire a cascade: random seed node, then neighbors
    const seed = nodes[Math.floor(Math.random() * nodes.length)];
    seed.firing = 1;

    // Propagate to nearby nodes with delay
    nodes.forEach((n) => {
      const dx = n.x - seed.x;
      const dy = n.y - seed.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 150 && n !== seed) {
        setTimeout(() => { n.firing = 0.8; }, dist * 2);
        setTimeout(() => { n.firing = 0.5; }, dist * 3);
      }
    });

    // Speed up briefly during firing, then settle
    nodes.forEach((n) => {
      n.vx *= 4;
      n.vy *= 4;
    });
    setTimeout(() => {
      nodes.forEach((n) => {
        n.vx = (Math.random() - 0.5) * 0.08;
        n.vy = (Math.random() - 0.5) * 0.08;
      });
    }, 1500);
  }, [fireCount]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let frame: number;

    function draw() {
      if (!ctx) return;
      ctx.clearRect(0, 0, width, height);

      const nodes = nodesRef.current;
      const anyFiring = nodes.some((n) => n.firing > 0.05);

      // Connections — only draw when firing is happening
      if (anyFiring) {
        for (let i = 0; i < nodes.length; i++) {
          for (let j = i + 1; j < nodes.length; j++) {
            const a = nodes[i];
            const b = nodes[j];
            const dx = a.x - b.x;
            const dy = a.y - b.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist > 120) continue;

            const firingBonus = Math.max(a.firing, b.firing);
            if (firingBonus < 0.05) continue;

            const alpha = (1 - dist / 120) * firingBonus * 0.5;
            ctx.beginPath();
            ctx.strokeStyle = `rgba(8,145,178,${alpha})`;
            ctx.lineWidth = 0.5 + firingBonus * 2;
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        }
      } else {
        // At rest: draw faint static connections
        for (let i = 0; i < nodes.length; i++) {
          for (let j = i + 1; j < nodes.length; j++) {
            const a = nodes[i];
            const b = nodes[j];
            const dx = a.x - b.x;
            const dy = a.y - b.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist > 90) continue;
            const alpha = (1 - dist / 90) * 0.08;
            ctx.beginPath();
            ctx.strokeStyle = `rgba(8,145,178,${alpha})`;
            ctx.lineWidth = 0.5;
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        }
      }

      // Nodes
      for (const node of nodes) {
        node.phase += 0.008; // very slow pulse at rest
        node.firing = Math.max(0, node.firing - 0.018);

        const pulse = Math.sin(node.phase) * 0.5 + 0.5;
        const baseAlpha = anyFiring ? 0.2 : 0.12;
        const alpha = baseAlpha + pulse * 0.1 + node.firing * 0.6;
        const glowR = node.r + node.firing * 5;

        if (node.firing > 0.1) {
          const grad = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, glowR * 4);
          grad.addColorStop(0, `rgba(8,145,178,${node.firing * 0.2})`);
          grad.addColorStop(1, "transparent");
          ctx.beginPath();
          ctx.arc(node.x, node.y, glowR * 4, 0, Math.PI * 2);
          ctx.fillStyle = grad;
          ctx.fill();
        }

        ctx.beginPath();
        ctx.arc(node.x, node.y, node.r + node.firing * 3, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(8,145,178,${alpha})`;
        ctx.fill();

        node.x += node.vx;
        node.y += node.vy;
        if (node.x < 0 || node.x > width) node.vx *= -1;
        if (node.y < 0 || node.y > height) node.vy *= -1;
      }

      frame = requestAnimationFrame(draw);
    }

    draw();
    return () => cancelAnimationFrame(frame);
  }, [width, height]);

  return <canvas ref={canvasRef} width={width} height={height} className="w-full opacity-70" />;
}
