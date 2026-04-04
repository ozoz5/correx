"use client";

import { useEffect, useRef, useMemo } from "react";

// --- The "Kawaii" Tuning Knobs (Saji-Kagen) ---
const KAWAII_KNOBS = {
  BREATH_SPEED: 0.0008,         // 整合性のある、極めてゆっくりとした心地よいスイング
  SWAY_AMPLITUDE: 0.08,         // ラジアン（角度）ベースの自然な風の強さ
  PUDGY_TRUNK_WIDTH: 14,        // もっちり太めの可愛い幹の太さ
  LEAF_GLOW_BLUR: 12,           // 魔法のようにぼわっと光らせる度合い
  LEAF_SIZE: 10,                // ぽてっとした葉っぱのベースサイズ
  JEWEL_SIZE: 7,                // キャンディのような可愛い花や実のベースサイズ
  ANIMATION_SPRING: 0.15,       // データ更新時にぽよんと弾むバネの戻る力
};

const THEME_COLORS = {
  light: {
    pot: "rgba(8, 145, 178, 0.12)",
    potBorder: "rgba(8, 145, 178, 0.3)",
    trunk: "rgba(8, 145, 178, 0.5)",
    leaf: "#059669",
    leafPale: "#6ee7b7",
    flower: "#db2777",
    fruit: "#f59e0b",
    seed: "#818cf8",
    glow: "rgba(5, 150, 105, 0.3)",
  },
  dark: {
    pot: "rgba(6, 182, 212, 0.15)",
    potBorder: "rgba(6, 182, 212, 0.4)",
    trunk: "rgba(6, 182, 212, 0.3)",
    leaf: "#10b981",
    leafPale: "#34d399",
    flower: "#f472b6",
    fruit: "#fbbf24",
    seed: "#a5b4fc",
    glow: "rgba(16, 185, 129, 0.6)",
  }
};

interface Props {
  level: number;
  promotedRules: number;
  candidateRules: number;
  totalRules: number;
  meaningCount: number;
  principleCount: number;
  avgConfidence: number;
  deferredCount: number;
  dataVersion: number;
}

function LCG(seed: number) {
  let state = seed;
  return () => {
    state = (state * 1664525 + 1013904223) % 4294967296;
    return state / 4294967296;
  };
}

interface FairyNode {
  id: number;
  depth: number;
  length: number;
  curveDir: number; 
  children: FairyNode[];
  leaves: number;
  flowers: number;
  fruits: number;
}

interface JewelParticle {
  x: number; y: number;
  vx: number; vy: number;
  life: number;
  type: "sparkle" | "falling_leaf";
  rotation: number;
  color: string;
}

export default function GrowthPlant({
  level,
  promotedRules,
  candidateRules,
  totalRules,
  meaningCount,
  principleCount,
  avgConfidence,
  deferredCount,
  dataVersion,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const particlesRef = useRef<JewelParticle[]>([]);
  const bounceYRef = useRef(0);
  const bounceVRef = useRef(0);
  const prevVersionRef = useRef(dataVersion);
  const prevRulesRef = useRef(totalRules);

  const treeModel = useMemo(() => {
    const rng = LCG(108); 
    
    let l_left = totalRules;
    let f_left = meaningCount;
    let p_left = principleCount;
    
    const targetBranches = Math.min(level, 4);

    function buildNode(depth: number, length: number, dir: number): FairyNode {
      const node: FairyNode = {
        id: Math.floor(rng() * 1000),
        depth,
        length,
        curveDir: dir,
        children: [],
        leaves: 0, flowers: 0, fruits: 0
      };
      
      if (depth < targetBranches - 1) {
        node.children.push(buildNode(depth + 1, length * (0.6 + rng() * 0.1), -1));
        if (rng() > 0.2 || depth === 0) {
          node.children.push(buildNode(depth + 1, length * (0.6 + rng() * 0.1), 1));
        }
      }
      return node;
    }
    
    // 【修正】計算可能な長さに上限を設ける
    const rootLen = Math.min(50, 30 + (level * 4));
    const root = buildNode(0, rootLen, 0); 
    
    const allNodes: FairyNode[] = [];
    function gather(n: FairyNode) {
      allNodes.push(n);
      n.children.forEach(gather);
    }
    gather(root);
    allNodes.sort((a, b) => b.depth - a.depth); 
    
    while(p_left > 0 || f_left > 0 || l_left > 0) {
      const n = allNodes[Math.floor(rng() * allNodes.length)];
      if (p_left > 0) { n.fruits++; p_left--; }
      else if (f_left > 0) { n.flowers++; f_left--; }
      else if (l_left > 0) { n.leaves++; l_left--; }
    }
    return root;
  }, [level, totalRules, meaningCount, principleCount]);

  useEffect(() => {
    if (dataVersion !== prevVersionRef.current) {
      prevVersionRef.current = dataVersion;
      bounceVRef.current = 1.0; // バウンス強度を正規化 (1.0)
      
      const W = canvasRef.current?.width || 300;
      const H = canvasRef.current?.height || 250;
      const canopyY = H - 100;

      const isDark = document.documentElement.classList.contains("dark");
      const c = isDark ? THEME_COLORS.dark : THEME_COLORS.light;

      if (totalRules < prevRulesRef.current) {
        particlesRef.current.push({
          x: (Math.random() - 0.5) * 40, 
          y: canopyY - H + 35, // local coordinates (based on pot center)
          vx: (Math.random() - 0.5) * 1.0,
          vy: Math.random() * 0.5 + 0.8,
          life: 1.2,
          type: "falling_leaf",
          rotation: Math.random() * Math.PI,
          color: c.leafPale
        });
      } else if (totalRules > 0) {
        for(let i=0; i<6; i++) {
          particlesRef.current.push({
            x: (Math.random()-0.5)*60,
            y: (canopyY - H + 35) + (Math.random()-0.5)*50,
            vx: (Math.random()-0.5)*0.8,
            vy: (Math.random()-0.5)*0.8 - 0.5,
            life: 1.0,
            type: "sparkle",
            rotation: 0,
            color: Math.random() > 0.5 ? c.leaf : c.flower
          });
        }
      }
      prevRulesRef.current = totalRules;
    }
  }, [dataVersion, totalRules]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    const cx = W / 2;
    const potY = H - 25; // 下げた

    let time = 0;

    const drawCurveBranch = (
      node: FairyNode, 
      startX: number, 
      startY: number, 
      baseAngle: number,   // 親がどの方向に向いているか（FKのベース角度）
      swayPhase: number, 
      colors: typeof THEME_COLORS.light
    ) => {
      // --- 整合性のあるForward Kinematicsアニメーション ---
      // ゴムのように伸縮させるのはやめ、長さは固定（node.length）
      // 「曲がり」は基部の角度（baseAngle）に、このノードのローカルな曲がり（curveDir）と、風のうねり（swayPhase）を足して決める
      
      const localBend = node.curveDir * 0.4;
      // 先端に行くほど風の影響（しなり）が強くなる
      const windAngle = swayPhase * (1 + node.depth * 0.3);
      const targetAngle = baseAngle + localBend + windAngle;
      
      const endX = startX + Math.sin(targetAngle) * node.length;
      const endY = startY - Math.cos(targetAngle) * node.length; // -Y is UP

      // ベジェ曲線の制御点は、親の接線（baseAngle）の方向に伸ばすことで「折れ曲がり」を防ぎ、滑らかにしならせる
      const cpDist = node.length * 0.5;
      const cpX = startX + Math.sin(baseAngle) * cpDist;
      const cpY = startY - Math.cos(baseAngle) * cpDist;

      // 幹（枝）の描画
      ctx.beginPath();
      ctx.moveTo(startX, startY);
      ctx.quadraticCurveTo(cpX, cpY, endX, endY);
      
      const trunkWidth = Math.max(2, KAWAII_KNOBS.PUDGY_TRUNK_WIDTH - (node.depth * 3) + (promotedRules * 0.1));
      
      ctx.strokeStyle = colors.trunk;
      ctx.lineWidth = trunkWidth;
      ctx.lineCap = "round";
      ctx.stroke();

      ctx.strokeStyle = "rgba(255,255,255,0.2)";
      ctx.lineWidth = trunkWidth * 0.3;
      ctx.stroke();

      const rng = LCG(node.id * 10);
      
      // アイテム描画
      const drawJewel = (x: number, y: number, color: string, size: number, hasHighlight: boolean) => {
        ctx.shadowColor = color;
        ctx.shadowBlur = KAWAII_KNOBS.LEAF_GLOW_BLUR;
        ctx.fillStyle = color;
        
        ctx.beginPath();
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;

        if (hasHighlight) {
          ctx.fillStyle = "rgba(255,255,255,0.6)";
          ctx.beginPath();
          ctx.arc(x - size*0.3, y - size*0.3, size*0.25, 0, Math.PI*2);
          ctx.fill();
        }
      };

      for(let i=0; i<node.leaves; i++) {
        const lx = endX + (rng() - 0.5) * 15;
        const ly = endY + (rng() - 0.5) * 15;
        const conf = Math.max(0, Math.min(100, avgConfidence * 100));
        const color = rng() > (conf / 100) ? colors.leafPale : colors.leaf;
        
        ctx.save();
        ctx.translate(lx, ly); 
        ctx.rotate(targetAngle + (rng()-0.5));
        
        ctx.shadowColor = colors.glow;
        ctx.shadowBlur = 8;
        ctx.fillStyle = color;
        
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.bezierCurveTo(8, -4, 6, -16, 0, -18);
        ctx.bezierCurveTo(-6, -16, -8, -4, 0, 0);
        ctx.fill();
        ctx.shadowBlur = 0;
        
        ctx.fillStyle = "rgba(255,255,255,0.4)";
        ctx.beginPath();
        ctx.ellipse(0, -9, 1.5, 3, Math.PI/4, 0, Math.PI*2);
        ctx.fill();
        ctx.restore();
      }

      for(let i=0; i<node.flowers; i++) {
        drawJewel(endX + (rng()-0.5)*20, endY + (rng()-0.5)*20, colors.flower, KAWAII_KNOBS.JEWEL_SIZE, true);
      }

      for(let i=0; i<node.fruits; i++) {
        const fx = endX + (rng()-0.5)*20;
        const fy = endY + 8 + (rng()-0.5)*10; 
        
        ctx.strokeStyle = colors.trunk;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(fx, fy-KAWAII_KNOBS.JEWEL_SIZE);
        ctx.lineTo(endX, endY);
        ctx.stroke();

        drawJewel(fx, fy, colors.fruit, KAWAII_KNOBS.JEWEL_SIZE + 2, true);
      }

      // 子ノードの描画 (baseAngleは描画した枝の末端の角度となるtargetAngleを引き継ぐ)
      node.children.forEach(child => 
        drawCurveBranch(child, endX, endY, targetAngle, swayPhase, colors)
      );
    };

    const draw = () => {
      // Date.now()で安定した時間を取得しブラウザタブ切替時の破綻を防ぐ
      time = Date.now() * KAWAII_KNOBS.BREATH_SPEED; 
      ctx.clearRect(0, 0, W, H);

      const isDark = document.documentElement.classList.contains("dark");
      const colors = isDark ? THEME_COLORS.dark : THEME_COLORS.light;

      // 摩擦とバネ力を持つSquash&Stretch物理
      bounceYRef.current += bounceVRef.current;
      bounceVRef.current -= bounceYRef.current * KAWAII_KNOBS.ANIMATION_SPRING;
      bounceVRef.current *= 0.82; // 摩擦
      const bounce = bounceYRef.current; // 0.0付近を振動

      const sway = Math.sin(time) * KAWAII_KNOBS.SWAY_AMPLITUDE;

      // --- 1. グラスモーフィズムなプランター ---
      // プランター自体はSquashの影響を受けない固定座標
      ctx.fillStyle = colors.pot;
      ctx.strokeStyle = colors.potBorder;
      ctx.lineWidth = 1.5;
      
      const potW = 60;
      const potH = 20;
      
      ctx.beginPath();
      ctx.roundRect(cx - potW/2, potY, potW, potH, 8);
      ctx.fill();
      ctx.stroke();
      
      ctx.fillStyle = "rgba(255,255,255,0.15)";
      ctx.beginPath();
      ctx.roundRect(cx - potW/2 + 2, potY + 2, potW - 4, potH/3, 4);
      ctx.fill();

      const seedRng = LCG(77);
      for(let i=0; i<Math.min(deferredCount, 5); i++) {
         const sx = cx + (seedRng() - 0.5) * (potW - 10);
         const sy = potY + potH/2 + (seedRng() - 0.5) * 10;
         
         ctx.shadowColor = colors.seed;
         ctx.shadowBlur = 5;
         ctx.fillStyle = colors.seed;
         ctx.beginPath();
         ctx.arc(sx, sy, 2, 0, Math.PI*2);
         ctx.fill();
         ctx.shadowBlur = 0;
      }

      // --- 2. 妖精の木（全体Squash & Stretch） ---
      ctx.save();
      // 植物の根元を原点とする
      ctx.translate(cx, potY + 2);
      // データ更新時「全体がムニッ」と縮んで伸びる（Squash & Stretch法）
      // Yが縮んだらXが伸びる（体積保存）
      ctx.scale(1.0 + bounce * 0.1, 1.0 - bounce * 0.1);

      if (level > 0 || totalRules > 0) {
        drawCurveBranch(treeModel, 0, 0, 0, sway, colors);
      } else {
        ctx.strokeStyle = colors.trunk;
        ctx.lineWidth = 6;
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(0, -12);
        ctx.stroke();
        
        ctx.fillStyle = colors.leaf;
        ctx.beginPath();
        ctx.ellipse(-5, -12, 4, 6, -Math.PI/4, 0, Math.PI*2);
        ctx.ellipse(5, -12, 4, 6, Math.PI/4, 0, Math.PI*2);
        ctx.fill();
      }

      // --- 3. パーティクル ---
      particlesRef.current = particlesRef.current.filter((p) => {
        p.x += p.vx;
        p.y += p.vy;
        
        if (p.type === "sparkle") {
           p.life -= 0.02;
           ctx.globalAlpha = Math.max(0, p.life);
           ctx.shadowColor = p.color;
           ctx.shadowBlur = 8;
           ctx.fillStyle = "#fff";
           ctx.beginPath();
           ctx.arc(p.x, p.y, Math.random() * 2 + 1, 0, Math.PI * 2);
           ctx.fill();
           ctx.shadowBlur = 0;
        } else if (p.type === "falling_leaf") {
           p.life -= 0.005;
           p.vx += Math.sin(time) * 0.02; 
           p.rotation += p.vx * 0.05;
           
           ctx.save();
           ctx.translate(p.x, p.y);
           ctx.rotate(p.rotation);
           ctx.globalAlpha = Math.max(0, p.life);
           ctx.fillStyle = p.color;
           ctx.beginPath();
           ctx.moveTo(0, 0);
           ctx.bezierCurveTo(8, -4, 6, -16, 0, -18);
           ctx.bezierCurveTo(-6, -16, -8, -4, 0, 0);
           ctx.fill();
           ctx.restore();
        }
        ctx.globalAlpha = 1.0;
        return p.life > 0;
      });

      ctx.restore();

      animRef.current = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(animRef.current);
  }, [treeModel, level, avgConfidence, totalRules, deferredCount, promotedRules]);

  return (
    <canvas
      ref={canvasRef}
      width={300}
      height={250}
      className="mx-auto"
      style={{ maxWidth: "100%", height: "250px" }}
    />
  );
}
