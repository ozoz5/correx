"use client";

import { useEffect, useRef, useState } from "react";
import type { MemoryData } from "./types";

const empty: MemoryData = { episodes: [], turns: [], rules: [], growth: [], transitions: [], dreamLog: [], meanings: [], principles: [], profiles: null, experiments: [], benchCases: [], transferEvaluations: [], skills: [], adaptiveRules: [], adaptiveCorrections: [], ghosts: [], ghostTrajectories: [], policies: [], universalLaws: [], positiveLaws: [], personality: null };

const POLL_INTERVAL = 3000;

export function useMemory() {
  const [data, setData] = useState<MemoryData>(empty);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchData = () =>
    fetch("/api/memory", { cache: "no-store" })
      .then((r) => r.json())
      .then((d) => {
        setData(d);
        setLastUpdated(new Date());
      })
      .catch(() => {})
      .finally(() => setLoading(false));

  useEffect(() => {
    fetchData();
    timerRef.current = setInterval(fetchData, POLL_INTERVAL);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  return { data, loading, lastUpdated };
}
