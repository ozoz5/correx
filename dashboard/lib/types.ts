export interface ConversationTurn {
  id: string;
  recorded_at: string;
  task_scope: string;
  user_message: string;
  assistant_message: string;
  user_feedback: string;
  extracted_corrections: string[];
  tags: string[];
  reaction_score: number | null;
  guidance_applied: boolean;
  metadata: Record<string, unknown>;
}

export interface PreferenceRule {
  id: string;
  statement: string;
  normalized_statement: string;
  instruction: string;
  status: "promoted" | "candidate" | "demoted";
  evidence_count: number;
  first_recorded_at: string;
  last_recorded_at: string;
  applies_to_scope: string;
  applies_when_tags: string[];
  negative_conditions: string[];
  priority: number;
  version: number;
  expires_at: string;
  tags: string[];
  source_turn_ids: string[];
  // Extended fields from MCP
  support_score?: number;
  expected_gain?: number;
  confidence_score?: number;
  strong_signal_count?: number;
  success_count?: number;
  failure_count?: number;
  context_mode?: string;
  latent_context_count?: number;
}

export interface CorrectionRecord {
  recorded_at: string;
  decision_override: string;
  correction_note: string;
  reuse_note: string;
  reason: string;
  scope: string;
  bad_output: string;
  revised_output: string;
  tool_used: string;
  source_user: string;
  accepted: boolean;
}

export interface TrainingExample {
  updated_at: string;
  format: string;
  system_message: string;
  user_message: string;
  prompt: string;
  draft_output: string;
  rejected_output: string;
  accepted_output: string;
  feedback: string;
  accepted: boolean;
  model_id: string;
  policy_version: string;
  accepted_by: string;
  tags: string[];
  temperature: number | null;
  metadata: Record<string, unknown>;
}

export interface EpisodeRecord {
  id: string;
  timestamp: string;
  title: string;
  issuer: string;
  task_type: string;
  profile_name: string;
  source_text: string;
  company_profile: Record<string, unknown>;
  corrections: CorrectionRecord[];
  last_corrected_at: string;
  output: Record<string, unknown>;
  training_example: TrainingExample | null;
  metadata: Record<string, unknown>;
}

export interface GrowthRun {
  output: string;
  guidance_text: string;
}

export interface GrowthRecord {
  record_id: string;
  case_id: string;
  case_title: string;
  baseline_score: number;
  guided_score: number;
  delta: number;
  recorded_at: string;
  baseline_run?: GrowthRun;
  guided_run?: GrowthRun;
}

export interface ContextTransition {
  id: string;
  from_scope: string;
  to_scope: string;
  from_tags: string[];
  to_tags: string[];
  evidence_count: number;
  success_weight: number;
  failure_weight: number;
  confidence_score: number;
  last_seen_at: string;
}

export interface DreamLogEntry {
  ran_at: string;
  rules_before: number;
  rules_after: number;
  merges: number;
  conflicts_found: number;
  turns_archived: number;
  rules_strengthened: number;
  rules_flagged_revision: number;
  errors: string[];
}

export interface ProfileInfo {
  active: string;
  list: Record<string, {
    name: string;
    description: string;
    source: string;
    rules_count: number;
    created_at: string;
  }>;
  personalCount: number;
  publicCount: number;
}

export interface Meaning {
  id: string;
  principle: string;
  normalized_principle: string;
  summary: string;
  source_rule_ids: string[];
  scopes: string[];
  tags: string[];
  strength: number;
  cross_scope_count: number;
  confidence: number;
  first_seen_at: string;
  last_seen_at: string;
  personal_settings_overlap: string[];
  status: string;
}

export interface MeaningPrinciple {
  id: string;
  declaration: string;
  normalized_declaration: string;
  source_meaning_ids: string[];
  source_rule_count: number;
  depth: number;
  scopes: string[];
  confidence: number;
  first_seen_at: string;
  last_seen_at: string;
  personal_settings_overlap: string[];
  status: string;
}

// --- SQLite-sourced types (from Python core) ---

export interface Experiment {
  id: string;
  benchCaseId: string;
  branchName: string;
  createdAt: string;
  findings: string[];
  outcomeScore?: number;
  adoptionDecision?: { adopt: boolean; reason?: string };
  [key: string]: unknown;
}

export interface BenchCase {
  id: string;
  description: string;
  expectedOutcome: string[];
  createdAt: string;
  [key: string]: unknown;
}

export interface TransferEvaluation {
  id: string;
  transferConfirmed: boolean;
  searchCostBefore: number;
  searchCostAfter: number;
  createdAt: string;
  [key: string]: unknown;
}

export interface Skill {
  id: string;
  name: string;
  domain: string;
  reuseCount: number;
  createdAt: string;
  [key: string]: unknown;
}

export interface DataSource {
  sqlite: string | null;
  json: string;
  timestamp: string;
}

// --- Ghost Engine types ---

export interface Ghost {
  id: string;
  rejected_output: string;
  origin: "scolded" | "corrected" | "rejected";
  task_scope: string;
  tags: string[];
  trajectory_id: string;
  prediction_error: number;
  created_at: string;
  [key: string]: unknown;
}

export interface GhostTrajectory {
  id: string;
  theme: string;
  status: "open" | "fired";
  source_ghost_count: number;
  cumulative_pe: number;
  fired: boolean;
  sublimated_principle: string | null;
  fired_at: string | null;
  created_at: string;
  [key: string]: unknown;
}

// --- Unified data shape ---

export interface MemoryData {
  episodes: EpisodeRecord[];
  turns: ConversationTurn[];
  rules: PreferenceRule[];
  growth: GrowthRecord[];
  transitions: ContextTransition[];
  dreamLog: DreamLogEntry[];
  meanings: Meaning[];
  principles: MeaningPrinciple[];
  profiles: ProfileInfo | null;
  // SQLite-sourced (may be empty if Python bridge unavailable)
  experiments: Experiment[];
  benchCases: BenchCase[];
  transferEvaluations: TransferEvaluation[];
  skills: Skill[];
  adaptiveRules: Record<string, unknown>[];
  adaptiveCorrections: Record<string, unknown>[];
  // Ghost Engine
  ghosts: Ghost[];
  ghostTrajectories: GhostTrajectory[];
  // Doctrine (policies + laws)
  policies: Record<string, unknown>[];
  universalLaws: Record<string, unknown>[];
  positiveLaws: Record<string, unknown>[];
  // Personality profile
  personality: Record<string, unknown> | null;
  _source?: DataSource;
}
