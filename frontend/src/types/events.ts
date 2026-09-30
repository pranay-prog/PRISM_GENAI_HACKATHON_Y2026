export type Decision = "WAIT" | "RETRIEVE" | "REFINE" | "SUPPRESS";

export interface BackendEvent<P = any> {
  event_id?: string;
  type: string;
  timestamp: number;
  session_id: string;
  payload: P;
}

export interface ControllerPayload {
  decision: Decision;
  confidence: number;
  reason: string;
  query: string;
  changed_information: string[];
  features: Record<string, number>;
  contributions: Record<string, number>;
  score: number;
  band: string;
  retrieve_intent_ids: string[];
  refine_intent_ids: string[];
  skipped_redundant: { label: string; query: string; novelty: number }[];
  general_inquiry: boolean;
}

export interface Citation { chunk_id: string; document_id: string; section: string; claim?: string }
export interface Claim { claim: string; citations: Citation[] }
export interface AnswerSection {
  intent_id: string;
  label: string;
  display: string;
  supported: boolean;
  regenerated: boolean;
  claims: Claim[];
  uncertainty?: string;
  grounding?: number;
}
export interface AnswerVersion {
  version: number;
  timestamp: number;
  trigger: string;
  answer_text: string;
  sections: AnswerSection[];
  citations: Citation[];
  uncertainties: string[];
  affected_intents: string[];
  citation_coverage: number | null;
  generator?: string;
  format?: string;
  source_version?: number;
  changes: {
    added_intents: string[];
    regenerated_intents: string[];
    carried_over_intents: string[];
    added_claims: string[];
    removed_claims: string[];
  };
}

export interface IntentView {
  intent_id: string;
  label: string;
  display: string;
  confidence: number;
  status: "detected" | "retrieved" | "answered";
  query: string;
  retrieval_count: number;
  evidence_chunk_ids: string[];
}

export interface Candidate { label: string; display: string; confidence: number }

export interface RankedRow { chunk_id: string; document_id: string; section: string; score: number | null }
export interface Latency { dense: number; bm25: number; fusion: number; rerank: number; total: number }

export interface RetrievalCall {
  intent_id: string;
  label: string;
  query: string;
  mode: string;
  parallel_group: string;
  dense: RankedRow[];
  bm25: RankedRow[];
  chunk_ids: string[];
  latency_ms: Latency;
  new_chunk_ratio: number;
  empty: boolean;
  started: number;
  timestamp: number;
}

export interface FusedRow { chunk_id: string; rrf: number; ranks: Record<string, number> }

export interface RetrievalGroup {
  id: string;
  started: number;
  finished?: number;
  trigger: Decision;
  calls: RetrievalCall[];
  wallMs?: number;
  sumMs?: number;
  fusedPerQuery: Record<string, FusedRow[]>;
  reranked: Record<string, RankedRow[]>;
  reranker?: string;
}

export interface EvidenceItem {
  chunk_id: string;
  document_id: string;
  section: string;
  title: string;
  text: string;
  matched_intents: string[];
  rrf_score: number;
  rerank_score: number | null;
  firstSeen: number;
}

export interface LatencyPoint extends Partial<Latency> { t: number; intent_id: string; query: string }

export interface Metrics {
  transcript_chunks: number;
  decision_counts: Partial<Record<Decision, number>>;
  decision_history: { t: number; decision: Decision; reason: string; confidence: number; score: number | null }[];
  retrieval_calls: number;
  unnecessary_retrievals: number;
  suppressed: number;
  refinements: number;
  first_retrieval_latency_s: number | null;
  first_answer_latency_s: number | null;
  final_answer_after_last_chunk_s: number | null;
  retrieval_latency_ms: LatencyPoint[];
  mean_retrieval_latency_ms: number | null;
  answer_versions: number;
  citation_coverage: number | null;
  event_count: number;
}

export interface SystemInfo {
  embedder: string;
  dense_index: string;
  bm25: string;
  reranker: string;
  llm_provider: string;
  llm_status: string;
  documents: number;
  chunks: number;
  domain: string;
  retrieval_mode: string;
  controller_mode: string;
  refinement_mode: string;
  thresholds: { retrieve: number; monitor: number; redundancy: number };
}

export interface Scenario {
  id: string;
  title: string;
  demo: boolean;
  description?: string;
  chunks: { time: number; text: string; final?: boolean }[];
}

export interface LateConstraint {
  t: number;
  trigger: string;
  constraint_text: string;
  new_constraints: { fact: string; value: unknown; evidence: string }[];
  invalidated_assumptions: { fact: string; previous: unknown; now: unknown }[];
  affected_intents: { intent_id: string; label: string; because_of: string[]; previous_query: string; new_query: string }[];
}

export interface Suppression { t: number; reason: string; instruction: string; source_version: number; reused_citations: string[] }
