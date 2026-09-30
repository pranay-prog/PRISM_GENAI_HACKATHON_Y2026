/** Pure derivation of the dashboard view from the backend event stream.
 *  Everything displayed comes from these events; nothing is synthesised. */
import type {
  AnswerVersion, BackendEvent, Candidate, ControllerPayload, Decision, EvidenceItem, IntentView,
  LateConstraint, Metrics, RetrievalGroup, Suppression,
} from "../types/events";

export interface ChunkView { index: number; text: string; final: boolean; t: number; decision?: Decision; reason?: string }
export interface DecisionView extends ControllerPayload { t: number; chunkIndex: number }

export interface SessionView {
  chunks: ChunkView[];
  decisions: DecisionView[];
  intents: IntentView[];
  candidates: Candidate[];
  facts: Record<string, unknown>;
  factChanges: { key: string; old: unknown; new: unknown; chunk_index: number }[];
  groups: RetrievalGroup[];
  evidence: EvidenceItem[];
  answers: AnswerVersion[];
  lateConstraints: LateConstraint[];
  suppressions: Suppression[];
  metrics: Metrics | null;
  errors: { t: number; message: string }[];
  refinedIntentIds: Set<string>;
}

export function derive(events: BackendEvent[]): SessionView {
  const v: SessionView = {
    chunks: [], decisions: [], intents: [], candidates: [], facts: {}, factChanges: [], groups: [],
    evidence: [], answers: [], lateConstraints: [], suppressions: [], metrics: null, errors: [],
    refinedIntentIds: new Set(),
  };
  const groups = new Map<string, RetrievalGroup>();
  const started = new Map<string, number>();
  const evidence = new Map<string, EvidenceItem>();
  let lastDecision: Decision = "RETRIEVE";

  for (const e of events) {
    const p = e.payload;
    switch (e.type) {
      case "transcript_chunk":
        v.chunks.push({ index: p.index, text: p.text, final: p.final, t: e.timestamp });
        break;
      case "facts_updated":
        v.facts = p.facts;
        v.factChanges.push(...p.changes);
        break;
      case "intent_detected":
        v.candidates = p.candidates;
        break;
      case "controller_decision": {
        const chunkIndex = v.chunks.length - 1;
        v.decisions.push({ ...(p as ControllerPayload), t: e.timestamp, chunkIndex });
        const ch = v.chunks[chunkIndex];
        if (ch) {
          ch.decision = p.decision;
          ch.reason = p.reason;
        }
        lastDecision = p.decision;
        if (p.decision === "REFINE") (p.refine_intent_ids as string[]).forEach((id) => v.refinedIntentIds.add(id));
        break;
      }
      case "subquery_created":
        if (!groups.has(p.parallel_group)) {
          groups.set(p.parallel_group, {
            id: p.parallel_group, started: e.timestamp, trigger: lastDecision, calls: [], fusedPerQuery: {}, reranked: {},
          });
        }
        break;
      case "retrieval_started":
        started.set(`${p.parallel_group}:${p.intent_id}`, e.timestamp);
        break;
      case "retrieval_completed": {
        const g = groups.get(p.parallel_group);
        if (g) g.calls.push({ ...p, started: started.get(`${p.parallel_group}:${p.intent_id}`) ?? e.timestamp, timestamp: e.timestamp });
        break;
      }
      case "fusion_completed": {
        const g = groups.get(p.parallel_group);
        if (g) {
          g.wallMs = p.wall_ms;
          g.sumMs = p.sum_of_query_ms;
          g.fusedPerQuery = p.fused_per_query;
          g.finished = e.timestamp;
        }
        for (const u of p.unified_evidence as EvidenceItem[]) {
          const prev = evidence.get(u.chunk_id);
          evidence.set(u.chunk_id, {
            ...u,
            matched_intents: Array.from(new Set([...(prev?.matched_intents ?? []), ...u.matched_intents])).sort(),
            firstSeen: prev?.firstSeen ?? e.timestamp,
          });
        }
        break;
      }
      case "rerank_completed": {
        const g = groups.get(p.parallel_group);
        if (g) {
          g.reranked = p.per_intent;
          g.reranker = p.reranker;
        }
        break;
      }
      case "late_constraint":
        v.lateConstraints.push({ ...p, t: e.timestamp });
        break;
      case "retrieval_suppressed":
        v.suppressions.push({ ...p, t: e.timestamp });
        break;
      case "answer_generated":
        v.answers.push(p.answer);
        break;
      case "session_updated":
        v.intents = p.intents;
        v.facts = p.facts;
        break;
      case "metrics_updated":
        v.metrics = p as Metrics;
        break;
      case "error":
        v.errors.push({ t: e.timestamp, message: p.message });
        break;
    }
  }
  v.groups = Array.from(groups.values());
  v.evidence = Array.from(evidence.values()).sort((a, b) => b.rrf_score - a.rrf_score);
  return v;
}

export const DECISION_HELP: Record<string, string> = {
  stable_intent: "The intent is stable and specific enough to retrieve now.",
  multi_intent_stable: "Several intents are stable; they are retrieved in parallel.",
  utterance_complete: "The speaker finished, so pending intents are retrieved.",
  utterance_complete_general_inquiry: "The speaker finished without a recognised intent; retrieving on the full utterance.",
  late_constraint: "A new statement contradicts an earlier assumption; affected answers are re-retrieved.",
  late_context: "New context changes intents that were already answered; they are re-retrieved.",
  presentation_restructure: "Only the presentation changes. The grounded answer is reused, no retrieval.",
  no_grounded_answer_to_restructure: "Asked to reformat, but there is no answer yet.",
  no_new_information: "Nothing new since the last retrieval; existing evidence is reused.",
  redundant_query: "The candidate query is near-identical to one already executed.",
  ablation_every_chunk: "Ablation mode: retrieving on every chunk.",
};

export function describeReason(reason: string): string {
  if (DECISION_HELP[reason]) return DECISION_HELP[reason];
  const m = reason.match(/^(monitoring_low|insufficient)_(.+)$/);
  if (m) {
    const feature = m[2].replace(/_/g, " ");
    return m[1] === "monitoring_low"
      ? `Close to the threshold; waiting because ${feature} is the weakest signal.`
      : `Too little to go on yet; ${feature} is the weakest signal.`;
  }
  return reason.replace(/_/g, " ");
}
