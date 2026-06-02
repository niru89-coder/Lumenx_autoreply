export interface Draft {
  id: string;
  thread_id: string;
  intent: string;
  sensitivity: string;
  draft_text: string;
  cited_sources: string[];
  uncertainty_flags: string[];
  confidence_label: "high" | "low" | "blocked";
  auto_sendable: boolean;
  guardrail_triggered: boolean;
  model: string;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  latency_ms: number;
  parse_attempts: number;
  context_cache_key: string;
  created_at: string;
  // extended — present in get_draft_with_actions
  human_actions?: HumanAction[];
  confidence_predictions?: ConfidencePrediction[];
  context_snapshot?: Record<string, unknown>;
}

export interface HumanAction {
  id: number;
  draft_id: string;
  action: "approved" | "edited" | "rejected" | "auto_sent";
  final_text: string | null;
  edit_distance: number | null;
  reviewer: string;
  decided_at: string;
}

export interface ConfidencePrediction {
  id: number;
  draft_id: string;
  features: Record<string, number>;
  score: number;
  threshold: number;
  would_auto_send: boolean;
  model_version: string | null;
  created_at: string;
}

export interface FeedbackStats {
  total_drafts: number;
  pending_review: number;
  total_actions: number;
  action_breakdown: Record<string, number>;
  confidence_breakdown: Record<string, number>;
  intent_breakdown: Record<string, number>;
  guardrail_hits: number;
  total_cost_usd: number;
  total_input_tokens: number;
  total_output_tokens: number;
}

export interface Checkpoint {
  version: number;
  val_auc: number | null;
  val_bce: number | null;
  n_train: number | null;
  n_val: number | null;
  n_pos_train: number | null;
  n_pos_val: number | null;
  n_epochs: number | null;
  temperature: number | null;
  feature_version: number | null;
  train_seed: number | null;
  is_active: boolean;
  created_at: string | null;
  path: string;
}

export interface ActiveRecord {
  active_version: number;
  previous_version: number | null;
  promoted_at: string;
  promoted_by: string;
}

export interface ModelsResponse {
  active_version: number | null;
  active_record: ActiveRecord | null;
  checkpoints: Checkpoint[];
}

export interface HealthStatus {
  status: string;
  version: string;
  drafts: number;
  pending_review: number;
  auto_send_enabled: boolean;
  auto_send_threshold: number;
  poller: {
    cycles: number;
    threads_processed: number;
    auto_sent: number;
    human_review: number;
    errors: number;
    last_poll_ts: string | null;
  };
}
