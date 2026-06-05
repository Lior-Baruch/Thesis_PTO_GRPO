"""
_shared — cross-method helpers for the Exp3 trainers (GRPO_Exp3 and PTO_Exp3).

Single canonical copy of the modules both trainers need: runtime detection,
model + tokenizer + LoRA + checkpoint utilities, conversation lifecycle (state,
generation, prompt extraction with MCL filter), oracle scoring + look-ahead
reward, and TensorBoard parsing/plotting.

The `Exp3_PTO_GRPO/code/system_prompts_builder.py` and `.../questionnaires.py`
files live at the `code/` root (one canonical copy each); the EDA package
also reaches them via the same `sys.path` prepend pattern.
"""

from .runtime import (
    RuntimeInfo,
    detect_runtime,
    init_openai_client,
    authenticate,
    verify_helpers,
)

from .model import (
    CHATML_TEMPLATE,
    ITER_PREFIX,
    HF_CKPT_PREFIX,
    ADAPTER_SUBDIR,
    ADAPTER_FILES,
    HF_TRAINER_FILES,
    setup_tokenizer,
    build_quantization_config,
    list_iteration_checkpoints,
    get_latest_iteration,
    validate_iteration_checkpoint,
    list_hf_checkpoints,
    get_latest_hf_checkpoint,
    validate_hf_checkpoint,
    load_base_model,
    sync_pad_token,
    patch_generate,
    build_multi_adapter_model_iterative,
    get_adapter_param_count,
    setup_permutations,
    resolve_start_state,
    compute_cumulative_step_offset,
)

from .convs import (
    ConversationState,
    load_conversation_from_csv,
    save_conversation_csv,
    reconstruct_conversation_text,
    print_conversation,
    initialize_conversation,
    update_conversation,
    handle_session_end,
    generate_patient_response_async,
    generate_patient_responses_batch,
    generate_therapist_responses_batch,
    conversation_loop_batch,
    synthesize_conversations_batch,
    run_synthesis_batch,
    generate_all_conversations,
    format_conversation_for_oracle,
    turns_to_messages,
    turns_to_patient_messages,
    extract_prompts_from_conversations,
    extract_prompts_from_saved_conversations,
)

from .reward import (
    OracleConfig,
    LookaheadConfig,
    OracleAsyncPrimitives,
    get_evaluation_json,
    make_reward_fn,
    simulate_lookahead_single,
    simulate_lookahead_batch,
)

from .tb_plots import (
    find_event_files,
    parse_tensorboard_logs,
    compute_iteration_boundaries,
    scan_scalar_tags,
    summarize_available_tags,
    plot_iteration_metrics,
    RunTBLogger,
    CheckpointMetadataCallback,
    CumulativeStepCallback,
    init_iteration_logging,
    finish_iteration_logging,
    setup_tensorboard_logging,
    patch_trainer_tensorboard_callback,
)

from .eda_recorder import EDARecorder


__all__ = [
    # runtime
    "RuntimeInfo", "detect_runtime", "init_openai_client", "authenticate", "verify_helpers",
    # model
    "CHATML_TEMPLATE",
    "ITER_PREFIX", "HF_CKPT_PREFIX", "ADAPTER_SUBDIR", "ADAPTER_FILES", "HF_TRAINER_FILES",
    "setup_tokenizer", "build_quantization_config",
    "list_iteration_checkpoints", "get_latest_iteration", "validate_iteration_checkpoint",
    "list_hf_checkpoints", "get_latest_hf_checkpoint", "validate_hf_checkpoint",
    "load_base_model", "sync_pad_token", "patch_generate",
    "build_multi_adapter_model_iterative", "get_adapter_param_count",
    "setup_permutations",
    "resolve_start_state", "compute_cumulative_step_offset",
    # convs
    "ConversationState",
    "load_conversation_from_csv", "save_conversation_csv",
    "reconstruct_conversation_text", "print_conversation",
    "initialize_conversation", "update_conversation", "handle_session_end",
    "generate_patient_response_async", "generate_patient_responses_batch",
    "generate_therapist_responses_batch",
    "conversation_loop_batch", "synthesize_conversations_batch",
    "run_synthesis_batch", "generate_all_conversations",
    "format_conversation_for_oracle", "turns_to_messages", "turns_to_patient_messages",
    "extract_prompts_from_conversations", "extract_prompts_from_saved_conversations",
    # reward
    "OracleConfig", "LookaheadConfig", "OracleAsyncPrimitives",
    "get_evaluation_json", "make_reward_fn",
    "simulate_lookahead_single", "simulate_lookahead_batch",
    # tb_plots
    "find_event_files", "parse_tensorboard_logs", "compute_iteration_boundaries",
    "scan_scalar_tags", "summarize_available_tags", "plot_iteration_metrics", "RunTBLogger",
    "CheckpointMetadataCallback", "CumulativeStepCallback",
    "init_iteration_logging", "finish_iteration_logging",
    "setup_tensorboard_logging", "patch_trainer_tensorboard_callback",
    # eda_recorder
    "EDARecorder",
]
