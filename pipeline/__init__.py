# The core orchestration + metric-classification engine.
#
# pipeline.py    -- run_pipeline(): ties everything below together, turns a
#                    filtered transcript into every metric's classification.
# registry.py    -- METRIC_PROMPTS: the per-metric LLM prompt configs, and
#                    classify(), the one function that actually calls a
#                    provider for an LLM-assisted metric.
# prefilter.py   -- should_run(): skips LLM calls where the answer is
#                    already knowable without one (cheap, deterministic).
# direct_computation.py -- the 4 metrics computed directly, no LLM call at
#                    all: FORMULAIC, FILLED_PAUSE, UNFILLED_PAUSE, WPM.
# metric_types.py -- MetricPromptConfig, the shared dataclass every
#                    prompts/<metric>.py module and registry.py use.
#
# Depends on transcript_processing/ (turns a raw transcript into these
# modules' input shapes) and prompts/ (the actual per-metric LLM prompt
# content). See lomb_prompts/README.md for the full system map.
