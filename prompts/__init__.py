# Makes prompts/ a package. One file per LLM-assisted metric lives here --
# each defines a single module-level constant named CONFIG
# (a pipeline.metric_types.MetricPromptConfig -- metric_types.py moved into
# pipeline/ in the 2026-09 reorg, hence the pipeline.metric_types import
# each of these files now starts with). See pipeline/registry.py for how
# these get wired together.
#
# formulaic.py, filled_pause.py, and unfilled_pause.py are the exception:
# each now holds only a reference constant (BUNDLES / FILLER_TOKENS /
# PAUSE_THRESHOLD_SECONDS), no CONFIG -- those three metrics moved off the
# LLM entirely and are computed directly by pipeline/direct_computation.py.
