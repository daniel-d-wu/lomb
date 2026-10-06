# Makes prompts/ a package. One file per fluenceme, each defining a single
# module-level CONFIG that pipeline/registry.py discovers automatically:
#
#   ErrorTagConfig      an error fluenceme (gdd1..gdd6, gvt1, gvt2, lp, lpf):
#                       a definition + real examples the error sorter uses to
#                       tag errors the error finder found. No LLM call of its
#                       own -- adding one adds a category, not a call.
#   MetricPromptConfig  a labeler fluenceme (structure_breadth): its own
#                       prompt, one batch call per chunk.
#
# formulaic.py, filled_pause.py and unfilled_pause.py hold only reference
# constants (BUNDLES / FILLER_TOKENS / PAUSE_THRESHOLD_SECONDS), no CONFIG --
# those fluencemes are computed directly by pipeline/direct_computation.py.
