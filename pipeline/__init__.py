# The core orchestration + fluenceme engine.
#
# pipeline.py        -- run_pipeline_from_turns(): ties everything below
#                        together for one target speaker.
# error_finder.py    -- call 1 of error detection: find EVERY grammar and
#                        word-choice error in a chunk of sentences.
# error_sorter.py    -- call 2: give each found error one error-fluenceme
#                        tag (GDD-1 ... LP) or OTHER_ERROR.
# registry.py        -- discovers every fluenceme: ERROR_TAGS and
#                        LLM_LABELERS from prompts/*.py, DIRECT_FLUENCEMES.
# batching.py        -- one call labels a whole sentence list (labelers
#                        such as STRUCTURE_BREADTH).
# direct_computation.py -- fluencemes computed with no LLM call at all:
#                        FORMULAIC, FILLED_PAUSE, UNFILLED_PAUSE, WPM.
# flag_quality.py    -- which error flags are trustworthy enough to count
#                        and show (shared by storage and Report 1).
# metric_types.py    -- PromptConfig, MetricPromptConfig, ErrorTagConfig.
# fake_provider.py   -- FakeProvider for offline wiring tests.
#
# Depends on transcript_processing/ (raw transcript -> Turn objects) and
# prompts/ (one file per fluenceme). See lomb_prompts/README.md.
