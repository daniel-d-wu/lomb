# Lomb

A speech-analytics demo that turns a recorded German conversation into a fluency, accuracy, and complexity report — grounded in the CAF framework (Brand & Götz 2011; Huang & Gráf 2025).

**Live demo:** https://daniel-d-wu.github.io/lomb/

## Layout

Reorganized 2026-09 from a flat pile of `.py` files into subsystem folders, so the codebase has a mental map instead of ~40 files in one directory. Every file that imports another local file does so via a fully-qualified dotted path from the repo root (e.g. `from pipeline.registry import ...`), with a small `sys.path` bootstrap at the top of the file so imports resolve the same way whether that file is run directly or imported by something in a different folder.

```
pipeline/                the fluenceme engine
    pipeline.py               orchestrates a full run: transcript in, every fluenceme's result out
    error_finder.py           error detection, call 1: find EVERY grammar/word-choice error in a chunk
    error_sorter.py           error detection, call 2: tag each found error (GDD-1 ... LP, or OTHER_ERROR)
    registry.py               discovers every fluenceme (error tags, labelers, direct) -- single source of truth
    batching.py               one LLM call labels a whole sentence list (STRUCTURE_BREADTH)
    direct_computation.py     fluencemes computed with plain code, not an LLM (FORMULAIC, FILLED_PAUSE, UNFILLED_PAUSE, WPM)
    flag_quality.py           which error flags are trustworthy enough to count and show
    metric_types.py           PromptConfig, MetricPromptConfig, ErrorTagConfig
    fake_provider.py          FakeProvider for free, offline wiring tests

    LLM calls per 15-min chunk: 2 for all error fluencemes together (find, then sort) + 1 per
    labeler -- 3 today. Error counts are per error (two in one sentence count twice).

prompts/                  one file per fluenceme
    gdd1.py ... gdd6.py, gvt1.py, gvt2.py, lp.py, lpf.py
        error fluencemes: CONFIG = ErrorTagConfig -- a definition + real examples the
        sorter uses. Adding one = adding one file here; it becomes a new category for
        the sorter (no new LLM call), a storage metric and (if report1_tag is set) a
        Report 1 card type.
    structure_breadth.py
        a labeler: CONFIG = MetricPromptConfig, its own prompt and call
    formulaic.py, filled_pause.py, unfilled_pause.py
        exception: these moved to direct computation, so each holds only a reference
        constant (BUNDLES / FILLER_TOKENS / PAUSE_THRESHOLD_SECONDS), no CONFIG

providers/                LLM backends, swappable behind one shared interface
    llm_provider.py           LLMProvider, the abstract interface
    openai_provider.py        the primary provider (verified live)
    gemini_provider.py        second provider, same interface

transcript_processing/    turns a raw transcript into what the pipeline scores
    speaker_filter.py         Turn/Word types; filters a multi-speaker transcript to one speaker
    assemblyai_adapter.py     AssemblyAI's real API response shape -> Turn/Word objects
    whisperx_adapter.py       whisperX+pyannote JSON -> Turn/Word objects (with the repetition-loop guardrail)

voice_enrollment/         the voice-enrollment slice (record a clip, store a voiceprint)
    voice_enrollment.py       SpeakerEmbedder interface + SQLiteVoiceprintRepository
    speechbrain_embedder.py   the real ECAPA-TDNN embedder (heavy; torch/speechbrain not installed)
    enroll_api.py             FastAPI app tying the widget to the embedder + repository
    enroll_widget.html        browser widget: record audio, POST it, show the result
    audio_decode.py           ffmpeg-based decode of whatever the browser recorded
    identify_target_speaker.py   picks which enrolled speaker is present in a transcript

reporting/                turns a pipeline run into the human-facing report
    report.py                  build_report() (result JSON -> report_contract.json) + render_html()
    report_contract.json, report.html, report_preview_new_format.html, lomb_report_standalone.html
        reference/output files, not imported by anything

scripts/                  entry points you actually run
    hardened_whisperx_batch_job.py   Stage 1: audio -> whisperX+pyannote transcript
    run_end_to_end_whisperx_test.py  Stage 2: transcript -> voiceprint match -> fluencemes -> all tables + reports
    run_pipeline_live.py       runs the full pipeline against the AssemblyAI sample transcript
    run_dev_server.py          starts enroll_api.py's FastAPI app for local testing
    run_e2e_widget_test.py     Playwright + fake-camera/mic browser test of the enrollment widget
    build_sample_transcript_v3.py   regenerates data/sample_transcript_assemblyai_v3.json
    build_source_doc.py        regenerates claude/lomb_prompts_source.md from the files on disk
    list_gemini_models.py      lists available Gemini models for the configured API key

tests/                    everything that asserts something
    test_voice_enrollment.py   unit tests for the voice_enrollment package
    test_find_sort_live.py     live finder + sorter on known corpus errors (2 calls)
    test_openai_live.py, test_gemini_live.py   one live finder call per provider
    smoketest_openai_offline.py   request/schema smoke test for every prompt, no live API call

data/                     inputs the scripts above read
    sample_transcript_assemblyai_v3.json
    pipeline_result.json       last run's output (regenerated by run_pipeline_live.py)

demos/                     static HTML, not part of the Python package
    index.html, lomb_demo_live.html, lomb_demo_source.html

archive/                  retired files, kept for reference, not part of the package structure above
```

## Running things

From the repo root (`lomb_prompts/`), not from inside a subfolder — the imports above assume repo root is on `sys.path`, which happens automatically for anything under `scripts/` and `tests/` via each file's own bootstrap:

```
python pipeline/pipeline.py          # offline wiring self-test (FakeProvider)
python tests/smoketest_openai_offline.py
python scripts/run_pipeline_live.py
python scripts/run_dev_server.py
python scripts/run_e2e_widget_test.py
python tests/test_voice_enrollment.py
```
