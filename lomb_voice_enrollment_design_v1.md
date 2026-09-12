# Voice Enrollment — Design v1

*Answers `lomb_prd_v1.md` Section 9, Open Question 1 ("Speaker enrollment via voiceprint... is the likely path for scalable speaker ID"). Extends, does not replace, `lomb_backend_prd_v1.md` Section 6.2's manual speaker-picker flow. Companion implementation: `voice_enrollment.py`, `speechbrain_embedder.py`, `test_voice_enrollment.py`. UX flow diagrammed at [Voice Enrollment Flow](https://claude.ai/code/artifact/a8c4fc8c-d139-47ce-9180-e0dd0a06a651).*

## What changed vs. Section 6.2

Section 6.2 committed to: diarize → show every detected speaker's clip → visitor clicks which one is them → every single upload. Correct for a first-time/anonymous visitor; there's nothing to match against yet.

This adds a layer underneath it: once a user has an enrolled voiceprint, `POST /analyze` can auto-resolve which detected speaker is them and skip the picker entirely. The picker doesn't go away — it's the fallback for a low-confidence match or a wrong-guess correction, and every one of those fallback confirmations reinforces the stored voiceprint.

## Two ways a voiceprint gets created

**Passive (`enroll()`)** — the default, zero-extra-friction path. The very first upload always shows the picker anyway (Section 6.2's existing, unavoidable requirement — there's nothing to compare against yet), so the click that picker already forces becomes the enrollment sample. No new screen. Every later confirmation, whether a fresh picker click or an unconfirmed pass-through after a confident auto-match, reinforces the same voiceprint via a running mean, each worth `sample_count += 1`.

**Active (`enroll_active()`)** — a dedicated onboarding step: the visitor records a short clean sample directly (read a passage aloud, or just talk for a few seconds) before ever uploading a real session. Two real differences from passive, not just a different call site:
- No diarization, no speaker label — the clip is single-speaker by construction, so it embeds directly.
- Seeded with `initial_weight=4` (configurable) instead of the `1` a passive confirmation gets, since a purpose-recorded clip is typically cleaner than a slice of a real tutoring call and earns more say in the running mean before ordinary sessions start diluting it. A head start, not a lock — real `enroll()` confirmations still move it over time.

The payoff active enrollment adds that passive can't: **the visitor's first real upload can skip the picker too.** Passive enrollment can't do this by construction — there's no voiceprint until after the first click. Both paths write into the exact same store, and `resolve()` doesn't know or care which one produced the voiceprint it's matching against — it's the same 33-line decision function either way.

`enroll_active()` also doubles as the "reset my voice profile" affordance flagged as a gap in the first draft of this doc: it refuses to overwrite an existing voiceprint unless called with `force=True`, so a visitor-initiated reset from account settings is just `force=True` re-enrollment, not a separate delete endpoint.

## How resolution decides

Two independent gates, both must pass, or it falls back to the manual picker (never a silent guess), regardless of which enrollment path produced the stored voiceprint:

- **Match threshold** — absolute cosine similarity to the stored voiceprint must be high (default **0.75**).
- **Margin threshold** — the gap between best and second-best candidate must be large enough that it isn't a coin flip (default **0.10**).

This two-gate design exists because of a real failure found in Dan's own personal corpus: `natasja_italki6_4_14_2026` was hand-labeled with high confidence using a Q&A-direction heuristic, and it was backwards — a pure "closest match wins" rule would have made the same mistake if the runner-up was nearly as close. Absolute similarity alone isn't enough; the margin catches the coin-flip case the similarity score alone can't.

**Where 0.75 / 0.10 actually come from — read before changing them.** They are a deliberately conservative placeholder, not a calibrated result. Checked against the literature and it doesn't converge on one number: SpeechBrain's own `SpeakerRecognition.verify_batch()` ships a default decision threshold of **0.25** on this same raw-cosine score (per a maintainer's comment, [speechbrain/speechbrain#2148](https://github.com/speechbrain/speechbrain/discussions/2148)) — but that's the *balanced* operating point (~EER) on VoxCeleb1-test-cleaned: curated, largely-English, single-utterance studio recordings, not German-language tutoring calls over videocall compression with real conversational turn-taking. A separate community thread on the model's own [HuggingFace page](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/discussions/7) cites 0.7–0.75 as typical, but "scaled to [0,1]" — possibly a different convention than the raw cosine score used here. The two sources don't reconcile cleanly, and neither is a validated stand-in for this domain.

Rather than block shipping on a proper calibration (needs a labeled batch of real uploads that doesn't exist yet), the thresholds are set well above SpeechBrain's own balanced default — deliberately trading recall for precision, because the two failure directions aren't symmetric: too high just means more visitors see the picker than strictly necessary (mildly annoying, self-correcting on click); too low risks a confident, silent, wrong auto-resolution — `natasja_italki6`'s failure mode, now automated. **When in doubt, fail toward the picker.**

Getting a real number later is cheap, not a re-architecture: every `SpeakerResolution` already carries the full ranked candidate list with raw similarity scores, whichever way a resolution went. Production just needs to persist that struct somewhere queryable (a log line is enough to start) — once a few hundred real resolutions have accumulated, plot similarity for confirmed-correct vs. confirmed-wrong auto-matches (caught via "Switch speaker" corrections) and pick the threshold that actually separates them, the same way `identify_target_speaker.py` already does offline for Dan's personal corpus.

Embedding model: SpeechBrain ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`), the same family pyannote's own clustering uses internally, and the same model `identify_target_speaker.py` uses for the personal-corpus diagnostic — kept identical so findings transfer between the two rather than diverging.

Voiceprints are stored as an incremental spherical mean (running sum of unit-normalized embeddings + a sample count), not a frozen first sample or a plain arithmetic mean — the voiceprint keeps adapting session over session (new mic, a cold, background noise) instead of anchoring to whatever the first enrollment happened to sound like.

## The dependency this introduces (not yet resolved)

A stored voiceprint has to be keyed to *something* that persists across uploads — that requires the persistent-identity/data-store decision `lomb_prd_v1.md` Section 8 currently marks unscoped. This design doesn't resolve that; `VoiceprintRepository` is an interface specifically so `user_id` can be a real account id, an email-gate token (Section 10 already assigns something identity-shaped for the beta), or a browser-local anonymous id — whichever the rest of the backend lands on. The one non-negotiable: some notion of "the same visitor came back" has to exist before enrollment pays off at all.

## Storage

`SQLiteVoiceprintRepository` ships as the default — zero added infra, works on the single Fly.io/Cloud Run instance already chosen in `lomb_build_roadmap.md` 2.7, plenty for the beta learner counts Stage 3.3 describes (3–5 learners). `VoiceprintRepository` is the seam to swap in Postgres+pgvector later if learner count or multi-instance deployment makes SQLite's single-file-lock model an actual bottleneck — not before.

## Open items before this ships

1. **Persistent identity model** (Section 8, above) — blocks enrollment from paying off at all, not just this module's storage choice.
2. **Threshold calibration** — 0.75 / 0.10 are deliberately conservative placeholders (see "How resolution decides" above), not validated against a labeled dataset. No new plumbing is needed to fix this later — `SpeakerResolution.candidates` already carries what a future calibration pass needs; production just has to log it.
3. **API contract addition** — `POST /analyze`'s response needs an `auto_resolved_speaker_id` / `confidence` field so the frontend knows whether to skip the picker; this is additive to Section 4/6.2's existing contract, not a breaking change.
4. **Active-enrollment surface: built as a dev slice, not yet wired to real identity.** `audio_decode.py` (blob → 16kHz mono float32 via ffmpeg), `enroll_api.py` (`POST /enroll`, `GET /status`, `GET /` via a `create_app(embedder, repository)` factory), and `enroll_widget.html` (`getUserMedia`/`MediaRecorder`, an 8s-minimum timer surfaced before upload, a force/reset checkbox) now exist and have been verified with a genuine end-to-end test (`run_e2e_widget_test.py`) — real Chromium with a fake mic device, driven by Playwright, recording real audio through the actual widget and hitting a real running `uvicorn` server: first enrollment, refused re-enrollment without `force`, and a clean `force=True` reset all confirmed via `/status`, not just unit-tested in isolation. What's still missing before this is real: `user_id` is hardcoded to `"1"` (`USER_ID_FOR_NOW` in `enroll_api.py`, pending Section 8's identity decision above), the embedder used in the e2e test is a deterministic fake (no torch/speechbrain in this sandbox — production wiring needs `speechbrain_embedder.EcapaSpeakerEmbedder` via a real `main.py`), and the Section 6.2 picker UI itself still doesn't exist.
5. **Embedding portability** — if the embedding model ever changes (e.g. a future move to pyannoteAI's paid `/voiceprint` API), every stored voiceprint needs re-enrollment; embeddings from different models aren't comparable.
