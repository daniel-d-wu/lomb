<#
Removes the OLD flat-layout copies of files that now live inside the new
lomb_prompts/ subfolders (pipeline/, prompts/, providers/, reporting/,
transcript_processing/, voice_enrollment/, scripts/, tests/, data/, demos/).

Run this ONLY after confirming the new subfolders actually contain the
files (open a couple and check), so you don't end up with neither copy if
something went wrong.

Deliberately does NOT touch:
  - prompts/  and  archive/   (already subfolders before the reorg; left as-is)
  - voiceprints_dev.sqlite    (see the warning printed below -- not auto-deleted)
  - .git, __pycache__, .ipynb_checkpoints (handled by cleanup_safe.ps1 already)

Usage:
    powershell -ExecutionPolicy Bypass -File cleanup_old_flat_files.ps1
#>

$root = "C:\Users\Danie\Desktop\lomb_sandbox\lomb_prompts"
Set-Location $root

$oldFlatFiles = @(
    "pipeline.py",
    "registry.py",
    "direct_computation.py",
    "metric_types.py",
    "prefilter.py",
    "llm_provider.py",
    "gemini_provider.py",
    "openai_provider.py",
    "speaker_filter.py",
    "assemblyai_adapter.py",
    "audio_decode.py",
    "enroll_api.py",
    "enroll_widget.html",
    "identify_target_speaker.py",
    "speechbrain_embedder.py",
    "voice_enrollment.py",
    "test_voice_enrollment.py",
    "report.py",
    "run_pipeline_live.py",
    "run_dev_server.py",
    "run_e2e_widget_test.py",
    "build_sample_transcript_v3.py",
    "build_source_doc.py",
    "list_gemini_models.py",
    "smoketest_openai_offline.py",
    "test_all_metrics_live.py",
    "test_gemini_live.py",
    "test_openai_live.py",
    "pipeline_result.json",
    "sample_transcript_assemblyai_v3.json",
    "index.html",
    "lomb_demo_live.html",
    "lomb_demo_source.html",
    "lomb_report_standalone.html",
    "report.html",
    "report_preview_new_format.html",
    "report_contract.json"
)

Write-Host "Removing old flat-layout files now superseded by the subfolders..."
foreach ($f in $oldFlatFiles) {
    $path = Join-Path $root $f
    if (Test-Path $path) {
        Remove-Item $path -Force
        Write-Host "  removed: $f"
    }
}

Write-Host ""
Write-Host "NOT touched -- voiceprints_dev.sqlite:"
Write-Host "  The old copy is still at $root\voiceprints_dev.sqlite"
Write-Host "  A new copy now also lives at $root\scripts\voiceprints_dev.sqlite"
Write-Host "  If you enrolled any voices through the widget more recently than"
Write-Host "  this reorg, the root copy may be newer -- compare the two"
Write-Host "  ('dir' both, check 'Last write time') before deleting either one."
Write-Host ""
Write-Host "Done. Remaining top-level items should now just be the new"
Write-Host "subfolders, README.md, prompts/, archive/, and .git."
