# Turns a raw transcript into the shapes pipeline/ actually consumes.
#
# assemblyai_adapter.py -- raw AssemblyAI JSON -> Turn/Word objects.
# speaker_filter.py     -- filters Turns down to the confirmed target
#                           speaker, then reshapes them into per-metric
#                           input windows (sentences, sentence windows,
#                           formulaic-match candidates).
#
# Everything downstream in pipeline/ assumes it's already been handed
# target-speaker-only input; guaranteeing that is this package's one job.
