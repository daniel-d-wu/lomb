# One file per LLM provider, plus the shared interface they implement.
#
# llm_provider.py    -- LLMProvider, the abstract interface: classify(config,
#                        input_data) -> the same answer shape regardless of
#                        which provider answered.
# gemini_provider.py -- the real, working one.
# openai_provider.py -- an unverified reference implementation proving the
#                        same metric content can target a second provider --
#                        see its own docstring.
#
# Switching providers means writing one new file here, not touching
# pipeline/registry.py or any file in prompts/.
