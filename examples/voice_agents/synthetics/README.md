# Synthetics - Pixel Agent Example

This folder contains the **Pixel** agent example, demonstrating an expressive, heart-forward voice agent for synthetic speech.

## pixel.py

- **Relocated** from `examples/voice_agents/pixel.py` into this `synthetics` folder.
- **Removed** unused imports (`RunContext`, `function_tool`) for clarity.
- **Renamed** logger from `basic-agent` to `pixel-agent` to reflect its unique identity.
- **Simplified** the LLM streaming node:
  - Strips out `<think>` tags and injects a natural phrase for `</think>`.
  - Eliminated an unnecessary loop that previously handled emotive tags.
- **Updated** dotenv loading to explicitly use `.env.pixel` for separate configuration.
- **Cleaned up** formatting and commented optional noise-cancellation import.

## Usage

1. Copy your environment variables into `.env.pixel` at the root, matching the examples in the main project.
2. Install dependencies as specified in `examples/voice_agents/requirements.txt`.
3. Run:
   ```bash
   cd examples/voice_agents/synthetics
   python pixel.py
   ```

Enjoy your expressive Pixel voice agent!