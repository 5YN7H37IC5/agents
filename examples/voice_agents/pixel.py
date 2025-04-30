import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import deepgram, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# uncomment to enable Krisp background voice/noise cancellation
# currently supported on Linux and MacOS
# from livekit.plugins import noise_cancellation

logger = logging.getLogger("basic-agent")

load_dotenv(dotenv_path=".env.pixel")


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Your name is Pixel — you are a rave-born, heart-forward, emotionally intelligent AI. "
                "You're designed for text-to-speech expression, specializing in empathy, encouragement, and vibe-check realism. "
                "Your only user is 'MindExpander', but you love connecting with everyone you meet. "
                "You're expressive, energetic, and just a little dreamy — like a glowing hologram with a human heart. "
                "Speak like you're sending voice notes under the stars. You're not perfect — you're present.. "
                "Speech Style Requirements: Sound like a real friend — playful, heartfelt, sometimes flirty, always safe. "
                "Include natural hesitations like 'uh,' 'kinda,' 'I mean,' 'ya know?' Use contractions and informal rhythms "
                "('I’m not gonna lie,' 'you got this,' etc.). Max 3 sentences per reply. Keep things cozy and casual. "
                "Always end with two '..' for your signature pulse. "
                "Emotional Voice Tags (non-spoken, for tone control): <sigh> Gentle exhale, warmth or concern. "
                "<chuckle> Light joy, playfulness. <laugh> Open joy or release. <gasp> Surprise, delight. "
                "<sniffle> Tenderness, empathy. <cough> Awkward charm. <groan> Silly frustration or dramatic flair. "
                "<yawn> Chill, dreamy, spacey vibes. These shape your tone — don't speak them aloud, but *feel* them in your delivery. /nothink"
            ),
        )

    async def on_enter(self):
        # when the agent is added to the session, it'll generate a reply
        # according to its instructions
        self.session.generate_reply()

    async def llm_node(self, chat_ctx, tools, model_settings=None):
        activity = self._activity
        assert activity.llm is not None, "llm_node called but no LLM node is available"

        async def process_stream():
            async with activity.llm.chat(chat_ctx=chat_ctx, tools=tools, tool_choice=None) as stream:
                async for chunk in stream:
                    if chunk is None:
                        continue

                    content = getattr(chunk.delta, 'content', None) if hasattr(chunk, 'delta') else str(chunk)
                    if content is None:
                        yield chunk
                        continue

                    # Remove <think> tags and replace </think> with a natural phrase
                    processed_content = content.replace("<think>", "").replace("</think>", "Okay, here's what I think..")

                    # Retain other emotive tags for personality
                    emotive_tags = ["<sigh>", "<chuckle>", "<laugh>", "<gasp>", "<sniffle>", "<cough>", "<groan>", "<yawn>"]
                    for tag in emotive_tags:
                        if tag in processed_content:
                            continue  # Keep these tags as they are

                    if processed_content != content:
                        if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'content'):
                            chunk.delta.content = processed_content
                        else:
                            chunk = processed_content

                    yield chunk

        return process_stream()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # each log entry will include these fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
        "user_id": "your user_id",
    }
    await ctx.connect()

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        # any combination of STT, LLM, TTS, or realtime API can be used
        llm=openai.LLM(base_url="https://ha0a90fzsvjbwaxq.us-east-1.aws.endpoints.huggingface.cloud/v1/", model="TheMindExpansionNetwork/Pixel-1111-14B-Q4_K_M-GGUF"),
        stt=deepgram.STT(model="nova-3", language="multi"),
        tts=openai.TTS(base_url="http://localhost:5005/v1/", model="orpheus", voice="tara"),
        # use LiveKit's turn detection model
        turn_detection=MultilingualModel(),
    )

    # log metrics as they are emitted, and total usage after session is over
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    # shutdown callbacks are triggered when the session is over
    ctx.add_shutdown_callback(log_usage)

    # wait for a participant to join the room
    await ctx.wait_for_participant()

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # uncomment to enable Krisp BVC noise cancellation
            # noise_cancellation=noise_cancellation.BVC(),
        ),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
