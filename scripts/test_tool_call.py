from anthropic import Anthropic
from langfuse import get_client, observe

from customer_support_agent import config
from customer_support_agent.tools import ECHO_TOOL, EchoArgs, echo


@observe()
def main() -> None:
    langfuse = get_client()
    client = Anthropic(
        api_key=config.ANTHROPIC_API_KEY,
        base_url=config.ANTHROPIC_BASE_URL,
    )

    user_message = "Use the echo tool to echo back the phrase 'hello agent'."
    langfuse.update_current_span(input={"user_message": user_message})

    first_response = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=config.ANTHROPIC_MAX_TOKENS,
        temperature=config.ANTHROPIC_TEMPERATURE,
        tools=[ECHO_TOOL],
        messages=[{"role": "user", "content": user_message}],
    )

    print(f"Turn 1 stop_reason: {first_response.stop_reason}")
    print(f"Turn 1 content blocks: {[b.type for b in first_response.content]}")

    tool_use_blocks = [b for b in first_response.content if b.type == "tool_use"]
    if len(tool_use_blocks) != 1:
        raise RuntimeError(
            f"Expected exactly one tool_use block, got {len(tool_use_blocks)}"
        )
    tool_use = tool_use_blocks[0]
    print(f"Tool called: {tool_use.name}")
    print(f"Tool input (raw): {tool_use.input}")

    args = EchoArgs.model_validate(tool_use.input)
    result = echo(args)
    print(f"Tool result: {result.model_dump_json()}")

    second_response = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=config.ANTHROPIC_MAX_TOKENS,
        temperature=config.ANTHROPIC_TEMPERATURE,
        tools=[ECHO_TOOL],
        messages=[
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": first_response.content},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": result.model_dump_json(),
                    }
                ],
            },
        ],
    )

    print(f"\nTurn 2 stop_reason: {second_response.stop_reason}")
    print(f"Turn 2 content blocks: {[b.type for b in second_response.content]}")

    text_blocks = [b.text for b in second_response.content if b.type == "text"]
    final_answer = "".join(text_blocks)
    print(f"\nFinal answer:\n{final_answer}")

    langfuse.update_current_span(output={"final_answer": final_answer})


if __name__ == "__main__":
    main()
    get_client().flush()
