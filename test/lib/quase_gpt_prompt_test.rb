require "test_helper"

class QuaseGptPromptTest < ActiveSupport::TestCase
  test "format wraps turns and ends with Assistant cue" do
    p = QuaseGptPrompt.format([ { role: "user", content: "Tell me a story" } ])
    assert_includes p, "User: Tell me a story"
    assert p.rstrip.end_with?("Assistant:")
  end

  test "format keeps both roles in order" do
    p = QuaseGptPrompt.format([
                                { role: "user", content: "hi" },
                                { role: "assistant", content: "hello" },
                                { role: "user", content: "more" }
                              ])
    assert p.index("User: hi") < p.index("Assistant: hello")
    assert p.index("Assistant: hello") < p.index("User: more")
  end

  test "build_context keeps the newest message under a tight budget" do
    msgs = (0...10).map { |i| { role: "user", content: "message number #{i} " * 10 } }
    ctx = QuaseGptPrompt.build_context(msgs, block_size: 128, max_new_tokens: 8)
    assert_includes ctx, "message number 9"
  end

  test "build_context always keeps at least the last message" do
    msgs = [ { role: "user", content: "hello " * 500 } ]
    ctx = QuaseGptPrompt.build_context(msgs, block_size: 64, max_new_tokens: 8)
    assert_includes ctx, "Assistant:"
  end
end
