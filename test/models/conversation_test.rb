require "test_helper"

class ConversationTest < ActiveSupport::TestCase
  test "title_for trims to first user message" do
    assert_equal "Tell me a story about dragons", Conversation.title_for("Tell me a story about dragons")
  end

  test "title_for truncates long titles at word boundary" do
    long = ("word " * 100).strip
    title = Conversation.title_for(long)
    assert title.length <= 55
    assert title.end_with?("…")
  end

  test "title_for falls back for blank input" do
    assert_equal "New chat", Conversation.title_for("")
    assert_equal "New chat", Conversation.title_for(nil)
  end

  test "destroys messages with conversation" do
    conv = Conversation.create!(title: "t")
    conv.messages.create!(role: "user", content: "hi", status: "complete")
    assert_difference("Message.count", -1) { conv.destroy! }
  end

  test "ordered_messages returns chronological order" do
    conv = Conversation.create!(title: "t")
    b = conv.messages.create!(role: "user", content: "b", status: "complete")
    a = conv.messages.create!(role: "assistant", content: "a", status: "complete")
    assert_equal [ b.id, a.id ], conv.ordered_messages.map(&:id)
  end
end
