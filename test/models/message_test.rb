require "test_helper"

class MessageTest < ActiveSupport::TestCase
  setup do
    @conv = Conversation.create!(title: "t")
  end

  test "valid user message" do
    m = @conv.messages.build(role: "user", content: "hello", status: "complete")
    assert m.valid?
  end

  test "user message requires content" do
    m = @conv.messages.build(role: "user", content: "  ", status: "complete")
    assert_not m.valid?
  end

  test "assistant placeholder may start empty while streaming" do
    m = @conv.messages.build(role: "assistant", content: "", status: "streaming")
    assert m.valid?
  end

  test "rejects unknown roles" do
    m = @conv.messages.build(role: "system", content: "x", status: "complete")
    assert_not m.valid?
  end

  test "rejects unknown statuses" do
    m = @conv.messages.build(role: "user", content: "x", status: "typing")
    assert_not m.valid?
  end

  test "rejects fake terminal states on create path roles" do
    %w[pending streaming complete failed].each do |s|
      m = @conv.messages.build(role: "assistant", content: "x", status: s)
      assert m.valid?, "expected status #{s} to be valid"
    end
  end
end
