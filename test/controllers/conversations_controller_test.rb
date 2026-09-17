require "test_helper"

class ConversationsControllerTest < ActionDispatch::IntegrationTest
  test "index renders welcome and composer" do
    get root_path
    assert_response :success
    assert_select "form.composer"
    assert_match "QuaseGPT", response.body
  end

  test "create persists conversation plus user and assistant messages" do
    assert_difference("Conversation.count", 1) do
      assert_difference("Message.count", 2) do
        post conversations_path, params: { content: "Once upon a time" }
      end
    end
    conv = Conversation.last
    assert_equal "Once upon a time", conv.title
    assert_redirected_to conversation_path(conv, stream: conv.messages.last.id)
    user, assistant = conv.ordered_messages
    assert_equal "user", user.role
    assert_equal "Once upon a time", user.content
    assert_equal "assistant", assistant.role
    assert_equal "streaming", assistant.status
    assert_equal "", assistant.content, "no fake assistant text is ever written"
  end

  test "create rejects blank content" do
    assert_no_difference([ "Conversation.count", "Message.count" ]) do
      post conversations_path, params: { content: "   " }
    end
    assert_redirected_to root_path
  end

  test "show lists persisted messages" do
    conv = Conversation.create!(title: "t")
    conv.messages.create!(role: "user", content: "hi", status: "complete")
    conv.messages.create!(role: "assistant", content: "hello", status: "complete")
    get conversation_path(conv)
    assert_response :success
    assert_match "hi", response.body
    assert_match "hello", response.body
  end

  test "destroy removes conversation and messages" do
    conv = Conversation.create!(title: "t")
    conv.messages.create!(role: "user", content: "hi", status: "complete")
    assert_difference("Conversation.count", -1) do
      assert_difference("Message.count", -1) do
        delete conversation_path(conv)
      end
    end
    assert_redirected_to root_path
  end
end
