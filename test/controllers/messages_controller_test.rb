require "test_helper"

# Controller tests swap the inference runtime for fakes at the
# MessagesController.inference_client_class seam: no server, checkpoint,
# or GPU is needed to exercise Rails behavior.
class MessagesControllerTest < ActionDispatch::IntegrationTest
  FakeResult = Struct.new(:text, :generated_tokens)

  class FakeOk
    def generate(prompt:, **)
      MessagesControllerTest::FakeResult.new("local reply", 2)
    end

    def stream(prompt:, **)
      yield "hel"
      yield "lo"
      2
    end
  end

  class FakeDown
    def generate(prompt:, **)
      raise QuaseGptClient::Unavailable, "down"
    end

    def stream(prompt:, **)
      raise QuaseGptClient::Unavailable, "down"
    end
  end

  setup do
    @conv = Conversation.create!(title: "t")
    @previous_client = MessagesController.inference_client_class
  end

  teardown do
    MessagesController.inference_client_class = @previous_client
  end

  test "create persists turn and redirects to streaming show" do
    MessagesController.inference_client_class = FakeOk
    assert_difference("Message.count", 2) do
      post conversation_messages_path(@conv), params: { content: "hello" }
    end
    assistant = @conv.messages.order(:id).last
    assert_equal "streaming", assistant.status
    assert_redirected_to conversation_path(@conv, stream: assistant.id)
  end

  test "create rejects blank content" do
    assert_no_difference("Message.count") do
      post conversation_messages_path(@conv), params: { content: "" }
    end
    assert_redirected_to conversation_path(@conv)
  end

  test "create rejects oversized content" do
    assert_no_difference("Message.count") do
      post conversation_messages_path(@conv), params: { content: "x" * 9000 }
    end
    assert_redirected_to conversation_path(@conv)
  end

  test "sync create saves runtime text on success" do
    MessagesController.inference_client_class = FakeOk
    post conversation_messages_path(@conv), params: { content: "hi", sync: "1" }
    assistant = @conv.messages.order(:id).last
    assert_equal "complete", assistant.status
    assert_equal "local reply", assistant.content
    assert_redirected_to conversation_path(@conv)
  end

  test "sync create marks failed without crashing when runtime is down" do
    MessagesController.inference_client_class = FakeDown
    post conversation_messages_path(@conv), params: { content: "hi", sync: "1" }
    assistant = @conv.messages.order(:id).last
    assert_equal "failed", assistant.status
    assert_redirected_to conversation_path(@conv)
  end

  test "retry regenerates without duplicating the user message" do
    MessagesController.inference_client_class = FakeOk
    user = @conv.messages.create!(role: "user", content: "hi", status: "complete")
    assistant = @conv.messages.create!(role: "assistant", content: "", status: "failed")
    assert_no_difference("Message.count") do
      post conversation_messages_path(@conv), params: { retry_assistant_id: assistant.id, sync: "1" }
    end
    assert_equal "local reply", assistant.reload.content
    assert_equal "complete", assistant.status
    assert_equal user.id, @conv.messages.where(role: "user").sole.id
  end

  test "stream proxies deltas and persists final text" do
    MessagesController.inference_client_class = FakeOk
    @conv.messages.create!(role: "user", content: "hi", status: "complete")
    assistant = @conv.messages.create!(role: "assistant", content: "", status: "streaming")
    get stream_conversation_path(@conv, assistant_id: assistant.id)
    assert_response :success
    assert_equal "text/event-stream", response.media_type
    assert_includes response.body, "hel"
    assert_includes response.body, '"done":true'
    assistant.reload
    assert_equal "complete", assistant.status
    assert_equal "hello", assistant.content
  end

  test "stream marks failed gracefully when runtime is down" do
    MessagesController.inference_client_class = FakeDown
    @conv.messages.create!(role: "user", content: "hi", status: "complete")
    assistant = @conv.messages.create!(role: "assistant", content: "", status: "streaming")
    get stream_conversation_path(@conv, assistant_id: assistant.id)
    assert_response :success
    assert_includes response.body, "error"
    assert_equal "failed", assistant.reload.status
  end
end
