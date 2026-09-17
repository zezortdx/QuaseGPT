class MessagesController < ApplicationController
  include ActionController::Live

  MAX_CONTENT_CHARS = 8000
  MAX_NEW_TOKENS = 128

  # Sealed for tests: test/lib fakes swap this without touching the network.
  class_attribute :inference_client_class, default: QuaseGptClient

  # POST /conversations/:conversation_id/messages
  # Persists the user message + an assistant placeholder, then:
  # - Turbo Stream / HTML: redirect to #show with ?stream= so the
  #   Stimulus controller pulls tokens live from #stream.
  # - sync=1 (no-JS fallback and tests): blocks, generates once, saves.
  def create
    conversation = Conversation.find(params[:conversation_id])

    # Retry a failed turn without duplicating the user message.
    if params[:retry_assistant_id].present?
      assistant = conversation.messages.find(params[:retry_assistant_id])
      assistant.update!(status: "streaming")
      if params[:sync].present?
        run_sync(assistant)
        redirect_to conversation_path(conversation)
      else
        redirect_to conversation_path(conversation, stream: assistant.id)
      end
      return
    end

    content = params[:content].to_s.strip
    if content.empty?
      redirect_to conversation_path(conversation), alert: "Type a message first."
      return
    end
    if content.length > MAX_CONTENT_CHARS
      redirect_to conversation_path(conversation), alert: "Message too long (max #{MAX_CONTENT_CHARS} chars)."
      return
    end

    assistant = add_turn!(conversation, content)

    if params[:sync].present?
      run_sync(assistant)
      redirect_to conversation_path(conversation)
    else
      redirect_to conversation_path(conversation, stream: assistant.id)
    end
  end

  # GET /conversations/:id/stream?assistant_id=123  (Server-Sent Events)
  # Proxies the local QuaseGPT runtime token-by-token, persists the final
  # text on the placeholder message, and never leaks internals.
  def stream
    conversation = Conversation.find(params[:id])
    assistant = conversation.messages.find(params[:assistant_id])

    response.headers["Content-Type"] = "text/event-stream"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    # Allow the fetch() EventSource from the same origin to read the stream.
    response.headers["Last-Modified"] = Time.now.httpdate

    begin
      prompt = conversation_prompt(conversation, assistant)
      full = +""
      client.stream(prompt: prompt, max_new_tokens: MAX_NEW_TOKENS) do |delta|
        full << delta
        sse_write(delta: delta)
      end
      assistant.update!(content: full, status: "complete")
      conversation.touch
      sse_write(done: true)
    rescue QuaseGptClient::Error => e
      Rails.logger.warn("[quasegpt] stream failed: #{e.class}: #{e.message}")
      assistant.update!(status: "failed", content: failure_note) if assistant.status != "complete"
      sse_write(error: "The local model is unavailable right now. Retry in a moment.")
    rescue IOError, ActionController::Live::ClientDisconnected
      # Browser navigated away: keep whatever streamed so far as failed
      # rather than persisting a silent truncation as complete.
      assistant.update!(status: "failed", content: failure_note) if assistant.status != "complete"
    ensure
      response.stream.close
    end
  end

  private

  def client
    @client ||= self.class.inference_client_class.new
  end

  # Blocking single-shot generation (no-JS fallback / tests).
  def run_sync(assistant)
    prompt = conversation_prompt(assistant.conversation, assistant)
    result = client.generate(prompt: prompt, max_new_tokens: MAX_NEW_TOKENS)
    assistant.update!(content: result.text, status: "complete")
    assistant.conversation.touch
  rescue QuaseGptClient::Error => e
    Rails.logger.warn("[quasegpt] sync generate failed: #{e.class}: #{e.message}")
    assistant.update!(status: "failed", content: failure_note)
  end

  # Prompt from everything before the placeholder: only persisted,
  # finished turns. The empty streaming row itself is excluded.
  def conversation_prompt(conversation, assistant)
    prior = conversation.messages.where("id < ?", assistant.id).order(:id)
    history = prior.map { |m| { role: m.role, content: m.content } }
    QuaseGptPrompt.build_context(history, block_size: runtime_block_size,
                                          max_new_tokens: MAX_NEW_TOKENS)
  end

  def runtime_block_size
    Integer(ENV.fetch("QUASEGPT_BLOCK_SIZE", QuaseGptPrompt::DEFAULT_BLOCK_SIZE))
  rescue ArgumentError
    QuaseGptPrompt::DEFAULT_BLOCK_SIZE
  end

  def failure_note
    "The local QuaseGPT runtime is unavailable. Make sure it is running " \
      "(bin/dev) and retry — your message above was kept."
  end

  def sse_write(obj)
    response.stream.write("data: #{JSON.generate(obj)}\n\n")
  end
end
