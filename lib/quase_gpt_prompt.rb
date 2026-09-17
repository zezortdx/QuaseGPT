# Chat adapter for a BASE language model (not instruction-tuned).
#
# Mirrors ml/quasegpt/prompt.py: the UI collects chat messages, but QuaseGPT
# only completes text, so this module isolates the "User:/Assistant:" template
# in one replaceable place. Chatbot behavior depends on how QuaseGPT was
# trained — a base model continues text plausibly rather than following
# instructions.
#
# Context budgeting: Ruby cannot run the real BPE tokenizer (it lives in
# ml/quasegpt/tokenizer.py), so the client truncates with a conservative
# ~4-chars-per-token estimate. The runtime ALWAYS enforces the exact limit by
# cropping to block_size tokens before generating, so this estimate can only
# under-fill, never overflow, the context.
module QuaseGptPrompt
  CHARS_PER_TOKEN_ESTIMATE = 4
  RESERVE_TOKENS = 16
  DEFAULT_BLOCK_SIZE = 256
  DEFAULT_MAX_NEW_TOKENS = 128

  module_function

  def format(messages)
    parts = messages.map do |m|
      role = m[:role] || m["role"]
      content = (m[:content] || m["content"]).to_s.strip
      case role.to_s
      when "user" then "User: #{content}"
      when "assistant" then "Assistant: #{content}"
      else content
      end
    end
    "#{parts.join("\n")}\nAssistant: "
  end

  # Newest-first truncation so the latest user message is always kept.
  def build_context(messages, block_size: DEFAULT_BLOCK_SIZE, max_new_tokens: DEFAULT_MAX_NEW_TOKENS)
    budget_chars = [ (block_size - max_new_tokens - RESERVE_TOKENS), 16 ].max * CHARS_PER_TOKEN_ESTIMATE
    kept = [] # newest-first while building
    messages.reverse_each do |m|
      trial = (kept + [ m ]).reverse # chronological candidate
      break if format(trial).length > budget_chars && kept.any?
      kept << m
    end
    ordered = kept.reverse # chronological
    ordered = [ messages.last ].compact if ordered.empty? && messages.any?
    text = format(ordered)
    if text.length > budget_chars
      text = text[-budget_chars, budget_chars]
      text = "…#{text}"
      text += "\nAssistant: " unless text.include?("Assistant:")
    end
    text
  end

  def title_for(first_user_text, limit: 50)
    Conversation.title_for(first_user_text, limit: limit)
  end
end
