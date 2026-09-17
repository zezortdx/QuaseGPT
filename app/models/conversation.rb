class Conversation < ApplicationRecord
  has_many :messages, -> { order(:id) }, dependent: :destroy, inverse_of: :conversation

  # Deterministic title from the first user message. No LLM involved
  # (mirrors ml/quasegpt/prompt.py conversation_title).
  def self.title_for(first_user_text, limit: 50)
    t = first_user_text.to_s.split.join(" ")
    return "New chat" if t.empty?
    return t if t.length <= limit
    cut = t[0, limit].rpartition(" ").first
    cut = t[0, limit] if cut.empty?
    "#{cut}…"
  end

  def ordered_messages
    messages.order(:id)
  end
end
