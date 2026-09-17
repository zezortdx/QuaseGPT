class ApplicationController < ActionController::Base
  # Only allow modern browsers supporting webp images, web push, badges, import maps, CSS nesting, and CSS :has.
  allow_browser versions: :modern

  # Changes to the importmap will invalidate the etag for HTML responses
  stale_when_importmap_changes

  private

  # Persists one chat turn: the user message plus an empty assistant
  # placeholder in "streaming" state. Never writes fake assistant text —
  # the placeholder is filled by MessagesController from the real runtime.
  def add_turn!(conversation, content)
    assistant = nil
    ActiveRecord::Base.transaction do
      conversation.messages.create!(role: "user", content: content, status: "complete")
      assistant = conversation.messages.create!(role: "assistant", content: "", status: "streaming")
      conversation.update!(title: Conversation.title_for(content)) if conversation.messages.where(role: "user").count == 1
      conversation.touch
    end
    assistant
  end
end
