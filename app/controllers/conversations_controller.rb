class ConversationsController < ApplicationController
  before_action :set_conversation, only: %i[show destroy]

  def index
    @conversations = Conversation.order(updated_at: :desc).limit(100)
    @conversation = nil
  end

  def show
    @conversations = Conversation.order(updated_at: :desc).limit(100)
    @messages = @conversation.ordered_messages
    @streaming_id = params[:stream].to_i if params[:stream].present?
  end

  # Landing-page composer: creates the conversation plus the first
  # user/assistant message pair, then hands off to #show for streaming.
  def create
    content = params[:content].to_s.strip
    if content.empty?
      redirect_to root_path, alert: "Type a message first."
      return
    end

    @conversation = Conversation.create!(title: Conversation.title_for(content))
    assistant = add_turn!(@conversation, content)
    redirect_to conversation_path(@conversation, stream: assistant.id)
  end

  def destroy
    @conversation.destroy!
    redirect_to root_path, notice: "Conversation deleted."
  end

  private

  def set_conversation
    @conversation = Conversation.find(params[:id])
  end
end
