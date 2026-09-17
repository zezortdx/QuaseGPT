class Message < ApplicationRecord
  belongs_to :conversation, inverse_of: :messages

  ROLES = %w[user assistant].freeze
  STATUSES = %w[pending streaming complete failed].freeze

  validates :role, presence: true, inclusion: { in: ROLES }
  validates :status, presence: true, inclusion: { in: STATUSES }
  validates :content, presence: true, if: :user?

  # User messages must carry real text; assistant placeholders may start empty
  # while streaming and are filled in when the run finishes.
  def user?
    role == "user"
  end

  def assistant?
    role == "assistant"
  end
end
