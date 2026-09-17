module ApplicationHelper
  # Local-runtime badge. One fast probe per request; never raises, so the
  # UI stays usable when the inference server is down.
  def runtime_status
    @runtime_status ||= QuaseGptClient.health
  end
end
