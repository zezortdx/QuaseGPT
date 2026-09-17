# Client for the local QuaseGPT inference runtime (ml/server.py).
#
# Only generation parameters are ever sent. Model paths come from trusted app
# config (ENV), never from user input. No shell/Python execution, no file
# paths, no checkpoint selection from the web.
#
# All failure modes (server down, timeout, malformed response, generation
# failure, interrupted stream) raise QuaseGptClient::Error so controllers can
# mark the assistant message failed instead of crashing the app.
require "json"
require "net/http"
require "uri"

class QuaseGptClient
  class Error < StandardError; end
  class Unavailable < Error; end # connection refused / timeout / 503
  class BadResponse < Error; end # malformed payload
  class GenerationFailed < Error; end # 500 from runtime

  DEFAULT_URL = "http://127.0.0.1:8000".freeze
  OPEN_TIMEOUT = 3
  READ_TIMEOUT = 120
  HEALTH_TIMEOUT = 2

  GenerateResult = Struct.new(:text, :generated_tokens, keyword_init: true)

  def initialize(base_url: ENV.fetch("QUASEGPT_INFERENCE_URL", DEFAULT_URL))
    @base_url = base_url
  end

  # Quick liveness probe for the UI badge. Never raises.
  def self.health(base_url: ENV.fetch("QUASEGPT_INFERENCE_URL", DEFAULT_URL))
    uri = URI.join(base_url + "/", "health")
    res = Net::HTTP.start(uri.host, uri.port, open_timeout: HEALTH_TIMEOUT,
                                                 read_timeout: HEALTH_TIMEOUT) do |http|
      http.get(uri.request_uri)
    end
    body = JSON.parse(res.body)
    { loaded: body["model_loaded"] == true, device: body["device"], error: body["error"] }
  rescue StandardError => e
    { loaded: false, device: nil, error: e.message }
  end

  def generate(prompt:, max_new_tokens: 128, temperature: 0.8, top_k: 40, top_p: nil, seed: nil)
    payload = { prompt: prompt, max_new_tokens: max_new_tokens,
                temperature: temperature, top_k: top_k, top_p: top_p, seed: seed }.compact
    res = post_json("/generate", payload)
    case res
    when Net::HTTPSuccess
      body = parse_json(res.body)
      text = body["text"]
      raise BadResponse, "missing 'text' in runtime response" unless text.is_a?(String)
      GenerateResult.new(text: text, generated_tokens: body["generated_tokens"].to_i)
    when Net::HTTPServiceUnavailable
      raise Unavailable, detail(res)
    when Net::HTTPUnprocessableEntity
      raise BadResponse, detail(res)
    else
      raise GenerationFailed, detail(res)
    end
  end

  # Streams incremental text deltas via the runtime's SSE endpoint.
  # Yields each delta String; returns the number of deltas seen.
  # Raises the same Error subclasses as #generate (mid-stream runtime errors
  # reported as {"error": ...} events raise GenerationFailed).
  def stream(prompt:, max_new_tokens: 128, temperature: 0.8, top_k: 40, top_p: nil, seed: nil)
    raise ArgumentError, "block required" unless block_given?
    payload = { prompt: prompt, max_new_tokens: max_new_tokens,
                temperature: temperature, top_k: top_k, top_p: top_p, seed: seed }.compact
    uri = URI.join(@base_url + "/", "generate/stream")
    count = 0
    begin
      Net::HTTP.start(uri.host, uri.port, open_timeout: OPEN_TIMEOUT, read_timeout: READ_TIMEOUT) do |http|
        req = Net::HTTP::Post.new(uri.request_uri, "Content-Type" => "application/json")
        req.body = JSON.generate(payload)
        http.request(req) do |res|
          case res
          when Net::HTTPSuccess
            buffer = +""
            res.read_body do |chunk|
              buffer << chunk
              while (line = extract_sse_line(buffer))
                event = parse_sse_data(line)
                next if event.nil?
                raise GenerationFailed, event["error"] if event["error"]
                next unless event["done"] || event["delta"]
                if event["delta"]
                  count += 1
                  yield event["delta"]
                end
              end
            end
          when Net::HTTPServiceUnavailable
            raise Unavailable, detail(res)
          when Net::HTTPUnprocessableEntity
            raise BadResponse, detail(res)
          else
            raise GenerationFailed, detail(res)
          end
        end
      end
    rescue QuaseGptClient::Error
      raise
    rescue Net::OpenTimeout, Net::ReadTimeout, Errno::ECONNREFUSED, Errno::EHOSTUNREACH, SocketError => e
      raise Unavailable, "inference runtime unreachable (#{e.class}: #{e.message})"
    end
    count
  end

  private

  def post_json(path, payload)
    uri = URI.join(@base_url + "/", path.sub(%r{\A/}, ""))
    Net::HTTP.start(uri.host, uri.port, open_timeout: OPEN_TIMEOUT, read_timeout: READ_TIMEOUT) do |http|
      req = Net::HTTP::Post.new(uri.request_uri, "Content-Type" => "application/json")
      req.body = JSON.generate(payload)
      http.request(req)
    end
  rescue Net::OpenTimeout, Net::ReadTimeout, Errno::ECONNREFUSED, Errno::EHOSTUNREACH, SocketError => e
    raise Unavailable, "inference runtime unreachable (#{e.class}: #{e.message})"
  end

  def parse_json(body)
    JSON.parse(body)
  rescue JSON::ParserError => e
    raise BadResponse, "malformed JSON from runtime: #{e.message}"
  end

  def detail(res)
    body = res.body.to_s[0, 300]
    "runtime responded #{res.code}: #{body}"
  end

  # Pulls one complete "data: {...}" line off the buffer, or nil.
  def extract_sse_line(buffer)
    idx = buffer.index("\n")
    return nil if idx.nil?
    line = buffer.slice!(0, idx + 1)
    line.strip!
    return extract_sse_line(buffer) if line.empty? || line.start_with?(":")
    line
  end

  def parse_sse_data(line)
    return nil unless line.start_with?("data:")
    parse_json(line.sub(/\Adata:\s*/, ""))
  rescue BadResponse
    nil # ignore a garbled event line; the terminal done/error event still decides
  end
end
