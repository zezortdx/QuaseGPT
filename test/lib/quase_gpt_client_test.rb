require "test_helper"

# Unit tests for the inference client. No real server is contacted: a tiny
# TCP stub stands in for ml/server.py, plus an unroutable port proves the
# unavailable path. Never touches a real checkpoint.
class QuaseGptClientTest < ActiveSupport::TestCase
  def stub_runtime(responses)
    server = TCPServer.new("127.0.0.1", 0)
    port = server.addr[1]
    thread = Thread.new do
      responses.each do |status, body|
        sock = server.accept
        request_line = sock.gets
        len = 0
        while (line = sock.gets) && line != "\r\n"
          len = Regexp.last_match(1).to_i if line =~ /Content-Length:\s*(\d+)/i
        end
        body_len = request_line&.start_with?("GET") ? 0 : len
        sock.read(body_len) if body_len.positive?
        payload = body.is_a?(String) ? body : JSON.generate(body)
        sock.write("HTTP/1.1 #{status}\r\nContent-Type: application/json\r\n" \
                   "Content-Length: #{payload.bytesize}\r\nConnection: close\r\n\r\n#{payload}")
        sock.close
      end
      server.close
    end
    [ port, thread ]
  end

  test "health reports loaded runtime" do
    port, th = stub_runtime([ [ "200 OK", { status: "ok", model_loaded: true, device: "cpu", error: nil } ] ])
    h = QuaseGptClient.health(base_url: "http://127.0.0.1:#{port}")
    assert_equal true, h[:loaded]
    assert_equal "cpu", h[:device]
    th.join(5)
  end

  test "health never raises when runtime is down" do
    h = QuaseGptClient.health(base_url: "http://127.0.0.1:1")
    assert_equal false, h[:loaded]
    assert h[:error].present?
  end

  test "generate returns text" do
    port, th = stub_runtime([ [ "200 OK", { text: "hello there", generated_tokens: 2 } ] ])
    r = QuaseGptClient.new(base_url: "http://127.0.0.1:#{port}")
                        .generate(prompt: "hi", max_new_tokens: 2, temperature: 0.0, top_k: nil)
    assert_equal "hello there", r.text
    assert_equal 2, r.generated_tokens
    th.join(5)
  end

  test "generate raises Unavailable on 503" do
    port, th = stub_runtime([ [ "503 Service Unavailable", { detail: "model not loaded" } ] ])
    assert_raises(QuaseGptClient::Unavailable) do
      QuaseGptClient.new(base_url: "http://127.0.0.1:#{port}").generate(prompt: "hi")
    end
    th.join(5)
  end

  test "generate raises Unavailable when server is down" do
    assert_raises(QuaseGptClient::Unavailable) do
      QuaseGptClient.new(base_url: "http://127.0.0.1:1").generate(prompt: "hi")
    end
  end

  test "generate raises BadResponse on malformed payload" do
    port, th = stub_runtime([ [ "200 OK", { nope: 1 } ] ])
    assert_raises(QuaseGptClient::BadResponse) do
      QuaseGptClient.new(base_url: "http://127.0.0.1:#{port}").generate(prompt: "hi")
    end
    th.join(5)
  end

  test "stream yields deltas" do
    sse = "data: {\"delta\": \"hel\"}\n\ndata: {\"delta\": \"lo\"}\n\ndata: {\"done\": true}\n\n"
    server = TCPServer.new("127.0.0.1", 0)
    port = server.addr[1]
    th = Thread.new do
      sock = server.accept
      sock.gets
      len = 0
      while (line = sock.gets) && line != "\r\n"
        len = Regexp.last_match(1).to_i if line =~ /Content-Length:\s*(\d+)/i
      end
      sock.read(len) if len.positive?
      sock.write("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n" \
                 "Content-Length: #{sse.bytesize}\r\nConnection: close\r\n\r\n#{sse}")
      sock.close
      server.close
    end
    got = []
    n = QuaseGptClient.new(base_url: "http://127.0.0.1:#{port}")
                       .stream(prompt: "hi", max_new_tokens: 4) { |d| got << d }
    assert_equal %w[hel lo], got
    assert_equal 2, n
    th.join(5)
  end

  test "stream raises GenerationFailed on error event" do
    sse = "data: {\"error\": \"boom\"}\n\n"
    server = TCPServer.new("127.0.0.1", 0)
    port = server.addr[1]
    th = Thread.new do
      sock = server.accept
      sock.gets
      len = 0
      while (line = sock.gets) && line != "\r\n"
        len = Regexp.last_match(1).to_i if line =~ /Content-Length:\s*(\d+)/i
      end
      sock.read(len) if len.positive?
      sock.write("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n" \
                 "Content-Length: #{sse.bytesize}\r\nConnection: close\r\n\r\n#{sse}")
      sock.close
      server.close
    end
    assert_raises(QuaseGptClient::GenerationFailed) do
      QuaseGptClient.new(base_url: "http://127.0.0.1:#{port}").stream(prompt: "hi") { |_| }
    end
    th.join(5)
  end
end
