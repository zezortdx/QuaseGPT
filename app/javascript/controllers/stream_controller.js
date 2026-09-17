// Live token streaming over Server-Sent Events (plain fetch, no websockets).
//
// The server persists the final text; this controller only paints deltas
// into the assistant row as they arrive.
import { Controller } from "@hotwired/stimulus"

export default class extends Controller {
  static values = { url: String }
  static targets = ["output"]

  connect() {
    this.done = false
    this.reader = null
    this.start()
  }

  disconnect() {
    this.done = true
    if (this.reader) this.reader.cancel().catch(() => {})
  }

  async start() {
    const box = this.outputTarget
    box.textContent = ""
    try {
      const res = await fetch(this.urlValue, { headers: { Accept: "text/event-stream" } })
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`)
      this.reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ""
      for (;;) {
        const { value, done } = await this.reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        let idx
        while ((idx = buffer.indexOf("\n\n")) !== -1) {
          const chunk = buffer.slice(0, idx)
          buffer = buffer.slice(idx + 2)
          for (const line of chunk.split("\n")) {
            const t = line.trim()
            if (!t.startsWith("data:")) continue
            let event
            try {
              event = JSON.parse(t.slice(5).trim())
            } catch {
              continue
            }
            if (event.delta) box.textContent += event.delta
            if (event.error) {
              box.innerHTML = ""
              const p = document.createElement("p")
              p.className = "msg-failed"
              p.textContent = event.error
              box.appendChild(p)
            }
            if (event.done) {
              this.finish()
              return
            }
          }
        }
        if (this.done) return
      }
      this.finish()
    } catch (e) {
      if (!this.done) {
        box.innerHTML = ""
        const p = document.createElement("p")
        p.className = "msg-failed"
        p.textContent = "The connection dropped before the reply finished. Reload to retry."
        box.appendChild(p)
      }
    }
  }

  finish() {
    this.done = true
    const row = this.element.closest(".msg")
    if (row) row.classList.add("msg-done")
    const chat = document.getElementById("chat")
    if (chat) chat.scrollTop = chat.scrollHeight
  }
}
