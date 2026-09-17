// Mobile drawer sidebar: toggles the conversation panel on narrow screens.
import { Controller } from "@hotwired/stimulus"

export default class extends Controller {
  static targets = ["panel", "scrim"]

  toggle() {
    const open = this.panelTarget.classList.toggle("open")
    this.scrimTarget.hidden = !open
  }

  close() {
    this.panelTarget.classList.remove("open")
    this.scrimTarget.hidden = true
  }
}
