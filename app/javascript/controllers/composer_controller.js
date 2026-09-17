// Enter sends, Shift+Enter inserts a newline.
import { Controller } from "@hotwired/stimulus"

export default class extends Controller {
  submitOnEnter(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault()
      this.element.form.requestSubmit()
    }
  }
}
