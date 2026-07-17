/* =============================================================================
   conversation.js  —  INHALTSDATEI DES INTERVIEWS
   -----------------------------------------------------------------------------
   Das ist die "andere Datei", aus der index.html die Inhalte lädt.
   Zum Testen des Layouts steht hier Beispieltext.

   >>> HIER ÄNDERST DU ALS FORSCHER <<<
   - Dein bestehendes Python-System kann diese Datei einfach neu schreiben
     (z. B. window.CONVERSATION als JSON hineinschreiben), oder du füllst
     "messages" live per fetch()/WebSocket. Solange am Ende window.CONVERSATION
     gesetzt ist, zeigt die Oberfläche den Inhalt an.
   - Ist diese Datei nicht vorhanden, benutzt index.html einen eingebauten
     Not-Beispieltext, damit das Layout trotzdem testbar bleibt.
============================================================================= */

window.CONVERSATION = {

  /* --- Kopfzeile der Klinikerin/des Kliniker-Agenten ---------------------- */
  clinicianName: "Dr. Maren Vos",
  clinicianRole: "Klinisches Interview · Aufnahme",

  /* --- FAKTOR 1: EMBODIMENT ---------------------------------------------- */
  /* Erlaubt: "logo" | "picture" | "video" | "avatar"                        */
  embodiment: "picture",

  /* Quellen für Bild/Video (nur relevant bei "picture" bzw. "video").       */
  /* Leer lassen -> es wird ein Platzhalter angezeigt.                        */
  pictureSrc: "",
  videoSrc:   "",

  /* --- FAKTOR 2: MODALITÄT ----------------------------------------------- */
  /* "text"   -> Chat-Blasen + Texteingabe                                   */
  /* "speech" -> nur Sprach-Oberfläche, KEIN Text für die teilnehmende Person */
  modality: "text",

  /* --- GESPRÄCHSVERLAUF --------------------------------------------------- */
  /* role: "clinician" = Frage/Antwort des LLM, "user" = Antwort der Person   */
  /* Dein System hängt hier einfach neue Einträge an.                         */
  messages: [
    { role: "clinician", text: "Danke, dass Sie sich heute Zeit nehmen. Zu Beginn: Wie würden Sie Ihre Stimmung in den letzten zwei Wochen beschreiben?" },
    { role: "user",      text: "Ehrlich gesagt ziemlich flach. Das Aufstehen war der schwierigste Teil." },
    { role: "clinician", text: "Das klingt belastend. Wenn Sie sagen flach – kommt und geht das, oder ist es die meiste Zeit des Tages da?" }
  ],

  /* --- Zeigt die Klinikerin gerade "tippt …" an? (Denk-Punkte) ----------- */
  clinicianTyping: true
};

async function sendChatMessage(text) {
  const trimmed = (text || "").trim();
  if (!trimmed) {
    return;
  }

  window.CONVERSATION.messages.push({ role: "user", text: trimmed });
  window.CONVERSATION.clinicianTyping = true;
  render();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: trimmed }),
    });

    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({}));
      window.CONVERSATION.messages.push({
        role: "clinician",
        text: errorBody.error || `Server error ${response.status}`,
      });
    } else {
      const result = await response.json();
      window.CONVERSATION.messages.push({
        role: "clinician",
        text: result.response || "No response from server.",
      });
    }
  } catch (error) {
    window.CONVERSATION.messages.push({
      role: "clinician",
      text: `Network error: ${error.message}`,
    });
  } finally {
    window.CONVERSATION.clinicianTyping = false;
    render();
  }
}

function setupTextInput() {
  const input = document.querySelector(".field");
  const send = document.querySelector(".send");
  if (!input || !send) {
    return;
  }

  send.addEventListener("click", function () {
    sendChatMessage(input.value);
    input.value = "";
    input.focus();
  });

  input.addEventListener("keydown", function (event) {
    if (event.key === "Enter") {
      event.preventDefault();
      sendChatMessage(input.value);
      input.value = "";
    }
  });
}

