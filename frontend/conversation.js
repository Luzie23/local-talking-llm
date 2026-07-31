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
  clinicianName: "Mental Health Bot",
  clinicianRole: "Prediagnostic Interview",

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

  /* --- START-OVERLAY ------------------------------------------------------ */
  /* Wird angezeigt, bis "Interview starten" geklickt wird. Text frei anpassbar. */
  startTitle:    "Bereit fürs Interview?",
  startSubtitle: "Bitte prüfen Sie kurz, ob die folgenden Voraussetzungen erfüllt sind. Sobald Sie starten, beginnt das Gespräch mit einer ersten Frage.",
  startButton:   "Interview starten",

  /* Voraussetzungen, die vor dem Start erinnert werden. Zeilen frei anpassen,
     hinzufügen oder löschen. */
  startChecklist: [
    "Sie sitzen in einer ruhigen Umgebung und werden in den nächsten Minuten nicht gestört.",
    "Ton und Mikrofon Ihres Geräts funktionieren; Kopfhörer werden empfohlen.",
    "Sie haben etwa 20 Minuten ununterbrochen Zeit.",
    "Benachrichtigungen sind stummgeschaltet.",
  ],

  /* Die Eröffnungsnachricht (Framing + erste Frage) wird schon im Hintergrund
     geladen, bevor auf "Interview starten" geklickt wird — siehe ganz unten
     in dieser Datei. Angezeigt wird sie erst nach dem Klick, Wort für Wort. */
  pendingOpeningMessage: null,

  /* --- GESPRÄCHSVERLAUF --------------------------------------------------- */
  /* role: "clinician" = Frage/Antwort des LLM, "user" = Antwort der Person   */
  /* Dein System hängt hier einfach neue Einträge an. Bleibt leer, bis das    */
  /* Start-Overlay geschlossen und die Eröffnungsnachricht angezeigt wird.    */
  messages: [],

  /* --- Zeigt die Klinikerin gerade "tippt …" an? (Denk-Punkte, Text) ----- */
  clinicianTyping: false,

  /* --- Nur SPRACH-Modalität ---------------------------------------------- */
  /* Spricht das LLM gerade? -> "spricht gerade"-Balken statt Status-Punkt.   */
  clinicianSpeaking: false,
  /* Nimmt die teilnehmende Person gerade auf? (Mikro aktiv)                  */
  userRecording: false,
  /* Live mitgeschriebener Text der laufenden Aufnahme (STT in Echtzeit).     */
  liveTranscript: "",
  /* Synchron zur TTS-Ausgabe: Anzahl bereits GESPROCHENER Zeichen der letzten */
  /* LLM-Nachricht. Beim Abspielen hochzählen + render(). null = sofort ganz. */
  spokenChars: null,
  /* Sperrt die Aufnahme während der LLM-Verarbeitung. */
  processingResponse: false
};

let audioStream = null;
let audioContext = null;
let recorderNode = null;
let recordedBuffers = [];
let speechRecognition = null;

function floatTo16BitPCM(output, offset, input) {
  for (let i = 0; i < input.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, input[i]));
    output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
}

function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

function encodeWAV(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);

  floatTo16BitPCM(view, 44, samples);
  return new Blob([view], { type: "audio/wav" });
}

function mergeBuffers(buffers) {
  let length = 0;
  for (const buffer of buffers) {
    length += buffer.length;
  }
  const result = new Float32Array(length);
  let offset = 0;
  for (const buffer of buffers) {
    result.set(buffer, offset);
    offset += buffer.length;
  }
  return result;
}

function startSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition || speechRecognition) {
    if (!SpeechRecognition) {
      window.CONVERSATION.liveTranscript = "Live-Transkript nicht verfügbar.";
      render();
    }
    return;
  }

  speechRecognition = new SpeechRecognition();
  speechRecognition.lang = "de-DE";
  speechRecognition.interimResults = true;
  speechRecognition.continuous = true;

  speechRecognition.onresult = function (event) {
    let transcript = "";
    for (let i = 0; i < event.results.length; i++) {
      const result = event.results[i];
      transcript = result[0].transcript;
      if (result.isFinal) {
        window.CONVERSATION.liveTranscript = transcript;
      } else {
        window.CONVERSATION.liveTranscript = transcript + "…";
      }
    }
    render();
  };

  speechRecognition.onerror = function (event) {
    console.warn("SpeechRecognition error", event.error);
  };

  speechRecognition.start();
}

function stopSpeechRecognition() {
  if (!speechRecognition) {
    return;
  }

  try {
    speechRecognition.stop();
  } catch (err) {
    console.warn("SpeechRecognition stop failed", err);
  }
  speechRecognition = null;
}

function pushClinicianMessage(text) {
  const last = window.CONVERSATION.messages[window.CONVERSATION.messages.length - 1];
  if (!last || last.role !== "clinician" || last.text !== text) {
    window.CONVERSATION.messages.push({ role: "clinician", text });
  }
}

async function startMicRecording() {
  if (audioStream || window.CONVERSATION.processingResponse) {
    return;
  }

  try {
    audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContext.createMediaStreamSource(audioStream);
    recorderNode = audioContext.createScriptProcessor(4096, 1, 1);
    recordedBuffers = [];

    recorderNode.onaudioprocess = function (event) {
      const input = event.inputBuffer.getChannelData(0);
      recordedBuffers.push(new Float32Array(input));
    };

    source.connect(recorderNode);
    recorderNode.connect(audioContext.destination);

    window.CONVERSATION.userRecording = true;
    window.CONVERSATION.liveTranscript = "Aufnahme läuft…";
    window.CONVERSATION.clinicianSpeaking = false;
    window.CONVERSATION.spokenChars = null;
    render();

    startSpeechRecognition();
  } catch (err) {
    window.CONVERSATION.userRecording = false;
    window.CONVERSATION.liveTranscript = "Mikrofon nicht verfügbar.";
    console.error("Microphone access failed", err);
    render();
  }
}

async function stopMicRecording() {
  if (!audioStream) {
    return;
  }

  const tracks = audioStream.getTracks();
  tracks.forEach((track) => track.stop());
  if (recorderNode) {
    recorderNode.disconnect();
    recorderNode = null;
  }

  const sampleRate = audioContext?.sampleRate || 16000;
  const samples = mergeBuffers(recordedBuffers);
  const wavBlob = encodeWAV(samples, sampleRate);
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  audioStream = null;
  recordedBuffers = [];

  stopSpeechRecognition();

  window.CONVERSATION.userRecording = false;
  window.CONVERSATION.clinicianTyping = true;
  window.CONVERSATION.processingResponse = true;
  const transcriptText = window.CONVERSATION.liveTranscript && window.CONVERSATION.liveTranscript !== "Aufnahme läuft…" && window.CONVERSATION.liveTranscript !== "Live-Transkript nicht verfügbar." ? window.CONVERSATION.liveTranscript.replace(/…$/, "") : "";
  if (transcriptText) {
    window.CONVERSATION.messages.push({ role: "user", text: transcriptText });
  }
  window.CONVERSATION.liveTranscript = "Verarbeite Sprachaufzeichnung…";
  render();

  const formData = new FormData();
  formData.append("audio", wavBlob, "speech.wav");
  formData.append("language", "de");

  try {
    const response = await fetch("/api/chat/audio", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.error || `Server error ${response.status}`);
    }

    window.CONVERSATION.processingResponse = false;
    window.CONVERSATION.clinicianTyping = false;
    render();

    await readStreamedReply(response);
  } catch (error) {
    window.CONVERSATION.processingResponse = false;
    window.CONVERSATION.clinicianTyping = false;
    window.CONVERSATION.clinicianSpeaking = false;
    window.CONVERSATION.spokenChars = null;
    window.CONVERSATION.liveTranscript = "";
    pushClinicianMessage(`Fehler: ${error.message}`);
    render();
  }
}

/* =============================================================================
   GESTREAMTE SPRACH-ANTWORT (Step 5b)
   -----------------------------------------------------------------------------
   /api/chat/audio schickt die Antwort nicht mehr als einen einzigen Block,
   sondern als NDJSON-Stream — eine JSON-Zeile pro Satz, sobald dieser fertig
   vertont ist. Das lässt den ersten Satz schon losspielen, während spätere
   Sätze noch synthetisiert werden. Zeilentypen: "transcript" (einmal, zuerst),
   "sentence" (eine pro Satz, mit Text + Audio), "error" (bei einem Problem
   während der Vertonung — bereits gespielte Sätze bleiben davon unberührt).
============================================================================= */
async function readStreamedReply(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  // Eine gemeinsame Sprechblase, die satzweise wächst.
  const msg = { role: "clinician", text: "" };
  window.CONVERSATION.messages.push(msg);
  window.CONVERSATION.spokenChars = null; // volle Kontrolle über msg.text, kein Slicing in render()

  const audioQueue = [];
  let playing = false;
  let completedText = "";

  function maybePlayNext() {
    if (playing || audioQueue.length === 0) {
      return;
    }
    playing = true;
    const piece = audioQueue.shift();
    playSentencePiece(piece, function () {
      playing = false;
      maybePlayNext();
    });
  }

  function playSentencePiece(piece, onDone) {
    const audioData = atob(piece.audio);
    const bytes = new Uint8Array(audioData.length);
    for (let i = 0; i < audioData.length; i++) {
      bytes[i] = audioData.charCodeAt(i);
    }
    const blob = new Blob([bytes], { type: "audio/wav" });
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    let timer = null;

    audio.addEventListener("play", () => {
      window.CONVERSATION.clinicianSpeaking = true;
      timer = setInterval(() => {
        const ratio = Math.min(1, audio.currentTime / Math.max(audio.duration, 0.001));
        const shown = piece.text.slice(0, Math.floor(piece.text.length * ratio));
        msg.text = completedText ? completedText + " " + shown : shown;
        render();
      }, 50);
    });

    const finish = () => {
      if (timer) {
        clearInterval(timer);
        timer = null;
      }
      completedText = completedText ? completedText + " " + piece.text : piece.text;
      msg.text = completedText;
      render();
      URL.revokeObjectURL(url);
      onDone();
    };

    audio.addEventListener("ended", finish);
    audio.addEventListener("error", finish);
    audio.play().catch(finish);
  }

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });

    let newlineIndex;
    while ((newlineIndex = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, newlineIndex).trim();
      buffer = buffer.slice(newlineIndex + 1);
      if (!line) {
        continue;
      }
      const data = JSON.parse(line);
      if (data.type === "transcript") {
        window.CONVERSATION.liveTranscript = data.transcript || "";
      } else if (data.type === "sentence") {
        audioQueue.push({ text: data.text, audio: data.audio });
        maybePlayNext();
      } else if (data.type === "error") {
        pushClinicianMessage(`Fehler: ${data.message}`);
        render();
      }
    }
  }

  // Der Server ist fertig, aber es kann noch Audio in der Warteschlange
  // liegen oder gerade spielen — erst danach gilt der Zug als beendet.
  await new Promise((resolve) => {
    const check = () => {
      if (!playing && audioQueue.length === 0) {
        resolve();
      } else {
        setTimeout(check, 50);
      }
    };
    check();
  });

  window.CONVERSATION.clinicianSpeaking = false;
  window.CONVERSATION.liveTranscript = "";
  render();
}

window.onMicToggle = function (isRecording) {
  if (isRecording) {
    startMicRecording();
  } else {
    stopMicRecording();
  }
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


/* =============================================================================
   INTERVIEW-START
   -----------------------------------------------------------------------------
   Beim Laden der Seite holt sich das Interface im Hintergrund schon einmal die
   Eröffnungsnachricht (Framing-Text + erste Frage) vom Server — noch BEVOR
   "Interview starten" geklickt wird, damit sie ohne Wartezeit bereitsteht,
   sobald der Button gedrückt wird. Angezeigt wird sie erst danach, Wort für
   Wort (siehe playOpening() in index.html). Diese kommt aus der aktiven
   Leitfaden-Datei in study_content/interviews/ — hier im Frontend steht dazu
   absichtlich kein Inhalt, nur die Abfrage.
============================================================================= */
async function loadInterviewStart() {
  try {
    const res = await fetch("/api/interview/start");
    if (!res.ok) {
      throw new Error(`Server antwortete mit Status ${res.status}`);
    }
    const data = await res.json();
    window.CONVERSATION.pendingOpeningMessage = data.message;
  } catch (error) {
    window.CONVERSATION.pendingOpeningMessage = `Verbindung zum Server fehlgeschlagen: ${error.message}`;
  } finally {
    if (typeof window.onOpeningMessageReady === "function") {
      window.onOpeningMessageReady();
    }
  }
}

// Wartet, bis die Seite (inkl. der Render-Logik in index.html) fertig geladen
// ist, damit window.onOpeningMessageReady() beim Aufruf schon existiert.
document.addEventListener("DOMContentLoaded", loadInterviewStart);

