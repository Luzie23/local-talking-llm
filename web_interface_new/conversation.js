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

  /* --- GESPRÄCHSVERLAUF --------------------------------------------------- */
  /* role: "clinician" = Frage/Antwort des LLM, "user" = Antwort der Person   */
  /* Dein System hängt hier einfach neue Einträge an.                         */
  messages: [
    { role: "clinician", text: "Danke, dass Sie sich heute Zeit nehmen. Zu Beginn: Wie würden Sie Ihre Stimmung in den letzten zwei Wochen beschreiben?" },
    { role: "user",      text: "Ehrlich gesagt ziemlich flach. Das Aufstehen war der schwierigste Teil." },
    { role: "clinician", text: "Das klingt belastend. Wenn Sie sagen flach – kommt und geht das, oder ist es die meiste Zeit des Tages da?" }
  ],

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
let playbackTimer = null;
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

function stopMicRecording() {
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

  fetch("/api/chat/audio", {
    method: "POST",
    body: formData,
  })
    .then(async (response) => {
          if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.error || `Server error ${response.status}`);
      }
      return response.json();
    })
    .then((result) => {
      if (!result.response) {
        throw new Error("Ungültige Antwort vom Server.");
      }

      window.CONVERSATION.processingResponse = false;
      window.CONVERSATION.clinicianTyping = false;
      window.CONVERSATION.liveTranscript = result.transcript || window.CONVERSATION.liveTranscript || "";
      window.CONVERSATION.clinicianSpeaking = true;
      window.CONVERSATION.spokenChars = 0;
      window.CONVERSATION.messages.push({ role: "clinician", text: result.response });
      render();

      playResponseAudio(result.audio, result.response);
    })
    .catch((error) => {
      window.CONVERSATION.processingResponse = false;
      window.CONVERSATION.clinicianTyping = false;
      window.CONVERSATION.clinicianSpeaking = false;
      window.CONVERSATION.spokenChars = null;
      window.CONVERSATION.liveTranscript = "";
      pushClinicianMessage(`Fehler: ${error.message}`);
      render();
    });
}

function playResponseAudio(base64Audio, responseText) {
  if (playbackTimer) {
    clearInterval(playbackTimer);
    playbackTimer = null;
  }

  const audioData = atob(base64Audio);
  const buffer = new Uint8Array(audioData.length);
  for (let i = 0; i < audioData.length; i++) {
    buffer[i] = audioData.charCodeAt(i);
  }
  const blob = new Blob([buffer], { type: "audio/wav" });
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);

  const fullLength = responseText.length;
  audio.addEventListener("play", () => {
    window.CONVERSATION.clinicianSpeaking = true;
    window.CONVERSATION.spokenChars = 0;
    render();

    playbackTimer = setInterval(() => {
      const ratio = Math.min(1, audio.currentTime / Math.max(audio.duration, 0.001));
      window.CONVERSATION.spokenChars = Math.floor(fullLength * ratio);
      render();
    }, 50);
  });

  audio.addEventListener("ended", () => {
    if (playbackTimer) {
      clearInterval(playbackTimer);
      playbackTimer = null;
    }
    window.CONVERSATION.clinicianSpeaking = false;
    window.CONVERSATION.spokenChars = fullLength;
    window.CONVERSATION.liveTranscript = "";
    render();
    URL.revokeObjectURL(url);
  });

  audio.addEventListener("error", () => {
    window.CONVERSATION.clinicianSpeaking = false;
    window.CONVERSATION.liveTranscript = "Audiowiedergabe fehlgeschlagen.";
    render();
    URL.revokeObjectURL(url);
  });

  audio.play().catch((error) => {
    console.error("Audio playback failed", error);
    window.CONVERSATION.clinicianSpeaking = false;
    window.CONVERSATION.liveTranscript = "Audiowiedergabe fehlgeschlagen.";
    render();
  });
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

