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
  modality: "speech",

  /* --- START-OVERLAY ------------------------------------------------------ */
  /* Wird angezeigt, bis "Interview starten" geklickt wird. Text frei anpassbar. */
  startTitle:    "Bereit fürs Interview?",
  startSubtitle: "Bitte prüfen Sie kurz, ob die folgenden Voraussetzungen erfüllt sind. Sobald Sie starten, beginnt das Gespräch mit einer ersten Frage.",
  startButton:   "Interview starten",

  /* Beschriftungen für den optionalen Ton-/Mikrofon-Test im Start-Overlay.
     Der Test selbst blockiert "Interview starten" nicht — reines Angebot. */
  startSoundTestLabel: "Testton abspielen",
  startMicTestLabel:   "Mikrofon testen",

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

  /* Nur Sprach-Modalität: die Eröffnungsnachricht bereits als Ton, satzweise
     vertont, im Hintergrund geladen (siehe loadInterviewStart() unten) —
     damit sie sofort abgespielt werden kann, sobald "Interview starten"
     geklickt wird, statt erst dann mit der Vertonung zu beginnen. */
  pendingOpeningAudioPieces: null,

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

      // Echte Lautstärke fürs Balken-Feedback in renderSpeech() — dieselbe
      // Berechnung wie beim Mikrofon-Test im Start-Overlay, hier aus dem
      // ohnehin schon laufenden Aufnahme-Signal, ohne eigenen AnalyserNode.
      let sumSquares = 0;
      for (let i = 0; i < input.length; i++) {
        sumSquares += input[i] * input[i];
      }
      const rms = Math.sqrt(sumSquares / input.length);
      if (typeof window.updateListeningLevel === "function") {
        window.updateListeningLevel(Math.min(1, rms * 4));
      }
    };

    source.connect(recorderNode);
    recorderNode.connect(audioContext.destination);

    window.CONVERSATION.userRecording = true;
    window.CONVERSATION.clinicianSpeaking = false;
    render();
  } catch (err) {
    window.CONVERSATION.userRecording = false;
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

  window.CONVERSATION.userRecording = false;
  window.CONVERSATION.clinicianTyping = true;
  window.CONVERSATION.processingResponse = true;

  // Platzhalter für die kommende Antwort — der Text wird NICHT live
  // angezeigt (Sprach-Modalität zeigt nur den Gesprächszustand, siehe
  // renderSpeech() in index.html), aber trotzdem mitgeschrieben, für den
  // späteren Verlauf ("Zurück zum Interview"-Ansicht). Die Nutzer-Antwort
  // selbst wird erst in readStreamedReply() eingetragen, sobald das
  // tatsächliche (server-seitige, verlässliche) Transkript da ist.
  const replyMsg = { role: "clinician", text: "" };
  window.CONVERSATION.messages.push(replyMsg);
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

    await readStreamedReply(response, replyMsg);
  } catch (error) {
    window.CONVERSATION.processingResponse = false;
    window.CONVERSATION.clinicianTyping = false;
    window.CONVERSATION.clinicianSpeaking = false;
    replyMsg.text = `Fehler: ${error.message}`;
    render();
  }
}

/* =============================================================================
   GEMEINSAMER AUDIO-PLAYER MIT ECHTER AMPLITUDE
   -----------------------------------------------------------------------------
   Wird sowohl für gestreamte Antworten als auch für die gesprochene
   Eröffnung genutzt. EIN AudioContext pro Antwort (nicht pro Satz!), ein
   AnalyserNode liest die tatsächliche Lautstärke der gerade spielenden
   Tonspur aus — window.updateSpeakingLevel() (siehe index.html) macht
   daraus die "spricht gerade"-Bögen am Lautsprecher-Symbol. KEINE
   vorgetäuschte Animation: Wenn kein Ton kommt, bewegt sich auch nichts.

   Wichtig beim Verdrahten: createMediaElementSource() lenkt die Wiedergabe
   eines <audio>-Elements exklusiv durch den Web-Audio-Graphen — vergisst
   man, den AnalyserNode am Ende wieder mit audioContext.destination zu
   verbinden, hört man plötzlich gar nichts mehr. Deshalb hier bewusst an
   einer einzigen Stelle gebaut und getestet, statt an mehreren Stellen
   dieselbe Verdrahtung zu wiederholen.

   Fällt Web Audio aus (nicht verfügbar, Browser verweigert o. Ä.), wird
   ganz normal ohne Amplituden-Anzeige weitergespielt — nie stiller Absturz.
============================================================================= */
function createSpokenAudioPlayer() {
  let audioContext = null;
  let analyser = null;
  let dataArray = null;
  let usingAnalyser = false;
  let rafId = null;

  try {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    analyser.connect(audioContext.destination); // <- ohne das: kein Ton mehr!
    dataArray = new Uint8Array(analyser.frequencyBinCount);
    usingAnalyser = true;
  } catch (err) {
    console.warn("Web Audio nicht verfügbar — Sprachausgabe ohne Amplituden-Anzeige.", err);
  }

  function startLevelLoop() {
    if (!usingAnalyser) {
      return;
    }
    function tick() {
      analyser.getByteTimeDomainData(dataArray);
      let sumSquares = 0;
      for (let i = 0; i < dataArray.length; i++) {
        const v = (dataArray[i] - 128) / 128;
        sumSquares += v * v;
      }
      const rms = Math.sqrt(sumSquares / dataArray.length);
      if (typeof window.updateSpeakingLevel === "function") {
        window.updateSpeakingLevel(Math.min(1, rms * 4));
      }
      rafId = requestAnimationFrame(tick);
    }
    tick();
  }

  function stopLevelLoop() {
    if (rafId) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }
    if (typeof window.updateSpeakingLevel === "function") {
      window.updateSpeakingLevel(0);
    }
  }

  function playPiece(piece, onDone) {
    const audioData = atob(piece.audio);
    const bytes = new Uint8Array(audioData.length);
    for (let i = 0; i < audioData.length; i++) {
      bytes[i] = audioData.charCodeAt(i);
    }
    const blob = new Blob([bytes], { type: "audio/wav" });
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);

    if (usingAnalyser) {
      try {
        const source = audioContext.createMediaElementSource(audio);
        source.connect(analyser);
      } catch (err) {
        // Seltene Browser-Eigenheiten — Wiedergabe funktioniert trotzdem
        // normal, nur ohne Amplituden-Anzeige für dieses eine Häppchen.
        console.warn("Amplituden-Anbindung für dieses Audio-Stück fehlgeschlagen.", err);
      }
    }

    audio.addEventListener("play", () => {
      window.CONVERSATION.clinicianTyping = false;
      window.CONVERSATION.processingResponse = false;
      window.CONVERSATION.clinicianSpeaking = true;
      render();
      startLevelLoop();
    });

    const finish = () => {
      stopLevelLoop();
      URL.revokeObjectURL(url);
      onDone();
    };
    audio.addEventListener("ended", finish);
    audio.addEventListener("error", finish);
    audio.play().catch(finish);
  }

  function close() {
    stopLevelLoop();
    if (audioContext) {
      audioContext.close().catch(function () {});
      audioContext = null;
    }
  }

  return { playPiece: playPiece, close: close };
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
async function readStreamedReply(response, msg) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const player = createSpokenAudioPlayer();
  const audioQueue = [];
  let playing = false;
  let assembledText = "";

  function maybePlayNext() {
    if (playing || audioQueue.length === 0) {
      return;
    }
    playing = true;
    const piece = audioQueue.shift();
    player.playPiece(piece, function () {
      assembledText = assembledText ? assembledText + " " + piece.text : piece.text;
      playing = false;
      maybePlayNext();
    });
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
        // Die Nutzer-Antwort wird jetzt erst eingetragen — mit dem echten,
        // server-seitigen Transkript, nicht der groben Browser-Vorschau.
        // Muss VOR der Kliniker-Platzhalter-Nachricht (msg) stehen, die
        // schon vorher angelegt wurde, damit die Reihenfolge stimmt.
        const messages = window.CONVERSATION.messages;
        const insertIndex = messages.indexOf(msg);
        messages.splice(insertIndex, 0, { role: "user", text: data.transcript || "" });
      } else if (data.type === "sentence") {
        audioQueue.push({ text: data.text, audio: data.audio });
        maybePlayNext();
      } else if (data.type === "error") {
        window.CONVERSATION.clinicianTyping = false;
        window.CONVERSATION.processingResponse = false;
        if (!assembledText) {
          assembledText = `Fehler: ${data.message}`;
        }
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

  // Text wird erst jetzt, am Ende, ins Nachrichten-Objekt geschrieben — er
  // wurde ja nie live angezeigt, sondern nur fürs spätere Verlauf-Feature
  // mitgeschrieben ("Zurück zum Interview"-Ansicht).
  player.close();
  msg.text = assembledText;
  window.CONVERSATION.clinicianSpeaking = false;
  window.CONVERSATION.clinicianTyping = false;
  window.CONVERSATION.processingResponse = false;
  window.CONVERSATION.liveTranscript = "";
  window.CONVERSATION.turnDone = true; // Verlauf-Button darf erscheinen (folgt als nächster Schritt)
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
   sobald der Button gedrückt wird. In der Sprach-Modalität wird zusätzlich
   die bereits vertonte Tonspur dazu geladen (siehe playOpeningSpoken() unten),
   in der Text-Modalität die Wort-für-Wort-Anzeige (siehe playOpening() in
   index.html). Beides kommt aus der aktiven Leitfaden-Datei in
   study_content/interviews/ — hier im Frontend steht dazu absichtlich kein
   Inhalt, nur die Abfrage.
============================================================================= */
async function loadInterviewStart() {
  try {
    const res = await fetch("/api/interview/start");
    if (!res.ok) {
      throw new Error(`Server antwortete mit Status ${res.status}`);
    }
    const data = await res.json();
    window.CONVERSATION.pendingOpeningMessage = data.message;

    if (window.CONVERSATION.modality === "speech") {
      const audioRes = await fetch("/api/interview/start/audio");
      if (audioRes.ok) {
        const audioData = await audioRes.json();
        window.CONVERSATION.pendingOpeningAudioPieces = audioData.pieces || [];
      }
    }
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

/* =============================================================================
   ERÖFFNUNGSNACHRICHT HÖRBAR MACHEN (Sprach-Modalität)
   -----------------------------------------------------------------------------
   Die erste Frage muss auch in der Sprach-Bedingung tatsächlich gesprochen
   werden, bevor die teilnehmende Person antworten kann — genau wie jede
   spätere Antwort auch. Die Tonspur dafür wurde schon im Hintergrund
   geladen (siehe loadInterviewStart() oben), während das Start-Overlay noch
   sichtbar war, liegt hier also i. d. R. schon fertig vor. Nutzt denselben
   Player wie die späteren Antworten, also auch dieselbe echte
   Amplituden-Anzeige — keine zwei unterschiedlichen Verhalten für
   "LLM spricht".
============================================================================= */
function playOpeningSpoken() {
  const pieces = window.CONVERSATION.pendingOpeningAudioPieces || [];

  // Sperrt das Mikrofon, bis die Eröffnung fertig gesprochen ist — die
  // teilnehmende Person soll erst antworten können, nachdem sie die erste
  // Frage tatsächlich gehört hat.
  window.CONVERSATION.clinicianTyping = false;
  window.CONVERSATION.processingResponse = true;
  render();

  // Text wird nicht live gezeigt (Sprach-Modalität), aber fürs spätere
  // Verlauf-Feature mitgeschrieben — genau wie bei jeder anderen Antwort.
  const fullText = pieces.length
    ? pieces.map(function (p) { return p.text; }).join(" ")
    : (window.CONVERSATION.pendingOpeningMessage || "");
  window.CONVERSATION.messages.push({ role: "clinician", text: fullText });

  if (pieces.length === 0) {
    // Kein Ton verfügbar (z. B. Server nicht erreichbar) — nicht ewig
    // blockieren, einfach normal freigeben.
    window.CONVERSATION.processingResponse = false;
    render();
    return;
  }

  const player = createSpokenAudioPlayer();
  let index = 0;

  function playNext() {
    if (index >= pieces.length) {
      player.close();
      window.CONVERSATION.clinicianSpeaking = false;
      window.CONVERSATION.processingResponse = false;
      render();
      return;
    }
    const piece = pieces[index];
    index += 1;
    player.playPiece(piece, playNext);
  }

  playNext();
}

window.playOpeningSpoken = playOpeningSpoken;

