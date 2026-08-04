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

  /* Bild-/Videodatei einfach in study_content/assets/ ablegen (dort auch die
     Herkunft/Quelle in SOURCES.md eintragen — wichtig für KI-generierte
     Bilder wie ein Logo, falls es später mal ersetzt werden soll), dann hier
     den Pfad "/study-assets/<Dateiname>" eintragen, z. B.:
       pictureSrc: "/study-assets/logo.png"
     Leer lassen -> es wird ein Platzhalter angezeigt. logoSrc ist ein
     EIGENES Feld (nicht dasselbe wie pictureSrc) — so lassen sich Logo- und
     Portrait-Bedingung unabhängig voneinander konfigurieren, ohne dass ein
     Wechsel der Bedingung versehentlich das falsche Bild zeigt.
     NOCH NICHT UMGESETZT: Diese Felder selbst liegen noch hier im
     Frontend-Code, nicht in study_content/ — das wäre ein größerer,
     separat geplanter Umbau (siehe Projektplan, Abschnitt „Offene Punkte"). */
  pictureSrc: "",
  videoSrc:   "",
  logoSrc:    "",

  /* --- FAKTOR 2: MODALITÄT ----------------------------------------------- */
  /* "text"   -> Chat-Blasen + Texteingabe                                   */
  /* "speech" -> nur Sprach-Oberfläche, KEIN Text für die teilnehmende Person */
  modality: "speech",

  /* --- START-OVERLAY ------------------------------------------------------ */
  /* Wird angezeigt, bis "Interview starten" geklickt wird. Text frei anpassbar. */
  startTitle:    "Bereit fürs Interview?",
  startSubtitle: "Bitte prüfen Sie kurz, ob die folgenden Voraussetzungen erfüllt sind. Sobald Sie starten, beginnt das Gespräch mit einer ersten Frage.",
  startButton:   "Interview starten",

  /* Steht auf dem Start-Button, solange er noch deaktiviert ist — das
     System lädt im Hintergrund das Sprachmodell und vertont die
     Eröffnungsnachricht schon einmal vorab. Das kann besonders beim ALLERERSTEN
     Aufruf (Modell noch nicht "warmgelaufen") einen Moment dauern. */
  startLoadingLabel: "Bereitet sich vor – einen Moment bitte …",

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
  processingResponse: false,
  /* Wird nach der ersten vollständig abgeschlossenen Runde true — schaltet
     den Verlauf-Button frei (siehe .convofoot in index.html), bleibt dann
     für den Rest der Sitzung true. */
  turnDone: false,
  /* Ist die Verlauf-Ansicht gerade geöffnet? (Reine Rückschau, kein Live-
     Mitlesen — siehe historyMessages() unten.) */
  historyOpen: false,
  /* Wird true, sobald der Leitfaden vollständig durchlaufen ist (die
     Abschlussnachricht war gerade dran oder schon vorher) — sperrt weitere
     Antwortversuche, statt die Abschlussnachricht endlos zu wiederholen.
     Zeigt "Interview beenden" statt der normalen Antwort-Bedienung (siehe
     .endbar in index.html) — führt in denselben Exit-Flow wie ein
     freiwilliger Abbruch, nur ohne Rückfrage und ohne Begründungs-Schritt. */
  interviewFinished: false,
  /* Wird true, nachdem im Exit-Flow tatsächlich eine Entscheidung
     (weiter/Studie beenden) getroffen wurde — zeigt die letzte, einfache
     Bestätigungsseite. */
  interviewEnded: false,

  /* --- ABBRUCH & EXIT-FLOW -------------------------------------------------
     Zustand für Rückfrage-Dialog + dreistufigen Exit-Flow. Die eigentlichen
     Texte/Inhalte stehen weiter unten in cfg.exit. */
  confirmOpen: false,   // Rückfrage-Dialog ("wirklich beenden?") sichtbar?
  exitOpen: false,      // Exit-Flow (Unterstützung/Begründung/Entscheid) sichtbar?
  exitStepIndex: 0,     // Index innerhalb des aktiven Schritt-Ablaufs (s. u.)
  exitViaAbort: false,  // true = freiwilliger Abbruch (mit Begründungs-Schritt),
                         // false = normaler Abschluss (Begründungs-Schritt entfällt)
  exitReason: "",        // Freitext aus Schritt 2, falls ausgefüllt
  exitFinalChoice: null, // "continue" oder "withdraw" — bestimmt den Text auf der allerletzten Seite

  /* --- SICHERHEITS-HINWEIS -------------------------------------------------
     Wird true, wenn der Server ein Stichwort aus study_content/
     safety_keywords.yaml in der ANTWORT DER TEILNEHMENDEN PERSON erkannt hat
     (siehe backend/safety_check.py). Zeigt einen ruhigen, nicht blockierenden
     Hinweis mit Hilfsangeboten — das Interview läuft normal weiter. Bewusst
     KEIN Gate/Interrupt. Nutzt dieselben Kontakte wie der Abbruch-Flow
     (cfg.exit.supportResources), um Inhalte nicht doppelt pflegen zu müssen. */
  safetyHintVisible: false,
  safety: {
    bannerTitle: "Falls Sie gerade über Suizidgedanken oder Selbstverletzung sprechen",
    bannerText: "Die folgenden Stellen sind kostenlos und unabhängig von dieser Studie erreichbar. Das Interview läuft in der Zwischenzeit ganz normal weiter.",
    dismissLabel: "Verstanden"
  },

  exit: {
    /* Button oben rechts, jederzeit während des aktiven Interviews. */
    abortButton: "Interview vorzeitig beenden",

    /* Schritt 0 — Rückfrage bei freiwilligem Abbruch (entfällt beim
       normalen Abschluss über .endbar). */
    confirmTitle:   "Interview wirklich vorzeitig beenden?",
    confirmText:    "Sie können jederzeit zum Gespräch zurückkehren. Wenn Sie beenden, wird das Interview geschlossen und nicht fortgesetzt.",
    confirmCancel:  "Zurück zum Interview",
    confirmConfirm: "Interview vorzeitig beenden",

    /* Schritt 1 — Unterstützung. Sachlich halten; die meisten Abbrüche haben
       keinen krisenhaften Grund, ein dramatischer Ton wirkt fehlplatziert. */
    supportTitle: "Danke, dass Sie bis hierher mitgemacht haben",
    supportText:  "Interviews zu psychischem Befinden können nachwirken. Falls Sie darüber sprechen möchten, sind die folgenden Stellen kostenlos und unabhängig von dieser Studie erreichbar.",
    supportResources: [
      { name: "Telefonseelsorge",              detail: "0800 111 0 111 · rund um die Uhr, kostenlos und anonym" },
      { name: "Info-Telefon Depression",       detail: "0800 33 44 533 · zu festen Sprechzeiten" },
      { name: "Ärztlicher Bereitschaftsdienst", detail: "116 117 · bei akutem Bedarf" }
    ],
    supportNext: "Weiter",

    /* Schritt 2 — optionale Begründung. NUR beim freiwilligen Abbruch,
       nie verpflichtend. */
    reasonTitle: "Möchten Sie uns den Grund nennen?",
    reasonText:  "Sie können optional angeben, warum Sie das Interview beendet haben. Das ist freiwillig und hat keine Auswirkung auf Ihre Vergütung.",
    reasonPlaceholder: "Grund (optional) …",
    reasonSkip: "Überspringen",
    reasonNext: "Weiter",

    /* Schritt 3 — Studienentscheid. Die Datenlage MUSS eindeutig benannt
       werden, sonst ist die Wahl nicht informiert. Beide Optionen bewusst
       gleich gewichtet. */
    decisionTitle: "Wie möchten Sie weitermachen?",
    decisionText:  "Ihre bisherigen Interviewantworten bleiben in beiden Fällen pseudonymisiert gespeichert und werden ausgewertet. Wie Sie deren Löschung beantragen können, steht unten.",
    continueLabel: "Nur das Interview beenden",
    continueDesc:  "Sie füllen die restlichen Fragebögen aus und schließen die Studie regulär ab.",
    withdrawLabel: "Die Studie vollständig beenden",
    withdrawDesc:  "Es werden keine weiteren Daten erhoben. Sie kehren ohne Abschluss zu Prolific zurück.",
    deletionNote:  "Löschung Ihrer Daten: Schreiben Sie mit Ihrem Prolific-Code an studie@example.org. Wir löschen Ihren Datensatz dann vollständig, ohne Angabe von Gründen.",

    /* Letzte Seite — unterschiedlich je nachdem, wie man hier ankam.
       Normaler Abschluss UND "nur Interview beenden" (nach Abbruch) zeigen
       denselben Dank; "Studie vollständig beenden" zeigt eine eigene,
       nüchterne Bestätigung ohne Dank (das wäre hier fehl am Platz). Bei
       gesetzter completionUrl/withdrawUrl (Prolific) wird automatisch
       dorthin weitergeleitet; ohne URL (Vor-Ort-Nutzung) bleibt nur der
       Text stehen. */
    finalThanksTitle:   "Vielen Dank!",
    finalThanksText:    "Das Interview ist beendet. Sie können dieses Fenster jetzt schließen.",
    finalWithdrawTitle: "Studie beendet",
    finalWithdrawText:  "Es werden keine weiteren Daten von Ihnen erhoben. Sie können dieses Fenster jetzt schließen.",

    /* NOCH NICHT VERDRAHTET — soll später bestimmen, ob Prolific-spezifische
       Texte ("zurück zu Prolific" o. Ä.) oder Vor-Ort-Texte gezeigt werden.
       Bis dahin: Prolific-Formulierungen als Standard, manuell anpassen. */
    deploymentMode: "prolific",

    /* Weiterleitungen zu Prolific. Leer lassen -> es wird nicht weitergeleitet
       (praktisch beim Testen der Seite, oder bei Vor-Ort-Nutzung). */
    completionUrl: "",
    withdrawUrl:   ""
  }
};

let audioStream = null;
let audioContext = null;
let recorderNode = null;
let recordedBuffers = [];
// Die zuletzt tatsächlich gesprochenen Satz-Häppchen (Text+Audio) — genutzt
// von "Frage wiederholen" (replayLastQuestion(), siehe weiter unten), damit
// nichts neu synthetisiert werden muss, um die letzte Frage erneut zu hören.
let lastReplyAudioPieces = [];

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
  if (audioStream || window.CONVERSATION.processingResponse || window.CONVERSATION.interviewFinished) {
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
  let currentAudio = null;
  let stopped = false;

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
    if (stopped) {
      // Wurde zwischen dem Einreihen und dem Abspielen abgebrochen (z. B.
      // Interview-Abbruch) — dieses Häppchen erst gar nicht mehr abspielen.
      return;
    }

    const audioData = atob(piece.audio);
    const bytes = new Uint8Array(audioData.length);
    for (let i = 0; i < audioData.length; i++) {
      bytes[i] = audioData.charCodeAt(i);
    }
    const blob = new Blob([bytes], { type: "audio/wav" });
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    currentAudio = audio;

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
      if (stopped) {
        // Zwischen play() und diesem Event kam der Abbruch — sofort wieder
        // anhalten, statt trotzdem weiterzuspielen.
        audio.pause();
        return;
      }
      window.CONVERSATION.clinicianTyping = false;
      window.CONVERSATION.processingResponse = false;
      window.CONVERSATION.clinicianSpeaking = true;
      render();
      startLevelLoop();
    });

    const finish = () => {
      stopLevelLoop();
      URL.revokeObjectURL(url);
      if (currentAudio === audio) {
        currentAudio = null;
      }
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

  // Sofort-Stopp von außen (siehe stopActiveTurnNow() weiter unten) — z. B.
  // wenn ein Abbruch bestätigt wird, während das LLM noch spricht oder
  // gerade eine Antwort verarbeitet: "Abbruch ist Abbruch", die Ausgabe
  // darf nicht im Hintergrund weiterlaufen, während schon die Exit-Seite
  // gezeigt wird.
  function stopNow() {
    stopped = true;
    stopLevelLoop();
    if (currentAudio) {
      try {
        currentAudio.pause();
      } catch (err) {
        // egal, wir wollen nur sicherstellen, dass nichts mehr tönt
      }
      currentAudio = null;
    }
    close();
  }

  return { playPiece: playPiece, close: close, stopNow: stopNow };
}

/* =============================================================================
   AKTIVER VORGANG — für sofortigen Abbruch von außen
   -----------------------------------------------------------------------------
   Genau EIN "laufender Vorgang" (gestreamte Antwort, gesprochene Eröffnung,
   oder eine Wiederholung) kann zu einem Zeitpunkt aktiv sein. Bei einem
   bestätigten Interview-Abbruch wird er hart gestoppt — Fetch/Stream wird
   abgebrochen, Audio wird sofort angehalten. Ohne dieses Zentralstück würde
   eine bereits laufende Sprachausgabe einfach weiterlaufen, während die
   Person schon auf der Exit-Seite ist.
============================================================================= */
let activeTurnStopFn = null;

function registerActiveTurn(stopFn) {
  activeTurnStopFn = stopFn;
}

function clearActiveTurn(stopFn) {
  // Nur zurücksetzen, wenn es wirklich noch derselbe Vorgang ist — ein
  // inzwischen neu gestarteter Vorgang soll durch das Aufräumen des alten
  // nicht versehentlich mit gelöscht werden.
  if (activeTurnStopFn === stopFn) {
    activeTurnStopFn = null;
  }
}

function stopActiveTurnNow() {
  if (activeTurnStopFn) {
    try {
      activeTurnStopFn();
    } catch (err) {
      console.warn("Fehler beim Stoppen des laufenden Vorgangs", err);
    }
    activeTurnStopFn = null;
  }
}

window.stopActiveTurnNow = stopActiveTurnNow;

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
  const playedPieces = [];
  let playing = false;
  let assembledText = "";
  let interviewFinished = false;
  let aborted = false;

  // "Abbruch ist Abbruch": wird von stopActiveTurnNow() aufgerufen, sobald
  // ein Interview-Abbruch bestätigt wird, während diese Antwort noch läuft.
  // Stoppt die Tonausgabe sofort und bricht den noch laufenden Stream ab,
  // statt im Hintergrund weiterzuspielen, während schon die Exit-Seite zu
  // sehen ist.
  function stopThisTurn() {
    aborted = true;
    player.stopNow();
    try {
      reader.cancel();
    } catch (err) {
      // Verbindung war ggf. schon zu Ende — kein Problem.
    }
  }
  registerActiveTurn(stopThisTurn);

  function maybePlayNext() {
    if (aborted || playing || audioQueue.length === 0) {
      return;
    }
    playing = true;
    const piece = audioQueue.shift();
    playedPieces.push(piece);
    player.playPiece(piece, function () {
      assembledText = assembledText ? assembledText + " " + piece.text : piece.text;
      playing = false;
      maybePlayNext();
    });
  }

  while (!aborted) {
    let readResult;
    try {
      readResult = await reader.read();
    } catch (err) {
      break; // Stream wurde abgebrochen (reader.cancel() durch stopThisTurn)
    }
    const { value, done } = readResult;
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
        if (data.safety_flag) {
          // Sofort anzeigen, nicht erst warten, bis die ganze Antwort da
          // ist — je früher der Hinweis, desto besser.
          window.CONVERSATION.safetyHintVisible = true;
          render();
        }
      } else if (data.type === "sentence") {
        audioQueue.push({ text: data.text, audio: data.audio, kind: data.kind });
        if (data.finished) {
          interviewFinished = true;
        }
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
  // liegen oder gerade spielen — erst danach gilt der Zug als beendet. Bricht
  // sofort ab, falls währenddessen ein Interview-Abbruch dazwischenkommt.
  await new Promise((resolve) => {
    const check = () => {
      if (aborted || (!playing && audioQueue.length === 0)) {
        resolve();
      } else {
        setTimeout(check, 50);
      }
    };
    check();
  });

  clearActiveTurn(stopThisTurn);

  if (aborted) {
    // Wir sind längst auf der Exit-Seite — am Gesprächszustand gibt es
    // nichts mehr sinnvoll zu ändern.
    return;
  }

  // Text wird erst jetzt, am Ende, ins Nachrichten-Objekt geschrieben — er
  // wurde ja nie live angezeigt, sondern nur fürs spätere Verlauf-Feature
  // mitgeschrieben ("Zurück zum Interview"-Ansicht).
  player.close();
  msg.text = assembledText;
  // "Frage wiederholen" soll nur die Frage nochmal abspielen, nicht die
  // Rückmeldung davor — deshalb hier auf die passenden "kind"s filtern.
  lastReplyAudioPieces = playedPieces.filter(function (p) { return p.kind !== "feedback"; });
  window.CONVERSATION.clinicianSpeaking = false;
  window.CONVERSATION.clinicianTyping = false;
  window.CONVERSATION.processingResponse = false;
  window.CONVERSATION.liveTranscript = "";
  window.CONVERSATION.turnDone = true; // Verlauf-Button darf erscheinen
  if (interviewFinished) {
    window.CONVERSATION.interviewFinished = true;
  }
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
  if (!trimmed || window.CONVERSATION.interviewFinished) {
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
      if (result.finished) {
        window.CONVERSATION.interviewFinished = true;
      }
      if (result.safety_flag) {
        window.CONVERSATION.safetyHintVisible = true;
      }
    }
  } catch (error) {
    window.CONVERSATION.messages.push({
      role: "clinician",
      text: `Network error: ${error.message}`,
    });
  } finally {
    window.CONVERSATION.clinicianTyping = false;
    window.CONVERSATION.turnDone = true; // Verlauf-Button darf ab jetzt erscheinen
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

  // "Frage wiederholen" soll nur die erste Frage abspielen, nicht die
  // Begrüßung/Framing-Nachricht davor.
  lastReplyAudioPieces = pieces.filter(function (p) { return p.kind !== "framing"; });
  const player = createSpokenAudioPlayer();
  let index = 0;
  let aborted = false;

  function stopThisTurn() {
    aborted = true;
    player.stopNow();
  }
  registerActiveTurn(stopThisTurn);

  function playNext() {
    if (aborted) {
      clearActiveTurn(stopThisTurn);
      return;
    }
    if (index >= pieces.length) {
      clearActiveTurn(stopThisTurn);
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

/* =============================================================================
   FRAGE WIEDERHOLEN (nur Sprach-Modalität)
   -----------------------------------------------------------------------------
   Da in der Sprach-Bedingung bewusst kein Text mitläuft, braucht es einen
   Weg, die zuletzt gestellte Frage nochmal zu hören, falls sie akustisch
   verpasst wurde. Spielt einfach die zuletzt tatsächlich gesprochenen
   Audio-Häppchen erneut ab (siehe lastReplyAudioPieces oben) — keine neue
   Synthese nötig. Nur die aktuellste Frage, keine älteren aus dem Verlauf
   (bewusst einfach gehalten).
============================================================================= */
function replayLastQuestion() {
  if (!lastReplyAudioPieces.length) {
    return;
  }
  // Während einer laufenden Antwort/Aufnahme nicht gleichzeitig abspielen.
  if (window.CONVERSATION.processingResponse || window.CONVERSATION.clinicianSpeaking || window.CONVERSATION.userRecording) {
    return;
  }

  window.CONVERSATION.processingResponse = true;
  render();

  const player = createSpokenAudioPlayer();
  let index = 0;
  let aborted = false;

  function stopThisTurn() {
    aborted = true;
    player.stopNow();
  }
  registerActiveTurn(stopThisTurn);

  function playNext() {
    if (aborted) {
      clearActiveTurn(stopThisTurn);
      return;
    }
    if (index >= lastReplyAudioPieces.length) {
      clearActiveTurn(stopThisTurn);
      player.close();
      window.CONVERSATION.clinicianSpeaking = false;
      window.CONVERSATION.processingResponse = false;
      render();
      return;
    }
    const piece = lastReplyAudioPieces[index];
    index += 1;
    player.playPiece(piece, playNext);
  }

  playNext();
}

window.replayLastQuestion = replayLastQuestion;
window.hasReplayableQuestion = function () {
  return lastReplyAudioPieces.length > 0;
};

/* =============================================================================
   VERLAUF (beide Modalitäten)
   -----------------------------------------------------------------------------
   Zeigt vergangene Fragen/Antworten als Text — auch in der Sprach-Bedingung,
   wo live bewusst kein Text mitläuft (das hier ist reine Rückschau, kein
   Live-Mitlesen, siehe Absprache im Projektplan). Schneidet die gerade
   laufende, noch unbeantwortete Kliniker-Nachricht ab, damit im Verlauf nur
   wirklich abgeschlossene Runden auftauchen.
============================================================================= */
function historyMessages() {
  const msgs = window.CONVERSATION.messages;
  if (msgs.length && msgs[msgs.length - 1].role === "clinician" && !msgs[msgs.length - 1].text) {
    return msgs.slice(0, -1);
  }
  return msgs;
}

window.historyMessages = historyMessages;

/* =============================================================================
   SICHERHEITS-HINWEIS SCHLIESSEN
   -----------------------------------------------------------------------------
   Schließt nur die aktuell sichtbare Anzeige — ein neuer Treffer in einer
   späteren Antwort blendet den Hinweis unabhängig davon wieder ein.
============================================================================= */
function dismissSafetyHint() {
  window.CONVERSATION.safetyHintVisible = false;
  render();
}

window.dismissSafetyHint = dismissSafetyHint;

/* =============================================================================
   ABBRUCH + EXIT-FLOW
   -----------------------------------------------------------------------------
   Zwei Wege führen zum selben Exit-Flow (Unterstützung -> [Begründung] ->
   Studienentscheid), unterscheiden sich nur am Anfang:
     - Freiwilliger Abbruch (jederzeit, "Interview beenden" oben rechts):
       erst Rückfrage ("wirklich beenden?"), dann Begründungs-Schritt dabei.
     - Normaler Abschluss (Leitfaden fertig, siehe .endbar): keine Rückfrage
       nötig (es gibt nichts "abzubrechen"), Begründungs-Schritt entfällt.
   Die eigentlichen Hooks (onAbortIntent usw.) sind bewusst einfache
   Konsolen-Logs — echtes Speichern kommt erst mit der Datenspeicherung
   (siehe Projektplan). Sie lassen sich hier direkt überschreiben, falls dein
   System eigene Logging-Funktionen mitbringt.
============================================================================= */
window.onAbortIntent    = window.onAbortIntent    || function () { console.log("[exit] Abbruch-Absicht (Beenden-Button geklickt)"); };
window.onAbortCancelled = window.onAbortCancelled || function () { console.log("[exit] Abbruch abgebrochen — zurück zum Interview"); };
window.onAbortConfirmed = window.onAbortConfirmed || function () { console.log("[exit] Abbruch bestätigt"); };
window.onAbortReason    = window.onAbortReason    || function (reason) { console.log("[exit] Begründung:", reason || "(keine angegeben)"); };
window.onExitChoice     = window.onExitChoice     || function (choice) { console.log("[exit] Studienentscheid:", choice); };

function activeExitSteps() {
  // 1 = Unterstützung, 2 = Begründung, 3 = Studienentscheid.
  // Beim normalen Abschluss nur Schritt 1 — die Studienentscheid-Frage
  // ("weitermachen oder Studie abbrechen?") betrifft nur den vorzeitigen
  // Abbruch; wer regulär fertig ist, hat ja nichts mehr zu entscheiden.
  return window.CONVERSATION.exitViaAbort ? [1, 2, 3] : [1];
}

function currentExitStep() {
  const steps = activeExitSteps();
  return steps[window.CONVERSATION.exitStepIndex] ?? steps[steps.length - 1];
}

function advanceExitStep() {
  const steps = activeExitSteps();
  if (window.CONVERSATION.exitStepIndex < steps.length - 1) {
    window.CONVERSATION.exitStepIndex += 1;
    render();
  } else {
    // Letzter Schritt im aktiven Ablauf erreicht. Beim normalen Abschluss
    // ist das bereits Schritt 1 (Unterstützung) — es gibt keine
    // Studienentscheid-Frage, also direkt abschließen ("continue" ist hier
    // nur ein interner Wert, entspricht "ganz normal fertig").
    finishExit("continue");
  }
}

// Vom Abbruch-Button (oben rechts, jederzeit während des Interviews) —
// öffnet erst die Rückfrage, noch keine endgültige Entscheidung.
function startAbortFlow() {
  window.onAbortIntent();
  window.CONVERSATION.confirmOpen = true;
  render();
}

function cancelAbort() {
  window.CONVERSATION.confirmOpen = false;
  window.onAbortCancelled();
  render();
}

function confirmAbort() {
  // "Abbruch ist Abbruch": stoppt sofort jede noch laufende Tonausgabe/
  // Antwortverarbeitung, statt sie im Hintergrund weiterlaufen zu lassen,
  // während schon die Exit-Seite gezeigt wird.
  stopActiveTurnNow();
  window.CONVERSATION.clinicianSpeaking = false;
  window.CONVERSATION.clinicianTyping = false;
  window.CONVERSATION.processingResponse = false;
  window.CONVERSATION.userRecording = false;

  window.CONVERSATION.confirmOpen = false;
  window.CONVERSATION.exitViaAbort = true;
  window.CONVERSATION.exitStepIndex = 0;
  window.CONVERSATION.exitOpen = true;
  window.onAbortConfirmed();
  render();
}

// Vom .endbar-Button nach normalem Abschluss — direkt in den Exit-Flow,
// keine Rückfrage, kein Begründungs- oder Studienentscheid-Schritt. Ruft
// stopActiveTurnNow() ebenfalls (Sicherheitsnetz) auf, auch wenn hier im
// Normalfall nichts mehr laufen sollte, sobald der Leitfaden fertig ist.
function startNaturalExit() {
  stopActiveTurnNow();
  window.CONVERSATION.exitViaAbort = false;
  window.CONVERSATION.exitStepIndex = 0;
  window.CONVERSATION.exitOpen = true;
  render();
}

function submitExitReason(reasonText) {
  window.CONVERSATION.exitReason = (reasonText || "").trim();
  window.onAbortReason(window.CONVERSATION.exitReason);
  advanceExitStep();
}

function skipExitReason() {
  window.CONVERSATION.exitReason = "";
  window.onAbortReason("");
  advanceExitStep();
}

// choice: "continue" (Studie regulär fortsetzen — auch der Normalfall nach
// vollständigem Abschluss) oder "withdraw" (Studie vollständig beenden,
// keine weiteren Daten). Wird sowohl vom Studienentscheid-Schritt (Abbruch)
// als auch automatisch nach dem normalen Abschluss aufgerufen.
//
// WICHTIG: exitOpen bleibt hier bewusst true (nicht zurück auf false) —
// die Exit-Seite ist eine echte, eigene Seite ohne Rückweg, keine
// Overlay-Ebene über dem Interview. Die finale Bestätigung wird auf
// DERSELBEN Seite gezeigt (siehe renderExitFlow() in index.html), nicht in
// einer separaten Ansicht, die das Interview daneben wieder sichtbar macht.
function finishExit(choice) {
  window.onExitChoice(choice);
  window.CONVERSATION.exitFinalChoice = choice;
  window.CONVERSATION.interviewEnded = true;
  const url = choice === "withdraw" ? window.CONVERSATION.exit.withdrawUrl : window.CONVERSATION.exit.completionUrl;
  render();
  if (url) {
    window.location.href = url;
  }
}

// Beibehalten als Alias, falls index.html noch den alten Namen aufruft.
function chooseExitDecision(choice) {
  finishExit(choice);
}

window.startAbortFlow = startAbortFlow;
window.cancelAbort = cancelAbort;
window.confirmAbort = confirmAbort;
window.startNaturalExit = startNaturalExit;
window.currentExitStep = currentExitStep;
window.activeExitSteps = activeExitSteps;
window.advanceExitStep = advanceExitStep;
window.submitExitReason = submitExitReason;
window.skipExitReason = skipExitReason;
window.chooseExitDecision = chooseExitDecision;

