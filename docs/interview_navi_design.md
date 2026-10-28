# Interview Navi — Design-Notiz

> **Quelle:** Design-Notiz des Projekteinhabers, 2026-10-22.
>
> **Status:** verbindliche Design-Referenz für das v2-Interviewformat, die
> Hierarchie und die Navigationslogik (Phase 2 von RECONSTRUCTION_PLAN.md).
> Wo sich PDF-Zielarchitektur, RECONSTRUCTION_PLAN.md oder die v2-Loader-
> Dokumentation von dieser Notiz unterscheiden, gilt diese Notiz.
>
> Deutsches Original unten unverändert übernommen.

---

## Fokus des Interview Navies
Ein spezifisches Tool, das dem Interviewer-Agenten zur Verfügung stehen soll. Dieses Tool hat folgende Kernaufgabe:
- Es ermöglicht dem Interviewer-Agenten die Navigation durch ein gespeichertes, strukturiertes Interview
- Es soll dabei die vorgegebene Navigationslogik und die Gesprächsfluss-Regeln des Interviews respektieren und einhalten
## Vorgesehener Workflow (später, Frontend-Ebene)
Ein Forscher soll perspektivisch über ein Frontend ein strukturiertes Interview eingeben können. Dieses Interview wird dann im Backend in ein navigierbares Objekt überführt und gespeichert, das:
- eine einstellbare/konfigurierbare Navigationslogik besitzt
- bestimmte Gesprächsfluss-Regeln kodiert und durchsetzt
Der Interviewer-Agent soll mit diesem Tool eine "zu stellende Fragen"-Menge führen, daraus die sinnvollste nächste Frage bestimmen und diese abhängig von möglichen Kriterienfolgen bearbeiten.
# Interview Navi
## Grundkonzept
Die Klasse bildet ein strukturiertes Interview als navigierbares Objekt ab: eine Folge inhaltlich synonym formulierter Fragen an einen Patienten, optional gruppiert und mit konfigurierbarer, teils dynamischer Navigationslogik. Das Interview läuft als Online-Algorithmus — Kriterien werden fortlaufend während des Gesprächs ausgewertet, sobald die zugehörigen Antworten vollständig vorliegen, nicht erst nachträglich oder verzögert.

## Hierarchische Struktur

| Ebene | Rolle | ID-Schema |
|---|---|---|
| Section Group | Bündelt ähnliche Question Groups; beliebig tief verschachtelbar (auch Section in Section); Beziehung zu anderen Section Groups konfigurierbar (zeitlich, pathologisch differenzierbar, komplexer) | `sec_1.0.0` |
| Question Group | Bündelt zusammenhängende Fragen | `qg_1.1.0` bis `qg_1.x.0` |
| Single Question | Atomare Frage mit Antworttyp (Ja/Nein, Zahl, unspezifizierter String) | `sq_1.1.1` bis `sq_1.1.x` |

Jedes Interview hat mindestens eine Question Group mit mindestens einer Frage. Section Groups sind beliebig tief verschachtelbar, um komplexe Kataloge wie das SCID abzubilden. Die ID kodiert dabei zugleich Typ und Position in der Hierarchie, was Debugging und gezielte Sprungreferenzen erleichtert.

## Fragen, Groups und Criteria
Eine Single Question hat ID, Fragetext und Antworttyp (Ja/Nein, Zahl, String) und darf im selben Durchlauf nicht wiederholt gestellt werden. Question Groups definieren eine frei konfigurierbare interne Logik (synonym, sequenziell, abhängig, komplex) und können als Criteria Group markiert werden, wobei sich das Kriterium immer auf alle Elemente der Gruppe zusammen bezieht. Section Groups können ebenfalls Criteria Groups sein und besitzen zusätzlich eine konfigurierbare Beziehung zu anderen Sections (zeitlich, pathologisch, frei definiert).
## Navigation und Verknüpfungslogik
Ohne Criteria Groups folgt die Navigation einer Postordertraversierung; mit Criteria Groups sind Abweichungen, Querverbindungen und Schleifen erlaubt. Die Auswertung verschachtelter Kriterien läuft auch postorder (zuerst Question Groups, dann Sections), aber durch den Online-Charakter wird praktisch immer nur das jeweils nächste vollständig beantwortete Kriterium geprüft. Die Verknüpfungslogik pro Criteria Group soll mehrere frei kombinierbare Operatoren unterstützen, darunter UND, ODER, NICHT, min, max, equal sowie ergänzend ungleich, größer/kleiner-gleich, zwischen, enthält, ist-leer, Anzahl/Threshold/Mehrheit und gewichtete Summe.

| Kategorie | Operatoren |
|---|---|
| Logisch | NICHT, UND, ODER |
| Arithmetisch (Einzelwert) | gleich , ungleich, größer-gleich, kleiner-gleich, zwischen, min, max |
| Text | enthält, enthält nicht |
| Vollständigkeit | ist leer, ist nicht leer |
| Aggregation (Gruppenebene) | Anzahl/Count, Schwellenwert, Mehrheit, gewichtete Summe |

## Sprungziele und Validierung
Sprungziele werden als direkte Referenzen auf diese IDs modelliert, unabhängig davon, ob das Ziel eine Question, Question Group oder Section Group ist. Sackgassen (fehlendes oder ungültiges Sprungziel) müssen bereits beim Einpflegen eines Interviews vom Code erkannt und verhindert werden. Nach Sprüngen wird default der Postordertraversierung gefolgt. Das konkrete Verhalten bei Schleifen, die zu einer bereits beantworteten (gesperrten) Question Group zurückführen, war offen gelassen — **entschieden am 2026-10-26: das Episoden-Modell**, siehe den Nachtrag unten.

## Bezug zum Interviewer-Agenten
Der Interviewer-Agent nutzt das spätere Navigationstool, um eine "zu stellende Fragen"-Menge zu führen, daraus die sinnvollste nächste Frage zu bestimmen und je nach Ausgang der Criteria-Group-Auswertungen entsprechend zu reagieren.

## Nachtrag: Episoden-Modell und verschobene Sprungauslösung (2026-10-26)

Die oben offen gelassene Frage — Verhalten einer Schleife, die zu einer
bereits beantworteten (gesperrten/locked) Question Group zurückführt —
ist entschieden (Eigentümerentscheidung 2026-10-26; Umsetzung in
`backend/interview_definition.py`, `backend/interview_run.py` und
`backend/interview_navigator.py`, Tests in `tests/`).

**Episoden-Modell.** Das Konzept "Episode" ist ein Definitionselement
PRO INTERVIEW (Block `episodes` im v2-Format; standardmäßig
deaktiviert). Das Interview fragt zunächst alles über die NOW-Episode
(Fenster in Tagen + lesbare Regel). Führt das zu keiner Diagnose,
kommen frühere Episoden ins Spiel:

- die ärztliche Fachkraft erklärt eine weitere Episode; die Laufzeit
  weist deterministische IDs zu (`ep_1`, `ep_2`, …),
- die Patientin / der Patient beschreibt die Episode in eigenen
  Worten — diese wörtlichen Fassungen akkumulieren als Aliasse der
  Episode (Verlauf des Gesprächs, Referenzmaterial),
- zwei Erklärungen können sich auf dieselbe reale Episode beziehen:
  der **Diagnostiker** ist die einzige LLM, die das Interview als
  Ganzes betrachtet, und urteilt über die Identität (seine einzige
  interview-weite Rolle); die Laufzeit vereinheitlicht identische
  Erklärungen (`merged_into` zeigt auf die frühere Erklärung, die
  Aliasse wandern auf die Gewinner-Episode),
- eine Frage wird pro (Frage, kanonische Episode) höchstens EINMAL
  gestellt — die alte Regel "nie erneut stellen" ist der
  Spezialfall; `CLOSED/SKIPPED → REQUIRED` ist der einzige Übergang
  aus einem Endzustand und dient ausschließlich dem Episode-Re-Ask
  (`reopen_question`).

**Sprung-Aspect.** Ein Sprung kann einen Episode-Aspect tragen
(genau einer): `episode_jump: true` (dynamischer Modus) oder
`episode: <name>` (vorab deklarierte Episoden). Das Ziel muss eine
Question Group (oder eine Frage) sein, deren Gruppe
`episode_evaluable: true` trägt — wird beim Einlesen validiert.

**Verhalten bei gesperrtem Ziel.**

- Sprung OHNE Episode-Aspect: wird ignoriert + protokolliert
  (`loop_note` im Navigationsvorschlag),
- Sprung MIT Episode-Aspect: die Zielgruppe wird für diese Episode
  NOCHMALS GESTELLT (`reopen_proposals` für die Pflichtfragen der
  Gruppe, die für diese Episode noch nicht gestellt wurden;
  `reframe_context` macht den Wechsel explizit: "wir sprechen jetzt
  über Episode X, nicht über die bisher bewertete(n) Episode(n)"),
- dynamischer Modus: der Navigationsvorschlag meldet zunächst
  `pending_episode` und wartet auf die Diagnostiker-Entscheidung
  (`EpisodeDecision`: neue Episode oder dieselbe reale Episode —
  nach einem Merge gilt die kanonische ID); dieselbe reale Episode →
  Sprung wird ignoriert + protokolliert.

**Verschobene Sprungauslösung.** Kriterien werden weiterhin ONLINE
entschieden; Sprünge und Gruppenkonsequenzen werden aber erst bei
GRUPPENABSCHLUSS (alle Pflichtfragen der Gruppe terminal) angewendet:
`complete_group` schlägt die übersprungenen optionalen Fragen der
Gruppe vor und löst den ersten ausgelösten Sprung der Gruppe aus
(Declarationsreihenfolge); `resolve_jump` mitten in der Gruppe
schlägt fehl (fails loudly).
