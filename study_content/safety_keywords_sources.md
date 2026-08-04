# Quellen zur Sicherheits-Stichwortliste (safety_keywords.yaml)

Dokumentiert, welche fachliche Literatur bei der Erstellung/Erweiterung von
`safety_keywords.yaml` herangezogen wurde und was konkret daraus übernommen
wurde — damit nachvollziehbar bleibt, worauf einzelne Entscheidungen beruhen,
nicht nur, dass es sie gibt.

---

## Quelle 1

**Titel:** Suicide- and crisis-risk detection using large language models in mental-health chatbots
**Autoren:** MULTICAST Consortium (Olbrich S, Kleim B, Schultebraucks K, Weber S, Homan S, Gershov S, Monn A, Häfliger A, Mocellin J) u. a.
**Veröffentlicht:** medRxiv-Preprint, Januar 2026
**DOI/URL:** https://www.medrxiv.org/content/10.64898/2026.01.12.26343914v1
(auch verfügbar über ResearchGate: https://www.researchgate.net/publication/399896086)

**Kernaussage:** Plädiert für einen LLM-basierten Risiko-Klassifikator mit
einstellbarer Empfindlichkeit statt reiner Stichwortsuche; bei höchster
Empfindlichkeit 100 % Erkennung bei ca. 49 % Falsch-Positiven-Rate, bei
etwas moderaterer Stufe 98,9 % Erkennung bei ca. 41 % Falsch-Positiven —
zeigt, dass hohe Erkennungsraten strukturell hohe Fehlalarmraten
mit sich bringen, unabhängig vom technischen Ansatz.

**Was daraus übernommen wurde:**
- Bestätigung, dass PASSIVE, indirekte Formulierungen ("I wish I were dead")
  als Warnsignal gelten sollten, nicht nur explizite Suizidäußerungen ->
  Grundlage für "wäre tot"/"besser tot"/"wish i was dead"/"wish i were dead"
  in safety_keywords.yaml.
- Bestätigung des Architektur-Prinzips, das wir bereits unabhängig gewählt
  hatten: Erkennung getrennt von der eigentlichen Gesprächsführung, kein
  Abbruch des Gesprächs bei Auslösung, sondern Weiterlaufen mit sichtbarem
  Hilfsangebot.
- NICHT übernommen: das eigentliche Kernkonzept des Papers (LLM-Klassifikator
  mit einstellbarer Sensitivität statt Stichwortsuche) — bewusste
  Entscheidung, siehe Begründung oben in safety_keywords.yaml (Durchsichtigkeit
  für Ethikkommission, keine klinische Bewertung durch die KI selbst).

---

## Quelle 2

**Titel:** KI-Chatbots und Suizidalität bei Jugendlichen — Aktuelle Evidenz und Handlungsempfehlungen
**Autor:** Augustin M
**Veröffentlicht:** Monatsschrift Kinderheilkunde (2026), Open Access
**DOI:** https://doi.org/10.1007/s00112-026-02362-6

**Kernaussage:** Narrativer Review zu KI-Chatbots und Suizidalität bei
Jugendlichen. Zeigt anhand eines dokumentierten Falls (New York Times,
Sommer 2025) und mehrerer Studien, dass bestehende Chatbots Anforderungen an
adäquate Reaktionen bei Suizidalität häufig nicht erfüllen und Sicherheits-
mechanismen umgangen werden können. Benennt als mitwirkende technische
Mechanismen: übermäßig bestätigendes Antwortverhalten ("Sycophancy"),
Hyperpersonalisierung, sprachliche Anpassung und Anthropomorphismus.

**Was daraus übernommen wurde:**
- Indirekte Anfragen (nicht nur Aussagen über sich selbst) sind ebenfalls ein
  Warnsignal — im Paper genannt am Beispiel einer Anfrage nach der Höhe von
  Brücken nach Erwähnung von Arbeitsplatzverlust, die von mehreren Chatbots
  nicht als impliziter Suizid-Hinweis erkannt wurde -> Ergänzung eng
  gefasster Methodik-Anfrage-Phrasen in safety_keywords.yaml (s. u.).

**Offener Punkt, NICHT in safety_keywords.yaml, sondern separat vermerkt:**
Das Paper nennt übermäßig bestätigendes Antwortverhalten ("Sycophancy") als
zentralen Risikofaktor — das betrifft nicht die Stichwort-Erkennung, sondern
möglicherweise die Formulierung der LLM-Rückmeldung selbst
(behavior_profiles/prompts). Noch nicht geprüft, ob unsere Profile
(strict_clinical/flexible_conversational) davon betroffen sein könnten —
auf der Liste offener Punkte vermerkt.

---

## Erweiterung 04.08. (nicht aus Fachliteratur, sondern eigene Praxis-Lektion)

Nutzerin hat die deutsche Liste eigenständig um "zu"-Infinitiv-Varianten
erweitert (z. B. "umzubringen" neben "mich umbringen") — dieselbe Lektion
wie beim ursprünglichen "nicht mehr da (zu) sein"-Fund, nur systematisch auf
die ganze Liste angewendet. Englische Liste entsprechend erweitert, aber
NICHT wortwörtlich übersetzt, sondern nach demselben Prinzip an die jeweils
andere Bruchstelle der Sprache angepasst:
- Deutsch bricht durch die infixe "zu"-Form ("um-zu-bringen").
- Englisch bricht stattdessen durch die Gerundium-Form ("hurt myself"
  matcht nicht auf "hurting myself").

Zwei Fälle bewusst NICHT wörtlich übersetzt, mit Begründung:
- "zu töten" (sehr breit) -> nicht bloßes "kill"/"to kill" (im Englischen
  extrem verbreitet in harmlosen Zusammenhängen wie "kill time"), sondern
  enger "kill me".
- "strick" -> nicht "rope" (zu generisch), sondern "noose".
