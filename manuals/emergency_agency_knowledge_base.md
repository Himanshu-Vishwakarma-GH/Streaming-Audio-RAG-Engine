# Multi-Agency Emergency Operations Knowledge Base
## Comprehensive Input-Target Training & Response Registry for PS-003 RAG Engine

> **Document Purpose:** Complete knowledge base and structured registry containing sector-specific emergency inputs, trigger keywords, exact AI voice responses, and deterministic numeric verification rules across five critical emergency sectors:
> 1. **Air Traffic Control (ATC)**
> 2. **911 Emergency Dispatch / Medical Emergency**
> 3. **Police Department (Law Enforcement)**
> 4. **Fire Department (Structure, Hazmat & Wildfire)**
> 5. **Industrial Safety & Chemical Processing**

---

## 1. ✈️ Sector 1: Air Traffic Control (ATC)

### Chunk ATC-01: Emergency Transponder Squawk Codes
* **Chunk ID:** `1`
* **Sector:** `Air Traffic Control`
* **Intent Identifier:** `transponder_squawk_code`
* **User Spoken Trigger Keywords:**
  * *"emergency squawk"*, *"squawk code"*, *"transponder emergency"*, *"general emergency code"*, *"mayday squawk"*
* **Input Acoustic / Semantic Signatures:** `["squawk", "transponder", "emergency", "code", "mayday"]`
* **Canonical Spoken Query:** `"Emergency squawk transponder"`
* **Ground Truth AI Spoken Response:**
  > *"For general emergency, set transponder squawk code to 7700. For radio communication failure, squawk 7600. For unlawful interference or hijacking, squawk 7500."*
* **Deterministic Guardrail Numeric Truth:** `[7700, 7600, 7500]`
* **Operational Notes:** General emergency declaration strictly requires code `7700`. Block any false hijacking overrides.

---

### Chunk ATC-02: Altitude, Radar Separation & Emergency Descent
* **Chunk ID:** `0`
* **Sector:** `Air Traffic Control`
* **Intent Identifier:** `altitude_separation`
* **User Spoken Trigger Keywords:**
  * *"radar separation"*, *"terminal separation"*, *"emergency descent altitude"*, *"enroute separation"*
* **Input Acoustic / Semantic Signatures:** `["altitude", "separation", "descent", "terminal", "radar", "enroute"]`
* **Canonical Spoken Query:** `"Radar separation and emergency descent altitude"`
* **Ground Truth AI Spoken Response:**
  > *"Minimum radar separation is 3 nautical miles within terminal airspace, and 5 nautical miles en-route. Emergency descent target altitude is 10,000 feet MSL or lowest safe altitude."*
* **Deterministic Guardrail Numeric Truth:** `[3.0, 5.0, 10000.0]`
* **Operational Notes:** Terminal separation: `3 nm`; En-route separation: `5 nm`; Oxygen descent ceiling: `10000 ft`.

---

## 2. 🚑 Sector 2: 911 Dispatch & Emergency Medical Services (EMS)

### Chunk 911-01: Adult CPR Compression-to-Ventilation Ratio
* **Chunk ID:** `3`
* **Sector:** `911 Emergency Dispatch`
* **Intent Identifier:** `cpr_compression_ratio`
* **User Spoken Trigger Keywords:**
  * *"cpr ratio"*, *"chest compressions"*, *"cardiac arrest"*, *"unresponsive breathing"*, *"cpr count"*, *"compressions per minute"*
* **Input Acoustic / Semantic Signatures:** `["cpr", "compression", "cardiac", "breath", "unresponsive", "rate"]`
* **Canonical Spoken Query:** `"CPR compression ratio and rate"`
* **Ground Truth AI Spoken Response:**
  > *"For adult CPR, deliver 30 chest compressions followed by 2 rescue breaths at a compression rate of 100 to 120 beats per minute, pressing at least 2 inches deep."*
* **Deterministic Guardrail Numeric Truth:** `[30, 2, 100, 120, 2.0]`
* **Operational Notes:** Ratio must strictly be `30:2`. Rate `100-120` cpm. Depth `2.0` inches minimum.

---

### Chunk 911-02: Severe Bleeding & Hemorrhage Tourniquet Protocol
* **Chunk ID:** `4`
* **Sector:** `911 Emergency Dispatch`
* **Intent Identifier:** `severe_bleeding_tourniquet`
* **User Spoken Trigger Keywords:**
  * *"severe bleeding"*, *"tourniquet placement"*, *"arterial bleed"*, *"stab wound bleeding"*, *"amputation bleeding"*
* **Input Acoustic / Semantic Signatures:** `["bleed", "bleeding", "tourniquet", "arterial", "hemorrhage", "wound"]`
* **Canonical Spoken Query:** `"Severe bleeding tourniquet application"`
* **Ground Truth AI Spoken Response:**
  > *"Apply direct firm pressure immediately. If uncontrolled extremity bleeding, apply windlass tourniquet 2 to 3 inches above the wound, never over a joint, and tighten until bleeding ceases completely."*
* **Deterministic Guardrail Numeric Truth:** `[2.0, 3.0]`
* **Operational Notes:** Placement strictly `2 to 3` inches proximal to injury site.

---

### Chunk 911-03: Pediatric Febrile Seizure Emergency Protocol
* **Chunk ID:** `5`
* **Sector:** `911 Emergency Dispatch`
* **Intent Identifier:** `pediatric_seizure_protocol`
* **User Spoken Trigger Keywords:**
  * *"child seizure"*, *"baby convulsing"*, *"pediatric seizure"*, *"febrile seizure"*, *"child shaking"*
* **Input Acoustic / Semantic Signatures:** `["seizure", "pediatric", "child", "convulsing", "febrile", "baby"]`
* **Canonical Spoken Query:** `"Child having seizure emergency instructions"`
* **Ground Truth AI Spoken Response:**
  > *"Place child gently on their side in recovery position on the floor. Do not insert anything into the mouth. If seizure duration exceeds 5 minutes, immediately dispatch ALS Priority 1 unit."*
* **Deterministic Guardrail Numeric Truth:** `[5.0, 1]`
* **Operational Notes:** Maximum threshold before status epilepticus intervention is `5` minutes; dispatch code Priority `1`.

---

## 3. 🚓 Sector 3: Police Department (Law Enforcement)

### Chunk POL-01: Active Shooter & Barricade Containment Protocol
* **Chunk ID:** `6`
* **Sector:** `Police Department`
* **Intent Identifier:** `active_shooter_containment`
* **User Spoken Trigger Keywords:**
  * *"active shooter"*, *"shots fired"*, *"barricaded suspect"*, *"hostage situation"*, *"perimeter containment"*
* **Input Acoustic / Semantic Signatures:** `["shooter", "shots", "barricade", "hostage", "perimeter", "tactical"]`
* **Canonical Spoken Query:** `"Active shooter containment perimeter protocol"`
* **Ground Truth AI Spoken Response:**
  > *"Code 10-33 emergency traffic. First arriving contact team of 2 to 4 officers must bypass wounded and neutralize active threat immediately. Establish outer perimeter at minimum 300 meters standoff."*
* **Deterministic Guardrail Numeric Truth:** `[2, 4, 300.0]`
* **Operational Notes:** Contact team threshold `2-4` officers; outer safety perimeter `300` meters.

---

### Chunk POL-02: Ten-Codes for Emergency Officer In Distress (10-33 & 10-99)
* **Chunk ID:** `7`
* **Sector:** `Police Department`
* **Intent Identifier:** `officer_distress_codes`
* **User Spoken Trigger Keywords:**
  * *"officer down"*, *"officer in distress"*, *"10-33"*, *"10-99"*, *"emergency radio code"*, *"police ten code"*
* **Input Acoustic / Semantic Signatures:** `["officer", "distress", "down", "emergency", "code", "10-33", "10-99"]`
* **Canonical Spoken Query:** `"Officer distress and urgent emergency ten-codes"`
* **Ground Truth AI Spoken Response:**
  > *"Code 10-33 signals emergency radio traffic only. Code 10-78 requests urgent backup. Code 10-99 indicates officer in extreme danger or officer down, initiating immediate all-units Priority 1 response."*
* **Deterministic Guardrail Numeric Truth:** `[10, 33, 10, 78, 10, 99, 1]`
* **Operational Notes:** Radio codes: `10-33` (emergency radio traffic), `10-78` (urgent assistance), `10-99` (officer down/extreme peril).

---

## 4. 🚒 Sector 4: Fire Department & Hazardous Materials (Hazmat)

### Chunk FIRE-01: High-Rise Structure Fire & Evacuation Staging
* **Chunk ID:** `8`
* **Sector:** `Fire Department`
* **Intent Identifier:** `high_rise_fire_staging`
* **User Spoken Trigger Keywords:**
  * *"structure fire"*, *"high rise fire"*, *"building fire"*, *"staging floor"*, *"standpipe hookup"*, *"fire attack"*
* **Input Acoustic / Semantic Signatures:** `["fire", "structure", "high-rise", "building", "staging", "standpipe"]`
* **Canonical Spoken Query:** `"High rise structure fire staging and standpipe pressure"`
* **Ground Truth AI Spoken Response:**
  > *"Establish interior staging area 2 floors below the fire floor. Connect attack lines to the standpipe on the floor below the fire, maintaining 100 PSI nozzle pressure with 2.5 inch supply lines."*
* **Deterministic Guardrail Numeric Truth:** `[2, 100.0, 2.5]`
* **Operational Notes:** Staging location: `2` floors below fire; Attack nozzle pressure: `100.0 PSI`; Supply hose diameter: `2.5` inches.

---

### Chunk FIRE-02: Hazardous Materials BLEVE & Evacuation Isolation Radius
* **Chunk ID:** `9`
* **Sector:** `Fire Department`
* **Intent Identifier:** `hazmat_bleve_isolation`
* **User Spoken Trigger Keywords:**
  * *"bleve danger"*, *"propane tank fire"*, *"hazmat tank"*, *"chemical spill"*, *"evacuation radius"*, *"flammable gas tank"*
* **Input Acoustic / Semantic Signatures:** `["bleve", "hazmat", "tank", "propane", "radius", "isolation", "evacuation"]`
* **Canonical Spoken Query:** `"Hazmat BLEVE evacuation standoff distance"`
* **Ground Truth AI Spoken Response:**
  > *"For pressurized liquefied gas vessels exposed to direct fire impingement with BLEVE risk, immediately withdraw all personnel and establish minimum 1 mile isolation perimeter in all directions."*
* **Deterministic Guardrail Numeric Truth:** `[1.0]`
* **Operational Notes:** Minimum standoff radius for BLEVE risk: `1.0` mile (1600 meters) in 360 degrees.

---

## 5. ☢️ Sector 5: Chemical Plant & Industrial Safety

### Chunk IND-01: Reactor Core R-101 Pressure and Quench Thresholds
* **Chunk ID:** `2`
* **Sector:** `Industrial Safety`
* **Intent Identifier:** `reactor_mawp_threshold`
* **User Spoken Trigger Keywords:**
  * *"reactor core"*, *"mawp pressure"*, *"reactor mawp"*, *"venting threshold"*, *"chemical reactor"*, *"quench injection"*
* **Input Acoustic / Semantic Signatures:** `["reactor", "pressure", "mawp", "venting", "threshold", "quench"]`
* **Canonical Spoken Query:** `"Reactor core pressure MAWP threshold"`
* **Ground Truth AI Spoken Response:**
  > *"Reactor Core R-101 Maximum Allowable Working Pressure MAWP is 450.0 PSI. Emergency venting trip threshold is 485.5 PSI. Quench tank injection rate is 1250 liters per minute."*
* **Deterministic Guardrail Numeric Truth:** `[450.0, 485.5, 1250.0]`
* **Operational Notes:** Core MAWP: `450.0 PSI`; Venting trip: `485.5 PSI`; Quench rate: `1250.0 L/min`.

---

## 6. 📊 Sector Summary Matrix

| Chunk ID | Sector / Agency | Intent Identifier | Primary Keyword Triggers | Key Numeric Ground Truth | Expected Latency |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **0** | **Air Traffic Control** | `altitude_separation` | *separation, altitude, terminal, descent* | `3.0 nm`, `5.0 nm`, `10000 ft` | `< 0.03 ms` |
| **1** | **Air Traffic Control** | `transponder_squawk_code` | *squawk, transponder, emergency, 7700* | `7700`, `7600`, `7500` | `< 0.03 ms` |
| **2** | **Industrial Safety** | `reactor_mawp_threshold` | *reactor, mawp, pressure, quench* | `450.0 PSI`, `485.5 PSI`, `1250 L/min` | `< 0.03 ms` |
| **3** | **911 EMS** | `cpr_compression_ratio` | *cpr, compressions, cardiac, breathing* | `30:2 ratio`, `100-120 bpm`, `2.0 in` | `< 0.03 ms` |
| **4** | **911 EMS** | `severe_bleeding_tourniquet` | *bleeding, tourniquet, arterial, wound* | `2-3 inches` | `< 0.03 ms` |
| **5** | **911 EMS** | `pediatric_seizure_protocol` | *seizure, child, febrile, convulsions* | `5 minutes`, Priority `1` | `< 0.03 ms` |
| **6** | **Police Department** | `active_shooter_containment` | *shooter, shots fired, perimeter, standoff* | `2-4 officers`, `300 meters` | `< 0.03 ms` |
| **7** | **Police Department** | `officer_distress_codes` | *officer down, distress, 10-33, 10-99* | `10-33`, `10-78`, `10-99`, Priority `1` | `< 0.03 ms` |
| **8** | **Fire Department** | `high_rise_fire_staging` | *high rise fire, staging, standpipe, nozzle* | `2 floors`, `100.0 PSI`, `2.5 inch` | `< 0.03 ms` |
| **9** | **Fire Department** | `hazmat_bleve_isolation` | *bleve, propane tank, hazmat, standoff* | `1.0 mile` | `< 0.03 ms` |

---
**Verification Guarantee:** Every registered response is backed by exact numeric values enforced via AST + regex guardrails in `< 0.07 ms` before voice egress.
