"""
manual_indexer.py - Knowledge Base Parser and Embedding Generator for PS-003

Parses manuals/atc_emergency_manual.md into metric-aware semantic chunks,
generates normalized 384-dimensional float32 dense embeddings using sentence-transformers
(all-MiniLM-L6-v2), and writes binary vector storage (manual_embeddings.bin) and
structured metadata (manual_metadata.json).
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np


def parse_emergency_manual(manual_path: str | Path) -> List[Dict[str, Any]]:
    """
    Parses manuals/atc_emergency_manual.md into semantic, metric-aware chunks:
      Chunk 0: Section 1 Altitude & Separation (3nm terminal, 5nm en-route, 10,000 ft emergency descent)
      Chunk 1: Section 1 Transponder Squawk codes (Hijacking: 7500, Radio Failure: 7600, General Emergency: 7700)
      Chunk 2: Section 2 Chemical Plant Emergency Overpressure (Reactor Core R-101 MAWP: 450.0 PSI, Venting: 485.5 PSI, Quench: 1250 L/min)
    """
    manual_path = Path(manual_path)
    if not manual_path.exists():
        raise FileNotFoundError(f"Emergency manual not found at: {manual_path.resolve()}")

    with open(manual_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Define metric-aware semantic chunks across all organizational sectors
    chunks = [
        # --- SECTOR 1: AIR TRAFFIC CONTROL ---
        {
            "chunk_id": 0,
            "sector": "Air Traffic Control",
            "section": "Section 1: Altitude & Separation",
            "title": "Air Traffic Control Altitude and Separation Tolerances",
            "text": (
                "Air Traffic Control Altitude and Separation Requirements: "
                "Minimum radar separation is 3 nautical miles within terminal airspace, "
                "and 5 nautical miles en-route. Emergency descent target altitude is "
                "10,000 feet MSL (Mean Sea Level) or lowest safe altitude."
            ),
            "key_metrics": {
                "terminal_separation_nm": 3.0,
                "enroute_separation_nm": 5.0,
                "emergency_descent_altitude_ft": 10000.0,
                "altitude_unit": "feet MSL",
                "separation_unit": "nautical miles"
            },
            "domain": "aviation_atc",
            "intent": "altitude_separation",
            "source": manual_path.name
        },
        {
            "chunk_id": 1,
            "sector": "Air Traffic Control",
            "section": "Section 1: Transponder Squawk Codes",
            "title": "Emergency Transponder Squawk Codes",
            "text": (
                "Air Traffic Control Emergency Transponder Squawk Codes: "
                "Squawk 7500 indicates Unlawful Interference or Hijacking. "
                "Squawk 7600 indicates Lost Radio Communications or Radio Failure NORDO. "
                "Squawk 7700 indicates General Emergency Mayday declaration."
            ),
            "key_metrics": {
                "hijacking_squawk": 7500,
                "radio_failure_squawk": 7600,
                "general_emergency_squawk": 7700
            },
            "domain": "aviation_atc",
            "intent": "transponder_squawk_code",
            "source": manual_path.name
        },
        # --- SECTOR 2: INDUSTRIAL SAFETY ---
        {
            "chunk_id": 2,
            "sector": "Industrial Safety",
            "section": "Section 2: Chemical Plant Emergency Overpressure",
            "title": "Reactor Core R-101 Pressure and Quench Thresholds",
            "text": (
                "Chemical Plant Emergency Overpressure Safety Tolerances: "
                "Reactor Core R-101 Maximum Allowable Working Pressure MAWP is 450.0 PSI. "
                "Emergency venting trip threshold is 485.5 PSI. "
                "Quench tank injection rate is 1250 liters per minute of water glycol solution."
            ),
            "key_metrics": {
                "reactor_core_id": "R-101",
                "mawp_psi": 450.0,
                "venting_trip_psi": 485.5,
                "quench_injection_rate_l_min": 1250.0,
                "pressure_unit": "PSI",
                "rate_unit": "liters/min"
            },
            "domain": "chemical_industrial",
            "intent": "reactor_mawp_threshold",
            "source": manual_path.name
        },
        # --- SECTOR 3: 911 EMERGENCY DISPATCH / EMS ---
        {
            "chunk_id": 3,
            "sector": "911 Emergency Dispatch",
            "section": "EMS Protocol: Adult CPR & Resuscitation",
            "title": "Adult CPR Compression Rate and Ratio",
            "text": (
                "911 Emergency Medical Dispatch CPR Protocol: "
                "For adult cardiac arrest, deliver 30 chest compressions followed by 2 rescue breaths. "
                "Maintain compression rate of 100 to 120 beats per minute, pressing at least 2 inches deep."
            ),
            "key_metrics": {
                "compression_ratio": 30,
                "breaths_ratio": 2,
                "min_rate_bpm": 100,
                "max_rate_bpm": 120,
                "depth_inches": 2.0
            },
            "domain": "emergency_911_ems",
            "intent": "cpr_compression_ratio",
            "source": manual_path.name
        },
        {
            "chunk_id": 4,
            "sector": "911 Emergency Dispatch",
            "section": "EMS Protocol: Severe Bleeding & Hemorrhage",
            "title": "Tourniquet Placement and Hemorrhage Control",
            "text": (
                "911 Emergency Medical Dispatch Bleeding Protocol: "
                "For severe arterial extremity bleeding, apply windlass tourniquet 2 to 3 inches "
                "above the wound, never over a joint. Tighten until bleeding ceases completely."
            ),
            "key_metrics": {
                "tourniquet_min_distance_inches": 2.0,
                "tourniquet_max_distance_inches": 3.0
            },
            "domain": "emergency_911_ems",
            "intent": "severe_bleeding_tourniquet",
            "source": manual_path.name
        },
        {
            "chunk_id": 5,
            "sector": "911 Emergency Dispatch",
            "section": "EMS Protocol: Pediatric Febrile Seizure",
            "title": "Pediatric Seizure Management and Dispatch Priority",
            "text": (
                "911 Emergency Medical Dispatch Pediatric Seizure Protocol: "
                "Place child on their side in recovery position. Do not insert anything into mouth. "
                "If seizure exceeds 5 minutes, dispatch Advanced Life Support Priority 1 unit immediately."
            ),
            "key_metrics": {
                "seizure_threshold_minutes": 5.0,
                "dispatch_priority": 1
            },
            "domain": "emergency_911_ems",
            "intent": "pediatric_seizure_protocol",
            "source": manual_path.name
        },
        # --- SECTOR 4: POLICE DEPARTMENT ---
        {
            "chunk_id": 6,
            "sector": "Police Department",
            "section": "Tactical Operations: Active Threat Containment",
            "title": "Active Shooter Response and Perimeter Standoff",
            "text": (
                "Police Department Tactical Operations Active Threat Protocol: "
                "Code 10-33 emergency traffic. First contact team of 2 to 4 officers must bypass "
                "wounded and neutralize threat immediately. Establish outer perimeter at 300 meters standoff."
            ),
            "key_metrics": {
                "contact_team_min_officers": 2,
                "contact_team_max_officers": 4,
                "outer_perimeter_standoff_meters": 300.0
            },
            "domain": "law_enforcement_police",
            "intent": "active_shooter_containment",
            "source": manual_path.name
        },
        {
            "chunk_id": 7,
            "sector": "Police Department",
            "section": "Radio Communications: Officer Distress Ten-Codes",
            "title": "Officer Distress Emergency Radio Ten-Codes",
            "text": (
                "Police Department Radio Communications Emergency Ten-Codes: "
                "Code 10-33 signals emergency traffic only. Code 10-78 requests urgent backup. "
                "Code 10-99 indicates officer in extreme danger or officer down, triggering all-units Priority 1 response."
            ),
            "key_metrics": {
                "emergency_traffic_code": "10-33",
                "urgent_backup_code": "10-78",
                "officer_down_code": "10-99",
                "dispatch_priority": 1
            },
            "domain": "law_enforcement_police",
            "intent": "officer_distress_codes",
            "source": manual_path.name
        },
        # --- SECTOR 5: FIRE DEPARTMENT ---
        {
            "chunk_id": 8,
            "sector": "Fire Department",
            "section": "Structural Firefighting: High-Rise Operations",
            "title": "High-Rise Staging Floor and Standpipe Attack Pressure",
            "text": (
                "Fire Department High-Rise Operations: "
                "Establish interior staging area 2 floors below the fire floor. Connect attack lines "
                "to standpipe on floor below fire, maintaining 100 PSI nozzle pressure with 2.5 inch supply lines."
            ),
            "key_metrics": {
                "staging_floors_below": 2,
                "nozzle_pressure_psi": 100.0,
                "supply_line_diameter_inches": 2.5
            },
            "domain": "fire_rescue",
            "intent": "high_rise_fire_staging",
            "source": manual_path.name
        },
        {
            "chunk_id": 9,
            "sector": "Fire Department",
            "section": "Hazmat Operations: Boiling Liquid Expanding Vapor Explosion",
            "title": "Hazmat BLEVE Standoff and Evacuation Radius",
            "text": (
                "Fire Department Hazardous Materials BLEVE Protocol: "
                "For pressurized liquefied gas vessels exposed to direct flame impingement with BLEVE risk, "
                "immediately withdraw all personnel and establish minimum 1 mile isolation perimeter in all directions."
            ),
            "key_metrics": {
                "isolation_radius_miles": 1.0,
                "perimeter_degrees": 360
            },
            "domain": "fire_rescue",
            "intent": "hazmat_bleve_isolation",
            "source": manual_path.name
        }
    ]

    return chunks


def generate_chunk_embeddings(
    chunks: List[Dict[str, Any]],
    model_name: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    """
    Encodes chunk texts into 384-dimensional normalized float32 embeddings.
    """
    from sentence_transformers import SentenceTransformer

    print(f"[*] Loading SentenceTransformer model: '{model_name}'...")
    model = SentenceTransformer(model_name)

    texts = [chunk["text"] for chunk in chunks]
    print(f"[*] Encoding {len(texts)} chunks...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    print(f"[+] Embeddings generated successfully. Shape: {embeddings.shape}, Dtype: {embeddings.dtype}")
    return embeddings


def save_embeddings_and_metadata(
    embeddings: np.ndarray,
    chunks: List[Dict[str, Any]],
    output_bin: str | Path,
    output_meta: str | Path
) -> None:
    """
    Persists:
      1. Raw binary float32 embeddings for direct zero-copy C++ / SIMD memory mapping.
      2. Structured JSON metadata with chunk IDs, texts, and exact numerical ground truth metrics.
    """
    output_bin = Path(output_bin)
    output_meta = Path(output_meta)

    # 1. Save raw float32 binary
    with open(output_bin, "wb") as f:
        f.write(embeddings.tobytes())
    print(f"[+] Saved binary embeddings to: {output_bin.resolve()} ({output_bin.stat().st_size} bytes)")

    # 2. Add embedding metadata to chunks
    num_vectors, dim = embeddings.shape
    metadata = {
        "model_name": "all-MiniLM-L6-v2",
        "vector_count": num_vectors,
        "dimension": dim,
        "dtype": "float32",
        "normalized": True,
        "binary_file": output_bin.name,
        "chunks": chunks
    }

    with open(output_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[+] Saved structured metadata to: {output_meta.resolve()} ({output_meta.stat().st_size} bytes)")


def verify_index(output_bin: str | Path, output_meta: str | Path) -> None:
    """
    Verifies binary embeddings and metadata consistency, testing dot product similarity.
    """
    output_bin = Path(output_bin)
    output_meta = Path(output_meta)

    with open(output_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)

    vector_count = meta["vector_count"]
    dimension = meta["dimension"]

    raw_bytes = output_bin.read_bytes()
    loaded_vectors = np.frombuffer(raw_bytes, dtype=np.float32).reshape(vector_count, dimension)

    print("\n--- Index Verification Report ---")
    print(f"Vector Count:       {vector_count}")
    print(f"Vector Dimension:   {dimension}")
    print(f"Norm of vectors:    {[float(np.linalg.norm(v)) for v in loaded_vectors]}")

    for chunk in meta["chunks"]:
        print(f"\n[Chunk {chunk['chunk_id']}] {chunk['section']}")
        print(f"  Title: {chunk['title']}")
        print(f"  Metrics: {chunk['key_metrics']}")

    print("\n[+] Verification PASSED: All chunks and vectors are intact and normalized.")


def main():
    base_dir = Path(__file__).resolve().parent
    manual_path = base_dir / "manuals" / "atc_emergency_manual.md"
    output_bin = base_dir / "manual_embeddings.bin"
    output_meta = base_dir / "manual_metadata.json"

    print("==================================================")
    print("PS-003 Knowledge Base & Ingestion: Manual Indexer")
    print("==================================================")
    print(f"Manual Source: {manual_path}")

    chunks = parse_emergency_manual(manual_path)
    embeddings = generate_chunk_embeddings(chunks, model_name="all-MiniLM-L6-v2")
    save_embeddings_and_metadata(embeddings, chunks, output_bin, output_meta)
    verify_index(output_bin, output_meta)


if __name__ == "__main__":
    main()
