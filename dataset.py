from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

from torch.utils.data import DataLoader, Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SummarizationExample:
    """Single source → target pair with optional gold reference and doc ID."""
    source: str
    target: str
    reference: Optional[str] = None   # gold summary for eval (may equal target)
    doc_id: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Base dataset
# ─────────────────────────────────────────────────────────────────────────────

class ReMemorizeDataset(Dataset):
    """PyTorch Dataset wrapping a list of SummarizationExamples."""

    def __init__(self, examples: List[SummarizationExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> SummarizationExample:
        return self.examples[idx]

    def __iter__(self) -> Iterator[SummarizationExample]:
        return iter(self.examples)

    @classmethod
    def from_jsonl(
        cls,
        path: str,
        source_key: str = "source",
        target_key: str = "target",
        reference_key: Optional[str] = None,
        doc_id_key: Optional[str] = None,
        max_examples: Optional[int] = None,
    ) -> "ReMemorizeDataset":
        """
        Load from a JSONL file with configurable field names.

        Args:
            path           : path to .jsonl file
            source_key     : JSON key for the source text
            target_key     : JSON key for the target summary
            reference_key  : optional JSON key for a gold reference
            doc_id_key     : optional JSON key for document identifier
            max_examples   : truncate to first N examples if set
        """
        examples: List[SummarizationExample] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                examples.append(
                    SummarizationExample(
                        source=obj[source_key],
                        target=obj[target_key],
                        reference=obj.get(reference_key) if reference_key else None,
                        doc_id=str(obj[doc_id_key]) if doc_id_key else None,
                    )
                )
                if max_examples is not None and len(examples) >= max_examples:
                    break
        return cls(examples)


def _collate_fn(batch: List[SummarizationExample]) -> List[SummarizationExample]:
    """DataLoader collate: keep as a list (tokenisation happens inside the model)."""
    return batch


def build_dataloader(
    dataset: ReMemorizeDataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    """Build a DataLoader with a reproducible generator."""
    import torch
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_fn,
        num_workers=num_workers,
        generator=generator,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MIMIC-IV-Ext-BHC dataset
# ─────────────────────────────────────────────────────────────────────────────

class MIMICDataset(ReMemorizeDataset):
    """
    MIMIC-IV-Ext-BHC: EHR clinical notes → Brief Hospital Course (BHC).

    Real data format (MIMIC-IV-Ext-BHC, Johnson et al. 2023):
        {"hadm_id": "12345", "note": "<EHR text>", "bhc": "<BHC text>"}

    Access requires PhysioNet credentialing. Once downloaded, load with:
        dataset = MIMICDataset.from_jsonl("path/to/mimic_bhc.jsonl")

    For local testing without data access, use MIMICDataset.placeholder(n).
    """

    # Standard system prompt prepended to each source at inference time.
    SYSTEM_PROMPT = (
        "You are a clinical documentation assistant. "
        "Given the following hospital notes, write a concise Brief Hospital Course "
        "that accurately summarises the patient's admission, treatment, and outcome."
    )

    # Approximate token-length statistics from the real dataset.
    # Source: ~4 200 tokens (EHR notes),  Target: ~350 tokens (BHC)
    APPROX_SOURCE_TOKENS = 4_200
    APPROX_TARGET_TOKENS = 350

    @classmethod
    def from_jsonl(cls, path: str, max_examples: Optional[int] = None) -> "MIMICDataset":  # type: ignore[override]
        return super().from_jsonl(
            path,
            source_key="note",
            target_key="bhc",
            reference_key="bhc",
            doc_id_key="hadm_id",
            max_examples=max_examples,
        )

    @classmethod
    def placeholder(cls, n: int = 200, seed: int = 42) -> "MIMICDataset":
        """
        Generate n synthetic MIMIC-style examples for offline development.
        Covers the full structural diversity of real BHC notes without PHI.
        """
        rng = random.Random(seed)

        conditions = [
            "community-acquired pneumonia", "sepsis secondary to UTI",
            "COPD exacerbation", "acute decompensated heart failure",
            "acute kidney injury", "ischemic stroke", "upper GI bleed",
            "pulmonary embolism", "diabetic ketoacidosis", "hypertensive emergency",
        ]
        comorbidities = [
            "HTN, DM2, HLD", "CKD stage 3, CAD", "COPD, OSA, obesity",
            "atrial fibrillation, prior stroke", "CHF (EF 35%), CKD",
        ]
        interventions = [
            "IV piperacillin-tazobactam", "norepinephrine drip", "IV methylprednisolone",
            "IV furosemide diuresis", "continuous renal replacement therapy (CRRT)",
            "tPA thrombolysis", "PPI infusion + upper endoscopy",
            "anticoagulation with heparin bridge", "insulin drip", "labetalol drip",
        ]
        dispositions = [
            "discharged home with outpatient follow-up",
            "transferred to rehab facility",
            "discharged to skilled nursing facility",
            "expired due to multi-organ failure",
        ]

        examples: List[SummarizationExample] = []
        for i in range(n):
            cond = rng.choice(conditions)
            comorbid = rng.choice(comorbidities)
            intv = rng.choice(interventions)
            dispo = rng.choice(dispositions)
            age = rng.randint(42, 87)
            sex = "male" if rng.random() > 0.5 else "female"
            los = rng.randint(2, 18)
            hr = rng.randint(88, 125)
            sbp = rng.randint(88, 178)
            dbp = rng.randint(52, 100)
            temp = round(36.5 + rng.random() * 2.5, 1)
            spo2 = rng.randint(84, 99)

            source = (
                f"ADMISSION NOTE\n"
                f"Patient: {age}yo {sex} | Admitting diagnosis: {cond}\n"
                f"PMH: {comorbid}\n"
                f"Vitals: T {temp}°C, HR {hr}, BP {sbp}/{dbp}, SpO2 {spo2}%\n"
                f"HPI: Pt presented to ED with worsening symptoms over "
                f"{rng.randint(1, 7)} days. Evaluated and admitted for further management.\n"
                f"Hospital course: Started on {intv}. Monitored on telemetry. "
                f"Daily labs obtained. Dietary consult placed. Social work involved for dispo planning.\n"
                f"LOS: {los} days\n"
                f"Disposition: {dispo.capitalize()}.\n"
            )

            target = (
                f"{age}yo {sex} with history of {comorbid} admitted for {cond}. "
                f"Patient was treated with {intv} with "
                f"{'good' if 'expired' not in dispo else 'poor'} response. "
                f"Hospital course was notable for close monitoring of vitals and labs. "
                f"{dispo.capitalize()}."
            )

            examples.append(
                SummarizationExample(
                    source=source,
                    target=target,
                    reference=target,
                    doc_id=f"mimic_placeholder_{i:06d}",
                )
            )
        return cls(examples)


# ─────────────────────────────────────────────────────────────────────────────
# MTS-Dialog dataset
# ─────────────────────────────────────────────────────────────────────────────

class MTSDialogDataset(ReMemorizeDataset):
    """
    MTS-Dialog: doctor-patient conversation transcript → SOAP note.

    Real data format (Ben Abacha et al. 2023):
        {"ID": "...", "dialogue": "<transcript>", "note": "<SOAP note>"}

    Download from: https://github.com/abachaa/MTS-Dialog
    Once downloaded, load with:
        dataset = MTSDialogDataset.from_jsonl("path/to/mts_dialog.jsonl")

    For local testing without data access, use MTSDialogDataset.placeholder(n).
    """

    SOAP_TEMPLATE = (
        "SUBJECTIVE:\n{subj}\n\n"
        "OBJECTIVE:\n{obj}\n\n"
        "ASSESSMENT:\n{assess}\n\n"
        "PLAN:\n{plan}"
    )

    APPROX_SOURCE_TOKENS = 800
    APPROX_TARGET_TOKENS = 200

    @classmethod
    def from_jsonl(cls, path: str, max_examples: Optional[int] = None) -> "MTSDialogDataset":  # type: ignore[override]
        return super().from_jsonl(
            path,
            source_key="dialogue",
            target_key="note",
            reference_key="note",
            doc_id_key="ID",
            max_examples=max_examples,
        )

    @classmethod
    def from_huggingface(
        cls,
        split: str = "train",
        val_size: int = 150,
        test_size: int = 150,
        seed: int = 42,
        max_examples: Optional[int] = None,
    ) -> "MTSDialogDataset":
        """
        Load SubashNeupane/dataset_SOAP_summary from HuggingFace (1,473 examples).
        The HF dataset has only a train split, so we carve out val and test here.

        split: one of "train", "val", "test"
        val_size / test_size: number of examples held out for val and test
        """
        from datasets import load_dataset as hf_load
        hf = hf_load("SubashNeupane/dataset_SOAP_summary")["train"]

        all_examples = [
            SummarizationExample(
                source=row["input"],
                target=row["output"],
                reference=row["output"],
                doc_id=f"soap_{i:05d}",
            )
            for i, row in enumerate(hf)
        ]

        rng = random.Random(seed)
        rng.shuffle(all_examples)

        test_ex = all_examples[:test_size]
        val_ex  = all_examples[test_size:test_size + val_size]
        train_ex = all_examples[test_size + val_size:]

        chosen = {"train": train_ex, "val": val_ex, "test": test_ex}[split]
        if max_examples is not None:
            chosen = chosen[:max_examples]
        return cls(chosen)

    @classmethod
    def placeholder(cls, n: int = 200, seed: int = 42) -> "MTSDialogDataset":
        """
        Generate n synthetic MTS-Dialog-style examples for offline development.
        """
        rng = random.Random(seed)

        complaints = [
            "persistent headache", "sharp chest pain", "shortness of breath on exertion",
            "lower abdominal pain", "generalised fatigue and weakness",
            "knee pain worsened by stairs", "intermittent palpitations",
            "chronic lower back pain", "recurrent sore throat", "insomnia",
        ]
        exam_findings = [
            "vital signs within normal limits",
            "mild tachycardia (HR 102)",
            "decreased air entry at left base",
            "diffuse mild abdominal tenderness",
            "bilateral ankle oedema 1+",
        ]
        diagnoses = [
            "tension-type headache", "musculoskeletal chest pain",
            "mild intermittent asthma", "irritable bowel syndrome",
            "iron-deficiency anaemia", "patellofemoral pain syndrome",
            "paroxysmal supraventricular tachycardia", "lumbar radiculopathy",
            "group A streptococcal pharyngitis", "chronic insomnia disorder",
        ]
        plan_actions = [
            "prescribe ibuprofen PRN and follow up in 4 weeks",
            "order ECG and refer to cardiology",
            "start salbutamol inhaler and short-course prednisolone",
            "trial low-FODMAP diet and refer to gastroenterology if no improvement",
            "prescribe ferrous sulphate and repeat CBC in 6 weeks",
        ]

        examples: List[SummarizationExample] = []
        for i in range(n):
            cc = rng.choice(complaints)
            finding = rng.choice(exam_findings)
            dx = rng.choice(diagnoses)
            plan_str = rng.choice(plan_actions)
            duration = rng.randint(1, 30)
            severity = rng.choice(["mild", "moderate", "severe"])

            dialogue = (
                f"Doctor: Good morning. What brings you in today?\n"
                f"Patient: I've been having {cc} for about {duration} days. "
                f"It's been {severity}.\n"
                f"Doctor: I'm sorry to hear that. Any other symptoms — "
                f"fever, nausea, changes in appetite?\n"
                f"Patient: A bit of nausea, but no fever.\n"
                f"Doctor: Let me examine you.\n"
                f"[EXAMINATION]\n"
                f"Doctor: On examination, {finding}.\n"
                f"Patient: Is there anything serious?\n"
                f"Doctor: Based on the history and exam, this looks most like {dx}. "
                f"The plan is to {plan_str}.\n"
                f"Patient: Thank you, doctor.\n"
            )

            note = cls.SOAP_TEMPLATE.format(
                subj=(
                    f"Patient presents with {severity} {cc} for {duration} days. "
                    f"Associated mild nausea. Denies fever."
                ),
                obj=(
                    f"Examination: {finding.capitalize()}. "
                    f"No acute distress."
                ),
                assess=f"Most likely {dx}.",
                plan=f"Will {plan_str}. Patient educated on diagnosis and advised to return if worsening.",
            )

            examples.append(
                SummarizationExample(
                    source=dialogue,
                    target=note,
                    reference=note,
                    doc_id=f"mts_placeholder_{i:06d}",
                )
            )
        return cls(examples)
