# Cardiomegaly Clinical Decision Support Dashboard

<p align="center">
  <img src="https://img.shields.io/badge/Domain-Medical%20AI%20%7C%20Radiology-0052cc.svg?style=for-the-badge&logo=medscape&logoColor=white" alt="Medical AI">
  <img src="https://img.shields.io/badge/Specialty-Cardiothoracic%20Imaging-a8663c.svg?style=for-the-badge" alt="Cardiothoracic Imaging">
  <img src="https://img.shields.io/badge/Task-Cardiomegaly%20Detection%20(CTR)-dc2626.svg?style=for-the-badge&logo=heart&logoColor=white" alt="Cardiomegaly Detection">
  <img src="https://img.shields.io/badge/System-Clinical%20Decision%20Support%20(CDSS)-16a34a.svg?style=for-the-badge" alt="CDSS">
  <img src="https://img.shields.io/badge/Architecture-Dual%20Ensembled%20(GNN%20%2B%20CNN)-6366f1.svg?style=for-the-badge" alt="Architecture">
  <img src="https://img.shields.io/badge/Streamlit-1.50.0-ff4b4b.svg?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/PyTorch-2.9.0-ee4c2c.svg?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Intended%20Use-Research%20%26%20Decision%20Support-0f172a.svg?style=for-the-badge" alt="Intended Use">
</p>

---

### Medical AI & Clinical Taxonomy Tags
`#MedicalAI` · `#RadiologyAI` · `#ClinicalDecisionSupport` · `#CardiothoracicImaging` · `#ChestRadiography` · `#Cardiomegaly` · `#CardiothoracicRatio` · `#AnatomicalSegmentation` · `#GraphNeuralNetworks` · `#AlgorithmicFairness` · `#ConsensusVerification` · `#SafetyValidation` · `#ExplainableAI` · `#HumanInTheLoop`

---

### Clinical & Technical System Specifications

| Clinical Attribute | System Specification |
| :--- | :--- |
| **Clinical Modality** | Digital & Film-Screen Chest Radiography (CXR: DICOM, PNG, JPEG) |
| **Pathology Target** | Cardiomegaly (Enlarged Cardiac Silhouette) |
| **Biomarker** | Cardiothoracic Ratio (**CTR** = Cardiac Width / Thoracic Width) |
| **Vision Extractors** | Dual Ensembled: **Tool A** (HybridGNet GNN) + **Tool B** (ianpan/chest-x-ray-basic CNN) |
| **Consensus Threshold** | Inter-tool agreement tolerance $\le 3.0\%$ CTR delta |
| **Decision Architecture** | **Layer 1**: Demographic-Aware Evidence-Based Threshold Rules (Age, Sex, Ancestry, View)<br>**Layer 2**: Multi-Factor Safety Gate (Inspiration, Quality, Rotation, Confidence) |
| **Interpretability (XAI)** | Deterministic Rule ID, Evidence Tier (A/B/C), Written Clinical Rationale, Segmentation Mask Overlays |
| **Intended Clinical Role** | Second-reader Clinical Decision Support System (CDSS) / Radiologist Triage |

---

So here's the deal: this is a demographic-aware decision support tool for spotting cardiomegaly (an enlarged heart) on chest X-rays. You upload an X-ray, punch in some patient info, and the system tells you whether it thinks the heart looks enlarged, normal, or whether a human should just take a look themselves.

It's not trying to replace a radiologist. Think of it as a second pair of (very literal, very consistent) eyes that flags things worth a closer look.

## What's actually happening under the hood

There's a lot going on when you hit "Run Analysis," so let's walk through it.

### Step 1: Two vision tools independently measure the heart

We don't trust a single model's opinion here — we run **two separate segmentation tools** on your X-ray and see if they agree.

- **Tool A — HybridGNet** (`extraction_tool_a.py`): a landmark-based GNN model from IEEE TMI 2022 that traces the outline of both lungs and the heart with anatomically-plausible contours.
- **Tool B — ianpan/chest-x-ray-basic** (`extraction_tool_b.py`): a Hugging Face model that segments lungs/heart, and *also* tells us the view (AP/PA/Lateral), estimated age, and estimated sex — handy for sanity-checking whatever the user typed in.

Both tools independently compute the **Cardiothoracic Ratio (CTR)** — basically, heart width divided by thorax width, expressed as a percentage. This is the classic clinical proxy for "is the heart too big."

### Step 2: Do the two tools agree?

If Tool A and Tool B's CTR estimates differ by more than 3%, we don't trust either one blindly — the case gets automatically flagged for human review instead of being pushed through the decision logic. If they agree, we average them into a single "resolved CTR."

### Step 3: The rule-based decision engine

This is the demographic-aware part. `decision_engine.py` takes the resolved CTR plus the patient's age, gender, ancestry, and view (AP/PA), and checks it against a lookup table of evidence-backed CTR thresholds. Different demographic groups have genuinely different normal ranges for CTR, and the engine accounts for that instead of using one blanket cutoff for everyone.

Out comes one of three verdicts:
- **No Cardiomegaly**
- **Cardiomegaly Present**
- **Flag for Review** (when the evidence isn't strong enough to be confident either way)

### Step 4: The safety gate (Layer 2)

Even if the rule engine is confident, `validation_gate.py` runs a set of sanity checks — image quality, inspiration adequacy, rotation, tool confidence, etc. If any of these fail, the gate **overrides** the verdict and forces a "Flag for Review," no matter what the rule engine said. Safety first.

### Step 5: You get a full report

The dashboard shows you the final verdict, both tools' raw CTR numbers, the resolved CTR, which rule fired, the evidence tier behind it, a plain-English clinical rationale, a breakdown of every safety gate check, and Tool B's segmentation overlay so you can visually confirm the heart/lung boundaries it detected.

## Project structure

```
.
├── dashboard.py            # Streamlit UI — the thing you actually run
├── pipeline.py              # Orchestrates metadata parsing, tool agreement, and the two decision layers
├── decision_engine.py        # The demographic-aware rule engine (Layer 1)
├── validation_gate.py         # Safety checks that can override the verdict (Layer 2)
├── extraction_tool_a.py       # Tool A: HybridGNet segmentation
├── extraction_tool_b.py       # Tool B: ianpan/chest-x-ray-basic segmentation
├── samples/                  # Up to 10 sample chest X-rays for users with none of their own
└── requirements.txt
```

## Setting it up

**1. Clone the repo and get into it.**
```bash
git clone <your-repo-url>
cd Github_Cardiomegally
```

**2. Set up HybridGNet's weights.** Tool A reuses the model code and weights from the public HybridGNet Hugging Face Space, so you need to pull that down once:
```bash
git lfs install
git clone https://huggingface.co/spaces/ngaggion/Chest-x-ray-HybridGNet-Segmentation hybridgnet_space
```
This creates a `hybridgnet_space/` folder that `extraction_tool_a.py` imports from directly. If this folder isn't there, Tool A just quietly disables itself and the dashboard runs on Tool B alone.

**3. Create a virtual environment** (we built this on Python 3.9.11, so stick close to that if you can):
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

**4. Install dependencies:**
```bash
pip install -r requirements.txt
```

**5. Run it:**
```bash
streamlit run dashboard.py
```
Streamlit will pop open a browser tab. That's it, you're live.

## Using the dashboard

You've got two ways to feed it an image:
- **Upload your own** chest X-ray (jpg/png/dcm)
- **Pick a sample** from the built-in set if you don't have one handy — great for a quick demo

Fill in age, gender, ancestry, and view (or leave view on "Auto" and let Tool B guess it), tick the image-quality checkboxes if they apply, hit **Run Analysis**, and read the verdict.

## Hosting it live

Since this needs a running Python backend (not just static files), plain GitHub Pages won't cut it. The easiest free option is **Streamlit Community Cloud** — point it at your GitHub repo and it deploys automatically. Hugging Face Spaces works too if you'd rather keep it alongside your ML stuff.

## A quick disclaimer

This tool produces a decision-support signal, not a diagnosis. It's built to assist, not replace, review by a qualified radiologist or physician. Every verdict comes with a clinical rationale precisely so a human can sanity-check the reasoning, not just take the label at face value.
