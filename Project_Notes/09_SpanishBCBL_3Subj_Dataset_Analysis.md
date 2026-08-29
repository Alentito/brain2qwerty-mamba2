# SpanishBCBL Dataset Analysis (3 Subjects: S15, S16, S6)

## 1. Dataset Overview & Recording Profile
The **SpanishBCBL (`Pinet2024Meg`)** benchmark measures continuous typing from human participants writing memorised Spanish sentences under a 306-channel whole-head magnetoencephalography (MEG) helmet.

* **Modality**: 306-Channel Elekta Neuromag MEG (204 Planar Gradiometers + 102 Magnetometers)
* **Preprocessed Sampling Rate**: 50 Hz ($20\text{ ms}$ per sample)
* **Keystroke Window**: $500\text{ ms}$ ($-200\text{ ms}$ motor preparation to $+300\text{ ms}$ tactile feedback, 25 samples/window)
* **Active Subjects**: $S15$, $S16$, $S6$

```
======================================================================
Split             Keystrokes       Sentences     Mean Keys/Sent
======================================================================
Train                 17,811             456               39.1
Val                    2,211              66               33.5
Test                   2,280              54               42.2
----------------------------------------------------------------------
Total                 22,302             576               38.7
======================================================================
```

---

## 2. Participant Keystroke Contributions
* **Subject S16**: 9,754 keystrokes (43.7%)
* **Subject S15**: 7,756 keystrokes (34.8%)
* **Subject S6**: 4,792 keystrokes (21.5%)
* Total Windows: **22,302 windows** across 576 unique sentence trials.

---

## 3. Linguistic & Character Frequency Profile
Spanish natural language statistics govern the character label distribution:
1. **Space token (`' '`)**: 2,922 occurrences (13.1%) — most frequent token (word boundary marker).
2. **Vowels (`'a'`, `'e'`, `'o'`, `'i'`, `'u'`)**:
   - `'a'`: 2,611
   - `'e'`: 2,304
   - `'o'`: 1,720
   - `'i'`: 1,490
   - `'u'`: 684
3. **High-Frequency Consonants**: `'s'` (2,017), `'l'` (1,881), `'n'` (1,540), `'r'` (1,510), `'c'` (1,402), `'d'` (1,150).
4. **Rare Characters**: `'x'`, `'k'`, `'w'`, `'z'` ($< 50$ occurrences).

---

## 4. Electrophysiological Signal Dynamics (Evoked ERF & GFP)
* **Motor Preparation Dip ($-50\text{ ms}$ to $0\text{ ms}$)**: Bereitschaftspotential / readiness field observed across premotor and motor cortical sensors.
* **Somatosensory Feedback Peak ($+20\text{ ms}$ to $+60\text{ ms}$)**: Sharp global field power (GFP) deflection corresponding to tactile key striking and kinesthetic mechanoreceptor feedback.
* **Sensor Variance**: Maximum signal power is localized over the **sensorimotor parietal-rolandic MEG channels**.

---

## 5. Generated Figures
All high-resolution EDA figures are saved in [`dataset_eda_out/`](../dataset_eda_out):
* `01_character_distribution.png`: 29-Class character frequency histogram.
* `02_subject_split_breakdown.png`: Keystrokes per subject across train/val/test splits.
* `03_sentence_length_distribution.png`: Histogram and KDE of sentence lengths.
* `04_meg_evoked_response.png`: Grand-average 306-channel butterfly plot + Global Field Power (GFP).
