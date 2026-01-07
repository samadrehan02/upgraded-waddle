"""Test script to verify all fixes work correctly."""

from medical_vocab_hi import SYMPTOMS, MEDICATIONS, DURATION_PATTERNS
from extract_hi import extract_symptoms, extract_medications
from segment_hi import segment_hi
from normalize_hi import normalize_hi
from diagnosis_from_doctor_hi import extract_doctor_diagnosis
import re

print("=" * 80)
print("TESTING FIXES")
print("=" * 80)

# Test 1: Duration extraction
test_text_1 = "मुझे परसों से बुखार लग रहा है"
sentences = segment_hi(test_text_1)
symptoms, _ = extract_symptoms(sentences)
print(f"\n✓ Test 1 - Duration extraction:")
print(f"  Input: {test_text_1}")
print(f"  Symptoms: {symptoms}")
assert any("परसों" in s.get("duration", "") for s in symptoms), "Duration not extracted!"
print("  PASSED ✓")

# Test 2: Hinglish weakness
test_text_2 = "वीकनेस लगती है"
sentences = segment_hi(test_text_2)
symptoms, _ = extract_symptoms(sentences)
print(f"\n✓ Test 2 - Hinglish 'weakness':")
print(f"  Input: {test_text_2}")
print(f"  Symptoms: {symptoms}")
assert any(s["name"] == "कमजोरी" for s in symptoms), "Weakness not recognized!"
print("  PASSED ✓")

# Test 3: Medication spelling variant
test_text_3 = "पेरासिटामोल लें"
meds = extract_medications(test_text_3)
print(f"\n✓ Test 3 - Medication spelling:")
print(f"  Input: {test_text_3}")
print(f"  Medications: {meds}")
assert "पैरासिटामोल" in meds, "Medication spelling variant not recognized!"
print("  PASSED ✓")

# Test 4: Diagnosis extraction
test_entries = [
    {"speaker": "doctor", "text": "यह बुखार का"},
    {"speaker": "doctor", "text": "केस लग रहा है"}
]
diagnoses = extract_doctor_diagnosis(test_entries)
print(f"\n✓ Test 4 - Diagnosis extraction:")
print(f"  Input: {test_entries}")
print(f"  Diagnosis: {diagnoses}")
assert len(diagnoses) > 0, "Diagnosis not extracted!"
assert "बुखार" in diagnoses[0], "Diagnosis doesn't contain 'बुखार'!"
print("  PASSED ✓")

# Test 5: Filler removal - Various cases
print(f"\n✓ Test 5 - Filler removal:")

# 5a: Simple filler line
test_5a = "हेलो हेलो आवाज आ रही है\nमुझे बुखार है"
cleaned_5a = normalize_hi(test_5a)
print(f"  5a) Simple filler:")
print(f"      Input: {repr(test_5a)}")
print(f"      Cleaned: {repr(cleaned_5a)}")
assert "हेलो" not in cleaned_5a, "हेलो not removed!"
assert "आवाज" not in cleaned_5a, "आवाज not removed!"
assert "बुखार" in cleaned_5a, "Medical content removed!"
print(f"      PASSED ✓")

# 5b: Multiple filler lines
test_5b = "हेलो\nजी\nमुझे सिर दर्द है\nठीक है"
cleaned_5b = normalize_hi(test_5b)
print(f"  5b) Multiple fillers:")
print(f"      Input: {repr(test_5b)}")
print(f"      Cleaned: {repr(cleaned_5b)}")
assert "हेलो" not in cleaned_5b and "जी" not in cleaned_5b and "ठीक है" not in cleaned_5b, "Fillers not removed!"
assert "सिर दर्द" in cleaned_5b, "Medical content removed!"
print(f"      PASSED ✓")

# 5c: Filler mixed with medical (should keep)
test_5c = "हाँ मुझे बुखार है"
cleaned_5c = normalize_hi(test_5c)
print(f"  5c) Mixed content (should keep line):")
print(f"      Input: {repr(test_5c)}")
print(f"      Cleaned: {repr(cleaned_5c)}")
# Line should be kept because it has medical content
assert "बुखार" in cleaned_5c, "Medical content removed!"
print(f"      PASSED ✓")

print("\n" + "=" * 80)
print("ALL TESTS PASSED! ✓✓✓")
print("=" * 80)
print("\nThe system is now ready for production testing.")
print("Key improvements:")
print("  1. Duration extraction (परसों से, आज से, etc.)")
print("  2. Hinglish symptom recognition (वीकनेस, weakness)")
print("  3. Medication spelling variants (पेरासिटामोल)")
print("  4. Better diagnosis pattern matching")
print("  5. Proper filler line removal")
print("  6. Strengthened LLM prompt to prevent hallucination")
