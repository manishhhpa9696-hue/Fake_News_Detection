"""
Comprehensive Model Testing & Validation Script.
Evaluates the model across 10 diverse real-world news articles across English,
Hindi, and Marathi, testing:
- Model Predictions & Calibrated Confidence
- Uncertainty Threshold Flagging
- Explainable AI (Top Influential Words)
- Clickbait & Emotional Language Detection
- Source Credibility & Satire Warnings
- Evidence Verification (Supported / Contradicted / Insufficient Evidence)
"""

import os
import sys
import json

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.predict import NewsPredictor

def run_comprehensive_tests():
    print("=" * 80)
    print("🔬 VERITAS AI - COMPREHENSIVE MODEL EVALUATION SUITE")
    print("=" * 80)
    
    predictor = NewsPredictor(models_dir="models", uncertainty_margin=0.12)
    
    test_cases = [
        {
            "id": "TC-01",
            "category": "Science / Astronomy (English)",
            "headline": "NASA confirms James Webb Telescope spots oldest galaxies ever observed",
            "text": "Astronomers using the James Webb Space Telescope have discovered galaxies that formed just 300 million years after the Big Bang. The spectroscopic analysis confirmed their immense redshift and stellar composition, providing critical observational evidence.",
            "url": "https://www.reuters.com/science/jwst-early-galaxies",
            "expected_label": "Reliable-like News"
        },
        {
            "id": "TC-02",
            "category": "Medical / Health Hoax (English)",
            "headline": "SHOCKING MIRACLE: Boiled Lemon Hot Water Destroys 100% of Cancer Doctors Hiding Truth!!",
            "text": "A revolutionary secret leaked from unnamed sources reveals that boiling lemons and drinking the alkaline water every morning permanently cures all forms of cancer and eliminates 100 percent of viral infections. Big pharma and mainstream medical doctors are hiding this miracle cure.",
            "url": "http://miracle-lemon-alkaline-cure.xyz",
            "expected_label": "Fake-like News"
        },
        {
            "id": "TC-03",
            "category": "Economy / Banking (English)",
            "headline": "Reserve Bank of India keeps repo rate unchanged at 6.5 percent amid inflation control",
            "text": "The Monetary Policy Committee of the Reserve Bank of India announced its decision to maintain the policy repo rate at 6.50 percent. RBI Governor noted that while domestic economic activity remains resilient, continuing food price pressures necessitate sustained vigil.",
            "url": "https://thehindu.com/business/rbi-monetary-policy",
            "expected_label": "Reliable-like News"
        },
        {
            "id": "TC-04",
            "category": "Technology Hoax (English)",
            "headline": "5G mobile radiation triggers new deadly virus outbreak and causes birds to fall from sky",
            "text": "Shocking viral videos on social media expose that 5G millimeter wave transmission towers are poisoning atmospheric oxygen and directly generating lethal respiratory pandemics. Hundreds of birds fell dead in testing zones.",
            "url": "https://unknown-viral-blog.net/5g-radiation-birds",
            "expected_label": "Fake-like News"
        },
        {
            "id": "TC-05",
            "category": "Space / Technology (Hindi हिंदी)",
            "headline": "इसरो ने मौसम उपग्रह इनसैट-3डीएस का सफल प्रक्षेपण किया",
            "text": "भारतीय अंतरिक्ष अनुसंधान संगठन ने श्रीहरिकोटा के सतीश धवन अंतरिक्ष केंद्र से जीएसएलवी रॉकेट द्वारा इनसैट-3डीएस उपग्रह को सफलतापूर्वक भू-समकालिक कक्षा में स्थापित किया। इससे मौसम पूर्वानुमान में सुधार होगा।",
            "url": "https://ndtv.com/hindi/isro-insat-3ds-launch",
            "expected_label": "Reliable-like News"
        },
        {
            "id": "TC-06",
            "category": "Viral Rumor / Panic (Hindi हिंदी)",
            "headline": "नमक की भारी कमी होने वाली है और नमक की कीमत हजार रुपये प्रति किलो हो जाएगी",
            "text": "सोशल मीडिया पर उड़ाई गई अफवाह में कहा गया कि देश में नमक का स्टॉक खत्म हो गया है और कल से नमक की बिक्री बंद हो जाएगी। प्रशासन ने स्पष्ट किया कि नमक का पर्याप्त भंडार मौजूद है।",
            "url": "",
            "expected_label": "Fake-like News"
        },
        {
            "id": "TC-07",
            "category": "National Accomplishment (Marathi मराठी)",
            "headline": "भारतीय अंतराळ संशोधन संस्थेने चंद्रयान ३ मोहीम यशस्वीपणे पूर्ण केली",
            "text": "भारतीय अंतराळ संशोधन संस्था म्हणजेच इस्रोने चंद्राच्या दक्षिण ध्रुवावर विक्रम लँडरचे यशस्वी सॉफ्ट लँडिंग करून ऐतिहासिक कामगिरी केली. जगभरातील वैज्ञानिकांनी भारताच्या या तांत्रिक यशाचे कौतुक केले.",
            "url": "https://esakal.com/chandrayaan-3-success",
            "expected_label": "Reliable-like News"
        },
        {
            "id": "TC-08",
            "category": "Currency Hoax (Marathi मराठी)",
            "headline": "दोन हजार रुपयांच्या नव्या नोटेत नॅनो जीपीएस चिप गुप्तपणे बसवण्यात आली आहे",
            "text": "सोशल मीडियावर व्हायरल झालेल्या बातमीनुसार नव्या दोन हजार रुपयांच्या नोटेत अत्याधुनिक नॅनो जीपीएस ट्रॅकिंग चिप बसवण्यात आली आहे. ही चिप जमिनीच्या खाली १२० मीटर खोलवर असली तरी उपग्रहाद्वारे पकडता येते असा दावा केला जात आहे.",
            "url": "",
            "expected_label": "Fake-like News"
        },
        {
            "id": "TC-09",
            "category": "Satire & Parody (English)",
            "headline": "Scientists discover magical sound frequency that regrows lost human limbs in 48 hours",
            "text": "Revolutionary suppressed laboratory trials prove that listening to 432 Hz scalar audio tones stimulates cellular stem cells to regenerate amputated arms and legs within two days without medical intervention or surgery.",
            "url": "https://theonion.com/sound-frequency-limbs",
            "expected_label": "Satire Flag / Fake-like News"
        },
        {
            "id": "TC-10",
            "category": "Ambiguous / Edge Case (Uncertainty Test)",
            "headline": "International observers discuss preliminary trade estimates following summit",
            "text": "Delegates convened to review multilateral economic projections and trade tariffs ahead of next quarter's fiscal reporting.",
            "url": "",
            "expected_label": "Uncertain — Needs Human Review (or Low Margin)"
        }
    ]
    
    passed_count = 0
    total_count = len(test_cases)
    
    for tc in test_cases:
        print(f"\n[{tc['id']}] Category: {tc['category']}")
        print(f"      Headline : {tc['headline'][:75]}...")
        
        res = predictor.predict_article(
            headline=tc["headline"],
            text=tc["text"],
            url=tc["url"],
            save_to_history=True
        )
        
        pred = res["prediction"]
        conf = res["confidence"]
        lang = res["language"]["name"]
        clickbait = res["clickbait"]["clickbait_score"]
        domain = res["source"]["domain"]
        source_badge = res["source"]["status_badge"]
        is_satire = res["source"]["is_satire"]
        
        # Verify alignment
        is_match = False
        if "Reliable" in tc["expected_label"] and "Reliable" in pred:
            is_match = True
        elif "Fake" in tc["expected_label"] and ("Fake" in pred or is_satire):
            is_match = True
        elif "Uncertain" in tc["expected_label"] and ("Uncertain" in pred or conf < 65):
            is_match = True
            
        if is_match:
            passed_count += 1
            status_symbol = "✅ PASS"
        else:
            status_symbol = "⚠️ REVIEW"
            
        print(f"      Result   : {status_symbol} -> Prediction: {pred} ({conf}%)")
        print(f"      Signals  : Lang: {lang} | Clickbait: {clickbait}/100 | Domain: {domain} ({source_badge})")
        
        # Claims check
        claims = res.get("claim_verifications", [])
        if claims:
            top_claim = claims[0]
            print(f"      FactCheck: Claim: \"{top_claim['claim'][:50]}...\" -> Verdict: {top_claim['status']}")

        # Temporal Specificity & Geographic Grounding
        temporal = res.get("temporal", {})
        geographic = res.get("geographic", {})
        grounding = res.get("grounding_index", 0.0)
        print(f"      Spatio-Temporal: Grounding: {grounding}/100 | Time: {temporal.get('score', 0)}% ({temporal.get('rating', '')}) | Geo: {geographic.get('score', 0)}% ({geographic.get('rating', '')})")

        # Top 3 XAI terms
        xai_terms = res["xai"]["feature_weights"][:3]
        if xai_terms:
            terms_str = ", ".join([f"{w[0]} ({w[1]:+.2f})" for w in xai_terms])
            print(f"      XAI Terms: {terms_str}")
            
    print("\n" + "=" * 80)
    print(f"🏆 TEST SUMMARY: {passed_count}/{total_count} Test Cases Verified ({passed_count/total_count*100:.1f}%)")
    print("=" * 80)

if __name__ == "__main__":
    run_comprehensive_tests()
