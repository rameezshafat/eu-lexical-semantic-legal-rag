# Inter-Annotator Agreement Task — EU Climate Law Retrieval

Thank you for helping with this annotation task. It should take approximately **30–45 minutes**.

---

## Background

We have built a retrieval system that searches EU climate legislation to answer natural-language legal questions. We need a second human annotator to judge the same 14 queries that the primary annotator already judged, so we can measure how consistently humans agree on relevance.

The measure we compute is **Cohen's κ** (kappa), a standard inter-annotator agreement statistic used in information retrieval research.

---

## What You Need to Judge

For each of the 14 queries below, you will see a list of **candidate documents** (EU legislative acts identified by their CELEX ID). For each candidate, judge whether it is **relevant** or **not relevant** to the query.

**Relevant** means: this EU legislative act directly addresses the question asked. A person trying to answer the question would need to consult this act.

**Not relevant** means: this act does not directly answer the question. It may share vocabulary with the question, but reading it would not help answer the query.

---

## Annotation File Format

Please fill in the JSON file below. For each query, add the CELEX IDs you judge **relevant** to the `"relevant_celex_ids"` list. Leave the list empty if you judge nothing relevant.

Do **not** remove or add to the `"candidate_celex_ids"` list — those are fixed.

```json
{
  "annotator": "YOUR NAME HERE",
  "date": "YYYY-MM-DD",
  "queries": [
    {
      "query_id": "q001",
      "query": "If my company runs cargo ships that call at European ports, what do we have to measure and report about their CO2?",
      "candidate_celex_ids": ["32015R0757", "32008L0101", "32003L0087", "32023L0959"],
      "relevant_celex_ids": []
    },
    {
      "query_id": "q002",
      "query": "How much pollution is allowed in total under the EU carbon market, and how do factories get their permits?",
      "candidate_celex_ids": ["32003L0087", "32023L0959", "32015R0757", "32023R0955"],
      "relevant_celex_ids": []
    },
    {
      "query_id": "q005",
      "query": "When goods come into the EU, how do they figure out how much carbon went into making them?",
      "candidate_celex_ids": ["32023R0956", "32003L0087", "32023R1773", "32025R0486"],
      "relevant_celex_ids": []
    },
    {
      "query_id": "q015",
      "query": "What exact tests prove an activity really helps fight climate change for green-investment purposes?",
      "candidate_celex_ids": ["32021R2139", "32020R0852", "32021R2178", "32020R1818"],
      "relevant_celex_ids": []
    },
    {
      "query_id": "q016",
      "query": "What do investment funds have to tell people about how green their products actually are?",
      "candidate_celex_ids": ["32021R2178", "32020R0852", "32021R2139", "32023R2631"],
      "relevant_celex_ids": []
    },
    {
      "query_id": "q022",
      "query": "How did flying get pulled into the EU carbon market, and which flights does it cover?",
      "candidate_celex_ids": ["32008L0101", "32003L0087", "32023L0959", "32015R0757"],
      "relevant_celex_ids": []
    },
    {
      "query_id": "q023",
      "query": "What do shipping companies now owe under the carbon market for their vessels' emissions?",
      "candidate_celex_ids": ["32023L0959", "32015R0757", "32008L0101", "32003L0087"],
      "relevant_celex_ids": []
    },
    {
      "query_id": "q037",
      "query": "Why do some heavy industries get free carbon permits, and how is that amount worked out?",
      "candidate_celex_ids": ["32003L0087", "32023L0959", "32023R0955", "32023R0956"],
      "relevant_celex_ids": []
    },
    {
      "query_id": "q039",
      "query": "What's the system that keeps track of who holds which carbon allowances?",
      "candidate_celex_ids": ["32018R0208", "32003L0087", "32015R1844", "32023L0959"],
      "relevant_celex_ids": []
    },
    {
      "query_id": "q040",
      "query": "Can a country carry forward, stockpile or trade away its yearly emissions budget?",
      "candidate_celex_ids": ["32018R0842", "32018R0841", "32023R0857", "32021R1119"],
      "relevant_celex_ids": []
    },
    {
      "query_id": "q041",
      "query": "How is the baseline for forest emissions set when judging how well a country is doing on land accounting?",
      "candidate_celex_ids": ["32018R0841", "32013R0525", "32023R0839", "32018R0842"],
      "relevant_celex_ids": []
    },
    {
      "query_id": "q044",
      "query": "How is the carbon-border bill trimmed to account for the free permits domestic producers already get?",
      "candidate_celex_ids": ["32025R2620", "32023R0956", "32003L0087", "32023L0959"],
      "relevant_celex_ids": []
    },
    {
      "query_id": "q049",
      "query": "What's the 'don't make anything else worse' test for activities in the EU green-investment rules?",
      "candidate_celex_ids": ["32021R2139", "32020R0852", "32020R1818", "32021R2178"],
      "relevant_celex_ids": []
    },
    {
      "query_id": "q067",
      "query": "What yearly land-sector emission and removal limits apply to countries from 2026 to 2029?",
      "candidate_celex_ids": ["32026R0893", "32018R0841", "32023R0839", "32018R0842"],
      "relevant_celex_ids": []
    }
  ]
}
```

---

## CELEX ID Reference

CELEX IDs identify EU legislation. The format is: `3` + year + instrument type + number.

| CELEX ID | Instrument |
|---|---|
| 32003L0087 | EU Emissions Trading System (ETS) Directive 2003/87 |
| 32008L0101 | Aviation ETS Directive 2008/101 |
| 32013R0525 | GHG Monitoring Mechanism Regulation 525/2013 |
| 32015R0757 | Maritime MRV Regulation 2015/757 |
| 32015R1844 | Registry implementing Regulation 2015/1844 (Kyoto credits) |
| 32018R0208 | Union Registry implementing Regulation 2018/208 |
| 32018R0841 | LULUCF Regulation 2018/841 (land use, land-use change and forestry) |
| 32018R0842 | Effort Sharing Regulation 2018/842 (non-ETS sector targets) |
| 32020R0852 | EU Taxonomy Regulation 2020/852 |
| 32020R1818 | Paris-aligned Benchmarks Delegated Regulation 2020/1818 |
| 32021R1119 | European Climate Law 2021/1119 |
| 32021R2139 | Taxonomy Climate Delegated Regulation 2021/2139 |
| 32021R2178 | Taxonomy Disclosure Delegated Regulation 2021/2178 |
| 32023L0959 | ETS Revision Directive 2023/959 |
| 32023R0839 | LULUCF Amendment 2023/839 |
| 32023R0856 | ETS Revision implementing details |
| 32023R0857 | Effort Sharing Amendment 2023/857 |
| 32023R0955 | Social Climate Fund Regulation 2023/955 |
| 32023R0956 | CBAM Regulation 2023/956 (Carbon Border Adjustment Mechanism) |
| 32023R1773 | CBAM Transitional implementing Regulation 2023/1773 |
| 32023R2631 | European Green Bond Standards Regulation 2023/2631 |
| 32025R0486 | CBAM Declarant Authorisation implementing Regulation 2025/486 |
| 32025R2620 | CBAM Free-allocation adjustment implementing Regulation 2025/2620 |
| 32026R0893 | LULUCF annual limits implementing Regulation 2026/893 |

---

## Instructions

1. Read each query carefully — they are written in informal language on purpose.
2. For each candidate in the list, ask: "Would this act help answer the question?"
3. Add the relevant CELEX IDs to `"relevant_celex_ids"`. Leave empty if nothing is relevant.
4. Do not look up the actual texts unless you are completely uncertain — judge from the CELEX reference table above.
5. Save the completed file as `iaa_annotator2.json` and send it back.

**If you are unsure**, err toward **including** rather than excluding. We are measuring agreement, not testing you.

---

## How results are used

Your annotations are combined with the primary annotator's judgments. We compute Cohen's κ to verify that the gold standard was not written to be intentionally easy or artificially consistent. A κ ≥ 0.60 is the threshold for publication-quality annotation.

Thank you for your time.
