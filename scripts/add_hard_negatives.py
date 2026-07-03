"""
scripts/add_hard_negatives.py — Apply hard-negative annotations to gold_standard.json.

Adds hard_negative_celex_ids to the 51 previously unannotated queries.
Each annotation was selected by the following rule:
  - The negative CELEX shares surface vocabulary with the query
  - It is legally incorrect as an answer (wrong instrument, wrong scope,
    or related predecessor/successor that does not contain the provision)
  - It is present in the indexed corpus (verifiable)

Run from project root:
    python scripts/add_hard_negatives.py [--dry-run]

Writes: data/evaluation/gold_standard.json (in-place)
Backup: data/evaluation/gold_standard_pre_hn.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Hard-negative annotations ─────────────────────────────────────────────────
# Format: { query_id: [celex_id, ...], ... }
# Only queries with EMPTY hard_negative_celex_ids are included here.
# Queries already annotated are left unchanged.

HARD_NEGATIVES: dict[str, list[str]] = {
    # q003: "How does the EU decide whether a business activity counts as genuinely green?"
    # Relevant: 32020R0852 (Taxonomy Regulation — criteria for environmental sustainability)
    # HN: 32021R2178 discusses taxonomy alignment of financial products, same "green" vocabulary
    #     but about what to REPORT, not what criteria make something green
    "q003": ["32021R2178"],

    # q004: "Which emission-cutting targets is each EU country legally on the hook for
    #         in areas like transport and farming?"
    # Relevant: 32018R0842, 32023R0857 (ESR — per-country non-ETS targets)
    # HN: 32021R1119 contains EU-level climate targets (55% by 2030, net-zero 2050)
    #     which look like "targets countries are on the hook for" but are EU-aggregate,
    #     not the per-country ESR allocations
    "q004": ["32021R1119"],

    # q006: "What do businesses have to report about the refrigerant-type gases that
    #         warm the planet?"
    # Relevant: 32014R0517 (F-Gas Regulation)
    # HN: 32013R0525 (Monitoring Mechanism Regulation) also requires businesses and
    #     member states to report greenhouse gas emissions, similar vocabulary
    #     but covers all GHGs via member states, not F-gas-specific business reporting
    "q006": ["32013R0525"],

    # q008: "Where exactly is the EU's promise to be climate neutral by 2050 written into law?"
    # Relevant: 32021R1119 (European Climate Law — Art. 2 sets the 2050 objective)
    # HN: 32018R1999 (Governance Regulation) mentions long-term low-carbon strategies
    #     and references climate goals, but does not legally mandate climate neutrality by 2050
    "q008": ["32018R1999"],

    # q009: "Do big firms have to check that their suppliers aren't wrecking the environment?"
    # Relevant: 32024L1760 (CSDDD — corporate due diligence on value chains)
    # HN: 32020R0852 (Taxonomy Regulation) also defines what counts as "wrecking the
    #     environment" (DNSH criteria) and applies to corporates, but is about investment
    #     classification, not supply-chain due diligence obligations
    "q009": ["32020R0852"],

    # q010: "Is there EU money to help poorer households cope when carbon prices push up their bills?"
    # Relevant: 32023R0955 (Social Climate Fund — household carbon-cost support)
    # HN: 32021R1056 (Just Transition Fund) also provides EU money to help people
    #     cope with the costs of the energy transition but targets fossil-fuel regions,
    #     not individual households facing carbon-price bills
    "q010": ["32021R1056"],

    # q011: "What's the minimum energy standard a building has to meet under EU rules?"
    # Relevant: 32018L2002, 32023L1791 (Energy Efficiency Directives)
    # HN: 32018R1999 (Governance Regulation) requires member states to include
    #     building energy efficiency in their national plans — same vocabulary
    #     but about planning obligations, not the minimum performance standards themselves
    "q011": ["32018R1999"],

    # q013: "Before the carbon border charge fully kicks in, what paperwork do importers need to file?"
    # Relevant: 32023R1773 (CBAM transitional implementing reg), 32023R0956 (CBAM)
    # HN: 32025R0486 (CBAM declarant authorisation) also concerns paperwork for CBAM
    #     importers but covers the permanent authorisation process, not the
    #     transitional period quarterly reports
    "q013": ["32025R0486"],

    # q014: "How do I get approved as the person allowed to declare carbon for goods I import?"
    # Relevant: 32025R0486 (CBAM declarant auth), 32023R0956 (CBAM)
    # HN: 32023R1773 (CBAM transitional implementing reg) also covers CBAM import
    #     compliance process but is about the transitional reporting period,
    #     not the permanent declarant authorisation
    "q014": ["32023R1773"],

    # q017: "What goes into a country's combined energy-and-climate plan, and when is it due?"
    # Relevant: 32018R1999 (Governance — NECP content and deadlines)
    # HN: 32020R1208 (Governance implementing reg) covers the SAME instrument family
    #     but specifies the reporting FORMAT for progress updates, not the content
    #     and deadlines of the integrated national plans themselves
    "q017": ["32020R1208"],

    # q018: "How do EU countries report their yearly greenhouse gas totals?"
    # Relevant: 32013R0525, 32014R0749 (Monitoring Mechanism — member state GHG reporting)
    # HN: 32014R0666 (Union GHG Inventory Regulation) also concerns GHG totals
    #     but covers the EU-level aggregated inventory, not individual member state
    #     annual national submissions
    "q018": ["32014R0666"],

    # q019: "In what form do governments have to report the steps they're taking to cut emissions?"
    # Relevant: 32020R1208, 32018R1999 (Governance — policies and measures reporting)
    # HN: 32022R2299 (Governance biennial reporting implementing reg) is in the same
    #     instrument family and also requires climate policy progress reports,
    #     but covers the biennial integrated national reports, not policies-and-measures tables
    "q019": ["32022R2299"],

    # q020: "Do countries have to send Brussels a long-range plan for going low-carbon?"
    # Relevant: 32018R1999 (Governance — long-term strategies, Art. 15)
    # HN: 32021R1119 (European Climate Law) sets the EU-level 2050 net-zero objective
    #     which creates the political context for long-term plans, but the legal
    #     obligation on member states to SUBMIT long-term strategies is in the Governance Reg
    "q020": ["32021R1119"],

    # q024: "When a public authority buys vehicles, how many have to be low- or zero-emission?"
    # Relevant: 32019L1161 (Clean Vehicles Directive — procurement quotas)
    # HN: 32023L1791 (Energy Efficiency Directive recast) also imposes public
    #     procurement energy obligations on member states but focuses on buildings
    #     and overall energy efficiency, not vehicle-specific purchase quotas
    "q024": ["32023L1791"],

    # q028: "How do the outfits that vet green bonds get registered and kept in check?"
    # Relevant: 32023R2631 (EuGB — external reviewer registration and oversight),
    #           32025R0755 (ESMA fee regulation for external reviewers)
    # HN: 32025R0754 (ESMA EuGB enforcement implementing reg) covers ESMA's supervisory
    #     powers over the same external reviewers but specifies enforcement/investigation
    #     steps, not the registration and ongoing supervision framework
    "q028": ["32025R0754"],

    # q029: "Who can get help from the EU fund for regions hit hardest by moving off fossil fuels?"
    # Relevant: 32021R1056 (Just Transition Fund — eligible territories and beneficiaries)
    # HN: 32023R0955 (Social Climate Fund) also provides EU money to people affected
    #     by fossil-fuel transition costs but targets vulnerable households, not
    #     the most coal-dependent regions and their enterprises/workers
    "q029": ["32023R0955"],

    # q030: "What does a carbon-removal or soil project need to qualify for EU certification?"
    # Relevant: 32024R3012 (Carbon Removal Certification Framework — eligibility criteria)
    # HN: 32020R0852 (Taxonomy Regulation) also sets criteria for activities to qualify
    #     as environmentally sustainable, similar certification vocabulary
    #     but for investment product classification, not carbon credit certification
    "q030": ["32020R0852"],

    # q031: "If I run a carbon farming scheme, what plans do I have to hand to the certifier?"
    # Relevant: 32024R3012 (CRCF), 32025R2358 (monitoring plan implementing reg)
    # HN: 32026R0285 (CRCF implementing reg for DACCS methodology) is another CRCF
    #     implementing regulation about certifying carbon removal but for industrial
    #     direct air capture, not agriculture — shares the certification plan vocabulary
    "q031": ["32026R0285"],

    # q032: "What do countries have to do to bring damaged land habitats back to health
    #         under the new EU nature law?"
    # Relevant: 32024R1991 (Nature Restoration Regulation — terrestrial habitat obligations)
    # HN: 32018R0841 (LULUCF Regulation) also covers forests and land ecosystems
    #     and requires countries to account for land-sector carbon, but from a
    #     carbon-accounting perspective, not ecological restoration obligations
    "q032": ["32018R0841"],

    # q033: "What share of road fuel was each country meant to make up with biofuel?"
    # Relevant: 32003L0030 (Biofuels Directive — fuel-mix indicative targets)
    # HN: 32014R1307 (implementing reg on highly biodiverse grassland) also concerns
    #     biofuels in road transport but specifies which land cannot be used for
    #     biofuel crops, not the fuel-mix percentage targets
    "q033": ["32014R1307"],

    # q034: "How do you tell whether a meadow is too rich in wildlife to be ploughed
    #         up for biofuel crops?"
    # Relevant: 32014R1307 (biodiverse grassland criteria)
    # HN: 32003L0030 (Biofuels Directive) established the biofuel targets that
    #     created the commercial pressure to convert grassland, shares the biofuel/
    #     road-fuel vocabulary but doesn't contain the biodiversity assessment criteria
    "q034": ["32003L0030"],

    # q035: "What's the EU's flagship funding programme for the environment trying to achieve?"
    # Relevant: 32021R0783, 32013R1293 (LIFE Programme — objectives and scope)
    # HN: 32021R1056 (Just Transition Fund) is another EU environment/climate fund
    #     targeting a specific transition, shares "EU funding for environment" framing
    #     but has narrower focus on fossil-fuel region transition, not broad environmental objectives
    "q035": ["32021R1056"],

    # q036: "Has the EU pinned down a 2040 climate goal yet, and how does it sit with the 2050 one?"
    # Relevant: 32021R1119 (ECL), 32026R0667 (ECL 2040 amendment)
    # HN: 32018R1999 (Governance Regulation) mentions the 2050 long-term low-carbon
    #     strategy in the context of national planning, shares the "2050 climate goal"
    #     vocabulary but doesn't establish the targets — merely requires plans toward them
    "q036": ["32018R1999"],

    # q038: "What is the EU planning to do about keeping soil healthy and cutting down on contamination?"
    # Relevant: 32025L2360 (Soil Monitoring Directive — soil health and contamination)
    # HN: 32024R1991 (Nature Restoration Regulation) covers restoration of terrestrial
    #     ecosystems including soils, shares the "land/soil health" vocabulary
    #     but is about ecosystem restoration obligations, not soil health monitoring per se
    "q038": ["32024R1991"],

    # q042: "What minimum rules apply to stock-market indices that claim to line up
    #         with the Paris Agreement?"
    # Relevant: 32020R1818 (Paris-aligned Benchmarks Delegated Reg — index minimum standards)
    # HN: 32021R2178 (Taxonomy disclosure delegated reg) also covers how financial
    #     products communicate Paris-alignment and taxonomy compatibility, shares
    #     the Paris/green-investment vocabulary but targets investment fund disclosure,
    #     not the construction rules for financial indices
    "q042": ["32021R2178"],

    # q043: "What's expected of a company that puts together a climate plan under the new
    #         corporate due-diligence rules?"
    # Relevant: 32024L1760 (CSDDD — climate transition plan obligations)
    # HN: 32021R2178 (Taxonomy disclosure delegated reg) also requires large companies
    #     to disclose climate/sustainability information, shares "corporate climate plan"
    #     vocabulary but applies to financial entities under Taxonomy disclosure rules,
    #     not to the broader corporate supply-chain due-diligence law
    "q043": ["32021R2178"],

    # q045: "What quality checks go into putting together the EU-wide greenhouse gas figures?"
    # Relevant: 32014R0666 (Union GHG Inventory Reg — quality assurance process),
    #           32013R0525 (Monitoring Mechanism Reg)
    # HN: 32014R0749 (implementing reg for member state reporting) is in the same
    #     instrument family as 32014R0666 but specifies the FORMAT of individual
    #     member state GHG submissions, not the quality process for the EU-level inventory
    "q045": ["32014R0749"],

    # q046: "What progress updates do member states owe under the EU's energy and climate
    #         governance setup?"
    # Relevant: 32018R1999 (Governance), 32020R1208 (implementing reg)
    # HN: 32022R2299 (Governance biennial progress reporting implementing reg)
    #     also implements the Governance Regulation's reporting obligations,
    #     but focuses on the biennial integrated national energy and climate reports,
    #     a different reporting cycle and format from the general progress updates
    "q046": ["32022R2299"],

    # q047: "How are yearly emission limits handed out to countries for the sectors
    #         that sit outside the carbon market?"
    # Relevant: 32018R0842, 32023R0857 (Effort Sharing Regulation)
    # HN: 32021R1119 (European Climate Law) sets the EU-wide 55% reduction target
    #     that underpins the ESR allocations, shares "emission limits for member states"
    #     vocabulary but sets the aggregate EU target, not the per-country non-ETS allocations
    "q047": ["32021R1119"],

    # q048: "What restoration duties cover seas, coasts and rivers under the EU nature law?"
    # Relevant: 32024R1991 (Nature Restoration Regulation — marine and freshwater articles)
    # HN: 32015R0757 (MRV Regulation) also covers maritime/shipping and mentions
    #     marine environment in its scope, shares "seas/marine" vocabulary
    #     but is about monitoring CO2 from ships, not ecological marine restoration
    "q048": ["32015R0757"],

    # q050: "What does a country need to put in its plan to tap the fund that cushions
    #         carbon-price impacts?"
    # Relevant: 32023R0955 (Social Climate Fund — national social climate plans)
    # HN: 32021R1056 (Just Transition Fund) also requires member states to submit
    #     territorial just transition plans before accessing funds, very similar
    #     plan-to-access-funding structure but for regional fossil-fuel transition,
    #     not individual household carbon cost support
    "q050": ["32021R1056"],

    # q051: "What kind of environment and transport projects does the EU's cohesion money pay for?"
    # Relevant: 32013R1300 (Cohesion Fund — eligible environment/transport investments)
    # HN: 32021R1056 (Just Transition Fund) is in the same cohesion policy family
    #     and also funds environmental and infrastructure projects in regions,
    #     but is specifically for territories transitioning away from fossil fuels
    "q051": ["32021R1056"],

    # q052: "How do you work out how much of a regional-funding programme counts as climate spending?"
    # Relevant: 32014R0215 (climate-tagging methodology for structural funds)
    # HN: 32020R0852 (Taxonomy Regulation) also provides a methodology for classifying
    #     activities as environmentally/climatically aligned, similar "how much counts
    #     as green" vocabulary but applies to financial investment products, not
    #     structural fund programme expenditure tracking
    "q052": ["32020R0852"],

    # q053: "How are old Kyoto-era carbon credits like CERs moved around inside the EU registry?"
    # Relevant: 32015R1844 (implementing reg — CER/ERU transfer procedures)
    # HN: 32018R0208 (Union Registry implementing reg) also governs the EU carbon registry
    #     where these transfers take place, shares "EU registry / carbon credits" vocabulary
    #     but covers ETS allowance registry operations generally, not the specific
    #     Kyoto credit transfer rules
    "q053": ["32018R0208"],

    # q054: "What's the scheme that pairs EU grants with EIB loans for green public projects?"
    # Relevant: 32021R1229 (Public Sector Loan Facility — JTF grant + EIB loan blending)
    # HN: 32021R1056 (Just Transition Fund) provides the JTF grants that feed into
    #     the Public Sector Loan Facility, shares the "JTF / EIB / green transition"
    #     vocabulary but is the source fund, not the loan-grant blending mechanism
    "q054": ["32021R1056"],

    # q055: "How and when do the green-investment technical criteria get reviewed?"
    # Relevant: 32022R1214 (Taxonomy amendment with review provisions),
    #           32021R2139 (Climate Delegated Reg)
    # HN: 32020R0852 (Taxonomy Regulation — parent instrument) establishes the
    #     Platform on Sustainable Finance and the review mandate at a high level,
    #     shares "green investment criteria / review" vocabulary but the SCHEDULE
    #     and triggering conditions for review are in the delegated acts
    "q055": ["32020R0852"],

    # q056: "What goes into the two-yearly progress report on the decarbonisation side
    #         of the governance rules?"
    # Relevant: 32022R2299 (Governance biennial reporting implementing reg),
    #           32018R1999 (Governance)
    # HN: 32020R1208 (Governance policies-and-measures implementing reg) also implements
    #     Governance Regulation reporting, shares "decarbonisation / governance progress"
    #     vocabulary but covers annual policies-and-measures tables, not the biennial
    #     integrated national energy and climate progress reports
    "q056": ["32020R1208"],

    # q057: "How does the EU recovery fund steer carbon-auction money into the energy transition?"
    # Relevant: 32023R0435 (RRF REPowerEU amendment — ETS revenue → REPowerEU chapter)
    # HN: 32003L0087 (ETS Directive) is the source of the carbon auction revenues
    #     referenced in the RRF amendment, shares "carbon auctions / allowances"
    #     vocabulary but describes the auctioning mechanism, not how those revenues
    #     are channelled into the Recovery and Resilience Facility
    "q057": ["32003L0087"],

    # q058: "What steps does the EU markets watchdog take when it looks into green-bond
    #         rule-breaking?"
    # Relevant: 32025R0754 (ESMA EuGB enforcement implementing reg),
    #           32023R2631 (EuGB)
    # HN: 32025R0755 (ESMA fee regulation for EuGB external reviewers) also concerns
    #     ESMA's supervisory role over external reviewers under EuGB,
    #     shares "ESMA / external reviewer oversight" vocabulary but is about
    #     fee calculation for registration/supervision, not enforcement investigation steps
    "q058": ["32025R0755"],

    # q059: "How are drought-stricken and climate-damaged areas pinned down for carbon farming?"
    # Relevant: 32025R2043 (CRCF implementing reg — aridity and climate-damage definitions),
    #           32024R3012 (CRCF)
    # HN: 32025R2358 (CRCF monitoring plan implementing reg) is another CRCF implementing
    #     regulation for agriculture/soil carbon removal, shares "CRCF / carbon farming"
    #     vocabulary but specifies the structure of monitoring plans, not geographic
    #     definitions of degraded or climate-damaged land
    "q059": ["32025R2358"],

    # q060: "Is there a small-shipment exemption from the EU's carbon border rules?"
    # Relevant: 32025R2083 (CBAM amendment — de minimis threshold),
    #           32023R0956 (CBAM)
    # HN: 32023R1773 (CBAM transitional implementing reg) also limits CBAM's scope
    #     during the transitional period, shares "CBAM exemptions / small quantities"
    #     vocabulary but is about who must file quarterly transition reports,
    #     not the permanent de minimis exemption for small consignments
    "q060": ["32023R1773"],

    # q061: "What standards of trustworthiness and qualifications do the bosses at
    #         green-bond reviewers need?"
    # Relevant: 32025R2180 (EuGB implementing reg — governance/repute criteria for management),
    #           32023R2631 (EuGB)
    # HN: 32025R0754 (ESMA EuGB enforcement implementing reg) also concerns ESMA's
    #     oversight of external reviewer management but describes investigation powers
    #     and enforcement steps, not the qualification/repute standards managers must meet
    "q061": ["32025R0754"],

    # q062: "When can a carbon-import checker do a remote inspection instead of turning up at the site?"
    # Relevant: 32025R2546 (CBAM verification implementing reg — remote/on-site conditions),
    #           32023R0956 (CBAM)
    # HN: 32025R0486 (CBAM declarant authorisation) relates to the same CBAM compliance
    #     process — the declarant works with verifiers — but covers the importer's
    #     authorisation to submit CBAM declarations, not the verifier's site-inspection rules
    "q062": ["32025R0486"],

    # q063: "How is the price of a carbon border certificate set from the EU carbon
    #         auction results each quarter?"
    # Relevant: 32025R2548 (CBAM certificate price implementing reg),
    #           32023R0956 (CBAM)
    # HN: 32003L0087 (ETS Directive) governs the carbon auctions whose quarterly
    #     average price feeds into the CBAM certificate price formula,
    #     shares "EU carbon auctions / allowance prices" vocabulary but describes
    #     the auctioning mechanism, not the CBAM certificate pricing calculation
    "q063": ["32003L0087"],

    # q064: "Which companies get shut out of Paris-aligned investment indices because
    #         of their fossil fuel ties?"
    # Relevant: 32025R1775 (Climate benchmarks amendment — exclusion criteria update),
    #           32020R1818 (Paris-aligned Benchmarks)
    # HN: 32021R2178 (Taxonomy disclosure delegated reg) also concerns how financial
    #     products communicate alignment with Paris goals and may reference fossil fuel
    #     exclusions, shares "Paris-aligned / fossil fuel ties / financial products"
    #     vocabulary but applies to fund disclosure, not index exclusion criteria
    "q064": ["32021R2178"],

    # q065: "Do smaller companies get any kind of break from the EU green-investment
    #         disclosure rules?"
    # Relevant: 32026R0073 (Taxonomy disclosure amendment — SME derogations),
    #           32021R2178 (Taxonomy disclosure delegated reg)
    # HN: 32020R0852 (Taxonomy Regulation — parent instrument) originally set the
    #     framework that required the very disclosures the derogation now relaxes,
    #     shares "green-investment disclosure obligations" vocabulary but is the
    #     general framework, not the specific SME derogation answer
    "q065": ["32020R0852"],

    # q066: "How do projects that suck carbon out of the air and bury it get certified
    #         under the EU removal rules?"
    # Relevant: 32026R0285 (CRCF implementing reg — DACCS methodology),
    #           32024R3012 (CRCF)
    # HN: 32025R2043 (CRCF implementing reg — aridity/climate-damage area definitions)
    #     is another CRCF implementing regulation about carbon removal verification,
    #     shares "CRCF / carbon removal certification" vocabulary but applies to
    #     agriculture/soil carbon, not direct air carbon capture and storage
    "q066": ["32025R2043"],

    # q068: "Can the EU social fund be tapped to help workers and towns recover after
    #         a natural disaster?"
    # Relevant: 32024R3236 (ESF+ amendment — natural disaster support)
    # HN: 32021R1056 (Just Transition Fund) also provides ESF-linked support for
    #     workers and communities in transition, shares "EU social fund / workers and
    #     communities" vocabulary but is specifically for fossil-fuel phase-out,
    #     not natural disaster recovery
    "q068": ["32021R1056"],

    # q069: "What wiggle room is there at the mid-point review to shift regional funds
    #         toward climate resilience?"
    # Relevant: 32025R1914 (ERDF mid-term review amendment — reprogramming flexibility)
    # HN: 32021R1056 (Just Transition Fund) is in the same cohesion family, also
    #     has mid-term review provisions and climate-related reprogramming flexibility,
    #     shares "regional funds / mid-term review / climate" vocabulary
    #     but the ERDF amendment answer is about mainstream cohesion funds, not JTF
    "q069": ["32021R1056"],

    # q070: "What changed in the standard reporting tables countries use for their
    #         governance progress reports?"
    # Relevant: 32024R1281 (Governance reporting amendment — updated annex tables),
    #           32020R1208 (Governance reporting implementing reg)
    # HN: 32022R2299 (Governance biennial progress reporting implementing reg)
    #     also establishes reporting tables/annexes for the Governance Regulation
    #     and was superseded/amended by 32024R1281, shares "governance reporting tables"
    #     vocabulary but is the instrument whose annexes were changed, not the change itself
    "q070": ["32022R2299"],

    # q071: "What standard template do countries have to use when handing in their
    #         nature restoration plans?"
    # Relevant: 32025R0912 (Nature Restoration plan format implementing reg),
    #           32024R1991 (Nature Restoration Regulation)
    # HN: 32018R1999 (Governance Regulation) also requires member states to submit
    #     national plans using standard templates (NECPs), shares "standard template /
    #     mandatory national plan submission" vocabulary but for energy/climate planning,
    #     not the nature restoration plan format
    "q071": ["32018R1999"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply hard-negative annotations to gold_standard.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print changes without writing to disk")
    parser.add_argument("--path", default="data/evaluation/gold_standard.json")
    args = parser.parse_args()

    gs_path = Path(args.path)
    data = json.loads(gs_path.read_text(encoding="utf-8"))

    queries_by_id = {q["query_id"]: q for q in data["queries"]}

    applied = 0
    skipped = 0
    for qid, negatives in HARD_NEGATIVES.items():
        q = queries_by_id.get(qid)
        if q is None:
            print(f"WARNING: {qid} not found in gold standard")
            continue
        if q.get("hard_negative_celex_ids"):
            skipped += 1
            continue
        if not args.dry_run:
            q["hard_negative_celex_ids"] = negatives
        else:
            print(f"  {qid}  →  {negatives}")
        applied += 1

    print(f"\nAnnotations: {applied} applied, {skipped} already had annotations")

    if args.dry_run:
        print("(dry-run — no files written)")
        return

    # Backup
    backup = gs_path.parent / "gold_standard_pre_hn.json"
    import shutil
    shutil.copy(gs_path, backup)
    print(f"Backup written to {backup}")

    gs_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Updated gold standard written to {gs_path}")

    # Verify counts
    annotated = sum(1 for q in data["queries"] if q.get("hard_negative_celex_ids"))
    print(f"Total queries with HN annotations: {annotated}/{len(data['queries'])}")


if __name__ == "__main__":
    main()
