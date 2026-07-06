"""
Build an NCBI-curated-like gene list for insulin / estrogen / insulin resistance pathways.
Uses KEGG + GO annotations via public APIs (KEGG REST + mygene.info).
Outputs: ncbi_curated_ie_gene_list.tsv  (columns: gene_id, symbol, pathway)
"""
import urllib.request
import urllib.parse
import json
import pandas as pd
from collections import defaultdict

OUT_TSV = "ncbi_curated_ie_gene_list.tsv"

# ---------------------------------------------------------------------------
# 1. KEGG pathways (mouse)
# ---------------------------------------------------------------------------
KEGG_PATHWAYS = {
    "Insulin_signaling":    "mmu04910",   # Insulin signaling pathway
    "Insulin_resistance":   "mmu04931",   # Insulin resistance
    "Estrogen_signaling":   "mmu04915",   # Estrogen signaling pathway
}

# ---------------------------------------------------------------------------
# 2. GO terms (mouse, downloaded via mygene.info)
# ---------------------------------------------------------------------------
GO_TERMS = {
    "Insulin_signaling":    ["GO:0008286",   # insulin receptor signaling pathway
                             "GO:0032868",   # response to insulin
                             "GO:0006006"],  # glucose metabolic process
    "Insulin_resistance":   ["GO:0042593",   # glucose homeostasis
                             "GO:0071333",   # cellular response to glucose stimulus
                             "GO:0032869"],  # cellular response to insulin stimulus
    "Estrogen_signaling":   ["GO:0030520",   # intracellular estrogen receptor signaling pathway
                             "GO:0043627",   # response to estrogen
                             "GO:0008210"],  # estrogen metabolic process
}


def kegg_link(pathway_id: str):
    """Return set of mmu:XXXX Entrez IDs belonging to a KEGG pathway."""
    url = f"https://rest.kegg.jp/link/mmu/{pathway_id}"
    with urllib.request.urlopen(url, timeout=60) as r:
        text = r.read().decode()
    ids = set()
    for line in text.strip().splitlines():
        if "\t" in line:
            _, gene = line.split("\t")
            ids.add(gene.replace("mmu:", ""))
    return ids


def mygene_batch_query(entrez_ids, scopes="entrezgene", fields="ensembl.gene,symbol"):
    """Batch query mygene.info; return list of dicts."""
    url = "https://mygene.info/v3/query"
    results = []
    # mygene.info batch limit is ~1000; chunk safely
    for i in range(0, len(entrez_ids), 500):
        chunk = entrez_ids[i:i+500]
        params = urllib.parse.urlencode({
            "q": ",".join(chunk),
            "scopes": scopes,
            "fields": fields,
            "species": "mouse"
        })
        req = urllib.request.Request(url, data=params.encode(),
                                     headers={"Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read().decode())
        results.extend(data)
    return results


def go_genes(go_id: str):
    """Query mygene.info for genes annotated to a GO term (paginated)."""
    hits = []
    page_size = 1000
    from_ = 0
    while True:
        params = urllib.parse.urlencode({
            "q": go_id,
            "scopes": "go",
            "fields": "ensembl.gene,symbol,entrezgene",
            "species": "mouse",
            "size": page_size,
            "from_": from_
        })
        url = f"https://mygene.info/v3/query?{params}"
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            print(f"  GO query failed for {go_id} (from={from_}): {e}")
            break
        page = data.get("hits", [])
        if not page:
            break
        hits.extend(page)
        if len(page) < page_size:
            break
        from_ += page_size
    return hits


def main():
    # ----- KEGG -----
    kegg_by_category = {}
    all_kegg_ids = set()
    for cat, pid in KEGG_PATHWAYS.items():
        ids = kegg_link(pid)
        kegg_by_category[cat] = ids
        all_kegg_ids.update(ids)
        print(f"[KEGG] {cat}: {len(ids)} genes")

    kegg_map = mygene_batch_query(list(all_kegg_ids))
    # {entrez: (ensembl, symbol)}
    entrez_to_info = {}
    for rec in kegg_map:
        q = rec.get("query")
        ens = rec.get("ensembl", {})
        if isinstance(ens, list):
            ens = ens[0] if ens else {}
        ens_id = ens.get("gene") if ens else None
        sym = rec.get("symbol")
        if q and (ens_id or sym):
            entrez_to_info[q] = (ens_id, sym)

    # ----- GO -----
    go_by_category = defaultdict(set)
    go_info = {}  # entrez -> (ensembl, symbol)
    for cat, go_ids in GO_TERMS.items():
        for go_id in go_ids:
            hits = go_genes(go_id)
            print(f"[GO] {cat} {go_id}: {len(hits)} genes")
            for h in hits:
                e = h.get("_id")
                if not e:
                    continue
                go_by_category[cat].add(str(e))
                ens = h.get("ensembl", {})
                if isinstance(ens, list):
                    ens = ens[0] if ens else {}
                ens_id = ens.get("gene") if ens else None
                sym = h.get("symbol")
                if ens_id or sym:
                    go_info[str(e)] = (ens_id, sym)

    # ----- Combine -----
    # membership per entrez id
    membership = defaultdict(set)
    for cat, ids in kegg_by_category.items():
        for eid in ids:
            membership[eid].add(cat)
    for cat, ids in go_by_category.items():
        for eid in ids:
            membership[eid].add(cat)

    # map to Ensembl; prefer KEGG-derived info, fallback GO
    rows = []
    for eid, cats in membership.items():
        info = entrez_to_info.get(eid) or go_info.get(eid)
        if not info or not info[0]:
            continue
        ens, sym = info
        if len(cats) > 1:
            pathway = "Combination"
        else:
            pathway = list(cats)[0]
        rows.append({"gene_id": ens, "symbol": sym, "pathway": pathway})

    df = pd.DataFrame(rows).drop_duplicates(subset=["gene_id"])
    print(f"\nTotal unique Ensembl genes: {len(df)}")
    print(df["pathway"].value_counts())

    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"Saved: {OUT_TSV}")


if __name__ == "__main__":
    main()
