"""Label-free rule for duplicate identities (the same human under two IDs). Uses only ID strings and face embeddings.
RULE: ID a is a proper suffix of ID b AND (len(a) >= 5 OR cosine(centroid a, centroid b) >= 0.4)  ->  both people are quarantined.
Evidence (D29): short suffixes (1-4 digits) are accidental (median cosine 0.03-0.09); 5-digit-in-6-digit pairs have median cosine 0.68."""
import numpy as np

COS_MIN = 0.4
LEN_MIN = 5

def suffix_pairs(ids):
    s = set(ids)
    return [(x[k:], x) for x in ids for k in range(1, len(x)) if x[k:] in s]

def quarantine_people(ids, centroid_cos):
    """ids: list of person-id strings; centroid_cos(a, b) -> float. Returns (set of quarantined ids, list of (short, long, cos, reason))."""
    out, q = [], set()
    for a, b in suffix_pairs(list(ids)):
        c = float(centroid_cos(a, b)); why = ("len>=%d" % LEN_MIN if len(a) >= LEN_MIN else "") + ("+" if len(a) >= LEN_MIN and c >= COS_MIN else "") + ("cos>=%.1f" % COS_MIN if c >= COS_MIN else "")
        if len(a) >= LEN_MIN or c >= COS_MIN: q.update((a, b)); out.append((a, b, c, why))
    return q, out
