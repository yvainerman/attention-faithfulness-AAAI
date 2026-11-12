"""
models/tasc.py
--------------
Task-Scaling (TaSc) for the FINAL attention layer, adapted from

    G. Chrysostomou and N. Aletras (2021). "Improving the Faithfulness of
    Attention-based Explanations with Task-specific Information for Text
    Classification." ACL-IJCNLP 2021.  Code: github.com/GChrysostomou/tasc

WHAT TaSc DOES (their Eq. 5)
    Vanilla attention:  c = sum_t  alpha_t * h_t
    TaSc:               c = sum_t  alpha_t * s_t * h_t
  alpha_t  the usual softmax attention, computed from the CONTEXTUALISED h_t
  s_t      a task-specific score computed ONLY from position t's own
           NON-contextualised representation e_t (in NLP: the word embedding,
           taken before the encoder mixes words together).
  Their motivation: an attention weight refers to "the word in context" because
  the encoder mixes information across positions (Tutek & Snajder, 2020).
  Scaling each weight by a score that depends on the position alone ties the
  weight back to that position.

WHY IT FITS OUR EKG MODELS
  Our BIBM paper found the same failure for EKGs: once the reach encoder mixes
  neighbouring segments/beats (lc, gl), attention stops agreeing with
  leave-one-out importance (P2). Both of our model families already have a
  NON-contextualised representation of every attention position by design:
    flat          e_t = the 100-ms PatchCNN token (non-overlapping patches, so
                        no information is shared between tokens), i.e. the
                        input to the reach encoder;
    hierarchical  e_b = the per-beat vector (each beat is encoded on its own,
                        then pooled by the within-beat attention), i.e. the
                        input to the reach encoder over beats.
  So s_t is computed from e_t, exactly where TaSc takes the word embedding.

ADAPTATION (the one change from the paper)
  All three TaSc variants index a learned parameter by VOCABULARY WORD TYPE
  (Lin-TaSc: one scalar u per word; Feat-TaSc: one vector U_w per word;
  Conv-TaSc: Lin-TaSc + a 1x1 conv). EKG segments are continuous -- there is no
  vocabulary to index -- so we keep the Feat-TaSc form (dot product of a learned
  vector with the embedding, summed over features, "sum-over" as in the paper)
  with ONE learned vector u shared by every position:

        s_t = u . e_t = sum_d u_d * e_t[d]

  s_t is still (1) computed from position t alone, (2) learned for the task,
  (3) used to scale position t's attention weight, and (4) independent of the
  encoder, so it plugs into all seven models unchanged. Without a vocabulary,
  Lin-TaSc would collapse to one scalar times sum_d e_t[d] (a fixed direction
  with a single free parameter), and Conv-TaSc (a 1x1 conv with no activation,
  then a sum over channels) to a linear function of e_t plus a constant -- the
  same form as below -- so Feat-TaSc is the variant that carries over to
  continuous inputs. Like the paper, s_t is unconstrained in sign and size and
  has no bias term.

WHAT IS EXPLAINED / TESTED
  The attention map under test stays alpha (the softmax distribution), as in
  the TaSc paper's own "attention" evaluation and in our P1-P5. The tests treat
  s_t * h_t as the values the final attention averages (faithfulness_common.py),
  so erasure, permutation, adversarial and uniform interventions act on alpha
  exactly as before.

SWITCH
  ECG_TASC = none (default: the original models, bit-identical) | feat
  Read by the trainers, tuners, performance tests and every faithfulness test,
  so they always agree on which checkpoints they train / load.
"""
import math
import os

import torch
import torch.nn as nn


TASC_CHOICES = ("feat",)
_OFF = ("", "0", "none", "off", "false", "no")
_ON = ("1", "on", "true", "yes")


def parse_tasc(value):
    """'none'/'0'/'' -> None ; 'feat'/'1'/'on' -> 'feat'. Anything else is an error."""
    v = "" if value is None else str(value).strip().lower()
    if v in _OFF:
        return None
    if v in _ON:
        return "feat"
    if v in TASC_CHOICES:
        return v
    raise ValueError(f"ECG_TASC={value!r}: use one of none | {' | '.join(TASC_CHOICES)}")


def tasc_from_env(name="ECG_TASC"):
    return parse_tasc(os.environ.get(name))


def ckpt_subdir(base, tasc=None, uniform=False):
    """Checkpoint folder name, e.g. flat_seq | flat_seq_uniform | flat_seq_tasc |
    flat_seq_tasc_uniform. The no-TaSc names are the ORIGINAL ones, so the
    original checkpoints are found unchanged."""
    return base + ("_tasc" if tasc else "") + ("_uniform" if uniform else "")


def method_tag(tasc=None):
    """Short label for logs / output folders: '' (original) or '_tasc'."""
    return "_tasc" if tasc else ""


class TaScScore(nn.Module):
    """Feat-TaSc score with one learned vector shared by all positions.

    (B, N, in_dim) non-contextualised representations -> (B, N) scores s_t.
    Initialised like the reference code (uniform in +/- stdv); the reference
    uses stdv = 1/sqrt(vocab_size), which has no EKG analogue, so we use the
    usual fan-in rule stdv = 1/sqrt(in_dim)."""
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.u = nn.Parameter(torch.empty(in_dim))
        stdv = 1.0 / math.sqrt(in_dim)
        self.u.data.uniform_(-stdv, stdv)

    def forward(self, e):                       # (B,N,in_dim)
        return (e * self.u).sum(dim=-1)         # (B,N)   "sum-over" of the scaled embedding


def build_tasc(tasc, in_dim):
    """None for the original models (no extra parameters, identical state_dict)."""
    if tasc is None:
        return None
    if tasc == "feat":
        return TaScScore(in_dim)
    raise ValueError(f"unknown TaSc variant {tasc!r}")
